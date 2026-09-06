"""
Tests for quantui.scf_robust — the shared SCF-convergence rescue helper
(M-SCF-ROBUST SCFR.1-2, SCFR.4).

Test strategy
-------------
* Control-flow tests use a small fake SCF object (``_FakeSCF``) so every
  branch (no-op on first-try convergence, bootstrap success, level-shift
  fallback, ``rescue=False``, ``max_stage=1``, both stages exhausted) is
  exercised deterministically and instantly — no real quantum chemistry, no
  flakiness, no CI cost.
* One small real-PySCF integration test (marked ``pyscf_only`` +
  ``slow``) proves the helper actually drives real ``mf.kernel()`` calls
  end-to-end on a cheap RHF/STO-3G water molecule, forcing non-convergence
  deterministically via an artificially low ``max_cycle`` rather than
  reproducing CHEM-3200 Lab 2's real (expensive) Mn2+ hexaaquo B3LYP/def2-SVP
  failing case — that case is documented in
  ``QuantUI-development-tracking/TODO/GOTCHAS.md`` but is far too costly to
  run on every CI build. The two rescue stages themselves were validated
  against that real failing case before this helper was written (see the
  module docstring); this test only proves the plumbing is correct.
* One real-PySCF no-op test confirms a calculation that already converges
  under plain defaults is bit-identical (same energy, same converged flag)
  whether or not it goes through the helper — the milestone's hard
  invariant (see roadmap 52's Success Criteria).
"""

from __future__ import annotations

import pytest

from quantui.scf_robust import (
    SCF_RESCUE_BOOTSTRAP,
    SCF_RESCUE_FAILED,
    SCF_RESCUE_LEVEL_SHIFT,
    SCF_RESCUE_NONE,
    run_scf_with_rescue,
)

# ---------------------------------------------------------------------------
# PySCF availability
# ---------------------------------------------------------------------------

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE,
    reason="PySCF not installed (Linux/macOS/WSL only)",
)


# ---------------------------------------------------------------------------
# Fake SCF object — deterministic control-flow tests, no real chemistry
# ---------------------------------------------------------------------------


class _FakeMol:
    """Minimal stand-in for a PySCF ``gto.Mole`` — only ``.spin`` is read."""

    def __init__(self, spin: int = 0) -> None:
        self.spin = spin


class _FakeBootstrapSCF:
    """Stand-in for the internal bootstrap RHF/UHF object scf_robust builds.

    Patched in via monkeypatch of ``pyscf.scf.RHF``/``pyscf.scf.UHF`` so
    :func:`run_scf_with_rescue` never touches real PySCF in the fake-object
    tests below.
    """

    def __init__(self, mol) -> None:
        self.mol = mol
        self.verbose = 4

    def kernel(self):
        self.converged = True
        return -1.0

    def make_rdm1(self):
        return "bootstrap-dm"


class _FakeSCF:
    """Scripted fake mean-field object.

    ``kernel_results`` is a list of ``(energy, converged)`` pairs consumed in
    order, one per ``.kernel(...)`` call — lets a test script the exact
    sequence "first attempt fails, bootstrap retry succeeds" etc. without any
    real SCF math. Every call is recorded in ``self.calls`` so a test can
    assert exactly how many kernel calls happened and with what kwargs (the
    no-op/control-case invariant depends on this: converged-on-first-try must
    make exactly one call, never two).
    """

    def __init__(self, kernel_results, spin: int = 0) -> None:
        self._results = list(kernel_results)
        self.mol = _FakeMol(spin=spin)
        self.calls: list[dict] = []
        self.converged = False
        self.level_shift = 0.0
        self.init_guess = "minao"
        self.max_cycle = 50

    def kernel(self, **kwargs):
        self.calls.append(kwargs)
        energy, converged = self._results.pop(0)
        self.converged = converged
        return energy


@pytest.fixture
def _patch_bootstrap_scf(monkeypatch):
    """Route scf_robust's internal ``from pyscf import scf`` to the fake.

    Requested (via ``usefixtures``) only by the fake-object test classes
    below — never by ``TestRealPySCFIntegration``, which needs the genuine
    ``pyscf.scf`` module.
    """
    import types

    fake_scf_module = types.SimpleNamespace(
        RHF=_FakeBootstrapSCF, UHF=_FakeBootstrapSCF
    )
    monkeypatch.setitem(__import__("sys").modules, "pyscf.scf", fake_scf_module)
    # scf_robust does ``from pyscf import scf`` (attribute access on the
    # `pyscf` package), not ``import pyscf.scf`` — patch the attribute too so
    # both import styles are covered regardless of what's already cached.
    if _PYSCF_AVAILABLE:
        import pyscf

        monkeypatch.setattr(pyscf, "scf", fake_scf_module, raising=False)


@pytest.mark.usefixtures("_patch_bootstrap_scf")
class TestControlFlowNoOp:
    """The hard invariant: a first-try convergence must never be retried."""

    def test_converged_first_try_is_a_true_no_op(self):
        mf = _FakeSCF([(-76.0, True)])
        energy = run_scf_with_rescue(mf)
        assert energy == -76.0
        assert mf.converged is True
        assert len(mf.calls) == 1, "must not retry a calc that already converged"
        assert mf.scf_rescue_stage == SCF_RESCUE_NONE

    def test_rescue_false_skips_everything_even_on_non_convergence(self):
        mf = _FakeSCF([(-76.0, False)])
        energy = run_scf_with_rescue(mf, rescue=False)
        assert energy == -76.0
        assert mf.converged is False
        assert len(mf.calls) == 1, "rescue=False must never retry"
        assert mf.scf_rescue_stage == SCF_RESCUE_FAILED

    def test_dm0_forwarded_only_on_first_attempt(self):
        mf = _FakeSCF([(-76.0, True)])
        run_scf_with_rescue(mf, dm0="caller-dm0")
        assert mf.calls == [{"dm0": "caller-dm0"}]


@pytest.mark.usefixtures("_patch_bootstrap_scf")
class TestBootstrapRescue:
    def test_bootstrap_stage_fires_and_succeeds(self):
        mf = _FakeSCF([(-75.5, False), (-76.0, True)])
        energy = run_scf_with_rescue(mf)
        assert energy == -76.0
        assert mf.converged is True
        assert mf.scf_rescue_stage == SCF_RESCUE_BOOTSTRAP
        assert len(mf.calls) == 2
        # Second call is the bootstrap retry — must use the bootstrap's
        # density, not whatever dm0 (if any) the caller originally passed.
        assert mf.calls[1] == {"dm0": "bootstrap-dm"}

    def test_max_stage_1_stops_after_bootstrap(self):
        mf = _FakeSCF([(-75.5, False), (-75.6, False)])
        energy = run_scf_with_rescue(mf, max_stage=1)
        assert energy == -75.6
        assert mf.converged is False
        assert len(mf.calls) == 2, "max_stage=1 must not attempt level-shift"
        assert mf.scf_rescue_stage == SCF_RESCUE_FAILED
        assert mf.level_shift == 0.0, "level-shift stage must not have run"


@pytest.mark.usefixtures("_patch_bootstrap_scf")
class TestLevelShiftFallback:
    def test_level_shift_fires_when_bootstrap_alone_is_not_enough(self):
        mf = _FakeSCF([(-75.5, False), (-75.6, False), (-76.0, True)])
        energy = run_scf_with_rescue(mf)
        assert energy == -76.0
        assert mf.converged is True
        assert mf.scf_rescue_stage == SCF_RESCUE_LEVEL_SHIFT
        assert len(mf.calls) == 3
        assert mf.level_shift == 0.3
        assert mf.init_guess == "atom"
        assert mf.max_cycle == 100

    def test_level_shift_does_not_lower_an_already_higher_max_cycle(self):
        mf = _FakeSCF([(-75.5, False), (-75.6, False), (-76.0, True)])
        mf.max_cycle = 200
        run_scf_with_rescue(mf)
        assert mf.max_cycle == 200

    def test_both_stages_exhausted_reports_unconverged_as_is(self):
        mf = _FakeSCF([(-75.5, False), (-75.6, False), (-75.7, False)])
        energy = run_scf_with_rescue(mf)
        assert energy == -75.7
        assert mf.converged is False
        assert mf.scf_rescue_stage == SCF_RESCUE_FAILED


@pytest.mark.usefixtures("_patch_bootstrap_scf")
class TestStreamStatusLines:
    def test_status_lines_written_to_stream_when_rescue_fires(self):
        import io

        stream = io.StringIO()
        mf = _FakeSCF([(-75.5, False), (-76.0, True)])
        run_scf_with_rescue(mf, stream=stream)
        out = stream.getvalue()
        assert "[scf_rescue]" in out
        assert "bootstrap" in out.lower()

    def test_no_status_lines_when_first_attempt_converges(self):
        import io

        stream = io.StringIO()
        mf = _FakeSCF([(-76.0, True)])
        run_scf_with_rescue(mf, stream=stream)
        assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# Real PySCF integration (SCFR.4) — cheap, deterministic, not the literal
# Lab 2 case (see module docstring)
# ---------------------------------------------------------------------------


@pyscf_only
@pytest.mark.slow
class TestRealPySCFIntegration:
    def _water_mol(self):
        from pyscf import gto

        mol = gto.Mole()
        mol.atom = "O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24"
        mol.basis = "STO-3G"
        mol.charge = 0
        mol.spin = 0
        mol.verbose = 0
        mol.build()
        return mol

    def test_no_op_for_a_calculation_that_already_converges(self):
        """Zero behavior change for the common case (roadmap 52's invariant)."""
        from pyscf import scf

        mol = self._water_mol()

        mf_plain = scf.RHF(mol)
        mf_plain.verbose = 0
        expected_energy = mf_plain.kernel()

        mf_rescued = scf.RHF(mol)
        mf_rescued.verbose = 0
        rescued_energy = run_scf_with_rescue(mf_rescued)

        assert mf_rescued.converged is True
        assert mf_rescued.scf_rescue_stage == SCF_RESCUE_NONE
        assert rescued_energy == pytest.approx(expected_energy, abs=1e-10)
        assert mf_plain.cycles == mf_rescued.cycles, (
            "the rescue helper must not add SCF cycles for a calc that "
            "already converges"
        )

    def test_bootstrap_rescue_recovers_a_forced_non_convergence(self):
        """Force non-convergence via an artificially tiny max_cycle on the
        first attempt (a cheap, deterministic proxy for a genuinely marginal
        SCF — see module docstring for why the literal Lab 2 case isn't
        used here) and confirm the bootstrap stage still reaches the
        correct, converged energy.
        """
        from pyscf import scf

        mol = self._water_mol()

        mf_reference = scf.RHF(mol)
        mf_reference.verbose = 0
        reference_energy = mf_reference.kernel()
        assert mf_reference.converged is True

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.max_cycle = 1  # too low to converge from scratch
        energy = run_scf_with_rescue(mf)

        assert mf.converged is True
        assert mf.scf_rescue_stage == SCF_RESCUE_BOOTSTRAP
        assert energy == pytest.approx(reference_energy, abs=1e-6)
