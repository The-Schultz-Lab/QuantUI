"""Tests for quantui.tddft_calc.

Unit-level tests for run_tddft_calc(). Broader integration coverage (history
replay, panel activation) lives in test_tddft_analysis_history.py.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE, reason="PySCF not installed (Linux/macOS/WSL only)"
)


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
    )


# ============================================================================
# Post-HF method guard (M2 audit fix, 2026-07-14)
# ============================================================================


class TestRunTddftCalcPostHfGuard:
    """Post-HF methods raise a clear ValueError instead of a cryptic LibXC error.

    Regression: run_tddft_calc() had no special-casing for MP2/CCSD/CCSD(T)
    — the SCF-selection branch silently treated them as a DFT xc functional
    (mf.xc = "CCSD"), failing deep inside PySCF with "LibXCFunctional: name
    'CCSD' not found" instead of a clear message. The guard fires before any
    PySCF import, so it needs no PySCF.
    """

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        from quantui.tddft_calc import run_tddft_calc

        with pytest.raises(ValueError, match="post-HF"):
            run_tddft_calc(_water(), method=method, basis="STO-3G", nstates=2)


# ============================================================================
# Basic run — PySCF-gated
# ============================================================================


class TestRunTddftCalcBasic:
    @pyscf_only
    @pytest.mark.slow
    def test_returns_tddft_result(self):
        from quantui.tddft_calc import run_tddft_calc

        result = run_tddft_calc(_water(), method="RHF", basis="STO-3G", nstates=2)
        assert result.formula == "H2O"
        assert len(result.excitation_energies_ev) <= 2

    @pyscf_only
    @pytest.mark.slow
    def test_scf_variant_reports_rks_for_closed_shell_dft(self):
        """M-UX2 UXP2.10 — confirms the wiring, mirroring the identical
        RHF/UHF/RKS/UKS dispatch already thoroughly tested in
        test_session_calc.py::TestScfVariantProvenance."""
        from quantui.tddft_calc import run_tddft_calc

        result = run_tddft_calc(_water(), method="B3LYP", basis="STO-3G", nstates=2)
        assert result.scf_variant == "RKS"


# ============================================================================
# AUDIT F08 — per-root TD convergence must not be swallowed
# ============================================================================


class TestTddftConvergence:
    @pyscf_only
    @pytest.mark.slow
    def test_unconverged_roots_are_not_reported_as_converged(self, monkeypatch):
        """Controlled reproduction from the audit: restrict the real TDHF
        Davidson solve to one iteration (only max_cycle forced; the SCF and
        TD kernels themselves are real PySCF). A real one-iteration-limited
        TDHF/6-31G water solve gives converged=[False, False, False] and
        excitations 9.804744, 11.993916, 12.420652 eV — the old code
        reported these as a converged result because it never checked
        td.converged at all.
        """
        import pyscf.scf.hf as pyscf_hf
        import pyscf.tdscf.rhf  # noqa: F401 — import side effect registers RHF.TDHF

        from quantui.tddft_calc import run_tddft_calc

        # mf.TDHF() is registered via pyscf.lib.class_as_method, which binds
        # the TDHF class into RHF.TDHF as a plain function at pyscf import
        # time — monkeypatching pyscf.tdscf.rhf.TDHF afterward has no effect
        # on that already-captured reference, so patch RHF.TDHF itself.
        _original_tdhf_method = pyscf_hf.RHF.TDHF

        def _one_cycle_tdhf(self):
            obj = _original_tdhf_method(self)
            obj.max_cycle = 1
            return obj

        monkeypatch.setattr(pyscf_hf.RHF, "TDHF", _one_cycle_tdhf)

        result = run_tddft_calc(_water(), method="RHF", basis="6-31G", nstates=3)

        assert result.td_converged == [False, False, False]
        assert result.n_converged_states == 0
        assert result.converged is False
        # The excitations are still surfaced (so the UI can show what
        # actually came out of the solver) — just not stamped converged.
        assert len(result.excitation_energies_ev) == 3

    @pyscf_only
    @pytest.mark.slow
    def test_converged_roots_report_full_convergence(self):
        """Sanity check the happy path: a normal (unpatched) TDHF solve on
        a small system converges every requested root."""
        from quantui.tddft_calc import run_tddft_calc

        result = run_tddft_calc(_water(), method="RHF", basis="STO-3G", nstates=2)

        assert result.td_converged is not None
        assert all(result.td_converged)
        assert result.n_converged_states == len(result.td_converged)
        assert result.converged is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
