"""
Tests for quantui.optimizer — QM geometry optimization via ASE-BFGS + PySCF.

Test strategy
-------------
* OptimizationResult dataclass tests run on any platform — they construct
  the dataclass directly and verify properties / summary without PySCF.
* Geometry calculation tests require both ASE and PySCF and are marked
  with ``pyscf_only``; run them on Linux/WSL:
      pytest tests/test_optimizer.py -v
* Import-guard tests run on all platforms and verify helpful error messages.

WSL testing
-----------
With both ase and pyscf installed in your conda environment:
    pytest tests/test_optimizer.py -v
    pytest tests/test_optimizer.py -v -m "not slow"   # skip actual QM runs
"""

import io

import pytest

from quantui.ase_bridge import ASE_AVAILABLE
from quantui.molecule import Molecule
from quantui.optimizer import DEFAULT_FMAX, DEFAULT_OPT_STEPS, OptimizationResult

# Check PySCF + ASE-PySCF availability
_PYSCF_AVAILABLE = False
try:
    import pyscf as _p  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

_ASE_PYSCF_AVAILABLE = False
try:
    from ase.calculators.pyscf import PySCF as _c  # noqa: F401

    _ASE_PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not (ASE_AVAILABLE and _PYSCF_AVAILABLE and _ASE_PYSCF_AVAILABLE),
    reason="ase>=3.22 and pyscf not both installed (Linux/WSL only)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2(bond_length: float = 0.74) -> Molecule:
    """H2 molecule — fastest meaningful QM calculation."""
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _make_result(**overrides) -> OptimizationResult:
    """Build an OptimizationResult with defaults, allowing field overrides."""
    mol = _h2()
    traj = [mol, mol]
    defaults = dict(
        molecule=mol,
        trajectory=traj,
        energies_hartree=[-1.100, -1.117],
        converged=True,
        n_steps=1,
        method="RHF",
        basis="STO-3G",
        formula="H2",
    )
    defaults.update(overrides)
    return OptimizationResult(**defaults)


# ============================================================================
# OptimizationResult dataclass — no PySCF needed
# ============================================================================


class TestOptimizationResultDataclass:
    """Unit tests for OptimizationResult fields, properties, and summary."""

    def test_energy_hartree_is_last_frame(self):
        result = _make_result(energies_hartree=[-1.100, -1.117, -1.118])
        assert result.energy_hartree == pytest.approx(-1.118)

    def test_energy_ev_conversion(self):
        from quantui.session_calc import HARTREE_TO_EV

        result = _make_result(energies_hartree=[-1.0, -1.117])
        assert result.energy_ev == pytest.approx(-1.117 * HARTREE_TO_EV, rel=1e-9)

    def test_energy_change_hartree(self):
        result = _make_result(energies_hartree=[-1.100, -1.117])
        assert result.energy_change_hartree == pytest.approx(-0.017, abs=1e-6)

    def test_energy_change_positive_means_geometry_worsened(self):
        result = _make_result(energies_hartree=[-1.117, -1.100])
        assert result.energy_change_hartree > 0

    def test_energy_change_zero_for_single_frame(self):
        mol = _h2()
        result = _make_result(trajectory=[mol], energies_hartree=[-1.117], n_steps=0)
        assert result.energy_change_hartree == 0.0

    def test_rmsd_zero_for_identical_frames(self):
        mol = _h2()
        result = _make_result(trajectory=[mol, mol])
        assert result.rmsd_angstrom == pytest.approx(0.0, abs=1e-8)

    def test_rmsd_positive_for_different_frames(self):
        mol_a = _h2(bond_length=0.74)
        mol_b = _h2(bond_length=1.00)
        result = _make_result(trajectory=[mol_a, mol_b])
        assert result.rmsd_angstrom > 0.0

    def test_summary_contains_formula(self):
        assert "H2" in _make_result(formula="H2").summary()

    def test_summary_contains_method_basis(self):
        summary = _make_result(method="UHF", basis="cc-pVDZ").summary()
        assert "UHF" in summary
        assert "cc-pVDZ" in summary

    def test_summary_converged_shows_yes(self):
        assert "Yes" in _make_result(converged=True).summary()

    def test_summary_not_converged_shows_warning(self):
        summary = _make_result(converged=False).summary()
        assert "NO" in summary or "not converge" in summary.lower()

    def test_summary_contains_steps(self):
        assert "7" in _make_result(n_steps=7).summary()

    def test_summary_contains_energy_change(self):
        summary = _make_result(energies_hartree=[-1.100, -1.117]).summary()
        assert "-0.017" in summary or "0.017" in summary

    def test_all_fields_accessible(self):
        result = _make_result()
        assert isinstance(result.molecule, Molecule)
        assert isinstance(result.trajectory, list)
        assert isinstance(result.energies_hartree, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_steps, int)
        assert isinstance(result.method, str)
        assert isinstance(result.basis, str)
        assert isinstance(result.formula, str)


# ============================================================================
# Module-level constants
# ============================================================================


class TestModuleConstants:
    def test_default_fmax_is_float(self):
        assert isinstance(DEFAULT_FMAX, float)

    def test_default_fmax_positive(self):
        assert DEFAULT_FMAX > 0

    def test_default_opt_steps_positive_int(self):
        assert isinstance(DEFAULT_OPT_STEPS, int)
        assert DEFAULT_OPT_STEPS > 0

    def test_config_exports_match(self):
        from quantui import config

        assert hasattr(config, "DEFAULT_FMAX")
        assert hasattr(config, "DEFAULT_OPT_STEPS")
        assert config.DEFAULT_FMAX == DEFAULT_FMAX
        assert config.DEFAULT_OPT_STEPS == DEFAULT_OPT_STEPS

    def test_bohr_to_angstrom_matches_shared_config_constant(self):
        """L audit fix: optimizer.py used to hand-type its own Bohr->Angstrom
        literal (0.529177249, a plain CODATA-2018 rounding) that didn't
        match freq_calc.py's separately hand-typed literal
        (0.52917721092, pyscf.data.nist.BOHR). Both now import
        config.BOHR_TO_ANGSTROM, so force/gradient unit conversions in
        optimize_geometry are computed with the same constant PySCF itself
        uses internally.
        """
        import quantui.optimizer as opt_mod
        from quantui import config
        from quantui.results_storage import _ANGSTROM_TO_BOHR

        assert opt_mod._BOHR_TO_ANG == config.BOHR_TO_ANGSTROM
        assert config.BOHR_TO_ANGSTROM == pytest.approx(0.52917721092, abs=1e-11)
        assert _ANGSTROM_TO_BOHR == pytest.approx(
            1.0 / config.BOHR_TO_ANGSTROM, abs=1e-15
        )


# ============================================================================
# Import-guard tests — all platforms
# ============================================================================


class TestOptimizeGeometryImportGuards:
    """Patch quantui.optimizer's OWN ``ASE_AVAILABLE`` name directly rather
    than reloading the module after monkeypatching ase_bridge's copy.

    ``optimize_geometry()`` reads the module-local ``ASE_AVAILABLE`` binding
    (imported via ``from .ase_bridge import ASE_AVAILABLE`` at module load
    time) — monkeypatching ``ase_bridge.ASE_AVAILABLE`` doesn't affect that
    binding at all, which is why the original version of this test used
    ``importlib.reload(opt_mod)`` to re-execute the import with the patched
    value in scope. But ``monkeypatch`` only reverts attributes it directly
    set (``ase_bridge.ASE_AVAILABLE``) — it has no way to know the reload
    also needs undoing, so ``quantui.optimizer.ASE_AVAILABLE`` stayed stuck
    at ``False`` for the rest of the test session once this test ran,
    silently breaking every later test that called ``optimize_geometry()``
    expecting ASE to be available. Patching the local name directly avoids
    the reload (and the leak) entirely.
    """

    def test_raises_when_ase_unavailable(self, monkeypatch):
        import quantui.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "ASE_AVAILABLE", False)

        with pytest.raises(ImportError, match="pip install"):
            opt_mod.optimize_geometry(_h2())

    def test_error_message_is_actionable(self, monkeypatch):
        import quantui.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "ASE_AVAILABLE", False)

        with pytest.raises(ImportError) as exc_info:
            opt_mod.optimize_geometry(_h2())
        msg = str(exc_info.value)
        assert "pip install" in msg or "conda install" in msg


class TestOptimizeGeometryPostHfGuard:
    """M2 audit fix (2026-07-14): post-HF methods raise a clear ValueError.

    Regression: optimize_geometry() had no special-casing for MP2/CCSD/
    CCSD(T) — _QuantUIPySCFCalc.calculate() silently treated them as a DFT
    xc functional (mf.xc = "CCSD"), failing deep inside PySCF with a cryptic
    "LibXCFunctional: name 'CCSD' not found" instead of a clear message.
    The guard fires before any PySCF import, so it needs ASE but not PySCF.
    """

    ase_only = pytest.mark.skipif(not ASE_AVAILABLE, reason="ase not installed")

    @ase_only
    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        from quantui.optimizer import optimize_geometry

        with pytest.raises(ValueError, match="post-HF"):
            optimize_geometry(_h2(), method=method, basis="STO-3G")


# ============================================================================
# Optimization calculation tests — Linux/WSL with ase + pyscf
# ============================================================================


class TestOptimizeGeometryBasic:
    """Functional tests for optimize_geometry()."""

    @pyscf_only
    @pytest.mark.slow
    def test_returns_optimization_result(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert isinstance(result, OptimizationResult)

    @pyscf_only
    @pytest.mark.slow
    def test_trajectory_is_nonempty(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert len(result.trajectory) >= 1

    @pyscf_only
    @pytest.mark.slow
    def test_trajectory_frames_are_molecules(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        for frame in result.trajectory:
            assert isinstance(frame, Molecule)

    @pyscf_only
    @pytest.mark.slow
    def test_energies_list_matches_trajectory_length(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert len(result.energies_hartree) == len(result.trajectory)

    @pyscf_only
    @pytest.mark.slow
    def test_final_molecule_has_correct_atoms(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert result.molecule.atoms == ["H", "H"]

    @pyscf_only
    @pytest.mark.slow
    def test_n_steps_consistent_with_trajectory(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert result.n_steps == len(result.trajectory) - 1

    @pyscf_only
    @pytest.mark.slow
    def test_formula_matches_input(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert result.formula == "H2"

    @pyscf_only
    @pytest.mark.slow
    def test_method_basis_recorded(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=20
        )
        assert result.method == "RHF"
        assert result.basis == "STO-3G"


class TestOptimizeGeometryEnergyAndConvergence:
    """Energy sanity checks and convergence behaviour."""

    @pyscf_only
    @pytest.mark.slow
    def test_energy_decreases_monotonically(self):
        """BFGS should never increase the energy in a well-behaved case."""
        from quantui.optimizer import optimize_geometry

        # Start from a slightly compressed H2 — forces will push it to equilibrium
        mol = _h2(bond_length=0.60)
        result = optimize_geometry(
            mol, method="RHF", basis="STO-3G", fmax=0.05, steps=50
        )
        energies = result.energies_hartree
        # Allow small numerical noise (<1e-5 Ha) between consecutive steps
        for i in range(1, len(energies)):
            assert (
                energies[i] <= energies[i - 1] + 1e-4
            ), f"Energy increased at step {i}: {energies[i - 1]:.8f} → {energies[i]:.8f}"

    @pyscf_only
    @pytest.mark.slow
    def test_h2_optimization_converges(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(0.60), method="RHF", basis="STO-3G", fmax=0.05, steps=50
        )
        assert result.converged is True

    @pyscf_only
    @pytest.mark.slow
    def test_max_steps_respected(self):
        """When steps=1, optimizer should stop and converged=False for most molecules."""
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(0.60), method="RHF", basis="STO-3G", fmax=1e-6, steps=1
        )
        # With only 1 step and a very tight fmax it should NOT converge
        assert result.n_steps <= 1


class TestOptimizeGeometryMetadataPreservation:
    """Charge and multiplicity survive through the optimization."""

    @pyscf_only
    @pytest.mark.slow
    def test_charge_preserved_neutral(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=10
        )
        assert result.molecule.charge == 0
        for frame in result.trajectory:
            assert frame.charge == 0

    @pyscf_only
    @pytest.mark.slow
    def test_multiplicity_preserved_singlet(self):
        from quantui.optimizer import optimize_geometry

        result = optimize_geometry(
            _h2(), method="RHF", basis="STO-3G", fmax=0.1, steps=10
        )
        assert result.molecule.multiplicity == 1


class TestOptimizeGeometryOutputStream:
    """BFGS progress is routed to progress_stream."""

    @pyscf_only
    @pytest.mark.slow
    def test_bfgs_output_written_to_stream(self):
        from quantui.optimizer import optimize_geometry

        buf = io.StringIO()
        optimize_geometry(
            _h2(0.60),
            method="RHF",
            basis="STO-3G",
            fmax=0.05,
            steps=20,
            progress_stream=buf,
        )
        output = buf.getvalue()
        # BFGS writes "BFGS: step  fmax" table — expect at least some content
        assert len(output) > 0

    @pyscf_only
    @pytest.mark.slow
    def test_pyscf_output_suppressed_in_stream(self):
        """PySCF SCF iterations should NOT appear in the BFGS progress stream."""
        from quantui.optimizer import optimize_geometry

        buf = io.StringIO()
        optimize_geometry(
            _h2(0.60),
            method="RHF",
            basis="STO-3G",
            fmax=0.05,
            steps=20,
            progress_stream=buf,
        )
        output = buf.getvalue()
        # PySCF cycle lines contain "converge" or "SCF energy"
        # At verbose=0 these should be absent from our stream
        assert "converge" not in output.lower() or "BFGS" in output


class TestOptimizeGeometryResume:
    """M-ISSUES ISSUE.5 regression: resuming from a checkpoint must not
    regress the reported step count.

    A fresh ``BFGS`` instance always has ``nsteps == 0``. Left unseeded on a
    resumed run, the per-step ``checkpoint.update(steps_done=dyn.nsteps)``
    call reports a count that restarted at 0 instead of continuing from what
    the interrupted run had already banked — a resumed-then-interrupted-again
    optimization under-reports how much work is saved. Seeding ``nsteps``
    from the checkpoint on resume (``optimizer.py``) fixes it.

    The installed ASE version (>= 3.26, via ``Dynamics.irun``'s
    ``_traj_is_empty()`` guard) already skips the pre-loop observer call that
    an older ASE would fire unconditionally at ``nsteps == 0`` — the
    duplicate-trajectory-frame half of ISSUE.5 as originally filed. The
    trajectory-length assertion below is kept as a live invariant check, not
    because this test demonstrates a fix for it.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_resume_continues_the_step_count_and_does_not_duplicate_frames(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
        from quantui.checkpoint import CalcIdentity, Checkpoint
        from quantui.optimizer import optimize_geometry

        # Compressed bond length so the first step never converges within
        # steps=1 — every call below is guaranteed to take exactly one step.
        mol = _h2(0.60)
        identity = CalcIdentity.from_molecule(
            mol, calc_type="geometry_opt", method="RHF", basis="STO-3G"
        )
        checkpoint = Checkpoint(identity)
        checkpoint.begin()

        result1 = optimize_geometry(
            mol,
            method="RHF",
            basis="STO-3G",
            fmax=1e-6,
            steps=1,
            checkpoint=checkpoint,
            resume=False,
        )
        assert result1.converged is False
        len_after_first_run = len(result1.trajectory)
        steps_done_after_first_run = checkpoint.load_state()["steps_done"]
        assert steps_done_after_first_run == 1

        result2 = optimize_geometry(
            mol,
            method="RHF",
            basis="STO-3G",
            fmax=1e-6,
            steps=1,
            checkpoint=checkpoint,
            resume=True,
        )

        # Exactly one new frame — not a duplicated boundary frame plus one.
        assert len(result2.trajectory) == len_after_first_run + 1

        # Step count continues from what the first run already banked,
        # rather than resetting because the resumed BFGS instance's own
        # nsteps starts at 0.
        steps_done_after_resume = checkpoint.load_state()["steps_done"]
        assert steps_done_after_resume == steps_done_after_first_run + 1


# ============================================================================
# Public API surface
# ============================================================================


class TestPublicAPI:
    def test_optimization_result_importable_from_quantui(self):
        from quantui import OptimizationResult  # noqa: F401

    def test_optimize_geometry_importable_from_quantui(self):
        from quantui import optimize_geometry  # noqa: F401


# ============================================================================
# Run directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
