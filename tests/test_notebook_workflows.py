"""
Headless tests for the core notebook workflows.

These tests exercise the same code paths the notebook UI calls, without
requiring a browser, Voilà, or ipywidgets. Run them inside the container:

    apptainer exec quantui.sif python -m pytest tests/test_notebook_workflows.py -v

Or locally (Linux/WSL with quantui env active):

    python -m pytest tests/test_notebook_workflows.py -v
"""

import threading
from io import StringIO

import pytest

from quantui import run_in_session
from quantui.molecule import Molecule
from quantui.preopt import preoptimize

# ---------------------------------------------------------------------------
# PySCF availability — used to skip tests that require it on Windows/no-PySCF
# ---------------------------------------------------------------------------

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

_requires_pyscf = pytest.mark.skipif(
    not _HAS_PYSCF, reason="PySCF not installed (Linux/macOS/WSL only)"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _methane() -> Molecule:
    return Molecule(
        atoms=["C", "H", "H", "H", "H"],
        coordinates=[
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    )


# ---------------------------------------------------------------------------
# preoptimize — tuple unpacking (regression: was returning tuple not Molecule)
# ---------------------------------------------------------------------------


class TestPreoptimize:
    def test_returns_tuple(self):
        result = preoptimize(_water())
        assert isinstance(result, tuple), "preoptimize must return a 2-tuple"
        assert len(result) == 2

    def test_first_element_is_molecule(self):
        mol, rmsd = preoptimize(_water())
        assert isinstance(mol, Molecule)

    def test_second_element_is_float_rmsd(self):
        mol, rmsd = preoptimize(_water())
        assert isinstance(rmsd, float)
        assert rmsd >= 0.0

    def test_optimized_molecule_has_same_formula(self):
        mol, _ = preoptimize(_water())
        assert mol.get_formula() == "H2O"

    def test_optimized_molecule_preserves_charge_and_mult(self):
        original = _water()
        optimized, _ = preoptimize(original)
        assert optimized.charge == original.charge
        assert optimized.multiplicity == original.multiplicity


# ---------------------------------------------------------------------------
# run_in_session — HF methods
# ---------------------------------------------------------------------------


@_requires_pyscf
class TestRunInSessionHF:
    def test_rhf_water_sto3g(self):
        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged
        assert abs(result.energy_hartree - (-74.963)) < 0.01
        assert result.method == "RHF"
        assert result.formula == "H2O"

    def test_uhf_water_sto3g(self):
        # Water is closed-shell but UHF should still converge to same energy
        result = run_in_session(_water(), method="UHF", basis="STO-3G", verbose=0)
        assert result.converged
        assert abs(result.energy_hartree - (-74.963)) < 0.01

    def test_rhf_methane_sto3g(self):
        result = run_in_session(_methane(), method="RHF", basis="STO-3G", verbose=0)
        assert result.converged
        assert result.energy_hartree < 0

    def test_result_has_homo_lumo_gap(self):
        result = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        assert result.homo_lumo_gap_ev is not None
        assert result.homo_lumo_gap_ev > 0

    def test_progress_stream_receives_output(self):
        buf = StringIO()
        run_in_session(
            _water(), method="RHF", basis="STO-3G", verbose=3, progress_stream=buf
        )
        output = buf.getvalue()
        assert len(output) > 0, "progress_stream should have received PySCF output"


# ---------------------------------------------------------------------------
# run_in_session — DFT methods
# ---------------------------------------------------------------------------


@_requires_pyscf
class TestRunInSessionDFT:
    def test_b3lyp_water_sto3g(self):
        result = run_in_session(_water(), method="B3LYP", basis="STO-3G", verbose=0)
        assert result.converged
        # B3LYP/STO-3G H2O ~ -75.31 Ha
        assert abs(result.energy_hartree - (-75.31)) < 0.05
        assert result.method == "B3LYP"

    def test_pbe_water_sto3g(self):
        result = run_in_session(_water(), method="PBE", basis="STO-3G", verbose=0)
        assert result.converged
        assert result.energy_hartree < 0

    def test_pbe0_water_sto3g(self):
        result = run_in_session(_water(), method="PBE0", basis="STO-3G", verbose=0)
        assert result.converged
        assert result.energy_hartree < 0

    def test_m062x_water_sto3g(self):
        result = run_in_session(_water(), method="M06-2X", basis="STO-3G", verbose=0)
        assert result.converged
        assert result.energy_hartree < 0

    def test_dft_lower_energy_than_hf(self):
        """DFT includes correlation — should give lower energy than HF for same basis."""
        hf = run_in_session(_water(), method="RHF", basis="STO-3G", verbose=0)
        dft = run_in_session(_water(), method="B3LYP", basis="STO-3G", verbose=0)
        assert dft.energy_hartree < hf.energy_hartree

    def test_invalid_method_raises(self):
        # Previously used "CCSD" as the unsupported-method stand-in; CCSD
        # became a real method in M8.1 (session 54).
        with pytest.raises(ValueError):
            run_in_session(_water(), method="NONEXISTENT", basis="STO-3G", verbose=0)


# ---------------------------------------------------------------------------
# Preopt → run_in_session pipeline (the exact sequence _do_run uses)
# ---------------------------------------------------------------------------


@_requires_pyscf
class TestPreoptToCalculation:
    def test_pipeline_rhf(self):
        """Simulates _do_run with preopt enabled."""
        mol = _water()
        calc_mol, rmsd = preoptimize(mol)  # must unpack — not assign tuple
        assert isinstance(calc_mol, Molecule)  # regression check
        result = run_in_session(calc_mol, method="RHF", basis="STO-3G", verbose=0)
        assert result.converged

    def test_pipeline_b3lyp(self):
        mol = _water()
        calc_mol, _ = preoptimize(mol)
        result = run_in_session(calc_mol, method="B3LYP", basis="STO-3G", verbose=0)
        assert result.converged


# ---------------------------------------------------------------------------
# Thread safety — simulates the background thread pattern from _do_run
# ---------------------------------------------------------------------------


@_requires_pyscf
class TestThreadSafety:
    def test_run_in_session_from_thread(self):
        """run_in_session must work correctly when called from a non-main thread."""
        results = []
        errors = []

        def _worker():
            try:
                result = run_in_session(
                    _water(), method="RHF", basis="STO-3G", verbose=0
                )
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=60)

        assert not errors, f"Thread raised: {errors[0]}"
        assert len(results) == 1
        assert results[0].converged

    def test_preopt_then_calculate_from_thread(self):
        """Full _do_run pipeline from a background thread."""
        results = []
        errors = []

        def _worker():
            try:
                calc_mol, _ = preoptimize(_water())
                result = run_in_session(
                    calc_mol, method="RHF", basis="STO-3G", verbose=0
                )
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=60)

        assert not errors, f"Thread raised: {errors[0]}"
        assert results[0].converged
