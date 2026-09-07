"""
Tests for quantui.freq_calc — ThermoData dataclass and FreqResult thermo field.

Test strategy
-------------
* ThermoData and FreqResult dataclass tests run unconditionally — no PySCF needed.
* run_freq_calc() tests are marked pyscf_only and skipped on Windows.
"""

from __future__ import annotations

import pytest

from quantui.freq_calc import FreqResult, ThermoData

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
# Helpers
# ---------------------------------------------------------------------------

_HARTREE_TO_JMOL = 2625499.6


def _make_thermo(**overrides) -> ThermoData:
    defaults = dict(
        zpve_hartree=0.020734,
        H_hartree=-76.003456,
        S_jmol=198.7,
        G_hartree=-76.032952,
    )
    defaults.update(overrides)
    return ThermoData(**defaults)


def _make_freq_result(**overrides) -> FreqResult:
    defaults = dict(
        energy_hartree=-76.023190,
        homo_lumo_gap_ev=9.5,
        converged=True,
        n_iterations=10,
        method="RHF",
        basis="STO-3G",
        formula="H2O",
        frequencies_cm1=[1600.0, 3600.0, 3800.0],
        zpve_hartree=0.020734,
    )
    defaults.update(overrides)
    return FreqResult(**defaults)


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


# ============================================================================
# ThermoData dataclass
# ============================================================================


class TestThermoData:
    def test_fields_stored(self):
        td = _make_thermo()
        assert td.zpve_hartree == pytest.approx(0.020734)
        assert td.H_hartree == pytest.approx(-76.003456)
        assert td.S_jmol == pytest.approx(198.7)
        assert td.G_hartree == pytest.approx(-76.032952)

    def test_default_temperature(self):
        td = _make_thermo()
        assert td.temperature_k == pytest.approx(298.15)

    def test_g_less_than_h(self):
        """G = H - T*S, so G < H for positive entropy."""
        td = _make_thermo()
        assert td.G_hartree < td.H_hartree

    def test_g_consistent_with_h_and_s(self):
        """Verify G ≈ H - T*S within floating-point tolerance."""
        td = _make_thermo()
        expected_g = td.H_hartree - td.temperature_k * td.S_jmol / _HARTREE_TO_JMOL
        assert td.G_hartree == pytest.approx(expected_g, abs=0.01)


# ============================================================================
# FreqResult.thermo field
# ============================================================================


class TestFreqResultThermoField:
    def test_thermo_defaults_to_none(self):
        result = _make_freq_result()
        assert result.thermo is None

    def test_thermo_stored_when_provided(self):
        td = _make_thermo()
        result = _make_freq_result(thermo=td)
        assert result.thermo is td

    def test_thermo_h_accessible(self):
        td = _make_thermo(H_hartree=-76.003456)
        result = _make_freq_result(thermo=td)
        assert result.thermo.H_hartree == pytest.approx(-76.003456)  # type: ignore[union-attr]

    def test_thermo_s_accessible(self):
        td = _make_thermo(S_jmol=198.7)
        result = _make_freq_result(thermo=td)
        assert result.thermo.S_jmol == pytest.approx(198.7)  # type: ignore[union-attr]

    def test_thermo_g_accessible(self):
        td = _make_thermo(G_hartree=-76.032952)
        result = _make_freq_result(thermo=td)
        assert result.thermo.G_hartree == pytest.approx(-76.032952)  # type: ignore[union-attr]


# ============================================================================
# run_freq_calc() — PySCF required
# ============================================================================


class TestRunFreqCalcThermo:
    @pyscf_only
    @pytest.mark.slow
    def test_thermo_populated_for_rhf(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.thermo is not None

    @pyscf_only
    @pytest.mark.slow
    def test_scf_variant_reports_rks_for_closed_shell_dft(self):
        """M-UX2 UXP2.10 — confirms the wiring, mirroring the identical
        RHF/UHF/RKS/UKS dispatch already thoroughly tested in
        test_session_calc.py::TestScfVariantProvenance."""
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="B3LYP", basis="STO-3G")
        assert result.scf_variant == "RKS"

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_h_is_finite(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert abs(result.thermo.H_hartree) < 1e6

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_s_positive(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert result.thermo.S_jmol > 0

    @pyscf_only
    @pytest.mark.slow
    def test_thermo_g_less_than_h(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.thermo is not None:
            assert result.thermo.G_hartree < result.thermo.H_hartree


# ============================================================================
# pyscf_mol_atom unit convention (H2 audit fix, 2026-07-14)
# ============================================================================


class TestPyscfMolAtomUnits:
    """``pyscf_mol_atom`` must be Angstrom, matching session_calc/optimizer.

    Regression for a bug where freq_calc built this field from PySCF's
    internal ``mol._atom`` (always Bohr), while every consumer (Molden
    export, cube generation, orbital replay) assumes Angstrom — silently
    inflating exported geometries ~1.89x for Frequency results only.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_pyscf_mol_atom_matches_input_geometry_in_angstrom(self):
        from quantui.freq_calc import run_freq_calc

        molecule = _water()
        result = run_freq_calc(molecule, method="RHF", basis="STO-3G")
        assert result.pyscf_mol_atom is not None
        for (sym, coords), (orig_sym, orig_coords) in zip(
            result.pyscf_mol_atom, zip(molecule.atoms, molecule.coordinates)
        ):
            assert sym == orig_sym
            for c, orig_c in zip(coords, orig_coords):
                assert c == pytest.approx(orig_c, abs=1e-9)


# ============================================================================
# Post-HF method guard (M2 audit fix, 2026-07-14)
# ============================================================================


class TestRunFreqCalcPostHfGuard:
    """Post-HF methods raise a clear ValueError instead of a cryptic LibXC error.

    Regression: run_freq_calc() had no special-casing for MP2/CCSD/CCSD(T)
    — the SCF-selection branch silently treated them as a DFT xc functional
    (mf.xc = "CCSD"), failing deep inside PySCF with "LibXCFunctional: name
    'CCSD' not found" instead of a clear message. The guard fires before any
    PySCF import, so it needs neither PySCF nor ASE.
    """

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        from quantui.freq_calc import run_freq_calc

        with pytest.raises(ValueError, match="post-HF"):
            run_freq_calc(_water(), method=method, basis="STO-3G")


# ============================================================================
# IR intensities — PySCF required
# ============================================================================


class TestIRIntensities:
    """make_ir_intensity() should return real km/mol values for H₂O / RHF.

    H₂O has 3 vibrational modes: bending (~1600 cm⁻¹), symmetric stretch
    (~3700 cm⁻¹), antisymmetric stretch (~3800 cm⁻¹).  All three are
    IR-active (A1 and B2 symmetry), so all intensities must be positive.
    """

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_non_empty(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.ir_intensities, "ir_intensities should be non-empty for H₂O/RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_length_matches_frequencies(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert len(result.ir_intensities) == len(result.frequencies_cm1)

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_all_non_negative(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        for i, inten in enumerate(result.ir_intensities):
            assert inten >= 0, f"mode {i}: intensity {inten:.3f} < 0"

    @pyscf_only
    @pytest.mark.slow
    def test_ir_intensities_physically_reasonable(self):
        """All H₂O modes are IR-active; max intensity should be > 1 km/mol."""
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        if result.ir_intensities:
            assert max(result.ir_intensities) > 1.0


# ============================================================================
# IR-intensity inner-loop dm0/method dispatch (M5 audit fix, 2026-07-14)
# ============================================================================


class TestIrIntensityUhfClosedShellDispatch:
    """The inner displaced-SCF loop must dispatch on dm0's actual shape.

    Regression: the serial (_displaced_scf_dipole in freq_calc.py) and
    parallel (run_displaced_scf in freq_ir_workers.py) inner loops both
    picked RHF/UHF based on mol.spin == 0 alone. That agrees with the
    parent SCF's actual type only when the user's method choice matches
    the molecule's natural spin state. Explicitly selecting UHF for a
    closed-shell molecule (mol.spin == 0, but the parent mf — and its
    dm0 — is still UHF-shaped, a legitimate technique for probing
    symmetry-broken solutions) used to raise a shape-mismatch ValueError
    deep in PySCF, silently dropping IR intensities for the whole run
    (caught by the broad except around the entire IR-intensity block).
    """

    @pyscf_only
    @pytest.mark.slow
    def test_uhf_on_closed_shell_molecule_still_gets_ir_intensities(self):
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="UHF", basis="STO-3G")
        assert result.ir_intensities, (
            "UHF on a closed-shell molecule should still produce IR "
            "intensities via the serial inner-loop dispatch fix"
        )
        assert len(result.ir_intensities) == len(result.frequencies_cm1)

    def test_freq_ir_workers_dispatches_on_dm0_shape_not_spin(self):
        """Unit-level check of the parallel worker's dispatch logic directly.

        Builds a UHF parent for a closed-shell (spin=0) molecule — the
        exact scenario that used to crash — and confirms the worker
        picks UHF (matching the (2, nao, nao) dm0) rather than RHF
        (which would raise on that dm0 shape).
        """
        pytest.importorskip("pyscf")
        import os
        import pickle
        import tempfile

        from pyscf import gto, scf

        from quantui.freq_ir_workers import init_worker, run_displaced_scf

        atom_str = "O 0 0 0.119; H 0 0.763 -0.477; H 0 -0.763 -0.477"
        mol = gto.M(atom=atom_str, basis="sto-3g", spin=0, charge=0, verbose=0)
        mf = scf.UHF(mol)
        mf.kernel()
        dm0 = mf.make_rdm1()
        assert dm0.ndim == 3  # UHF-shaped despite mol.spin == 0

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        try:
            pickle.dump(dm0, tmp)
            tmp.close()
            init_worker(atom_str, "sto-3g", 0, 0, None, tmp.name, 1)
            coords = mol.atom_coords(unit="Bohr").flatten().tolist()
            dip = run_displaced_scf(coords)  # must not raise
            assert len(dip) == 3
        finally:
            os.unlink(tmp.name)

    @pyscf_only
    @pytest.mark.slow
    def test_parallel_failure_falls_back_to_serial(self, monkeypatch):
        """A parallel-path failure must fall back to serial, not give up.

        Regression: run_displaced_scf's own docstring claims "The
        freq_calc driver catches such failures and falls back to the
        serial loop so the user's calc still completes" — but no such
        fallback existed; any single worker failure propagated out to
        the broad except around the whole IR-intensity block, dropping
        IR intensities entirely. Forces the parallel path to be
        selected and to fail immediately, then checks IR intensities
        still come out via the serial fallback.
        """
        import concurrent.futures as cf
        import io as _io

        import quantui.freq_ir_workers as ir_workers
        from quantui.freq_calc import run_freq_calc

        monkeypatch.setattr(ir_workers, "parallel_enabled_for_run", lambda **kw: True)

        class _FailingExecutor:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                raise RuntimeError("simulated worker pool failure")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(cf, "ProcessPoolExecutor", _FailingExecutor)

        buf = _io.StringIO()
        result = run_freq_calc(
            _water(), method="RHF", basis="STO-3G", progress_stream=buf
        )
        log = buf.getvalue()
        assert "falling back to serial" in log
        assert result.ir_intensities, (
            "IR intensities should still be populated via the serial "
            "fallback after a simulated parallel-path failure"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
