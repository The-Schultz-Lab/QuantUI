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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
