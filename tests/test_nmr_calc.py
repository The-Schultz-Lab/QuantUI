"""
Tests for quantui.nmr_calc — M5 acceptance criteria.

NMRResult dataclass tests run unconditionally.
run_nmr_calc() tests are PySCF-gated.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule
from quantui.nmr_calc import NMRResult, run_nmr_calc

# ---------------------------------------------------------------------------
# PySCF gate
# ---------------------------------------------------------------------------

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE, reason="PySCF not installed (Linux/macOS/WSL only)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
    )


def _methane() -> Molecule:
    return Molecule(
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    )


# ---------------------------------------------------------------------------
# NMRResult dataclass
# ---------------------------------------------------------------------------


class TestNMRResult:
    def _make_result(self) -> NMRResult:
        return NMRResult(
            atom_symbols=["O", "H", "H"],
            shielding_iso_ppm=[320.1, 28.5, 28.5],
            chemical_shifts_ppm={1: 3.22, 2: 3.22},
            method="B3LYP",
            basis="6-31G*",
            formula="H2O",
        )

    def test_h_shifts_only_returns_H(self):
        r = self._make_result()
        hs = r.h_shifts()
        assert all(r.atom_symbols[i] == "H" for i, _ in hs)

    def test_h_shifts_count(self):
        r = self._make_result()
        assert len(r.h_shifts()) == 2

    def test_c_shifts_empty_for_water(self):
        r = self._make_result()
        assert r.c_shifts() == []

    def test_c_shifts_for_methane(self):
        r = NMRResult(
            atom_symbols=["C", "H", "H", "H", "H"],
            shielding_iso_ppm=[150.0, 29.0, 29.0, 29.0, 29.0],
            chemical_shifts_ppm={0: 33.71, 1: 2.72, 2: 2.72, 3: 2.72, 4: 2.72},
            method="B3LYP",
            basis="6-31G*",
            formula="CH4",
        )
        cs = r.c_shifts()
        assert len(cs) == 1
        assert cs[0][0] == 0  # atom index 0 = C

    def test_default_reference_is_tms(self):
        r = self._make_result()
        assert r.reference_compound == "TMS"

    def test_default_converged_true(self):
        r = self._make_result()
        assert r.converged is True

    def test_h_shifts_sorted_by_index(self):
        r = NMRResult(
            atom_symbols=["H", "C", "H"],
            shielding_iso_ppm=[29.0, 150.0, 28.0],
            chemical_shifts_ppm={0: 2.72, 1: 33.71, 2: 3.72},
            method="B3LYP",
            basis="6-31G*",
            formula="CH2",
        )
        indices = [i for i, _ in r.h_shifts()]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# config NMR constants
# ---------------------------------------------------------------------------


class TestNMRConfig:
    def test_reference_shieldings_has_b3lyp(self):
        from quantui.config import NMR_REFERENCE_SHIELDINGS

        assert "B3LYP/6-31G*" in NMR_REFERENCE_SHIELDINGS

    def test_reference_shieldings_has_H_and_C(self):
        from quantui.config import NMR_REFERENCE_SHIELDINGS

        entry = NMR_REFERENCE_SHIELDINGS["B3LYP/6-31G*"]
        assert "H" in entry
        assert "C" in entry

    def test_default_reference_present(self):
        from quantui.config import NMR_DEFAULT_REFERENCE

        assert "H" in NMR_DEFAULT_REFERENCE
        assert "C" in NMR_DEFAULT_REFERENCE

    def test_h_reference_plausible(self):
        from quantui.config import NMR_DEFAULT_REFERENCE

        assert 25.0 < NMR_DEFAULT_REFERENCE["H"] < 40.0

    def test_c_reference_plausible(self):
        from quantui.config import NMR_DEFAULT_REFERENCE

        assert 150.0 < NMR_DEFAULT_REFERENCE["C"] < 220.0


class TestResolveNmrReference:
    """M4 audit fix (2026-07-14): record which reference was actually used.

    Regression: any method/basis combination not in
    config.NMR_REFERENCE_SHIELDINGS silently used the B3LYP/6-31G*
    constants with no record anywhere that a substitution happened —
    chemical shifts could be off by a few ppm with no indication to the
    student. resolve_nmr_reference() (and the NMRResult.reference_key /
    is_fallback_reference fields it feeds) makes that substitution visible.
    No PySCF needed — pure lookup logic.
    """

    def test_exact_match_is_not_fallback(self):
        from quantui.nmr_calc import resolve_nmr_reference

        ref_map, key, is_fallback = resolve_nmr_reference("B3LYP", "6-31G*")
        assert key == "B3LYP/6-31G*"
        assert is_fallback is False
        assert ref_map == {"H": 31.72, "C": 183.71}

    def test_unlisted_combination_is_fallback(self):
        from quantui.nmr_calc import resolve_nmr_reference

        ref_map, key, is_fallback = resolve_nmr_reference("CAM-B3LYP", "6-31G*")
        assert is_fallback is True
        assert key == "B3LYP/6-31G*"
        from quantui.config import NMR_DEFAULT_REFERENCE

        assert ref_map == NMR_DEFAULT_REFERENCE

    def test_lookup_is_case_insensitive(self):
        from quantui.nmr_calc import resolve_nmr_reference

        ref_map, key, is_fallback = resolve_nmr_reference("rhf", "6-31g*")
        assert is_fallback is False
        assert key == "RHF/6-31G*"


# ---------------------------------------------------------------------------
# run_nmr_calc — PySCF-gated
# ---------------------------------------------------------------------------


class TestRunNMRCalc:
    def test_raises_importerror_without_pyscf(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "pyscf", None)
        with pytest.raises(ImportError, match="PySCF"):
            run_nmr_calc(_water())

    @pyscf_only
    @pytest.mark.slow
    def test_water_returns_nmr_result(self):
        result = run_nmr_calc(_water(), method="RHF", basis="STO-3G")
        assert isinstance(result, NMRResult)

    @pyscf_only
    @pytest.mark.slow
    def test_water_has_two_h_shifts(self):
        result = run_nmr_calc(_water(), method="RHF", basis="STO-3G")
        assert len(result.h_shifts()) == 2

    @pyscf_only
    @pytest.mark.slow
    def test_water_no_c_shifts(self):
        result = run_nmr_calc(_water(), method="RHF", basis="STO-3G")
        assert result.c_shifts() == []

    @pyscf_only
    @pytest.mark.slow
    def test_methane_has_c_and_h_shifts(self):
        result = run_nmr_calc(_methane(), method="RHF", basis="STO-3G")
        assert len(result.c_shifts()) == 1
        assert len(result.h_shifts()) == 4

    @pyscf_only
    @pytest.mark.slow
    def test_water_h_shifts_reasonable_range(self):
        result = run_nmr_calc(_water(), method="B3LYP", basis="6-31G*")
        for _i, delta in result.h_shifts():
            # Water ¹H is typically 1–5 ppm at this level (gas phase)
            assert -5.0 < delta < 15.0, f"Unexpected ¹H shift {delta:.2f} ppm"

    @pyscf_only
    @pytest.mark.slow
    def test_formula_matches_molecule(self):
        result = run_nmr_calc(_water(), method="RHF", basis="STO-3G")
        assert "O" in result.formula
        assert "H" in result.formula

    @pyscf_only
    @pytest.mark.slow
    def test_shielding_iso_length_matches_atoms(self):
        mol = _water()
        result = run_nmr_calc(mol, method="RHF", basis="STO-3G")
        assert len(result.shielding_iso_ppm) == len(list(mol.atoms))

    @pyscf_only
    @pytest.mark.slow
    def test_exact_match_reference_not_flagged_as_fallback(self):
        """M4 audit fix: RHF/STO-3G is tabulated -> not a fallback."""
        result = run_nmr_calc(_water(), method="RHF", basis="STO-3G")
        assert result.reference_key == "RHF/STO-3G"
        assert result.is_fallback_reference is False

    @pyscf_only
    @pytest.mark.slow
    def test_untabulated_combination_flagged_as_fallback(self):
        """M4 audit fix: an untabulated method/basis is flagged, not silent."""
        result = run_nmr_calc(_water(), method="CAM-B3LYP", basis="6-31G*")
        assert result.is_fallback_reference is True
        assert result.reference_key == "B3LYP/6-31G*"


class TestNmrCompatPatchIdempotency:
    """L audit fix: the pyscf.prop.nmr compat patches must apply once per
    process, not redefine + reassign the same closures on every NMR call.
    """

    @pyscf_only
    def test_patch_functions_are_not_redefined_on_repeat_calls(self):
        import pyscf.prop.nmr.rhf as _prop_nmr_rhf
        import pyscf.prop.nmr.rks as _prop_nmr_rks

        from quantui.nmr_calc import _ensure_nmr_compat_patches_applied

        _ensure_nmr_compat_patches_applied()
        gen_vind_first = _prop_nmr_rhf.gen_vind
        get_vxc_giao_first = _prop_nmr_rks.get_vxc_giao

        _ensure_nmr_compat_patches_applied()
        # Same function object — the second call must be a no-op, not a
        # fresh `def` + reassignment (which would install a *different*
        # (if behaviorally identical) closure each time).
        assert _prop_nmr_rhf.gen_vind is gen_vind_first
        assert _prop_nmr_rks.get_vxc_giao is get_vxc_giao_first

    @pyscf_only
    def test_patched_functions_carry_version_sentinel(self):
        import pyscf.prop.nmr.rhf as _prop_nmr_rhf
        import pyscf.prop.nmr.rks as _prop_nmr_rks

        from quantui.nmr_calc import (
            _NMR_COMPAT_PATCH_VERSION,
            _NMR_PATCH_VERSION_ATTR,
            _ensure_nmr_compat_patches_applied,
        )

        _ensure_nmr_compat_patches_applied()
        assert (
            getattr(_prop_nmr_rhf.gen_vind, _NMR_PATCH_VERSION_ATTR, None)
            == _NMR_COMPAT_PATCH_VERSION
        )
        assert (
            getattr(_prop_nmr_rks.get_vxc_giao, _NMR_PATCH_VERSION_ATTR, None)
            == _NMR_COMPAT_PATCH_VERSION
        )


# ============================================================================
# Post-HF method guard (M2 audit fix, 2026-07-14)
# ============================================================================


class TestRunNmrCalcPostHfGuard:
    """Post-HF methods raise a clear ValueError instead of a cryptic LibXC error.

    Regression: run_nmr_calc() had no special-casing for MP2/CCSD/CCSD(T)
    — the SCF-selection branch silently treated them as a DFT xc functional
    (mf.xc = "CCSD"), failing deep inside PySCF with "LibXCFunctional: name
    'CCSD' not found" instead of a clear message. The guard fires before any
    PySCF import, so it needs no PySCF.
    """

    @pytest.mark.parametrize("method", ["MP2", "CCSD", "CCSD(T)"])
    def test_post_hf_method_raises_value_error(self, method):
        with pytest.raises(ValueError, match="post-HF"):
            run_nmr_calc(_water(), method=method, basis="STO-3G")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
