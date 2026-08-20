"""Pre-run guards for inorganic / metal calculations — M-METAL MET.5.

Turn two cryptic mid-run PySCF crashes into a clear message before the run:
a basis with no parameters for an element, and a charge/multiplicity that is
impossible for the electron count. The charge/multiplicity logic is pure; the
basis check uses PySCF's loader and is gated.
"""

from __future__ import annotations

import pytest

from quantui.inorganic_guards import (
    check_basis_coverage,
    check_charge_multiplicity,
    ecp_for_basis,
    preflight_messages,
)

# cisplatin (PtCl2(NH3)2) as element list + a geometry, reused across ECP tests.
_CISPLATIN_ELEMENTS = ["Pt", "Cl", "Cl", "N", "N"] + ["H"] * 6
_CISPLATIN_ATOM = (
    "Pt 0 0 0; Cl 1.648 1.648 0; Cl -1.648 1.648 0; "
    "N -1.45 -1.45 0; N 1.45 -1.45 0; "
    "H -1.01 -2.37 0; H -2.03 -1.35 0.833; H -2.03 -1.35 -0.833; "
    "H 2.37 -1.01 0; H 1.35 -2.03 0.833; H 1.35 -2.03 -0.833"
)

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


class TestChargeMultiplicity:
    def test_even_electrons_singlet_ok(self):
        assert check_charge_multiplicity(36, 1) is None

    def test_odd_electrons_doublet_ok(self):
        assert check_charge_multiplicity(37, 2) is None

    def test_odd_electrons_singlet_is_flagged(self):
        # The exact cisplatin-adjacent trap: odd count with the default mult 1.
        msg = check_charge_multiplicity(37, 1)
        assert msg is not None
        assert "odd electron count needs an even multiplicity" in msg

    def test_even_electrons_doublet_is_flagged(self):
        msg = check_charge_multiplicity(36, 2)
        assert msg is not None
        assert "even electron count needs an odd multiplicity" in msg

    def test_multiplicity_below_one(self):
        assert check_charge_multiplicity(10, 0) is not None

    def test_more_unpaired_than_electrons(self):
        msg = check_charge_multiplicity(1, 4)  # 3 unpaired > 1 electron
        assert msg is not None
        assert "only 1" in msg


@pyscf_only
class TestBasisCoverage:
    def test_pople_basis_lacks_a_metal(self):
        msg = check_basis_coverage(["C", "H", "Pt"], "6-31G")
        assert msg is not None
        assert "Pt" in msg
        assert "def2" in msg

    def test_def2_covers_metals(self):
        assert check_basis_coverage(["C", "N", "Pt", "Zn"], "def2-SVP") is None

    def test_lanl2dz_covers_metals_and_ligands(self):
        # MET.5: LANL2DZ is offered for heavy metals and (unlike a mixed-basis
        # setup) its PySCF definition also covers the ligand atoms, so a whole
        # complex runs under it as QuantUI's single molecule-wide basis.
        assert check_basis_coverage(["C", "H", "N", "Cl", "Pt"], "LANL2DZ") is None
        assert check_basis_coverage(["Fe", "Ru", "Pd"], "LANL2DZ") is None

    def test_organic_basis_covers_organics(self):
        assert check_basis_coverage(["C", "H", "O", "N"], "6-31G*") is None


@pyscf_only
class TestEcpForBasis:
    """MET.5/MET.8: ECP-carrying bases must actually attach their ECP.

    Regression guard for the cisplatin geometry-opt divergence — LANL2DZ/def2
    were run with ``mol.ecp`` unset, so Pt kept all 78 electrons and the SCF and
    gradients were garbage.
    """

    def test_lanl2dz_selects_all_ecp_atoms(self):
        ecp = ecp_for_basis("LANL2DZ", _CISPLATIN_ELEMENTS)
        # LANL2DZ carries an ECP for every atom heavier than Ne — so Pt *and*
        # Cl — while the first-row N/H ligand atoms stay all-electron.
        assert ecp == {"Pt": "LANL2DZ", "Cl": "LANL2DZ"}

    def test_def2_selects_only_the_heavy_metal(self):
        # def2's ECP boundary is higher (Z >= 37): Cl is all-electron in def2,
        # so only Pt is selected — a real, correct difference from LANL2DZ.
        assert ecp_for_basis("def2-SVP", _CISPLATIN_ELEMENTS) == {"Pt": "def2-SVP"}
        assert ecp_for_basis("def2-TZVP", ["Pt"]) == {"Pt": "def2-TZVP"}

    def test_all_electron_basis_returns_empty(self):
        # Pople / cc / STO have no ECP table — even over a metal, and never raise.
        assert ecp_for_basis("6-31G", ["C", "H", "O", "N"]) == {}
        assert ecp_for_basis("6-31G", ["Pt", "Cl"]) == {}
        assert ecp_for_basis("cc-pVDZ", ["C", "H"]) == {}

    def test_light_only_molecule_under_ecp_basis_is_empty(self):
        # def2 over an organic: the bundled ECP applies to no atom present.
        assert ecp_for_basis("def2-SVP", ["C", "H", "O"]) == {}

    def test_duplicate_elements_collapse(self):
        assert ecp_for_basis("LANL2DZ", ["Pt", "Pt", "H", "H"]) == {"Pt": "LANL2DZ"}

    def test_ecp_drops_core_electrons_on_cisplatin(self):
        # The bug, stated as a number: all-electron cisplatin is 132 electrons.
        # LANL2DZ puts an ECP on Pt (removes 60) and both Cl (removes 10 each),
        # so the correct count is 132 - 60 - 20 = 52.
        from pyscf import gto

        mol = gto.Mole()
        mol.atom = _CISPLATIN_ATOM
        mol.basis = "LANL2DZ"
        mol.ecp = ecp_for_basis("LANL2DZ", _CISPLATIN_ELEMENTS)
        mol.build()
        assert mol.nelectron == 52

    def test_def2_ecp_drops_only_pt_core(self):
        # def2 keeps Cl all-electron, so only Pt's 60 core electrons go:
        # 132 - 60 = 72. Confirms the helper tracks each basis's own boundary.
        from pyscf import gto

        mol = gto.Mole()
        mol.atom = _CISPLATIN_ATOM
        mol.basis = "def2-SVP"
        mol.ecp = ecp_for_basis("def2-SVP", _CISPLATIN_ELEMENTS)
        mol.build()
        assert mol.nelectron == 72

    def test_no_ecp_keeps_all_electrons(self):
        # Same molecule, ECP omitted — the pre-fix behaviour, pinned so the
        # 60-electron difference is unmistakable.
        from pyscf import gto

        mol = gto.Mole()
        mol.atom = _CISPLATIN_ATOM
        mol.basis = "LANL2DZ"
        mol.build()
        assert mol.nelectron == 132


@pyscf_only
class TestPreflight:
    def test_clean_organic_run_has_no_problems(self):
        # water, singlet, an organic basis — nothing to flag.
        assert preflight_messages(["O", "H", "H"], 10, "6-31G", 1) == []

    def test_metal_on_pople_basis_is_blocked(self):
        # cisplatin (H6Cl2N2Pt), even electrons so only the basis fails.
        elements = ["Pt", "Cl", "Cl", "N", "N"] + ["H"] * 6
        problems = preflight_messages(elements, 132, "6-31G", 1)
        assert len(problems) == 1
        assert "Pt" in problems[0]

    def test_both_problems_reported_together(self):
        # A metal on a Pople basis AND an impossible multiplicity.
        problems = preflight_messages(["Pt", "H"], 79, "6-31G", 1)
        assert len(problems) == 2

    def test_def2_with_right_spin_is_clean(self):
        assert preflight_messages(["Pt", "H"], 79, "def2-SVP", 2) == []
