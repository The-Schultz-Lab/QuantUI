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
    preflight_messages,
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

    def test_organic_basis_covers_organics(self):
        assert check_basis_coverage(["C", "H", "O", "N"], "6-31G*") is None


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
