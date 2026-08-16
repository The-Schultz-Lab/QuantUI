"""Spin-state / multiplicity suggestion engine (M-METAL MET.5).

The assertions below are the textbook cases the suggestions must reproduce —
this file doubles as the chemistry-review record for the d-count → spin-state
logic. Pure logic, no PySCF.
"""

from __future__ import annotations

import pytest

from quantui.spin_presets import (
    d_electron_count,
    suggest_spin_states,
    supported_metals,
)


class TestDCount:
    @pytest.mark.parametrize(
        "element,ox,expected",
        [
            ("Sc", 3, 0),
            ("Ti", 3, 1),
            ("V", 3, 2),
            ("Cr", 3, 3),
            ("Mn", 2, 5),
            ("Fe", 3, 5),
            ("Fe", 2, 6),
            ("Co", 3, 6),
            ("Co", 2, 7),
            ("Ni", 2, 8),
            ("Cu", 2, 9),
            ("Zn", 2, 10),
            ("Pt", 2, 8),
            ("Pt", 4, 6),
            ("Ru", 2, 6),
            ("Pd", 2, 8),
        ],
    )
    def test_d_count(self, element, ox, expected):
        assert d_electron_count(element, ox) == expected

    def test_unsupported_element(self):
        with pytest.raises(ValueError):
            d_electron_count("Xx", 2)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            d_electron_count("Sc", 5)  # d-2, impossible


def _mults(element, ox, geometry="octahedral"):
    s = suggest_spin_states(element, ox, geometry)
    return s, sorted(st.multiplicity for st in s.states)


class TestOctahedralUnambiguous:
    @pytest.mark.parametrize(
        "element,ox,mult",
        [
            ("Sc", 3, 1),  # d0
            ("Ti", 3, 2),  # d1
            ("V", 3, 3),  # d2
            ("Cr", 3, 4),  # d3 quartet
            ("Ni", 2, 3),  # d8 triplet
            ("Cu", 2, 2),  # d9 doublet
            ("Zn", 2, 1),  # d10 singlet
        ],
    )
    def test_single_state(self, element, ox, mult):
        s, mults = _mults(element, ox)
        assert not s.is_ambiguous
        assert mults == [mult]


class TestOctahedralHighLowSpin:
    @pytest.mark.parametrize(
        "element,ox,hs_mult,ls_mult",
        [
            ("Cr", 2, 5, 3),  # d4: HS quintet / LS triplet
            ("Mn", 2, 6, 2),  # d5: HS sextet / LS doublet
            ("Fe", 3, 6, 2),  # d5
            ("Fe", 2, 5, 1),  # d6: HS quintet / LS singlet
            ("Co", 3, 5, 1),  # d6
            ("Ru", 2, 5, 1),  # d6 (4d — same d-count rules)
            ("Co", 2, 4, 2),  # d7: HS quartet / LS doublet
        ],
    )
    def test_two_states(self, element, ox, hs_mult, ls_mult):
        s, mults = _mults(element, ox)
        assert s.is_ambiguous
        assert mults == sorted([hs_mult, ls_mult])
        # Labels present and the numbers pair with the right label.
        by_label = {st.label: st.multiplicity for st in s.states}
        assert by_label["high-spin"] == hs_mult
        assert by_label["low-spin"] == ls_mult
        assert "high-spin" in s.explanation and "low-spin" in s.explanation


class TestSquarePlanar:
    def test_d8_is_diamagnetic_singlet(self):
        # Cisplatin's Pt(II): square-planar d8 → all paired, multiplicity 1.
        s, mults = _mults("Pt", 2, "square_planar")
        assert not s.is_ambiguous
        assert mults == [1]
        assert "diamagnetic" in s.explanation

    def test_pd_ii_square_planar_singlet(self):
        _, mults = _mults("Pd", 2, "square_planar")
        assert mults == [1]


class TestTetrahedral:
    @pytest.mark.parametrize(
        "element,ox,mult",
        [
            ("Fe", 2, 5),  # d6 tetrahedral → high-spin quintet
            ("Co", 2, 4),  # d7 → quartet
            ("Ni", 2, 3),  # d8 → triplet
        ],
    )
    def test_always_high_spin_single_state(self, element, ox, mult):
        s, mults = _mults(element, ox, "tetrahedral")
        assert not s.is_ambiguous
        assert mults == [mult]


class TestMisc:
    def test_supported_metals_covers_scope(self):
        metals = supported_metals()
        for el in ("Sc", "Zn", "Fe", "Co", "Ru", "Rh", "Pd", "Pt"):
            assert el in metals

    def test_bad_geometry(self):
        with pytest.raises(ValueError):
            suggest_spin_states("Fe", 3, "linear")

    def test_multiplicity_is_unpaired_plus_one(self):
        for st in suggest_spin_states("Fe", 3).states:
            assert st.multiplicity == st.n_unpaired + 1
