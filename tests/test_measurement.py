"""Tests for quantui.measurement (M-MEASURE MEAS.1) — pure geometry, no browser."""

from __future__ import annotations

import pytest

from quantui.ase_bridge import ASE_AVAILABLE
from quantui.measurement import angle, atom_label, describe_picks, dihedral, distance
from quantui.molecule import Molecule

ase_only = pytest.mark.skipif(not ASE_AVAILABLE, reason="ase not installed")


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


def _linear_plus_one() -> Molecule:
    # Atoms 0-1-2 collinear (a linear triatomic), atom 3 off-axis: the
    # dihedral 0-1-2-3 has an undefined inner-angle plane.
    return Molecule(
        atoms=["H", "H", "H", "H"],
        coordinates=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
    )


@ase_only
class TestDistance:
    def test_water_oh_bond_length(self):
        d = distance(_water(), 0, 1)
        assert d == pytest.approx(0.958, abs=1e-3)

    def test_distance_is_symmetric(self):
        mol = _water()
        assert distance(mol, 0, 1) == pytest.approx(distance(mol, 1, 0))

    def test_same_atom_is_zero(self):
        assert distance(_water(), 0, 0) == pytest.approx(0.0)


@ase_only
class TestAngle:
    def test_water_h_o_h_angle(self):
        a = angle(_water(), 1, 0, 2)
        assert a == pytest.approx(104.5, abs=0.2)

    def test_methane_tetrahedral_angle(self):
        a = angle(_methane(), 1, 0, 2)
        assert a == pytest.approx(109.47, abs=0.1)

    def test_linear_angle_is_180(self):
        a = angle(_linear_plus_one(), 0, 1, 2)
        assert a == pytest.approx(180.0, abs=1e-6)

    def test_duplicate_vertex_neighbor_is_undefined(self):
        with pytest.raises(ZeroDivisionError):
            angle(_water(), 0, 1, 1)


@ase_only
class TestDihedral:
    def test_linear_inner_angle_is_undefined(self):
        # The documented edge case: atoms 0-1-2 are collinear, so the
        # dihedral 0-1-2-3 has no well-defined plane.
        with pytest.raises(ZeroDivisionError):
            dihedral(_linear_plus_one(), 0, 1, 2, 3)

    def test_well_defined_dihedral_returns_a_float(self):
        # Non-degenerate 4-atom chain (staggered-ish, no collinear triple).
        mol = Molecule(
            atoms=["H", "H", "H", "H"],
            coordinates=[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        )
        dh = dihedral(mol, 0, 1, 2, 3)
        assert isinstance(dh, float)
        # ASE's get_dihedral returns degrees in [0, 360), not signed.
        assert 0.0 <= dh <= 360.0


class TestAtomLabel:
    def test_one_based_element_prefixed(self):
        mol = _water()
        assert atom_label(mol, 0) == "O1"
        assert atom_label(mol, 1) == "H2"
        assert atom_label(mol, 2) == "H3"


@ase_only
class TestDescribePicks:
    def test_no_picks(self):
        assert "Click an atom" in describe_picks(_water(), [])

    def test_one_pick_shows_only_label(self):
        text = describe_picks(_water(), [0])
        assert text == "Picked: O1"

    def test_two_picks_show_bond_length(self):
        text = describe_picks(_water(), [0, 1])
        assert "O1" in text
        assert "H2" in text
        assert "Å" in text
        assert "0.958" in text

    def test_three_picks_add_angle(self):
        text = describe_picks(_water(), [1, 0, 2])
        assert "°" in text
        assert "104.4" in text or "104.5" in text

    def test_four_picks_add_dihedral(self):
        mol = Molecule(
            atoms=["H", "H", "H", "H"],
            coordinates=[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        )
        text = describe_picks(mol, [0, 1, 2, 3])
        assert text.count("°") == 2  # angle AND dihedral both shown

    def test_four_picks_with_collinear_inner_angle_reports_undefined(self):
        text = describe_picks(_linear_plus_one(), [0, 1, 2, 3])
        assert "undefined" in text.lower()
        assert "collinear" in text.lower()
