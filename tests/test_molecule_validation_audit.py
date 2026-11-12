"""Molecule / XYZ validation hardening (audit DATA-8).

Non-finite coordinates, impossible spin states (more unpaired electrons than
electrons, or a charge that leaves a negative electron count) and truncated
XYZ pastes used to pass validation; numpy charges/rows were rejected with a
misleading message; and ``Molecule`` aliased the caller's lists.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantui import utils
from quantui.molecule import Molecule, parse_xyz_input

H2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]


class TestNonFiniteCoordinates:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_validator_rejects(self, bad):
        assert utils.validate_coordinates([0.0, 0.0, bad]) is False

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_molecule_rejects(self, bad):
        with pytest.raises(ValueError, match="finite"):
            Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, bad]])

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf", "NaN"])
    def test_paste_parser_rejects(self, token):
        with pytest.raises(ValueError, match="finite"):
            parse_xyz_input(f"H 0 0 0\nH 0 0 {token}")


class TestImpossibleSpinStates:
    def test_multiplicity_above_electron_count(self):
        # H2 has 2 electrons: at most 2 unpaired (triplet). Quintet passes
        # the parity check but is impossible.
        with pytest.raises(ValueError, match="only 2"):
            Molecule(["H", "H"], H2, multiplicity=5)

    def test_triplet_h2_still_allowed(self):
        assert Molecule(["H", "H"], H2, multiplicity=3).multiplicity == 3

    def test_negative_electron_count(self):
        with pytest.raises(ValueError, match="Charge"):
            Molecule(["H"], [[0.0, 0.0, 0.0]], charge=2)
        with pytest.raises(ValueError, match="Charge"):
            Molecule(["H", "H"], H2, charge=4)

    def test_bare_proton_is_allowed(self):
        assert Molecule(["H"], [[0.0, 0.0, 0.0]], charge=1).get_electron_count() == 0

    def test_geometry_only_load_still_skips_spin_checks(self):
        # validate_spin=False (XYZ/upload loads) keeps editing charge/mult
        # separately possible; the pre-run guard checks them later.
        m = Molecule(["H", "H"], H2, multiplicity=5, validate_spin=False)
        assert m.multiplicity == 5


class TestTruncatedXyz:
    def test_header_count_mismatch_raises(self):
        text = "5\ntitle\nO 0 0 0\nH 0 0 0.96\nH 0 0.93 -0.24\n"
        with pytest.raises(ValueError, match="header says 5 atoms"):
            parse_xyz_input(text)

    def test_matching_header_still_parses(self):
        atoms, _ = parse_xyz_input("2\ntitle\nH 0 0 0\nH 0 0 0.74\n")
        assert atoms == ["H", "H"]


class TestNumpyInputsAndCopies:
    def test_numpy_charge_and_rows_accepted(self):
        m = Molecule(
            np.array(["O", "H", "H"]).tolist(),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]]),
            charge=np.int64(0),
            multiplicity=np.int64(1),
        )
        assert m.charge == 0 and type(m.charge) is int
        assert type(m.multiplicity) is int
        assert all(type(v) is float for row in m.coordinates for v in row)

    def test_validate_charge_accepts_numpy_int(self):
        assert utils.validate_charge(np.int64(-1)) is True
        assert utils.validate_coordinates(np.array([0.0, 1.0, 2.0])) is True

    def test_caller_lists_are_copied(self):
        atoms = ["H", "H"]
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        m = Molecule(atoms, coords)
        atoms.append("O")
        coords[1][2] = 9.0
        coords.append([1.0, 1.0, 1.0])
        assert m.atoms == ["H", "H"]
        assert m.coordinates == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
