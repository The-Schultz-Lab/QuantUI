"""Structure editing and frozen-atom optimization (M-INTERACT INT.8, DEC-023 #8)."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from quantui import structure_edit as se
from quantui.molecule import Molecule

# Methanol: C1 O2 H3(O) H4 H5 H6 (on C).
METHANOL = Molecule(
    ["C", "O", "H", "H", "H", "H"],
    [
        [-0.0469, 0.6640, 0.0],
        [-0.0469, -0.7590, 0.0],
        [0.8960, -1.0390, 0.0],
        [-1.0870, 1.0010, 0.0],
        [0.4430, 1.0870, 0.8920],
        [0.4430, 1.0870, -0.8920],
    ],
)
WATER = Molecule(
    ["O", "H", "H"], [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]
)
H2O2 = Molecule(
    ["O", "O", "H", "H"],
    [[0, 0.7, 0], [0, -0.7, 0], [0.9, 0.9, 0.3], [-0.9, -0.9, 0.3]],
)
CYCLOPROPANE_C = Molecule(
    ["C", "C", "C"], [[0, 0.866, 0], [0.75, -0.433, 0], [-0.75, -0.433, 0]]
)


def _d(m: Molecule, i: int, j: int) -> float:
    return float(np.linalg.norm(np.subtract(m.coordinates[i], m.coordinates[j])))


class TestGeometryEdits:
    def test_bond_length_moves_the_whole_far_side(self):
        new, note = se.set_bond_length(METHANOL, 0, 1, 1.50)
        assert _d(new, 0, 1) == pytest.approx(1.50, abs=1e-9)
        # The hydroxyl H rides along with O; the methyl side stays put.
        assert _d(new, 1, 2) == pytest.approx(_d(METHANOL, 1, 2), abs=1e-9)
        assert new.coordinates[3] == pytest.approx(METHANOL.coordinates[3])
        assert "1.500" in note

    def test_angle_keeps_bond_lengths(self):
        from quantui.measurement import angle

        new, _ = se.set_angle(WATER, 1, 0, 2, 120.0)
        assert angle(new, 1, 0, 2) == pytest.approx(120.0, abs=1e-6)
        assert _d(new, 0, 2) == pytest.approx(_d(WATER, 0, 2), abs=1e-9)

    @pytest.mark.parametrize("target", [0.0, 60.0, 120.0, 180.0, 300.0, -60.0])
    def test_dihedral_lands_on_target(self, target):
        from quantui.measurement import dihedral

        new, _ = se.set_dihedral(H2O2, 2, 0, 1, 3, target)
        got = dihedral(new, 2, 0, 1, 3)
        assert (got - target % 360.0 + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)
        # Rigid rotation: the rotated O–H keeps its length.
        assert _d(new, 1, 3) == pytest.approx(_d(H2O2, 1, 3), abs=1e-9)

    def test_ring_bond_moves_only_the_end_atom_and_says_so(self):
        side, in_ring = se.moving_side(CYCLOPROPANE_C, 0, 1)
        assert in_ring and side == [1]
        new, note = se.set_bond_length(CYCLOPROPANE_C, 0, 1, 1.6)
        assert "ring" in note
        assert new.coordinates[2] == pytest.approx(CYCLOPROPANE_C.coordinates[2])

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            se.set_bond_length(WATER, 0, 0, 1.0)
        with pytest.raises(ValueError):
            se.set_bond_length(WATER, 0, 9, 1.0)
        with pytest.raises(ValueError):
            se.set_angle(WATER, 1, 0, 2, 190.0)


class TestAtomEdits:
    def test_delete_and_renumber(self):
        new, note = se.delete_atoms(WATER, [2])
        assert new.atoms == ["O", "H"] and "H3" in note
        with pytest.raises(ValueError):
            se.delete_atoms(WATER, [0, 1, 2])

    def test_change_element_normalizes_symbol(self):
        new, _ = se.change_element(WATER, 0, "s")
        assert new.atoms[0] == "S"

    def test_add_hydrogen_points_away_from_neighbours(self):
        # CH3 radical (planar) → the new H goes perpendicular to the plane.
        ch3 = Molecule(
            ["C", "H", "H", "H"],
            [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.935, 0], [-0.54, -0.935, 0]],
            multiplicity=2,
        )
        new, _ = se.add_hydrogen(ch3, 0)
        assert new.atoms[-1] == "H"
        assert _d(new, 0, 4) == pytest.approx(1.09)
        h = np.asarray(new.coordinates[4])
        assert abs(h[2]) == pytest.approx(1.09, abs=1e-6)

    def test_add_hydrogen_to_water_oxygen(self):
        new, _ = se.add_hydrogen(WATER, 0)
        assert _d(new, 0, 3) == pytest.approx(0.96)
        # Away from both H: larger distance to them than the O–H bond.
        assert _d(new, 3, 1) > 1.2 and _d(new, 3, 2) > 1.2

    def test_parse_atom_list(self):
        assert se.parse_atom_list("1, 3 5-7") == [0, 2, 4, 5, 6]
        with pytest.raises(ValueError):
            se.parse_atom_list("0")
        with pytest.raises(ValueError):
            se.parse_atom_list("4", n_atoms=3)


class TestFrozenOptimization:
    def test_frozen_atom_does_not_move(self):
        pytest.importorskip("pyscf")
        pytest.importorskip("ase")
        from quantui.optimizer import optimize_geometry

        start = Molecule(
            ["O", "H", "H"], [[0, 0, 0.0], [0, 0.85, -0.55], [0, -0.70, -0.50]]
        )
        res = optimize_geometry(
            start, "RHF", "STO-3G", progress_stream=io.StringIO(), frozen_atoms=[0, 1]
        )
        assert res.frozen_atoms == [0, 1]
        assert res.molecule.coordinates[0] == pytest.approx(
            start.coordinates[0], abs=1e-9
        )
        assert res.molecule.coordinates[1] == pytest.approx(
            start.coordinates[1], abs=1e-9
        )
        assert res.molecule.coordinates[2] != pytest.approx(
            start.coordinates[2], abs=1e-3
        )

    def test_bad_frozen_lists(self):
        pytest.importorskip("pyscf")  # engine check runs before the list check
        pytest.importorskip("ase")
        from quantui.optimizer import optimize_geometry

        with pytest.raises(ValueError, match="does not exist"):
            optimize_geometry(WATER, "RHF", "STO-3G", frozen_atoms=[7])
        with pytest.raises(ValueError, match="nothing to optimize"):
            optimize_geometry(WATER, "RHF", "STO-3G", frozen_atoms=[0, 1, 2])


class TestEditPanel:
    def _app(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._set_molecule(WATER, "test", sync_charge_mult=False)
        return app

    def test_atoms_field_prefills_the_current_value(self):
        app = self._app()
        app._edit_atoms_txt.value = "1 2"
        assert app._edit_value.value == pytest.approx(round(_d(WATER, 0, 1), 3))
        assert "distance" in app._edit_current_html.value

    def test_set_bond_then_undo(self):
        from quantui.app_structure_edit import on_set_bond, on_undo

        app = self._app()
        app._edit_atoms_txt.value = "1 2"
        app._edit_value.value = 1.10
        on_set_bond(app)
        assert _d(app._molecule, 0, 1) == pytest.approx(1.10)
        assert "1.100" in app._edit_msg.value
        on_undo(app)
        assert _d(app._molecule, 0, 1) == pytest.approx(_d(WATER, 0, 1))

    def test_wrong_pick_count_is_explained(self):
        from quantui.app_structure_edit import on_set_angle

        app = self._app()
        app._edit_atoms_txt.value = "1 2"
        on_set_angle(app)
        assert "exactly 3 atoms" in app._edit_msg.value

    def test_viewer_clicks_accumulate_picks(self):
        from quantui.app_structure_edit import current_picks

        app = self._app()
        for idx in ("1", "0", "2"):
            app._edit_pick_inbox.value = idx
        assert current_picks(app) == [1, 0, 2]
        assert app._edit_pick_inbox.value == ""

    def test_freeze_fills_the_geometry_opt_field_and_request(self):
        from quantui.app_runflow import frozen_atom_indices
        from quantui.app_structure_edit import on_freeze

        app = self._app()
        app._edit_atoms_txt.value = "1 3"
        on_freeze(app)
        assert app.frozen_atoms_txt.value == "1, 3"
        assert frozen_atom_indices(app) == [0, 2]
        app.calc_type_dd.value = "Geometry Opt"
        from quantui.backends.dispatch import build_calculation_request

        req = build_calculation_request(app)
        assert req.options["frozen_atoms"] == [0, 2]

    def test_frozen_field_validation(self):
        from quantui.app_runflow import frozen_atom_indices

        app = self._app()
        app.frozen_atoms_txt.value = "9"
        with pytest.raises(ValueError, match="Freeze atoms"):
            frozen_atom_indices(app)

    def test_opening_the_panel_shows_atom_numbers(self):
        app = self._app()
        app._viz_backend = "py3dmol"
        captured = {}
        orig = app._set_html_output

        def _grab(out, html):
            captured["html"] = html
            return orig(out, html)

        app._set_html_output = _grab
        app._edit_accordion.selected_index = 0
        html = captured.get("html", "")
        if "3dmolviewer_" in html:  # py3Dmol available
            assert "quantui-edit-pick-inbox" in html

    def test_math_sanity(self):
        assert math.isclose(se._angle_diff(350.0, 10.0), -20.0)
