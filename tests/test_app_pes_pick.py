"""Click-to-pick atoms for PES Scan setup (quantui.app_pes_pick).

Audit INFRA-22: this module had no tests. Driven with a duck-typed app
(plain value holders for the widgets), so no front end, PySCF or py3Dmol is
needed; the JS transport itself is covered in test_ui_display_audit.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui import app_pes_pick as pp
from quantui.molecule import Molecule


def _w(value):
    return SimpleNamespace(value=value)


@pytest.fixture
def app():
    return SimpleNamespace(
        _molecule=Molecule(
            ["O", "H", "H", "H"],
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0], [0, 0, 1.0]],
            charge=1,
        ),
        _scan_type_dd=_w("Bond"),
        _scan_atom1=_w(1),
        _scan_atom2=_w(2),
        _scan_atom3=_w(3),
        _scan_atom4=_w(4),
        _scan_pick_inbox=_w(""),
        _scan_pick_readout=_w(""),
        _scan_pick_buffer=[],
        calc_type_dd=_w("PES Scan"),
    )


@pytest.fixture
def summary():
    """Record the scan-coordinate summary refresh a completed pick triggers."""
    with patch("quantui.app_runflow._update_scan_coord_summary") as fn:
        yield fn


def _click(app, idx):
    app._scan_pick_inbox.value = str(idx)
    pp.on_pes_pick_inbox_changed(app, {"new": str(idx)})


class TestNeededPicks:
    @pytest.mark.parametrize(
        "scan_type, n",
        [("Bond", 2), ("angle", 3), ("DIHEDRAL", 4), ("something else", 2)],
    )
    def test_atoms_per_coordinate(self, scan_type, n):
        assert pp._needed_picks(scan_type) == n


class TestScanPickIndices:
    def test_dropdowns_are_one_based_indices_zero_based(self, app):
        app._scan_type_dd.value = "Dihedral"
        assert pp._scan_pick_indices(app) == [0, 1, 2, 3]
        app._scan_type_dd.value = "Angle"
        assert pp._scan_pick_indices(app) == [0, 1, 2]
        app._scan_type_dd.value = "Bond"
        app._scan_atom1.value, app._scan_atom2.value = 3, 1
        assert pp._scan_pick_indices(app) == [2, 0]


class TestSequentialClicks:
    def test_two_clicks_fill_a_bond(self, app, summary):
        _click(app, 0)
        assert app._scan_atom1.value == 1
        assert app._scan_pick_buffer == [0]
        assert "1 more" in app._scan_pick_readout.value
        summary.assert_not_called()

        _click(app, 2)
        assert (app._scan_atom1.value, app._scan_atom2.value) == (1, 3)
        assert app._scan_pick_buffer == []  # complete: next click starts over
        assert "Picked O1 → H3" in app._scan_pick_readout.value
        summary.assert_called_once_with(app)
        assert app._scan_pick_inbox.value == ""  # inbox drained each click

    def test_dihedral_needs_four(self, app, summary):
        app._scan_type_dd.value = "Dihedral"
        for i in (3, 0, 1):
            _click(app, i)
        summary.assert_not_called()
        _click(app, 2)
        summary.assert_called_once()
        assert [
            w.value
            for w in (
                app._scan_atom1,
                app._scan_atom2,
                app._scan_atom3,
                app._scan_atom4,
            )
        ] == [4, 1, 2, 3]

    def test_clicking_the_same_atom_twice_is_ignored(self, app, summary):
        _click(app, 1)
        _click(app, 1)
        assert app._scan_pick_buffer == [1]
        assert app._scan_pick_inbox.value == ""
        summary.assert_not_called()

    def test_a_shorter_coordinate_restarts_a_longer_pick(self, app, summary):
        # Three atoms picked for a dihedral, then the user switches to Bond:
        # the stale buffer must not overflow into atom3/atom4.
        app._scan_pick_buffer = [0, 1, 2]
        _click(app, 3)
        assert app._scan_pick_buffer == [3]
        assert app._scan_atom1.value == 4
        summary.assert_not_called()

    @pytest.mark.parametrize("raw", ["7", "-1", "x", "1.5"])
    def test_out_of_range_or_garbage_clears_the_inbox(self, app, raw, summary):
        app._scan_pick_inbox.value = raw
        pp.on_pes_pick_inbox_changed(app, {"new": raw})
        assert app._scan_pick_inbox.value == ""
        assert app._scan_pick_buffer == []
        assert app._scan_atom1.value == 1  # untouched

    def test_no_molecule_ignores_clicks(self, app, summary):
        app._molecule = None
        _click(app, 0)
        assert app._scan_pick_buffer == []
        assert app._scan_pick_inbox.value == ""

    def test_empty_change_is_a_no_op(self, app, summary):
        pp.on_pes_pick_inbox_changed(app, {"new": ""})
        pp.on_pes_pick_inbox_changed(app, {})
        assert app._scan_pick_buffer == []

    def test_each_click_pushes_the_highlight(self, app, summary):
        with patch.object(pp, "push_scan_highlight") as push:
            _click(app, 0)
            _click(app, 2)
        assert [c.args[1] for c in push.call_args_list] == [[0], [0, 2]]


class TestHighlightAndHtml:
    def test_push_is_a_no_op_without_a_bridge(self, app):
        with patch.object(pp, "push_highlight") as ph:
            pp.push_scan_highlight(app, [0, 1])
        ph.assert_not_called()

    def test_push_borrows_the_bridge_and_restores_it_on_error(self, app):
        app._scan_pick_bridge = "scan-bridge"
        app._measure_js_bridge = "measure-bridge"
        seen = []

        def _boom(a, indices, *, slot):
            seen.append((a._measure_js_bridge, list(indices), slot))
            raise RuntimeError("display failed")

        with patch.object(pp, "push_highlight", side_effect=_boom):
            with pytest.raises(RuntimeError):
                pp.push_scan_highlight(app, [0, 1])
        assert seen == [("scan-bridge", [0, 1], pp.PES_PICK_INBOX_CLASS)]
        assert app._measure_js_bridge == "measure-bridge"

    def test_finalize_wires_clicks_only_for_a_py3dmol_pes_scan(self, app):
        inject_p = patch.object(pp, "inject_click_js", return_value="wired")
        push_p = patch.object(pp, "push_scan_highlight")
        with inject_p as inject, push_p as push:
            assert pp.finalize_pes_calc_html(app, "html", "plotlymol") == "html"
            app.calc_type_dd.value = "Single Point"
            assert pp.finalize_pes_calc_html(app, "html", "py3dmol") == "html"
            inject.assert_not_called()

            app.calc_type_dd.value = "PES Scan"
            assert pp.finalize_pes_calc_html(app, "html", "py3dmol") == "wired"
        assert inject.call_args.kwargs["inbox_class"] == pp.PES_PICK_INBOX_CLASS
        push.assert_called_once_with(app, [0, 1])

    def test_finalize_without_a_calc_type_dropdown(self, app):
        app.calc_type_dd = None
        assert pp.finalize_pes_calc_html(app, "html", "py3dmol") == "html"

    def test_clear_resets_buffer_and_readout(self, app):
        app._scan_pick_buffer = [2]
        with patch.object(pp, "push_scan_highlight") as push:
            pp.clear_pes_pick(app)
        assert app._scan_pick_buffer == []
        assert "Click atoms" in app._scan_pick_readout.value
        push.assert_called_once_with(app, [0, 1])
