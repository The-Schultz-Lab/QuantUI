"""Tests for click-to-measure wiring (M-MEASURE MEAS.2/3/5/6/7).

Cloud-doable per the roadmap: HTML-content assertions for the injected click
JS (mirrors test_orbital_export_and_resolution.py's TestCaptureButtonIsOptIn
for the same inbox-textarea mechanism), the inbox -> pick-state observer, and
a router-unchanged check for STRUCTURE_VIEW_RESULTS. The actual browser
click-through is LOCAL/Voila-only (MEAS.7) and not attempted here.
"""

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest

from quantui.app_measurement import (
    MAX_PICKS,
    MEASURE_INBOX_CLASS,
    finalize_analysis_html,
    inject_click_js,
    on_measure_clear,
    on_measure_inbox_changed,
    push_highlight,
    reset_picks,
    update_panel_for_backend,
)
from quantui.molecule import Molecule
from quantui.visualization_py3dmol import PY3DMOL_AVAILABLE
from quantui.viz_backend_router import VizBackend

pytestmark = pytest.mark.skipif(not PY3DMOL_AVAILABLE, reason="py3Dmol not installed")


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _py3dmol_html() -> str:
    from quantui.visualization_py3dmol import render_molecule_html

    return render_molecule_html(_water(), backend="py3dmol")


class TestInjectClickJS:
    def test_setclickable_is_wired(self):
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert "setClickable" in html

    def test_targets_the_class_the_widget_actually_carries(self):
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert MEASURE_INBOX_CLASS in html

    def test_the_sync_event_is_dispatched(self):
        # Same mechanism as ORBX.1: setting .value alone is invisible to the
        # widget model — 'input' is what the kernel actually observes.
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert 'dispatchEvent(new Event("input"' in html
        assert "bubbles:true" in html

    def test_binds_to_this_viewer(self):
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        uid = re.search(r"3dmolviewer_(\w+)", html).group(1)
        assert f'var UID="{uid}"' in html

    def test_highlight_bridge_function_is_defined(self):
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert "__quantuiMeasureHighlight" in html
        assert "addSphere" in html
        assert "addLine" in html
        assert "drawAngleArc" in html
        assert "drawDihedralArc" in html

    def test_measure_heading_exists(self):
        app = Mock()
        # build via QuantUIApp is heavy; heading is wired in app_builders.
        from quantui.app import QuantUIApp

        real = QuantUIApp()
        assert hasattr(real, "_measure_heading")
        assert "Measurement" in real._measure_heading.value

    def test_highlight_radius_scales_with_atom_vdw(self):
        # Fixed radius:0.4 sat inside C/O/Cl ball+stick spheres (VDW×0.3).
        # The overlay must look up 3Dmol's VDW table and clear the atom.
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert "highlightRadius" in html
        assert "vdwRadii" in html
        assert "radius:0.4" not in html
        assert "radius:highlightRadius(a)" in html

    def test_highlight_color_is_saturated(self):
        html = inject_click_js(_py3dmol_html(), inbox_class=MEASURE_INBOX_CLASS)
        assert "#FFEA00" in html
        assert "opacity:0.80" in html
        # Named CSS "yellow" was the old washed-out value.
        assert 'color:"yellow"' not in html

    def test_missing_viewer_id_returns_html_unchanged(self):
        html = "<p>not a py3dmol viewer</p>"
        out = inject_click_js(html, inbox_class=MEASURE_INBOX_CLASS)
        assert out == html


class TestPickStateMachine:
    @staticmethod
    def _app() -> Mock:
        app = Mock()
        app._measure_inbox = Mock(value="pending")
        app._measure_readout = Mock(value="")
        app._measure_js_bridge = None  # push_highlight no-ops cleanly
        app._analysis_displayed_molecule = _water()
        app._measure_picks = []
        return app

    def test_first_click_shows_only_the_label(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "0"})
        assert app._measure_picks == [0]
        assert "O1" in app._measure_readout.value

    def test_two_clicks_show_a_bond_length(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "0"})
        on_measure_inbox_changed(app, {"new": "1"})
        assert app._measure_picks == [0, 1]
        assert "Å" in app._measure_readout.value

    def test_three_clicks_add_an_angle(self):
        app = self._app()
        for idx in (1, 0, 2):
            on_measure_inbox_changed(app, {"new": str(idx)})
        assert app._measure_picks == [1, 0, 2]
        assert "°" in app._measure_readout.value

    def test_a_fifth_click_starts_a_new_chain(self):
        # Water only has 3 atoms, so borrow a 4-atom molecule for this one.
        app = self._app()
        app._analysis_displayed_molecule = Molecule(
            atoms=["H", "H", "H", "H"],
            coordinates=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        )
        for idx in (0, 1, 2, 3):
            on_measure_inbox_changed(app, {"new": str(idx)})
        assert len(app._measure_picks) == MAX_PICKS
        # A genuine 5th distinct pick resets to a fresh one-atom chain.
        on_measure_inbox_changed(app, {"new": "0"})
        assert app._measure_picks == [0]

    def test_repeating_an_already_picked_atom_is_ignored(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "0"})
        on_measure_inbox_changed(app, {"new": "0"})
        assert app._measure_picks == [0]  # not [0, 0] — would be degenerate

    def test_the_inbox_is_cleared_after_every_click(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "0"})
        assert app._measure_inbox.value == ""

    def test_empty_payload_is_a_noop(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": ""})
        assert app._measure_picks == []

    def test_garbage_payload_does_not_raise(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "not-a-number"})
        assert app._measure_picks == []

    def test_out_of_range_index_is_ignored(self):
        app = self._app()
        on_measure_inbox_changed(app, {"new": "99"})
        assert app._measure_picks == []

    def test_no_molecule_loaded_is_a_noop(self):
        app = self._app()
        app._analysis_displayed_molecule = None
        on_measure_inbox_changed(app, {"new": "0"})
        assert app._measure_picks == []

    def test_collinear_dihedral_reports_undefined_rather_than_raising(self):
        app = self._app()
        app._analysis_displayed_molecule = Molecule(
            atoms=["H", "H", "H", "H"],
            coordinates=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
        )
        for idx in (0, 1, 2, 3):
            on_measure_inbox_changed(app, {"new": str(idx)})
        assert "undefined" in app._measure_readout.value.lower()


class TestClearButton:
    def test_clear_empties_picks_and_resets_readout(self):
        app = Mock()
        app._measure_picks = [0, 1, 2]
        app._measure_readout = Mock(value="stale")
        app._measure_js_bridge = None
        on_measure_clear(app)
        assert app._measure_picks == []
        assert "Click an atom" in app._measure_readout.value


class TestResetPicks:
    def test_reset_clears_state_and_readout(self):
        app = Mock()
        app._measure_picks = [0, 1]
        app._measure_readout = Mock(value="stale")
        reset_picks(app)
        assert app._measure_picks == []
        assert "Click an atom" in app._measure_readout.value


class TestPanelBackendSwitch:
    @staticmethod
    def _app() -> Mock:
        app = Mock()
        app._measure_controls = Mock()
        app._measure_controls.layout = Mock()
        app._measure_fallback_msg = Mock()
        app._measure_fallback_msg.layout = Mock()
        return app

    def test_py3dmol_shows_controls(self):
        app = self._app()
        update_panel_for_backend(app, VizBackend.PY3DMOL)
        assert app._measure_controls.layout.display == ""
        assert app._measure_fallback_msg.layout.display == "none"

    def test_plotlymol_shows_fallback_message(self):
        app = self._app()
        update_panel_for_backend(app, VizBackend.PLOTLYMOL)
        assert app._measure_controls.layout.display == "none"
        assert app._measure_fallback_msg.layout.display == ""

    def test_missing_panel_widgets_is_a_noop(self):
        # Guards a bare Mock() app (as other test modules construct) from
        # crashing when the panel was never built.
        app = Mock()
        app._measure_controls = None
        app._measure_fallback_msg = None
        update_panel_for_backend(app, VizBackend.PY3DMOL)  # must not raise


class TestFinalizeAnalysisHtml:
    @staticmethod
    def _app() -> Mock:
        app = Mock()
        app._measure_picks = [0, 1]
        app._measure_readout = Mock(value="stale")
        app._measure_controls = Mock()
        app._measure_controls.layout = Mock()
        app._measure_fallback_msg = Mock()
        app._measure_fallback_msg.layout = Mock()
        return app

    def test_py3dmol_gets_click_js_and_resets_picks(self):
        app = self._app()
        html = finalize_analysis_html(app, _py3dmol_html(), VizBackend.PY3DMOL)
        assert "setClickable" in html
        assert app._measure_picks == []

    def test_plotlymol_gets_no_click_js(self):
        app = self._app()
        html = finalize_analysis_html(app, _py3dmol_html(), VizBackend.PLOTLYMOL)
        assert "setClickable" not in html
        assert app._measure_picks == []


class TestPushHighlightIsSafe:
    def test_no_bridge_is_a_noop(self):
        app = Mock()
        app._measure_js_bridge = None
        push_highlight(app, [0, 1])  # must not raise


class TestRouterUnaffected:
    """MEAS.7: confirm ANALYSIS_STRUCTURE_VIEW's sibling task is untouched —
    click-to-measure must be a per-VizTask opt-in, never a global one."""

    def test_results_tab_render_carries_no_click_js(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        app._show_result_3d(mol, extra_output=None)  # results_tab only
        out = app.result_viz_output.outputs
        combined = "".join(
            o.get("data", {}).get("text/html", "") for o in out if "data" in o
        )
        assert "quantui-measure-inbox" not in combined
