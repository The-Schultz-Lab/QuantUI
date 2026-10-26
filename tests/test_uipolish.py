"""Clarity / visual-friendliness pass (roadmap M-UIPOLISH).

Widget- and helper-level guards; the rendering itself is checked in Voilà.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantui.app import QuantUIApp


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    return QuantUIApp()


WATER_YZ = [[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]


def _rotation_matrix(angle_deg, axis):
    a = np.radians(angle_deg)
    n = np.asarray(axis, dtype=float)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


class TestP1Orientation:
    def test_planar_molecule_faces_the_camera(self):
        from quantui.viz_assets import principal_view_rotation

        angle, axis = principal_view_rotation(WATER_YZ)
        rotated = (np.array(WATER_YZ) - np.mean(WATER_YZ, axis=0)) @ _rotation_matrix(
            angle, axis
        ).T
        # 3Dmol looks down z: the molecular plane must end up in x/y, with the
        # H···H direction (longest extent) horizontal.
        assert np.allclose(rotated[:, 2], 0, atol=1e-9)
        assert abs(rotated[1, 0] - rotated[2, 0]) == pytest.approx(1.514, abs=1e-6)

    def test_already_facing_needs_no_rotation(self):
        from quantui.viz_assets import principal_view_rotation

        flat = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
        assert principal_view_rotation(flat) is None
        assert principal_view_rotation([[0, 0, 0]]) is None

    def test_viewers_emit_the_rotation(self):
        pytest.importorskip("py3Dmol")
        from quantui.molecule import Molecule
        from quantui.visualization_py3dmol import visualize_molecule_py3dmol
        from quantui.viz_assets import orient_js

        view = visualize_molecule_py3dmol(Molecule(["O", "H", "H"], WATER_YZ))
        assert ".rotate(" in view._make_html()
        assert orient_js("vw", WATER_YZ).startswith("vw.rotate(")

    def test_cube_and_xyz_coordinates_parse(self):
        from quantui.viz_assets import coords_from_model_text

        xyz = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
        assert coords_from_model_text(xyz, "xyz")[1] == [0.0, 0.757, -0.469]
        cube = (
            "c\nc\n    2   0.0 0.0 0.0\n 2 1 0 0\n 2 0 1 0\n 2 0 0 1\n"
            " 8 8.0 0.0 0.0 0.2\n 1 1.0 0.0 1.4 -0.9\n"
        )
        assert coords_from_model_text(cube, "cube") == [[0, 0, 0.2], [0, 1.4, -0.9]]

    def test_plotly_camera_looks_along_the_flattest_axis(self):
        from quantui.viz_assets import plotly_camera

        cam = plotly_camera(WATER_YZ)
        assert abs(cam["eye"]["x"]) == pytest.approx(1.8)


class TestB1GpuBadge:
    def test_settings_tab_resends_the_badge(self, app, monkeypatch):
        sent = []
        monkeypatch.setattr(
            app._status_html, "send_state", lambda key=None: sent.append(key)
        )
        app.root_tab.selected_index = app._tab_index("settings")
        assert sent == ["value"]


class TestB2ReportIssue:
    def test_header_button_toggles_the_panel(self, app):
        app._on_issue_btn()
        assert app._issue_overlay.layout.display == ""
        app._on_issue_btn()
        assert app._issue_overlay.layout.display == "none"

    def test_esc_also_closes_report_issue(self, app):
        from quantui.app import _HELP_ESC_JS

        assert "quantui-issue-cancel" in app._issue_cancel_btn._dom_classes
        assert '".quantui-issue-cancel"' in _HELP_ESC_JS


class TestV1HeaderButtons:
    def test_report_issue_and_exit_are_not_alarm_coloured(self, app):
        assert app._issue_btn.button_style == ""
        assert app._exit_btn.button_style == ""
