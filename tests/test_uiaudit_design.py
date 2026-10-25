"""Rendered-UI audit design items D4–D17 (roadmap M-UIAUDIT).

Widget-level guards for layout/copy fixes that were found by screenshotting
the running app; the rendering itself is checked in Voilà.
"""

from __future__ import annotations

import pytest

from quantui.app import QuantUIApp


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    return QuantUIApp()


class TestD7Truncation:
    def test_long_checkbox_labels_size_to_their_text(self, app):
        # ipywidgets' 300 px default cut "— Raman is not computed in solvent"
        # and "(QM, slower)" off.
        assert app.solvent_cb.layout.width == "auto"
        assert app._freq_preopt_cb.layout.width == "auto"

    def test_small_help_buttons_have_no_padding(self, app):
        # 28 px buttons with the default 12 px side padding rendered "?..".
        for btn in (app.method_help_btn, app.basis_help_btn, app._scan_help_btn):
            assert btn.layout.padding == "0"

    def test_orbital_count_input_has_no_truncated_description(self, app):
        assert app._orb_n_orb_input.description == ""

    def test_raman_mode_toggle_has_no_orphan_label(self, app):
        assert app._raman_mode_toggle.description == ""
        assert app._ir_mode_toggle.style.button_width == "96px"


class TestD13Copy:
    def test_hero_points_at_the_help_button(self, app):
        assert "Help</b> button" in app._welcome_html.value
        assert "Help</b> tab" not in app._welcome_html.value


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


def _output_html(out) -> str:
    return "".join(
        o.get("data", {}).get("text/html", "")
        for o in out.outputs
        if o.get("output_type") == "display_data"
    )


class TestD8DuplicateInfo:
    def test_calculate_viewer_has_no_info_box(self, app):
        pytest.importorskip("py3Dmol")
        app._set_molecule(_water())
        html = _output_html(app.viz_output)
        assert "<div" in html  # the viewer rendered
        assert "Molecule Information" not in html
        assert "available:" not in html

    def test_viewer_space_is_reserved_only_with_a_molecule(self, app):
        pytest.importorskip("py3Dmol")
        assert app.viz_output.layout.min_height == "0px"
        app._set_molecule(_water())
        assert app.viz_output.layout.min_height == "540px"

    def test_launcher_notebook_has_no_duplicate_heading(self):
        import json
        import pathlib

        nb_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "notebooks"
            / "molecule_computations.ipynb"
        )
        cells = json.loads(nb_path.read_text())["cells"]
        assert all(c["cell_type"] == "code" for c in cells)


class TestD9ResultTable:
    def test_result_table_hugs_its_content_and_aligns_left(self):
        from quantui.app import _APP_CSS
        from quantui.app_formatters import _result_card_table_open

        assert "width:auto" in _result_card_table_open()
        assert "text-align: left !important" in _APP_CSS
