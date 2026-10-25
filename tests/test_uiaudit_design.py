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
