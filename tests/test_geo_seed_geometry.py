"""Seed geometry for Geometry Opt runs (M-UX2, user request 2026-07-30).

Frequency and UV-Vis could already start from a previously optimised geometry;
Geometry Opt could not. The motivating workflow is "optimise at a cheap level of
theory, then refine at a higher one" — which needs the *starting* geometry to
come from history.

Platform-independent: widget wiring only, no PySCF.
"""

from __future__ import annotations

import pytest

from quantui.app import QuantUIApp


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    return QuantUIApp()


class TestSeedWidgetExists:
    def test_dropdown_defaults_to_current_molecule(self):
        # The empty value is what the run path checks; a non-empty default would
        # silently redirect every optimisation through history.
        a = QuantUIApp()
        assert a._geo_seed_dd.value == ""
        assert a._geo_seed_dd.options[0] == ("(use current molecule)", "")

    def test_shown_in_the_geometry_opt_panel(self, app):
        app.calc_type_dd.value = "Geometry Opt"
        rendered = []
        for child in app.calc_extra_opts.children:
            rendered.extend(getattr(child, "children", [child]))
        assert app._geo_seed_dd in rendered
        assert app._geo_seed_refresh_btn in rendered

    def test_not_shown_for_single_point(self, app):
        app.calc_type_dd.value = "Single Point"
        rendered = []
        for child in app.calc_extra_opts.children:
            rendered.extend(getattr(child, "children", [child]))
        assert app._geo_seed_dd not in rendered

    def test_fmax_and_max_steps_still_present(self, app):
        # The seed row is additive — it must not displace the existing controls.
        app.calc_type_dd.value = "Geometry Opt"
        rendered = []
        for child in app.calc_extra_opts.children:
            rendered.extend(getattr(child, "children", [child]))
        assert app.fmax_fi in rendered
        assert app.max_steps_si in rendered


class TestSeedNote:
    def test_note_appears_when_a_seed_is_selected(self, app):
        app._on_geo_seed_changed({"new": "/some/result/dir"})
        assert "geometry" in app._geo_seed_note.value.lower()
        assert app._geo_seed_note.value != ""

    def test_note_clears_when_deselected(self, app):
        app._on_geo_seed_changed({"new": "/some/result/dir"})
        app._on_geo_seed_changed({"new": ""})
        assert app._geo_seed_note.value == ""

    def test_does_not_touch_the_preopt_checkbox(self, app):
        # The Frequency/UV-Vis handlers disable _freq_preopt_cb because a seed is
        # already optimised and re-optimising first would be redundant. For
        # Geometry Opt that checkbox means "optimise before the calculation",
        # which is meaningless when the optimisation IS the calculation — so this
        # handler must leave it alone.
        app._freq_preopt_cb.disabled = False
        app._freq_preopt_cb.value = True
        app._on_geo_seed_changed({"new": "/some/result/dir"})
        assert app._freq_preopt_cb.disabled is False
        assert app._freq_preopt_cb.value is True


class TestSeedWiring:
    def test_refresh_button_targets_the_geo_dropdown(self, app, monkeypatch):
        # Regression guard: an early revision bound this to _freq_seed_refresh_btn,
        # so the Geo Opt refresh icon repopulated the Frequency dropdown.
        called = []
        monkeypatch.setattr(
            app, "_refresh_geo_seed_options", lambda: called.append("geo")
        )
        for handler in app._geo_seed_refresh_btn._click_handlers.callbacks:
            handler(app._geo_seed_refresh_btn)
        assert called == ["geo"]

    def test_each_calc_type_has_its_own_dropdown(self, app):
        # Sharing one dropdown across calc types would leak a Frequency seed into
        # a Geometry Opt run.
        assert app._geo_seed_dd is not app._freq_seed_dd
        assert app._geo_seed_dd is not app._tddft_seed_dd

    def test_refresh_populates_without_error(self, app):
        app._refresh_geo_seed_options()
        assert app._geo_seed_dd.options[0] == ("(use current molecule)", "")
