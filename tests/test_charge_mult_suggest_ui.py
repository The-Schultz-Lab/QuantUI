"""Tests for the geometry-based charge/multiplicity suggester."""

from quantui.app import QuantUIApp


def _make_app():
    return QuantUIApp()


class TestChargeMultSuggestUi:
    def test_widgets_exist(self):
        app = _make_app()
        assert hasattr(app, "charge_mult_suggest_btn")
        assert hasattr(app, "charge_mult_apply_btn")

    def test_suggest_for_loaded_nitrogen(self):
        app = _make_app()
        app.xyz_area.value = "N 0 0 0"
        app._on_load_xyz(None)
        app._on_charge_mult_suggest()
        assert app._charge_mult_suggestion is not None
        assert app._charge_mult_suggestion.multiplicity == 2
        assert "Suggested: charge 0" in app.charge_mult_suggest_output.value
        assert app.charge_mult_apply_btn.layout.display != "none"

    def test_apply_sets_fields_and_molecule(self):
        app = _make_app()
        app.xyz_area.value = "N 0 0 0"
        app._on_load_xyz(None)
        app.mult_si.value = 1
        app._on_charge_mult_suggest()
        app._on_charge_mult_apply()
        assert app.charge_si.value == 0
        assert app.mult_si.value == 2
        assert app._molecule.multiplicity == 2
        assert app.charge_mult_apply_btn.layout.display == "none"

    def test_suggest_without_geometry_shows_error(self):
        app = _make_app()
        app.xyz_area.value = ""
        app._molecule = None
        app._on_charge_mult_suggest()
        assert app._charge_mult_suggestion is None
        assert "⚠" in app.charge_mult_suggest_output.value

    def test_suggest_from_textarea_without_load(self):
        app = _make_app()
        app.xyz_area.value = "H 0 0 0\nH 0 0 0.74"
        app._on_charge_mult_suggest()
        assert app._charge_mult_suggestion is not None
        assert app._charge_mult_suggestion.multiplicity == 1
