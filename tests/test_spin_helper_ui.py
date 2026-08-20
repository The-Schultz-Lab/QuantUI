"""Pick-and-apply spin-state helper UI (M-METAL MET.5).

Suggests a multiplicity for a metal centre; the student clicks Apply. Suggests
only — never sets charge, never auto-applies, and surfaces every caveat/refusal.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


def _suggest(app, metal, ox, geom):
    app.spin_metal_dd.value = metal
    app.spin_ox_si.value = ox
    app.spin_geom_dd.value = geom
    app._on_spin_suggest()


class TestSuggest:
    def test_ambiguous_shows_two_apply_buttons(self, app):
        _suggest(app, "Co", 3, "octahedral")  # d6: HS 5 / LS 1
        assert [b.layout.display for b in app.spin_apply_btns] == ["", ""]
        assert app._spin_suggested_mults == [5, 1]
        labels = [b.description for b in app.spin_apply_btns]
        assert "5" in labels[0] and "high-spin" in labels[0]
        assert "1" in labels[1] and "low-spin" in labels[1]

    def test_unambiguous_shows_one_apply_button(self, app):
        _suggest(app, "Zn", 2, "octahedral")  # d10 → singlet only
        assert app.spin_apply_btns[0].layout.display == ""
        assert app.spin_apply_btns[1].layout.display == "none"
        assert app._spin_suggested_mults == [1]

    def test_explanation_rendered(self, app):
        _suggest(app, "Fe", 3, "octahedral")
        assert "d5" in app.spin_helper_output.value


class TestApply:
    def test_apply_sets_multiplicity_not_charge(self, app):
        app.charge_si.value = 3
        _suggest(app, "Co", 3, "octahedral")
        app._on_spin_apply(1)  # low-spin, multiplicity 1
        assert app.mult_si.value == 1
        assert app.charge_si.value == 3  # charge never touched
        assert "charge" in app.spin_helper_output.value.lower()

    def test_apply_high_spin(self, app):
        _suggest(app, "Co", 3, "octahedral")
        app._on_spin_apply(0)  # high-spin, multiplicity 5
        assert app.mult_si.value == 5

    def test_apply_out_of_range_index_is_safe(self, app):
        _suggest(app, "Zn", 2, "octahedral")  # only one state
        before = app.mult_si.value
        app._on_spin_apply(1)  # no second state — must be a no-op
        assert app.mult_si.value == before


class TestFlagsAndRefusals:
    def test_non_d8_square_planar_is_flagged_no_buttons(self, app):
        _suggest(app, "Fe", 3, "square_planar")  # d5 — refused
        assert "d8" in app.spin_helper_output.value
        assert [b.layout.display for b in app.spin_apply_btns] == ["none", "none"]
        assert app._spin_suggested_mults == []

    def test_caveat_is_surfaced(self, app):
        _suggest(app, "Fe", 2, "tetrahedral")  # high-spin assumption caveat
        assert "⚠" in app.spin_helper_output.value

    def test_out_of_range_dcount_is_flagged(self, app):
        _suggest(app, "Sc", 5, "octahedral")  # d-2, impossible
        assert "⚠" in app.spin_helper_output.value
        assert all(b.layout.display == "none" for b in app.spin_apply_btns)
