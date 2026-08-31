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
    def test_refresh_button_click_populates_options(self, app, monkeypatch):
        # Regression guard, updated for UXP2.5's consolidation: an early
        # revision bound the Geo Opt refresh button to
        # _refresh_freq_seed_options, so its click repopulated the Frequency
        # dropdown instead of its own. Now that all three calc types share one
        # dropdown/button (there IS only "its own" in the singular), the
        # meaningful guard is that clicking the button actually triggers the
        # real refresh implementation, not a specific per-calc-type alias name.
        called = []
        monkeypatch.setattr(
            app, "_refresh_seed_options", lambda: called.append("refreshed")
        )
        for handler in app._geo_seed_refresh_btn._click_handlers.callbacks:
            handler(app._geo_seed_refresh_btn)
        assert called == ["refreshed"]

    def test_geo_freq_tddft_share_one_widget_group(self, app):
        # UXP2.5 (M-UX2, 2026-07-31): Geometry Opt / Frequency / UV-Vis
        # (TD-DFT) used to each build their own near-identical seed dropdown +
        # refresh button + note. Only one of the three panels is ever visible
        # at a time, so there is no benefit to three separate widgets — and a
        # seed a user picked on one panel legitimately applies if they switch
        # to another before running, rather than needing to be re-selected.
        # These attribute names are now aliases onto one shared widget each.
        assert app._geo_seed_dd is app._freq_seed_dd is app._tddft_seed_dd
        assert (
            app._geo_seed_refresh_btn
            is app._freq_seed_refresh_btn
            is app._tddft_seed_refresh_btn
        )
        assert app._geo_seed_note is app._freq_seed_note is app._tddft_seed_note

    def test_refresh_populates_without_error(self, app):
        app._refresh_geo_seed_options()
        assert app._geo_seed_dd.options[0] == ("(use current molecule)", "")


class TestPreoptCheckboxDoesNotGetStuckDisabled:
    """Regression tests for a bug the UXP2.5 consolidation fixed as a side
    effect, not the thing it set out to fix.

    Before consolidation, on_calc_type_changed only ever set
    ``_freq_preopt_cb.disabled`` to True (via the per-calc-type seed-changed
    handlers) and never reset it back to False on a plain calc-type switch —
    only Geometry Opt/Reorganization Energy explicitly handled the checkbox at
    all, by hiding it. So: pick a seed on Frequency (disables the checkbox),
    switch to Single Point (checkbox becomes visible again, but stays
    disabled), and there was no path back to enabled short of re-selecting and
    then clearing the seed on the now-hidden Frequency panel.
    """

    def _seeded_result(self, tmp_path, monkeypatch, app):
        # Minimal on-disk geometry_opt result so the dropdown has a real,
        # selectable option (a Dropdown rejects values not in .options).
        import json

        result_dir = tmp_path / "seed_result"
        result_dir.mkdir()
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "calc_type": "geometry_opt",
                    "formula": "H2O",
                    "method": "RHF",
                    "basis": "STO-3G",
                    "timestamp": "2026-01-01T00:00:00",
                }
            )
        )
        (result_dir / "trajectory.json").write_text(
            json.dumps([{"atoms": ["O", "H", "H"], "coordinates": [[0, 0, 0]] * 3}])
        )
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app._molecule = None  # skip the formula/RMSD filter for this test
        app._refresh_seed_options()
        return str(result_dir)

    def test_switching_to_single_point_re_enables_the_checkbox(
        self, app, tmp_path, monkeypatch
    ):
        seed = self._seeded_result(tmp_path, monkeypatch, app)
        app.calc_type_dd.value = "Frequency"
        app._seed_dd.value = seed
        assert app._freq_preopt_cb.disabled is True  # sanity: engaged first

        app.calc_type_dd.value = "Single Point"

        assert app._freq_preopt_cb.disabled is False

    def test_a_seed_still_disables_it_after_switching_between_freq_and_uvvis(
        self, app, tmp_path, monkeypatch
    ):
        # The shared dropdown legitimately carries the seed across these two
        # calc types (UXP2.5) — the checkbox must stay gated, not just avoid
        # being stuck.
        seed = self._seeded_result(tmp_path, monkeypatch, app)
        app.calc_type_dd.value = "Frequency"
        app._seed_dd.value = seed

        app.calc_type_dd.value = "UV-Vis (TD-DFT)"

        assert app._freq_preopt_cb.disabled is True

    def test_switching_to_geometry_opt_hides_it_regardless_of_a_pending_seed(
        self, app, tmp_path, monkeypatch
    ):
        seed = self._seeded_result(tmp_path, monkeypatch, app)
        app.calc_type_dd.value = "Frequency"
        app._seed_dd.value = seed

        app.calc_type_dd.value = "Geometry Opt"

        assert app._freq_preopt_cb.layout.display == "none"
        assert app._freq_preopt_cb.value is False


class TestPesPreoptCheckboxPlacement:
    def test_preopt_checkbox_is_above_pes_extra_opts(self, app):
        app.calc_type_dd.value = "PES Scan"
        children = list(app.calc_setup_panel.children)
        preopt_idx = children.index(app._freq_preopt_cb)
        extra_idx = children.index(app.calc_extra_opts)
        assert preopt_idx < extra_idx


class TestPesSeedMatchesFinalGeometry:
    def test_seed_lists_geo_opt_when_current_matches_optimized_frame(
        self, app, tmp_path, monkeypatch
    ):
        import json

        from quantui.molecule import Molecule

        result_dir = tmp_path / "geo_result"
        result_dir.mkdir()
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "calc_type": "geometry_opt",
                    "formula": "H2O",
                    "method": "RHF",
                    "basis": "STO-3G",
                    "timestamp": "2026-01-01T00:00:00",
                }
            )
        )
        start = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
        final = [[0.0, 0.0, 0.1], [0.95, 0.0, 0.0], [-0.25, 0.92, 0.0]]
        (result_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "atoms": ["O", "H", "H"],
                    "steps": [
                        {"coords": start},
                        {"coords": final},
                    ],
                }
            )
        )
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app._molecule = Molecule(
            atoms=["O", "H", "H"],
            coordinates=final,
        )
        from quantui.app_runflow import _refresh_seed_options

        _refresh_seed_options(app, app._scan_seed_dd)
        values = [v for _, v in app._scan_seed_dd.options]
        assert str(result_dir) in values
