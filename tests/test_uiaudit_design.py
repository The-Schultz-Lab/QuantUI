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


class TestD4DipoleConvention:
    def test_chemistry_convention_reverses_only_the_drawing(self):
        from quantui.populations_overlay import build_overlay_payload

        mol = _water()
        kw = dict(
            charges=[-0.8, 0.4, 0.4],
            color_enabled=False,
            dipole_vector=[0.0, 2.6, 0.0],
            dipole_enabled=True,
            symbols=mol.atoms,
            coordinates=mol.coordinates,
        )
        phys = build_overlay_payload(**kw)["arrow"]
        chem = build_overlay_payload(**kw, arrow_convention="chemistry")["arrow"]
        # μ points toward the H atoms (+y): physics head at +y, chemistry at −y.
        assert phys["end"]["y"] > phys["start"]["y"]
        assert chem["end"] == phys["start"] and chem["start"] == phys["end"]

    def test_choice_is_captioned_and_remembered(self, app, tmp_path):
        assert app._mulliken_dipole_dir_dd.value == "physics"
        assert "positive</b> end" in app._mulliken_dipole_caption.value
        app._mulliken_dipole_dir_dd.value = "chemistry"
        assert "negative</b> end" in app._mulliken_dipole_caption.value
        assert QuantUIApp()._mulliken_dipole_dir_dd.value == "chemistry"


class TestD6FrequencyCard:
    @staticmethod
    def _result():
        from quantui.freq_calc import FreqResult, ThermoData

        return FreqResult(
            energy_hartree=-76.406815,
            homo_lumo_gap_ev=9.5,
            converged=True,
            n_iterations=10,
            method="B3LYP",
            basis="6-31G*",
            formula="H2O",
            frequencies_cm1=[1713.0, 3727.0, 3849.0],
            scf_variant="RKS",
            thermo=ThermoData(
                zpve_hartree=0.021723,
                H_hartree=-76.381313,
                S_jmol=188.56,
                G_hartree=-76.402725,
            ),
        )

    def test_title_has_no_nested_parentheses(self):
        from quantui.app_formatters import format_freq_result

        html = format_freq_result(self._result())
        assert "H2O &middot; B3LYP/6-31G* (RKS)</b>" in html
        assert "(B3LYP/6-31G* (RKS))" not in html

    def test_thermo_shows_corrections_not_absolute_kj(self):
        from quantui.app_formatters import format_freq_result

        html = format_freq_result(self._result())
        # G − E = (−76.402725 + 76.406815) Ha × 2625.4996 ≈ +10.74 kJ/mol
        assert "+10.74 kJ/mol vs E" in html
        assert "-200595" not in html
        assert "(298 K)</td>" not in html  # T is in the section header
