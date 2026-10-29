"""Clarity / visual-friendliness pass (roadmap M-UIPOLISH).

Widget- and helper-level guards; the rendering itself is checked in Voilà.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantui.app import QuantUIApp

try:
    import pyscf  # noqa: F401

    _PYSCF = True
except ImportError:
    _PYSCF = False

# Labels / the engine chip are computed with PySCF; the Windows CI job has none.
pyscf_only = pytest.mark.skipif(not _PYSCF, reason="PySCF not installed")


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


class TestP2Library:
    @pytest.mark.parametrize(
        "query, name",
        [("water", "Water"), ("ammonia", "Ammonia"), ("methane", "Methane")]
        + [("benzene", "Benzene"), ("C6H6", "Benzene"), ("H2O", "Water")],
    )
    def test_common_names_lead(self, query, name):
        from quantui import molecule_library as ml

        assert ml.search(query, limit=3)[0]["name"] == name

    def test_qm9_after_curated_and_no_duplicate_names(self):
        from quantui import molecule_library as ml

        rows = ml.search("C6H6", limit=50)
        cats = [r["category"] for r in rows]
        assert cats.index("bulk-qm9") > cats.index(rows[0]["category"])
        assert all(c == "bulk-qm9" for c in cats[cats.index("bulk-qm9") :])
        names = [r["name"] for r in ml.search("methanol", limit=10)]
        assert names.count("Methanol") == 1


class TestP3UvVis:
    def test_default_range_shows_vacuum_uv_bands(self):
        from quantui.app_visualization import uv_default_range

        # RHF/STO-3G water: bands at 57-106 nm (and two core states at 2 nm).
        wl = [106.3, 88.8, 85.4, 69.4, 57.6, 46.2, 33.6, 33.5, 2.39, 2.38]
        lo, hi = uv_default_range(wl, 10.0)
        assert lo < 33.5 and hi > 106.3 and lo > 2.39

    def test_excitation_table(self):
        from quantui.app_visualization import uv_excitation_table_html

        html = uv_excitation_table_html(
            [200.0, 150.0], [0.1, 0.5], [6.2, 8.27], (160, 300)
        )
        assert "<th style='padding:2px 12px;text-align:right'>f</th>" in html
        assert html.index("6.200") < html.index("8.270")  # S1 first
        assert "1 state outside the plotted" in html


class TestP4Nmr:
    def test_equivalent_nuclei_share_one_labelled_stick(self):
        from quantui.nmr_plot import plot_nmr_spectrum

        fig = plot_nmr_spectrum([(1, -3.17), (2, -3.17)], ["O", "H", "H"])
        peaks = fig.data[-1]
        assert list(peaks.text) == ["H2, H3"]
        assert list(peaks.y) == [2.0]

    def test_one_table_numbered_like_the_viewer(self):
        from quantui.app_visualization import _nmr_summary_html

        html = _nmr_summary_html(
            ["O", "H", "H"], [365.6, 33.7, 33.7], [(1, -3.17), (2, -3.17)], [], "TMS"
        )
        assert "H2</td>" in html and "H3</td>" in html
        assert "H-1" not in html
        assert html.count("<table") == 1


class TestP6P7Formulas:
    def test_display_formula(self):
        from quantui.app_formatters import display_formula

        got = [display_formula(f) for f in ["H3N", "H2O", "CH4", "H4Si", "HCl", "C6H6"]]
        assert got == ["NH3", "H2O", "CH4", "SiH4", "HCl", "C6H6"]

    @pyscf_only
    def test_frequency_row_lists_modes_with_symmetry(self):
        import json
        import pathlib
        from types import SimpleNamespace

        from quantui.app_formatters import _frequency_list_html

        path = pathlib.Path(__file__).parent / "data" / "normal_modes_hf_sto3g.json"
        m = json.loads(path.read_text())["NH3"]
        r = SimpleNamespace(
            frequencies_cm1=m["freqs"],
            displacements=m["disps"],
            pyscf_mol_atom=m["atoms"],
        )
        assert _frequency_list_html(r).startswith(
            "1091 (a<sub>1</sub>), 2020 (e) &times;2"
        )


class TestP5SameCard:
    def test_live_and_history_frequency_cards_share_rows(self):
        from quantui.app_formatters import format_freq_result, format_past_result
        from quantui.freq_calc import FreqResult

        r = FreqResult(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=9.5,
            converged=True,
            n_iterations=10,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
            frequencies_cm1=[1700.0, 3700.0, 3800.0],
            zpve_hartree=0.02,
        )
        data = {
            "calc_type": "frequency",
            "formula": "H2O",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -76.0,
            "energy_ev": -76.0 * 27.211386245988,
            "homo_lumo_gap_ev": 9.5,
            "converged": True,
            "n_iterations": 10,
            "spectra": {"ir": {"frequencies_cm1": [1700.0, 3700.0, 3800.0]}},
        }
        live, past = format_freq_result(r), format_past_result(data)
        for label in ("Total energy", "SCF iterations", "Real modes", "Frequencies"):
            assert label in live and label in past


class TestPhase2CopyAndStyle:
    def test_settings_say_once_that_they_persist(self, app):
        from quantui.app_builders import build_status_panel  # noqa: F401

        html = "".join(
            getattr(w, "value", "")
            for w in app._status_tab_panel.children[1].children
            if hasattr(w, "value") and isinstance(w.value, str)
        )
        assert "persists across launches" not in html
        assert "saved automatically" in html

    def test_one_line_header(self, app):
        assert "font-size:44px" not in app._welcome_html.value
        assert str(app._welcome_logo.width) == "44"

    def test_cancel_is_red_only_while_running(self, app):
        assert app.cancel_btn.disabled and app.cancel_btn.button_style == ""
        app.cancel_btn.disabled = False
        assert app.cancel_btn.button_style == "danger"
        app.cancel_btn.disabled = True
        assert app.cancel_btn.button_style == ""

    @pyscf_only
    def test_engine_is_a_chip_without_the_platform_paragraph(self, app):
        import sys

        html = app.engine_capability_html.value
        assert "Engine: <b>" in html
        if sys.platform != "win32" and "PySCF" in html:
            assert "Native Windows requires" not in html.split("</span>")[-1]

    def test_run_first_text_grammar(self):
        from quantui.app_analysis import _run_first_text

        assert _run_first_text("NMR Shielding").startswith("Run an NMR")
        assert _run_first_text("UV-Vis (TD-DFT)").startswith("Run a UV-Vis")

    def test_imaginary_note_only_with_imaginary_modes(self):
        from types import SimpleNamespace

        from quantui.app_visualization import vib_mode_table_html

        real = SimpleNamespace(frequencies_cm1=[1600.0, 3700.0])
        imag = SimpleNamespace(frequencies_cm1=[-400.0, 3700.0])
        assert "Imaginary modes" not in vib_mode_table_html(real)
        assert "Imaginary modes" in vib_mode_table_html(imag)

    def test_minimal_basis_card_is_filled(self):
        from quantui import theme
        from quantui.descriptor_cards import basis_card_html

        assert theme.css.SURFACE_MUTED_BG in basis_card_html("STO-3G")

    def test_history_card_has_no_thumbnail(self, tmp_path):
        from quantui.app_formatters import format_past_result

        (tmp_path / "thumbnail.png").write_bytes(b"\x89PNG")
        data = {
            "calc_type": "single_point",
            "formula": "H2O",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -75.0,
            "energy_ev": -2040.0,
            "converged": True,
        }
        assert "data:image/png" not in format_past_result(data, tmp_path)


def _walk(w):
    yield w
    for c in getattr(w, "children", ()):
        yield from _walk(c)


class TestPhase3Layout:
    def test_one_line_stepper(self, app):
        html = app.step_progress.widget.value
        assert "Step 1:" not in html and "›" in html

    def test_calculate_tab_has_two_columns(self, app):
        row = app._calculate_tab_panel.children[1]
        left, right = row.children
        assert app.mol_input_container in left.children
        assert app.calc_setup_panel in right.children
        assert row.layout.flex_flow == "row wrap"

    def test_completion_banner_sits_under_run(self, app):
        kids = list(app.run_panel.children)
        run_row = next(
            i for i, c in enumerate(kids) if app.run_btn in getattr(c, "children", ())
        )
        assert kids[run_row + 1] is app._completion_banner

    def test_molecule_tools_appear_with_a_molecule(self, app):
        from quantui.molecule import Molecule

        kids = app.mol_input_container.children
        assert app._edit_accordion not in kids
        assert app.preopt_preview_row not in kids
        app._set_molecule(Molecule(["O", "H", "H"], WATER_YZ))
        kids = app.mol_input_container.children
        assert app._edit_accordion in kids and app.preopt_preview_row in kids
        assert all(app.preopt_preview_row is not w for w in _walk(app.calc_setup_panel))

    def test_available_panels_listed_first(self, app):
        from quantui.app_analysis import order_analysis_panels

        app._ana_available = {"IR Spectrum"}
        order_analysis_panels(app)
        kids = list(app.analysis_tab_panel.children)
        assert kids.index(app._ir_accordion) < kids.index(app._orb_accordion)
        assert app._ana_unavail_divider.layout.display == ""

    def test_compare_uses_checkboxes(self, app):
        from quantui.checklist_widget import CheckList

        sel = app.compare_select
        assert isinstance(sel, CheckList)
        sel.options = [("a", "/a"), ("b", "/b"), ("(note)", "")]
        sel.children[1].value = True
        assert sel.value == ("/b",)
        sel.value = ()
        assert sel.value == ()

    def test_history_recent_buttons_follow_the_picker(self, app):
        app.past_dd.options = [("(select)", ""), ("first", "/r1"), ("second", "/r2")]
        btns = [c for c in app._history_recent_box.children if hasattr(c, "on_click")]
        assert [b.description for b in btns] == ["first", "second"]

    def test_help_and_report_are_drawers(self, app):
        assert "quantui-drawer" in app.help_tab_panel._dom_classes
        assert "quantui-drawer" in app._issue_overlay._dom_classes

    def test_raman_follows_ir_axis(self):
        from quantui.raman_plot import plot_raman_spectrum

        fig = plot_raman_spectrum([1000.0, 2000.0], [1.0, 2.0], descending=True)
        lo, hi = fig.layout.xaxis.range
        assert lo > hi

    def test_live_log_hides_bookkeeping(self):
        from quantui.app import _friendly_log_text

        text = "[checkpoint] saved — status=complete\n[QuantUI_STATUS] Hessian\nE=1\n"
        assert _friendly_log_text(text) == "→ Hessian\nE=1\n"

    def test_plots_resize_when_panels_open(self):
        from quantui.app import _PLOT_RESIZE_JS

        assert "jupyter-widget-Collapse-header" in _PLOT_RESIZE_JS
        assert 'new Event("resize")' in _PLOT_RESIZE_JS

    def test_compare_table_rows_are_theme_coloured(self):
        from types import SimpleNamespace

        from quantui.comparison import comparison_table_html

        s = SimpleNamespace(
            label="A",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            homo_lumo_gap_ev=1.0,
            converged=True,
            n_iterations=5,
        )
        assert "var(--q-bg-panel" in comparison_table_html([s])

    def test_side_by_side_viewers(self, app):
        for widget in (app._mulliken_mol_output, app.vib_output, app._orb_iso_output):
            parents = [
                w
                for w in _walk(app.analysis_tab_panel)
                if widget in getattr(w, "children", ())
            ]
            row = [
                w
                for w in _walk(app.analysis_tab_panel)
                if parents[0] in getattr(w, "children", ())
            ][0]
            assert row.layout.flex_flow == "row wrap"
