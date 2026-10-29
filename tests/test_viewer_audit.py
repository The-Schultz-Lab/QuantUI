"""Pre-release viewer audit (VA.1-VA.12): fixes for the History + Analysis
viewer, which runs without PySCF (desktop installer, browser build)."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from quantui import app_viewer
from quantui.app_viewer import RESULTS_DIR_ENV

WATER_ATOMS = ["O", "H", "H"]
WATER_XYZ = [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path / "results"))
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    (tmp_path / "results").mkdir()
    return tmp_path / "results"


@pytest.fixture
def no_pyscf(monkeypatch):
    """What the viewer sees: PySCF not installed."""
    import quantui.engines

    monkeypatch.setattr(quantui.engines, "is_pyscf_available", lambda: False)


def _water():
    from quantui.molecule import Molecule

    return Molecule(WATER_ATOMS, WATER_XYZ)


def _freq_result(**over):
    thermo = SimpleNamespace(
        zpve_hartree=0.0236,
        H_hartree=-74.94,
        S_jmol=188.7,
        G_hartree=-74.96,
        temperature_k=298.15,
        E_thermal_hartree=-74.941,
        Cv_jmolk=25.3,
        Cp_jmolk=33.6,
        symmetry_number=2,
        pressure_atm=1.0,
    )
    base = dict(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-74.9659,
        homo_lumo_gap_ev=26.5,
        converged=True,
        n_iterations=6,
        frequencies_cm1=[2170.0, 4141.0, 4392.0],
        ir_intensities=[10.0, 30.0, 20.0],
        raman_activities=[8.0, 40.0, 17.0],
        zpve_hartree=0.0236,
        displacements=None,
        thermo=thermo,
        imaginary_mode_retry=None,
        max_force_ev_per_ang=0.0096,
        mode_irreps=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── VA.12 + VA.9: one Frequency payload, mode labels saved with it ─────────


class TestFreqPayload:
    def test_calibration_save_keeps_thermo_and_force(self, env):
        from quantui.benchmarks import _save_calibration_step

        rd = _save_calibration_step(
            _freq_result(),
            calc_type="frequency",
            pyscf_log="",
            calibration_run_id="",
            mol=_water(),
        )
        ir = json.loads((rd / "result.json").read_text())["spectra"]["ir"]
        assert ir["thermo"]["G_hartree"] == pytest.approx(-74.96)
        assert ir["max_force_ev_per_ang"] == pytest.approx(0.0096)
        assert "imaginary_mode_retry" in ir and "mode_irreps" in ir

    def test_calibration_opt_keeps_populations(self, env):
        from quantui.benchmarks import _save_calibration_step

        mol = _water()
        opt = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-74.96],
            converged=True,
            n_steps=1,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
            energy_hartree=-74.96,
            atom_symbols=WATER_ATOMS,
            mulliken_charges=[-0.37, 0.18, 0.18],
            dipole_moment_debye=1.7,
            dipole_vector_debye=[0.0, 0.0, -1.7],
        )
        rd = _save_calibration_step(
            opt, calc_type="geometry_opt", pyscf_log="", calibration_run_id="", mol=mol
        )
        data = json.loads((rd / "result.json").read_text())
        assert data["mulliken_charges"] == pytest.approx([-0.37, 0.18, 0.18])
        assert data["geometry"]["atoms"] == WATER_ATOMS

    def test_saved_labels_round_trip(self):
        from quantui.symmetry import mode_irreps_from_saved

        irreps = mode_irreps_from_saved(
            {"labels": ["a1", "a1", "b2"], "group": "C2v", "gamma": "2A1 + B2"}
        )
        assert irreps.labels == ["a1", "a1", "b2"] and not irreps.is_subgroup
        assert mode_irreps_from_saved({"labels": "a1"}) is None
        assert mode_irreps_from_saved(None) is None

    def test_card_uses_saved_labels_without_displacements(self):
        from quantui.app_formatters import _frequency_list_html

        html = _frequency_list_html(
            _freq_result(
                mode_irreps={"labels": ["a1", "a1", "b2"], "group": "C2v"},
                pyscf_mol_atom=None,
            )
        )
        assert html.startswith("2170 (a<sub>1</sub>)") and "(b<sub>2</sub>)" in html

    def test_analysis_uses_saved_labels(self):
        from quantui.app_visualization import freq_mode_irreps

        irreps = freq_mode_irreps(
            _freq_result(mode_irreps={"labels": ["a1", "a1", "b2"], "group": "C2v"})
        )
        assert irreps is not None and irreps.labels[2] == "b2"


# ── VA.1: thermochemistry without PySCF ────────────────────────────────────


class TestThermoWithoutPySCF:
    def _app(self):
        from quantui.app import QuantUIApp

        return QuantUIApp(viewer=True)

    def test_saved_values_shown_and_inputs_locked(self, env, no_pyscf):
        from quantui.app_visualization import show_thermo_box

        app = self._app()
        saved = {"G_hartree": -74.96, "H_hartree": -74.94, "S_jmol": 188.7}
        show_thermo_box(
            app, _water(), -74.9659, [2170.0], saved_thermo=saved
        )
        html = app._thermo_html.value
        assert "Gibbs energy G" in html and "Saved with the result" in html
        assert "failed" not in html
        assert app._thermo_T.disabled and app._thermo_P.disabled

    def test_no_saved_values_is_a_note_not_an_error(self, env, no_pyscf):
        from quantui.app_visualization import show_thermo_box

        app = self._app()
        show_thermo_box(app, _water(), -74.9, [2170.0], max_force_ev_per_ang=3.2)
        html = app._thermo_html.value
        assert "Not a minimum" in html  # D1 survives the missing engine
        assert "no saved values" in html and "failed" not in html


# ── VA.2: isosurface panel without PySCF ───────────────────────────────────


def test_isosurface_panel_is_a_note_without_pyscf(env, no_pyscf):
    from quantui.app import QuantUIApp

    app = QuantUIApp(viewer=True)

    def _walk(w):
        yield w
        for c in getattr(w, "children", ()):
            yield from _walk(c)

    widgets = list(_walk(app._iso_accordion))
    assert any("doesn't include" in str(getattr(w, "value", "")) for w in widgets)
    assert app._iso_generate_btn not in widgets
    assert app._orb_gallery_btn not in widgets


# ── VA.3 / VA.8: theme CSS ─────────────────────────────────────────────────


def test_theme_css_overrides_table_stripes_and_caps_3dmol_width():
    from quantui.theme import _widget_chrome_css

    css = _widget_chrome_css()
    assert ".jp-RenderedHTMLCommon tbody tr:nth-child(odd)" in css
    assert 'div[id^="3dmolviewer_"] { max-width: 100% !important; }' in css


# ── VA.4: structure viewer for results with only top-level geometry ───────


def test_history_structure_uses_saved_geometry(env):
    from quantui.app_history import _mol_lookup_data, mol_from_result_dir
    from quantui.results_storage import save_result

    res = SimpleNamespace(
        formula="H2O", method="RHF", basis="STO-3G", energy_hartree=-74.9
    )
    rd = save_result(res, results_dir=env, calc_type="nmr", molecule=_water())
    ctx = SimpleNamespace(calc_type="nmr", spectra_data={})
    mol = mol_from_result_dir(rd, _mol_lookup_data(rd, ctx))
    assert mol is not None and list(mol.atoms) == WATER_ATOMS


# ── VA.5 / VA.7 / VA.11 / cosmetics: viewer chrome ─────────────────────────


class TestViewerChrome:
    def test_tab_switch_does_not_raise(self, env):
        from quantui.app import QuantUIApp

        app = QuantUIApp(viewer=True)
        for idx in (1, 0):
            app._on_root_tab_changed({"new": idx})
        assert not app._is_tab(0, "files") and app._is_tab(1, "analysis")

    def test_header_help_and_activity(self, env):
        from quantui.app import QuantUIApp

        app = QuantUIApp(viewer=True)
        assert ">Viewer</span>" in app._welcome_html.value
        assert app._activity_btn.layout.display == "none"
        keys = [k for _, k in app.help_topic_dd.options]
        assert keys[0] == "viewer_getting_started" and "calc_type" not in keys
        assert "Calculate tab" not in app._help_intro_html.value
        assert "Viewer" in app.help_content_html.value

    def test_full_app_unchanged(self, env):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert ">Viewer</span>" not in app._welcome_html.value
        assert "viewer_getting_started" not in [k for _, k in app.help_topic_dd.options]

    def test_browser_build_chrome(self, env, monkeypatch):
        from quantui.app import QuantUIApp

        monkeypatch.setattr(app_viewer, "IN_BROWSER", True)
        app = QuantUIApp(viewer=True)
        assert app._exit_btn.layout.display == "none"
        assert app.copy_path_btn.layout.display == "none"
        assert app._viewer_folder_txt.disabled
        assert app._viewer_open_btn.layout.display == "none"
        issue_text = app._issue_overlay.children[0].value
        assert app_viewer.ISSUES_URL in issue_text and "gone" in issue_text

    def test_copy_path_without_threads(self, env, monkeypatch):
        from quantui.app import QuantUIApp
        from quantui.app_runflow import on_copy_results_path

        app = QuantUIApp(viewer=True)

        def _no_threads(self, *a, **k):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", _no_threads)
        on_copy_results_path(app, None)  # must not raise
        assert "Copied" in app.results_path_lbl.value

    def test_trajectory_title_keeps_its_hint(self, env):
        from quantui.app import QuantUIApp
        from quantui.results_storage import save_result

        res = SimpleNamespace(
            formula="H2O", method="RHF", basis="STO-3G", energy_hartree=-74.9
        )
        rd = save_result(res, calc_type="single_point", molecule=_water())
        app = QuantUIApp(viewer=True)
        ctx = app._build_history_context(rd)
        app._apply_analysis_context(ctx)
        assert app.traj_accordion.get_title(0).startswith("Trajectory Viewer — ")


# ── Cosmetic: energy-level labels out of the gridlines ─────────────────────


def test_homo_lumo_labels_sit_in_the_margin():
    pytest.importorskip("plotly")
    import numpy as np

    from quantui.orbital_visualization import plot_orbital_diagram_plotly

    info = SimpleNamespace(
        formula="H2O",
        mo_energies_ev=np.array([-560.0, -36.0, -18.0, -14.0, -10.6, 16.5, 19.0]),
        n_occupied=5,
        homo_energy_ev=-10.6,
        lumo_energy_ev=16.5,
        homo_lumo_gap_ev=27.1,
        irreps=None,
    )
    fig = plot_orbital_diagram_plotly(info)
    labels = [a for a in fig.layout.annotations if a.text and "HOMO" in a.text]
    assert labels and labels[0].xref == "paper" and labels[0].x == 1.0
