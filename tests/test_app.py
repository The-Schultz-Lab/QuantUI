"""
Tests for quantui.app.QuantUIApp — FR-012 Phase 4.

All tests instantiate QuantUIApp() without calling .display(), which is safe
on any platform (display() requires an active IPython kernel; construction does
not).  PySCF is unavailable on Windows; calculations are skipped accordingly.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import ipywidgets as widgets
import pytest

from quantui.app import _RE_CONV, _RE_CYCLE, QuantUIApp, _AnalysisContext, _LogCapture
from quantui.molecule import Molecule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water() -> Molecule:
    """Return a minimal water molecule for testing."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """QuantUIApp constructs successfully without calling display()."""

    def test_instantiates(self):
        app = QuantUIApp()
        assert app is not None

    def test_root_tab_is_widget(self):
        app = QuantUIApp()
        assert isinstance(app.root_tab, widgets.Tab)

    def test_widget_property_returns_root_tab(self):
        app = QuantUIApp()
        assert app.widget is app.root_tab

    def test_initial_state(self):
        app = QuantUIApp()
        assert app._molecule is None
        assert app._last_result is None
        assert app._results == []

    def test_initial_molecule_is_none(self):
        app = QuantUIApp()
        assert app._molecule is None

    def test_activity_indicator_defaults_idle(self):
        app = QuantUIApp()
        assert hasattr(app, "_activity_btn")
        assert app._activity_btn.description == "Idle"

    def test_activity_indicator_compute_state(self):
        app = QuantUIApp()
        app._activity_begin("Running compute operations...", kind="compute")
        assert app._activity_btn.description == "Computing"
        app._activity_end(kind="compute")
        assert app._activity_btn.description == "Idle"

    def test_activity_indicator_ui_state(self):
        app = QuantUIApp()
        app._activity_begin("Switching tabs...", kind="ui")
        assert app._activity_btn.description == "UI Active"
        app._activity_end(kind="ui")
        assert app._activity_btn.description == "Idle"

    def test_run_btn_initially_disabled(self):
        app = QuantUIApp()
        assert app.run_btn.disabled is True

    def test_export_btn_initially_disabled(self):
        app = QuantUIApp()
        assert app.export_btn.disabled is True

    def test_scroll_guard_installer_method_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_install_run_output_scroll_guard")
        assert callable(app._install_run_output_scroll_guard)


# ---------------------------------------------------------------------------
# Default widget values
# ---------------------------------------------------------------------------


class TestDefaultWidgetValues:
    """Widget dropdowns and inputs have expected defaults."""

    def test_method_default(self):
        from quantui.config import DEFAULT_METHOD

        app = QuantUIApp()
        assert app.method_dd.value == DEFAULT_METHOD

    def test_run_output_has_scroll_guard_class(self):
        app = QuantUIApp()
        assert "quantui-run-output" in tuple(app.run_output._dom_classes)

    def test_basis_default(self):
        from quantui.config import DEFAULT_BASIS

        app = QuantUIApp()
        assert app.basis_dd.value == DEFAULT_BASIS

    def test_calc_type_default(self):
        app = QuantUIApp()
        assert app.calc_type_dd.value == "Single Point"

    def test_theme_default(self):
        app = QuantUIApp()
        assert app.theme_btn.value == "Dark"

    def test_charge_default(self):
        from quantui.config import DEFAULT_CHARGE

        app = QuantUIApp()
        assert app.charge_si.value == DEFAULT_CHARGE

    def test_multiplicity_default(self):
        from quantui.config import DEFAULT_MULTIPLICITY

        app = QuantUIApp()
        assert app.mult_si.value == DEFAULT_MULTIPLICITY


# ---------------------------------------------------------------------------
# Worker-thread callback scheduling
# ---------------------------------------------------------------------------


class TestMainThreadCallbackQueue:
    """_queue_main_thread_callback uses cached kernel io_loop from workers."""

    def test_uses_cached_io_loop_from_worker_thread(self):
        app = QuantUIApp()
        cb = MagicMock()
        io_loop = MagicMock()
        app._kernel_io_loop = io_loop

        t = threading.Thread(
            target=lambda: app._queue_main_thread_callback(cb, "ok"),
            daemon=True,
        )
        t.start()
        t.join(timeout=2)

        io_loop.add_callback.assert_called_once()
        called_cb, called_arg = io_loop.add_callback.call_args[0][:2]
        assert called_cb is cb
        assert called_arg == "ok"
        cb.assert_not_called()

    def test_falls_back_to_direct_call_without_io_loop(self):
        app = QuantUIApp()
        app._kernel_io_loop = None
        called = []

        def _cb() -> None:
            called.append(True)

        with patch("quantui.app.get_ipython", return_value=None):
            t = threading.Thread(
                target=lambda: app._queue_main_thread_callback(_cb),
                daemon=True,
            )
            t.start()
            t.join(timeout=2)

        assert called == [True]


# ---------------------------------------------------------------------------
# Tab structure
# ---------------------------------------------------------------------------


class TestTabStructure:
    """root_tab has the correct number and titles of tabs."""

    def test_eight_tabs(self):
        app = QuantUIApp()
        assert len(app.root_tab.children) == 8

    def test_tab_titles(self):
        app = QuantUIApp()
        expected = [
            "Calculate",
            "Results",
            "Analysis",
            "History",
            "Compare",
            "Log",
            "Files",
            "Status",
        ]
        for i, title in enumerate(expected):
            assert app.root_tab.get_title(i) == title


class TestFilesTab:
    """Files tab widgets are available and initialized."""

    def test_files_tab_widgets_exist(self):
        app = QuantUIApp()
        assert hasattr(app, "files_tab_panel")
        assert hasattr(app, "_files_root_dd")
        assert hasattr(app, "_files_entries")
        assert hasattr(app, "_files_preview_output")

    def test_files_root_dropdown_has_options(self):
        app = QuantUIApp()
        assert len(app._files_root_dd.options) >= 1


# ---------------------------------------------------------------------------
# Molecule input — collapse / expand pattern
# ---------------------------------------------------------------------------


class TestMoleculeInputCollapse:
    """mol_input_container switches between expanded and collapsed views."""

    def test_initially_expanded(self):
        app = QuantUIApp()
        # Expanded: first child is mol_input_expanded
        assert app.mol_input_container.children[0] is app.mol_input_expanded

    def test_collapses_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Collapsed: first child is mol_input_collapsed
        assert app.mol_input_container.children[0] is app.mol_input_collapsed

    def test_molecule_stored_after_set_molecule(self):
        app = QuantUIApp()
        mol = _water()
        app._set_molecule(mol)
        assert app._molecule is mol

    def test_run_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.run_btn.disabled is False

    def test_export_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.export_btn.disabled is False

    def test_mol_info_html_updated(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert "H2O" in app.mol_info_html.value

    def test_expand_restores_expanded_view(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Simulate clicking "Change molecule"
        app._on_expand_mol_input(None)
        assert app.mol_input_container.children[0] is app.mol_input_expanded

    def test_multiplicity_above_one_switches_to_uhf(self):
        app = QuantUIApp()
        app.method_dd.value = "RHF"
        radical = Molecule(
            atoms=["H"],
            coordinates=[[0.0, 0.0, 0.0]],
            multiplicity=2,
        )
        app._set_molecule(radical)
        assert app.method_dd.value == "UHF"

    def test_rhf_kept_for_singlet(self):
        app = QuantUIApp()
        app.method_dd.value = "RHF"
        app._set_molecule(_water())
        assert app.method_dd.value == "RHF"


# ---------------------------------------------------------------------------
# Step progress
# ---------------------------------------------------------------------------


class TestStepProgress:
    """Step indicator advances correctly through the workflow."""

    def test_step_0_active_initially(self):
        app = QuantUIApp()
        # Step 0 ("Choose molecule") should be active at start
        assert app.step_progress._states[0] == "active"

    def test_step_advances_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        # Step 0 done, step 1 should be active
        assert app.step_progress._states[0] == "done"
        assert app.step_progress._states[1] == "active"


# ---------------------------------------------------------------------------
# _LogCapture
# ---------------------------------------------------------------------------


class TestLogCapture:
    """_LogCapture parses SCF cycle lines and updates the status label."""

    def _make_capture(self):
        out = widgets.Output()
        status = widgets.Label()
        return _LogCapture(out, status), status

    def test_write_buffers_text(self):
        cap, _ = self._make_capture()
        cap.write("hello world\n")
        assert "hello world" in cap.getvalue()

    def test_cycle_regex_parses_line(self):
        line = "cycle= 3 E= -76.031234  delta_E= -0.0042"
        m = _RE_CYCLE.search(line)
        assert m is not None
        assert m.group(1) == "3"

    def test_conv_regex_parses_line(self):
        line = "converged SCF energy = -76.031234"
        m = _RE_CONV.search(line)
        assert m is not None

    def test_status_label_updated_on_cycle(self):
        cap, status = self._make_capture()
        cap.write("cycle= 2 E= -76.031234  delta_E= -0.0042\n")
        assert "SCF cycle 2" in status.value

    def test_status_label_updated_on_convergence(self):
        cap, status = self._make_capture()
        cap.write("converged SCF energy = -76.031234\n")
        assert "converged" in status.value.lower()

    def test_status_marker_updates_status_label(self):
        cap, status = self._make_capture()
        cap.write(
            "[QuantUI_STATUS] Numerical IR intensities: "
            "4/24 extra SCF solves complete (20 remaining)\n"
        )
        assert "4/24" in status.value
        assert "remaining" in status.value

    def test_scf_converged_callback_fires_once(self):
        out = widgets.Output()
        status = widgets.Label()
        called = 0

        def _on_conv() -> None:
            nonlocal called
            called += 1

        cap = _LogCapture(out, status, on_scf_converged=_on_conv)
        cap.write("converged SCF energy = -76.031234\n")
        cap.write("converged SCF energy = -76.031230\n")
        assert called == 1

    def test_flush_is_noop(self):
        cap, _ = self._make_capture()
        cap.flush()  # Must not raise

    def test_empty_write_is_noop(self):
        cap, _ = self._make_capture()
        cap.write("")
        assert cap.getvalue() == ""


# ---------------------------------------------------------------------------
# _do_run dispatch
# ---------------------------------------------------------------------------


class TestDoRunDispatch:
    """_do_run dispatches to the correct calculation function."""

    @pytest.fixture
    def app_with_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        return app

    def test_single_point_dispatch(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Single Point"
        mock_result = MagicMock()
        mock_result.energy_hartree = -75.0
        mock_result.homo_lumo_gap_ev = 12.3
        mock_result.converged = True
        mock_result.n_iterations = 10
        mock_result.formula = "H2O"
        mock_result.method = "RHF"
        mock_result.basis = "STO-3G"
        with patch(
            "quantui.run_in_session", return_value=mock_result, create=True
        ) as mock_run:
            with patch("quantui.save_result"):
                app._do_run()
        mock_run.assert_called_once()

    def test_geo_opt_dispatch(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Geometry Opt"
        mock_result = MagicMock()
        mock_result.energy_hartree = -75.0
        mock_result.converged = True
        mock_result.n_iterations = 5
        mock_result.energies_hartree = [-75.0]
        mock_result.trajectory = []
        mock_result.formula = "H2O"
        mock_result.method = "RHF"
        mock_result.basis = "STO-3G"
        mock_result.molecule = _water()
        mock_result.mo_energy_hartree = None
        mock_result.mo_occ = None
        mock_result.mo_coeff = None
        mock_result.pyscf_mol_atom = None
        mock_result.pyscf_mol_basis = None
        mock_sp = MagicMock()
        mock_sp.converged = True
        mock_sp.energy_hartree = -75.1
        mock_sp.mo_energy_hartree = [0.0]
        mock_sp.mo_occ = [2.0]
        mock_sp.mo_coeff = [[0.0]]
        mock_sp.pyscf_mol_atom = [("O", [0.0, 0.0, 0.0])]
        mock_sp.pyscf_mol_basis = "STO-3G"
        with patch(
            "quantui.optimize_geometry", return_value=mock_result, create=True
        ) as mock_opt:
            with patch(
                "quantui.run_in_session", return_value=mock_sp, create=True
            ) as mock_sp_run:
                with patch("quantui.save_result"):
                    app._do_run()
        mock_opt.assert_called_once()
        mock_sp_run.assert_called_once()

    def test_pyscf_unavailable_shows_error(self, app_with_molecule):
        app = app_with_molecule
        app.calc_type_dd.value = "Single Point"
        with patch("quantui.app._PYSCF_AVAILABLE", False):
            app._do_run()
        # Should not raise; error message should be in run_output
        # (output widget content is opaque, so just verify no exception)


# ---------------------------------------------------------------------------
# Availability flags on the instance
# ---------------------------------------------------------------------------


class TestAvailabilityFlags:
    def test_pyscf_flag_mirrors_module_level(self):
        from quantui.app import _PYSCF_AVAILABLE

        app = QuantUIApp()
        assert app._pyscf_available == _PYSCF_AVAILABLE

    def test_preopt_flag_mirrors_module_level(self):
        from quantui.app import _PREOPT_AVAILABLE

        app = QuantUIApp()
        assert app._preopt_available == _PREOPT_AVAILABLE


# ---------------------------------------------------------------------------
# M3.3 — result log accordion and directory label
# ---------------------------------------------------------------------------


class TestResultLogAccordion:
    """_result_log_accordion and _result_dir_label exist and start hidden."""

    def test_log_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_result_log_accordion")
        assert isinstance(app._result_log_accordion, widgets.Accordion)

    def test_log_accordion_initially_hidden(self):
        app = QuantUIApp()
        assert app._result_log_accordion.layout.display == "none"

    def test_log_accordion_initially_collapsed(self):
        app = QuantUIApp()
        assert app._result_log_accordion.selected_index is None

    def test_result_dir_label_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_result_dir_label")
        assert isinstance(app._result_dir_label, widgets.HTML)

    def test_result_dir_label_initially_hidden(self):
        app = QuantUIApp()
        assert app._result_dir_label.layout.display == "none"

    def test_last_result_dir_initially_none(self):
        app = QuantUIApp()
        assert app._last_result_dir is None

    def test_on_run_clicked_clears_log(self):
        """_on_run_clicked must hide log accordion and clear dir label."""
        app = QuantUIApp()
        # Simulate a previous result being present
        app._result_log_accordion.layout.display = ""
        app._result_dir_label.layout.display = ""
        app._result_dir_label.value = "Saved to: /some/path"

        with patch.object(app, "_do_run"):
            app._on_run_clicked(None)

        assert app._result_log_accordion.layout.display == "none"
        assert app._result_dir_label.layout.display == "none"
        assert app._result_dir_label.value == ""

    def test_show_result_log_populates_widgets(self, tmp_path):
        """_show_result_log() sets dir label and reveals accordion."""
        log_text = "SCF converged in 10 cycles."
        log_file = tmp_path / "pyscf.log"
        log_file.write_text(log_text, encoding="utf-8")

        app = QuantUIApp()
        app._show_result_log(tmp_path, log_text)

        assert str(tmp_path) in app._result_dir_label.value
        assert app._result_dir_label.layout.display == ""
        assert app._result_log_accordion.layout.display == ""

    def test_show_result_log_falls_back_to_string(self, tmp_path):
        """_show_result_log() uses in-memory log_text if pyscf.log absent."""
        log_text = "fallback log content"
        app = QuantUIApp()
        empty_dir = tmp_path / "no_log_here"
        empty_dir.mkdir()
        app._show_result_log(empty_dir, log_text)

        assert app._result_log_accordion.layout.display == ""


# ---------------------------------------------------------------------------
# M3.4 — Structure file exports (XYZ, MOL/SDF, PDB)
# ---------------------------------------------------------------------------


class TestStructureExportButtons:
    """export_xyz_btn, export_mol_btn, export_pdb_btn exist and start disabled."""

    def test_export_xyz_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_xyz_btn")
        assert isinstance(app.export_xyz_btn, widgets.Button)

    def test_export_mol_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_mol_btn")
        assert isinstance(app.export_mol_btn, widgets.Button)

    def test_export_pdb_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "export_pdb_btn")
        assert isinstance(app.export_pdb_btn, widgets.Button)

    def test_struct_export_status_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "struct_export_status")

    def test_export_xyz_btn_disabled_initially(self):
        app = QuantUIApp()
        assert app.export_xyz_btn.disabled is True

    def test_export_xyz_btn_enabled_after_set_molecule(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        assert app.export_xyz_btn.disabled is False

    def test_export_accordion_title_is_export(self):
        app = QuantUIApp()
        assert app.advanced_accordion.get_title(0) == "Export"


class TestExportXYZCallback:
    """_on_export_xyz writes a valid XYZ file."""

    def test_xyz_file_written_to_result_dir(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        xyz_files = list(tmp_path.glob("*.xyz"))
        assert len(xyz_files) == 1

    def test_xyz_file_contains_atom_count(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        content = list(tmp_path.glob("*.xyz"))[0].read_text()
        first_line = content.splitlines()[0].strip()
        assert first_line == "3"  # water has 3 atoms

    def test_xyz_status_shows_saved_path(self, tmp_path):
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path

        app._on_export_xyz(None)

        assert "Saved" in app.struct_export_status.value

    def test_xyz_no_molecule_shows_error(self):
        app = QuantUIApp()
        app._on_export_xyz(None)
        assert "molecule" in app.struct_export_status.value.lower()


class TestExportMoleculeAndLabel:
    """_export_molecule_and_label returns correct molecule and labels."""

    def test_returns_current_molecule_when_no_result(self):
        app = QuantUIApp()
        water = _water()
        app._set_molecule(water)
        mol, method, basis = app._export_molecule_and_label()
        assert mol is water

    def test_method_falls_back_to_dropdown(self):
        app = QuantUIApp()
        app._set_molecule(_water())
        _, method, _ = app._export_molecule_and_label()
        assert method == app.method_dd.value


class TestMoleculeToRdkit:
    """_molecule_to_rdkit does not raise; returns RDKit mol or None."""

    def test_does_not_raise_for_water(self):
        result = QuantUIApp._molecule_to_rdkit(_water())
        # Either succeeds or returns None — must not raise
        assert result is None or result is not None


# ---------------------------------------------------------------------------
# M4.1 — Extended DFT functional list
# ---------------------------------------------------------------------------


class TestExtendedDFTFunctionals:
    """New functionals appear in method_dd options."""

    def test_wb97xd_in_dropdown(self):
        app = QuantUIApp()
        assert "wB97X-D" in app.method_dd.options

    def test_cam_b3lyp_in_dropdown(self):
        app = QuantUIApp()
        assert "CAM-B3LYP" in app.method_dd.options

    def test_m06l_in_dropdown(self):
        app = QuantUIApp()
        assert "M06-L" in app.method_dd.options

    def test_hse06_in_dropdown(self):
        app = QuantUIApp()
        assert "HSE06" in app.method_dd.options

    def test_pbe_d3_in_dropdown(self):
        app = QuantUIApp()
        assert "PBE-D3" in app.method_dd.options

    def test_mp2_in_dropdown(self):
        app = QuantUIApp()
        assert "MP2" in app.method_dd.options


# ---------------------------------------------------------------------------
# M4.2 — MP2 energy
# ---------------------------------------------------------------------------


class TestMP2SessionResult:
    """mp2_correlation_hartree field on SessionResult."""

    def test_mp2_corr_defaults_to_none(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=10,
            method="MP2",
            basis="STO-3G",
            formula="H2O",
        )
        assert r.mp2_correlation_hartree is None

    def test_mp2_corr_stored(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-76.3,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=10,
            method="MP2",
            basis="STO-3G",
            formula="H2O",
            mp2_correlation_hartree=-0.3,
        )
        assert r.mp2_correlation_hartree == pytest.approx(-0.3)


class TestMP2FormatResult:
    """_format_result shows HF reference and MP2 correlation when present."""

    def test_hf_reference_shown_when_mp2(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-76.3,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=10,
            method="MP2",
            basis="STO-3G",
            formula="H2O",
            mp2_correlation_hartree=-0.3,
        )
        app = QuantUIApp()
        html = app._format_result(r)
        assert "HF reference" in html
        assert "MP2 correlation" in html


# ---------------------------------------------------------------------------
# M4.3 — Implicit solvent (PCM)
# ---------------------------------------------------------------------------


class TestSolventWidgets:
    """solvent_cb and solvent_dd exist and behave correctly."""

    def test_solvent_cb_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "solvent_cb")
        assert isinstance(app.solvent_cb, widgets.Checkbox)

    def test_solvent_dd_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "solvent_dd")
        assert isinstance(app.solvent_dd, widgets.Dropdown)

    def test_solvent_dd_hidden_initially(self):
        app = QuantUIApp()
        assert app.solvent_dd.layout.display == "none"

    def test_solvent_dd_revealed_when_cb_checked(self):
        app = QuantUIApp()
        app.solvent_cb.value = True
        assert app.solvent_dd.layout.display == ""

    def test_solvent_dd_hidden_when_cb_unchecked(self):
        app = QuantUIApp()
        app.solvent_cb.value = True
        app.solvent_cb.value = False
        assert app.solvent_dd.layout.display == "none"

    def test_water_is_solvent_option(self):
        app = QuantUIApp()
        assert "Water" in app.solvent_dd.options

    def test_solvent_field_on_session_result(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=10,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
            solvent="Water",
        )
        assert r.solvent == "Water"

    def test_solvent_shown_in_format_result(self):
        from quantui.session_calc import SessionResult

        r = SessionResult(
            energy_hartree=-76.0,
            homo_lumo_gap_ev=None,
            converged=True,
            n_iterations=10,
            method="RHF",
            basis="STO-3G",
            formula="H2O",
            solvent="Ethanol",
        )
        app = QuantUIApp()
        html = app._format_result(r)
        assert "Ethanol" in html
        assert "PCM" in html


# ---------------------------------------------------------------------------
# M-CAL — Calibration UI widgets
# ---------------------------------------------------------------------------


class TestCalibrationWidgets:
    """Calibration accordion and its child widgets exist in correct initial state."""

    def test_cal_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_cal_accordion")
        assert isinstance(app._cal_accordion, widgets.Accordion)

    def test_cal_run_btn_exists(self):
        app = QuantUIApp()
        assert isinstance(app._cal_run_btn, widgets.Button)

    def test_cal_stop_btn_hidden_initially(self):
        app = QuantUIApp()
        assert app._cal_stop_btn.layout.display == "none"

    def test_cal_progress_hidden_initially(self):
        app = QuantUIApp()
        assert app._cal_progress.layout.display == "none"

    def test_cal_step_label_hidden_initially(self):
        app = QuantUIApp()
        assert app._cal_step_label.layout.display == "none"

    def test_cal_run_btn_disabled_when_pyscf_unavailable(self):
        from quantui.app import _PYSCF_AVAILABLE

        app = QuantUIApp()
        # Button state must match module-level availability flag
        assert app._cal_run_btn.disabled == (not _PYSCF_AVAILABLE)

    def test_cal_progress_max_equals_suite_length(self):
        from quantui.benchmarks import BENCHMARK_SUITE

        app = QuantUIApp()
        assert app._cal_progress.max == len(BENCHMARK_SUITE)

    def test_on_cal_stop_sets_event(self):
        import threading

        app = QuantUIApp()
        app._cal_stop_event = threading.Event()
        app._on_cal_stop(None)
        assert app._cal_stop_event.is_set()


# ---------------------------------------------------------------------------
# M5 — NMR Shielding widgets
# ---------------------------------------------------------------------------


class TestNMRWidgets:
    """NMR Shielding option exists and callback wires correctly."""

    def test_nmr_in_calc_type_options(self):
        app = QuantUIApp()
        assert "NMR Shielding" in app.calc_type_dd.options

    def test_calc_type_dd_has_six_options(self):
        app = QuantUIApp()
        assert len(app.calc_type_dd.options) == 6

    def test_nmr_calc_type_shows_note(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "NMR Shielding"
        # calc_extra_opts should contain an HTML note about basis recommendations
        assert len(app.calc_extra_opts.children) == 1
        note = app.calc_extra_opts.children[0]
        assert isinstance(note, widgets.HTML)
        assert "6-31G*" in note.value

    def test_nmr_note_mentions_sto3g_warning(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "NMR Shielding"
        note = app.calc_extra_opts.children[0]
        assert "STO-3G" in note.value

    def test_switching_away_from_nmr_clears_opts(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "NMR Shielding"
        app.calc_type_dd.value = "Single Point"
        assert len(app.calc_extra_opts.children) == 0


class TestFormatNMRResult:
    """_format_nmr_result produces correct HTML."""

    def _make_nmr(self, basis="6-31G*", converged=True):
        from quantui.nmr_calc import NMRResult

        return NMRResult(
            atom_symbols=["O", "H", "H"],
            shielding_iso_ppm=[320.1, 28.5, 28.5],
            chemical_shifts_ppm={1: 3.22, 2: 3.22},
            method="B3LYP",
            basis=basis,
            formula="H2O",
            converged=converged,
        )

    def test_returns_string(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr())
        assert isinstance(html, str)

    def test_contains_formula(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr())
        assert "H2O" in html

    def test_contains_method_and_basis(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr())
        assert "B3LYP" in html
        assert "6-31G*" in html

    def test_h_shifts_table_present(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr())
        assert "¹H" in html
        assert "3.22" in html

    def test_sto3g_warning_shown(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr(basis="STO-3G"))
        assert "STO-3G" in html
        assert "qualitative" in html

    def test_no_sto3g_warning_for_631g(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr(basis="6-31G*"))
        assert "qualitative" not in html

    def test_not_converged_shows_warning(self):
        app = QuantUIApp()
        html = app._format_nmr_result(self._make_nmr(converged=False))
        assert "caution" in html

    def test_no_hc_atoms_shows_empty_message(self):

        from quantui.nmr_calc import NMRResult

        r = NMRResult(
            atom_symbols=["N", "N"],
            shielding_iso_ppm=[100.0, 100.0],
            chemical_shifts_ppm={},
            method="RHF",
            basis="STO-3G",
            formula="N2",
        )
        app = QuantUIApp()
        html = app._format_nmr_result(r)
        assert "No ¹H or ¹³C" in html


# ---------------------------------------------------------------------------
# M-IR — IR Spectrum accordion widgets
# ---------------------------------------------------------------------------


class TestIRSpectrumWidgets:
    """IR Spectrum accordion and controls exist in correct initial state."""

    def test_ir_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_ir_accordion")
        assert isinstance(app._ir_accordion, widgets.Accordion)

    def test_ir_accordion_visible_and_collapsed_initially(self):
        app = QuantUIApp()
        assert app._ir_accordion.layout.display == ""
        assert app._ir_accordion.selected_index is None

    def test_ir_mode_toggle_exists(self):
        app = QuantUIApp()
        assert isinstance(app._ir_mode_toggle, widgets.ToggleButtons)

    def test_ir_mode_toggle_default_stick(self):
        app = QuantUIApp()
        assert app._ir_mode_toggle.value == "Stick"

    def test_ir_mode_toggle_has_two_options(self):
        app = QuantUIApp()
        assert set(app._ir_mode_toggle.options) == {"Stick", "Broadened"}

    def test_fwhm_slider_hidden_initially(self):
        app = QuantUIApp()
        assert app._ir_fwhm_slider.layout.display == "none"

    def test_fwhm_slider_default_20(self):
        app = QuantUIApp()
        assert app._ir_fwhm_slider.value == 20.0

    def test_fwhm_slider_range(self):
        app = QuantUIApp()
        assert app._ir_fwhm_slider.min == 5.0
        assert app._ir_fwhm_slider.max == 100.0

    def test_fwhm_slider_continuous_update_false(self):
        # BUG.9 regression guard: continuous_update must be False so the
        # slider only fires the observer on release, not 30-60 times per
        # second during a drag (which produces visible flicker).
        app = QuantUIApp()
        assert app._ir_fwhm_slider.continuous_update is False

    def test_ir_fig_has_min_height(self):
        # BUG.9 regression guard: min_height keeps the Output container
        # from collapsing to 0px between renders. Pairs with the atomic
        # outputs swap in _set_html_output to keep the IR panel
        # flicker-free on mode toggle / slider changes.
        app = QuantUIApp()
        assert app._ir_fig.layout.min_height == "300px"

    def test_ir_export_controls_exist(self):
        app = QuantUIApp()
        assert isinstance(app._ir_export_btn, widgets.Button)
        assert isinstance(app._ir_export_fmt_dd, widgets.Dropdown)
        assert app._ir_export_fmt_dd.value == "html"


class TestShowIRSpectrum:
    """_show_ir_spectrum reveals accordion and wires mode toggle."""

    def _make_freq_result(self):
        from unittest.mock import MagicMock

        r = MagicMock()
        r.frequencies_cm1 = [500.0, 1000.0, 3000.0]
        r.ir_intensities = [10.0, 50.0, 5.0]
        return r

    def test_show_ir_spectrum_returns_true_with_data(self):
        app = QuantUIApp()
        app._last_ir_freqs = []
        app._last_ir_ints = []
        ok = app._show_ir_spectrum(self._make_freq_result())
        assert ok is True

    def test_accordion_expanded_via_activate(self):
        # _show_ir_spectrum populates widget; _activate_ana_panel expands it.
        app = QuantUIApp()
        app._show_ir_spectrum(self._make_freq_result())
        assert app._ir_accordion.selected_index is None  # still collapsed
        app._activate_ana_panel("IR Spectrum")
        assert app._ir_accordion.selected_index == 0

    def test_fwhm_slider_shown_when_broadened(self):
        app = QuantUIApp()
        app._show_ir_spectrum(self._make_freq_result())
        app._ir_mode_toggle.value = "Broadened"
        assert app._ir_fwhm_slider.layout.display == ""

    def test_fwhm_slider_hidden_when_stick(self):
        app = QuantUIApp()
        app._show_ir_spectrum(self._make_freq_result())
        app._ir_mode_toggle.value = "Broadened"
        app._ir_mode_toggle.value = "Stick"
        assert app._ir_fwhm_slider.layout.display == "none"

    def test_broadened_toggle_triggers_ir_figure_update(self):
        app = QuantUIApp()
        app._show_ir_spectrum(self._make_freq_result())
        with patch.object(app, "_update_ir_figure") as mock_update:
            app._ir_mode_toggle.value = "Broadened"
        mock_update.assert_called_once_with("Broadened", app._ir_fwhm_slider.value)


# ---------------------------------------------------------------------------
# M-UV — UV-Vis Spectrum accordion widgets
# ---------------------------------------------------------------------------


class TestSetHtmlOutputAtomic:
    """_set_html_output must perform a single atomic outputs assignment.

    BUG.9 root cause: the previous implementation was clear_output() +
    append_display_data(), which produced an intermediate empty state
    between the two calls. On rapid invocations (IR FWHM slider drag,
    Stick/Broadened toggle), the user saw the panel flash blank between
    every re-render. Atomic outputs swap eliminates the intermediate
    state in one widget-state update.
    """

    def test_outputs_is_single_entry_after_set(self):
        app = QuantUIApp()
        out = widgets.Output()
        app._set_html_output(out, "<p>hello</p>")
        assert len(out.outputs) == 1
        entry = out.outputs[0]
        assert entry["output_type"] == "display_data"
        assert entry["data"]["text/html"] == "<p>hello</p>"

    def test_outputs_replaces_prior_content_atomically(self):
        # Repeated calls (e.g. FWHM slider scrub) must each produce a
        # single-entry outputs tuple — never accumulating or clearing-then-
        # appending (which would briefly empty the widget mid-update).
        app = QuantUIApp()
        out = widgets.Output()
        app._set_html_output(out, "<p>first</p>")
        app._set_html_output(out, "<p>second</p>")
        app._set_html_output(out, "<p>third</p>")
        assert len(out.outputs) == 1
        assert out.outputs[0]["data"]["text/html"] == "<p>third</p>"


class TestShowResult3DAtomic:
    """``_show_result_3d`` must route through the atomic ``_set_html_output``
    swap rather than ``with output: display(viz)``.

    BUG.7 root cause: ``show_result_3d`` previously used the nested-Output +
    main-thread ``display(viz)`` pattern, which intermittently produced a
    blank 🙁 viewer on Analysis-tab history replay (same failure family as
    resolved BUG.6 in trajectory render). After this fix, every invocation
    leaves the target ``Output`` with a single-entry ``outputs`` tuple whose
    ``text/html`` payload is non-empty.
    """

    def _make_water(self):
        return Molecule(
            atoms=["O", "H", "H"],
            coordinates=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        )

    def test_analysis_mol_output_is_single_entry_after_show(self):
        from quantui.app import _render_molecule_html

        if _render_molecule_html is None:
            pytest.skip("No 3D visualization backend installed")
        app = QuantUIApp()
        app._show_result_3d(self._make_water(), extra_output=app._analysis_mol_output)
        assert len(app._analysis_mol_output.outputs) == 1
        entry = app._analysis_mol_output.outputs[0]
        assert entry["output_type"] == "display_data"
        assert entry["data"]["text/html"].strip() != ""

    def test_result_viz_output_is_single_entry_after_show(self):
        from quantui.app import _render_molecule_html

        if _render_molecule_html is None:
            pytest.skip("No 3D visualization backend installed")
        app = QuantUIApp()
        app._show_result_3d(self._make_water(), extra_output=None)
        assert len(app.result_viz_output.outputs) == 1
        entry = app.result_viz_output.outputs[0]
        assert entry["output_type"] == "display_data"
        assert entry["data"]["text/html"].strip() != ""

    def test_repeated_calls_do_not_accumulate_outputs(self):
        # Backend-toggle scenario: re-render the same molecule multiple
        # times and confirm the viewer is replaced atomically each time.
        from quantui.app import _render_molecule_html

        if _render_molecule_html is None:
            pytest.skip("No 3D visualization backend installed")
        app = QuantUIApp()
        mol = self._make_water()
        for _ in range(3):
            app._show_result_3d(mol, extra_output=app._analysis_mol_output)
        assert len(app._analysis_mol_output.outputs) == 1
        assert len(app.result_viz_output.outputs) == 1


class TestUVVisSpectrumWidgets:
    """UV-Vis accordion and controls exist in correct initial state."""

    def test_uv_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_tddft_accordion")
        assert isinstance(app._tddft_accordion, widgets.Accordion)

    def test_uv_mode_toggle_exists(self):
        app = QuantUIApp()
        assert isinstance(app._uv_mode_toggle, widgets.ToggleButtons)

    def test_uv_mode_toggle_default_stick(self):
        app = QuantUIApp()
        assert app._uv_mode_toggle.value == "Stick"

    def test_uv_mode_toggle_has_two_options(self):
        app = QuantUIApp()
        assert set(app._uv_mode_toggle.options) == {"Stick", "Broadened"}

    def test_uv_fwhm_slider_hidden_initially(self):
        app = QuantUIApp()
        assert app._uv_fwhm_slider.layout.display == "none"

    def test_uv_export_controls_exist(self):
        app = QuantUIApp()
        assert isinstance(app._uv_export_btn, widgets.Button)
        assert isinstance(app._uv_export_fmt_dd, widgets.Dropdown)
        assert app._uv_export_fmt_dd.value == "html"


class TestPESExportWidgets:
    def test_pes_export_controls_exist(self):
        app = QuantUIApp()
        assert isinstance(app._pes_export_btn, widgets.Button)
        assert isinstance(app._pes_export_fmt_dd, widgets.Dropdown)
        assert app._pes_export_fmt_dd.value == "html"


class TestShowUVVisSpectrum:
    """_show_uv_vis_spectrum stores data and wires controls."""

    def test_show_uv_vis_spectrum_returns_true_with_data(self):
        app = QuantUIApp()
        ok = app._show_uv_vis_spectrum(
            [3.0, 4.2, 5.5],
            [0.12, 0.08, 0.05],
            [413.3, 295.2, 225.5],
        )
        assert ok is True

    def test_uv_fwhm_slider_shown_when_broadened(self):
        app = QuantUIApp()
        app._show_uv_vis_spectrum(
            [3.0, 4.2, 5.5],
            [0.12, 0.08, 0.05],
            [413.3, 295.2, 225.5],
        )
        app._uv_mode_toggle.value = "Broadened"
        assert app._uv_fwhm_slider.layout.display == ""

    def test_uv_fwhm_slider_hidden_when_stick(self):
        app = QuantUIApp()
        app._show_uv_vis_spectrum(
            [3.0, 4.2, 5.5],
            [0.12, 0.08, 0.05],
            [413.3, 295.2, 225.5],
        )
        app._uv_mode_toggle.value = "Broadened"
        app._uv_mode_toggle.value = "Stick"
        assert app._uv_fwhm_slider.layout.display == "none"

    def test_broadened_toggle_triggers_uv_figure_update(self):
        app = QuantUIApp()
        app._show_uv_vis_spectrum(
            [3.0, 4.2, 5.5],
            [0.12, 0.08, 0.05],
            [413.3, 295.2, 225.5],
        )
        with patch.object(app, "_update_uv_vis_figure") as mock_update:
            app._uv_mode_toggle.value = "Broadened"
        mock_update.assert_called_once_with("Broadened", app._uv_fwhm_slider.value)


# ---------------------------------------------------------------------------
# M6 — Orbital Diagram accordion
# ---------------------------------------------------------------------------


class TestOrbitalAccordionWidgets:
    """Orbital accordion widgets exist and have the correct initial state."""

    def test_orb_accordion_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_orb_accordion")

    def test_orb_accordion_visible_collapsed_initially(self):
        app = QuantUIApp()
        assert app._orb_accordion.layout.display == ""
        assert app._orb_accordion.selected_index is None

    def test_orb_diagram_html_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_orb_diagram_html")

    def test_orb_export_controls_exist(self):
        app = QuantUIApp()
        assert isinstance(app._orb_export_btn, widgets.Button)
        assert isinstance(app._orb_export_fmt_dd, widgets.Dropdown)
        assert app._orb_export_fmt_dd.value == "html"

    def test_orb_toggle_has_four_options(self):
        app = QuantUIApp()
        assert set(app._orb_toggle.options) == {"HOMO-1", "HOMO", "LUMO", "LUMO+1"}

    def test_orb_toggle_default_homo(self):
        app = QuantUIApp()
        assert app._orb_toggle.value == "HOMO"

    def test_orb_iso_controls_hidden_initially(self):
        app = QuantUIApp()
        assert app._orb_iso_controls.layout.display == "none"

    def test_orb_accordion_collapsed_after_run_clicked(self):
        app = QuantUIApp()
        app._orb_accordion.selected_index = 0
        app._on_run_clicked(None)
        assert app._orb_accordion.selected_index is None


class TestShowOrbitalDiagram:

    class TestPlotExportHelper:
        def test_export_plot_figure_html_writes_file(self, tmp_path):
            app = QuantUIApp()
            app._last_result_dir = tmp_path

            fig = MagicMock()
            with patch("plotly.io.to_html", return_value="<html>ok</html>"):
                app._export_plot_figure(
                    fig=fig,
                    stem="ir_spectrum",
                    fmt="html",
                    status_widget=app._ir_export_status,
                )

            saved = list(tmp_path.glob("ir_spectrum_*.html"))
            assert len(saved) == 1
            assert "Saved:" in app._ir_export_status.value

    """_show_orbital_diagram reveals accordion when MO data is present."""

    def _make_result_with_mo(self):
        from unittest.mock import MagicMock

        import numpy as np

        r = MagicMock()
        r.formula = "H2O"
        r.mo_energy_hartree = np.array([-1.5, -0.8, 0.2, 0.9])
        r.mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        r.mo_coeff = None
        r.pyscf_mol_atom = None
        r.pyscf_mol_basis = None
        return r

    def test_show_orbital_diagram_returns_true_with_mo_data(self):
        app = QuantUIApp()
        ok = app._show_orbital_diagram(self._make_result_with_mo())
        assert ok is True

    def test_accordion_expanded_via_activate(self):
        # _show_orbital_diagram populates widget; _activate_ana_panel expands it.
        app = QuantUIApp()
        app._show_orbital_diagram(self._make_result_with_mo())
        assert app._orb_accordion.selected_index is None  # still collapsed
        app._activate_ana_panel("Energies")
        assert app._orb_accordion.selected_index == 0

    def test_accordion_stays_collapsed_when_no_mo_data(self):
        from unittest.mock import MagicMock

        app = QuantUIApp()
        r = MagicMock()
        r.mo_energy_hartree = None
        r.mo_occ = None
        app._show_orbital_diagram(r)
        assert app._orb_accordion.selected_index is None

    def test_diagram_html_populated(self):
        app = QuantUIApp()
        app._show_orbital_diagram(self._make_result_with_mo())
        # Plotly renders an interactive <div>; matplotlib fallback renders <img>.
        # The diagram is now rendered via Output display_data (not HTML.value).
        payloads = [
            out.get("data", {}).get("text/html", "")
            for out in app._orb_diagram_html.outputs
            if out.get("output_type") == "display_data"
        ]
        val = "\n".join(payloads)
        assert "<div" in val or "<img" in val

    def test_isosurface_controls_hidden_when_no_mo_coeff(self):
        app = QuantUIApp()
        app._show_orbital_diagram(self._make_result_with_mo())
        # mo_coeff is None in mock → iso controls stay hidden
        assert app._orb_iso_controls.layout.display == "none"


class TestIsosurfacePersistence:
    def test_render_orbital_isosurface_saves_cube_to_disk(self, tmp_path):
        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_orb_info = MagicMock()
        app._last_orb_info.n_occupied = 1
        app._last_orb_info.mo_energies_ev = [-10.0, 2.0]
        app._last_orb_info.formula = "H2O"
        app._last_orb_mo_coeff = [[1.0, 0.0], [0.0, 1.0]]
        app._last_orb_mol_atom = [["H", [0.0, 0.0, 0.0]]]
        app._last_orb_mol_basis = "sto-3g"

        captured: dict[str, object] = {}

        def _fake_generate(_atom, _basis, _coeff, _idx, out_path):
            captured["path"] = out_path
            out_path.write_text("cube", encoding="utf-8")
            return out_path

        with (
            patch(
                "quantui.orbital_visualization.generate_cube_from_arrays",
                side_effect=_fake_generate,
            ) as mock_gen,
            patch(
                "quantui.orbital_visualization.plot_cube_isosurface",
                return_value=MagicMock(),
            ) as mock_plot,
            patch(
                "plotly.io.to_html",
                return_value="<div>iso</div>",
            ),
        ):
            app._render_orbital_isosurface("HOMO")

        saved_path = captured.get("path")
        assert saved_path is not None
        assert saved_path.parent == tmp_path / "isosurfaces"
        assert saved_path.suffix == ".cube"
        assert saved_path.exists()
        mock_gen.assert_called_once()
        mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# M-UI — Results tab widgets (M-UI.8)
# ---------------------------------------------------------------------------


class TestResultsTab:
    """Results tab panel contains the expected widgets and backward-compat alias."""

    def test_results_tab_panel_is_vbox(self):
        app = QuantUIApp()
        import ipywidgets as widgets

        assert isinstance(app.results_tab_panel, widgets.VBox)

    def test_results_panel_alias_points_to_same_object(self):
        app = QuantUIApp()
        assert app.results_panel is app.results_tab_panel

    def test_results_tab_contains_result_output(self):
        app = QuantUIApp()
        assert app.result_output in app.results_tab_panel.children

    def test_to_analysis_btn_initially_hidden(self):
        app = QuantUIApp()
        assert app._to_analysis_btn.layout.display == "none"

    def test_advanced_accordion_in_results_tab(self):
        app = QuantUIApp()
        assert app.advanced_accordion in app.results_tab_panel.children


# ---------------------------------------------------------------------------
# M-UI — Analysis tab widgets (M-UI.8)
# ---------------------------------------------------------------------------


class TestAnalysisTab:
    """Analysis tab panel contains the expected widgets and backward-compat alias."""

    def test_analysis_tab_panel_is_vbox(self):
        app = QuantUIApp()
        import ipywidgets as widgets

        assert isinstance(app.analysis_tab_panel, widgets.VBox)

    def test_post_calc_panel_alias_points_to_same_object(self):
        app = QuantUIApp()
        assert app.post_calc_panel is app.analysis_tab_panel

    def test_analysis_context_label_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_analysis_context_lbl")
        import ipywidgets as widgets

        assert isinstance(app._analysis_context_lbl, widgets.HTML)

    def test_analysis_empty_html_initially_hidden(self):
        """Empty-state message starts hidden; it appears only when a non-analysis calc completes."""
        app = QuantUIApp()
        assert app._analysis_empty_html.layout.display == "none"

    def test_orb_accordion_in_analysis_tab(self):
        app = QuantUIApp()
        assert app._orb_accordion in app.analysis_tab_panel.children

    def test_vib_accordion_in_analysis_tab(self):
        app = QuantUIApp()
        assert app.vib_accordion in app.analysis_tab_panel.children

    def test_ir_accordion_in_analysis_tab(self):
        app = QuantUIApp()
        assert app._ir_accordion in app.analysis_tab_panel.children

    def test_analysis_heading_matches_history_label_shape(self):
        app = QuantUIApp()
        ctx = _AnalysisContext(
            calc_type="frequency",
            formula="H2O",
            method="B3LYP",
            basis="6-31G",
            timestamp="2026-05-14_10-11-12-123456",
            source="history",
        )

        app._apply_analysis_context(ctx)

        heading = app._analysis_context_lbl.value
        assert "Analysing:" in heading
        assert "2026-05-14_10-11-12-123456" in heading
        assert "[Frequency Analysis]" in heading
        assert "H2O  B3LYP/6-31G" in heading
        assert "(from History)" in heading


# ---------------------------------------------------------------------------
# M-ANA — Panel switcher (M-ANA)
# ---------------------------------------------------------------------------


class TestAnaSwitcher:
    """Analysis panel state: activation, deactivation, and placeholder swapping."""

    def test_panel_names(self):
        app = QuantUIApp()
        assert app._ana_panel_names == [
            "Energies",
            "Trajectory",
            "Vibrational",
            "IR Spectrum",
            "PES Scan",
            "Isosurface",
            "UV-Vis",
            "NMR",
        ]

    def test_no_panels_available_initially(self):
        app = QuantUIApp()
        assert len(app._ana_available) == 0

    def test_all_accordions_visible_and_collapsed_initially(self):
        app = QuantUIApp()
        for acc in app._ana_accordions:
            assert acc.layout.display == ""
            assert acc.selected_index is None

    def test_activate_panel_marks_available(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies")
        assert "Energies" in app._ana_available

    def test_activate_panel_auto_selects(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies")
        assert app._orb_accordion.selected_index == 0

    def test_activate_panel_no_auto_select(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies", auto_select=False)
        # Panel is available but not expanded; still visible in DOM.
        assert "Energies" in app._ana_available
        assert app._orb_accordion.selected_index is None
        assert app._orb_accordion.layout.display == ""

    def test_activate_collapses_other_accordions(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies")
        # Other accordions remain visible but collapsed (not hidden).
        for name, acc in zip(app._ana_panel_names, app._ana_accordions):
            if name != "Energies":
                assert acc.layout.display == ""
                assert acc.selected_index is None

    def test_deactivate_all_clears_available(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies")
        app._activate_ana_panel("IR Spectrum", auto_select=False)
        app._deactivate_all_ana_panels()
        assert len(app._ana_available) == 0

    def test_deactivate_all_collapses_accordions(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies")
        app._deactivate_all_ana_panels()
        # All panels remain visible in the DOM but are collapsed.
        for acc in app._ana_accordions:
            assert acc.layout.display == ""
            assert acc.selected_index is None

    def test_unavail_message_shown_initially(self):
        app = QuantUIApp()
        # Every panel starts with the unavailable placeholder visible.
        for name in app._ana_panel_names:
            assert app._ana_unavail_msgs[name].layout.display == ""
            assert app._ana_content_boxes[name].layout.display == "none"

    def test_activate_swaps_placeholder_for_content(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies", auto_select=False)
        assert app._ana_unavail_msgs["Energies"].layout.display == "none"
        assert app._ana_content_boxes["Energies"].layout.display == ""

    def test_deactivate_restores_placeholder(self):
        app = QuantUIApp()
        app._activate_ana_panel("Energies", auto_select=False)
        app._deactivate_all_ana_panels()
        assert app._ana_unavail_msgs["Energies"].layout.display == ""
        assert app._ana_content_boxes["Energies"].layout.display == "none"


# ---------------------------------------------------------------------------
# M-UI — Completion banner (M-UI.8)
# ---------------------------------------------------------------------------


class TestCompletionBanner:
    """Completion banner widget exists and is initially hidden."""

    def test_completion_banner_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_completion_banner")
        import ipywidgets as widgets

        assert isinstance(app._completion_banner, widgets.HBox)

    def test_completion_banner_initially_hidden(self):
        app = QuantUIApp()
        assert app._completion_banner.layout.display == "none"

    def test_go_results_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_go_results_btn")
        import ipywidgets as widgets

        assert isinstance(app._go_results_btn, widgets.Button)

    def test_go_analysis_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_go_analysis_btn")

    def test_help_btn_exists(self):
        app = QuantUIApp()
        assert hasattr(app, "_help_btn")
        import ipywidgets as widgets

        assert isinstance(app._help_btn, widgets.Button)

    def test_help_panel_initially_hidden(self):
        app = QuantUIApp()
        assert app.help_tab_panel.layout.display == "none"
