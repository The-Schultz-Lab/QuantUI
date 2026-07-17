"""
Tests for quantui.app.QuantUIApp — FR-012 Phase 4.

All tests instantiate QuantUIApp() without calling .display(), which is safe
on any platform (display() requires an active IPython kernel; construction does
not).  PySCF is unavailable on Windows; calculations are skipped accordingly.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
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

    def test_seven_tabs(self):
        # POLISH.8 (M-POLISH, 2026-05-25): Log moved into the History
        # tab as a sub-accordion → 8 root tabs → 7.
        app = QuantUIApp()
        assert len(app.root_tab.children) == 7

    def test_tab_titles(self):
        app = QuantUIApp()
        expected = [
            "Calculate",
            "Results",
            "Analysis",
            "History",
            "Compare",
            # POLISH.8 (M-POLISH, 2026-05-25): Log tab moved into the
            # History tab as a sub-accordion; Files + System Settings
            # renumber to indices 5 and 6.
            "Files",
            # POLISH.4 (M-POLISH, 2026-05-25): "Status" → "System Settings".
            "System Settings",
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

    def test_close_is_noop_and_does_not_raise(self):
        """Regression (found via the L6 audit fix's Python 3.9 CI matrix):
        ase.utils.IOContext.openfile() — used by BFGS(..., logfile=...) in
        optimizer.py / pes_scan.py — checks hasattr(file, "close") to decide
        whether *file* is an already-open stream it should leave alone vs. a
        path string it should open() itself. ase==3.26.0 (the newest version
        pip resolves for Python 3.9) enforces this strictly and raised
        TypeError for a _LogCapture instance, which had no close() method;
        ase==3.29.0 (resolved for 3.10/3.11) happened to tolerate it via a
        later refactor, masking the gap until 3.9 was added to CI.
        """
        cap, _ = self._make_capture()
        cap.close()  # Must not raise
        assert hasattr(cap, "close")

    def test_satisfies_ase_openfile_already_open_contract(self):
        """Directly exercises the exact duck-typing check ASE performs."""
        cap, _ = self._make_capture()
        assert hasattr(cap, "close"), (
            "ase.utils.IOContext.openfile() treats any object without a "
            "'close' attribute as a path to open() itself, which fails for "
            "a non-path file-like object like _LogCapture"
        )


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

    def test_xyz_filename_sanitizes_basis_with_asterisk(self, tmp_path):
        """M11 audit fix (2026-07-14): a basis like "6-31G*" embedded
        verbatim in a filename is invalid on Windows ("*" is a reserved
        character there) and glob-hostile on POSIX. The exported filename
        must not contain "*".
        """
        app = QuantUIApp()
        app._set_molecule(_water())
        app._last_result_dir = tmp_path
        app.basis_dd.value = "6-31G*"

        app._on_export_xyz(None)

        xyz_files = list(tmp_path.glob("*.xyz"))
        assert len(xyz_files) == 1
        assert "*" not in xyz_files[0].name
        assert "Error" not in app.struct_export_status.value


class TestExportScriptCallback:
    """_on_export (standalone PySCF script) sanitizes its filename too."""

    def test_script_filename_sanitizes_basis_with_asterisk(self, tmp_path, monkeypatch):
        """M11 audit fix (2026-07-14): same filename-sanitization bug as
        the XYZ export, for the "Export Script" button — the script is
        written to a bare relative filename in the current directory.
        """
        monkeypatch.chdir(tmp_path)
        app = QuantUIApp()
        app._set_molecule(_water())
        app.basis_dd.value = "6-31G*"

        app._on_export(None)

        py_files = list(tmp_path.glob("*.py"))
        assert len(py_files) == 1
        assert "*" not in py_files[0].name
        assert "Error" not in app.export_status.value


class TestUpdateNotesBoldRendering:
    """_update_notes converts every **bold** span, not just the first.

    Regression (M12 audit fix, 2026-07-14): the old implementation was
    ``notes.replace("**", "<b>", 1).replace("**", "</b>", 1)`` — string
    .replace(..., 1) only touches the FIRST occurrence in the whole
    string, so only the first "**bold**" pair converted; every later one
    (get_educational_notes() typically returns 2-3 separate
    "**Label**: description" paragraphs) kept its literal "**" markers
    and leaked into the rendered panel instead of rendering bold.
    """

    def test_multiple_bold_spans_all_converted(self, monkeypatch):
        # UHF + 6-31G* + multiplicity 2 -> 3 separate "**Label**" spans
        # in get_educational_notes()'s output.
        #
        # `with app.notes_output: display(HTML(...))` only populates the
        # widget's `.outputs` under a live IPython display hook, which
        # isn't present in a plain pytest process — so this test captures
        # what's passed to `display()` directly instead of inspecting
        # `notes_output.outputs` (the pattern used by tests of the
        # newer `_set_html_output` atomic-swap helper, which manipulates
        # `.outputs` directly and doesn't have this limitation).
        import quantui.app_runflow as app_runflow

        captured: list = []
        monkeypatch.setattr(app_runflow, "display", lambda obj: captured.append(obj))

        app = QuantUIApp()
        mol = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], multiplicity=2)
        app._set_molecule(mol)
        app.method_dd.value = "UHF"
        app.basis_dd.value = "6-31G*"

        captured.clear()  # drop any renders triggered by the value changes above
        app._update_notes()

        assert len(captured) == 1
        html = captured[0].data
        assert "**" not in html, f"literal ** markers leaked into rendered HTML: {html}"
        assert html.count("<b>") >= 2
        assert html.count("<b>") == html.count("</b>")


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

    def test_calc_type_dd_has_expected_options(self):
        app = QuantUIApp()
        assert len(app.calc_type_dd.options) == 7
        assert "Reorganization Energy" in app.calc_type_dd.options

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


class TestReorganizationEnergyUI:
    """UI wiring for the Reorganization Energy calc-type + auto-setup button."""

    def test_reorg_mode_shows_channel_selector(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "Reorganization Energy"
        kids = app.calc_extra_opts.children
        assert app._reorg_mode_dd in kids
        assert app._reorg_note in kids
        assert app._reorg_mode_dd.value == "both"

    def test_reorg_hides_preopt_checkbox(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "Reorganization Energy"
        assert app._freq_preopt_cb.layout.display == "none"

    def test_auto_button_enabled_after_molecule_load(self):
        app = QuantUIApp()
        assert app._reorg_auto_btn.disabled is True
        mol = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        app._set_molecule(mol)
        assert app._reorg_auto_btn.disabled is False

    def test_auto_button_sets_up_mode(self):
        app = QuantUIApp()
        mol = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        app._set_molecule(mol)
        # Drive only the setup portion (not the background run thread) by
        # replicating what the handler does before dispatch.
        app.calc_type_dd.value = "Reorganization Energy"
        app._reorg_mode_dd.value = "both"
        assert app.calc_type_dd.value == "Reorganization Energy"
        assert app._reorg_mode_dd in app.calc_extra_opts.children


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


class TestFreqSeedDropdownFilter:
    """The Freq seed-geometry dropdown should only list prior geo-opts of
    the currently-active molecule.

    Rationale: users selecting "Seed geometry" on the Frequency tab want a
    geometry compatible with their current molecule. Listing a CH₄ geo-opt
    while the user is working on H₂O is misleading and risks an accidental
    geometry replacement. Filter is by formula (cheap and good enough for
    the common case); strict atom-list match is queued under
    M-HISTORY-HARDENING for later.
    """

    def _make_geo_opt_dir(self, root, formula, method="RHF", basis="STO-3G", offset=0):
        # Offset the timestamp microseconds so directories sort
        # deterministically when multiple fixtures share the same second.
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + f"{offset:06d}"
        d = root / f"{ts}_{formula}_{method}_{basis}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "geometry_opt",
                    "formula": formula,
                    "method": method,
                    "basis": basis,
                }
            )
        )
        (d / "trajectory.json").write_text("[]")
        return d

    def _water(self):
        return Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        )

    def test_unfiltered_when_no_molecule_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        self._make_geo_opt_dir(tmp_path, "CH4", offset=2)
        app = QuantUIApp()
        assert app._molecule is None
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert any("H2O" in lbl for lbl in labels)
        assert any("CH4" in lbl for lbl in labels)

    def test_filtered_to_current_molecule_formula(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        self._make_geo_opt_dir(tmp_path, "CH4", offset=2)
        app = QuantUIApp()
        app._molecule = self._water()
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert labels[0] == "(use current molecule)"
        assert any("H2O" in lbl for lbl in labels)
        assert not any("CH4" in lbl for lbl in labels)

    def test_set_molecule_triggers_filter(self, tmp_path, monkeypatch):
        # Loading a new molecule should auto-refresh the dropdown so stale
        # cross-molecule options drop out without the user clicking refresh.
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        self._make_geo_opt_dir(tmp_path, "CH4", offset=2)
        app = QuantUIApp()
        app._set_molecule(self._water(), label="test")
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert any("H2O" in lbl for lbl in labels)
        assert not any("CH4" in lbl for lbl in labels)


class TestPopIsosurfaceBug8:
    """Regression tests for BUG.8: ``_pop_isosurface`` raised AttributeError
    on single-point history replay when ``orbitals.npz`` was missing.

    Root cause: ``_last_orb_mo_coeff`` (and siblings) were only assigned by
    ``show_orbital_diagram`` during a successful Energies-panel populate.
    When that path returned early (no orbitals file or missing fields), the
    attributes never existed, and the immediately-following Isosurface
    populator's direct ``app._last_orb_mo_coeff is not None`` read blew up.

    Fix: initialize the attributes in ``__init__`` so they always exist,
    reset them at the start of ``apply_analysis_context`` so stale state
    can't leak between contexts, and use defensive ``getattr`` in the
    populator as belt-and-suspenders.
    """

    def test_orb_state_initialized_on_fresh_app(self):
        app = QuantUIApp()
        # All three attributes must exist (initialized to None) so the
        # populator can read them safely.
        assert app._last_orb_mo_coeff is None
        assert app._last_orb_mol_atom is None
        assert app._last_orb_mol_basis is None
        assert app._last_orb_info is None

    def test_pop_isosurface_on_fresh_app_returns_false_without_error(self):
        # The exact crash scenario from the user's 2026-05-20 event log:
        # a fresh QuantUIApp where no orbital data has been loaded yet
        # should NOT raise; it should report the panel as unavailable.
        from quantui.app_analysis import pop_isosurface

        app = QuantUIApp()
        ctx = _AnalysisContext(
            calc_type="single_point",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
        )
        result = pop_isosurface(app, ctx)
        assert result is False

    def test_apply_analysis_context_resets_orbital_state(self, tmp_path, monkeypatch):
        # After running an SP that populated orbital state, replaying a
        # different result (no orbitals.npz on disk) must NOT leak the
        # previous calc's orbital arrays into the Isosurface panel.
        from quantui.app_analysis import apply_analysis_context

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        # Simulate a prior live SP having populated orbital state.
        app._last_orb_mo_coeff = [[1.0, 0.0], [0.0, 1.0]]
        app._last_orb_mol_atom = [["H", [0.0, 0.0, 0.0]]]
        app._last_orb_mol_basis = "sto-3g"
        app._last_orb_info = MagicMock()

        # Now replay a context with no result_dir and no live_result —
        # i.e. nothing to repopulate orbital state from.
        ctx = _AnalysisContext(
            calc_type="single_point",
            formula="CH4",
            method="RHF",
            basis="STO-3G",
            result_dir=None,
            live_result=None,
        )
        apply_analysis_context(app, ctx)

        # State must have been wiped — stale H2O orbitals must not survive
        # into the CH4 context.
        assert app._last_orb_mo_coeff is None
        assert app._last_orb_mol_atom is None
        assert app._last_orb_mol_basis is None
        assert app._last_orb_info is None


class TestTDDFTSeedDropdown:
    """The UV-Vis (TD-DFT) Calculate-tab tab exposes a seed-geometry dropdown
    that mirrors the Frequency tab's behaviour (BUG.5).

    Acceptance:
    - The dropdown widget exists with the placeholder option.
    - On_calc_type_changed places the dropdown into ``calc_extra_opts``
      when UV-Vis (TD-DFT) is selected, but not for other calc types.
    - Like the Frequency seed dropdown, options are filtered to saved
      ``geometry_opt`` results whose formula matches the active molecule.
    - Picking a seed disables the QM pre-opt checkbox (seed = already
      optimised) and surfaces the green confirmation note.
    - ``_set_molecule`` auto-refreshes both seed dropdowns.
    """

    def _make_geo_opt_dir(self, root, formula, method="RHF", basis="STO-3G", offset=0):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + f"{offset:06d}"
        d = root / f"{ts}_{formula}_{method}_{basis}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "geometry_opt",
                    "formula": formula,
                    "method": method,
                    "basis": basis,
                }
            )
        )
        (d / "trajectory.json").write_text("[]")
        return d

    def _water(self):
        return Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        )

    def test_seed_widgets_exist(self):
        app = QuantUIApp()
        assert isinstance(app._tddft_seed_dd, widgets.Dropdown)
        assert isinstance(app._tddft_seed_refresh_btn, widgets.Button)
        assert isinstance(app._tddft_seed_note, widgets.HTML)
        # Initial placeholder option is present.
        labels = [lbl for lbl, _ in app._tddft_seed_dd.options]
        assert labels[0] == "(use current molecule)"

    def test_calc_type_uvvis_shows_seed_dropdown(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "UV-Vis (TD-DFT)"
        # The seed dropdown should now be one of the calc_extra_opts children.
        descendants = list(app.calc_extra_opts.children)
        # The seed dropdown is wrapped in an HBox with the refresh button.
        found = False
        for child in descendants:
            if isinstance(child, widgets.HBox):
                for sub in child.children:
                    if sub is app._tddft_seed_dd:
                        found = True
                        break
        assert found, "UV-Vis tab should include the seed-geometry dropdown"

    def test_calc_type_single_point_does_not_show_seed_dropdown(self):
        app = QuantUIApp()
        app.calc_type_dd.value = "Single Point"
        descendants = list(app.calc_extra_opts.children)
        for child in descendants:
            if isinstance(child, widgets.HBox):
                for sub in child.children:
                    assert sub is not app._tddft_seed_dd

    def test_seed_options_filtered_by_formula(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        self._make_geo_opt_dir(tmp_path, "CH4", offset=2)
        app = QuantUIApp()
        app._molecule = self._water()
        app._refresh_tddft_seed_options()
        labels = [lbl for lbl, _ in app._tddft_seed_dd.options]
        assert any("H2O" in lbl for lbl in labels)
        assert not any("CH4" in lbl for lbl in labels)

    def test_set_molecule_triggers_tddft_seed_filter(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        self._make_geo_opt_dir(tmp_path, "CH4", offset=2)
        app = QuantUIApp()
        app._set_molecule(self._water(), label="test")
        labels = [lbl for lbl, _ in app._tddft_seed_dd.options]
        assert any("H2O" in lbl for lbl in labels)
        assert not any("CH4" in lbl for lbl in labels)

    def test_picking_seed_disables_preopt_and_shows_note(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        seed_dir = self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        app = QuantUIApp()
        app._molecule = self._water()
        app._refresh_tddft_seed_options()
        # Pre-condition: pre-opt checkbox is enabled and toggled on.
        app._freq_preopt_cb.disabled = False
        app._freq_preopt_cb.value = True
        # Pick the seed.
        app._tddft_seed_dd.value = str(seed_dir)
        # Post-condition: pre-opt is disabled and value cleared; note set.
        assert app._freq_preopt_cb.disabled is True
        assert app._freq_preopt_cb.value is False
        assert "✓" in app._tddft_seed_note.value

    def test_clearing_seed_re_enables_preopt_and_clears_note(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        seed_dir = self._make_geo_opt_dir(tmp_path, "H2O", offset=1)
        app = QuantUIApp()
        app._molecule = self._water()
        app._refresh_tddft_seed_options()
        app._tddft_seed_dd.value = str(seed_dir)
        # Now clear the seed back to the placeholder.
        app._tddft_seed_dd.value = ""
        assert app._freq_preopt_cb.disabled is False
        assert app._tddft_seed_note.value == ""


class TestVibExportAnimation:
    """The Vibrational accordion exposes an export-to-HTML button that
    writes the current mode as a self-contained animation file.

    Backend resolution is decoupled from the user's default backend
    preference: plotlymol3d (preferred for export quality) with a py3Dmol
    fallback. This separation is enforced inside ``build_vib_export_html``
    so a user whose default render backend is py3Dmol can still get the
    higher-quality plotlymol animation when exporting.
    """

    def _water(self):
        return Molecule(
            atoms=["O", "H", "H"],
            coordinates=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        )

    def _seed_vib_state(self, app):
        """Populate the minimal state the export handler depends on.

        Mirrors what ``_render_vib_mode_py3dmol`` reads but does not exercise
        the live-render path — keeps the test focused on the export surface.
        """
        from types import SimpleNamespace

        mol = self._water()
        freq_stub = SimpleNamespace(
            frequencies_cm1=[100.0, 200.0, 300.0],
            ir_intensities=[1.0, 1.0, 1.0],
            displacements=[
                [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]],
            ],
        )
        app._last_vib_freq_result = freq_stub
        app._last_vib_molecule = mol
        app._last_vib_data = None  # forces the py3dmol fallback in this test
        app.vib_mode_dd.options = [
            ("Mode 1: 100.0 cm⁻¹", 1),
            ("Mode 2: 200.0 cm⁻¹", 2),
            ("Mode 3: 300.0 cm⁻¹", 3),
        ]
        app.vib_mode_dd.value = 1

    def test_export_button_and_status_exist(self):
        app = QuantUIApp()
        assert hasattr(app, "_vib_export_btn")
        assert isinstance(app._vib_export_btn, widgets.Button)
        assert hasattr(app, "_vib_export_status")
        assert isinstance(app._vib_export_status, widgets.HTML)
        assert app._vib_export_status.value == ""

    def test_export_bad_mode_index_chains_original_exception(self):
        """L audit fix (ruff B904): build_vib_export_html's py3Dmol-fallback
        path must chain the original IndexError via `raise ... from exc`
        when displacements[mode_number - 1] is out of range, not swallow it.
        """
        from types import SimpleNamespace

        from quantui.app_visualization import build_vib_export_html
        from quantui.viz_backend_router import BackendAvailability

        if not BackendAvailability.from_environment().py3dmol:
            pytest.skip("py3Dmol not available for export fallback test")

        freq_stub = SimpleNamespace(displacements=[[[0.1, 0.0, 0.0]]])
        app_stub = SimpleNamespace(
            _last_vib_freq_result=freq_stub,
            _last_vib_molecule=self._water(),
            _viz_availability=BackendAvailability(py3dmol=True, plotlymol=False),
        )
        with pytest.raises(ValueError) as exc_info:
            build_vib_export_html(app_stub, mode_number=5)  # out of range
        assert isinstance(exc_info.value.__cause__, IndexError)

    def test_export_without_vib_state_shows_error_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        # No _last_vib_freq_result / _last_vib_molecule yet.
        app._on_vib_export_animation(None)
        assert "color:#b91c1c" in app._vib_export_status.value
        assert "No vibrational mode loaded" in app._vib_export_status.value

    def test_export_writes_html_and_reports_backend(self, tmp_path, monkeypatch):
        from quantui.viz_backend_router import BackendAvailability

        if not BackendAvailability.from_environment().py3dmol:
            pytest.skip("py3Dmol not available for export fallback test")

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        self._seed_vib_state(app)
        # Force the py3Dmol fallback regardless of plotlymol installation —
        # the goal here is to assert the fallback writes a real HTML file.
        app._viz_availability = BackendAvailability(py3dmol=True, plotlymol=False)
        app._last_result_dir = tmp_path

        app._on_vib_export_animation(None)

        assert "color:#16a34a" in app._vib_export_status.value
        assert "Saved (py3dmol)" in app._vib_export_status.value
        # Find the file the handler wrote.
        files = list(tmp_path.glob("vib_*_mode1_*.html"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        # py3Dmol HTML includes a 3dmoljs viewer block.
        assert "viewer" in content.lower() or "3dmol" in content.lower()

    def test_export_no_backend_available_surfaces_error(self, tmp_path, monkeypatch):
        from quantui.viz_backend_router import BackendAvailability

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        self._seed_vib_state(app)
        app._viz_availability = BackendAvailability(py3dmol=False, plotlymol=False)

        app._on_vib_export_animation(None)
        assert "color:#b91c1c" in app._vib_export_status.value
        assert "No visualization backend available" in app._vib_export_status.value


class TestHistoryHardeningHist2:
    """HIST.2: every history-load operation emits a single
    ``history_load_timing`` event capturing total elapsed_ms + per-stage
    breakdown.

    Acceptance:
    - ``_LoadTimer.stage`` records elapsed_ms for each named sub-stage.
    - ``_LoadTimer.emit`` calls ``calc_log.log_event`` with event_type
      ``history_load_timing``, the total_ms, the op name, and per-stage
      ``<name>_ms`` keys.
    - ``history_load_analysis`` emits exactly one timing event per call
      with all expected stages.
    - ``status="error"`` is reported when the loader raises mid-load.
    - Telemetry failures (e.g. log_event itself raising) must NOT block
      the load — they're swallowed inside ``emit``.
    """

    def _make_sp_result_dir(self, tmp_path):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + "000001"
        d = tmp_path / f"{ts}_H2O_RHF_STO-3G"
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "single_point",
                    "formula": "H2O",
                    "method": "RHF",
                    "basis": "STO-3G",
                    "energy_hartree": -75.0,
                    "energy_ev": -2041.0,
                    "homo_lumo_gap_ev": 8.0,
                    "converged": True,
                    "n_iterations": 10,
                }
            )
        )
        return d

    def test_load_timer_stage_records_elapsed_ms(self):
        from quantui.app_history import _LoadTimer

        timer = _LoadTimer("test_op", Path("/tmp/dummy"))
        with timer.stage("phase_a"):
            pass  # near-zero elapsed
        with timer.stage("phase_b"):
            pass
        assert "phase_a" in timer._stages
        assert "phase_b" in timer._stages
        assert timer._stages["phase_a"] >= 0.0
        assert timer._stages["phase_b"] >= 0.0

    def test_load_timer_emit_logs_event_with_stage_breakdown(self):
        from quantui.app_history import _LoadTimer

        timer = _LoadTimer("test_op", Path("/tmp/dummy"))
        with timer.stage("foo"):
            pass
        with patch("quantui.calc_log.log_event") as mock_log:
            timer.emit(status="ok")
        mock_log.assert_called_once()
        event_type, _message = mock_log.call_args.args[:2]
        kwargs = mock_log.call_args.kwargs
        assert event_type == "history_load_timing"
        assert kwargs["op"] == "test_op"
        assert kwargs["status"] == "ok"
        assert kwargs["total_ms"] >= 0.0
        assert "foo_ms" in kwargs

    def test_load_timer_emit_swallows_log_event_failures(self):
        # If log_event raises (e.g. disk full), the timer's emit MUST NOT
        # propagate the exception — telemetry must never block the load.
        from quantui.app_history import _LoadTimer

        timer = _LoadTimer("test_op", Path("/tmp/dummy"))
        with patch("quantui.calc_log.log_event", side_effect=RuntimeError("disk full")):
            timer.emit(status="ok")  # must not raise

    def test_history_load_analysis_emits_one_timing_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = self._make_sp_result_dir(tmp_path)
        app = QuantUIApp()
        with (
            patch("quantui.calc_log.log_event") as mock_log,
            patch.object(app, "_activity_pulse"),
        ):
            app._history_load_analysis(result_dir)

        # Find the history_load_timing event (mock_log captures many other
        # events too — e.g. _refresh_file_browser may log nothing, but other
        # observers do).
        timing_calls = [
            call
            for call in mock_log.call_args_list
            if call.args and call.args[0] == "history_load_timing"
        ]
        assert len(timing_calls) == 1, (
            f"Expected exactly one history_load_timing event, got "
            f"{len(timing_calls)}"
        )
        kwargs = timing_calls[0].kwargs
        assert kwargs["op"] == "history_load_analysis"
        assert kwargs["status"] == "ok"
        assert kwargs["total_ms"] >= 0.0
        # All five expected stages must appear.
        expected_stages = {
            "read_pyscf_log_ms",
            "update_log_panel_ms",
            "build_context_ms",
            "mol_reconstruction_ms",
            "show_result_3d_ms",
            "apply_analysis_context_ms",
            "nav_tab_ms",
        }
        actual_stages = set(kwargs.keys()) & expected_stages
        assert actual_stages == expected_stages, (
            f"Missing stages: {expected_stages - actual_stages}; "
            f"unexpected stages: {actual_stages - expected_stages}"
        )

    def test_history_load_analysis_reports_error_status_on_raise(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = self._make_sp_result_dir(tmp_path)
        app = QuantUIApp()
        with (
            patch("quantui.calc_log.log_event") as mock_log,
            patch.object(
                app,
                "_apply_analysis_context",
                side_effect=RuntimeError("simulated"),
            ),
            patch.object(app, "_activity_pulse"),
        ):
            try:
                app._history_load_analysis(result_dir)
            except RuntimeError:
                pass

        timing_calls = [
            call
            for call in mock_log.call_args_list
            if call.args and call.args[0] == "history_load_timing"
        ]
        assert len(timing_calls) == 1
        assert timing_calls[0].kwargs["status"] == "error"


class TestHistoryHardeningHist6:
    """HIST.6: strict atom-list + coordinate match for the seed-geometry
    dropdown filter, replacing the formula-only filter shipped in session 54.

    Acceptance:
    - Two same-formula candidates with DIFFERENT starting geometries
      (different isomers / conformers) are correctly excluded from each
      other's seed dropdown when the active molecule matches only one of
      them by coordinates.
    - Two same-formula candidates with starting geometries within the RMSD
      tolerance of the active molecule's coordinates BOTH appear.
    - Malformed or missing ``trajectory.json`` falls through to a formula-
      only match (don't punish the user for a corrupt history entry).
    - ``_load_starting_geometry`` caches per-result results so repeated
      dropdown refreshes don't re-parse the same JSON files.
    """

    def _make_geo_opt_dir_with_trajectory(
        self,
        root,
        formula,
        atoms,
        starting_coords,
        offset=0,
        method="RHF",
        basis="STO-3G",
    ):
        from pathlib import Path

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + f"{offset:06d}"
        d = Path(root) / f"{ts}_{formula}_{method}_{basis}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "geometry_opt",
                    "formula": formula,
                    "method": method,
                    "basis": basis,
                }
            )
        )
        (d / "trajectory.json").write_text(
            json.dumps(
                {
                    "atoms": atoms,
                    "charge": 0,
                    "multiplicity": 1,
                    "steps": [
                        {
                            "coords": [
                                list(map(float, row)) for row in starting_coords
                            ],
                            "energy": -75.0,
                        }
                    ],
                }
            )
        )
        return d

    def _water_coords(self, displacement=0.0):
        # Returns water coords; ``displacement`` lets us produce a second
        # water at a controllable RMSD distance from the canonical one.
        return [
            [0.0 + displacement, 0.0, 0.0],
            [0.96 + displacement, 0.0, 0.0],
            [-0.24 + displacement, 0.93, 0.0],
        ]

    def _water_molecule(self):
        return Molecule(atoms=["O", "H", "H"], coordinates=self._water_coords(0.0))

    def setup_method(self, _method):
        # Tests share a module-level cache (_SEED_GEOMETRY_CACHE) for
        # geometry parses; clear it before each test for determinism.
        from quantui.app_runflow import _SEED_GEOMETRY_CACHE

        _SEED_GEOMETRY_CACHE.clear()

    def test_same_formula_different_geometry_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        # Active molecule = water at canonical coords.
        # Saved A: same coords → matches.
        # Saved B: coords shifted by 2 Å → RMSD ≈ 2 Å ≫ 0.1 Å → excluded.
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(0.0), offset=1
        )
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(2.0), offset=2
        )
        app = QuantUIApp()
        app._molecule = self._water_molecule()
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert len(labels) == 2, labels
        assert labels[0] == "(use current molecule)"
        assert "H2O" in labels[1]

    def test_same_formula_within_tolerance_included(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        # Two candidates, both within 0.1 Å RMSD of the active mol.
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(0.0), offset=1
        )
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(0.02), offset=2
        )
        app = QuantUIApp()
        app._molecule = self._water_molecule()
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert len(labels) == 3, labels
        assert sum(1 for lbl in labels if "H2O" in lbl) == 2

    def test_atom_order_mismatch_excluded(self, tmp_path, monkeypatch):
        # Strict atom-order policy: ["H","O","H"] is not the same as
        # ["O","H","H"] even though the formula matches.
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(0.0), offset=1
        )
        self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["H", "O", "H"], self._water_coords(0.0), offset=2
        )
        app = QuantUIApp()
        app._molecule = self._water_molecule()
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert len(labels) == 2
        assert "H2O" in labels[1]

    def test_malformed_trajectory_falls_back_to_formula_match(
        self, tmp_path, monkeypatch
    ):
        # Malformed trajectory.json must NOT crash — and must fall through
        # to formula-only match so the candidate still appears.
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + "000001"
        d = tmp_path / f"{ts}_H2O_RHF_STO-3G"
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "geometry_opt",
                    "formula": "H2O",
                    "method": "RHF",
                    "basis": "STO-3G",
                }
            )
        )
        (d / "trajectory.json").write_text("[]")  # malformed (list, not dict)
        app = QuantUIApp()
        app._molecule = self._water_molecule()
        app._refresh_freq_seed_options()
        labels = [lbl for lbl, _ in app._freq_seed_dd.options]
        assert any("H2O" in lbl for lbl in labels)

    def test_starting_geometry_cache_hit_avoids_reread(self, tmp_path):
        # _load_starting_geometry must cache per-result so back-to-back
        # refreshes (e.g. when both Freq and UV-Vis dropdowns refresh from
        # the same _set_molecule call) don't re-parse the JSON.
        from quantui.app_runflow import (
            _SEED_GEOMETRY_CACHE,
            _load_starting_geometry,
        )

        _SEED_GEOMETRY_CACHE.clear()
        d = self._make_geo_opt_dir_with_trajectory(
            tmp_path, "H2O", ["O", "H", "H"], self._water_coords(0.0), offset=1
        )
        first = _load_starting_geometry(d)
        assert first is not None
        # Second call must return the cached object without touching disk.
        with patch("pathlib.Path.read_text") as mock_read:
            second = _load_starting_geometry(d)
        assert second is first
        mock_read.assert_not_called()


class TestMExportCopyPlotData:
    """M-EXPORT / EXPORT.4: every spectrum / diagram panel offers a
    "Copy data" button that exports the plot's (x, y) data to CSV and
    attempts a clipboard copy via the browser's clipboard API.

    Acceptance:
    - ``_fig_to_csv`` extracts per-trace (x, y) data from a Plotly figure
      in the documented CSV layout; empty figure → empty string (caller
      treats as "nothing to copy" rather than writing an empty file).
    - Each plot panel (IR, UV-Vis, orbital, PES) exposes a
      ``_*_copy_data_btn`` widget.
    - The handler writes a CSV file to the active result directory and
      updates the panel's status widget.
    - The status reports an error when no figure has been rendered yet.
    - Output CSV round-trips cleanly via stdlib ``csv.reader``.
    """

    def _make_simple_fig(self):
        import plotly.graph_objects as go

        return go.Figure(
            go.Scatter(x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], name="trace0")
        )

    def _make_two_trace_fig(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[100, 200], y=[5, 8], name="Stick"))
        fig.add_trace(go.Scatter(x=[100, 150, 200], y=[1, 4, 8], name="Broadened"))
        return fig

    def test_fig_to_csv_returns_empty_string_for_none(self):
        assert QuantUIApp._fig_to_csv(None) == ""

    def test_fig_to_csv_returns_empty_string_when_no_traces(self):
        import plotly.graph_objects as go

        fig = go.Figure()  # no data
        assert QuantUIApp._fig_to_csv(fig) == ""

    def test_fig_to_csv_extracts_single_trace(self):
        fig = self._make_simple_fig()
        csv_text = QuantUIApp._fig_to_csv(fig, title="Test Plot")
        assert "# Test Plot" in csv_text
        assert "# trace0" in csv_text
        assert "x,y" in csv_text
        assert "1.0,10.0" in csv_text
        assert "3.0,30.0" in csv_text

    def test_fig_to_csv_extracts_multi_trace_with_separator_sections(self):
        fig = self._make_two_trace_fig()
        csv_text = QuantUIApp._fig_to_csv(fig)
        assert "# Stick" in csv_text
        assert "# Broadened" in csv_text
        # Each section gets its own "x,y" header — the layout is
        # repeated, not merged into one wide table.
        assert csv_text.count("x,y") == 2

    def test_fig_to_csv_output_round_trips_via_stdlib_csv(self):
        import csv as _csv
        import io as _io

        fig = self._make_simple_fig()
        text = QuantUIApp._fig_to_csv(fig, title="Roundtrip")
        # Strip the "# ..." comment lines, leaving the actual rows.
        lines = [
            line for line in text.splitlines() if line and not line.startswith("#")
        ]
        reader = _csv.reader(_io.StringIO("\n".join(lines)))
        rows = list(reader)
        assert rows[0] == ["x", "y"]
        assert rows[1:] == [
            ["1.0", "10.0"],
            ["2.0", "20.0"],
            ["3.0", "30.0"],
        ]

    def test_all_four_panels_expose_copy_data_button(self):
        app = QuantUIApp()
        for prefix in ("ir", "uv", "orb", "pes"):
            btn = getattr(app, f"_{prefix}_copy_data_btn", None)
            assert isinstance(btn, widgets.Button), f"missing _{prefix}_copy_data_btn"
            assert btn.description == "Copy data"

    def test_copy_data_with_no_figure_shows_error_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        app._last_ir_fig = None
        app._on_ir_copy_data(None)
        assert "color:#b91c1c" in app._ir_export_status.value
        assert "No plot data" in app._ir_export_status.value

    def test_copy_data_writes_csv_to_result_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_ir_fig = self._make_simple_fig()
        app._on_ir_copy_data(None)
        assert "color:#16a34a" in app._ir_export_status.value
        assert "Saved CSV" in app._ir_export_status.value
        csv_files = list(tmp_path.glob("ir_spectrum_data_*.csv"))
        assert len(csv_files) == 1
        content = csv_files[0].read_text(encoding="utf-8")
        assert "trace0" in content
        assert "1.0,10.0" in content

    def test_copy_data_handles_figure_with_no_extractable_traces(
        self, tmp_path, monkeypatch
    ):
        import plotly.graph_objects as go

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_ir_fig = go.Figure()  # empty
        app._on_ir_copy_data(None)
        assert "color:#b91c1c" in app._ir_export_status.value
        assert "no extractable" in app._ir_export_status.value.lower()


class TestHistoryHardeningHist1:
    """HIST.1: clicking View Results / View Analysis on a History selection
    must give the user immediate visual feedback.

    Acceptance:
    - ``_activity_count`` increments while the loader runs (toolbar
      indicator turns to "UI Active") and decrements back to 0 on completion.
    - Source buttons (View Results, View Analysis) are disabled at start of
      load and re-enabled at end — prevents double-click + signals "loading".
    - The feedback contract holds even if the load raises (try/finally).
    """

    def _make_sp_result_dir(self, tmp_path):
        """Create a minimal saved single-point result the loader can read."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + "000001"
        d = tmp_path / f"{ts}_H2O_RHF_STO-3G"
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": "single_point",
                    "formula": "H2O",
                    "method": "RHF",
                    "basis": "STO-3G",
                    "energy_hartree": -75.0,
                    "energy_ev": -2041.0,
                    "homo_lumo_gap_ev": 8.0,
                    "converged": True,
                    "n_iterations": 10,
                }
            )
        )
        return d

    def test_history_load_analysis_lights_activity_indicator(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = self._make_sp_result_dir(tmp_path)
        app = QuantUIApp()
        # The loader bumps _activity_count up by 1 inside its body and back
        # down on exit. Patch _apply_analysis_context to capture the live
        # count mid-load. Patch out the tab-switch pulse so its timer doesn't
        # race the assertion (the load ends by setting root_tab.selected_index
        # which fires _activity_pulse on a 160ms Timer thread — separate from
        # the loader's own begin/end pair we're verifying here).
        captured_count: list[int] = []
        original_apply = app._apply_analysis_context

        def _capture_count(ctx):
            captured_count.append(app._activity_count)
            return original_apply(ctx)

        with (
            patch.object(app, "_apply_analysis_context", side_effect=_capture_count),
            patch.object(app, "_activity_pulse"),
        ):
            app._history_load_analysis(result_dir)
        assert captured_count == [1]  # exactly one active op while loading
        assert app._activity_count == 0  # restored after completion

    def test_history_load_analysis_disables_source_buttons(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = self._make_sp_result_dir(tmp_path)
        app = QuantUIApp()
        btn_a = widgets.Button(description="View Results")
        btn_b = widgets.Button(description="View Analysis")
        # Both buttons start enabled.
        assert btn_a.disabled is False
        assert btn_b.disabled is False

        # Capture disabled state mid-load.
        captured: dict[str, bool] = {}
        original_apply = app._apply_analysis_context

        def _capture(ctx):
            captured["a"] = btn_a.disabled
            captured["b"] = btn_b.disabled
            return original_apply(ctx)

        with patch.object(app, "_apply_analysis_context", side_effect=_capture):
            app._history_load_analysis(result_dir, source_btns=(btn_a, btn_b))
        assert captured == {"a": True, "b": True}
        # Buttons restored after the load.
        assert btn_a.disabled is False
        assert btn_b.disabled is False

    def test_feedback_restored_even_on_exception(self, tmp_path, monkeypatch):
        # If the loader raises mid-load, the activity counter and buttons
        # must still be restored — try/finally contract.
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = self._make_sp_result_dir(tmp_path)
        app = QuantUIApp()
        btn = widgets.Button(description="View")

        with patch.object(
            app,
            "_apply_analysis_context",
            side_effect=RuntimeError("simulated failure"),
        ):
            try:
                app._history_load_analysis(result_dir, source_btns=(btn,))
            except RuntimeError:
                pass
        assert app._activity_count == 0
        assert btn.disabled is False


class TestHistoryHardeningHist5:
    """HIST.5: history dropdown labels must expose calc type before selection.

    The current ``refresh_results_browser`` formats each option as
    ``"<timestamp>  ·  [<calc-badge>]  <formula>  <method>/<basis>"``,
    where the badge is the friendly name from ``_calc_type_badge``. This
    test locks in that contract — particularly the bracketed badge — so
    a future refactor can't accidentally drop the calc-type prefix that the
    user originally reported missing in the M-PLOT user report.
    """

    def _make_result(self, tmp_path, formula, calc_type, offset):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + f"{offset:06d}"
        d = tmp_path / f"{ts}_{formula}_RHF_STO-3G"
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "timestamp": ts,
                    "calc_type": calc_type,
                    "formula": formula,
                    "method": "RHF",
                    "basis": "STO-3G",
                }
            )
        )
        # Geometry opt needs trajectory.json for the seed-dropdown side-path,
        # but refresh_results_browser doesn't gate on it.
        return d

    def test_dropdown_label_includes_calc_badge_for_each_type(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        self._make_result(tmp_path, "H2O", "single_point", offset=1)
        self._make_result(tmp_path, "H2O", "geometry_opt", offset=2)
        self._make_result(tmp_path, "H2O", "frequency", offset=3)
        self._make_result(tmp_path, "H2O", "tddft", offset=4)
        self._make_result(tmp_path, "H2O", "nmr", offset=5)
        self._make_result(tmp_path, "H2O", "pes_scan", offset=6)
        app = QuantUIApp()
        app._refresh_results_browser()
        labels = [lbl for lbl, _ in app.past_dd.options]
        # POLISH.6 (M-POLISH, 2026-05-25) prepends a
        # "(select a calculation to view)" placeholder so the dropdown
        # opens in an explicit no-selection state. Strip it before
        # asserting per-row badge contents.
        result_labels = [lbl for lbl in labels if "select a calculation" not in lbl]
        # Every result row must include a bracketed badge.
        assert all("[" in lbl and "]" in lbl for lbl in result_labels), result_labels
        joined = " ".join(result_labels)
        for expected in ("[SP]", "[GeoOpt]", "[Freq]", "[UV-Vis]", "[NMR]", "[PES]"):
            assert expected in joined, f"missing badge {expected} in {result_labels}"


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
        # Force the Plotly fallback path (M-ORBVIZ: the default routes to
        # py3Dmol when available). Backend is pinned via _resolve_backend so the
        # test is independent of which backends are installed.
        app._resolve_backend = lambda task: "plotlymol"

        captured: dict[str, object] = {}

        def _fake_generate(_atom, _basis, _coeff, _idx, out_path, **_kwargs):
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

    def test_render_orbital_isosurface_py3dmol_path(self, tmp_path):
        # When the backend resolves to py3Dmol, the renderer is the py3Dmol
        # cube path (not Plotly), and the cube is still saved to disk.
        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_orb_info = MagicMock()
        app._last_orb_info.n_occupied = 1
        app._last_orb_info.mo_energies_ev = [-10.0, 2.0]
        app._last_orb_info.formula = "H2O"
        app._last_orb_mo_coeff = [[1.0, 0.0], [0.0, 1.0]]
        app._last_orb_mol_atom = [["H", [0.0, 0.0, 0.0]]]
        app._last_orb_mol_basis = "sto-3g"
        app._resolve_backend = lambda task: "py3dmol"

        captured: dict[str, object] = {}

        def _fake_generate(_atom, _basis, _coeff, _idx, out_path, **_kwargs):
            captured["path"] = out_path
            out_path.write_text("cube", encoding="utf-8")
            return out_path

        with (
            patch(
                "quantui.orbital_visualization.generate_cube_from_arrays",
                side_effect=_fake_generate,
            ) as mock_gen,
            patch(
                "quantui.orbital_visualization.render_orbital_isosurface_py3dmol",
                return_value="<div>py3dmol iso</div>",
            ) as mock_py3dmol,
            patch(
                "quantui.orbital_visualization.plot_cube_isosurface",
                return_value=MagicMock(),
            ) as mock_plot,
        ):
            app._render_orbital_isosurface("HOMO")

        saved_path = captured.get("path")
        assert saved_path is not None and saved_path.exists()
        mock_gen.assert_called_once()
        mock_py3dmol.assert_called_once()
        mock_plot.assert_not_called()


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
