"""
QuantUI application class.

All widget creation, state management, callbacks, and tab wiring live here.
The notebook is a thin launcher::

    from quantui.app import QuantUIApp
    QuantUIApp().display()

CSS is injected inside ``display()`` — not on import — so importing this
module in tests or tutorials does not pollute the IPython display.
"""

from __future__ import annotations

import asyncio
import html as _html
import io
import re
import threading
import time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, List, Literal, Optional

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import HTML, Javascript, display

import quantui
import quantui.calc_log as _calc_log
import quantui.issue_tracker as _issue_tracker
from quantui import molecule_library as _ml
from quantui.app_analysis import (
    activate_ana_panel as _ana_activate_ana_panel,
)
from quantui.app_analysis import (
    apply_analysis_context as _ana_apply_analysis_context,
)
from quantui.app_analysis import (
    build_ana_switcher as _ana_build_ana_switcher,
)
from quantui.app_analysis import (
    deactivate_all_ana_panels as _ana_deactivate_all_ana_panels,
)
from quantui.app_analysis import (
    pop_energies as _ana_pop_energies,
)
from quantui.app_analysis import (
    pop_geo_trajectory as _ana_pop_geo_trajectory,
)
from quantui.app_analysis import (
    pop_ir_spectrum as _ana_pop_ir_spectrum,
)
from quantui.app_analysis import (
    pop_isosurface as _ana_pop_isosurface,
)
from quantui.app_analysis import (
    pop_nmr_shielding as _ana_pop_nmr_shielding,
)
from quantui.app_analysis import (
    pop_pes_plot as _ana_pop_pes_plot,
)
from quantui.app_analysis import (
    pop_pes_trajectory as _ana_pop_pes_trajectory,
)
from quantui.app_analysis import (
    pop_preopt_trajectory as _ana_pop_preopt_trajectory,
)
from quantui.app_analysis import (
    pop_uv_vis as _ana_pop_uv_vis,
)
from quantui.app_analysis import (
    pop_vibrational as _ana_pop_vibrational,
)
from quantui.app_analysis import (
    select_ana_panel as _ana_select_ana_panel,
)
from quantui.app_builders import (
    build_calc_setup as _bld_build_calc_setup,
)
from quantui.app_builders import (
    build_compare_section as _bld_build_compare_section,
)
from quantui.app_builders import (
    build_files_tab as _bld_build_files_tab,
)
from quantui.app_builders import (
    build_help_section as _bld_build_help_section,
)
from quantui.app_builders import (
    build_history_section as _bld_build_history_section,
)
from quantui.app_builders import (
    build_issue_widgets as _bld_build_issue_widgets,
)
from quantui.app_builders import (
    build_molecule_section as _bld_build_molecule_section,
)
from quantui.app_builders import (
    build_output_tab as _bld_build_output_tab,
)
from quantui.app_builders import (
    build_results_section as _bld_build_results_section,
)
from quantui.app_builders import (
    build_run_section as _bld_build_run_section,
)
from quantui.app_builders import (
    build_shared_widgets as _bld_build_shared_widgets,
)
from quantui.app_builders import (
    build_status_panel as _bld_build_status_panel,
)
from quantui.app_builders import (
    build_theme_selector as _bld_build_theme_selector,
)
from quantui.app_builders import (
    build_welcome_header as _bld_build_welcome_header,
)
from quantui.app_builders import (
    library_result_options as _bld_library_result_options,
)
from quantui.app_exports import (
    export_molecule_and_label as _exp_export_molecule_and_label,
)
from quantui.app_exports import (
    molecule_to_rdkit as _exp_molecule_to_rdkit,
)
from quantui.app_exports import (
    on_export as _exp_on_export,
)
from quantui.app_exports import (
    on_export_bundle as _exp_on_export_bundle,
)
from quantui.app_exports import (
    on_export_mol as _exp_on_export_mol,
)
from quantui.app_exports import (
    on_export_pdb as _exp_on_export_pdb,
)
from quantui.app_exports import (
    on_export_xyz as _exp_on_export_xyz,
)
from quantui.app_exports import (
    on_iso_export_cube as _exp_on_iso_export_cube,
)
from quantui.app_formatters import (
    format_freq_result as _fmt_freq_result,
)
from quantui.app_formatters import (
    format_nmr_result as _fmt_nmr_result,
)
from quantui.app_formatters import (
    format_opt_result as _fmt_opt_result,
)
from quantui.app_formatters import (
    format_past_result as _fmt_past_result,
)
from quantui.app_formatters import (
    format_pes_scan_result as _fmt_pes_scan_result,
)
from quantui.app_formatters import (
    format_result as _fmt_result,
)
from quantui.app_formatters import (
    format_tddft_result as _fmt_tddft_result,
)
from quantui.app_history import (
    build_history_context as _hist_build_history_context,
)
from quantui.app_history import (
    history_load_analysis as _hist_history_load_analysis,
)
from quantui.app_history import (
    history_load_results as _hist_history_load_results,
)
from quantui.app_history import (
    mol_from_result_dir as _hist_mol_from_result_dir,
)
from quantui.app_history import (
    on_past_dd_changed as _hist_on_past_dd_changed,
)
from quantui.app_history import (
    on_view_log as _hist_on_view_log,
)
from quantui.app_runflow import (
    do_calibration as _run_do_calibration,
)
from quantui.app_runflow import (
    on_accumulate as _run_on_accumulate,
)
from quantui.app_runflow import (
    on_basis_help as _run_on_basis_help,
)
from quantui.app_runflow import (
    on_cal_run as _run_on_cal_run,
)
from quantui.app_runflow import (
    on_cal_skip as _run_on_cal_skip,
)
from quantui.app_runflow import (
    on_cal_stop as _run_on_cal_stop,
)
from quantui.app_runflow import (
    on_calc_type_changed as _run_on_calc_type_changed,
)
from quantui.app_runflow import (
    on_clear as _run_on_clear,
)
from quantui.app_runflow import (
    on_clear_log as _run_on_clear_log,
)
from quantui.app_runflow import (
    on_clear_log_cache as _run_on_clear_log_cache,
)
from quantui.app_runflow import (
    on_clear_log_cache_confirm as _run_on_clear_log_cache_confirm,
)
from quantui.app_runflow import (
    on_compare as _run_on_compare,
)
from quantui.app_runflow import (
    on_compare_clear as _run_on_compare_clear,
)
from quantui.app_runflow import (
    on_compare_refresh as _run_on_compare_refresh,
)
from quantui.app_runflow import (
    on_confirm_no as _run_on_confirm_no,
)
from quantui.app_runflow import (
    on_confirm_yes as _run_on_confirm_yes,
)
from quantui.app_runflow import (
    on_copy_results_path as _run_on_copy_results_path,
)
from quantui.app_runflow import (
    on_exit_clicked as _run_on_exit_clicked,
)
from quantui.app_runflow import (
    on_expand_mol_input as _run_on_expand_mol_input,
)
from quantui.app_runflow import (
    on_freq_seed_changed as _run_on_freq_seed_changed,
)
from quantui.app_runflow import (
    on_help_toggle as _run_on_help_toggle,
)
from quantui.app_runflow import (
    on_help_topic_changed as _run_on_help_topic_changed,
)
from quantui.app_runflow import (
    on_issue_btn as _run_on_issue_btn,
)
from quantui.app_runflow import (
    on_issue_cancel as _run_on_issue_cancel,
)
from quantui.app_runflow import (
    on_issue_submit as _run_on_issue_submit,
)
from quantui.app_runflow import (
    on_log_clear as _run_on_log_clear,
)
from quantui.app_runflow import (
    on_method_help as _run_on_method_help,
)
from quantui.app_runflow import (
    on_past_refresh as _run_on_past_refresh,
)
from quantui.app_runflow import (
    on_reset_click as _run_on_reset_click,
)
from quantui.app_runflow import (
    on_run_clicked as _run_on_run_clicked,
)
from quantui.app_runflow import (
    on_solvent_cb_changed as _run_on_solvent_cb_changed,
)
from quantui.app_runflow import (
    on_tddft_seed_changed as _run_on_tddft_seed_changed,
)
from quantui.app_runflow import (
    populate_compare_list as _run_populate_compare_list,
)
from quantui.app_runflow import (
    refresh_comparison as _run_refresh_comparison,
)
from quantui.app_runflow import (
    refresh_freq_seed_options as _run_refresh_freq_seed_options,
)
from quantui.app_runflow import (
    refresh_results_browser as _run_refresh_results_browser,
)
from quantui.app_runflow import (
    refresh_tddft_seed_options as _run_refresh_tddft_seed_options,
)
from quantui.app_runflow import (
    update_estimate as _run_update_estimate,
)
from quantui.app_runflow import (
    update_notes as _run_update_notes,
)
from quantui.app_runflow import (
    update_scan_widgets as _run_update_scan_widgets,
)
from quantui.app_visualization import (
    build_vib_data_from_freq_result as _viz_build_vib_data_from_freq_result,
)
from quantui.app_visualization import (
    build_vib_data_inner as _viz_build_vib_data_inner,
)
from quantui.app_visualization import (
    build_vib_export_html as _viz_build_vib_export_html,
)
from quantui.app_visualization import (
    on_ir_fwhm_changed as _viz_on_ir_fwhm_changed,
)
from quantui.app_visualization import (
    on_ir_mode_changed as _viz_on_ir_mode_changed,
)
from quantui.app_visualization import (
    on_iso_generate as _viz_on_iso_generate,
)
from quantui.app_visualization import (
    on_orb_range_changed as _viz_on_orb_range_changed,
)
from quantui.app_visualization import (
    on_traj_expand as _viz_on_traj_expand,
)
from quantui.app_visualization import (
    on_uv_fwhm_changed as _viz_on_uv_fwhm_changed,
)
from quantui.app_visualization import (
    on_uv_mode_changed as _viz_on_uv_mode_changed,
)
from quantui.app_visualization import (
    on_vib_mode_changed as _viz_on_vib_mode_changed,
)
from quantui.app_visualization import (
    render_orbital_isosurface as _viz_render_orbital_isosurface,
)
from quantui.app_visualization import (
    render_traj_frame as _viz_render_traj_frame,
)
from quantui.app_visualization import (
    render_vib_mode as _viz_render_vib_mode,
)
from quantui.app_visualization import (
    show_ir_spectrum as _viz_show_ir_spectrum,
)
from quantui.app_visualization import (
    show_opt_trajectory as _viz_show_opt_trajectory,
)
from quantui.app_visualization import (
    show_orbital_diagram as _viz_show_orbital_diagram,
)
from quantui.app_visualization import (
    show_pes_scan_result as _viz_show_pes_scan_result,
)
from quantui.app_visualization import (
    show_result_3d as _viz_show_result_3d,
)
from quantui.app_visualization import (
    show_uv_vis_spectrum as _viz_show_uv_vis_spectrum,
)
from quantui.app_visualization import (
    show_vib_animation as _viz_show_vib_animation,
)
from quantui.app_visualization import (
    traj_step_html as _viz_traj_step_html,
)
from quantui.app_visualization import (
    update_ir_figure as _viz_update_ir_figure,
)
from quantui.app_visualization import (
    update_uv_vis_figure as _viz_update_uv_vis_figure,
)
from quantui.app_visualization import (
    wire_ir_controls as _viz_wire_ir_controls,
)
from quantui.app_visualization import (
    wire_uv_controls as _viz_wire_uv_controls,
)

# Import directly from submodules to avoid circular-import issues.
# quantui/__init__.py imports this module (app.py), so using
# `from quantui import X` at module load time would see a partially-
# initialised package namespace (symbols defined after the app import
# in __init__.py would not yet exist).
from quantui.config import (
    DEFAULT_BASIS,
    DEFAULT_CHARGE,
    DEFAULT_FMAX,
    DEFAULT_METHOD,
    DEFAULT_MULTIPLICITY,
    DEFAULT_OPT_STEPS,
    SUPPORTED_BASIS_SETS,
    SUPPORTED_METHODS,
)
from quantui.help_content import HELP_TOPICS
from quantui.molecule import Molecule, parse_xyz_input
from quantui.progress import StepProgress
from quantui.user_settings import UserSettings
from quantui.utils import get_session_resources
from quantui.viz_backend_router import (
    BackendAvailability,
    VizBackend,
    VizPreference,
    VizTask,
    select_backend,
)

# ── Availability flags (computed once at import, not per-instantiation) ───────
try:
    from quantui.ase_bridge import ASE_AVAILABLE
except ImportError:
    ASE_AVAILABLE = False

try:
    from quantui.visualization_py3dmol import (
        DEFAULT_LIGHTING as _DEFAULT_LIGHTING,
    )
    from quantui.visualization_py3dmol import (
        DEFAULT_STYLE as _DEFAULT_VIZ_STYLE,
    )
    from quantui.visualization_py3dmol import (
        LIGHTING_OPTIONS as _LIGHTING_OPTIONS,
    )
    from quantui.visualization_py3dmol import (
        PLOTLYMOL_AVAILABLE as _PLOTLYMOL_VIZ,
    )
    from quantui.visualization_py3dmol import (
        PY3DMOL_AVAILABLE as _PY3DMOL_VIZ,
    )
    from quantui.visualization_py3dmol import (
        VIZ_STYLE_OPTIONS as _VIZ_STYLE_OPTIONS,
    )
    from quantui.visualization_py3dmol import (
        display_molecule as _display_molecule,
    )
    from quantui.visualization_py3dmol import (
        render_molecule_html as _render_molecule_html,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    _display_molecule = None  # type: ignore[assignment]
    _render_molecule_html = None  # type: ignore[assignment]
    _PLOTLYMOL_VIZ = False
    _PY3DMOL_VIZ = False
    _DEFAULT_VIZ_STYLE = "ball+stick"
    _DEFAULT_LIGHTING = "soft"
    _VIZ_STYLE_OPTIONS = [
        ("Ball & Stick", "ball+stick"),
        ("Stick", "stick"),
        ("Sphere (VDW)", "sphere"),
        ("Line", "line"),
    ]
    _LIGHTING_OPTIONS = [
        ("Soft", "soft"),
        ("Default", "default"),
        ("Bright", "bright"),
        ("Metallic", "metallic"),
        ("Dramatic", "dramatic"),
    ]

_VizBackend = Literal["auto", "py3dmol", "plotlymol"]
_BOTH_VIZ_AVAILABLE: bool = _PLOTLYMOL_VIZ and _PY3DMOL_VIZ
_DEFAULT_VIZ_BACKEND: _VizBackend = "plotlymol" if _PLOTLYMOL_VIZ else "py3dmol"

try:
    from quantui.pubchem import (
        RDKIT_AVAILABLE as _PUBCHEM_RDKIT_AVAILABLE,
    )
    from quantui.structure_providers import (
        search_candidates as _struct_search_candidates,
    )
    from quantui.structure_providers import (
        student_friendly_resolve as _student_friendly_resolve,
    )

    PUBCHEM_AVAILABLE = _PUBCHEM_RDKIT_AVAILABLE
except ImportError:
    PUBCHEM_AVAILABLE = False
    _student_friendly_resolve = None  # type: ignore[assignment]
    _struct_search_candidates = None  # type: ignore[assignment]

try:
    from quantui.session_calc import SessionResult, run_in_session  # noqa: F401

    _PYSCF_AVAILABLE = True
except (ImportError, AttributeError):
    _PYSCF_AVAILABLE = False

try:
    from quantui.preopt import preoptimize

    _PREOPT_AVAILABLE = True
except (ImportError, AttributeError):
    _PREOPT_AVAILABLE = False

_RDKIT_AVAILABLE: bool = bool(PUBCHEM_AVAILABLE)

from quantui.benchmarks import (  # noqa: E402
    BENCHMARK_SUITE as _BENCHMARK_SUITE,
)
from quantui.benchmarks import (  # noqa: E402
    BENCHMARK_SUITE_LONG as _BENCHMARK_SUITE_LONG,
)
from quantui.benchmarks import (  # noqa: E402
    load_last_calibration as _load_last_calibration_raw,
)


def _load_last_calibration_label() -> str:
    """Return a human-readable timestamp of the last calibration, or ''."""
    data = _load_last_calibration_raw()
    if data is None:
        return ""
    ts = str(data.get("timestamp", ""))
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(ts).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return ts[:19] if ts else ""


# ── Module-level constants ────────────────────────────────────────────────────
_THEME_HUE: dict = {"Dark": 180}

_APP_CSS: str = """<style>
/* System font stack ---------------------------------------------------- */
body, p, span, li, td, th, label, input, select, textarea, blockquote,
.jp-OutputArea-output, .widget-html-content, .widget-label-basic,
.widget-label {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* App title (h1 in the markdown cell) ---------------------------------- */
h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: -0.01em !important;
    margin: 10px 0 4px !important;
    border-bottom: none !important;
}

/* Section headers ------------------------------------------------------- */
h3 {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin: 24px 0 10px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

/* Rounded corners on inputs, dropdowns, and buttons -------------------- */
.widget-text input, .widget-textarea textarea {
    border-color: #d1d5db !important;
    border-radius: 5px !important;
}
.widget-dropdown select { border-radius: 5px !important; }
.widget-button, .widget-toggle-button { border-radius: 5px !important; }

/* Suppress Jupyter stderr pink — invert+hue-rotate turns it dark red in Dark mode */
.jp-OutputArea-stderr, .output_stderr {
    background: transparent !important;
}
</style>"""

_LAYOUT_TRAITS: frozenset[str] = frozenset(widgets.Layout.class_trait_names())


def _layout(**kwargs: Any) -> widgets.Layout:
    """Create a Layout while dropping unsupported kwargs to avoid traitlets noise."""
    normalized = dict(kwargs)
    if "overflow_y" in normalized and "overflow" not in normalized:
        normalized["overflow"] = normalized["overflow_y"]
    normalized.pop("overflow_y", None)
    if "gap" in normalized and "grid_gap" not in normalized:
        normalized["grid_gap"] = normalized["gap"]
    normalized.pop("gap", None)
    if "flex_wrap" in normalized and "flex_flow" not in normalized:
        normalized["flex_flow"] = f"row {normalized['flex_wrap']}"
    normalized.pop("flex_wrap", None)
    filtered = {k: v for k, v in normalized.items() if k in _LAYOUT_TRAITS}
    return widgets.Layout(**filtered)


# ── SCF regex (module-level so _LogCapture can use them) ─────────────────────
_RE_CYCLE = re.compile(
    r"cycle=\s*(\d+)\s+E=\s*([\-\d\.]+)\s+delta_E=\s*([\-\d\.Ee+\-]+)"
)
_RE_CONV = re.compile(r"converged SCF energy\s*=\s*([\-\d\.]+)")
_RE_Q_STATUS = re.compile(r"\[QuantUI_STATUS\]\s*(.+)")


# ══ LOG CAPTURE ══════════════════════════════════════════════════════════════


class _LogCapture:
    """Write PySCF output to an Output widget and capture it to a buffer."""

    def __init__(
        self,
        output_widget: widgets.Output,
        status_label: Optional[widgets.Label] = None,
        on_scf_converged: Optional[Callable[[], None]] = None,
    ) -> None:
        self._w = output_widget
        self._buf = io.StringIO()
        self._line_buf = ""
        self._status = status_label
        self._on_scf_converged = on_scf_converged
        self._scf_converged_seen = False

    def write(self, text: str) -> None:
        if not text:
            return
        self._w.append_stdout(text)
        self._buf.write(text)
        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            m = _RE_Q_STATUS.search(line)
            if m and self._status is not None:
                self._status.value = m.group(1).strip()
                continue
            m = _RE_CYCLE.search(line)
            if m and self._status is not None:
                n, delta = m.group(1), m.group(3)
                try:
                    self._status.value = f"SCF cycle {n}  ·  ΔE = {float(delta):.4g} Ha"
                except Exception:
                    self._status.value = f"SCF cycle {n}"
                continue
            m = _RE_CONV.search(line)
            if m:
                if self._status is not None:
                    self._status.value = "SCF converged ✓"
                if not self._scf_converged_seen and self._on_scf_converged is not None:
                    self._scf_converged_seen = True
                    try:
                        self._on_scf_converged()
                    except Exception:
                        pass

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ══ ANALYSIS CONTEXT ═════════════════════════════════════════════════════════


@dataclass
class _AnalysisContext:
    """All data needed to populate Analysis panels for one result.

    Created by ``_do_run()`` for live results and by the history loaders for
    saved results.  Passed to ``QuantUIApp._apply_analysis_context()``, which
    uses ``_PANEL_REGISTRY`` to populate and activate the appropriate panels.
    """

    calc_type: str  # "single_point" | "geometry_opt" | etc.
    formula: str
    method: str
    basis: str
    live_result: Any = None  # result object from _do_run; None for history
    result_dir: Optional[Any] = None  # Path to saved dir; None before save_result
    molecule: Optional[Any] = None  # molecule used for the calculation
    spectra_data: dict = field(default_factory=dict)  # from save_spectra / disk
    preopt_result: Optional[Any] = None  # OptimizationResult from pre-opt step
    timestamp: str = ""  # result timestamp shown in history dropdown labels
    source: str = "live"  # "live" | "history"

    @property
    def label(self) -> str:
        if self.method:
            return f"{self.formula}  {self.method}/{self.basis}"
        return self.formula


# ══ APP CLASS ════════════════════════════════════════════════════════════════


class QuantUIApp:
    """
    Self-contained QuantUI application widget.

    Instantiate once; call ``display()`` to inject CSS and show the UI::

        app = QuantUIApp()
        app.display()
    """

    if TYPE_CHECKING:
        # Attributes initialized in companion builder modules. Keeping these
        # declarations here avoids attr-defined churn during phased extraction.
        _clear_log_cache_btn: Any
        _clear_log_cache_confirm_btn: Any
        _exit_btn: Any
        _exit_output: Any
        _help_btn: Any
        _issue_btn: Any
        _issue_cancel_btn: Any
        _issue_overlay: Any
        _issue_status_html: Any
        _issue_submit_btn: Any
        _issue_textarea: Any
        _cal_accordion: Any
        _cal_mode_toggle: Any
        _cal_progress: Any
        _cal_results_html: Any
        _cal_run_btn: Any
        _cal_step_label: Any
        _cal_stop_btn: Any
        _log_clear_btn: Any
        _log_output_html: Any
        _log_source_lbl: Any
        _perf_accordion: Any
        _perf_events_html: Any
        _perf_stats_html: Any
        _reset_btn: Any
        _reset_confirm_box: Any
        _reset_confirm_html: Any
        _reset_confirm_no: Any
        _reset_confirm_yes: Any
        _status_html: Any
        _status_tab_panel: Any
        _theme_style: Any
        _welcome_html: Any
        _activity_btn: Any
        advanced_accordion: Any
        calc_setup_panel: Any
        change_mol_btn: Any
        copy_path_btn: Any
        compare_btn: Any
        compare_clear_btn: Any
        compare_output: Any
        compare_panel: Any
        compare_refresh_btn: Any
        compare_select: Any
        files_tab_panel: Any
        _files_entries: Any
        _files_open_btn: Any
        _files_path_html: Any
        _files_preview_output: Any
        _files_refresh_btn: Any
        _files_root_dd: Any
        _files_status_html: Any
        _files_up_btn: Any
        help_content_html: Any
        help_tab_panel: Any
        help_topic_dd: Any
        history_panel: Any
        log_tab_panel: Any
        mol_input_collapsed: Any
        mol_input_container: Any
        mol_input_expanded: Any
        past_dd: Any
        past_output: Any
        past_refresh_btn: Any
        lib_category_dd: Any
        lib_search_txt: Any
        lib_results_dd: Any
        lib_count_lbl: Any
        pubchem_btn: Any
        pubchem_msg: Any
        pubchem_txt: Any
        pubchem_candidates_dd: Any
        result_output: Any
        result_viz_output: Any
        results_path_lbl: Any
        run_btn: Any
        run_output: Any
        run_panel: Any
        run_status: Any
        solvent_cb: Any
        solvent_dd: Any
        step_progress: Any
        theme_btn: Any
        vib_framerate_si: Any
        viz_backend_label_ana: Any
        viz_backend_toggle: Any
        viz_backend_toggle_ana: Any
        viz_controls_box: Any
        viz_default_backend_dd: Any
        viz_lighting_dd: Any
        viz_output: Any
        viz_style_dd: Any
        view_log_btn: Any
        xyz_area: Any
        xyz_btn: Any
        xyz_msg: Any
        _freq_preopt_cb: Any
        _freq_seed_dd: Any
        _freq_seed_note: Any
        _freq_seed_refresh_btn: Any
        _go_analysis_btn: Any
        _go_results_btn: Any
        _ir_export_btn: Any
        _ir_export_fmt_dd: Any
        _ir_export_status: Any
        _ir_fig: Any
        _ir_fwhm_slider: Any
        _ir_mode_toggle: Any
        _ir_accordion: Any
        _iso_accordion: Any
        _iso_generate_btn: Any
        _last_result_dir: Any
        _nmr_accordion: Any
        _nmr_output: Any
        _orb_accordion: Any
        _orb_diagram_box: Any
        _orb_diagram_html: Any
        _orb_export_btn: Any
        _orb_export_fmt_dd: Any
        _orb_export_status: Any
        _orb_iso_controls: Any
        _orb_iso_output: Any
        _orb_n_orb_input: Any
        _orb_toggle: Any
        _orb_ymax_input: Any
        _orb_ymin_input: Any
        _pes_export_btn: Any
        _pes_export_fmt_dd: Any
        _pes_export_status: Any
        _pes_plot_html: Any
        _pes_scan_accordion: Any
        _result_dir_label: Any
        _result_log_accordion: Any
        _result_log_output: Any
        _scan_atom1: Any
        _scan_atom2: Any
        _scan_atom3: Any
        _scan_atom34_box: Any
        _scan_atom4: Any
        _scan_start: Any
        _scan_steps: Any
        _scan_stop: Any
        _scan_type_dd: Any
        _scan_unit_lbl: Any
        _tddft_accordion: Any
        _tddft_fig: Any
        _uv_export_btn: Any
        _uv_export_fmt_dd: Any
        _uv_export_status: Any
        _uv_fwhm_slider: Any
        _uv_mode_toggle: Any
        _to_analysis_btn: Any
        _viz_backend: Any
        _viz_label: Any
        _viz_lighting: Any
        _viz_style: Any
        _analysis_context_lbl: Any
        _analysis_empty_html: Any
        _analysis_mol_output: Any
        _ana_unavail_html: Any
        accumulate_btn: Any
        analysis_tab_panel: Any
        basis_dd: Any
        basis_help_btn: Any
        calc_extra_opts: Any
        calc_type_dd: Any
        charge_si: Any
        clear_btn: Any
        _completion_banner: Any
        _completion_mol_lbl: Any
        comparison_output: Any
        export_btn: Any
        export_mol_btn: Any
        export_pdb_btn: Any
        export_status: Any
        export_xyz_btn: Any
        fmax_fi: Any
        log_clear_btn: Any
        max_steps_si: Any
        method_dd: Any
        method_help_btn: Any
        mol_info_html: Any
        mol_summary_compact: Any
        mult_si: Any
        notes_output: Any
        nstates_si: Any
        perf_estimate_html: Any
        post_calc_panel: Any
        preopt_cb: Any
        results_panel: Any
        results_tab_panel: Any
        struct_export_status: Any
        traj_accordion: Any
        traj_output: Any
        vib_accordion: Any
        vib_mode_dd: Any
        vib_output: Any

    def __init__(self) -> None:
        # ── Instance state ────────────────────────────────────────────────
        self._molecule: Optional[Molecule] = None
        self._last_result: Any = None
        self._last_calc_type: Optional[str] = None  # e.g. "frequency", "single_point"
        self._results: List = []
        self._pending_traj_result: Any = None
        # Cached for the fresh-path safety net in on_traj_expand: if the
        # initial render's outputs go missing before the user views the
        # Analysis tab, on_traj_expand re-renders from this cache. Cleared
        # by apply_analysis_context at every context reset.
        self._last_traj_result: Any = None
        # Generation counter for vibrational-animation renders. Each
        # mode-dropdown change bumps this so older worker-thread renders
        # bail out before they overwrite the newer render's output.
        self._vib_render_token: int = 0
        self._traj_render_token: int = 0
        self._iso_render_token: int = 0
        self._last_uv_wavelengths_nm: list[float] = []
        self._last_uv_oscillator_strengths: list[float] = []
        self._last_ir_fig: Any = None
        self._last_uv_fig: Any = None
        self._last_orb_fig: Any = None
        self._last_orb_info: Any = None
        # Orbital state consumed by the Isosurface panel populator. Always
        # initialized to None so ``pop_isosurface`` can read the attributes
        # via direct access without raising AttributeError on a fresh app
        # or on a history-replay where ``orbitals.npz`` is missing (BUG.8).
        # ``_apply_analysis_context`` resets these between contexts so stale
        # state from a prior calc cannot leak into the next molecule.
        self._last_orb_mo_coeff: Any = None
        self._last_orb_mol_atom: Any = None
        self._last_orb_mol_basis: Any = None
        # Last-generated cube file path + orbital label (M-EXPORT / EXPORT.5).
        # Set by the isosurface render path; consumed by the Export cube
        # button. Initialized here so the button handler reads ``None``
        # cleanly when no isosurface has been generated yet.
        self._last_cube_path: Optional[Path] = None
        self._last_cube_orbital: Optional[str] = None
        self._last_pes_fig: Any = None
        self._run_output_scroll_guard_installed: bool = False
        self._files_current_dir: Optional[Path] = None
        self._files_selected_path: Optional[Path] = None
        self._files_updating: bool = False
        self._activity_count: int = 0
        self._activity_compute_count: int = 0
        self._activity_lock = threading.Lock()
        # Cache kernel io_loop once on the main thread so worker threads can
        # reliably schedule UI callbacks even when get_ipython() is thread-local.
        self._kernel_io_loop: Any = getattr(
            getattr(get_ipython(), "kernel", None), "io_loop", None
        )
        self.root_tab: widgets.Tab
        self._session_id: str = _uuid.uuid4().hex[:12]

        # Availability (copied from module-level flags)
        self._pyscf_available: bool = _PYSCF_AVAILABLE
        self._preopt_available: bool = _PREOPT_AVAILABLE

        # User settings (persisted in ~/.quantui/settings.json) + viz
        # backend availability snapshot. The router consumes these; render
        # call sites will be migrated to the router in VIZBACK.4 ff.
        self._user_settings: UserSettings = UserSettings.load()
        self._viz_availability: BackendAvailability = (
            BackendAvailability.from_environment()
        )
        self._viz_backend_preference: str = self._user_settings.viz.default_backend

        # Synchronization state for Calculate/Analysis backend toggles.
        # When _set_viz_backend updates one toggle, it sets this flag so the
        # other toggle's observer can short-circuit and avoid an echo loop.
        self._viz_sync_in_progress: bool = False
        # Molecule currently rendered into _analysis_mol_output. Updated by
        # show_result_3d; consumed by _set_viz_backend to re-render the
        # Analysis-tab viewer when the toggle changes.
        self._analysis_displayed_molecule: Any = None

        # ── Build → wire → assemble ───────────────────────────────────────
        self._build_widgets()

        # Resolve the persisted preference through the router and align all
        # three preference widgets + _viz_backend with the router decision.
        # Observers are NOT yet wired so widget assignments don't trigger
        # render side-effects — this is pure initial-state alignment.
        self._initialize_viz_state_from_preference()

        self._wire_callbacks()
        self._assemble_tabs()

        # Log startup, but never let optional logging I/O break app startup.
        try:
            _calc_log.log_event("startup", f"QuantUI {quantui.__version__} started")
        except OSError:
            pass

    def display(self) -> None:
        """Inject global CSS and render the application widget."""
        display(HTML(_APP_CSS))
        display(
            widgets.VBox(
                [
                    self._welcome_header,
                    widgets.HBox(
                        [
                            self._activity_btn,
                            self.theme_btn,
                            self._help_btn,
                            self._issue_btn,
                            self._exit_btn,
                        ],
                        layout=_layout(justify_content="flex-end", margin="0 0 4px"),
                    ),
                    self._issue_overlay,
                    self._exit_output,
                    self._theme_style,
                    self.help_tab_panel,
                    self.root_tab,
                ]
            )
        )
        self._install_run_output_scroll_guard()

    @property
    def widget(self) -> widgets.Tab:
        """The root tab widget (for callers that want the widget object)."""
        return self.root_tab

    # ══ BUILD METHODS ════════════════════════════════════════════════════════

    def _build_widgets(self) -> None:
        self._build_theme_selector()
        self._build_status_panel()
        self._build_welcome_header()
        self._build_shared_widgets()
        self._build_molecule_section()
        self._build_calc_setup()
        self._build_run_section()
        self._build_results_section()
        self._build_history_section()
        self._build_compare_section()
        self._build_output_tab()
        self._build_files_tab()
        self._build_help_section()
        self._build_issue_widgets()

    # ── Theme selector ────────────────────────────────────────────────────

    def _build_theme_selector(self) -> None:
        _bld_build_theme_selector(self, layout_fn=_layout)

    def _theme_css(self, theme: str) -> str:
        """Return the CSS filter block for *theme*, or '' for Light."""
        if theme not in _THEME_HUE:
            return ""
        deg = _THEME_HUE[theme]
        return (
            "<style>"
            f"html {{ filter: invert(1) hue-rotate({deg}deg) !important; }}"
            "canvas, img, iframe, video "
            f"{{ filter: invert(1) hue-rotate({deg}deg) !important; }}"
            "</style>"
        )

    def _set_activity_indicator(self, state: str = "idle", message: str = "") -> None:
        """Update the toolbar activity light state and tooltip."""
        if state == "compute":
            self._activity_btn.description = "Computing"
            self._activity_btn.icon = "cog"
            self._activity_btn.button_style = "warning"
            self._activity_btn.tooltip = message or "Running compute operations..."
            return
        if state == "ui":
            self._activity_btn.description = "UI Active"
            self._activity_btn.icon = "bolt"
            self._activity_btn.button_style = "info"
            self._activity_btn.tooltip = message or "Running UI operations..."
            return

        self._activity_btn.description = "Idle"
        self._activity_btn.icon = "circle-o"
        self._activity_btn.button_style = "success"
        self._activity_btn.tooltip = "No active operations."

    def _refresh_activity_indicator(self, message: str = "") -> None:
        """Recompute activity light state from active operation counters."""
        if self._activity_count <= 0:
            self._set_activity_indicator("idle")
            return
        if self._activity_compute_count > 0:
            self._set_activity_indicator("compute", message)
            return
        self._set_activity_indicator("ui", message)

    def _activity_begin(self, message: str = "", kind: str = "ui") -> None:
        """Mark one operation as active."""
        with self._activity_lock:
            self._activity_count += 1
            if kind == "compute":
                self._activity_compute_count += 1
        self._refresh_activity_indicator(message)

    def _activity_end(self, kind: str = "ui") -> None:
        """Mark one operation as finished."""
        with self._activity_lock:
            if self._activity_count > 0:
                self._activity_count -= 1
            if kind == "compute" and self._activity_compute_count > 0:
                self._activity_compute_count -= 1
        self._refresh_activity_indicator()

    def _activity_pulse(
        self, message: str, hold_s: float = 0.18, kind: str = "ui"
    ) -> None:
        """Briefly light the activity indicator for quick operations."""
        self._activity_begin(message, kind=kind)
        timer = threading.Timer(
            max(0.05, hold_s),
            self._activity_end,
            kwargs={"kind": kind},
        )
        timer.daemon = True
        timer.start()

    def _on_root_tab_changed(self, _change) -> None:
        """Pulse the activity light on tab navigation actions."""
        self._activity_pulse("Switching tabs...", hold_s=0.16, kind="ui")

    def _go_to_results_tab(self, _btn) -> None:
        """Navigate to Results tab with a visible activity pulse."""
        self._activity_pulse("Navigating to Results tab...", hold_s=0.16, kind="ui")
        self.root_tab.selected_index = 1

    def _go_to_analysis_tab(self, _btn) -> None:
        """Navigate to Analysis tab with a visible activity pulse."""
        self._activity_pulse("Navigating to Analysis tab...", hold_s=0.16, kind="ui")
        self.root_tab.selected_index = 2

    # ── Status panel ──────────────────────────────────────────────────────

    def _build_status_panel(self) -> None:
        _bld_build_status_panel(
            self,
            layout_fn=_layout,
            get_session_resources_fn=get_session_resources,
            load_last_calibration_label_fn=_load_last_calibration_label,
            pyscf_available=_PYSCF_AVAILABLE,
            ase_available=ASE_AVAILABLE,
            pubchem_available=PUBCHEM_AVAILABLE,
            visualization_available=VISUALIZATION_AVAILABLE,
            viz_default_backend=self._user_settings.viz.default_backend,
            vib_framerate_fps=self._user_settings.viz.vib_framerate_fps,
        )

    # ── Welcome header ────────────────────────────────────────────────────

    def _build_welcome_header(self) -> None:
        _bld_build_welcome_header(self, layout_fn=_layout)

    # ── Shared widgets (Cell 3) ───────────────────────────────────────────

    def _build_shared_widgets(self) -> None:
        _bld_build_shared_widgets(
            self,
            layout_fn=_layout,
            step_progress_cls=StepProgress,
            supported_methods=SUPPORTED_METHODS,
            supported_basis_sets=SUPPORTED_BASIS_SETS,
            default_method=DEFAULT_METHOD,
            default_basis=DEFAULT_BASIS,
            default_charge=DEFAULT_CHARGE,
            default_multiplicity=DEFAULT_MULTIPLICITY,
            default_fmax=DEFAULT_FMAX,
            default_opt_steps=DEFAULT_OPT_STEPS,
            preopt_available=_PREOPT_AVAILABLE,
            visualization_available=VISUALIZATION_AVAILABLE,
            both_viz_available=_BOTH_VIZ_AVAILABLE,
            default_viz_backend=_DEFAULT_VIZ_BACKEND,
            default_viz_style=_DEFAULT_VIZ_STYLE,
            default_lighting=_DEFAULT_LIGHTING,
            viz_style_options=_VIZ_STYLE_OPTIONS,
            plotlymol_viz=_PLOTLYMOL_VIZ,
            lighting_options=_LIGHTING_OPTIONS,
            rdkit_available=_RDKIT_AVAILABLE,
        )

    # ── Molecule section (Cell 4) ─────────────────────────────────────────

    def _build_molecule_section(self) -> None:
        _bld_build_molecule_section(
            self,
            layout_fn=_layout,
            pubchem_available=PUBCHEM_AVAILABLE,
            visualization_available=VISUALIZATION_AVAILABLE,
        )

    # ── Calculation setup panel (Cell 5) ──────────────────────────────────

    def _build_calc_setup(self) -> None:
        _bld_build_calc_setup(self, layout_fn=_layout)

    # ── Run panel (Cell 6) ────────────────────────────────────────────────

    def _build_run_section(self) -> None:
        _bld_build_run_section(self, layout_fn=_layout)

    # ── Results panel (Cell 7) ────────────────────────────────────────────

    def _build_results_section(self) -> None:
        _bld_build_results_section(self, layout_fn=_layout)

    # ── Analysis panel switcher ───────────────────────────────────────────

    def _build_ana_switcher(self) -> None:
        _ana_build_ana_switcher(self, layout_fn=_layout)

    def _on_ir_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_ir_freqs", None):
            self._update_ir_figure(
                self._ir_mode_toggle.value, self._ir_fwhm_slider.value
            )

    def _on_tddft_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_uv_wavelengths_nm", None):
            self._update_uv_vis_figure(
                self._uv_mode_toggle.value,
                self._uv_fwhm_slider.value,
            )

    def _on_orb_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_orb_info", None) is not None:
            self._on_orb_range_changed()

    def _select_ana_panel(self, name: str) -> None:
        _ana_select_ana_panel(self, name)

    def _activate_ana_panel(self, name: str, auto_select: bool = True) -> None:
        _ana_activate_ana_panel(self, name, auto_select=auto_select)

    def _deactivate_all_ana_panels(self) -> None:
        _ana_deactivate_all_ana_panels(self)

    # ── Panel registry and unified applier ───────────────────────────────────
    #
    # _PANEL_META: ordered list of (name, accordion_attr, when_str) for every
    # analysis panel.  Single source of truth for names, accordion references,
    # and the "available after: …" tooltip text.
    #
    # _PANEL_REGISTRY maps calc_type → ordered list of
    # (panel_name, populate_method_name, auto_select) tuples.
    #
    # Rules:
    #   • populate_method_name is a string — looked up via getattr at runtime.
    #   • auto_select=True on the FIRST entry that returns True activates that
    #     panel as the default view; subsequent entries with auto_select=True are
    #     treated as False (only one panel is auto-selected per result).
    #   • If a populate method returns False / None the panel stays disabled.
    #   • Populate methods must NOT call _activate_ana_panel themselves.

    _PANEL_META: ClassVar[list] = [
        ("Energies", "_orb_accordion", "Single Point / Geometry Opt"),
        ("Trajectory", "traj_accordion", "Geometry Opt / PES Scan / Frequency pre-opt"),
        ("Vibrational", "vib_accordion", "Frequency"),
        ("IR Spectrum", "_ir_accordion", "Frequency"),
        ("PES Scan", "_pes_scan_accordion", "PES Scan"),
        ("Isosurface", "_iso_accordion", "Single Point (Linux/WSL only)"),
        ("UV-Vis", "_tddft_accordion", "UV-Vis (TD-DFT)"),
        ("NMR", "_nmr_accordion", "NMR Shielding"),
    ]

    _PANEL_REGISTRY: ClassVar[dict] = {
        "single_point": [
            ("Energies", "_pop_energies", True),
            ("Isosurface", "_pop_isosurface", False),
        ],
        "geometry_opt": [
            ("Trajectory", "_pop_geo_trajectory", True),
            ("Energies", "_pop_energies", False),
            ("Isosurface", "_pop_isosurface", False),
        ],
        "frequency": [
            ("Vibrational", "_pop_vibrational", True),
            ("IR Spectrum", "_pop_ir_spectrum", True),
            ("Trajectory", "_pop_preopt_trajectory", False),
            ("Energies", "_pop_energies", True),
        ],
        "tddft": [
            ("UV-Vis", "_pop_uv_vis", True),
        ],
        "nmr": [
            ("NMR", "_pop_nmr_shielding", True),
        ],
        "pes_scan": [
            ("PES Scan", "_pop_pes_plot", True),
            ("Trajectory", "_pop_pes_trajectory", False),
        ],
    }

    def _apply_analysis_context(self, ctx: _AnalysisContext) -> None:
        _ana_apply_analysis_context(self, ctx)

    # ── Panel populate methods ────────────────────────────────────────────────
    # Each receives an _AnalysisContext and returns True if data was rendered.

    def _pop_energies(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_energies(self, ctx)

    def _pop_isosurface(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_isosurface(self, ctx)

    def _pop_geo_trajectory(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_geo_trajectory(self, ctx)

    def _pop_preopt_trajectory(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_preopt_trajectory(self, ctx)

    def _pop_vibrational(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_vibrational(self, ctx)

    def _pop_ir_spectrum(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_ir_spectrum(self, ctx)

    def _pop_uv_vis(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_uv_vis(self, ctx)

    def _pop_nmr_shielding(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_nmr_shielding(self, ctx)

    def _pop_pes_plot(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_pes_plot(self, ctx)

    def _pop_pes_trajectory(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_pes_trajectory(self, ctx)

    # ── History panel (Cell 8) ────────────────────────────────────────────

    def _build_history_section(self) -> None:
        _bld_build_history_section(
            self,
            layout_fn=_layout,
            pyscf_available=_PYSCF_AVAILABLE,
            benchmark_suite=_BENCHMARK_SUITE,
            benchmark_suite_long=_BENCHMARK_SUITE_LONG,
            load_last_calibration_label_fn=_load_last_calibration_label,
        )

    # ── Compare panel (Cell 9) ────────────────────────────────────────────

    def _build_compare_section(self) -> None:
        _bld_build_compare_section(
            self,
            layout_fn=_layout,
            rdkit_available=_RDKIT_AVAILABLE,
        )

    # ── Output log tab (Cell 10) ──────────────────────────────────────────

    def _build_output_tab(self) -> None:
        _bld_build_output_tab(self, layout_fn=_layout)

    # ── Files tab (Cell 11) ───────────────────────────────────────────────

    def _build_files_tab(self) -> None:
        _bld_build_files_tab(self, layout_fn=_layout)
        self._refresh_file_browser()

    # ── Help section (Cell 12) ────────────────────────────────────────────

    def _build_help_section(self) -> None:
        _bld_build_help_section(self, layout_fn=_layout)

    def _build_issue_widgets(self) -> None:
        _bld_build_issue_widgets(self, layout_fn=_layout)

    # ── Tab assembly (Cell 10) ────────────────────────────────────────────

    def _assemble_tabs(self) -> None:
        _calculate_content = widgets.VBox(
            [
                self.step_progress.widget,
                self.mol_input_container,
                self.calc_setup_panel,
                self.run_panel,
                self._completion_banner,
            ],
            layout=_layout(padding="8px 0"),
        )

        # Splice advanced_accordion into results_tab_panel before _to_analysis_btn.
        # It cannot be referenced in _build_results_section because it is built later
        # in _build_compare_section.
        _rtp = list(self.results_tab_panel.children)
        _rtp.insert(_rtp.index(self._to_analysis_btn), self.advanced_accordion)
        self.results_tab_panel.children = tuple(_rtp)

        # POLISH.8 (M-POLISH, 2026-05-25): Log moved to be an
        # Accordion inside the History tab — see build_output_tab for
        # the wrap. Tab indices renumbered: Files 6→5, System Settings
        # 7→6. Update any caller that depended on tab-index 5 being
        # "Log" (notably _goto_output_tab — now navigates to History
        # and expands the log accordion).
        self.root_tab = widgets.Tab(
            children=[
                _calculate_content,
                self.results_tab_panel,
                self.analysis_tab_panel,
                self.history_panel,
                self.compare_panel,
                self.files_tab_panel,
                self._status_tab_panel,
            ]
        )
        self.root_tab.set_title(0, "Calculate")
        self.root_tab.set_title(1, "Results")
        self.root_tab.set_title(2, "Analysis")
        self.root_tab.set_title(3, "History")
        self.root_tab.set_title(4, "Compare")
        self.root_tab.set_title(5, "Files")
        # POLISH.4 (M-POLISH, 2026-05-25): "Status" was ambiguous —
        # status of what? "System Settings" is what the tab actually
        # holds (env info + calibration + GPU status + UI prefs).
        self.root_tab.set_title(6, "System Settings")
        self.root_tab.observe(
            self._safe_cb(self._on_root_tab_changed), names="selected_index"
        )

    # ══ CALLBACK WIRING ══════════════════════════════════════════════════════

    def _wire_callbacks(self) -> None:
        # 3D viewer backend toggle (only wired when both backends are available)
        if self.viz_backend_toggle is not None:
            self.viz_backend_toggle.observe(
                self._safe_cb(self._on_viz_backend_changed), names="value"
            )
        # Analysis-tab backend toggle (only wired when both backends available).
        if self.viz_backend_toggle_ana is not None:
            self.viz_backend_toggle_ana.observe(
                self._safe_cb(self._on_viz_backend_changed_ana), names="value"
            )
        # Settings → "Default 3D backend" preference (Status tab; persisted).
        self.viz_default_backend_dd.observe(
            self._safe_cb(self._on_viz_default_backend_changed), names="value"
        )
        # Settings → Vibrational animation framerate (Status tab; persisted).
        self.vib_framerate_si.observe(
            self._safe_cb(self._on_vib_framerate_changed), names="value"
        )
        # 3D viewer style and lighting controls
        if VISUALIZATION_AVAILABLE:
            self.viz_style_dd.observe(
                self._safe_cb(self._on_viz_style_changed), names="value"
            )
            self.viz_lighting_dd.observe(
                self._safe_cb(self._on_viz_lighting_changed), names="value"
            )
        # Theme
        self.theme_btn.observe(self._safe_cb(self._on_theme_changed), names="value")
        # Molecule input — library browse/search (STRUCT.9)
        self.lib_category_dd.observe(
            self._safe_cb(self._on_lib_filter_changed), names="value"
        )
        self.lib_search_txt.observe(
            self._safe_cb(self._on_lib_filter_changed), names="value"
        )
        self.lib_results_dd.observe(self._safe_cb(self._on_lib_select), names="value")
        self.xyz_btn.on_click(self._on_load_xyz)
        self.pubchem_btn.on_click(self._on_search_pubchem)
        self.pubchem_candidates_dd.observe(
            self._safe_cb(self._on_pubchem_candidate_selected), names="value"
        )
        self.change_mol_btn.on_click(self._on_expand_mol_input)
        # Calc type
        self.calc_type_dd.observe(
            self._safe_cb(self._on_calc_type_changed), names="value"
        )
        self._freq_seed_dd.observe(
            self._safe_cb(self._on_freq_seed_changed), names="value"
        )
        self._tddft_seed_dd.observe(
            self._safe_cb(self._on_tddft_seed_changed), names="value"
        )
        self._scan_type_dd.observe(
            self._safe_cb(self._update_scan_widgets), names="value"
        )
        self._freq_seed_refresh_btn.on_click(
            lambda _btn: self._refresh_freq_seed_options()
        )
        self._tddft_seed_refresh_btn.on_click(
            lambda _btn: self._refresh_tddft_seed_options()
        )
        # Notes + estimate
        self.method_dd.observe(self._safe_cb(self._update_notes), names="value")
        self.basis_dd.observe(self._safe_cb(self._update_notes), names="value")
        self.method_dd.observe(self._safe_cb(self._update_estimate), names="value")
        self.basis_dd.observe(self._safe_cb(self._update_estimate), names="value")
        # Help buttons
        self.method_help_btn.on_click(self._on_method_help)
        self.basis_help_btn.on_click(self._on_basis_help)
        # Run
        self.run_btn.on_click(self._on_run_clicked)
        self.log_clear_btn.on_click(self._on_clear_log)
        self._ir_mode_toggle.observe(
            self._safe_cb(self._on_ir_mode_changed), names="value"
        )
        self._ir_fwhm_slider.observe(
            self._safe_cb(self._on_ir_fwhm_changed), names="value"
        )
        self._uv_mode_toggle.observe(
            self._safe_cb(self._on_uv_mode_changed), names="value"
        )
        self._uv_fwhm_slider.observe(
            self._safe_cb(self._on_uv_fwhm_changed), names="value"
        )
        self._ir_export_btn.on_click(self._on_ir_export_plot)
        self._uv_export_btn.on_click(self._on_uv_export_plot)
        self._orb_export_btn.on_click(self._on_orb_export_plot)
        self._pes_export_btn.on_click(self._on_pes_export_plot)
        self._vib_export_btn.on_click(self._on_vib_export_animation)
        # M-EXPORT / EXPORT.4: per-panel CSV-to-clipboard / file buttons.
        self._ir_copy_data_btn.on_click(self._on_ir_copy_data)
        self._uv_copy_data_btn.on_click(self._on_uv_copy_data)
        self._orb_copy_data_btn.on_click(self._on_orb_copy_data)
        self._pes_copy_data_btn.on_click(self._on_pes_copy_data)
        # Accumulate / export
        self.accumulate_btn.on_click(self._on_accumulate)
        self.clear_btn.on_click(self._on_clear)
        self.solvent_cb.observe(
            self._safe_cb(self._on_solvent_cb_changed), names="value"
        )
        self._cal_run_btn.on_click(self._on_cal_run)
        self._cal_stop_btn.on_click(self._on_cal_stop)
        self._cal_skip_btn.on_click(self._on_cal_skip)
        self.export_btn.on_click(self._on_export)
        self.export_xyz_btn.on_click(self._on_export_xyz)
        self.export_mol_btn.on_click(self._on_export_mol)
        self.export_pdb_btn.on_click(self._on_export_pdb)
        # History
        self.past_dd.observe(self._safe_cb(self._on_past_dd_changed), names="value")
        self.past_refresh_btn.on_click(self._on_past_refresh)
        self.copy_path_btn.on_click(self._on_copy_results_path)
        self.view_log_btn.on_click(self._on_view_log)
        # Perf stats reset
        self._reset_btn.on_click(self._on_reset_click)
        self._reset_confirm_yes.on_click(self._on_confirm_yes)
        self._reset_confirm_no.on_click(self._on_confirm_no)
        # Compare
        self.compare_refresh_btn.on_click(self._on_compare_refresh)
        self.compare_btn.on_click(self._on_compare)
        self.compare_clear_btn.on_click(self._on_compare_clear)
        # Output log
        self._log_clear_btn.on_click(self._on_log_clear)
        # Clear log cache (event_log.jsonl)
        self._clear_log_cache_btn.on_click(self._on_clear_log_cache)
        self._clear_log_cache_confirm_btn.on_click(self._on_clear_log_cache_confirm)
        # Files tab
        self._files_root_dd.observe(
            self._safe_cb(self._on_files_root_changed), names="value"
        )
        self._files_entries.observe(
            self._safe_cb(self._on_files_entry_changed), names="value"
        )
        self._files_open_btn.on_click(self._on_files_open)
        self._files_up_btn.on_click(self._on_files_up)
        self._files_refresh_btn.on_click(self._on_files_refresh)
        # Issue reporting
        self._issue_btn.on_click(self._on_issue_btn)
        self._issue_submit_btn.on_click(self._on_issue_submit)
        self._issue_cancel_btn.on_click(self._on_issue_cancel)
        # Help [?] toggle
        self._help_btn.on_click(self._on_help_toggle)
        # Exit
        self._exit_btn.on_click(self._on_exit_clicked)
        self.help_topic_dd.observe(
            self._safe_cb(self._on_help_topic_changed), names="value"
        )
        # Tab navigation buttons
        self._go_results_btn.on_click(self._go_to_results_tab)
        self._go_analysis_btn.on_click(self._go_to_analysis_tab)
        self._to_analysis_btn.on_click(self._go_to_analysis_tab)
        # Vibrational mode selector
        self.vib_mode_dd.observe(
            self._safe_cb(self._on_vib_mode_changed), names="value"
        )
        self.vib_mode_dd.observe(
            self._safe_cb(self._update_vib_nav_buttons), names=["value", "options"]
        )
        self.vib_prev_btn.on_click(self._on_vib_prev_clicked)
        self.vib_next_btn.on_click(self._on_vib_next_clicked)
        # Orbital diagram axis controls
        self._orb_ymin_input.observe(
            self._safe_cb(self._on_orb_range_changed), names="value"
        )
        self._orb_ymax_input.observe(
            self._safe_cb(self._on_orb_range_changed), names="value"
        )
        self._orb_n_orb_input.observe(
            self._safe_cb(self._on_orb_range_changed), names="value"
        )
        # Orbital isosurface generate button
        self._iso_generate_btn.on_click(self._on_iso_generate)
        # M-EXPORT / EXPORT.5: cube + bundle exports
        self._iso_export_cube_btn.on_click(self._on_iso_export_cube)
        self._export_bundle_btn.on_click(self._on_export_bundle)

    # ── Files tab ────────────────────────────────────────────────────────

    def _files_allowed_roots(self) -> list[Path]:
        """Return the approved filesystem roots for the Files tab."""
        roots: list[Path] = []
        candidates: list[Optional[Path]] = [self._get_results_dir(), Path.cwd()]
        _last_dir = getattr(self, "_last_result_dir", None)
        if isinstance(_last_dir, Path):
            candidates.append(_last_dir)

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved not in roots:
                roots.append(resolved)

        return roots

    def _is_path_in_allowed_roots(self, path: Path, roots: list[Path]) -> bool:
        """True when *path* is inside any configured Files-tab root."""
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for root in roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _format_file_size(self, size_bytes: int) -> str:
        """Return a compact human-readable size label."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _set_files_status(self, message: str, color: str = "#64748b") -> None:
        """Update Files tab status text."""
        self._files_status_html.value = (
            f'<span style="font-size:12px;color:{color}">'
            f"{_html.escape(message)}</span>"
        )

    def _format_files_root_label(self, root: Path) -> str:
        """Return a readable dropdown label for a root path."""
        labels: list[tuple[str, Path]] = []
        try:
            labels.append(("Results", self._get_results_dir().resolve()))
        except OSError:
            pass
        try:
            labels.append(("Workspace CWD", Path.cwd().resolve()))
        except OSError:
            pass
        _last_dir = getattr(self, "_last_result_dir", None)
        if isinstance(_last_dir, Path):
            try:
                labels.append(("Current Result", _last_dir.resolve()))
            except OSError:
                pass

        for prefix, known_root in labels:
            if root == known_root:
                return f"{prefix} ({root})"
        return str(root)

    def _refresh_file_browser(self) -> None:
        """Refresh root options and the current directory listing."""
        roots = self._files_allowed_roots()
        try:
            results_root = self._get_results_dir().resolve()
        except OSError:
            results_root = None
        for root in roots:
            if results_root is not None and root == results_root:
                try:
                    root.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass

        if not roots:
            self._files_updating = True
            try:
                self._files_root_dd.options = [("(no roots)", "")]
                self._files_root_dd.value = ""
                self._files_entries.options = [("(no files)", "")]
                self._files_entries.value = ""
            finally:
                self._files_updating = False
            self._files_current_dir = None
            self._files_selected_path = None
            self._files_path_html.value = (
                '<span style="font-size:12px;color:#64748b">'
                "Current folder: unavailable</span>"
            )
            self._files_open_btn.disabled = True
            self._files_up_btn.disabled = True
            self._set_files_status("No readable roots available.", "#b91c1c")
            self._files_preview_output.clear_output(wait=True)
            return

        old_root_value = str(self._files_root_dd.value or "")
        root_options = [
            (self._format_files_root_label(root), str(root)) for root in roots
        ]
        valid_root_values = {value for _, value in root_options}
        selected_root = old_root_value if old_root_value in valid_root_values else ""
        if not selected_root:
            selected_root = root_options[0][1]

        self._files_updating = True
        try:
            self._files_root_dd.options = root_options
            self._files_root_dd.value = selected_root
        finally:
            self._files_updating = False

        selected_root_path = Path(selected_root)
        if (
            self._files_current_dir is None
            or not self._is_path_in_allowed_roots(self._files_current_dir, roots)
            or not self._files_current_dir.exists()
            or not self._files_current_dir.is_dir()
        ):
            self._files_current_dir = selected_root_path

        self._update_files_entries()
        self._set_files_status("File list refreshed.")

    def _update_files_entries(self) -> None:
        """Rebuild the directory listing for the current folder."""
        roots = self._files_allowed_roots()
        if not roots:
            self._files_entries.options = [("(no files)", "")]
            self._files_entries.value = ""
            self._files_selected_path = None
            self._files_open_btn.disabled = True
            self._files_up_btn.disabled = True
            self._files_preview_output.clear_output(wait=True)
            return

        current = self._files_current_dir or roots[0]
        if not self._is_path_in_allowed_roots(current, roots):
            current = Path(self._files_root_dd.value)
        if not current.exists() or not current.is_dir():
            current = Path(self._files_root_dd.value)

        self._files_current_dir = current
        self._files_path_html.value = (
            '<span style="font-size:12px;color:#475569">Current folder: '
            f"{_html.escape(str(current))}</span>"
        )

        try:
            children = list(current.iterdir())
        except OSError as exc:
            self._files_entries.options = [("(unreadable folder)", "")]
            self._files_entries.value = ""
            self._files_selected_path = None
            self._files_open_btn.disabled = True
            self._files_up_btn.disabled = True
            self._files_preview_output.clear_output(wait=True)
            self._set_files_status(f"Cannot list folder: {exc}", "#b91c1c")
            return

        children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        options: list[tuple[str, str]] = []
        for child in children:
            if child.is_dir():
                options.append((f"[DIR] {child.name}", str(child)))
                continue
            try:
                size_label = self._format_file_size(child.stat().st_size)
            except OSError:
                size_label = "?"
            options.append((f"{child.name} ({size_label})", str(child)))

        if not options:
            options = [("(empty directory)", "")]

        old_selection = str(self._files_entries.value or "")
        valid_values = {value for _, value in options if value}
        new_selection = old_selection if old_selection in valid_values else ""
        if not new_selection and valid_values:
            new_selection = next(iter(valid_values))

        self._files_updating = True
        try:
            self._files_entries.options = options
            self._files_entries.value = new_selection
        finally:
            self._files_updating = False

        self._files_selected_path = Path(new_selection) if new_selection else None
        self._files_open_btn.disabled = self._files_selected_path is None

        _parent = current.parent
        self._files_up_btn.disabled = (
            _parent == current or not self._is_path_in_allowed_roots(_parent, roots)
        )

        self._files_preview_output.clear_output(wait=True)

    def _preview_file_path(self, path: Path) -> None:
        """Render a safe preview for a selected file path."""
        roots = self._files_allowed_roots()
        if not self._is_path_in_allowed_roots(path, roots):
            self._set_files_status("Selected path is outside allowed roots.", "#b91c1c")
            return
        if not path.exists() or not path.is_file():
            self._set_files_status("Selected file no longer exists.", "#b91c1c")
            return

        self._files_preview_output.clear_output(wait=True)
        suffix = path.suffix.lower()

        image_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        text_ext = {
            ".txt",
            ".log",
            ".json",
            ".md",
            ".py",
            ".csv",
            ".yaml",
            ".yml",
            ".xyz",
            ".cube",
            ".molden",
        }

        if suffix in image_ext:
            from IPython.display import Image as _Image

            with self._files_preview_output:
                display(_Image(filename=str(path)))
            self._set_files_status(f"Previewing image: {path.name}")
            return

        if suffix == ".svg":
            # IPython.display.Image doesn't handle SVG well — use SVG.
            from IPython.display import SVG as _SVG

            with self._files_preview_output:
                display(_SVG(filename=str(path)))
            self._set_files_status(f"Previewing SVG: {path.name}")
            return

        # POLISH.5 (M-POLISH, 2026-05-25): specialized previews for
        # extensions where the generic text dump is unhelpful. Each
        # handler caps file reads at 256 KB. On any exception inside a
        # handler, fall through to the generic text dispatch below so
        # the user always sees SOMETHING. Order matters: 3D-structure
        # extensions (.xyz/.mol/.pdb) take precedence over their
        # text-ext membership.

        if suffix in {".xyz", ".mol", ".pdb"}:
            # 3D structure → py3Dmol viewer via raw model load. Falls
            # through to text dispatch on failure (so the user still
            # sees the raw coordinates).
            try:
                import py3Dmol as _p3d  # type: ignore[import]

                model_format = {".xyz": "xyz", ".mol": "mol", ".pdb": "pdb"}[suffix]
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                if len(raw_text) <= 256_000:
                    viewer = _p3d.view(width=500, height=380)
                    viewer.addModel(raw_text, model_format)
                    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
                    viewer.setBackgroundColor("white")
                    viewer.zoomTo()
                    html_str = viewer._make_html()
                    with self._files_preview_output:
                        display(HTML(html_str))
                    self._set_files_status(
                        f"3D structure preview: {path.name}"
                        f" ({model_format.upper()})"
                    )
                    return
            except Exception:  # noqa: BLE001 — fall through to text preview
                pass

        if suffix == ".json":
            try:
                import json as _json_pretty

                raw = path.read_bytes()[:256_000]
                parsed = _json_pretty.loads(raw.decode("utf-8", errors="replace"))
                pretty = _json_pretty.dumps(parsed, indent=2, ensure_ascii=False)
                # Cap line count so a 10k-key dict doesn't lock the viewport.
                lines = pretty.splitlines()
                truncated = False
                if len(lines) > 500:
                    lines = lines[:500]
                    truncated = True
                rendered = "\n".join(lines)
                if truncated:
                    rendered += "\n\n[truncated to first 500 lines]"
                with self._files_preview_output:
                    display(
                        HTML(
                            "<pre style='white-space:pre-wrap;word-break:break-word;"
                            "font-size:12px;line-height:1.35;margin:0'>"
                            f"{_html.escape(rendered)}</pre>"
                        )
                    )
                self._set_files_status(f"JSON preview: {path.name}")
                return
            except Exception:  # noqa: BLE001 — fall through to text preview
                pass

        if suffix == ".csv":
            try:
                import csv as _csv

                with open(path, encoding="utf-8", errors="replace", newline="") as fh:
                    reader = _csv.reader(fh)
                    rows: list[list[str]] = []
                    for i, row in enumerate(reader):
                        if i >= 50:
                            break
                        rows.append(row)
                if rows:
                    header = rows[0]
                    body = rows[1:]
                    head_html = "".join(
                        f'<th style="padding:4px 10px;text-align:left;'
                        f"border-bottom:1px solid #cbd5e1;font-size:12px;"
                        f'color:#1e293b">{_html.escape(str(c))}</th>'
                        for c in header
                    )
                    body_html = "".join(
                        "<tr>"
                        + "".join(
                            f'<td style="padding:3px 10px;font-size:12px;'
                            f"border-bottom:1px solid #f1f5f9;color:#334155;"
                            f'font-variant-numeric:tabular-nums">{_html.escape(str(c))}</td>'
                            for c in r
                        )
                        + "</tr>"
                        for r in body
                    )
                    note = (
                        f'<p style="font-size:11px;color:#94a3b8;margin:4px 0 6px">'
                        f"First {len(rows)} rows shown.</p>"
                        if len(rows) >= 50
                        else ""
                    )
                    table_html = (
                        f"{note}"
                        '<table style="border-collapse:collapse;width:100%">'
                        f"<thead><tr>{head_html}</tr></thead>"
                        f"<tbody>{body_html}</tbody></table>"
                    )
                    with self._files_preview_output:
                        display(HTML(table_html))
                    self._set_files_status(
                        f"CSV preview: {path.name} ({len(rows)} rows)"
                    )
                    return
            except Exception:  # noqa: BLE001 — fall through to text preview
                pass

        if suffix in {".html", ".htm"}:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                if len(raw) <= 1_000_000:
                    # Sandboxed iframe via srcdoc — embedded JS can't
                    # reach the parent app.
                    iframe_html = (
                        '<iframe sandbox="allow-scripts" '
                        'style="width:100%;height:400px;border:1px solid #cbd5e1;'
                        'border-radius:4px" '
                        f'srcdoc="{_html.escape(raw, quote=True)}"></iframe>'
                    )
                    with self._files_preview_output:
                        display(HTML(iframe_html))
                    self._set_files_status(f"HTML preview (sandboxed): {path.name}")
                    return
            except Exception:  # noqa: BLE001 — fall through to text preview
                pass

        if suffix == ".cube":
            # Cube files can be hundreds of MB (volumetric data). Don't
            # dump them — show the header + a size + a hint.
            try:
                stat = path.stat()
                with open(path, encoding="utf-8", errors="replace") as fh:
                    head_lines = []
                    for i, line in enumerate(fh):
                        if i >= 6:
                            break
                        head_lines.append(line.rstrip("\n"))
                header_text = "\n".join(head_lines)
                size_mb = stat.st_size / (1024 * 1024)
                msg_html = (
                    f'<p style="font-size:13px;color:#475569;margin:0 0 6px">'
                    f"<b>Cube file:</b> {_html.escape(path.name)} "
                    f"&middot; {size_mb:.2f} MB</p>"
                    '<p style="font-size:12px;color:#64748b;margin:0 0 6px">'
                    "Use the <b>Analysis</b> tab's Orbital Isosurface panel to "
                    "render volumetric data; the raw file is too large to "
                    "preview inline.</p>"
                    '<p style="font-size:11px;color:#94a3b8;margin:6px 0 4px">'
                    "Header (first 6 lines):</p>"
                    '<pre style="white-space:pre-wrap;font-size:11px;'
                    "line-height:1.35;margin:0;background:#f8fafc;padding:6px;"
                    'border-radius:4px">'
                    f"{_html.escape(header_text)}</pre>"
                )
                with self._files_preview_output:
                    display(HTML(msg_html))
                self._set_files_status(f"Cube file metadata: {path.name}")
                return
            except Exception:  # noqa: BLE001 — fall through to text preview
                pass

        is_text = suffix in text_ext
        if not is_text:
            try:
                sample = path.read_bytes()[:512]
            except OSError as exc:
                self._set_files_status(f"Cannot read file: {exc}", "#b91c1c")
                return
            is_text = b"\x00" not in sample

        if not is_text:
            with self._files_preview_output:
                display(
                    HTML(
                        "<p style='font-size:12px;color:#475569;margin:4px 0'>"
                        "Binary preview is not available for this file type."
                        "</p>"
                    )
                )
            self._set_files_status(f"Binary file selected: {path.name}")
            return

        max_bytes = 200_000
        try:
            raw = path.read_bytes()
        except OSError as exc:
            self._set_files_status(f"Cannot read file: {exc}", "#b91c1c")
            return

        truncated = len(raw) > max_bytes
        raw = raw[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
        if truncated:
            text += "\n\n[Preview truncated to 200 KB]"

        with self._files_preview_output:
            display(
                HTML(
                    "<pre style='white-space:pre-wrap;word-break:break-word;"
                    "font-size:12px;line-height:1.35;margin:0'>"
                    f"{_html.escape(text)}"
                    "</pre>"
                )
            )
        self._set_files_status(f"Previewing text file: {path.name}")

    def _on_files_root_changed(self, change) -> None:
        if self._files_updating:
            return
        new_value = str(change.get("new") or "")
        if not new_value:
            return

        new_root = Path(new_value)
        roots = self._files_allowed_roots()
        if not self._is_path_in_allowed_roots(new_root, roots):
            self._set_files_status("Selected root is not allowed.", "#b91c1c")
            return

        self._files_current_dir = new_root
        self._update_files_entries()
        self._set_files_status(f"Root changed to: {new_root}")

    def _on_files_entry_changed(self, change) -> None:
        if self._files_updating:
            return
        new_value = str(change.get("new") or "")
        self._files_selected_path = Path(new_value) if new_value else None
        self._files_open_btn.disabled = self._files_selected_path is None
        if self._files_selected_path is None:
            self._set_files_status("Select a folder or file.")
            return
        if self._files_selected_path.is_dir():
            self._set_files_status(
                f"Folder selected: {self._files_selected_path.name} — click Open to enter."
            )
        else:
            # Auto-preview on selection so the user doesn't need to click Open
            # for every file. Open remains useful for folders.
            self._preview_file_path(self._files_selected_path)

    def _on_files_open(self, _btn) -> None:
        self._activity_begin("Opening selected path...")
        try:
            selected = self._files_selected_path
            if selected is None:
                self._set_files_status("Select a folder or file first.")
                return
            if selected.is_dir():
                self._files_current_dir = selected
                self._update_files_entries()
                self._set_files_status(f"Opened folder: {selected}")
                return
            self._preview_file_path(selected)
        finally:
            self._activity_end()

    def _on_files_up(self, _btn) -> None:
        self._activity_begin("Moving to parent folder...")
        try:
            if self._files_current_dir is None:
                self._set_files_status("No current folder selected.", "#b91c1c")
                return

            parent = self._files_current_dir.parent
            roots = self._files_allowed_roots()
            if parent == self._files_current_dir or not self._is_path_in_allowed_roots(
                parent, roots
            ):
                self._set_files_status("Already at the top of the selected root.")
                return

            self._files_current_dir = parent
            self._update_files_entries()
            self._set_files_status(f"Moved to parent folder: {parent}")
        finally:
            self._activity_end()

    def _on_files_refresh(self, _btn) -> None:
        self._activity_begin("Refreshing files browser...")
        try:
            self._refresh_file_browser()
        finally:
            self._activity_end()

    # ══ CALLBACK METHODS ═════════════════════════════════════════════════════

    # ── Theme ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, change) -> None:
        self._theme_style.clear_output()
        css = self._theme_css(change["new"])
        if css:
            with self._theme_style:
                display(HTML(css))
        self._rerender_plotly_theme()

    def _plotly_theme_colors(self) -> dict:
        """Return plot colors tuned for the current theme.

        The dark theme is a CSS invert+hue-rotate filter on the whole page.
        For SVG/div elements (2D charts): html filter already inverts, so we
        use light values and let the filter make them dark.
        For WebGL canvas (3D scenes): canvas has a counter-filter that cancels
        the html filter, so the color appears as-is — use scene_bgcolor.
        """
        is_dark = self.theme_btn.value == "Dark"
        return {
            "plot_bgcolor": "white",  # html filter darkens this in dark mode
            "paper_bgcolor": "white",  # html filter darkens this in dark mode
            "font_color": "#111827",  # html filter lightens → white text in dark
            "grid_color": "#e5e7eb",  # html filter darkens → subtle grid in dark
            "scene_bgcolor": "#000000" if is_dark else "#ffffff",
        }

    def _apply_plotly_theme(self, fig) -> None:
        """Apply current theme colors to a plotly Figure in-place."""
        tc = self._plotly_theme_colors()
        fig.update_layout(
            plot_bgcolor=tc["plot_bgcolor"],
            paper_bgcolor=tc["paper_bgcolor"],
            font=dict(color=tc["font_color"]),
            xaxis=dict(gridcolor=tc["grid_color"]),
            yaxis=dict(gridcolor=tc["grid_color"]),
        )

    def _set_html_output(self, out: widgets.Output, html: str) -> None:
        """Render HTML into an Output widget via an atomic outputs swap.

        Plotly HTML contains <script> tags. Those scripts do not execute when
        assigned to widgets.HTML.value (innerHTML path), which leads to blank
        figure panels. Routing through ``Output.outputs`` executes the JS.

        The assignment is a single ``out.outputs = (display_data,)`` rather
        than ``clear_output() + append_display_data()`` so the browser never
        observes an intermediate empty state. This eliminates the flicker
        users were seeing on IR Stick/Broadened toggle and FWHM slider drag
        (BUG.9) and matches the atomic-swap pattern already used by
        ``_swap_frame_out`` (trajectory) and ``_swap_vib_output`` (vib).
        """
        if threading.current_thread() is not threading.main_thread():
            io_loop = self._get_kernel_io_loop()
            if io_loop is not None:
                io_loop.add_callback(self._set_html_output, out, html)
                return
        out.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/html": html},
                "metadata": {},
            },
        )

    def _refresh_calc_mol_viewer(self, *, backend: Optional[str] = None) -> None:
        """Re-render the Calculate-tab molecule viewer via an atomic HTML swap.

        Replaces the ``with self.viz_output: display_molecule(...)`` pattern
        that surfaced BUG B1/B2/B3 (2026-05-25 user report):

        - **B1** "viewer doesn't update on PubChem load until I toggle the
          backend" — the Output-context render path was racing the kernel's
          comms flush, so the initial display was sometimes never emitted.
          Atomic ``outputs = (display_data,)`` is a single synchronous
          assignment that the front-end always picks up.
        - **B3** "red log lines around the viewer on the Calculate tab" —
          ``with self.viz_output:`` captured every ``logger.info`` /
          ``logger.error`` line that ``display_molecule`` emitted while it
          ran. ``render_molecule_html`` returns the HTML string OUTSIDE any
          Output context, so the only thing that lands in the widget is
          the viewer itself.
        - **B2** "PlotlyMol valence error spills as red text" — the same
          helper wraps render failures into an inline error <div>, so
          plotlymol's RDKit-bond-perception failure on aromatic systems
          shows up as a friendly inline message instead of a logger.error
          line bleeding through the Output context.

        ``backend`` defaults to ``self._viz_backend`` (user's current
        Calculate-tab toggle); pass an explicit value when the router has
        chosen one (see ``_rerender_viz_for_backend_change``).
        """
        if self._molecule is None or _render_molecule_html is None:
            return
        backend_to_use = backend if backend is not None else self._viz_backend
        html = _render_molecule_html(
            self._molecule,
            backend=backend_to_use,
            style=self._viz_style,
            lighting=self._viz_lighting,
            bgcolor=self._plotly_theme_colors()["scene_bgcolor"],
        )
        self._set_html_output(self.viz_output, html)

    def _get_kernel_io_loop(self) -> Any:
        """Return a cached kernel io_loop, resolving it lazily when needed."""
        io_loop = getattr(self, "_kernel_io_loop", None)
        if io_loop is not None:
            return io_loop
        ip = get_ipython()
        io_loop = getattr(getattr(ip, "kernel", None), "io_loop", None)
        if io_loop is not None:
            self._kernel_io_loop = io_loop
        return io_loop

    def _render_plotly_figure(self, out: widgets.Output, fig) -> None:
        """Render a Plotly figure through display() in an Output widget."""
        self._clear_output_widget(out)
        with out:
            display(fig)

    def _set_plotly_figure_output(self, out: widgets.Output, fig) -> None:
        """Display a Plotly figure via the notebook display pipeline.

        This mirrors the PlotlyMol path (display(fig) in Output) and avoids
        reliance on raw HTML <script> execution inside widget content.
        """
        if threading.current_thread() is threading.main_thread():
            self._render_plotly_figure(out, fig)
        else:
            self._queue_main_thread_callback(self._render_plotly_figure, out, fig)

    def _clear_output_widget(self, out: widgets.Output) -> None:
        """Clear an Output widget in a way that is deterministic in tests."""
        out.clear_output()
        try:
            out.outputs = ()
        except Exception:
            pass

    def _rerender_plotly_theme(self) -> None:
        """Re-render all visible Plotly charts in the updated theme."""
        if getattr(self, "_last_orb_info", None) is not None:
            self._on_orb_range_changed()
        if getattr(self, "_last_ir_freqs", None) is not None:
            self._update_ir_figure(
                self._ir_mode_toggle.value,
                self._ir_fwhm_slider.value,
            )
        if getattr(self, "_last_uv_wavelengths_nm", None):
            self._update_uv_vis_figure(
                self._uv_mode_toggle.value,
                self._uv_fwhm_slider.value,
            )
        _last_pes = getattr(self, "_last_pes_result", None)
        if _last_pes is not None:
            self._show_pes_scan_result(_last_pes)
        # Re-render 3D molecule viewer so scene_bgcolor updates immediately.
        self._refresh_calc_mol_viewer()

    def _initialize_viz_state_from_preference(self) -> None:
        """Align _viz_backend and the three preference widgets with the
        persisted preference. Called at startup before observers are wired."""
        resolved = self._resolve_backend(VizTask.MOLECULE_PREVIEW)
        if resolved is not None:
            self._viz_backend = str(resolved)  # type: ignore[assignment]
            if (
                self.viz_backend_toggle is not None
                and self.viz_backend_toggle.value != str(resolved)
            ):
                self.viz_backend_toggle.value = str(resolved)
            if (
                self.viz_backend_toggle_ana is not None
                and self.viz_backend_toggle_ana.value != str(resolved)
            ):
                self.viz_backend_toggle_ana.value = str(resolved)
        # The Settings widget was already built with viz_default_backend
        # loaded from settings.json — no further alignment needed there.

    def _resolve_backend(self, task: VizTask) -> VizBackend | None:
        """Convenience wrapper: resolve backend for a render task via the
        router using current preference + availability. Returns None if no
        backend is available for the task."""
        decision = select_backend(
            task,
            VizPreference(self._viz_backend_preference),
            self._viz_availability,
        )
        try:
            _calc_log.log_event(
                "viz_route_decision",
                f"task={task} pref={self._viz_backend_preference} "
                f"chosen={decision.chosen} reason={decision.reason}"[:300],
            )
        except OSError:
            pass
        return decision.chosen

    def _set_viz_preference(self, new_pref: str, *, persist: bool) -> None:
        """Single source-of-truth setter for the backend preference.

        ``new_pref`` must be one of "auto" | "py3dmol" | "plotlymol". The
        Settings widget (Status tab) calls this with ``persist=True``; the
        Calculate/Analysis effective toggles call it with ``persist=False``
        (session-only override — clicking either toggle is treated as an
        explicit commit to a concrete preference, even if the prior
        preference was "auto").

        Updates ``self._viz_backend_preference``, resolves the effective
        backend for general static-structure rendering, syncs all three
        widgets under ``_viz_sync_in_progress`` (no echo loops), updates
        lighting-control visibility, and re-renders all visible 3D views.
        """
        if new_pref not in ("auto", "py3dmol", "plotlymol"):
            return
        if new_pref == self._viz_backend_preference:
            return
        self._viz_backend_preference = new_pref

        if persist:
            self._user_settings.viz.default_backend = new_pref
            self._user_settings.save()
            try:
                _calc_log.log_event(
                    "viz_default_backend_changed", f"preference={new_pref}"
                )
            except OSError:
                pass

        # Resolve the effective backend for general static-structure rendering.
        # MOLECULE_PREVIEW is used as the canonical task for the
        # Calculate/Analysis toggle display (all static-structure tasks
        # currently resolve to the same backend per the routing policy).
        resolved = self._resolve_backend(VizTask.MOLECULE_PREVIEW)
        if resolved is not None:
            self._viz_backend = str(resolved)  # type: ignore[assignment]

        # Sync all three widgets under the lock.
        self._viz_sync_in_progress = True
        try:
            if self.viz_default_backend_dd.value != new_pref:
                self.viz_default_backend_dd.value = new_pref
            # Effective toggles can only display concrete values.
            if resolved is not None:
                resolved_str = str(resolved)
                if (
                    self.viz_backend_toggle is not None
                    and self.viz_backend_toggle.value != resolved_str
                ):
                    self.viz_backend_toggle.value = resolved_str
                if (
                    self.viz_backend_toggle_ana is not None
                    and self.viz_backend_toggle_ana.value != resolved_str
                ):
                    self.viz_backend_toggle_ana.value = resolved_str
        finally:
            self._viz_sync_in_progress = False

        # Lighting only works with the PlotlyMol backend.
        _lighting_usable = _PLOTLYMOL_VIZ and self._viz_backend == "plotlymol"
        self.viz_lighting_dd.disabled = not _lighting_usable
        self.viz_lighting_dd.layout.visibility = (
            "visible" if _lighting_usable else "hidden"
        )

        # Re-render all currently-visible 3D molecule viewers via the router.
        self._rerender_3d_views()

    def _rerender_3d_views(self) -> None:
        """Re-render visible 3D molecule viewers using the router to pick a
        backend per task. Updates the "Rendering with: X" label widgets."""
        if _display_molecule is None:
            return

        # Calculate-tab molecule preview (MOLECULE_PREVIEW task).
        if self._molecule is not None:
            chosen = self._resolve_backend(VizTask.MOLECULE_PREVIEW)
            if chosen is not None:
                self._refresh_calc_mol_viewer(backend=str(chosen))

        # Analysis-tab molecule viewer (ANALYSIS_STRUCTURE_VIEW task).
        if self._analysis_displayed_molecule is not None:
            chosen = self._resolve_backend(VizTask.ANALYSIS_STRUCTURE_VIEW)
            if chosen is not None:
                self._analysis_mol_output.clear_output()
                with self._analysis_mol_output:
                    _display_molecule(
                        self._analysis_displayed_molecule,
                        backend=str(chosen),
                        style=self._viz_style,
                        lighting=self._viz_lighting,
                        bgcolor=self._plotly_theme_colors()["scene_bgcolor"],
                    )
                self._update_analysis_backend_label(chosen)

    def _update_analysis_backend_label(self, chosen: VizBackend) -> None:
        """Update the small 'Rendering with: X' label next to the Analysis
        molecule viewer. No-op if the label widget does not exist (built only
        when both backends are available)."""
        label = getattr(self, "viz_backend_label_ana", None)
        if label is None:
            return
        display_name = "py3Dmol" if chosen == VizBackend.PY3DMOL else "plotlymol3d"
        label.value = (
            f'<span style="font-size:11px;color:#94a3b8;font-style:italic">'
            f"Rendering with: {display_name}</span>"
        )

    def _on_viz_backend_changed(self, change) -> None:
        """Calculate-tab toggle observer — explicit override of preference."""
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=False)

    def _on_viz_backend_changed_ana(self, change) -> None:
        """Analysis-tab toggle observer — explicit override of preference."""
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=False)

    def _on_viz_default_backend_changed(self, change) -> None:
        """Settings widget observer — persistent preference change."""
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=True)

    def _on_vib_framerate_changed(self, change) -> None:
        """Persist the vibrational-animation framerate and re-render the
        current mode so the new fps applies immediately. Re-rendering also
        rebuilds the on-disk cache under the new fps key."""
        try:
            new_fps = int(change["new"])
        except (TypeError, ValueError):
            return
        if new_fps == self._user_settings.viz.vib_framerate_fps:
            return
        self._user_settings.viz.vib_framerate_fps = new_fps
        self._user_settings.save()
        try:
            _calc_log.log_event("vib_framerate_changed", f"fps={new_fps}")
        except OSError:
            pass
        # If a vibrational result is currently loaded, re-render the current
        # mode through the new fps so the change is visible immediately.
        if (
            getattr(self, "_last_vib_freq_result", None) is not None
            and getattr(self, "_last_vib_molecule", None) is not None
        ):
            current_mode = self.vib_mode_dd.value
            if current_mode is not None:
                self._on_vib_mode_changed({"new": current_mode})

    def _on_viz_style_changed(self, change) -> None:
        self._viz_style = change["new"]
        self._refresh_calc_mol_viewer()

    def _on_viz_lighting_changed(self, change) -> None:
        self._viz_lighting = change["new"]
        self._refresh_calc_mol_viewer()

    # ── Molecule input ────────────────────────────────────────────────────

    def _refresh_lib_results(self) -> None:
        """Repopulate the library results dropdown from the current filters."""
        category = self.lib_category_dd.value or None
        query = self.lib_search_txt.value.strip()
        opts, note = _bld_library_result_options(query, category)
        # Guard so resetting options/value doesn't fire _on_lib_select.
        self._lib_refreshing = True
        try:
            self.lib_results_dd.options = opts
            self.lib_results_dd.value = ""
        finally:
            self._lib_refreshing = False
        self.lib_count_lbl.value = (
            f'<span style="color:#888;font-size:12px">{note}</span>'
        )

    def _on_lib_filter_changed(self, change) -> None:
        self._refresh_lib_results()

    def _on_lib_select(self, change) -> None:
        if getattr(self, "_lib_refreshing", False):
            return
        entry_id = change["new"]
        if not entry_id:
            return
        entry = _ml.get(entry_id)
        if entry is None:
            return
        self._set_molecule(
            Molecule(
                atoms=entry["atoms"],
                coordinates=entry["coordinates"],
                charge=entry["charge"],
                multiplicity=entry["multiplicity"],
            ),
            entry.get("description") or entry["name"],
        )

    def _on_load_xyz(self, btn) -> None:
        try:
            atoms, coords = parse_xyz_input(self.xyz_area.value.strip())
            mol = Molecule(atoms=atoms, coordinates=coords)
            self._set_molecule(mol, "Loaded from XYZ input")
            self.xyz_msg.value = ""
        except Exception as exc:
            self.xyz_msg.value = f"Parse error: {exc}"

    def _apply_pubchem_search_result(
        self,
        query: str,
        mol: Optional[Molecule] = None,
        error: Optional[Exception] = None,
    ) -> None:
        if error is None and mol is not None:
            self._set_molecule(mol, f"PubChem: {query}")
            self.pubchem_msg.value = f"Loaded {mol.get_formula()} from PubChem."
        else:
            self.pubchem_msg.value = f"Not found: {error}"
            try:
                _calc_log.log_event(
                    "pubchem_search_failed",
                    f"PubChem query not found: '{query}'",
                    query=query,
                    error=str(error)[:200],
                    session_id=self._session_id,
                )
            except Exception:
                pass
        self.pubchem_btn.disabled = False

    def _resolve_and_apply(self, query: str, loop) -> None:
        """Resolve a single query (background thread) and apply on the main loop."""
        try:
            xyz_str, _msg = _student_friendly_resolve(query)
            if xyz_str is None:
                raise ValueError(_msg)
            atoms, coords = parse_xyz_input(xyz_str)
            mol = Molecule(atoms=atoms, coordinates=coords)
            loop.call_soon_threadsafe(
                self._apply_pubchem_search_result, query, mol, None
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                self._apply_pubchem_search_result, query, None, exc
            )

    def _hide_pubchem_candidates(self) -> None:
        """Clear + hide the disambiguation pick-list."""
        self._pubchem_cand_refreshing = True
        try:
            self.pubchem_candidates_dd.options = [("— pick a match —", "")]
            self.pubchem_candidates_dd.value = ""
            self.pubchem_candidates_dd.layout.display = "none"
        finally:
            self._pubchem_cand_refreshing = False

    def _show_pubchem_candidates(self, query: str, candidates: list) -> None:
        """Populate + reveal the pick-list when a query has multiple matches."""
        opts = [(f"pick one of {len(candidates)} matches…", "")]
        for c in candidates:
            mw = c.get("mw") or 0.0
            opts.append(
                (f"{c['title']}  ·  {c['formula']}  ·  {mw:.1f} g/mol", str(c["cid"]))
            )
        self._pubchem_cand_refreshing = True
        try:
            self.pubchem_candidates_dd.options = opts
            self.pubchem_candidates_dd.value = ""
            self.pubchem_candidates_dd.layout.display = ""
        finally:
            self._pubchem_cand_refreshing = False
        self.pubchem_msg.value = (
            f'{len(candidates)} matches for "{query}" — pick one below.'
        )
        self.pubchem_btn.disabled = False

    def _on_pubchem_candidate_selected(self, change) -> None:
        if getattr(self, "_pubchem_cand_refreshing", False):
            return
        cid = change["new"]
        if not cid:
            return
        self.pubchem_msg.value = f"Loading CID {cid}…"
        self.pubchem_btn.disabled = True
        loop = asyncio.get_running_loop()
        threading.Thread(
            target=lambda: self._resolve_and_apply(str(cid), loop), daemon=True
        ).start()

    def _on_search_pubchem(self, btn) -> None:
        query = self.pubchem_txt.value.strip()
        if not query:
            self.pubchem_msg.value = "Enter a molecule name, SMILES, CID, or InChI."
            return
        if _student_friendly_resolve is None:
            self.pubchem_msg.value = "Structure search not available."
            return
        self._hide_pubchem_candidates()
        self.pubchem_msg.value = f'Searching for "{query}"...'
        self.pubchem_btn.disabled = True

        loop = asyncio.get_running_loop()

        def _do():
            try:
                candidates = (
                    _struct_search_candidates(query)
                    if _struct_search_candidates is not None
                    else []
                )
            except Exception:
                candidates = []
            if len(candidates) > 1:
                loop.call_soon_threadsafe(
                    self._show_pubchem_candidates, query, candidates
                )
                return
            # 0 or 1 match → resolve via the full chain (PubChem → CACTUS →
            # offline library), which also handles SMILES/InChI/CID locally.
            self._resolve_and_apply(query, loop)

        threading.Thread(target=_do, daemon=True).start()

    def _on_expand_mol_input(self, btn) -> None:
        _run_on_expand_mol_input(
            self,
            btn,
            visualization_available=VISUALIZATION_AVAILABLE,
        )

    # ── Calc type ─────────────────────────────────────────────────────────

    def _on_calc_type_changed(self, change) -> None:
        _run_on_calc_type_changed(self, change, layout_fn=_layout)

    def _update_scan_widgets(self, _change=None) -> None:
        _run_update_scan_widgets(self, _change)

    def _refresh_freq_seed_options(self) -> None:
        _run_refresh_freq_seed_options(self)

    def _on_freq_seed_changed(self, change) -> None:
        _run_on_freq_seed_changed(self, change)

    def _refresh_tddft_seed_options(self) -> None:
        _run_refresh_tddft_seed_options(self)

    def _on_tddft_seed_changed(self, change) -> None:
        _run_on_tddft_seed_changed(self, change)

    # ── Help buttons ──────────────────────────────────────────────────────

    def _on_method_help(self, btn) -> None:
        _run_on_method_help(self, btn)

    def _on_basis_help(self, btn) -> None:
        _run_on_basis_help(self, btn)

    # ── Run ───────────────────────────────────────────────────────────────

    def _on_run_clicked(self, btn) -> None:
        self._activity_pulse(
            "Queueing calculation...",
            hold_s=0.18,
            kind="compute",
        )
        _run_on_run_clicked(self, btn)

    def _on_solvent_cb_changed(self, change) -> None:
        _run_on_solvent_cb_changed(self, change)

    def _on_clear_log(self, btn) -> None:
        _run_on_clear_log(self, btn)

    # ── Accumulate / export ───────────────────────────────────────────────

    def _on_accumulate(self, btn) -> None:
        _run_on_accumulate(self, btn)

    def _on_clear(self, btn) -> None:
        _run_on_clear(self, btn)

    def _on_export(self, btn) -> None:
        _exp_on_export(self, btn)

    def _on_export_xyz(self, btn) -> None:
        _exp_on_export_xyz(self, btn)

    def _on_export_mol(self, btn) -> None:
        _exp_on_export_mol(self, btn)

    def _on_export_pdb(self, btn) -> None:
        _exp_on_export_pdb(self, btn)

    def _on_iso_export_cube(self, btn) -> None:
        _exp_on_iso_export_cube(self, btn)

    def _on_export_bundle(self, btn) -> None:
        _exp_on_export_bundle(self, btn)

    def _on_ir_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_ir_fig", None),
            stem="ir_spectrum",
            fmt=self._ir_export_fmt_dd.value,
            status_widget=self._ir_export_status,
        )

    def _on_uv_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_uv_fig", None),
            stem="uv_vis_spectrum",
            fmt=self._uv_export_fmt_dd.value,
            status_widget=self._uv_export_status,
        )

    def _on_orb_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_orb_fig", None),
            stem="orbital_energy_diagram",
            fmt=self._orb_export_fmt_dd.value,
            status_widget=self._orb_export_status,
        )

    def _on_vib_export_animation(self, _btn) -> None:
        """Export the active vibrational mode as a self-contained HTML file.

        Backend selection is intentionally decoupled from the user's default
        ``viz.default_backend`` preference: plotlymol3d is preferred for export
        quality, with py3Dmol as a fallback when plotlymol3d is unavailable.
        This is enforced inside ``build_vib_export_html``.
        """
        import re as _re
        from datetime import datetime as _dt

        status = self._vib_export_status

        # Validate vib state before attempting anything else.
        if (
            getattr(self, "_last_vib_freq_result", None) is None
            or getattr(self, "_last_vib_molecule", None) is None
        ):
            status.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                "No vibrational mode loaded — run a Frequency calculation first."
                "</span>"
            )
            return

        try:
            mode_number = int(self.vib_mode_dd.value)
        except (TypeError, ValueError):
            status.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                "No vibrational mode selected.</span>"
            )
            return

        try:
            backend, html_str = _viz_build_vib_export_html(self, mode_number)
        except Exception as exc:
            status.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                f"Export failed: {exc}</span>"
            )
            try:
                _calc_log.log_event(
                    "vib_export_error",
                    f"mode={mode_number} {type(exc).__name__}: {exc}"[:300],
                )
            except Exception:
                pass
            return

        target_dir = (
            self._last_result_dir
            if isinstance(self._last_result_dir, Path)
            else self._get_results_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        formula = getattr(self._last_vib_molecule, "get_formula", lambda: "mol")()
        safe_formula = _re.sub(r"[^A-Za-z0-9_.-]+", "_", formula).strip("_") or "mol"
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest = target_dir / f"vib_{safe_formula}_mode{mode_number}_{ts}.html"

        try:
            dest.write_text(html_str, encoding="utf-8")
        except Exception as exc:
            status.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                f"Write failed: {exc}</span>"
            )
            return

        status.value = (
            '<span style="color:#16a34a;font-size:12px">'
            f"Saved ({backend}): {dest}</span>"
        )
        try:
            _calc_log.log_event(
                "vib_export_done",
                f"mode={mode_number} backend={backend} path={dest}",
            )
        except Exception:
            pass

    def _on_pes_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_pes_fig", None),
            stem="pes_scan_profile",
            fmt=self._pes_export_fmt_dd.value,
            status_widget=self._pes_export_status,
        )

    def _export_plot_figure(
        self,
        *,
        fig: Any,
        stem: str,
        fmt: str,
        status_widget: widgets.HTML,
    ) -> None:
        """Export a plotly figure to HTML or PNG in the current result folder."""
        if fig is None:
            status_widget.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                "No plot available to export yet.</span>"
            )
            return

        import re as _re
        from datetime import datetime as _dt

        import plotly.io as _pio

        target_dir = (
            self._last_result_dir
            if isinstance(self._last_result_dir, Path)
            else self._get_results_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = _re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip()) or "plot"
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = "png" if fmt == "png" else "html"
        dest = target_dir / f"{safe_stem}_{ts}.{ext}"

        try:
            if fmt == "png":
                # Requires kaleido for static image export.
                fig.write_image(str(dest), scale=2)
            else:
                html_str = _pio.to_html(
                    fig,
                    include_plotlyjs=True,
                    full_html=True,
                    config={"responsive": True},
                )
                dest.write_text(html_str, encoding="utf-8")

            status_widget.value = (
                '<span style="color:#16a34a;font-size:12px">' f"Saved: {dest}</span>"
            )
        except Exception as exc:
            msg = str(exc)
            if fmt == "png" and "kaleido" in msg.lower():
                msg = (
                    "PNG export requires kaleido. " "Install with: pip install kaleido"
                )
            status_widget.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                f"Export failed: {msg}</span>"
            )

    @staticmethod
    def _fig_to_csv(fig: Any, *, title: str = "") -> str:
        """Extract per-trace (x, y) pairs from a Plotly figure into CSV text.

        Used by ``_copy_plot_data`` to surface the underlying numerical
        data for every plot panel as a portable CSV. Layout:

        ```
        # <title>
        # <trace name>
        x,y
        <x>,<y>
        ...
        ```

        Multiple traces are emitted as separated sections so the user can
        see (e.g.) Stick + Broadened spectra in one file. Returns the
        empty string if the figure has no extractable data — caller treats
        that as "nothing to copy" rather than writing an empty file.
        (M-EXPORT / EXPORT.4)
        """
        if fig is None:
            return ""
        import io as _io

        out = _io.StringIO()
        if title:
            out.write(f"# {title}\n")
        any_trace = False
        for trace in getattr(fig, "data", []):
            name = getattr(trace, "name", None) or "trace"
            x = getattr(trace, "x", None)
            y = getattr(trace, "y", None)
            if x is None or y is None:
                continue
            out.write(f"\n# {name}\n")
            out.write("x,y\n")
            for xi, yi in zip(x, y):
                out.write(f"{xi},{yi}\n")
            any_trace = True
        return out.getvalue() if any_trace else ""

    def _copy_plot_data(
        self,
        *,
        fig: Any,
        stem: str,
        title: str,
        status_widget: widgets.HTML,
    ) -> None:
        """Write a Plotly figure's data to CSV + try to copy to clipboard.

        Saves ``<stem>_data_<timestamp>.csv`` into the active result
        directory (always works) and emits a JS snippet that copies the
        same CSV to the user's system clipboard via
        ``navigator.clipboard.writeText`` (best-effort — the API requires
        a secure context + user-gesture in some browsers; failures are
        invisible by design). Status widget surfaces the saved path so
        the user can find the file even when clipboard is unavailable.
        (M-EXPORT / EXPORT.4)
        """
        if fig is None:
            status_widget.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                "No plot data to copy yet.</span>"
            )
            return

        csv_text = self._fig_to_csv(fig, title=title)
        if not csv_text:
            status_widget.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                "Figure had no extractable (x, y) traces.</span>"
            )
            return

        import json as _json
        import re as _re
        from datetime import datetime as _dt

        target_dir = (
            self._last_result_dir
            if isinstance(self._last_result_dir, Path)
            else self._get_results_dir()
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = _re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip()) or "plot"
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest = target_dir / f"{safe_stem}_data_{ts}.csv"

        try:
            dest.write_text(csv_text, encoding="utf-8")
        except Exception as exc:
            status_widget.value = (
                '<span style="color:#b91c1c;font-size:12px">'
                f"Write failed: {exc}</span>"
            )
            return

        # Best-effort clipboard copy via the browser's clipboard API.
        # Wrapped in try/catch on the JS side so a permissions error
        # doesn't show up as a Voilà console exception.
        from IPython.display import Javascript, display

        try:
            js_payload = _json.dumps(csv_text)
            display(
                Javascript(
                    "try { navigator.clipboard.writeText("
                    f"{js_payload}); }} catch (e) {{ /* clipboard unavailable */ }}"
                )
            )
        except Exception:
            pass  # Clipboard is best-effort; the file is the canonical artifact.

        status_widget.value = (
            '<span style="color:#16a34a;font-size:12px">'
            f"Saved CSV: {dest} &mdash; copied to clipboard"
            "</span>"
        )

    def _on_ir_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_ir_fig", None),
            stem="ir_spectrum",
            title="IR Spectrum",
            status_widget=self._ir_export_status,
        )

    def _on_uv_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_uv_fig", None),
            stem="uv_vis_spectrum",
            title="UV-Vis Spectrum",
            status_widget=self._uv_export_status,
        )

    def _on_orb_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_orb_fig", None),
            stem="orbital_energy_diagram",
            title="Orbital Energy Diagram",
            status_widget=self._orb_export_status,
        )

    def _on_pes_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_pes_fig", None),
            stem="pes_scan_profile",
            title="PES Scan Profile",
            status_widget=self._pes_export_status,
        )

    def _export_molecule_and_label(self):
        return _exp_export_molecule_and_label(self)

    @staticmethod
    def _molecule_to_rdkit(mol):
        return _exp_molecule_to_rdkit(mol)

    # ── Compare ───────────────────────────────────────────────────────────

    def _on_compare_refresh(self, btn) -> None:
        self._activity_begin("Refreshing comparison choices...")
        try:
            _run_on_compare_refresh(self, btn)
        finally:
            self._activity_end()

    def _on_compare(self, btn) -> None:
        self._activity_begin("Building comparison view...")
        try:
            _run_on_compare(self, btn, layout_fn=_layout)
        finally:
            self._activity_end()

    def _on_compare_clear(self, btn) -> None:
        self._activity_begin("Clearing comparison output...")
        try:
            _run_on_compare_clear(self, btn)
        finally:
            self._activity_end()

    # ── History ───────────────────────────────────────────────────────────

    def _on_past_dd_changed(self, change) -> None:
        _hist_on_past_dd_changed(self, change, layout_fn=_layout)

    def _on_past_refresh(self, btn) -> None:
        self._activity_begin("Refreshing history list...")
        try:
            _run_on_past_refresh(self, btn)
        finally:
            self._activity_end()

    def _on_copy_results_path(self, btn) -> None:
        self._activity_begin("Copying results path...")
        try:
            _run_on_copy_results_path(self, btn)
        finally:
            self._activity_end()

    def _on_view_log(self, btn) -> None:
        self._activity_begin("Loading history log...")
        try:
            _hist_on_view_log(self, btn)
            self._refresh_file_browser()
        finally:
            self._activity_end()

    def _mol_from_result_dir(self, result_dir: Path, data: dict):
        return _hist_mol_from_result_dir(result_dir, data)

    def _history_load_results(
        self, data: dict, result_dir: Path, *, source_btns: tuple = ()
    ) -> None:
        # Activity indicator + button-disable feedback are handled inside the
        # inner ``history_load_results`` helper now (HIST.1). The wrapper just
        # forwards source_btns and refreshes the Files tab after the load.
        try:
            _hist_history_load_results(self, data, result_dir, source_btns=source_btns)
        finally:
            self._refresh_file_browser()

    def _history_load_analysis(
        self, result_dir: Path, *, source_btns: tuple = ()
    ) -> None:
        try:
            _hist_history_load_analysis(self, result_dir, source_btns=source_btns)
        finally:
            self._refresh_file_browser()

    def _build_history_context(self, result_dir: Path) -> Optional[_AnalysisContext]:
        return _hist_build_history_context(result_dir, context_cls=_AnalysisContext)

    # ── Perf stats reset ──────────────────────────────────────────────────

    def _on_reset_click(self, btn) -> None:
        _run_on_reset_click(self, btn)

    def _on_confirm_yes(self, btn) -> None:
        _run_on_confirm_yes(self, btn, reset_perf_log_fn=_calc_log.reset_perf_log)

    def _on_confirm_no(self, btn) -> None:
        _run_on_confirm_no(self, btn)

    # ── Calibration ───────────────────────────────────────────────────────

    def _on_cal_run(self, btn) -> None:
        _run_on_cal_run(
            self,
            btn,
            benchmark_suite=_BENCHMARK_SUITE,
            benchmark_suite_long=_BENCHMARK_SUITE_LONG,
        )

    def _on_cal_stop(self, btn) -> None:
        _run_on_cal_stop(self, btn)

    def _on_cal_skip(self, btn) -> None:
        _run_on_cal_skip(self, btn)

    def _do_calibration(self) -> None:
        _run_do_calibration(self, pyscf_available=_PYSCF_AVAILABLE)

    # ── Output log ────────────────────────────────────────────────────────

    def _on_log_clear(self, btn) -> None:
        _run_on_log_clear(self, btn)

    # ── Issue reporting ───────────────────────────────────────────────────

    def _on_issue_btn(self, _=None) -> None:
        _run_on_issue_btn(self, _)

    def _on_issue_cancel(self, _=None) -> None:
        _run_on_issue_cancel(self, _)

    def _on_issue_submit(self, _=None) -> None:
        _run_on_issue_submit(self, issue_tracker_mod=_issue_tracker)

    def _build_issue_context(self) -> dict:
        """Snapshot the current app state to attach to an issue report."""
        ctx: dict = {}
        if self._molecule is not None:
            try:
                ctx["molecule"] = {
                    "formula": self._molecule.get_formula(),
                    "n_atoms": len(self._molecule.atoms),
                    "charge": self._molecule.charge,
                    "multiplicity": self._molecule.multiplicity,
                }
            except Exception:
                pass
        try:
            ctx["settings"] = {
                "method": self.method_dd.value,
                "basis": self.basis_dd.value,
                "calc_type": self.calc_type_dd.value,
                "last_calc_type": getattr(self, "_last_calc_type", None),
            }
        except Exception:
            pass
        if self._last_result is not None:
            try:
                ctx["last_result"] = {
                    "formula": getattr(self._last_result, "formula", None),
                    "method": getattr(self._last_result, "method", None),
                    "basis": getattr(self._last_result, "basis", None),
                    "converged": getattr(self._last_result, "converged", None),
                    "energy_hartree": getattr(
                        self._last_result, "energy_hartree", None
                    ),
                }
            except Exception:
                pass
        try:
            _all_ev = _calc_log.get_recent_events(60)
            # Always include the 10 most recent non-startup events so that calc
            # events are not starved out by a burst of startup entries (e.g.
            # rapid notebook restarts).  Merge with the 5 most recent events of
            # any type to preserve immediate context, then re-sort by timestamp.
            _non_startup = [e for e in _all_ev if e.get("event") != "startup"]
            _keep_ids = {id(e) for e in _non_startup[-10:]} | {
                id(e) for e in _all_ev[-5:]
            }
            ctx["recent_events"] = [e for e in _all_ev if id(e) in _keep_ids]
        except Exception:
            pass
        return ctx

    # ── Clear log cache ───────────────────────────────────────────────────

    def _on_clear_log_cache(self, _=None) -> None:
        _run_on_clear_log_cache(self, _)

    def _on_clear_log_cache_confirm(self, _=None) -> None:
        _run_on_clear_log_cache_confirm(self, calc_log_mod=_calc_log)

    # ── Exit ──────────────────────────────────────────────────────────────

    def _on_exit_clicked(self, _=None) -> None:
        _run_on_exit_clicked(self, _)

    # ── Help ──────────────────────────────────────────────────────────────

    def _on_help_toggle(self, _=None) -> None:
        _run_on_help_toggle(self, _)

    def _on_help_topic_changed(self, change=None) -> None:
        _run_on_help_topic_changed(self, change)

    # ══ LOGIC METHODS ════════════════════════════════════════════════════════

    def _set_molecule(self, mol: Molecule, label: str = "") -> None:
        """Update shared state and refresh dependent widgets."""
        self._molecule = mol
        self.run_btn.disabled = False
        self.export_btn.disabled = False
        self.export_xyz_btn.disabled = False
        self.export_mol_btn.disabled = not _RDKIT_AVAILABLE
        self.export_pdb_btn.disabled = not _RDKIT_AVAILABLE

        try:
            _calc_log.log_event(
                "molecule_load",
                f"{mol.get_formula()} — {label or 'unknown source'}",
                formula=mol.get_formula(),
                n_atoms=len(mol.atoms),
                charge=mol.charge,
                multiplicity=mol.multiplicity,
                source=label or "unknown",
                session_id=self._session_id,
            )
        except Exception:
            pass

        try:
            n_e = mol.get_electron_count()
            e_str = f"{n_e} electrons"
        except Exception:
            e_str = ""

        _lbl = f'<br><small style="color:#777">{label}</small>' if label else ""
        _summary = (
            f'<b style="font-size:15px">{mol.get_formula()}</b>'
            f'&ensp;<span style="color:#555;font-size:13px">'
            f"{len(mol.atoms)} atoms"
            + (f" &bull; {e_str}" if e_str else "")
            + f" &bull; charge {mol.charge} &bull; mult {mol.multiplicity}"
            + f"</span>{_lbl}"
        )
        self.mol_info_html.value = _summary
        self.mol_summary_compact.value = (
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
            f'border-radius:6px;padding:7px 14px;font-size:14px;display:inline-block">'
            f"{_summary}</div>"
        )

        self.charge_si.value = mol.charge
        self.mult_si.value = mol.multiplicity
        if mol.multiplicity > 1 and self.method_dd.value == "RHF":
            self.method_dd.value = "UHF"

        # BUG B1/B2/B3 (2026-05-25): route through ``_refresh_calc_mol_viewer``
        # so the viewer renders via an atomic outputs swap rather than the
        # ``with self.viz_output: display(...)`` pattern that the BUG.7 fix
        # already replaced for the Analysis tab. The molecule attribute on
        # the app was set just above; the helper reads it.
        self._refresh_calc_mol_viewer()

        self._update_notes()

        # Advance step indicator
        if self.step_progress._states[2] != "active":
            if self.step_progress._states[2] in ("done", "fail"):
                self.step_progress.reset()
            self.step_progress.complete(0)
            self.step_progress.start(1)

        self._update_estimate()

        # Collapse molecule input to compact view
        _collapsed_children = [self.mol_input_collapsed, self.viz_output]
        if self.viz_backend_toggle is not None:
            _collapsed_children.append(self.viz_backend_toggle)
        if VISUALIZATION_AVAILABLE:
            _collapsed_children.append(self.viz_controls_box)
        self.mol_input_container.children = _collapsed_children

        # Re-filter seed-geometry dropdowns (Freq + UV-Vis) to only include
        # prior geo-opts of the now-active molecule (formula match). Best-
        # effort: failures must not block molecule loading.
        try:
            self._refresh_freq_seed_options()
        except Exception:
            pass
        try:
            self._refresh_tddft_seed_options()
        except Exception:
            pass

    def _queue_main_thread_callback(self, callback, *args, **kwargs) -> None:
        """Run a callback on the notebook/kernel thread when possible."""
        if threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
            return

        io_loop = self._get_kernel_io_loop()
        if io_loop is not None:
            io_loop.add_callback(callback, *args, **kwargs)
            return

        # Best-effort fallback for non-notebook contexts where no kernel loop
        # is available. This preserves existing behaviour, but the normal
        # notebook path above keeps rendering off the worker thread.
        callback(*args, **kwargs)

    def _install_run_output_scroll_guard(self) -> None:
        """Install a JS guard that preserves live-log scroll behavior.

        Two parts (BUG-SCROLL, reopened 2026-06-08):

        1. Disable browser **scroll-anchoring** (``overflow-anchor: none``) on the
           log and its scrollable ancestors — appending a line otherwise makes
           the browser nudge the *page* scroll, the "screen jumps up on each new
           output line" symptom.
        2. Keep the internal log pinned to the bottom while the user is already
           at the bottom, and preserve manual scroll-up.
        """
        if self._run_output_scroll_guard_installed:
            return

        js_code = r"""
(() => {
    const ROOT_CLASS = "quantui-run-output";
    const ROOT_MARK = "data-quantui-run-scroll-guard";

    function selectScroller(root) {
        const candidates = [
            root,
            ...root.querySelectorAll(
                ".jp-OutputArea-output, .output_scroll, .jupyter-widgets-output-area, .output_subarea"
            ),
        ];
        for (const el of candidates) {
            const style = window.getComputedStyle(el);
            const overflowY = (style && style.overflowY) || "";
            const canScroll = /auto|scroll/.test(overflowY);
            if (canScroll || el.scrollHeight > el.clientHeight + 2) {
                return el;
            }
        }
        return root;
    }

    function installForRoot(root) {
        if (!root || root.getAttribute(ROOT_MARK) === "1") {
            return;
        }

        const scroller = selectScroller(root);
        if (!scroller) {
            return;
        }

        root.setAttribute(ROOT_MARK, "1");

        // Disable browser scroll-anchoring on the log and every scrollable
        // ancestor (incl. the page scroller). Without this, appending a line
        // inside the log makes the browser nudge the *page* scroll to keep an
        // anchor element stable — the user-visible "screen jumps up" on each
        // new output line (BUG-SCROLL).
        try {
            scroller.style.overflowAnchor = "none";
            for (let el = root; el && el !== document.documentElement; el = el.parentElement) {
                el.style.overflowAnchor = "none";
            }
            document.documentElement.style.overflowAnchor = "none";
            if (document.body) document.body.style.overflowAnchor = "none";
        } catch (e) { /* styling best-effort */ }

        const thresholdPx = 24;
        let stickToBottom = true;
        let lastScrollHeight = scroller.scrollHeight;

        const updateStickFlag = () => {
            const dist = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
            stickToBottom = dist <= thresholdPx;
        };

        const onMutation = () => {
            // A large shrink means the log was cleared for a new run — re-arm
            // "follow". Without this, stickToBottom stays false from a previous
            // manual scroll-up and the new run's output streams while the log
            // sits at the top (BUG-SCROLL, confirmed via DevTools: scrollTop
            // stuck at 0 while content grew).
            if (scroller.scrollHeight < lastScrollHeight - thresholdPx) {
                stickToBottom = true;
            }
            lastScrollHeight = scroller.scrollHeight;
            if (stickToBottom) {
                scroller.scrollTop = scroller.scrollHeight;
            }
        };

        scroller.addEventListener("scroll", updateStickFlag, { passive: true });

        const obs = new MutationObserver(onMutation);
        obs.observe(root, { childList: true, subtree: true, characterData: true });

        updateStickFlag();
        onMutation();
    }

    function scanAndInstall() {
        const roots = document.querySelectorAll(`.${ROOT_CLASS}`);
        roots.forEach(installForRoot);
    }

    scanAndInstall();

    const bodyObserver = new MutationObserver(() => {
        scanAndInstall();
    });
    bodyObserver.observe(document.body, { childList: true, subtree: true });
})();
"""

        try:
            with self._exit_output:
                display(Javascript(js_code))
            self._run_output_scroll_guard_installed = True
        except Exception:
            # Non-notebook contexts may not support JS display; fail silently.
            self._run_output_scroll_guard_installed = False

    def _set_molecule_state_only(self, mol) -> None:
        """Apply only thread-safe molecule state updates."""
        self._molecule = mol

    def _set_molecule_threadsafe(self, mol, status_message: str) -> None:
        """Update molecule state safely and render on the main thread only."""
        if threading.current_thread() is threading.main_thread():
            self._set_molecule(mol, status_message)
            return

        self._set_molecule_state_only(mol)
        self._queue_main_thread_callback(self._set_molecule, mol, status_message)

    def _show_result_3d(self, molecule, extra_output=None) -> None:
        _viz_show_result_3d(
            self,
            molecule,
            extra_output,
            render_html_fn=_render_molecule_html,
        )

    def _show_result_log(self, saved_dir: Path, log_text: str) -> None:
        """Populate the result-directory label and output-log accordion.

        Safe to call from a background thread.
        """
        # Path label
        self._result_dir_label.value = (
            f'<span style="font-size:12px;color:#555;font-family:monospace">'
            f"Saved to: {saved_dir}</span>"
        )
        self._result_dir_label.layout.display = ""

        # Log accordion — prefer on-disk file (written by save_result) over in-memory string
        import html as _html_mod

        _log_path = saved_dir / "pyscf.log"
        try:
            log_content = _log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_content = log_text

        if not log_content.strip():
            log_content = "(No output captured for this calculation.)"

        self._result_log_output.clear_output()
        with self._result_log_output:
            display(
                HTML(
                    f'<pre style="font-size:11px;max-height:400px;overflow-y:auto;'
                    f'white-space:pre-wrap;word-break:break-all;margin:0;padding:6px">'
                    f"{_html_mod.escape(log_content)}</pre>"
                )
            )
        self._result_log_accordion.layout.display = ""

    def _on_traj_expand(self, change) -> None:
        _viz_on_traj_expand(self, change)

    def _show_opt_trajectory(
        self, opt_result, render_token: Optional[int] = None
    ) -> None:
        _viz_show_opt_trajectory(
            self,
            opt_result,
            layout_fn=_layout,
            render_token=render_token,
        )

    def _traj_step_html(self, step: int, traj, energies, rel_e) -> str:
        return _viz_traj_step_html(self, step, traj, energies, rel_e)

    def _render_traj_frame(self, molecule, output_widget) -> None:
        _viz_render_traj_frame(self, molecule, output_widget)

    def _build_vib_data_from_freq_result(self, freq_result, molecule):
        return _viz_build_vib_data_from_freq_result(self, freq_result, molecule)

    def _build_vib_data_inner(
        self, freq_result, molecule, np, VibrationalData, VibrationalMode
    ):
        return _viz_build_vib_data_inner(
            self, freq_result, molecule, np, VibrationalData, VibrationalMode
        )

    def _show_vib_animation(self, freq_result, molecule) -> bool:
        return _viz_show_vib_animation(self, freq_result, molecule)

    def _show_ir_spectrum(self, freq_result) -> bool:
        return _viz_show_ir_spectrum(self, freq_result)

    def _wire_ir_controls(self) -> None:
        _viz_wire_ir_controls(self)

    def _on_ir_mode_changed(self, change) -> None:
        _viz_on_ir_mode_changed(self, change)

    def _on_ir_fwhm_changed(self, change) -> None:
        _viz_on_ir_fwhm_changed(self, change)

    def _update_ir_figure(self, mode: str, fwhm: float) -> None:
        _viz_update_ir_figure(self, mode, fwhm)

    def _show_uv_vis_spectrum(
        self,
        energies_ev: list[float],
        oscillator_strengths: list[float],
        wavelengths_nm: list[float],
    ) -> bool:
        return _viz_show_uv_vis_spectrum(
            self,
            energies_ev,
            oscillator_strengths,
            wavelengths_nm,
        )

    def _wire_uv_controls(self) -> None:
        _viz_wire_uv_controls(self)

    def _on_uv_mode_changed(self, change) -> None:
        _viz_on_uv_mode_changed(self, change)

    def _on_uv_fwhm_changed(self, change) -> None:
        _viz_on_uv_fwhm_changed(self, change)

    def _update_uv_vis_figure(self, mode: str, fwhm: float) -> None:
        _viz_update_uv_vis_figure(self, mode, fwhm)

    def _show_orbital_diagram(self, result) -> bool:
        return _viz_show_orbital_diagram(self, result)

    def _on_iso_generate(self, btn) -> None:
        _viz_on_iso_generate(self, btn)

    def _on_orb_range_changed(self, _change=None) -> None:
        _viz_on_orb_range_changed(self, _change)

    def _render_orbital_isosurface(
        self, orbital_label: str, render_token: Optional[int] = None
    ) -> None:
        _viz_render_orbital_isosurface(
            self,
            orbital_label,
            render_token=render_token,
        )

    def _render_vib_mode(
        self,
        vib_data,
        molecule,
        mode_number: int,
        *,
        render_token: Optional[int] = None,
    ) -> None:
        _viz_render_vib_mode(
            self, vib_data, molecule, mode_number, render_token=render_token
        )

    def _on_vib_mode_changed(self, change) -> None:
        _viz_on_vib_mode_changed(self, change)

    def _update_vib_nav_buttons(self, change=None) -> None:
        """Enable/disable prev/next vib mode buttons based on current
        dropdown position. Called on both ``value`` and ``options`` changes
        so the buttons stay correct after a new freq result populates the
        options list."""
        opts = self.vib_mode_dd.options or ()
        if not opts:
            self.vib_prev_btn.disabled = True
            self.vib_next_btn.disabled = True
            return
        cur = self.vib_mode_dd.value
        idx = next(
            (i for i, (_lbl, num) in enumerate(opts) if num == cur),
            -1,
        )
        self.vib_prev_btn.disabled = idx <= 0
        self.vib_next_btn.disabled = idx < 0 or idx >= len(opts) - 1

    def _on_vib_prev_clicked(self, _btn) -> None:
        opts = self.vib_mode_dd.options or ()
        cur = self.vib_mode_dd.value
        for i, (_lbl, num) in enumerate(opts):
            if num == cur and i > 0:
                self.vib_mode_dd.value = opts[i - 1][1]
                return

    def _on_vib_next_clicked(self, _btn) -> None:
        opts = self.vib_mode_dd.options or ()
        cur = self.vib_mode_dd.value
        for i, (_lbl, num) in enumerate(opts):
            if num == cur and i < len(opts) - 1:
                self.vib_mode_dd.value = opts[i + 1][1]
                return

    def _do_run(self) -> None:
        """Main calculation dispatch — runs in a background thread."""
        mol = self._molecule
        if mol is None:
            self.run_status.value = "Load a molecule first."
            return
        self._activity_begin(
            "Running compute operations...",
            kind="compute",
        )
        self.run_btn.disabled = True
        self.run_status.value = "Starting..."

        self.step_progress.complete(1)
        self.step_progress.start(2)

        _calc_log.log_event(
            "calc_start",
            f"{self.method_dd.value}/{self.basis_dd.value} on {mol.get_formula()}",
            n_atoms=len(mol.atoms),
        )
        _run_wall_t = time.perf_counter()
        _run_cpu_t = time.process_time()
        _scf_converged_t: Optional[float] = None
        _tail_marks: dict[str, float] = {}

        # M-EST / EST.6 (2026-05-25): capture the estimator's pre-run
        # prediction so we can write a (predicted, actual) record to
        # ``prediction_log.jsonl`` after the calc completes. The
        # estimator may return None (insufficient history); we record
        # that as "no estimate" so the dashboard counts it separately
        # from "estimate was wrong by N%".
        _predicted_run_s: Optional[float] = None
        _predicted_run_confidence: str = "unknown"
        try:
            _ct_for_est = {
                "Single Point": "single_point",
                "Geometry Opt": "geometry_opt",
                "Frequency": "frequency",
                "UV-Vis (TD-DFT)": "tddft",
                "NMR Shielding": "nmr",
                "PES Scan": "pes_scan",
            }.get(self.calc_type_dd.value, "single_point")
            _nb_for_est = _calc_log.count_basis_functions(
                mol.atoms, self.basis_dd.value
            )
            # Match _update_estimate's GPU-prediction logic so the
            # recorded predicted_s is what the user SAW in the UI
            # before they hit Run.
            _predicted_gpu_used: Optional[bool] = None
            try:
                from quantui.gpu_offload import (
                    _GPU_UNSUPPORTED_METHODS as _GPU_NO,
                )
                from quantui.gpu_offload import (
                    is_gpu_available,
                )

                _gpu_avail, _ = is_gpu_available()
                if _gpu_avail and self.method_dd.value.upper() not in _GPU_NO:
                    _predicted_gpu_used = True
                else:
                    _predicted_gpu_used = False
            except Exception:  # noqa: BLE001 — fall back to device-agnostic
                _predicted_gpu_used = None

            _est = _calc_log.estimate_time(
                n_atoms=len(mol.atoms),
                n_electrons=mol.get_electron_count(),
                method=self.method_dd.value,
                basis=self.basis_dd.value,
                n_basis=_nb_for_est,
                calc_type=_ct_for_est,
                gpu_used=_predicted_gpu_used,
            )
            if _est is not None:
                _predicted_run_s = float(_est["seconds"])
                _predicted_run_confidence = str(_est.get("confidence", "unknown"))
        except Exception as _est_exc:
            # Estimator failure here is non-fatal — we just won't have a
            # predicted_s to compare against. Log to event_log so the
            # cause is at least surfaced for diagnosis.
            try:
                _calc_log.log_event(
                    "predict_capture_failed",
                    f"{type(_est_exc).__name__}: {_est_exc}"[:300],
                )
            except Exception:  # noqa: BLE001 — telemetry self-guard
                pass

        def _mark(stage: str) -> None:
            _tail_marks[stage] = time.perf_counter()

        def _span(stage_a: str, stage_b: str) -> Optional[float]:
            if stage_a not in _tail_marks or stage_b not in _tail_marks:
                return None
            return round(_tail_marks[stage_b] - _tail_marks[stage_a], 3)

        def _on_scf_converged() -> None:
            nonlocal _scf_converged_t
            if _scf_converged_t is None:
                _scf_converged_t = time.perf_counter()

        def _run_required_final_single_point(target_mol, reason: str):
            """Run a required post-optimization single point on target geometry."""
            from quantui import run_in_session

            _solvent = self.solvent_dd.value if self.solvent_cb.value else None
            self.run_status.value = (
                "Running required single-point on optimized geometry..."
            )
            log.write(
                f"\n-- Required single-point ({reason}) "
                "on optimized geometry --------------------------------\n"
            )
            sp_result = run_in_session(
                molecule=target_mol,
                method=self.method_dd.value,
                basis=self.basis_dd.value,
                progress_stream=log,  # type: ignore[arg-type]
                solvent=_solvent,
            )
            if not bool(getattr(sp_result, "converged", False)):
                raise RuntimeError(
                    "Required post-optimization single-point did not converge."
                )
            log.write("Required single-point converged on optimized geometry.\n")
            return sp_result

        log = _LogCapture(
            self.run_output,
            self.run_status,
            on_scf_converged=_on_scf_converged,
        )

        # Write structured log header immediately so it appears at the top of output
        try:
            from quantui.log_utils import format_log_header as _fmt_log_hdr

            _hdr_calc_type = {
                "Geometry Opt": "geometry_opt",
                "Frequency": "frequency",
                "UV-Vis (TD-DFT)": "tddft",
                "NMR Shielding": "nmr",
                "PES Scan": "pes_scan",
            }.get(self.calc_type_dd.value, "single_point")
            log.write(
                _fmt_log_hdr(
                    formula=mol.get_formula(),
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    calc_type=_hdr_calc_type,
                )
            )
        except Exception:
            pass

        try:
            calc_mol = mol
            if self.preopt_cb.value and _PREOPT_AVAILABLE:
                self.run_status.value = "Pre-optimizing..."
                calc_mol, _rmsd = preoptimize(mol)
                self._set_molecule_threadsafe(
                    calc_mol,
                    f"Geometry pre-optimized (LJ, RMSD={_rmsd:.3f} Å)",
                )

            ct = self.calc_type_dd.value
            result: Any = None
            result_html: str = ""
            save_spectra: dict = {}
            save_type: str = "single_point"
            _pre_opt: Any = None  # OptimizationResult from Frequency pre-opt step

            # Optional QM geometry optimization before non-frequency workflows.
            # Frequency and UV-Vis (TD-DFT) both have dedicated seed/pre-opt
            # handling in their own branches so they can layer a seed
            # geometry under the pre-opt step.
            if self._freq_preopt_cb.value and ct not in (
                "Geometry Opt",
                "Frequency",
                "UV-Vis (TD-DFT)",
            ):
                from quantui import optimize_geometry

                # POLISH.9 (M-POLISH, 2026-05-25): rename user-facing
                # "Pre-optimisation" → "Geometry optimization". The
                # wrapped operation is the full DFT geom-opt at the
                # user's selected method/basis — same code path as the
                # standalone Geometry Opt calc-type. The LJ classical
                # pre-opt earlier (around line 3488) keeps its name.
                self.run_status.value = f"Optimizing geometry before {ct}…"
                log.write(
                    f"\n── Geometry optimization (before {ct}) "
                    f"────────────────────────────\n"
                )
                # BUG C (2026-05-25): catch numerical failures (e.g.
                # singular matrix in cho_solve on tight rings) and fall
                # back to the user's input geometry rather than killing
                # the whole calc.
                try:
                    _pre_opt = optimize_geometry(
                        molecule=calc_mol,
                        method=self.method_dd.value,
                        basis=self.basis_dd.value,
                        progress_stream=log,  # type: ignore[arg-type]
                    )
                    calc_mol = _pre_opt.molecule
                    _conv_str = (
                        "converged" if _pre_opt.converged else "did NOT fully converge"
                    )
                    log.write(
                        f"\nGeometry optimization {_conv_str} in {_pre_opt.n_steps} steps."
                        f"  E = {_pre_opt.energies_hartree[-1]:.8f} Ha\n\n"
                    )
                    if not _pre_opt.converged:
                        log.write(
                            "⚠ Geometry optimization did not fully converge — "
                            "proceeding with best available geometry.\n\n"
                        )
                    if ct != "Single Point":
                        _run_required_final_single_point(
                            calc_mol,
                            f"after geometry optimization before {ct}",
                        )
                except Exception as _pre_exc:
                    log.write(
                        f"\n⚠ Geometry optimization failed: {_pre_exc}\n"
                        "  Proceeding with the user-provided geometry "
                        "as-is.\n\n"
                    )

            if ct == "Geometry Opt":
                self.run_status.value = "Optimizing geometry..."
                from quantui import optimize_geometry

                result = optimize_geometry(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    fmax=self.fmax_fi.value,
                    steps=self.max_steps_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                _sp_result = _run_required_final_single_point(
                    result.molecule,
                    "after geometry optimisation",
                )
                _sp_energy = getattr(_sp_result, "energy_hartree", None)
                if (
                    isinstance(getattr(result, "energies_hartree", None), list)
                    and result.energies_hartree
                    and isinstance(_sp_energy, (int, float))
                ):
                    result.energies_hartree[-1] = float(_sp_energy)
                result.converged = bool(result.converged) and bool(
                    getattr(_sp_result, "converged", False)
                )
                result.mo_energy_hartree = getattr(
                    _sp_result,
                    "mo_energy_hartree",
                    result.mo_energy_hartree,
                )
                result.mo_occ = getattr(_sp_result, "mo_occ", result.mo_occ)
                result.mo_coeff = getattr(_sp_result, "mo_coeff", result.mo_coeff)
                result.pyscf_mol_atom = getattr(
                    _sp_result,
                    "pyscf_mol_atom",
                    result.pyscf_mol_atom,
                )
                result.pyscf_mol_basis = getattr(
                    _sp_result,
                    "pyscf_mol_basis",
                    result.pyscf_mol_basis,
                )
                result_html = self._format_opt_result(result)
                save_spectra, save_type = {}, "geometry_opt"
            elif ct == "Frequency":
                from quantui.freq_calc import run_freq_calc

                # ── Step 1: resolve seed geometry ─────────────────────────────
                _seed_path = self._freq_seed_dd.value
                if _seed_path:
                    from quantui.results_storage import load_trajectory

                    self.run_status.value = "Loading seed geometry from history…"
                    _seed_traj, _ = load_trajectory(Path(_seed_path))
                    calc_mol = _seed_traj[-1]
                    log.write(
                        f"\nSeed geometry loaded from: {Path(_seed_path).name}\n"
                        f"  Formula: {calc_mol.get_formula()}  "
                        f"Atoms: {len(calc_mol.atoms)}\n\n"
                    )

                # ── Step 2: optional geometry optimization ────────────────────
                #
                # POLISH.9 (M-POLISH, 2026-05-25): renamed from
                # "pre-optimisation" — the wrapped operation is a full
                # DFT geometry optimization at the user's selected
                # method/basis. The LJ-classical pre-opt is in
                # quantui/preopt.py and keeps its "pre-opt" name.
                #
                # BUG C (2026-05-25): geom-opt can hit a singular matrix
                # in PySCF's ``cho_solve`` on tight rings (e.g. aromatic
                # benzene with B3LYP/6-31G). That raises out of the
                # optimizer and used to kill the whole calc. Wrap it: on
                # any failure log to the user log, keep ``calc_mol`` as
                # the input geometry, and proceed to the freq analysis —
                # the user can iterate if their input was actually wrong.
                if self._freq_preopt_cb.value:
                    from quantui import optimize_geometry

                    self.run_status.value = "Optimizing geometry before frequency…"
                    log.write(
                        "\n── Geometry optimization (before frequency analysis) ──────────────────\n"
                    )
                    try:
                        _pre_opt = optimize_geometry(
                            molecule=calc_mol,
                            method=self.method_dd.value,
                            basis=self.basis_dd.value,
                            progress_stream=log,  # type: ignore[arg-type]
                        )
                        calc_mol = _pre_opt.molecule
                        _conv_str = (
                            "converged"
                            if _pre_opt.converged
                            else "did NOT fully converge"
                        )
                        log.write(
                            f"\nGeometry optimization {_conv_str} in {_pre_opt.n_steps} steps."
                            f"  E = {_pre_opt.energies_hartree[-1]:.8f} Ha\n\n"
                        )
                        if not _pre_opt.converged:
                            log.write(
                                "⚠ Geometry optimization did not fully converge — "
                                "proceeding with best available geometry.\n\n"
                            )
                        _run_required_final_single_point(
                            calc_mol,
                            "after geometry optimization before frequency",
                        )
                    except Exception as _pre_exc:
                        log.write(
                            f"\n⚠ Geometry optimization failed: {_pre_exc}\n"
                            "  Proceeding with the user-provided geometry "
                            "as-is; if the molecule was already near a "
                            "stationary point this is usually fine.\n\n"
                        )

                # ── Step 3: frequency analysis ────────────────────────────────
                self.run_status.value = "Computing frequencies (SCF + Hessian)…"
                result = run_freq_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_freq_result(result)
                _displacements_serialized = None
                if result.displacements is not None:
                    try:
                        import numpy as _np_d

                        _displacements_serialized = _np_d.asarray(
                            result.displacements
                        ).tolist()
                    except Exception:
                        pass
                save_spectra = {
                    "ir": {
                        "frequencies_cm1": result.frequencies_cm1,
                        "ir_intensities": result.ir_intensities,
                        "zpve_hartree": result.zpve_hartree,
                        "displacements": _displacements_serialized,
                    },
                    "molecule": {
                        "atoms": list(calc_mol.atoms),
                        "coords": [
                            list(map(float, row)) for row in calc_mol.coordinates
                        ],
                        "charge": calc_mol.charge,
                        "multiplicity": calc_mol.multiplicity,
                    },
                }
                save_type = "frequency"
            elif ct == "UV-Vis (TD-DFT)":
                from quantui.tddft_calc import run_tddft_calc

                # ── Step 1: resolve seed geometry ─────────────────────────────
                _tddft_seed_path = self._tddft_seed_dd.value
                if _tddft_seed_path:
                    from quantui.results_storage import load_trajectory

                    self.run_status.value = "Loading seed geometry from history…"
                    _seed_traj, _ = load_trajectory(Path(_tddft_seed_path))
                    calc_mol = _seed_traj[-1]
                    log.write(
                        f"\nSeed geometry loaded from: {Path(_tddft_seed_path).name}\n"
                        f"  Formula: {calc_mol.get_formula()}  "
                        f"Atoms: {len(calc_mol.atoms)}\n\n"
                    )

                # ── Step 2: optional geometry optimization ────────────────────
                # POLISH.9 (M-POLISH, 2026-05-25): renamed from
                # "pre-optimisation" — DFT geom-opt is just geom-opt.
                if self._freq_preopt_cb.value:
                    from quantui import optimize_geometry

                    self.run_status.value = (
                        "Optimizing geometry before UV-Vis (TD-DFT)…"
                    )
                    log.write(
                        "\n── Geometry optimization (before UV-Vis (TD-DFT)) "
                        "─────────────\n"
                    )
                    # BUG C (2026-05-25): catch numerical failures and
                    # fall back to the user's seed geometry rather than
                    # killing the whole TD-DFT calc.
                    try:
                        _pre_opt = optimize_geometry(
                            molecule=calc_mol,
                            method=self.method_dd.value,
                            basis=self.basis_dd.value,
                            progress_stream=log,  # type: ignore[arg-type]
                        )
                        calc_mol = _pre_opt.molecule
                        _conv_str = (
                            "converged"
                            if _pre_opt.converged
                            else "did NOT fully converge"
                        )
                        log.write(
                            f"\nGeometry optimization {_conv_str} in {_pre_opt.n_steps} steps."
                            f"  E = {_pre_opt.energies_hartree[-1]:.8f} Ha\n\n"
                        )
                        if not _pre_opt.converged:
                            log.write(
                                "⚠ Geometry optimization did not fully converge — "
                                "proceeding with best available geometry.\n\n"
                            )
                        _run_required_final_single_point(
                            calc_mol,
                            "after geometry optimization before UV-Vis",
                        )
                    except Exception as _pre_exc:
                        log.write(
                            f"\n⚠ Geometry optimization failed: {_pre_exc}\n"
                            "  Proceeding with the seed geometry as-is.\n\n"
                        )

                # ── Step 3: TD-DFT excited-state calculation ─────────────────
                self.run_status.value = "Running TD-DFT excited states..."
                result = run_tddft_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    nstates=self.nstates_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_tddft_result(result)
                save_spectra = {
                    "uv_vis": {
                        "excitation_energies_ev": result.excitation_energies_ev,
                        "oscillator_strengths": result.oscillator_strengths,
                        "wavelengths_nm": result.wavelengths_nm(),
                    }
                }
                save_type = "tddft"
            elif ct == "NMR Shielding":
                self.run_status.value = "Running NMR shielding (SCF + GIAO)..."
                from quantui.nmr_calc import run_nmr_calc

                result = run_nmr_calc(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_nmr_result(result)
                save_spectra = {
                    "nmr": {
                        "atom_symbols": list(result.atom_symbols),
                        "shielding_iso_ppm": list(result.shielding_iso_ppm),
                        "chemical_shifts_ppm": {
                            str(k): v for k, v in result.chemical_shifts_ppm.items()
                        },
                        "reference_compound": result.reference_compound,
                    }
                }
                save_type = "nmr"
            elif ct == "PES Scan":
                self.run_status.value = "Running PES scan…"
                from quantui.pes_scan import run_pes_scan

                _st = self._scan_type_dd.value.lower()
                _atom_idx: list = [
                    self._scan_atom1.value - 1,
                    self._scan_atom2.value - 1,
                ]
                if _st in ("angle", "dihedral"):
                    _atom_idx.append(self._scan_atom3.value - 1)
                if _st == "dihedral":
                    _atom_idx.append(self._scan_atom4.value - 1)

                result = run_pes_scan(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    scan_type=_st,
                    atom_indices=_atom_idx,
                    start=self._scan_start.value,
                    stop=self._scan_stop.value,
                    steps=self._scan_steps.value,
                    progress_stream=log,  # type: ignore[arg-type]
                )
                result_html = self._format_pes_scan_result(result)
                save_spectra = {
                    "pes_scan": {
                        "scan_type": result.scan_type,
                        "atom_indices": result.atom_indices,
                        "scan_parameter_values": result.scan_parameter_values,
                        "energies_hartree": result.energies_hartree,
                    }
                }
                save_type = "pes_scan"
            else:  # Single Point
                self.run_status.value = "Calculating..."
                from quantui import run_in_session

                # MP2 heavy-atom warning
                if self.method_dd.value.upper() == "MP2":
                    _n_heavy = sum(1 for a in calc_mol.atoms if a != "H")
                    if _n_heavy > 20:
                        self.result_output.append_display_data(
                            HTML(
                                '<div style="background:#fffbe6;border-left:4px solid #f59e0b;'
                                'padding:8px 12px;border-radius:4px;margin:4px 0;font-size:13px">'
                                f"⚠️ MP2 scales as O(N⁵) — this molecule has {_n_heavy} heavy atoms "
                                "and may be slow. Consider using DFT instead.</div>"
                            )
                        )

                _solvent = self.solvent_dd.value if self.solvent_cb.value else None
                result = run_in_session(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    progress_stream=log,  # type: ignore[arg-type]
                    solvent=_solvent,
                )
                result_html = self._format_result(result)
                save_spectra, save_type = {}, "single_point"

            _mark("result_ready")
            _elapsed = time.perf_counter() - _run_wall_t
            _elapsed_cpu = time.process_time() - _run_cpu_t
            self._last_result = result
            self._last_calc_type = save_type
            self.accumulate_btn.disabled = False

            self.result_output.append_display_data(HTML(result_html))
            self.run_status.value = "Finalizing results..."

            # Show 3D structure in the result panel and mirrored in Analysis tab
            _viz_mol = result.molecule if ct == "Geometry Opt" else calc_mol
            if ct == "Geometry Opt":
                self._viz_label.value = (
                    '<p style="color:#555;font-size:12px;font-weight:600;'
                    'margin:6px 0 2px">Optimized geometry</p>'
                )
                self._viz_label.layout.display = ""
            self._queue_main_thread_callback(
                self._show_result_3d,
                _viz_mol,
                self._analysis_mol_output,
            )
            _mark("viz_done")

            # Populate Analysis panels via the unified registry
            _ana_ctx = _AnalysisContext(
                calc_type=save_type,
                formula=result.formula,
                method=self.method_dd.value,
                basis=self.basis_dd.value,
                live_result=result,
                molecule=calc_mol,
                spectra_data=save_spectra,
                preopt_result=_pre_opt,
                source="live",
            )

            self.step_progress.complete(2)
            self.step_progress.complete(3)

            # Update completion banner
            _mol_label = _ana_ctx.label
            self._completion_mol_lbl.value = (
                f'<span style="color:#1e293b;font-size:13px;font-weight:500">'
                f"{_mol_label}</span>"
            )
            self._completion_banner.layout.display = ""
            _mark("banner_ready")

            # Write structured log footer
            try:
                from quantui.log_utils import format_log_footer as _fmt_log_ftr

                log.write(
                    _fmt_log_ftr(
                        result=result,
                        wall_time=_elapsed,
                        cpu_time=_elapsed_cpu,
                        log_text=log.getvalue(),
                        success=True,
                    )
                )
            except Exception:
                pass

            # Persist to disk
            _mark("persist_begin")
            try:
                from quantui import load_result, save_result
                from quantui.results_storage import (
                    save_orbitals,
                    save_thumbnail,
                    save_trajectory,
                )

                _saved_dir = save_result(
                    result,
                    pyscf_log=log.getvalue(),
                    calc_type=save_type,
                    spectra=save_spectra,
                )
                self._last_result_dir = _saved_dir
                # M-EXPORT / EXPORT.5: result folder is now on disk —
                # the "Export bundle (.zip)" button has something to zip.
                try:
                    self._export_bundle_btn.disabled = False
                except Exception:
                    pass
                _saved_data = load_result(_saved_dir)
                save_thumbnail(_saved_dir, _saved_data)
                _ana_ctx.result_dir = _saved_dir
                _ana_ctx.timestamp = str(_saved_data.get("timestamp", ""))
                # Persist trajectory so history viewer can replay it.
                if ct in ("Geometry Opt", "PES Scan"):
                    _traj = getattr(
                        result,
                        "trajectory" if ct == "Geometry Opt" else "coordinates_list",
                        None,
                    )
                    _e_list = getattr(result, "energies_hartree", [])
                    if _traj:
                        save_trajectory(_saved_dir, _traj, _e_list or [])
                        # M-EXPORT / EXPORT.3 + EXPORT.7: also write
                        # external-tool-friendly trajectory formats.
                        # Multi-frame XYZ (any viewer) and ASE .traj
                        # (ASE-GUI + ASE Python post-processing). Both
                        # best-effort: failures are caught by the outer
                        # save try/except so the calc still completes.
                        try:
                            from quantui.results_storage import (
                                save_trajectory_ase as _save_traj_ase,
                            )
                            from quantui.results_storage import (
                                save_trajectory_xyz as _save_traj_xyz,
                            )

                            _save_traj_xyz(
                                _saved_dir,
                                frames=_traj,
                                energies=_e_list or [],
                            )
                            _save_traj_ase(
                                _saved_dir,
                                frames=_traj,
                                energies=_e_list or [],
                            )
                        except Exception:
                            pass
                # Persist pre-opt geometry trajectory for Frequency runs (DEC-007).
                if ct == "Frequency" and _pre_opt is not None:
                    _pre_traj = getattr(_pre_opt, "trajectory", None)
                    _pre_e = list(getattr(_pre_opt, "energies_hartree", []))
                    if _pre_traj:
                        save_trajectory(
                            _saved_dir,
                            _pre_traj,
                            _pre_e,
                            filename="preopt_trajectory.json",
                        )
                # Persist MO data for orbital diagram + isosurface replay.
                if ct in ("Single Point", "Geometry Opt", "Frequency"):
                    save_orbitals(_saved_dir, result)
                # M-EXPORT / EXPORT.1+2: write a Molden-format companion
                # file so users can open results in Avogadro / IQmol /
                # Jmol. Best-effort — failures are swallowed by the
                # outer try block above and the calc still completes.
                # For SP / GeoOpt this writes orbitals + structure; for
                # Frequency it writes structure + [FREQ] / [FR-NORM-COORD]
                # blocks so Avogadro can animate vibrations directly.
                if ct in ("Single Point", "Geometry Opt", "Frequency"):
                    try:
                        from quantui.results_storage import (
                            save_molden as _save_molden,
                        )

                        _save_molden(
                            _saved_dir,
                            mo_energy_hartree=getattr(
                                result, "mo_energy_hartree", None
                            ),
                            mo_occ=getattr(result, "mo_occ", None),
                            mo_coeff=getattr(result, "mo_coeff", None),
                            pyscf_mol_atom=getattr(result, "pyscf_mol_atom", None),
                            pyscf_mol_basis=getattr(result, "pyscf_mol_basis", None),
                            charge=int(getattr(calc_mol, "charge", 0)),
                            multiplicity=int(getattr(calc_mol, "multiplicity", 1)),
                            frequencies_cm1=getattr(result, "frequencies_cm1", None),
                            normal_modes=getattr(result, "displacements", None),
                        )
                    except Exception:
                        pass
                self._queue_main_thread_callback(self._refresh_results_browser)
                self._queue_main_thread_callback(self._populate_compare_list)
                self._queue_main_thread_callback(
                    self._update_log_panel,
                    log.getvalue(),
                    f"{result.formula}  {self.method_dd.value}/{self.basis_dd.value}",
                )
                self._queue_main_thread_callback(
                    self._show_result_log,
                    _saved_dir,
                    log.getvalue(),
                )
            except Exception as _save_exc:
                try:
                    from quantui import calc_log as _clog

                    _clog.log_event(
                        "save_error",
                        f"{type(_save_exc).__name__}: {_save_exc}"[:300],
                    )
                except Exception:
                    pass
            _mark("persist_done")

            # Activate analysis panels after scheduling refresh/update callbacks.
            # Refreshing the history browser may fire past_dd observers that clear
            # analysis state; queueing this callback after refresh keeps ordering
            # deterministic on the kernel UI loop.
            _mark("analysis_begin")
            self._queue_main_thread_callback(self._apply_analysis_context, _ana_ctx)
            _mark("analysis_done")

            # Log performance
            _mark("perf_begin")
            try:
                _elapsed_for_est = time.perf_counter() - _run_wall_t
                _calc_log.log_calculation(
                    formula=result.formula,
                    n_atoms=len(calc_mol.atoms),
                    n_electrons=calc_mol.get_electron_count(),
                    method=result.method,
                    basis=result.basis,
                    n_iterations=getattr(result, "n_iterations", None),
                    elapsed_s=_elapsed_for_est,
                    converged=result.converged,
                    n_basis=_calc_log.count_basis_functions(
                        calc_mol.atoms, result.basis
                    ),
                    n_cores=1,
                    calc_type=save_type,
                    gpu_used=getattr(result, "gpu_used", None),
                    gpu_name=getattr(result, "gpu_name", None),
                )
                _calc_log.log_event(
                    "calc_done",
                    f"{result.method}/{result.basis} on {result.formula}",
                    elapsed_s=round(_elapsed_for_est, 2),
                    converged=result.converged,
                    gpu_used=bool(getattr(result, "gpu_used", False)),
                    gpu_name=getattr(result, "gpu_name", None),
                )
                # M-EST / EST.6: persist the (predicted, actual) pair to
                # ``prediction_log.jsonl``. ``_predicted_run_s`` was
                # captured at the top of _do_run via the same
                # estimate_time(...) call that drives the UI estimate;
                # ``_elapsed_for_est`` is the actual wall-time the calc
                # took. The analytics dashboard reads both to surface
                # accuracy metrics + the "consider re-calibrating"
                # banner when the median error exceeds threshold.
                try:
                    _calc_log.log_prediction(
                        predicted_s=_predicted_run_s,
                        actual_s=_elapsed_for_est,
                        method=result.method,
                        basis=result.basis,
                        calc_type=save_type,
                        formula=result.formula,
                        confidence=_predicted_run_confidence,
                        gpu_used=getattr(result, "gpu_used", None),
                    )
                except Exception:  # noqa: BLE001 — telemetry self-guard
                    pass
                self._update_estimate()
            except Exception:
                pass
            _mark("perf_done")

            _mark("success_done")
            _elapsed_total = _tail_marks["success_done"] - _run_wall_t
            self.run_status.value = f"Done in {_elapsed_total:.1f} s."

            try:
                _tail_end = _tail_marks.get("success_done")
                _post_scf_to_done: Optional[float] = None
                if _tail_end is not None and _scf_converged_t is not None:
                    _post_scf_to_done = round(_tail_end - _scf_converged_t, 3)
                _post_result_to_done = _span("result_ready", "success_done")
                _calc_log.log_event(
                    "calc_tail_timing",
                    "Post-SCF completion timing checkpoint",
                    session_id=self._session_id,
                    formula=result.formula,
                    method=result.method,
                    basis=result.basis,
                    calc_type=save_type,
                    scf_converged_seen=_scf_converged_t is not None,
                    post_scf_to_done_s=_post_scf_to_done,
                    post_result_to_done_s=_post_result_to_done,
                    result_to_viz_s=_span("result_ready", "viz_done"),
                    result_to_banner_s=_span("result_ready", "banner_ready"),
                    persist_block_s=_span("persist_begin", "persist_done"),
                    analysis_apply_s=_span("analysis_begin", "analysis_done"),
                    perf_block_s=_span("perf_begin", "perf_done"),
                    banner_to_done_s=_span("banner_ready", "success_done"),
                )
            except Exception:
                pass

        except ImportError as _import_err:
            _err_detail = str(_import_err)
            _msg = (
                f"Import error: {_err_detail}\n\n"
                "A required calculation dependency could not be loaded.\n"
                "On Windows: use the Apptainer container.\n"
                "  apptainer run quantui.sif\n"
            )
            log.write(_msg)
            _err_html = (
                '<div style="background:#fef2f2;border:1px solid #fca5a5;'
                'border-radius:8px;padding:16px;margin:8px 0">'
                '<b style="color:#b91c1c">&#9888; Dependency Not Available</b><br>'
                f'<span style="color:#7f1d1d">{_err_detail}</span><br><br>'
                '<small style="color:#991b1b">On Windows, use the Apptainer container: '
                "<code>apptainer run quantui.sif</code>. "
                "Full details are in the <b>Output</b> tab.</small>"
                "</div>"
            )
            self.result_output.append_display_data(HTML(_err_html))
            self.run_status.value = "Dependency unavailable."
            self.step_progress.fail(2, _err_detail[:60])
            _calc_log.log_event("calc_error", _err_detail[:200])

        except Exception as exc:
            import traceback as _tb

            _elapsed = time.perf_counter() - _run_wall_t
            _elapsed_cpu = time.process_time() - _run_cpu_t
            _tb_str = _tb.format_exc()
            # Full details → Output tab (for debugging/instructors)
            log.write(f"\n--- Calculation Error ---\n{exc}\n\n{_tb_str}")
            # Structured failure footer
            try:
                from quantui.log_utils import format_log_footer as _fmt_log_ftr

                log.write(
                    _fmt_log_ftr(
                        result=None,
                        wall_time=_elapsed,
                        cpu_time=_elapsed_cpu,
                        log_text=log.getvalue(),
                        success=False,
                    )
                )
            except Exception:
                pass
            # Write to persistent error log
            try:
                import datetime as _dt
                import os as _os

                _log_dir = Path(
                    _os.environ.get(
                        "QUANTUI_LOG_DIR",
                        Path.home() / ".quantui" / "logs",
                    )
                )
                _log_dir.mkdir(parents=True, exist_ok=True)
                _ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _formula = mol.get_formula() if mol is not None else "unknown"
                _method = self.method_dd.value
                _basis = self.basis_dd.value
                with open(_log_dir / "error_log.txt", "a") as _ef:
                    _ef.write(
                        f"\n{'='*60}\n"
                        f"{_ts}  {_formula}  {_method}/{_basis}\n"
                        f"{_tb_str}"
                    )
            except Exception:
                pass
            # Clean summary → result panel (student-facing)
            _err_html = (
                '<div style="background:#fef2f2;border:1px solid #fca5a5;'
                'border-radius:8px;padding:16px;margin:8px 0">'
                '<b style="color:#b91c1c">&#9888; Calculation Failed</b><br>'
                f'<code style="color:#7f1d1d">{exc}</code><br><br>'
                '<small style="color:#991b1b">'
                "Tips: try a smaller basis set (STO-3G), use a geometry-optimized "
                "structure first, or check for unusually long/short bonds in your "
                "XYZ input. Full error details are in the <b>Output</b> tab.</small>"
                "</div>"
            )
            self.result_output.append_display_data(HTML(_err_html))
            self.run_status.value = "Calculation failed."
            self.step_progress.fail(2, str(exc)[:60])
            _calc_log.log_event(
                "calc_error", str(exc)[:200], elapsed_s=round(_elapsed, 2)
            )

        finally:
            self.run_btn.disabled = False
            self._activity_end(kind="compute")

    def _update_notes(self, change=None) -> None:
        _run_update_notes(self, change)

    def _update_estimate(self, change=None) -> None:
        _run_update_estimate(self, calc_log_mod=_calc_log, change=change)

    def _refresh_results_browser(self) -> None:
        _run_refresh_results_browser(self)

    def _refresh_comparison(self) -> None:
        _run_refresh_comparison(self)

    def _populate_compare_list(self) -> None:
        _run_populate_compare_list(self)

    def _show_help_topic(self, topic: str) -> None:
        if topic in HELP_TOPICS:
            self.help_topic_dd.value = topic
        self.help_tab_panel.layout.display = ""

    def _update_log_panel(self, log_text: str, label: str = "") -> None:
        self._render_log(log_text, label)

    def _safe_cb(self, fn):
        """Wrap an .observe() handler so exceptions are logged instead of silently dropped."""

        def _wrapper(change):
            self._activity_begin(
                f"Running {getattr(fn, '__name__', 'callback')}...",
                kind="ui",
            )
            try:
                fn(change)
            except Exception as _e:
                import traceback as _tb

                try:
                    from quantui import calc_log as _clog

                    _clog.log_event(
                        "callback_error",
                        f"{getattr(fn, '__name__', repr(fn))}: "
                        f"{type(_e).__name__}: {_e}\n{_tb.format_exc()[:800]}",
                    )
                except Exception:
                    pass
            finally:
                self._activity_end(kind="ui")

        return _wrapper

    def _goto_output_tab(self) -> None:
        # POLISH.8 (M-POLISH, 2026-05-25): the standalone Log tab is
        # gone; the PySCF output log now lives in an Accordion inside
        # the History tab (index 3). Switch tabs + expand the log
        # accordion so the user lands directly on the log content.
        self.root_tab.selected_index = 3
        if hasattr(self, "_history_log_accordion"):
            try:
                self._history_log_accordion.selected_index = 0
            except Exception:  # noqa: BLE001 — best-effort UI tweak
                pass

    def _render_log(self, text: str, source_label: str = "") -> None:
        import html as _html_mod
        import re as _re

        _bfgs_re = _re.compile(r"^BFGS:\s+(\d+)\s+\S+\s+([-\d.]+)\s+([\d.]+)")

        lines = text.splitlines()
        rows = []
        for line in lines:
            esc = _html_mod.escape(line)
            # ── Log header / footer structure ─────────────────────────────────
            if len(line) >= 40 and line == "=" * len(line):
                style = "color:#1e3a5f;font-weight:700"
            elif "QuantUI — Quantum Chemistry Interface" in line:
                style = "color:#6d28d9;font-weight:700"
            elif line.startswith("  ── "):
                style = "color:#334155;font-weight:700"
            elif line.startswith("  ✓"):
                style = "color:#16a34a;font-weight:700"
            elif line.startswith("  ✗"):
                style = "color:#dc2626;font-weight:700"
            elif (
                line.startswith("  Machine:")
                or line.startswith("  GPU:")
                or line.startswith("  Threads:")
            ):
                style = "color:#475569"
            elif (
                line.startswith("  Molecule:")
                or line.startswith("  Method/Basis:")
                or line.startswith("  Calc type:")
                or line.startswith("  Started:")
            ):
                style = "color:#1d4ed8"
            elif (
                line.startswith("    Energy:")
                or line.startswith("    HOMO-LUMO gap:")
                or line.startswith("    ZPVE:")
            ):
                style = "color:#0f766e;font-weight:600"
            elif line.startswith("    Wall time:"):
                style = "color:#64748b"
            elif line.startswith("    ✔") or line.startswith("    ⚠"):
                style = "color:#d97706"
            # ── Geometry optimisation (ASE BFGS) ──────────────────────────────
            elif line.startswith("BFGS:"):
                m = _bfgs_re.match(line)
                if m:
                    fmax = float(m.group(3))
                    # Colour by convergence: green when nearly converged, teal otherwise
                    style = (
                        "color:#16a34a;font-weight:600"
                        if fmax < 0.1
                        else "color:#0d9488"
                    )
                else:
                    style = "color:#0d9488"
            elif line.strip() == "Step Time Energy fmax":
                style = "color:#334155;font-weight:700"
            # ── Post-optimisation summary ──────────────────────────────────────
            elif line.startswith("── Final SCF"):
                style = "color:#6d28d9;font-weight:600"
            elif "HOMO-LUMO gap:" in line:
                style = "color:#6d28d9;font-weight:600"
            # ── SCF convergence ────────────────────────────────────────────────
            elif "converged SCF energy" in line or "SCF converged" in line:
                style = "color:#16a34a;font-weight:600"
            elif line.lstrip().startswith("cycle=") and "E=" in line:
                style = "color:#64748b"
            # ── MO / orbital info (verbose=4) ──────────────────────────────────
            elif "MO energies" in line or "** MO" in line:
                style = "color:#1d4ed8;font-weight:600"
            elif "HOMO" in line or "LUMO" in line or "All MO energies" in line:
                style = "color:#2563eb"
            elif line.lstrip().startswith("occupied:") or line.lstrip().startswith(
                "virtual:"
            ):
                style = "color:#3b82f6"
            # ── Thermo / properties ────────────────────────────────────────────
            elif "Mulliken" in line or "mulliken" in line:
                style = "color:#7c3aed"
            elif "dipole" in line.lower() or "Dipole" in line:
                style = "color:#7c3aed"
            elif "nuclear repulsion" in line.lower() or "Nuclear repulsion" in line:
                style = "color:#94a3b8"
            elif "E(MP2)" in line or "MP2 correlation" in line:
                style = "color:#0891b2"
            # ── Warnings / errors ──────────────────────────────────────────────
            elif "Warning" in line or "warning" in line:
                style = "color:#d97706"
            elif "Error" in line or "error" in line or "failed" in line:
                style = "color:#dc2626"
            else:
                style = "color:#1e293b"
            rows.append(f'<div style="{style}">{esc}</div>')
        self._log_output_html.value = (
            '<div style="font-family:monospace;font-size:12px;line-height:1.4;'
            "padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;"
            'border-radius:4px;overflow-x:auto;max-height:550px;overflow-y:auto">'
            + "".join(rows)
            + "</div>"
        )
        self._log_source_lbl.value = (
            f'<span style="font-size:12px;color:#64748b">Source: {source_label}</span>'
            if source_label
            else ""
        )

    def _render_help_topic(self, change=None) -> None:
        key = self.help_topic_dd.value
        if key and key in HELP_TOPICS:
            entry = HELP_TOPICS[key]
            self.help_content_html.value = (
                f'<div style="border:1px solid #e2e8f0;border-radius:6px;'
                f'padding:14px 18px;margin:8px 0;background:#f8fafc;max-width:700px">'
                f'<h4 style="margin:0 0 10px;color:#1e293b;font-size:15px;font-weight:700">'
                f'{entry["title"]}</h4>'
                f'<div style="font-size:14px;color:#334155;line-height:1.6">'
                f'{entry["body"]}</div>'
                f"</div>"
            )

    def _refresh_perf_stats(self) -> None:
        self._perf_stats_html.value = self._build_perf_stats_html()
        self._perf_events_html.value = self._build_events_html()

    def _build_perf_stats_html(self) -> str:
        from quantui.calc_log import get_perf_history

        records = get_perf_history()
        if not records:
            return (
                '<span style="color:#94a3b8;font-size:13px">'
                "No performance data recorded yet.</span>"
            )
        groups: dict = {}
        for r in records:
            key = (r.get("method", "?"), r.get("basis", "?"))
            groups.setdefault(key, []).append(r)
        rows = ""
        for (meth, bas), recs in sorted(groups.items()):
            times = [r["elapsed_s"] for r in recs if "elapsed_s" in r]
            n = len(recs)
            if times:
                avg = sum(times) / len(times)
                rows += (
                    "<tr>"
                    f'<td style="padding:2px 12px 2px 0">{meth}</td>'
                    f'<td style="padding:2px 12px 2px 0">{bas}</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{n}</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{avg:.1f} s</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{min(times):.1f} s</td>'
                    f'<td style="padding:2px 12px 2px 0;text-align:right">{max(times):.1f} s</td>'
                    "</tr>"
                )
        header = (
            "<tr>"
            '<th style="text-align:left;padding:2px 12px 2px 0;color:#64748b">Method</th>'
            '<th style="text-align:left;padding:2px 12px 2px 0;color:#64748b">Basis</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Runs</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Avg</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Min</th>'
            '<th style="text-align:right;padding:2px 12px 2px 0;color:#64748b">Max</th>'
            "</tr>"
        )
        return (
            '<table style="font-size:13px;border-collapse:collapse;width:100%">'
            f"{header}{rows}</table>"
        )

    def _build_events_html(self) -> str:
        from quantui.calc_log import get_recent_events

        events = get_recent_events(20)
        if not events:
            return (
                '<span style="color:#94a3b8;font-size:13px">'
                "No events recorded yet.</span>"
            )
        rows = ""
        for e in reversed(events):
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            evt = e.get("event", "")
            msg = e.get("message", "")
            rows += (
                "<tr>"
                f'<td style="padding:1px 10px 1px 0;color:#94a3b8;font-size:11px;white-space:nowrap">{ts}</td>'
                f'<td style="padding:1px 10px 1px 0;color:#475569;font-size:12px">{evt}</td>'
                f'<td style="padding:1px 0;color:#334155;font-size:12px">{msg}</td>'
                "</tr>"
            )
        return (
            '<table style="font-size:13px;border-collapse:collapse;width:100%">'
            f"{rows}</table>"
        )

    # ══ RESULT FORMATTERS ════════════════════════════════════════════════════

    def _format_result(self, r) -> str:
        return _fmt_result(r)

    def _format_opt_result(self, r) -> str:
        return _fmt_opt_result(r)

    def _format_freq_result(self, r) -> str:
        return _fmt_freq_result(r)

    def _format_tddft_result(self, r) -> str:
        return _fmt_tddft_result(r)

    def _format_nmr_result(self, r) -> str:
        return _fmt_nmr_result(r)

    def _format_pes_scan_result(self, r) -> str:
        return _fmt_pes_scan_result(r)

    def _show_pes_scan_result(self, result) -> bool:
        return _viz_show_pes_scan_result(self, result)

    def _format_past_result(self, data: dict, result_dir: Optional[Path] = None) -> str:
        return _fmt_past_result(data, result_dir=result_dir)

    # ══ HELPERS ══════════════════════════════════════════════════════════════

    def _get_results_dir(self) -> Path:
        from quantui.results_storage import _default_results_dir

        return _default_results_dir().resolve()
