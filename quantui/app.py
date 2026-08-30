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
from typing import TYPE_CHECKING, Any, Callable, ClassVar, List, Literal, Optional, cast

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import HTML, Javascript, display

import quantui
import quantui.calc_log as _calc_log
import quantui.issue_tracker as _issue_tracker
from quantui import molecule_library as _ml
from quantui import theme as _theme
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
    on_reorg_view_changed as _ana_on_reorg_view_changed,
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
    pop_mulliken as _ana_pop_mulliken,
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
    pop_raman_spectrum as _ana_pop_raman_spectrum,
)
from quantui.app_analysis import (
    pop_reorg_geometries as _ana_pop_reorg_geometries,
)
from quantui.app_analysis import (
    pop_uv_vis as _ana_pop_uv_vis,
)
from quantui.app_analysis import (
    pop_vibrational as _ana_pop_vibrational,
)
from quantui.app_analysis import (
    scroll_analysis_tab_to_top as _ana_scroll_analysis_tab_to_top,
)
from quantui.app_analysis import (
    select_ana_panel as _ana_select_ana_panel,
)
from quantui.app_analysis import (
    update_mulliken_figure as _ana_update_mulliken_figure,
)
from quantui.app_builders import (
    _MOL_ANALYSIS_PNG_INBOX_CLASS,
    _MOL_CALC_PNG_INBOX_CLASS,
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
    build_slurm_jobs_tab as _bld_build_slurm_jobs_tab,
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
    on_export_reorg_geometries as _exp_on_export_reorg_geometries,
)
from quantui.app_exports import (
    on_export_xyz as _exp_on_export_xyz,
)
from quantui.app_exports import (
    on_iso_export_cube as _exp_on_iso_export_cube,
)
from quantui.app_exports import (
    on_mol_analysis_png_captured as _exp_on_mol_analysis_png_captured,
)
from quantui.app_exports import (
    on_mol_calc_png_captured as _exp_on_mol_calc_png_captured,
)
from quantui.app_exports import (
    on_mol_results_png_captured as _exp_on_mol_results_png_captured,
)
from quantui.app_exports import (
    on_orb_png_captured as _exp_on_orb_png_captured,
)
from quantui.app_exports import (
    on_reorg_png_captured as _exp_on_reorg_png_captured,
)
from quantui.app_exports import (
    on_vib_png_captured as _exp_on_vib_png_captured,
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
    format_reorg_result as _fmt_reorg_result,
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
from quantui.app_measurement import (
    on_measure_clear as _measure_on_clear,
)
from quantui.app_measurement import (
    on_measure_inbox_changed as _measure_on_inbox_changed,
)
from quantui.app_runflow import (
    apply_vib_mode_for_frequency as _run_apply_vib_mode_for_frequency,
)
from quantui.app_runflow import (
    calc_type_key as _run_calc_type_key,
)
from quantui.app_runflow import (
    do_calibration as _run_do_calibration,
)
from quantui.app_runflow import (
    on_accumulate as _run_on_accumulate,
)
from quantui.app_runflow import (
    on_basis_fix as _run_on_basis_fix,
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
    on_calc_type_help as _run_on_calc_type_help,
)
from quantui.app_runflow import (
    on_charge_mult_apply as _run_on_charge_mult_apply,
)
from quantui.app_runflow import (
    on_charge_mult_suggest as _run_on_charge_mult_suggest,
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
    on_exit_cancel as _run_on_exit_cancel,
)
from quantui.app_runflow import (
    on_exit_clicked as _run_on_exit_clicked,
)
from quantui.app_runflow import (
    on_expand_mol_input as _run_on_expand_mol_input,
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
    on_preopt_accept as _run_on_preopt_accept,
)
from quantui.app_runflow import (
    on_preopt_preview as _run_on_preopt_preview,
)
from quantui.app_runflow import (
    on_preopt_reset as _run_on_preopt_reset,
)
from quantui.app_runflow import (
    on_reset_click as _run_on_reset_click,
)
from quantui.app_runflow import (
    on_run_clicked as _run_on_run_clicked,
)
from quantui.app_runflow import (
    on_seed_changed as _run_on_seed_changed,
)
from quantui.app_runflow import (
    on_solvent_cb_changed as _run_on_solvent_cb_changed,
)
from quantui.app_runflow import (
    on_spin_apply as _run_on_spin_apply,
)
from quantui.app_runflow import (
    on_spin_suggest as _run_on_spin_suggest,
)
from quantui.app_runflow import (
    populate_compare_list as _run_populate_compare_list,
)
from quantui.app_runflow import (
    refresh_comparison as _run_refresh_comparison,
)
from quantui.app_runflow import (
    refresh_results_browser as _run_refresh_results_browser,
)
from quantui.app_runflow import (
    refresh_seed_options as _run_refresh_seed_options,
)
from quantui.app_runflow import (
    resolve_seed_geometry as _run_resolve_seed_geometry,
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
from quantui.app_slurm import (
    on_slurm_jobs_cancel_clicked as _slurm_on_jobs_cancel_clicked,
)
from quantui.app_slurm import (
    on_slurm_jobs_refresh_clicked as _slurm_on_jobs_refresh_clicked,
)
from quantui.app_slurm import (
    on_slurm_jobs_remove_clicked as _slurm_on_jobs_remove_clicked,
)
from quantui.app_slurm import (
    on_slurm_jobs_view_clicked as _slurm_on_jobs_view_clicked,
)
from quantui.app_slurm import (
    on_slurm_reconnect_clicked as _slurm_on_reconnect_clicked,
)
from quantui.app_slurm import (
    refresh_slurm_jobs_tab as _slurm_refresh_jobs_tab,
)
from quantui.app_slurm import (
    slurm_jobs_tab_visible as _slurm_jobs_tab_visible,
)
from quantui.app_slurm import (
    startup_slurm_check as _slurm_startup_check,
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
    on_iso_appearance_changed as _viz_on_iso_appearance_changed,
)
from quantui.app_visualization import (
    on_iso_cancel as _viz_on_iso_cancel,
)
from quantui.app_visualization import (
    on_iso_generate as _viz_on_iso_generate,
)
from quantui.app_visualization import (
    on_nmr_nucleus_changed as _viz_on_nmr_nucleus_changed,
)
from quantui.app_visualization import (
    on_orb_range_changed as _viz_on_orb_range_changed,
)
from quantui.app_visualization import (
    on_raman_fwhm_changed as _viz_on_raman_fwhm_changed,
)
from quantui.app_visualization import (
    on_raman_mode_changed as _viz_on_raman_mode_changed,
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
    on_uv_range_changed as _viz_on_uv_range_changed,
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
    rerender_3d_scenes_for_theme as _viz_rerender_3d_scenes_for_theme,
)
from quantui.app_visualization import (
    show_ir_spectrum as _viz_show_ir_spectrum,
)
from quantui.app_visualization import (
    show_nmr_spectrum as _viz_show_nmr_spectrum,
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
    show_raman_spectrum as _viz_show_raman_spectrum,
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
    update_nmr_figure as _viz_update_nmr_figure,
)
from quantui.app_visualization import (
    update_raman_figure as _viz_update_raman_figure,
)
from quantui.app_visualization import (
    update_uv_vis_figure as _viz_update_uv_vis_figure,
)
from quantui.app_visualization import (
    wire_ir_controls as _viz_wire_ir_controls,
)
from quantui.app_visualization import (
    wire_nmr_controls as _viz_wire_nmr_controls,
)
from quantui.app_visualization import (
    wire_raman_controls as _viz_wire_raman_controls,
)
from quantui.app_visualization import (
    wire_uv_controls as _viz_wire_uv_controls,
)
from quantui.app_xyz_input import (
    on_load_xyz as _xyz_on_load_xyz,
)
from quantui.app_xyz_input import (
    on_xyz_add_atom as _xyz_on_add_atom,
)
from quantui.app_xyz_input import (
    on_xyz_apply_table as _xyz_on_apply_table,
)
from quantui.app_xyz_input import (
    on_xyz_cleanup as _xyz_on_cleanup,
)
from quantui.app_xyz_input import (
    on_xyz_cleanup_accept as _xyz_on_cleanup_accept,
)
from quantui.app_xyz_input import (
    on_xyz_cleanup_reject as _xyz_on_cleanup_reject,
)
from quantui.app_xyz_input import (
    on_xyz_fill_table as _xyz_on_fill_table,
)
from quantui.backends.dispatch import is_slurm_available
from quantui.cancellation import CalcCancelled as _CalcCancelled

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
        render_molecule_html as _render_molecule_html,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
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
        resolve_structure_with_message as _resolve_structure_with_message,
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
    _resolve_structure_with_message = None  # type: ignore[assignment]
    _struct_search_candidates = None  # type: ignore[assignment]


# Provider key → short, accurate label for the loaded-molecule card. Replaces
# the old hard-coded "PubChem: <query>" which mislabeled offline/library/SMILES
# hits (2026-06-15).
_STRUCT_SOURCE_PREFIX = {
    "pubchem": "PubChem",
    "cactus": "NCI CACTUS",
    "rdkit-smiles": "SMILES",
    "rdkit-inchi": "InChI",
    "library": "Library",
    "library-offline-fallback": "Library (offline)",
}

try:
    from quantui.session_calc import SessionResult, run_in_session  # noqa: F401

    _PYSCF_AVAILABLE = True
except (ImportError, AttributeError):
    _PYSCF_AVAILABLE = False

try:
    # Availability probe only — the classical pre-opt is invoked via the
    # interactive Preview flow (app_runflow uses preoptimize_with_trajectory),
    # not from app.py. ``_PREOPT_AVAILABLE`` gates the Preview button.
    import quantui.preopt  # noqa: F401

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
    color: var(--q-text-strong) !important;
    letter-spacing: -0.01em !important;
    margin: 10px 0 4px !important;
    border-bottom: none !important;
}

/* Live calculation log — must stay fixed-width --------------------------- */
/* The system-font rule above lists ``.jp-OutputArea-output``, which is exactly
   the element the streaming calc log renders into — so the log inherited a
   PROPORTIONAL font. Two things in the header depend on fixed-width cells and
   both broke together: the ASCII wordmark (letters slid into each other) and
   the padded ``Label           : value`` provenance rows (colons drifted out of
   line even though the padding is correct). Re-assert monospace for the log
   only. Two classes out-specifies the single-class rule above, and
   ``!important`` is required to beat its ``!important``. */
/* The first two selectors cover the historical widgets.Output rendering; the
   [class*=] selector covers the LiveLog container (M-LOGSCROLL route C), whose
   class carries a per-app uid suffix. LiveLog also sets the stack inline — this
   is belt-and-braces, since a directly-applied rule beats an inherited one. */
.quantui-run-output .jp-OutputArea-output,
.quantui-run-output pre,
[class*="quantui-live-log"] {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
                 "Liberation Mono", "Courier New", monospace !important;
    font-variant-ligatures: none !important;  /* no ligatures in ASCII art */
}

/* Section headers ------------------------------------------------------- */
h3 {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: var(--q-text-slate) !important;
    margin: 24px 0 10px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid var(--q-border) !important;
}

/* Rounded corners on inputs, dropdowns, and buttons -------------------- */
.widget-text input, .widget-textarea textarea {
    border-color: #d1d5db !important;
    border-radius: 5px !important;
}
.widget-dropdown select { border-radius: 5px !important; }
.widget-button, .widget-toggle-button { border-radius: 5px !important; }

/* Suppress Jupyter stderr pink in dark palettes */
.jp-OutputArea-stderr, .output_stderr {
    background: transparent !important;
}

/* Inline "calculating" spinner — shown next to slow on-demand
   controls (e.g. orbital-isosurface generation) while work is in flight. */
@keyframes quantui-spin { to { transform: rotate(360deg); } }
.quantui-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--q-border);
    border-top-color: var(--q-accent-info);
    border-radius: 50%;
    animation: quantui-spin 0.7s linear infinite;
    vertical-align: middle;
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
# TD-DFT root convergence. PySCF's Davidson solver prints
# "root %d converged  |r|= ...  e= <excitation energy, Ha>  max|de|= ..." at
# verbose=5 (DEBUG) — see tddft_calc.py's td.verbose. This is the only
# per-root progress signal the solve emits; without it the heartbeat's
# generic "still working" line is all a user sees during a multi-minute
# excited-state solve.
_RE_TD_ROOT = re.compile(
    r"root\s+(\d+)\s+converged\s+\|r\|=\s*[\d.eE+\-]+\s+e=\s*([\d.eE+\-]+)"
)
# Per-phase progress for long silent kernels (NMR, Hessian, post-HF, TD-DFT).
_RE_CCSD_CYCLE = re.compile(
    r"cycle\s*=\s*(\d+)\s+E_corr\(CCSD\)\s*=\s*[\-\d\.Ee+\-]+\s+dE\s*=\s*([\-\d\.Ee+\-]+)"
)
_RE_NMR_RANGE = re.compile(r"shielding for atoms range\(0,\s*(\d+)\)")
_RE_NMR_ATOM = re.compile(r"total shielding of atom\s+(\d+)")
_RE_HESS_ATOM = re.compile(r"contracting int2e_ip1ip2 for atom\s+(\d+)")
_RE_MP2_TRANSFORM = re.compile(r"transform \(ia\|jb\)")
_RE_MP2_KERNEL = re.compile(r"CPU time for kernel")
_HARTREE_TO_EV = 27.211386245988
# Step/point/state counters inside a status message. Removed before the
# message is used as a per-stage timing key — see _LogCapture._stage_key.
_RE_STAGE_NUMBERS = re.compile(r"\d+(?:[./]\d+)*")

# ── Silent-phase heartbeat ───────────────────────────────────────────────────
#
# Seconds of stream silence before the log says it is still alive.
#
# Sized from a real measurement, not a guess: a user timed an aspirin
# (21 atoms) B3LYP/6-31G* UV-Vis run and the log printed **nothing for 120 s**
# after "converged SCF energy" while the TD-DFT solve ran. The status label was
# advancing the whole time — Phase A covers that — but the log, which is what a
# user actually watches, looked frozen.
#
# 25 s yields ~4 lines across that gap: enough to prove liveness, few enough not
# to bloat the archived pyscf.log. Gaps grow steeply with system size, and
# aspirin is a *small* case, so err on the short side.
_HEARTBEAT_AFTER_S = 25.0

# How often the watchdog wakes to check. Well under _HEARTBEAT_AFTER_S so a beat
# lands close to its due time, but coarse enough to be free.
_HEARTBEAT_POLL_S = 2.0


# ══ LOG CAPTURE ══════════════════════════════════════════════════════════════


# ``_CalcCancelled`` is defined in quantui.cancellation (imported at the top of
# this module) so the calc modules (session_calc / optimizer / freq / tddft /
# nmr / pes) can raise the SAME class from their SCF callbacks + optimizer
# observers without importing the app layer. ``_do_run``'s
# ``except _CalcCancelled`` catches it whether it was raised by
# ``_LogCapture.write`` or by one of those hooks.


class _LogCapture:
    """Write PySCF output to an Output widget and capture it to a buffer."""

    def __init__(
        self,
        output_widget: widgets.Output,
        status_label: Optional[widgets.Label] = None,
        on_scf_converged: Optional[Callable[[], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._w = output_widget
        self._buf = io.StringIO()
        self._line_buf = ""
        self._status = status_label
        self._on_scf_converged = on_scf_converged
        self._scf_converged_seen = False
        self._cancel_check = cancel_check
        # Public alias so calc modules can duck-type the predicate off the
        # progress stream (see quantui.cancellation.cancel_check_from_stream).
        self.cancel_check = cancel_check
        # Completion fraction (0..1) reported by calc modules via
        # log_utils.emit_progress; read by the elapsed ticker. None = unknown.
        self._fraction: Optional[float] = None
        # Silent-phase heartbeat. Long kernels — the TD-DFT excited-state solve
        # most of all — print nothing for minutes, so the log looks hung even
        # though the status label is advancing. A watchdog appends a "still
        # working" line when the stream has gone quiet.
        self._last_write_t = time.monotonic()
        self._hb_started_t = self._last_write_t
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        # Per-stage wall times. Every calc type announces its phases through
        # log_utils.emit_status, and every announcement passes through this
        # object — stage boundaries can be timed here without threading a timer
        # through optimizer/freq/tddft/nmr one by one.
        self._stage_times: dict[str, float] = {}
        self._stage_name: Optional[str] = None
        self._stage_started_t = self._last_write_t
        # D3: atom totals parsed from PySCF's NMR header line.
        self._nmr_atom_total: Optional[int] = None

    # ── Per-stage timing ────────────────────────────────────────────────────

    @staticmethod
    def _stage_key(message: str) -> str:
        """Collapse a live status message to a stable stage name.

        Status messages carry per-step detail — "Opt step 7 — SCF…",
        "Solving TD-DFT excited states (10)…" — which is exactly right for
        the user watching the run and exactly wrong as a dictionary key: it
        would turn one stage into one entry per step. Stripping the numbers
        leaves the phase itself, which is the unit a cost model reasons in.
        """
        text = _RE_STAGE_NUMBERS.sub(" ", message)
        text = text.replace("…", " ").replace("(", " ").replace(")", " ")
        text = " ".join(text.split())
        return text.strip(" -—:·").lower()

    def _enter_stage(self, name: str) -> None:
        """Close the stage in progress and start *name*.

        Repeated announcements of the same stage (the optimizer re-announces
        every step) are treated as one continuous stage, so the recorded
        breakdown stays at the granularity a cost model can use rather than
        exploding into one entry per step.
        """
        now = time.monotonic()
        if self._stage_name is not None and name != self._stage_name:
            prev = self._stage_times.get(self._stage_name, 0.0)
            self._stage_times[self._stage_name] = prev + (now - self._stage_started_t)
        if name != self._stage_name:
            self._stage_name = name
            self._stage_started_t = now

    def stage_timings(self) -> dict[str, float]:
        """Return ``{stage: seconds}``, including the stage still running.

        Safe to call mid-run: the open stage is measured up to now rather
        than omitted, so a caller that logs this at the end of a calc gets a
        breakdown that actually sums to the run.
        """
        out = dict(self._stage_times)
        if self._stage_name is not None:
            elapsed = time.monotonic() - self._stage_started_t
            out[self._stage_name] = out.get(self._stage_name, 0.0) + elapsed
        return out

    # ── Silent-phase heartbeat ──────────────────────────────────────────────

    def start_heartbeat(self) -> None:
        """Begin watching for silent stretches. Idempotent."""
        if self._hb_thread is not None:
            return
        self._last_write_t = time.monotonic()
        self._hb_started_t = self._last_write_t
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="quantui-log-heartbeat"
        )
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the watchdog. Safe to call more than once, or if never started."""
        self._hb_stop.set()
        self._hb_thread = None

    def _heartbeat_loop(self) -> None:
        """Append a liveness line whenever the stream has been quiet too long.

        Deliberately writes **directly** to the widget and buffer rather than
        going through :meth:`write`: that path checks cancellation (which would
        raise ``_CalcCancelled`` on this thread, where nothing can catch it) and
        would also reset the very timer being measured.
        """
        while not self._hb_stop.wait(_HEARTBEAT_POLL_S):
            quiet_for = time.monotonic() - self._last_write_t
            if quiet_for < _HEARTBEAT_AFTER_S:
                continue
            stage = ""
            if self._status is not None:
                try:
                    stage = str(self._status.value).strip()
                except Exception:  # noqa: BLE001 — a label read must not kill it
                    stage = ""
            from quantui.log_utils import format_elapsed

            elapsed = time.monotonic() - self._hb_started_t
            line = "   … still working"
            if stage:
                line += f" — {stage}"
            line += f"  ·  {format_elapsed(elapsed)} elapsed\n"
            try:
                # Widget only — deliberately NOT self._buf. The buffer becomes
                # the result directory's pyscf.log, which should stay a faithful
                # record of what PySCF emitted. Heartbeats are UI chrome for the
                # live view; padding the archived log with them would make a
                # long silent run look chatty after the fact.
                self._w.append_stdout(line)
            except Exception:  # noqa: BLE001 — never let the log kill a run
                pass
            # Reset so the next beat is measured from this line, giving evenly
            # spaced heartbeats instead of one per poll once the gap is open.
            self._last_write_t = time.monotonic()

    def write(self, text: str) -> None:
        if not text:
            return
        # Cooperative cancellation: the calc prints frequently (per SCF cycle /
        # opt step), so raising here stops it at the next line. See _CalcCancelled.
        if self._cancel_check is not None and self._cancel_check():
            raise _CalcCancelled()
        # Any real output resets the silence timer, so a heartbeat only ever
        # appears in a genuinely quiet stretch.
        self._last_write_t = time.monotonic()
        self._w.append_stdout(text)
        self._buf.write(text)
        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            m = _RE_Q_STATUS.search(line)
            if m:
                message = m.group(1).strip()
                self._enter_stage(self._stage_key(message))
                if self._status is not None:
                    self._status.value = message
                continue
            m = _RE_CYCLE.search(line)
            if m and self._status is not None:
                n, delta = m.group(1), m.group(3)
                try:
                    self._status.value = f"SCF cycle {n}  ·  ΔE = {float(delta):.4g} Ha"
                except Exception:
                    self._status.value = f"SCF cycle {n}"
                continue
            m = _RE_TD_ROOT.search(line)
            if m and self._status is not None:
                root, e_ha = m.group(1), m.group(2)
                try:
                    root_n = int(root) + 1  # PySCF's root index is 0-based
                    ev = float(e_ha) * _HARTREE_TO_EV
                    self._status.value = (
                        f"TD-DFT root {root_n} converged  ·  {ev:.3f} eV"
                    )
                except Exception:
                    self._status.value = f"TD-DFT root {root} converged"
                continue
            m = _RE_CCSD_CYCLE.search(line)
            if m and self._status is not None:
                cycle, d_e = m.group(1), m.group(2)
                try:
                    self._status.value = (
                        f"CCSD cycle {cycle}  ·  ΔE = {float(d_e):.4g} Ha"
                    )
                except Exception:
                    self._status.value = f"CCSD cycle {cycle}"
                continue
            m = _RE_NMR_RANGE.search(line)
            if m:
                try:
                    self._nmr_atom_total = int(m.group(1))
                except Exception:
                    self._nmr_atom_total = None
                continue
            m = _RE_NMR_ATOM.search(line)
            if m and self._status is not None:
                atom_idx = m.group(1)
                try:
                    atom_n = int(atom_idx) + 1
                    if self._nmr_atom_total:
                        self._status.value = (
                            f"NMR GIAO · atom {atom_n}/{self._nmr_atom_total}"
                        )
                    else:
                        self._status.value = f"NMR GIAO · atom {atom_n}"
                except Exception:
                    self._status.value = f"NMR GIAO · atom {atom_idx}"
                continue
            m = _RE_HESS_ATOM.search(line)
            if m and self._status is not None:
                atom_idx = m.group(1)
                try:
                    self._status.value = f"Hessian build · atom {int(atom_idx) + 1}…"
                except Exception:
                    self._status.value = f"Hessian build · atom {atom_idx}…"
                continue
            if _RE_MP2_TRANSFORM.search(line) and self._status is not None:
                self._status.value = "MP2 · transforming integrals…"
                continue
            if _RE_MP2_KERNEL.search(line) and self._status is not None:
                self._status.value = "MP2 · correlation kernel…"
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

    def set_status(self, message: str) -> None:
        """Update the live status label WITHOUT appending to the log.

        Calc modules (optimizer / pes / reorg) call this via
        ``log_utils.emit_status`` to surface a stage label ("Optimizing —
        step k…") during silent (``verbose=0``) phases, without cluttering the
        output log the way a ``[QuantUI_STATUS]`` stream line would.
        """
        self._enter_stage(self._stage_key(message))
        if self._status is not None:
            try:
                self._status.value = message
            except Exception:
                pass

    def set_progress_fraction(self, fraction: float) -> None:
        """Record a completion fraction (0..1) for the live remaining-time chip.

        Calc modules call this via ``log_utils.emit_progress``
        when they know a real completion fraction (PES points, optimizer fmax
        trend). The elapsed ticker reads it off the active log and prefers a
        self-correcting ``elapsed·(1−f)/f`` estimate over the static total.
        """
        try:
            f = float(fraction)
        except (TypeError, ValueError):
            return
        # Clamp below 1.0 so the chip never claims "0s left" while work remains.
        self._fraction = max(0.0, min(f, 0.999))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        """No-op — required so ASE treats this as an already-open stream.

        ASE's ``IOContext.openfile()`` (used by ``BFGS(..., logfile=...)``
        in optimizer.py / pes_scan.py) checks ``hasattr(file, "close")`` to
        decide whether *file* is an already-open, file-like object it
        should leave alone, vs. a path string it should ``open()`` itself.
        Without this method, ase>=3.22 (the floor this project pins) still
        happened to work via a later refactor's more lenient check, but
        ase==3.26.0 (the newest version pip resolves for Python 3.9) hits
        the stricter ``openfile()`` and raises
        ``TypeError: expected str, bytes or os.PathLike object`` — a real
        Python-3.9-specific compatibility gap the CI matrix
        expansion caught.
        """

    def getvalue(self) -> str:
        return self._buf.getvalue()

    def seed_prior(self, text: str) -> None:
        """Prepend output from an earlier interrupted chunk (resume / ISSUE.9).

        Seeds the capture buffer and the live Output widget so the archived
        ``pyscf.log`` and the on-screen log both show the full story.
        """
        if not text:
            return
        try:
            self._w.append_stdout(text)
        except Exception:  # noqa: BLE001 — never let the log kill a run
            pass
        self._buf.write(text)


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
        _exit_cancel_btn: Any
        _exit_warn_html: Any
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
        _cal_skip_btn: Any
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
        _resume_cb: Any
        _resume_discard_btn: Any
        _resume_entries: Any
        _resume_list_box: Any
        _resume_list_dd: Any
        _resume_list_html: Any
        _resume_notice_html: Any
        _resume_restore_btn: Any
        _status_html: Any
        _status_tab_panel: Any
        _theme_style: Any
        _welcome_html: Any
        _welcome_header: Any
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
        gpu_enabled_cb: Any
        density_fit_enabled_cb: Any
        execution_backend_dd: Any
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
        history_search: Any
        history_filter_clear_btn: Any
        history_count_lbl: Any
        history_method_dd: Any
        history_basis_dd: Any
        history_date_from: Any
        history_date_to: Any
        _history_calc_chips: Any
        _history_status_chips: Any
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
        cancel_btn: Any
        basis_fix_btn: Any
        run_output: Any
        run_panel: Any
        run_status: Any
        _run_elapsed_lbl: Any
        _slurm_job_banner: Any
        _slurm_reconnect_btn: Any
        _slurm_jobs_summary_html: Any
        _slurm_jobs_table_html: Any
        _slurm_jobs_select: Any
        _slurm_jobs_refresh_btn: Any
        _slurm_jobs_view_btn: Any
        _slurm_jobs_cancel_btn: Any
        _slurm_jobs_remove_btn: Any
        _slurm_jobs_status_html: Any
        slurm_jobs_tab_panel: Any
        _slurm_jobs_tab_index: int | None
        _calculate_tab_panel: Any
        _root_tab_order_cache: list[str]
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
        xyz_add_atom_btn: Any
        xyz_apply_table_btn: Any
        xyz_cleanup_accept_btn: Any
        xyz_cleanup_btn: Any
        xyz_cleanup_preview: Any
        xyz_cleanup_preview_box: Any
        xyz_cleanup_reject_btn: Any
        xyz_fill_table_btn: Any
        xyz_table_box: Any
        _freq_preopt_cb: Any
        _freq_perturb_box: Any
        _freq_perturb_fraction: Any
        _freq_perturb_mode_dd: Any
        _seed_dd: Any
        _seed_note: Any
        _seed_refresh_btn: Any
        _geo_seed_dd: Any
        _geo_seed_note: Any
        _freq_seed_dd: Any
        _freq_seed_note: Any
        _freq_seed_refresh_btn: Any
        _tddft_seed_dd: Any
        _go_analysis_btn: Any
        _go_results_btn: Any
        _ir_export_btn: Any
        _ir_export_fmt_dd: Any
        _ir_export_status: Any
        _ir_fig: Any
        _ir_fwhm_slider: Any
        _ir_mode_toggle: Any
        _ir_accordion: Any
        _ir_copy_data_btn: Any
        _raman_export_btn: Any
        _raman_export_fmt_dd: Any
        _raman_export_status: Any
        _raman_fig: Any
        _raman_fwhm_slider: Any
        _raman_mode_toggle: Any
        _raman_accordion: Any
        _raman_copy_data_btn: Any
        _iso_accordion: Any
        _iso_generate_btn: Any
        _iso_cancel_btn: Any
        _iso_colors_dd: Any
        _iso_export_cube_btn: Any
        _iso_isovalue_slider: Any
        _iso_opacity_slider: Any
        _iso_wireframe_cb: Any
        _iso_resolution_dd: Any
        _last_result_dir: Any
        _measure_inbox: Any
        _measure_js_bridge: Any
        _measure_readout: Any
        _measure_clear_btn: Any
        _measure_help_btn: Any
        _measure_controls: Any
        _measure_fallback_msg: Any
        _measure_panel: Any
        _measure_picks: Any
        _mulliken_accordion: Any
        _mulliken_summary: Any
        _mulliken_table: Any
        _mulliken_fig: Any
        _mulliken_help_btn: Any
        _mulliken_color_cb: Any
        _mulliken_dipole_cb: Any
        _mulliken_vividness_slider: Any
        _mulliken_mol_output: Any
        _mulliken_overlay_note: Any
        _populations_js_bridge: Any
        _nmr_accordion: Any
        _nmr_fig: Any
        _nmr_nucleus_toggle: Any
        _nmr_output: Any
        _nmr_summary: Any
        _nmr_export_btn: Any
        _nmr_export_fmt_dd: Any
        _nmr_export_status: Any
        _nmr_copy_data_btn: Any
        _orb_accordion: Any
        _orb_diagram_box: Any
        _orb_diagram_html: Any
        _orb_export_btn: Any
        _orb_export_fmt_dd: Any
        _orb_export_status: Any
        _orb_copy_data_btn: Any
        _orb_iso_controls: Any
        _orb_iso_output: Any
        _orb_n_orb_input: Any
        _orb_index_input: Any
        _orb_png_inbox: Any
        _orb_toggle: Any
        _orb_ymax_input: Any
        _orb_ymin_input: Any
        _pes_export_btn: Any
        _pes_export_fmt_dd: Any
        _pes_export_status: Any
        _pes_copy_data_btn: Any
        _pes_plot_html: Any
        _pes_scan_accordion: Any
        _reorg_view_toggle: Any
        _reorg_overlay_pair: Any
        _reorg_exaggerate: Any
        _reorg_mode_dd: Any
        _reorg_export_btn: Any
        _reorg_export_status: Any
        _reorg_png_inbox: Any
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
        _uv_copy_data_btn: Any
        _uv_fwhm_slider: Any
        _uv_mode_toggle: Any
        _uv_xmin_input: Any
        _uv_xmax_input: Any
        _uv_range_hint: Any
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
        calc_type_help_btn: Any
        charge_si: Any
        charge_mult_apply_btn: Any
        charge_mult_suggest_btn: Any
        charge_mult_suggest_output: Any
        clear_btn: Any
        _completion_banner: Any
        _completion_mol_lbl: Any
        comparison_output: Any
        export_btn: Any
        export_mol_btn: Any
        export_pdb_btn: Any
        export_status: Any
        export_xyz_btn: Any
        _export_bundle_btn: Any
        fmax_fi: Any
        log_clear_btn: Any
        max_steps_si: Any
        method_dd: Any
        method_help_btn: Any
        mol_info_html: Any
        mol_summary_compact: Any
        mult_si: Any
        _method_card_html: Any
        _basis_card_html: Any
        _descriptor_cards_box: Any
        _open_shell_hint: Any
        spin_metal_dd: Any
        spin_ox_si: Any
        spin_geom_dd: Any
        spin_suggest_btn: Any
        spin_helper_output: Any
        spin_apply_btns: Any
        spin_helper_box: Any
        nstates_si: Any
        perf_estimate_html: Any
        post_calc_panel: Any
        preopt_preview_label: Any
        preopt_preview_btn: Any
        preopt_accept_btn: Any
        preopt_reset_btn: Any
        preopt_preview_status: Any
        preopt_preview_output: Any
        preopt_preview_box: Any
        results_panel: Any
        results_tab_panel: Any
        struct_export_status: Any
        traj_accordion: Any
        traj_output: Any
        vib_accordion: Any
        vib_mode_dd: Any
        vib_output: Any
        vib_prev_btn: Any
        vib_next_btn: Any
        _vib_apply_mode_btn: Any
        _vib_export_btn: Any
        _vib_export_status: Any
        _vib_png_inbox: Any
        _vib_png_status: Any
        _mol_calc_png_inbox: Any
        _mol_calc_png_status: Any
        _mol_results_png_inbox: Any
        _mol_results_png_status: Any
        _mol_analysis_png_inbox: Any
        _mol_analysis_png_status: Any
        _last_vib_molecule: Any

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
        self._last_nmr_fig: Any = None
        self._last_orb_fig: Any = None
        self._last_orb_info: Any = None
        # Orbital state consumed by the Isosurface panel populator. Always
        # initialized to None so ``pop_isosurface`` can read the attributes
        # via direct access without raising AttributeError on a fresh app
        # or on a history-replay where ``orbitals.npz`` is missing.
        # ``_apply_analysis_context`` resets these between contexts so stale
        # state from a prior calc cannot leak into the next molecule.
        self._last_orb_mo_coeff: Any = None
        self._last_orb_mo_occ: Any = None
        self._last_orb_mol_atom: Any = None
        self._last_orb_mol_basis: Any = None
        # Mulliken Populations panel state (table + Plotly bar chart).
        self._last_mulliken_symbols: Any = None
        self._last_mulliken_charges: Any = None
        self._last_mulliken_dipole: Any = None
        self._last_mulliken_dipole_vector: Any = None
        self._last_mulliken_fig: Any = None
        # Last-generated cube file path + orbital label.
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
        # Cancellation + run-in-flight state. ``_cancel_event`` is checked by
        # the run's _LogCapture each output line; ``_calc_running`` guards the
        # Clear button from wiping output mid-run.
        self._cancel_event = threading.Event()
        self._calc_running: bool = False
        # Stop signal for the live elapsed-time ticker thread.
        self._elapsed_stop_event: Optional[threading.Event] = None
        # Total run estimate the ticker turns into "time
        # remaining"; set by _do_run once estimate_time() has run.
        self._run_estimate_s: Optional[float] = None
        self._run_estimate_conf: str = "unknown"
        # The active run's _LogCapture, so the ticker can read the
        # completion fraction calc modules report onto it. None between runs.
        self._active_log: Optional[_LogCapture] = None
        # Calc types this session has already completed once. The first run
        # of a type pays import costs later ones don't (PySCF loads its
        # Hessian module on the first Frequency, for instance), so the
        # perf record carries a warm/cold flag rather than silently mixing
        # the two populations. See calc_log.log_calculation.
        self._warm_calc_types: set[str] = set()
        # Relaxed molecule from a pending pre-opt preview, awaiting Keep/Revert.
        self._preopt_relaxed_mol: Optional[Molecule] = None
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
        # call sites will be migrated to the router.
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
        self._mulliken_displayed_molecule: Any = None

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
        # Include the loaded viz backend preference so a "it reset" report
        # can be confirmed against what was actually persisted.
        try:
            _calc_log.log_event(
                "startup",
                f"QuantUI {quantui.__version__} started "
                f"(viz backend pref={self._viz_backend_preference})",
            )
        except OSError:
            pass

        # Kick off slow startup work (GPU detection, History/Compare loading)
        # off the synchronous construction path so the UI paints fast.
        self._start_deferred_startup_tasks()

    def _start_deferred_startup_tasks(self) -> None:
        """Run slow startup work AFTER widget construction so it doesn't block
        first paint.

        - **GPU detection** imports gpu4pyscf + cupy and queries CUDA (~7 s); it
          runs on a daemon thread, then re-renders the Status badge on the kernel
          loop. ``is_gpu_available`` is lru-cached, so the run dispatcher reuses
          the result with no extra cost.
        - **History + Compare** population loads every saved result; deferring it
          onto the kernel io loop lets it run right after the cell returns (UI
          already painted), and the summary sidecar keeps it fast. Falls back to
          inline when there is no kernel loop (tests / plain scripts).
        """

        def _detect_gpu() -> None:
            # Warm the run-header's system-info cache (lru_cache; may shell out
            # to nvidia-smi) off the main thread so the synchronous header write
            # in on_run_clicked stays instant on the first calc.
            try:
                from quantui.log_utils import get_system_info

                get_system_info()
            except Exception:  # noqa: BLE001 — warm-up is best-effort
                pass
            try:
                from quantui.gpu_offload import probe_gpu

                # probe_gpu (not is_gpu_available) so the badge can show the
                # actual reason offload isn't active instead of a generic
                # "not installed or no CUDA device".
                state = probe_gpu()
            except Exception:  # noqa: BLE001 — treat any failure as "no GPU"
                state = (False, None, "")
            render = getattr(self, "_render_status_html", None)
            html_widget = getattr(self, "_status_html", None)
            if render is None or html_widget is None:
                return

            def _apply() -> None:
                try:
                    html_widget.value = render(state)
                except Exception:
                    pass

            loop = self._get_kernel_io_loop()
            if loop is not None:
                loop.add_callback(_apply)
            else:
                _apply()

        threading.Thread(
            target=_detect_gpu, daemon=True, name="quantui-gpu-detect"
        ).start()

        loop = self._get_kernel_io_loop()
        if loop is not None:
            loop.add_callback(self._refresh_results_browser)
            loop.add_callback(self._populate_compare_list)
            # Startup is the moment that matters for CHK.6: after a restart
            # the targeted resume offer can't fire, because nothing is
            # configured yet.
            loop.add_callback(self._refresh_resume_list)
            loop.add_callback(_slurm_startup_check, self)
        else:
            self._refresh_results_browser()
            self._populate_compare_list()
            self._refresh_resume_list()
            _slurm_startup_check(self)

    def display(self) -> None:
        """Inject global CSS and render the application widget."""
        display(HTML(_APP_CSS))
        # NOTE: 3Dmol.js is loaded offline per-view via py3Dmol's own loader
        # (``viz_assets.make_view`` passes ``js=<vendored data: URI>``), NOT a
        # one-time page bootstrap. A startup-time bootstrap ran py3Dmol's
        # exports/module-juggling loader during Voilà's RequireJS bootstrap and
        # broke widget startup offline — never reintroduce it here.
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
                            self._exit_warn_html,
                            self._exit_cancel_btn,
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
        self._build_slurm_jobs_tab()
        self._build_files_tab()
        self._build_help_section()
        self._build_issue_widgets()

    # ── Theme selector ────────────────────────────────────────────────────

    def _build_theme_selector(self) -> None:
        _bld_build_theme_selector(self, layout_fn=_layout)

    def _theme_css(self, palette_id: str) -> str:
        """Return the CSS variable block for *palette_id*."""
        return _theme.theme_css_block(palette_id)

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

    def _on_root_tab_changed(self, change) -> None:
        """Pulse the activity light on tab navigation actions."""
        self._activity_pulse("Switching tabs...", hold_s=0.16, kind="ui")
        if change.get("new") == self._tab_index("analysis"):
            _ana_scroll_analysis_tab_to_top(self)
        if _slurm_jobs_tab_visible(self) and change.get("new") == self._tab_index(
            "slurm_jobs"
        ):
            _slurm_refresh_jobs_tab(self)

    def _go_to_calculate_tab(self) -> None:
        """Navigate to the Calculate tab."""
        self._activity_pulse("Navigating to Calculate tab...", hold_s=0.16, kind="ui")
        self.root_tab.selected_index = self._tab_index("calculate")

    def _go_to_results_tab(self, _btn) -> None:
        """Navigate to Results tab with a visible activity pulse."""
        self._activity_pulse("Navigating to Results tab...", hold_s=0.16, kind="ui")
        self.root_tab.selected_index = self._tab_index("results")

    def _go_to_analysis_tab(self, _btn) -> None:
        """Navigate to Analysis tab with a visible activity pulse."""
        self._activity_pulse("Navigating to Analysis tab...", hold_s=0.16, kind="ui")
        self.root_tab.selected_index = self._tab_index("analysis")
        _ana_scroll_analysis_tab_to_top(self)

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
            gpu_enabled=self._user_settings.compute.gpu_enabled,
            density_fit_enabled=self._user_settings.compute.density_fit,
            execution_backend=self._user_settings.compute.execution_backend,
            slurm_available=is_slurm_available(),
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

    def _on_raman_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_raman_freqs", None):
            self._update_raman_figure(
                self._raman_mode_toggle.value, self._raman_fwhm_slider.value
            )

    def _on_tddft_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_uv_wavelengths_nm", None):
            self._update_uv_vis_figure(
                self._uv_mode_toggle.value,
                self._uv_fwhm_slider.value,
            )

    def _on_nmr_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_nmr_atom_symbols", None):
            self._update_nmr_figure(self._nmr_nucleus_toggle.value)

    def _on_orb_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_orb_info", None) is not None:
            self._on_orb_range_changed()

    def _on_mulliken_accordion_show(self, change) -> None:
        if change["new"] == 0 and getattr(self, "_last_mulliken_charges", None):
            _ana_update_mulliken_figure(self)
            self._show_mulliken_viewer()

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
    #   • auto_select=True on the FIRST entry that returns True marks that
    #     panel as the primary scroll target; every panel with data expands.
    #   • If a populate method returns False / None the panel stays disabled.
    #   • Populate methods must NOT call _activate_ana_panel themselves.

    _PANEL_META: ClassVar[list] = [
        ("Energies", "_orb_accordion", "Single Point / Geometry Opt"),
        ("Populations", "_mulliken_accordion", "Single Point / Geometry Opt"),
        ("Trajectory", "traj_accordion", "Geometry Opt / PES Scan / Frequency pre-opt"),
        ("Vibrational", "vib_accordion", "Frequency"),
        ("IR Spectrum", "_ir_accordion", "Frequency"),
        ("Raman Spectrum", "_raman_accordion", "Frequency"),
        ("PES Scan", "_pes_scan_accordion", "PES Scan"),
        ("Isosurface", "_iso_accordion", "Single Point (Linux/WSL only)"),
        ("Geometries", "_reorg_geom_accordion", "Reorganization Energy"),
        ("UV-Vis", "_tddft_accordion", "UV-Vis (TD-DFT)"),
        ("NMR", "_nmr_accordion", "NMR Shielding"),
    ]

    _PANEL_REGISTRY: ClassVar[dict] = {
        "single_point": [
            ("Energies", "_pop_energies", True),
            ("Isosurface", "_pop_isosurface", False),
            ("Populations", "_pop_mulliken", False),
        ],
        "geometry_opt": [
            # ORDER MATTERS: the FIRST entry whose populator returns True and
            # carries auto_select=True becomes the default panel (see the rules
            # above). Isosurface therefore leads (requested 2026-08-04); an
            # earlier attempt put it last, which did nothing because Trajectory
            # had already claimed the selection.
            #
            # Trajectory keeps auto_select=True as the fallback: when a result
            # has no orbital data, _pop_isosurface returns False, Isosurface
            # never activates, and Trajectory becomes the default instead.
            # ORDER IS LOAD-BEARING TWICE OVER.
            #
            # 1. Execution: _pop_energies calls show_orbital_diagram, which is
            #    what populates _last_orb_mo_coeff / _mol_atom / _mol_basis —
            #    the very state _pop_isosurface checks. Energies MUST run
            #    first, or Isosurface reports "required data is missing" on a
            #    result that has it. (Putting Isosurface first did exactly
            #    that, 2026-08-04.)
            # 2. Selection: the FIRST entry with auto_select=True that returns
            #    True becomes the default panel. Energies is False, so
            #    Isosurface is the first candidate and opens by default —
            #    which is the request — while Trajectory keeps True as the
            #    fallback for results with no orbital data.
            ("Energies", "_pop_energies", False),
            ("Isosurface", "_pop_isosurface", True),
            ("Trajectory", "_pop_geo_trajectory", True),
            ("Populations", "_pop_mulliken", False),
        ],
        "frequency": [
            ("Vibrational", "_pop_vibrational", True),
            ("IR Spectrum", "_pop_ir_spectrum", True),
            ("Raman Spectrum", "_pop_raman_spectrum", False),
            ("Trajectory", "_pop_preopt_trajectory", False),
            ("Energies", "_pop_energies", True),
        ],
        "tddft": [
            ("UV-Vis", "_pop_uv_vis", True),
        ],
        "nmr": [
            ("NMR", "_pop_nmr_shielding", True),
        ],
        # reorganization_energy had NO entry at all until 2026-08-05, so the
        # Analysis tab populated nothing for these runs — not a missing panel,
        # no panels. Order matters twice over: the FIRST auto_select=True that
        # returns True wins, AND _pop_energies loads the orbital state
        # _pop_isosurface checks, so Energies must precede Isosurface.
        "reorganization_energy": [
            ("Energies", "_pop_energies", False),
            ("Geometries", "_pop_reorg_geometries", True),
            ("Isosurface", "_pop_isosurface", False),
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

    def _pop_reorg_geometries(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_reorg_geometries(self, ctx)

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

    def _pop_raman_spectrum(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_raman_spectrum(self, ctx)

    def _pop_uv_vis(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_uv_vis(self, ctx)

    def _pop_nmr_shielding(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_nmr_shielding(self, ctx)

    def _pop_mulliken(self, ctx: _AnalysisContext) -> bool:
        return _ana_pop_mulliken(self, ctx)

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

    def _build_slurm_jobs_tab(self) -> None:
        _bld_build_slurm_jobs_tab(self, layout_fn=_layout)

    def _build_files_tab(self) -> None:
        _bld_build_files_tab(self, layout_fn=_layout)
        self._refresh_file_browser()

    # ── Help section (Cell 12) ────────────────────────────────────────────

    def _build_help_section(self) -> None:
        _bld_build_help_section(self, layout_fn=_layout)

    def _build_issue_widgets(self) -> None:
        _bld_build_issue_widgets(self, layout_fn=_layout)

    # ── Tab assembly (Cell 10) ────────────────────────────────────────────

    def _root_tab_order(self) -> list[str]:
        order = ["calculate"]
        if _slurm_jobs_tab_visible(self):
            order.append("slurm_jobs")
        order.extend(["results", "analysis", "history", "compare", "files", "settings"])
        return order

    def _tab_index(self, name: str) -> int:
        return self._root_tab_order().index(name)

    def _tab_panel_for(self, name: str) -> Any:
        panels = {
            "calculate": self._calculate_tab_panel,
            "slurm_jobs": self.slurm_jobs_tab_panel,
            "results": self.results_tab_panel,
            "analysis": self.analysis_tab_panel,
            "history": self.history_panel,
            "compare": self.compare_panel,
            "files": self.files_tab_panel,
            "settings": self._status_tab_panel,
        }
        return panels[name]

    def _sync_root_tab_layout(self) -> None:
        """Insert or remove the Cluster Jobs tab when execution backend changes."""
        order = self._root_tab_order()
        prev_order = getattr(self, "_root_tab_order_cache", order)
        preserve_name: str | None = None
        try:
            old_idx = self.root_tab.selected_index
            if 0 <= old_idx < len(prev_order):
                preserve_name = prev_order[old_idx]
        except Exception:  # noqa: BLE001 — tab index is cosmetic
            preserve_name = None

        self.root_tab.children = tuple(self._tab_panel_for(name) for name in order)
        title_map = {
            "calculate": "Calculate",
            "slurm_jobs": "Cluster Jobs",
            "results": "Results",
            "analysis": "Analysis",
            "history": "History",
            "compare": "Compare",
            "files": "Files",
            "settings": "System Settings",
        }
        for index, name in enumerate(order):
            self.root_tab.set_title(index, title_map[name])

        self._slurm_jobs_tab_index = (
            order.index("slurm_jobs") if "slurm_jobs" in order else None
        )
        self._root_tab_order_cache = list(order)

        if "slurm_jobs" in order:
            _slurm_refresh_jobs_tab(self)
            from quantui.app_slurm import _update_slurm_jobs_tab_title

            _update_slurm_jobs_tab_title(self)

        if preserve_name in order:
            self.root_tab.selected_index = order.index(preserve_name)

    def _assemble_tabs(self) -> None:
        self._calculate_tab_panel = widgets.VBox(
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

        self._slurm_jobs_tab_index = None
        self._root_tab_order_cache: list[str] = []
        self.root_tab = widgets.Tab(children=())
        self._sync_root_tab_layout()
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
        # Settings → GPU offload on/off (Status tab; persisted).
        self.gpu_enabled_cb.observe(
            self._safe_cb(self._on_gpu_enabled_changed), names="value"
        )
        # Settings → density fitting (RI) on/off (Status tab; persisted).
        self.density_fit_enabled_cb.observe(
            self._safe_cb(self._on_density_fit_enabled_changed), names="value"
        )
        self.execution_backend_dd.observe(
            self._safe_cb(self._on_execution_backend_changed), names="value"
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
        # Molecule input — library browse/search
        self.lib_category_dd.observe(
            self._safe_cb(self._on_lib_filter_changed), names="value"
        )
        self.lib_search_txt.observe(
            self._safe_cb(self._on_lib_filter_changed), names="value"
        )
        self.lib_results_dd.observe(self._safe_cb(self._on_lib_select), names="value")
        self.xyz_btn.on_click(self._on_load_xyz)
        self.xyz_add_atom_btn.on_click(self._on_xyz_add_atom)
        self.xyz_fill_table_btn.on_click(self._on_xyz_fill_table)
        self.xyz_apply_table_btn.on_click(self._on_xyz_apply_table)
        self.xyz_cleanup_btn.on_click(self._on_xyz_cleanup)
        self.xyz_cleanup_accept_btn.on_click(self._on_xyz_cleanup_accept)
        self.xyz_cleanup_reject_btn.on_click(self._on_xyz_cleanup_reject)
        self.pubchem_btn.on_click(self._on_search_pubchem)
        self.pubchem_candidates_dd.observe(
            self._safe_cb(self._on_pubchem_candidate_selected), names="value"
        )
        self.change_mol_btn.on_click(self._on_expand_mol_input)
        # Calc type
        self.calc_type_dd.observe(
            self._safe_cb(self._on_calc_type_changed), names="value"
        )
        # Geometry Opt / Frequency / UV-Vis (TD-DFT) share one seed-geometry
        # dropdown + refresh button — only one observer/click binding is needed,
        # not three, since `_geo_seed_dd`, `_freq_seed_dd` and `_tddft_seed_dd`
        # are the same underlying widget.
        self._seed_dd.observe(self._safe_cb(self._on_seed_changed), names="value")
        self._seed_refresh_btn.on_click(lambda _btn: self._refresh_seed_options())
        self._scan_type_dd.observe(
            self._safe_cb(self._update_scan_widgets), names="value"
        )
        # Notes + estimate
        self.method_dd.observe(self._safe_cb(self._update_notes), names="value")
        self.basis_dd.observe(self._safe_cb(self._update_notes), names="value")
        # Multiplicity drives the open-shell hint (part of _update_notes).
        self.mult_si.observe(self._safe_cb(self._update_notes), names="value")
        # Keep the active molecule's charge/multiplicity in step with the fields,
        # so an edit here (or the spin-state helper's Apply) actually reaches the
        # run — the calc reads mol.charge/mol.multiplicity, and the pre-run guard
        # reads the widgets, so the two must not drift apart.
        self.charge_si.observe(
            self._safe_cb(self._sync_charge_to_molecule), names="value"
        )
        self.mult_si.observe(self._safe_cb(self._sync_mult_to_molecule), names="value")
        self.method_dd.observe(self._safe_cb(self._update_estimate), names="value")
        self.basis_dd.observe(self._safe_cb(self._update_estimate), names="value")
        # Unfinished-calculations list (CHK.6)
        self._resume_list_dd.observe(
            self._safe_cb(self._on_resume_entry_changed), names="value"
        )
        self._resume_restore_btn.on_click(self._safe_cb(self._on_resume_restore))
        self._resume_discard_btn.on_click(self._safe_cb(self._on_resume_discard))
        # Help buttons
        self.method_help_btn.on_click(self._on_method_help)
        self.basis_help_btn.on_click(self._on_basis_help)
        self.calc_type_help_btn.on_click(self._on_calc_type_help)
        # Run
        self.run_btn.on_click(self._on_run_clicked)
        self._slurm_reconnect_btn.on_click(
            self._safe_cb(lambda _btn: _slurm_on_reconnect_clicked(self, _btn))
        )
        self._slurm_jobs_refresh_btn.on_click(
            self._safe_cb(lambda _btn: _slurm_on_jobs_refresh_clicked(self, _btn))
        )
        self._slurm_jobs_view_btn.on_click(
            self._safe_cb(lambda _btn: _slurm_on_jobs_view_clicked(self, _btn))
        )
        self._slurm_jobs_cancel_btn.on_click(
            self._safe_cb(lambda _btn: _slurm_on_jobs_cancel_clicked(self, _btn))
        )
        self._slurm_jobs_remove_btn.on_click(
            self._safe_cb(lambda _btn: _slurm_on_jobs_remove_clicked(self, _btn))
        )
        self.cancel_btn.on_click(self._safe_cb(self._on_cancel))
        self.basis_fix_btn.on_click(self._safe_cb(self._on_basis_fix))
        self.charge_mult_suggest_btn.on_click(
            self._safe_cb(self._on_charge_mult_suggest)
        )
        self.charge_mult_apply_btn.on_click(self._safe_cb(self._on_charge_mult_apply))
        self.spin_suggest_btn.on_click(self._safe_cb(self._on_spin_suggest))
        self.spin_apply_btns[0].on_click(
            self._safe_cb(lambda _b: self._on_spin_apply(0))
        )
        self.spin_apply_btns[1].on_click(
            self._safe_cb(lambda _b: self._on_spin_apply(1))
        )
        self.preopt_preview_btn.on_click(self._safe_cb(self._on_preopt_preview))
        self.preopt_accept_btn.on_click(self._safe_cb(self._on_preopt_accept))
        self.preopt_reset_btn.on_click(self._safe_cb(self._on_preopt_reset))
        self.log_clear_btn.on_click(self._on_clear_log)
        self._ir_mode_toggle.observe(
            self._safe_cb(self._on_ir_mode_changed), names="value"
        )
        self._ir_fwhm_slider.observe(
            self._safe_cb(self._on_ir_fwhm_changed), names="value"
        )
        self._raman_mode_toggle.observe(
            self._safe_cb(self._on_raman_mode_changed), names="value"
        )
        self._raman_fwhm_slider.observe(
            self._safe_cb(self._on_raman_fwhm_changed), names="value"
        )
        self._uv_mode_toggle.observe(
            self._safe_cb(self._on_uv_mode_changed), names="value"
        )
        self._uv_fwhm_slider.observe(
            self._safe_cb(self._on_uv_fwhm_changed), names="value"
        )
        self._uv_xmin_input.observe(
            self._safe_cb(self._on_uv_range_changed), names="value"
        )
        self._uv_xmax_input.observe(
            self._safe_cb(self._on_uv_range_changed), names="value"
        )
        self._nmr_nucleus_toggle.observe(
            self._safe_cb(self._on_nmr_nucleus_changed), names="value"
        )
        self._ir_export_btn.on_click(self._on_ir_export_plot)
        self._raman_export_btn.on_click(self._on_raman_export_plot)
        self._uv_export_btn.on_click(self._on_uv_export_plot)
        self._nmr_export_btn.on_click(self._on_nmr_export_plot)
        self._orb_export_btn.on_click(self._on_orb_export_plot)
        self._pes_export_btn.on_click(self._on_pes_export_plot)
        self._vib_export_btn.on_click(self._on_vib_export_animation)
        # Per-panel CSV-to-clipboard / file buttons.
        self._ir_copy_data_btn.on_click(self._on_ir_copy_data)
        self._raman_copy_data_btn.on_click(self._on_raman_copy_data)
        self._uv_copy_data_btn.on_click(self._on_uv_copy_data)
        self._nmr_copy_data_btn.on_click(self._on_nmr_copy_data)
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
        self._reorg_export_btn.on_click(self._safe_cb(self._on_export_reorg_geometries))
        # History
        self.past_dd.observe(self._safe_cb(self._on_past_dd_changed), names="value")
        self.past_refresh_btn.on_click(self._on_past_refresh)
        self.copy_path_btn.on_click(self._on_copy_results_path)
        self.view_log_btn.on_click(self._on_view_log)
        # History search / faceted filters (HIST.7)
        for _w in (
            self.history_search,
            self.history_method_dd,
            self.history_basis_dd,
            self.history_date_from,
            self.history_date_to,
        ):
            _w.observe(self._safe_cb(self._on_history_filter_changed), names="value")
        for _chip in (
            *self._history_calc_chips.values(),
            *self._history_status_chips.values(),
        ):
            _chip.observe(self._safe_cb(self._on_history_filter_changed), names="value")
        self.history_filter_clear_btn.on_click(self._on_history_filter_clear)
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
        self._exit_cancel_btn.on_click(self._on_exit_cancel)
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
        self._vib_apply_mode_btn.on_click(
            self._safe_cb(lambda _btn: _run_apply_vib_mode_for_frequency(self))
        )
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
        # Reveal the free-entry MO-index input only in "By index" mode.
        self._orb_toggle.observe(
            self._safe_cb(self._on_orb_toggle_changed), names="value"
        )
        # Cube + bundle exports
        self._iso_export_cube_btn.on_click(self._on_iso_export_cube)
        self._iso_cancel_btn.on_click(self._safe_cb(self._on_iso_cancel))
        # Reorg geometry views (REORG.3): both redraw from data already in
        # memory, so they respond directly rather than behind an Apply button.
        for _w in (
            self._reorg_view_toggle,
            self._reorg_overlay_pair,
            self._reorg_exaggerate,
        ):
            _w.observe(self._safe_cb(self._on_reorg_view_changed), names="value")
        # PNG capture arrives from the browser, so there is no button to bind
        # here — the viewer's own Save-PNG button posts into this Textarea and
        # ipywidgets syncs it back, firing this observer (ORBX.1).
        self._orb_png_inbox.observe(
            self._safe_cb(self._on_orb_png_captured), names="value"
        )
        # Same bridge, own inbox — the reorg-geometry viewer's Save-PNG button
        # (M-EXPORT2 EXP2.2).
        self._reorg_png_inbox.observe(
            self._safe_cb(self._on_reorg_png_captured), names="value"
        )
        # Same bridge again — the vibrational single-viewer's Save-PNG button
        # (M-EXPORT2 EXP2.2). Only fires for that path; the legacy per-mode
        # plotlymol3d fallback never renders the button.
        self._vib_png_inbox.observe(
            self._safe_cb(self._on_vib_png_captured), names="value"
        )
        # Molecule (top) viewer — three independent Save-PNG buttons
        # (M-EXPORT2 EXP2.2), one per output slot (Calculate/Results/
        # Analysis). Only fires for the py3Dmol backend; see
        # visualization_py3dmol.render_molecule_html's capture_class
        # docstring for why plotlymol has no equivalent button.
        self._mol_calc_png_inbox.observe(
            self._safe_cb(self._on_mol_calc_png_captured), names="value"
        )
        self._mol_results_png_inbox.observe(
            self._safe_cb(self._on_mol_results_png_captured), names="value"
        )
        self._mol_analysis_png_inbox.observe(
            self._safe_cb(self._on_mol_analysis_png_captured), names="value"
        )
        # Click-to-measure (M-MEASURE MEAS.2/3): a click in the Analysis-tab
        # viewer posts an atom index into this inbox the same way the PNG
        # buttons post a data URI.
        self._measure_inbox.observe(
            self._safe_cb(self._on_measure_inbox_changed), names="value"
        )
        self._measure_clear_btn.on_click(self._on_measure_clear)
        self._measure_help_btn.on_click(self._on_measure_help)
        self._mulliken_help_btn.on_click(self._on_mulliken_help)
        self._mulliken_color_cb.observe(
            self._safe_cb(self._on_mulliken_overlay_changed), names="value"
        )
        self._mulliken_dipole_cb.observe(
            self._safe_cb(self._on_mulliken_overlay_changed), names="value"
        )
        self._mulliken_vividness_slider.observe(
            self._safe_cb(self._on_mulliken_overlay_changed), names="value"
        )
        # Persist the grid choice so it survives a relaunch (ORBX.2).
        self._iso_resolution_dd.observe(
            self._safe_cb(self._on_iso_resolution_changed), names="value"
        )
        # Appearance controls redraw from the cube on disk — no cubegen — so
        # they can respond directly rather than behind an Apply button.
        for _w in (
            self._iso_isovalue_slider,
            self._iso_opacity_slider,
            self._iso_wireframe_cb,
            self._iso_colors_dd,
            # NOT _iso_png_transparent: it is an export-only option, applied at
            # capture time. Observing it here would change the live viewer.
        ):
            _w.observe(self._safe_cb(self._on_iso_appearance_changed), names="value")
        self._export_bundle_btn.on_click(self._on_export_bundle)

    # ── Files tab ────────────────────────────────────────────────────────

    def _files_allowed_roots(self) -> list[Path]:
        """Return the approved filesystem roots for the Files tab."""
        roots: list[Path] = []
        candidates: list[Optional[Path]] = [self._get_results_dir(), Path.cwd()]
        _last_dir = getattr(self, "_last_result_dir", None)
        if isinstance(_last_dir, Path):
            candidates.append(_last_dir)
        # Expose the app's own log dir (~/.quantui/logs) so the event
        # log is reachable in-app. Resolves inside the runtime process, so it
        # correctly points at the WSL home when the app runs under WSL.
        try:
            candidates.append(_calc_log._log_dir())
        except Exception:
            pass

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

    def _set_files_status(
        self, message: str, color: str = _theme.css.TEXT_SLATE
    ) -> None:
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
        try:
            labels.append(("Logs", _calc_log._log_dir().resolve()))
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
                f'<span style="font-size:12px;color:{_theme.css.TEXT_SLATE}">'
                "Current folder: unavailable</span>"
            )
            self._files_open_btn.disabled = True
            self._files_up_btn.disabled = True
            self._set_files_status(
                "No readable roots available.", _theme.css.ACCENT_ERROR
            )
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
            f'<span style="font-size:12px;color:{_theme.css.TEXT_SLATE_DARK}">Current folder: '
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
            self._set_files_status(
                f"Cannot list folder: {exc}", _theme.css.ACCENT_ERROR
            )
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
            self._set_files_status(
                "Selected path is outside allowed roots.", _theme.css.ACCENT_ERROR
            )
            return
        if not path.exists() or not path.is_file():
            self._set_files_status(
                "Selected file no longer exists.", _theme.css.ACCENT_ERROR
            )
            return

        self._files_preview_output.clear_output(wait=True)
        suffix = path.suffix.lower()

        image_ext = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        text_ext = {
            ".txt",
            ".log",
            ".json",
            ".jsonl",
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

        # Specialized previews for
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
                from quantui.viz_assets import make_view

                model_format = {".xyz": "xyz", ".mol": "mol", ".pdb": "pdb"}[suffix]
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                if len(raw_text) <= 256_000:
                    viewer = make_view(width=500, height=380)
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

        if suffix == ".jsonl":
            # JSONL logs (event_log.jsonl) grow append-only, so the
            # newest — and most useful — records are at the END. The generic
            # text dispatch keeps the FIRST 200 KB (oldest events), so tail
            # the file here instead: show the last N lines, newest last.
            try:
                _MAX_TAIL_LINES = 300
                raw = path.read_bytes()
                total_bytes = len(raw)
                # Cap the decode window so a multi-MB log stays responsive;
                # the tail is all we render anyway.
                tail_raw = raw[-400_000:]
                text = tail_raw.decode("utf-8", errors="replace")
                lines = text.splitlines()
                # A leading partial line can appear after byte-slicing — drop it.
                if len(tail_raw) < total_bytes and lines:
                    lines = lines[1:]
                total_lines_shown = min(len(lines), _MAX_TAIL_LINES)
                shown = lines[-_MAX_TAIL_LINES:]
                rendered = "\n".join(shown)
                note = (
                    f"Showing the last {total_lines_shown} record(s)"
                    + (
                        " (file tail — older records not shown)"
                        if len(lines) > _MAX_TAIL_LINES or len(tail_raw) < total_bytes
                        else ""
                    )
                    + "."
                )
                with self._files_preview_output:
                    display(
                        HTML(
                            f"<p style='font-size:11px;color:{_theme.css.TEXT_SLATE};margin:0 0 4px'>"
                            f"{_html.escape(note)}</p>"
                            "<pre style='white-space:pre-wrap;word-break:break-word;"
                            "font-size:12px;line-height:1.35;margin:0'>"
                            f"{_html.escape(rendered)}</pre>"
                        )
                    )
                self._set_files_status(f"Log preview (tail): {path.name}")
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
                        f"border-bottom:1px solid {_theme.css.BORDER};font-size:12px;"
                        f'color:{_theme.css.TEXT_STRONG}">{_html.escape(str(c))}</th>'
                        for c in header
                    )
                    body_html = "".join(
                        "<tr>"
                        + "".join(
                            f'<td style="padding:3px 10px;font-size:12px;'
                            f"border-bottom:1px solid #f1f5f9;color:{_theme.css.TEXT_BODY};"
                            f'font-variant-numeric:tabular-nums">{_html.escape(str(c))}</td>'
                            for c in r
                        )
                        + "</tr>"
                        for r in body
                    )
                    note = (
                        f'<p style="font-size:11px;color:{_theme.css.TEXT_SUBTLE};margin:4px 0 6px">'
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
                html_text = path.read_text(encoding="utf-8", errors="replace")
                if len(html_text) <= 1_000_000:
                    # Sandboxed iframe via srcdoc — embedded JS can't
                    # reach the parent app.
                    iframe_html = (
                        '<iframe sandbox="allow-scripts" '
                        f'style="width:100%;height:400px;border:1px solid {_theme.css.BORDER};'
                        'border-radius:4px" '
                        f'srcdoc="{_html.escape(html_text, quote=True)}"></iframe>'
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
                    f'<p style="font-size:13px;color:{_theme.css.TEXT_SLATE_DARK};margin:0 0 6px">'
                    f"<b>Cube file:</b> {_html.escape(path.name)} "
                    f"&middot; {size_mb:.2f} MB</p>"
                    f'<p style="font-size:12px;color:{_theme.css.TEXT_SLATE};margin:0 0 6px">'
                    "Use the <b>Analysis</b> tab's Orbital Isosurface panel to "
                    "render volumetric data; the raw file is too large to "
                    "preview inline.</p>"
                    f'<p style="font-size:11px;color:{_theme.css.TEXT_SUBTLE};margin:6px 0 4px">'
                    "Header (first 6 lines):</p>"
                    '<pre style="white-space:pre-wrap;font-size:11px;'
                    f"line-height:1.35;margin:0;background:{_theme.css.BG_PANEL};padding:6px;"
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
                self._set_files_status(
                    f"Cannot read file: {exc}", _theme.css.ACCENT_ERROR
                )
                return
            is_text = b"\x00" not in sample

        if not is_text:
            with self._files_preview_output:
                display(
                    HTML(
                        f"<p style='font-size:12px;color:{_theme.css.TEXT_SLATE_DARK};margin:4px 0'>"
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
            self._set_files_status(f"Cannot read file: {exc}", _theme.css.ACCENT_ERROR)
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
            self._set_files_status(
                "Selected root is not allowed.", _theme.css.ACCENT_ERROR
            )
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
                self._set_files_status(
                    "No current folder selected.", _theme.css.ACCENT_ERROR
                )
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
        palette_id = change["new"]
        if palette_id not in _theme.PALETTE_IDS:
            return
        self._user_settings.theme.palette = palette_id
        self._user_settings.save()
        self._theme_style.clear_output()
        with self._theme_style:
            display(HTML(self._theme_css(palette_id)))
        self._rerender_plotly_theme()

    def _plotly_theme_colors(self) -> dict:
        """Return plot colours for the active palette."""
        return _theme.plotly_colors(self.theme_btn.value)

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
        users were seeing on IR Stick/Broadened toggle and FWHM slider drag,
        and matches the atomic-swap pattern already used by
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
        that surfaced three viewer issues (2026-05-25):

        - "viewer doesn't update on PubChem load until I toggle the
          backend" — the Output-context render path was racing the kernel's
          comms flush, so the initial display was sometimes never emitted.
          Atomic ``outputs = (display_data,)`` is a single synchronous
          assignment that the front-end always picks up.
        - "red log lines around the viewer on the Calculate tab" —
          ``with self.viz_output:`` captured every ``logger.info`` /
          ``logger.error`` line that ``display_molecule`` emitted while it
          ran. ``render_molecule_html`` returns the HTML string OUTSIDE any
          Output context, so the only thing that lands in the widget is
          the viewer itself.
        - "PlotlyMol valence error spills as red text" — the same
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
            capture_class=_MOL_CALC_PNG_INBOX_CLASS,
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
        if getattr(self, "_last_mulliken_charges", None):
            _ana_update_mulliken_figure(self)
        _last_pes = getattr(self, "_last_pes_result", None)
        if _last_pes is not None:
            self._show_pes_scan_result(_last_pes)
        # Re-render BOTH 3D molecule viewers so scene_bgcolor updates
        # immediately. This used to call _refresh_calc_mol_viewer directly,
        # which covered the Calculate tab only — the Analysis-tab viewer kept
        # its old background until something else happened to redraw it.
        # _rerender_3d_views already handled both; it just was not reached from
        # here. Reported 2026-08-04.
        self._rerender_3d_views()
        # ...and the isosurface / vibrational viewers, which bake the same
        # colour into their generated HTML. Without this they keep the old
        # background until the user regenerates — reported 2026-08-04. Both
        # re-render from cached inputs; neither re-runs a calculation.
        _viz_rerender_3d_scenes_for_theme(self)

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

        ``new_pref`` must be one of "auto" | "py3dmol" | "plotlymol". All three
        widgets (Settings dropdown + Calculate/Analysis toggles) edit the same
        single global preference, so every user-initiated change passes
        ``persist=True`` (a backend choice must survive the session).
        ``persist=False`` remains available for programmatic syncs that must
        not write settings.

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
        if _render_molecule_html is None:
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
                # Same render path as the first draw (app_visualization.
                # _show_result_3d), not display(): the fragment it returns
                # carries the viewer's own border. Going through display()
                # here would silently drop that border the moment a user
                # toggled backends on the Analysis tab.
                html = _render_molecule_html(
                    self._analysis_displayed_molecule,
                    # VizBackend is a StrEnum whose only members are
                    # "py3dmol"/"plotlymol", a subset of render_molecule_html's
                    # accepted Literal — cast documents that, str() alone widens
                    # to plain str for mypy.
                    backend=cast(Literal["auto", "py3dmol", "plotlymol"], str(chosen)),
                    style=self._viz_style,
                    lighting=self._viz_lighting,
                    bgcolor=self._plotly_theme_colors()["scene_bgcolor"],
                    capture_class=_MOL_ANALYSIS_PNG_INBOX_CLASS,
                )
                # M-MEASURE: same wiring as the post-calc render path — a
                # backend-toggle switch is still a fresh viewer, so stale
                # picks from the previous backend must not survive it.
                from quantui.app_measurement import finalize_analysis_html

                html = finalize_analysis_html(self, html, chosen)
                self._set_html_output(self._analysis_mol_output, html)
                self._update_analysis_backend_label(chosen)

    def _show_mulliken_viewer(self, molecule=None) -> None:
        from quantui.populations_overlay import show_mulliken_viewer

        show_mulliken_viewer(
            self,
            molecule,
            render_html_fn=_render_molecule_html,
        )

    def _update_analysis_backend_label(self, chosen: VizBackend) -> None:
        """Update the small 'Rendering with: X' label next to the Analysis
        molecule viewer. No-op if the label widget does not exist (built only
        when both backends are available)."""
        label = getattr(self, "viz_backend_label_ana", None)
        if label is None:
            return
        display_name = "py3Dmol" if chosen == VizBackend.PY3DMOL else "plotlymol3d"
        label.value = (
            f'<span style="font-size:11px;color:{_theme.css.TEXT_SUBTLE};font-style:italic">'
            f"Rendering with: {display_name}</span>"
        )

    def _on_viz_backend_changed(self, change) -> None:
        """Calculate-tab toggle observer — persists the chosen backend.

        Previously ``persist=False`` (session-only). That surprised users: the
        toggle visibly updated every view + the Settings dropdown, but the
        choice silently reverted next session (2026-06-15).
        All three widgets edit the one global preference, so any user-initiated
        change now persists.
        """
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=True)

    def _on_viz_backend_changed_ana(self, change) -> None:
        """Analysis-tab toggle observer — persists the chosen backend."""
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=True)

    def _on_viz_default_backend_changed(self, change) -> None:
        """Settings widget observer — persistent preference change."""
        if self._viz_sync_in_progress:
            return
        self._set_viz_preference(change["new"], persist=True)

    def _on_gpu_enabled_changed(self, change) -> None:
        """Persist the GPU-offload preference and re-probe immediately.

        The detection probe is cached for the process lifetime, so flipping this
        without clearing the cache would leave the next run using the old
        decision — the toggle would appear to do nothing until restart.
        """
        new_val = bool(change["new"])
        if new_val == self._user_settings.compute.gpu_enabled:
            return
        self._user_settings.compute.gpu_enabled = new_val
        self._user_settings.save()
        try:
            from quantui.gpu_offload import is_gpu_available, probe_gpu

            # cache_clear is forwarded from _probe_gpu's lru_cache onto this
            # function at definition time (gpu_offload.py); mypy can't see a
            # monkey-patched attribute across the module boundary.
            is_gpu_available.cache_clear()  # type: ignore[attr-defined]
            state = probe_gpu()
        except Exception:  # noqa: BLE001 — a probe failure must not break the UI
            state = (False, None, "")
        # Refresh the Status badge so it reflects the new decision right away.
        render = getattr(self, "_render_status_html", None)
        html_widget = getattr(self, "_status_html", None)
        if render is not None and html_widget is not None:
            try:
                html_widget.value = render(state)
            except Exception:  # noqa: BLE001 — best-effort badge refresh
                pass
        try:
            _calc_log.log_event("gpu_enabled_changed", f"gpu_enabled={new_val}")
        except OSError:
            pass

    def _on_density_fit_enabled_changed(self, change) -> None:
        """Persist the density-fitting (RI) preference.

        No cache to clear: the SCF path reads ``compute.density_fit`` fresh from
        settings on each run (:func:`quantui.density_fitting.try_density_fit`).
        Refresh the time estimate, since DF changes which history pool the
        estimator draws from.
        """
        new_val = bool(change["new"])
        if new_val == self._user_settings.compute.density_fit:
            return
        self._user_settings.compute.density_fit = new_val
        self._user_settings.save()
        try:
            self._update_estimate()
        except Exception:  # noqa: BLE001 — best-effort estimate refresh
            pass
        try:
            _calc_log.log_event("density_fit_enabled_changed", f"density_fit={new_val}")
        except OSError:
            pass

    def _on_execution_backend_changed(self, change) -> None:
        """Persist local vs SLURM batch execution preference."""
        new_val = str(change["new"])
        if new_val == self._user_settings.compute.execution_backend:
            return
        self._user_settings.compute.execution_backend = new_val
        self._user_settings.save()
        try:
            _calc_log.log_event("execution_backend_changed", f"backend={new_val}")
        except OSError:
            pass
        self._sync_root_tab_layout()

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
        # Single-viewer path: update the running animation's interval in place
        # (no rebuild, camera preserved). The legacy per-mode path re-renders.
        if getattr(self, "_vib_single_viewer_active", False):
            from quantui.app_visualization import _vib_bridge_set_fps

            _vib_bridge_set_fps(self, new_fps)
            return
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
            f'<span style="color:{_theme.css.TEXT_FAINT};font-size:12px">{note}</span>'
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
        _xyz_on_load_xyz(self, btn)

    def _on_xyz_add_atom(self, btn) -> None:
        _xyz_on_add_atom(self, btn)

    def _on_xyz_fill_table(self, btn) -> None:
        _xyz_on_fill_table(self, btn)

    def _on_xyz_apply_table(self, btn) -> None:
        _xyz_on_apply_table(self, btn)

    def _on_xyz_cleanup(self, btn) -> None:
        _xyz_on_cleanup(self, btn)

    def _on_xyz_cleanup_accept(self, btn) -> None:
        _xyz_on_cleanup_accept(self, btn)

    def _on_xyz_cleanup_reject(self, btn) -> None:
        _xyz_on_cleanup_reject(self, btn)

    def _apply_pubchem_search_result(
        self,
        query: str,
        mol: Optional[Molecule] = None,
        error: Optional[Exception] = None,
        source: Optional[str] = None,
    ) -> None:
        # Runs on the main loop. Terminal point of a search → end the activity
        # indicator started in the search handler. Best-effort + idempotent
        # via the counter, so an unbalanced begin can't pin the light "busy".
        try:
            self._activity_end(kind="ui")
        except Exception:
            pass
        if error is None and mol is not None:
            # Label by where the structure ACTUALLY came from — not always
            # "PubChem". Offline FALLBACK means the network was tried + failed,
            # so surface a no-network note; a plain library hit is not an error.
            prefix = _STRUCT_SOURCE_PREFIX.get(source or "", "Structure")
            self._set_molecule(mol, f"{prefix}: {query}")
            msg = f"Loaded {mol.get_formula()} from {prefix}."
            if source == "library-offline-fallback":
                msg = (
                    "⚠ No network detected — resolved offline from the bundled "
                    f"library (not PubChem). {msg}"
                )
            # MET.2: a fetched name can resolve to a disconnected ionic salt
            # (cisplatin → 2 NH₃ + 2 HCl + Pt²⁺) rather than the coordinated
            # complex. Warn rather than let a wrong geometry silently feed a run.
            try:
                from .connectivity import describe_disconnection

                warning = describe_disconnection(mol.atoms, mol.coordinates)
            except Exception:  # noqa: BLE001 — a detection failure must not block load
                warning = None
            if warning:
                msg = f"⚠ {warning} {msg}"
            self.pubchem_msg.value = msg
        else:
            self.pubchem_msg.value = f"Not found: {error}"
            try:
                _calc_log.log_event(
                    "structure_search_failed",
                    f"Structure query not found: '{query}'",
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
            xyz_str, _msg, source, _is_offline = _resolve_structure_with_message(query)
            if xyz_str is None:
                raise ValueError(_msg)
            atoms, coords = parse_xyz_input(xyz_str)
            mol = Molecule(atoms=atoms, coordinates=coords)
            loop.call_soon_threadsafe(
                self._apply_pubchem_search_result, query, mol, None, source
            )
        except Exception as exc:
            loop.call_soon_threadsafe(
                self._apply_pubchem_search_result, query, None, exc, None
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
        # Terminal point of the search phase (awaiting the user's pick) → end
        # the activity indicator started in _on_search_pubchem.
        try:
            self._activity_end(kind="ui")
        except Exception:
            pass

    def _on_pubchem_candidate_selected(self, change) -> None:
        if getattr(self, "_pubchem_cand_refreshing", False):
            return
        cid = change["new"]
        if not cid:
            return
        self.pubchem_msg.value = f"Loading CID {cid}…"
        self.pubchem_btn.disabled = True
        try:
            self._activity_begin(f"Loading CID {cid}…", kind="ui")
        except Exception:
            pass
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
        # Light the toolbar activity indicator so the resolver chain
        # (PubChem → CACTUS → library, up to ~8 s on a CACTUS timeout) doesn't
        # look like a hang. Ended at every terminal point below.
        try:
            self._activity_begin(f'Searching structures for "{query}"…', kind="ui")
        except Exception:
            pass

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

    def _refresh_seed_options(self) -> None:
        _run_refresh_seed_options(self)

    def _on_seed_changed(self, change) -> None:
        _run_on_seed_changed(self, change)

    # Geometry Opt / Frequency / UV-Vis (TD-DFT) used to each have their own
    # seed-refresh + seed-changed methods; the widget they operate on is now
    # one shared dropdown, so these are aliases of the two methods above rather
    # methods above rather than separate implementations. Kept under their
    # original names because app_runflow.py's per-calc-type branches and the
    # existing per-calc-type tests still call them by these names.
    _refresh_geo_seed_options = _refresh_seed_options
    _refresh_freq_seed_options = _refresh_seed_options
    _refresh_tddft_seed_options = _refresh_seed_options
    _on_geo_seed_changed = _on_seed_changed
    _on_freq_seed_changed = _on_seed_changed
    _on_tddft_seed_changed = _on_seed_changed

    # ── Help buttons ──────────────────────────────────────────────────────

    def _on_method_help(self, btn) -> None:
        _run_on_method_help(self, btn)

    def _on_basis_help(self, btn) -> None:
        _run_on_basis_help(self, btn)

    def _on_calc_type_help(self, btn) -> None:
        _run_on_calc_type_help(self, btn)

    # ── Run ───────────────────────────────────────────────────────────────

    def _on_run_clicked(self, btn) -> None:
        self._activity_pulse(
            "Queueing calculation...",
            hold_s=0.18,
            kind="compute",
        )
        _run_on_run_clicked(self, btn)

    def _on_cancel(self, btn=None) -> None:
        """Request graceful cancellation of the in-flight calculation.

        Sets the cancel event (checked by the run's _LogCapture each output
        line) and disables the button so a second click can't re-fire. The
        actual stop + UI reset happens in ``_do_run`` when the next write raises
        ``_CalcCancelled``.

        For SLURM batch jobs, requests ``scancel`` and stops the monitor thread.
        """
        if getattr(self, "_slurm_active_request_id", None):
            from quantui.app_slurm import cancel_slurm_run

            if cancel_slurm_run(self):
                self.cancel_btn.disabled = True
                self.cancel_btn.description = "Cancelling…"
            return
        if not self._calc_running:
            return
        self._cancel_event.set()
        self.cancel_btn.disabled = True
        self.cancel_btn.description = "Cancelling…"
        self.run_status.value = "Cancelling — stopping at the next cycle/step…"
        # Write an immediate marker to the live log so the click reads as
        # acknowledged even if the next cooperative checkpoint is a moment away.
        try:
            self.run_output.append_stdout(
                "\n⏹ Cancel requested — stopping at the next SCF cycle / "
                "optimizer step…\n"
            )
        except Exception:
            pass

    def _on_basis_fix(self, btn=None) -> None:
        _run_on_basis_fix(self, btn)

    def _on_charge_mult_suggest(self, btn=None) -> None:
        _run_on_charge_mult_suggest(self, btn)

    def _on_charge_mult_apply(self, btn=None) -> None:
        _run_on_charge_mult_apply(self, btn)

    def _on_spin_suggest(self, btn=None) -> None:
        _run_on_spin_suggest(self, btn)

    def _on_spin_apply(self, index: int) -> None:
        _run_on_spin_apply(self, index)

    def _on_preopt_preview(self, btn=None) -> None:
        _run_on_preopt_preview(self, btn)

    def _on_preopt_accept(self, btn=None) -> None:
        _run_on_preopt_accept(self, btn)

    def _on_preopt_reset(self, btn=None) -> None:
        _run_on_preopt_reset(self, btn)

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

    def _on_export_reorg_geometries(self, btn) -> None:
        _exp_on_export_reorg_geometries(self, btn)

    def _on_reorg_view_changed(self, change) -> None:
        _ana_on_reorg_view_changed(self, change)

    def _on_iso_cancel(self, btn) -> None:
        _viz_on_iso_cancel(self, btn)

    def _on_iso_appearance_changed(self, change) -> None:
        _viz_on_iso_appearance_changed(self, change)

    def _on_orb_png_captured(self, change) -> None:
        _exp_on_orb_png_captured(self, change)

    def _on_reorg_png_captured(self, change) -> None:
        _exp_on_reorg_png_captured(self, change)

    def _on_vib_png_captured(self, change) -> None:
        _exp_on_vib_png_captured(self, change)

    def _on_mol_calc_png_captured(self, change) -> None:
        _exp_on_mol_calc_png_captured(self, change)

    def _on_mol_results_png_captured(self, change) -> None:
        _exp_on_mol_results_png_captured(self, change)

    def _on_mol_analysis_png_captured(self, change) -> None:
        _exp_on_mol_analysis_png_captured(self, change)

    def _on_measure_inbox_changed(self, change) -> None:
        _measure_on_inbox_changed(self, change)

    def _on_measure_clear(self, btn=None) -> None:
        _measure_on_clear(self, btn)

    def _on_measure_help(self, btn) -> None:
        _ = btn
        self._show_help_topic("measure")

    def _on_mulliken_help(self, btn) -> None:
        _ = btn
        self._show_help_topic("mulliken")

    def _on_mulliken_overlay_changed(self, change=None) -> None:
        _ = change
        from quantui.populations_overlay import push_populations_overlay

        push_populations_overlay(self)

    def _on_iso_resolution_changed(self, change) -> None:
        """Persist the isosurface grid choice.

        Saved on change rather than at generate time so the preference sticks
        even if the user picks a grid and then closes the app without running
        anything.
        """
        new_val = (change or {}).get("new")
        if not new_val or new_val == self._user_settings.viz.iso_resolution:
            return
        self._user_settings.viz.iso_resolution = str(new_val)
        self._user_settings.save()

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

    def _on_raman_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_raman_fig", None),
            stem="raman_spectrum",
            fmt=self._raman_export_fmt_dd.value,
            status_widget=self._raman_export_status,
        )

    def _on_uv_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_uv_fig", None),
            stem="uv_vis_spectrum",
            fmt=self._uv_export_fmt_dd.value,
            status_widget=self._uv_export_status,
        )

    def _on_nmr_export_plot(self, btn) -> None:
        self._export_plot_figure(
            fig=getattr(self, "_last_nmr_fig", None),
            stem="nmr_spectrum",
            fmt=self._nmr_export_fmt_dd.value,
            status_widget=self._nmr_export_status,
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
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
                "No vibrational mode loaded — run a Frequency calculation first."
                "</span>"
            )
            return

        try:
            mode_number = int(self.vib_mode_dd.value)
        except (TypeError, ValueError):
            status.value = (
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
                "No vibrational mode selected.</span>"
            )
            return

        try:
            backend, html_str = _viz_build_vib_export_html(self, mode_number)
        except Exception as exc:
            status.value = (
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
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
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
                f"Write failed: {exc}</span>"
            )
            return

        status.value = (
            f'<span style="color:{_theme.css.ACCENT_SUCCESS};font-size:12px">'
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
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
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
                f'<span style="color:{_theme.css.ACCENT_SUCCESS};font-size:12px">'
                f"Saved: {dest}</span>"
            )
        except Exception as exc:
            msg = str(exc)
            if fmt == "png" and "kaleido" in msg.lower():
                msg = (
                    "PNG export requires kaleido. " "Install with: pip install kaleido"
                )
            status_widget.value = (
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
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
        """
        if fig is None:
            status_widget.value = (
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
                "No plot data to copy yet.</span>"
            )
            return

        csv_text = self._fig_to_csv(fig, title=title)
        if not csv_text:
            status_widget.value = (
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
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
                f'<span style="color:{_theme.css.ACCENT_ERROR};font-size:12px">'
                f"Write failed: {exc}</span>"
            )
            return

        # Best-effort clipboard copy via the browser's clipboard API.
        # Wrapped in try/catch on the JS side so a permissions error
        # doesn't show up as a Voilà console exception.
        from IPython.display import display

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
            f'<span style="color:{_theme.css.ACCENT_SUCCESS};font-size:12px">'
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

    def _on_raman_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_raman_fig", None),
            stem="raman_spectrum",
            title="Raman Spectrum",
            status_widget=self._raman_export_status,
        )

    def _on_uv_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_uv_fig", None),
            stem="uv_vis_spectrum",
            title="UV-Vis Spectrum",
            status_widget=self._uv_export_status,
        )

    def _on_nmr_copy_data(self, _btn) -> None:
        self._copy_plot_data(
            fig=getattr(self, "_last_nmr_fig", None),
            stem="nmr_spectrum",
            title="NMR Spectrum",
            status_widget=self._nmr_export_status,
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

    def _on_history_filter_changed(self, change=None) -> None:
        from quantui.app_history import apply_history_filter

        apply_history_filter(self)

    def _on_history_filter_clear(self, btn=None) -> None:
        from quantui.app_history import apply_history_filter

        # Reset every facet widget, suspending the observer so we run a single
        # filter pass at the end instead of one per widget reset.
        self._history_filter_suspend = True
        try:
            self.history_search.value = ""
            self.history_method_dd.value = ""
            self.history_basis_dd.value = ""
            self.history_date_from.value = None
            self.history_date_to.value = None
            for chip in (
                *self._history_calc_chips.values(),
                *self._history_status_chips.values(),
            ):
                chip.value = False
        finally:
            self._history_filter_suspend = False
        apply_history_filter(self)

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
        # inner ``history_load_results`` helper now. The wrapper just
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

    def _on_exit_cancel(self, _=None) -> None:
        _run_on_exit_cancel(self, _)

    # ── Help ──────────────────────────────────────────────────────────────

    def _on_help_toggle(self, _=None) -> None:
        _run_on_help_toggle(self, _)

    def _on_help_topic_changed(self, change=None) -> None:
        _run_on_help_topic_changed(self, change)

    # ══ LOGIC METHODS ════════════════════════════════════════════════════════

    def _sync_charge_to_molecule(self, change=None) -> None:
        """Push a Charge-field edit onto the active molecule (see the observer
        wiring): the run reads ``mol.charge``, so the field must not drift."""
        if self._molecule is not None:
            self._molecule.charge = int(self.charge_si.value)

    def _sync_mult_to_molecule(self, change=None) -> None:
        """Push a Multiplicity-field edit (or a spin-helper Apply) onto the
        active molecule: the run reads ``mol.multiplicity``."""
        if self._molecule is not None:
            self._molecule.multiplicity = int(self.mult_si.value)

    def _set_molecule(
        self, mol: Molecule, label: str = "", *, sync_charge_mult: bool = True
    ) -> None:
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

        _lbl = (
            f'<br><small style="color:{_theme.css.TEXT_MUTED_LIGHT}">{label}</small>'
            if label
            else ""
        )
        _summary = (
            f'<b style="font-size:15px">{mol.get_formula()}</b>'
            f'&ensp;<span style="color:{_theme.css.TEXT_SECONDARY};font-size:13px">'
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

        if sync_charge_mult:
            self.charge_si.value = mol.charge
            self.mult_si.value = mol.multiplicity
            if mol.multiplicity > 1 and self.method_dd.value == "RHF":
                self.method_dd.value = "UHF"
        else:
            # Geometry-only load: calc-setup fields own charge/mult.
            self._molecule.charge = int(self.charge_si.value)
            self._molecule.multiplicity = int(self.mult_si.value)

        # Route through ``_refresh_calc_mol_viewer`` (2026-05-25)
        # so the viewer renders via an atomic outputs swap rather than the
        # ``with self.viz_output: display(...)`` pattern already replaced
        # for the Analysis tab. The molecule attribute on
        # the app was set just above; the helper reads it.
        self._refresh_calc_mol_viewer()

        self._update_notes()

        # Any pending pre-opt preview was for the previous geometry — invalidate
        # it so a stale "Keep/Revert" can't apply to a different molecule.
        # (Accept captures the relaxed molecule before calling this, so this is
        # safe there; loading any new structure clears a leftover preview.)
        if getattr(self, "preopt_preview_box", None) is not None:
            self._preopt_relaxed_mol = None
            self.preopt_preview_box.layout.display = "none"

        # Loading a new molecule makes the previous run/preopt status stale
        # ("Pre-optimized geometry accepted.", "Cancelled.", "Done in …") —
        # clear it. Guarded on _calc_running so the mid-run pre-opt call to
        # _set_molecule_threadsafe doesn't wipe the live "Pre-optimizing…"
        # status. (Accept sets its own status right after this returns.)
        if not self._calc_running:
            self.run_status.value = ""

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

        # Re-filter the (shared) seed-geometry dropdown to only include prior
        # geo-opts of the now-active molecule (formula match). Best-effort:
        # failures must not block molecule loading.
        #
        # This used to refresh only the Frequency/UV-Vis dropdowns, because
        # Geometry Opt's seed dropdown didn't exist yet when this was written.
        # Now that all three calc types share one dropdown, a single call here
        # call here also fixes a real gap: switching molecules while already
        # on the Geometry Opt panel used to leave its seed list showing the
        # PREVIOUS molecule's matches until the user switched calc types away
        # and back.
        try:
            self._refresh_seed_options()
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
        """No-op: superseded by :class:`quantui.live_log.LiveLog` (M-LOGSCROLL).

        This used to inject a ``requestAnimationFrame`` loop that re-pinned the
        live log to the bottom every frame, to out-race ipywidgets' per-line
        ``scrollTop = 0`` reset. That made the log follow output, but at the cost
        of making it impossible to scroll up during a run — the reported bug.

        Route C removed the reset instead of racing it: the log is now a
        QuantUI-owned container that is appended to rather than re-rendered, so
        native ``overflow-anchor`` preserves the user's scroll position and no
        per-frame pinning is needed. Kept as a no-op because ``display()`` calls
        it unconditionally; delete once nothing references it.
        """
        self._run_output_scroll_guard_installed = True

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
            f'<span style="font-size:12px;color:{_theme.css.TEXT_SECONDARY};font-family:monospace">'
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

    def _show_raman_spectrum(self, freq_result) -> bool:
        return _viz_show_raman_spectrum(self, freq_result)

    def _wire_raman_controls(self) -> None:
        _viz_wire_raman_controls(self)

    def _on_raman_mode_changed(self, change) -> None:
        _viz_on_raman_mode_changed(self, change)

    def _on_raman_fwhm_changed(self, change) -> None:
        _viz_on_raman_fwhm_changed(self, change)

    def _update_raman_figure(self, mode: str, fwhm: float) -> None:
        _viz_update_raman_figure(self, mode, fwhm)

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

    def _on_uv_range_changed(self, change) -> None:
        _viz_on_uv_range_changed(self, change)

    def _show_nmr_spectrum(
        self,
        *,
        atom_symbols: list[str],
        shielding_iso_ppm: list[float],
        h_shifts: list[tuple[int, float]],
        c_shifts: list[tuple[int, float]],
        reference: str = "TMS",
    ) -> bool:
        return _viz_show_nmr_spectrum(
            self,
            atom_symbols=atom_symbols,
            shielding_iso_ppm=shielding_iso_ppm,
            h_shifts=h_shifts,
            c_shifts=c_shifts,
            reference=reference,
        )

    def _wire_nmr_controls(self) -> None:
        _viz_wire_nmr_controls(self)

    def _on_nmr_nucleus_changed(self, change) -> None:
        _viz_on_nmr_nucleus_changed(self, change)

    def _update_nmr_figure(self, nucleus: str) -> None:
        _viz_update_nmr_figure(self, nucleus)

    def _show_orbital_diagram(self, result) -> bool:
        return _viz_show_orbital_diagram(self, result)

    def _on_iso_generate(self, btn) -> None:
        _viz_on_iso_generate(self, btn)

    def _on_orb_toggle_changed(self, change) -> None:
        """Show/hide the free-entry MO-index input for the 'By index' mode."""
        self._orb_index_input.layout.display = (
            "" if change["new"] == "By index" else "none"
        )

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
        # Run-in-flight state: arm Cancel, lock out Clear (so it can't wipe the
        # live output mid-run), reset the cancel flag for this fresh run.
        self._calc_running = True
        self._cancel_event.clear()
        self.cancel_btn.disabled = False
        self.log_clear_btn.disabled = True

        self.step_progress.complete(1)
        self.step_progress.start(2)

        _calc_log.log_event(
            "calc_start",
            f"{self.method_dd.value}/{self.basis_dd.value} on {mol.get_formula()}",
            n_atoms=len(mol.atoms),
        )
        _run_wall_t = time.perf_counter()
        _run_cpu_t = time.process_time()
        # Start the live elapsed-time ticker.
        self._start_elapsed_ticker(_run_wall_t)
        _scf_converged_t: Optional[float] = None
        _tail_marks: dict[str, float] = {}

        # Capture the estimator's pre-run (2026-05-25)
        # prediction so we can write a (predicted, actual) record to
        # ``prediction_log.jsonl`` after the calc completes. The
        # estimator may return None (insufficient history); we record
        # that as "no estimate" so the dashboard counts it separately
        # from "estimate was wrong by N%".
        _predicted_run_s: Optional[float] = None
        _predicted_run_confidence: str = "unknown"
        try:
            _ct_for_est = _run_calc_type_key(self)
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

            # Match session_calc's density-fitting decision (M-DF) so the
            # estimate is drawn from the pool the run will land in. Post-HF
            # references are never fitted, so predict False for them.
            _predicted_density_fit: Optional[bool] = None
            try:
                if self.method_dd.value.upper() in ("MP2", "CCSD", "CCSD(T)"):
                    _predicted_density_fit = False
                else:
                    _predicted_density_fit = bool(
                        self._user_settings.compute.density_fit
                    )
            except Exception:  # noqa: BLE001 — fall back to fit-agnostic
                _predicted_density_fit = None

            _est = _calc_log.estimate_time(
                n_atoms=len(mol.atoms),
                n_electrons=mol.get_electron_count(),
                method=self.method_dd.value,
                basis=self.basis_dd.value,
                n_basis=_nb_for_est,
                calc_type=_ct_for_est,
                gpu_used=_predicted_gpu_used,
                source="app",
                density_fit=_predicted_density_fit,
            )
            if _est is not None:
                _predicted_run_s = float(_est["seconds"])
                _predicted_run_confidence = str(_est.get("confidence", "unknown"))
                # Hand the estimate to the live ticker so it can show a
                # "time remaining" readout alongside elapsed.
                self._run_estimate_s = _predicted_run_s
                self._run_estimate_conf = _predicted_run_confidence
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
            cancel_check=self._cancel_event.is_set,
        )
        # Expose this run's log to the elapsed ticker so it can read the
        # completion fraction calc modules report via emit_progress.
        self._active_log = log
        # Watch for silent stretches in the output log. Stopped in the `finally`
        # alongside the elapsed ticker, so it cannot outlive the run and keep
        # writing into a finished log.
        log.start_heartbeat()

        # --- Checkpoint for this run (M-CHECKPOINT) ---
        # Opened before any calculation starts, because the runs worth
        # checkpointing are exactly the ones that never reach the end. Failure
        # to open one leaves ``_ckpt`` as None and the calc runs
        # uncheckpointed, which is the pre-M-CHECKPOINT behaviour.
        #
        # Deliberately after ``log`` exists: the checkpoint writes its own
        # provenance lines into the run log, and those lines are the only
        # record that a resumed run did not start from the geometry at the
        # top of the file.
        _ckpt = self._begin_run_checkpoint(log)
        # Resume only when there is genuinely something to continue. The
        # checkbox defaults to checked and is *hidden* when nothing is
        # resumable, so consulting it alone would ask every ordinary run to
        # resume — and the optimizer would answer with a "no usable
        # checkpoint" warning on a calculation the user started from scratch.
        _resume = bool(
            _ckpt is not None
            and self._resume_cb.value
            and getattr(self, "_checkpoint_resumable", False)
        )
        if _ckpt is not None and not _resume:
            _ckpt.clear_run_log()
        elif _resume and _ckpt is not None:
            _prior_log = _ckpt.read_run_log()
            if _prior_log:
                log.seed_prior(_prior_log)

        _run_saved = False
        _run_was_cancelled = False

        # The run header (structured banner) is written synchronously + atomically
        # on the main thread by ``on_run_clicked`` → ``_write_run_header`` BEFORE
        # this background thread starts. Writing it here (bg thread) instead was
        # the pre-step-1 "blank window" bug (2026-07-18): for a large molecule the
        # long gap before the first optimizer/SCF line exposed a lost-early-output
        # race in the bg-thread ``append_stdout`` path. All later PySCF / optimizer
        # output appends onto that header via ``log`` as usual.

        try:
            # Classical pre-optimization is now an explicit Preview → Keep tool
            # (it mutates the active molecule before the run when the user keeps
            # it), not a silent step here. So the run uses the active geometry
            # as-is. (The QM "Geometry optimization before calculation" path is
            # separate and handled below per calc type.)
            calc_mol = mol

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

                # Rename user-facing
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
                # Catch numerical failures (e.g. singular matrix in
                # cho_solve on tight rings) and fall back to the user's
                # input geometry rather than killing the whole calc.
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
                # Optional seed: start from a previously optimised geometry
                # rather than the current molecule — the "optimise cheaply,
                # then refine at a higher level of theory" workflow. Mirrors
                # the Frequency / UV-Vis seed handling below.
                _geo_seed_path = self._geo_seed_dd.value
                if _geo_seed_path:
                    from quantui.results_storage import load_trajectory

                    self.run_status.value = "Loading seed geometry from history…"
                    _geo_seed_traj, _ = load_trajectory(Path(_geo_seed_path))
                    calc_mol = _geo_seed_traj[-1]
                    log.write(
                        f"\nSeed geometry loaded from: "
                        f"{Path(_geo_seed_path).name}\n"
                        f"  Formula: {calc_mol.get_formula()}  "
                        f"Atoms: {len(calc_mol.atoms)}\n"
                        "  Optimization starts from this geometry.\n\n"
                    )

                self.run_status.value = "Optimizing geometry..."
                from quantui import optimize_geometry

                # History-based expected step count → "step k/~N" + a floor
                # for the live progress fraction. None on cold history.
                _expected_steps = _calc_log.estimate_opt_steps(
                    self.method_dd.value, self.basis_dd.value
                )
                result = optimize_geometry(
                    molecule=calc_mol,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    fmax=self.fmax_fi.value,
                    steps=self.max_steps_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                    expected_steps=(
                        int(round(_expected_steps)) if _expected_steps else None
                    ),
                    checkpoint=_ckpt,
                    resume=_resume,
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

                # ── Step 1: resolve seed geometry (geo-opt or mode perturbation)
                calc_mol = _run_resolve_seed_geometry(self, calc_mol, log=log)

                # ── Step 2: optional geometry optimization ────────────────────
                #
                # Renamed from
                # "pre-optimisation" — the wrapped operation is a full
                # DFT geometry optimization at the user's selected
                # method/basis. The LJ-classical pre-opt is in
                # quantui/preopt.py and keeps its "pre-opt" name.
                #
                # Geom-opt can hit a singular matrix
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
                        "raman_activities": result.raman_activities,
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
                # Renamed from
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
                    # Catch numerical failures and fall back to the
                    # user's seed geometry rather than killing the whole
                    # TD-DFT calc.
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
                _nmr_seed_path = self._freq_seed_dd.value
                if _nmr_seed_path:
                    from quantui.results_storage import load_trajectory

                    self.run_status.value = "Loading seed geometry from history…"
                    _nmr_seed_traj, _ = load_trajectory(Path(_nmr_seed_path))
                    calc_mol = _nmr_seed_traj[-1]
                    log.write(
                        f"\nSeed geometry loaded from: {Path(_nmr_seed_path).name}\n"
                        f"  Formula: {calc_mol.get_formula()}  "
                        f"Atoms: {len(calc_mol.atoms)}\n\n"
                    )

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
                    checkpoint=_ckpt,
                    resume=_resume,
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
            elif ct == "Reorganization Energy":
                self.run_status.value = "Computing reorganization energy..."
                from quantui.reorganization_energy import (
                    run_reorganization_energy,
                )

                _solvent = self.solvent_dd.value if self.solvent_cb.value else None
                result = run_reorganization_energy(
                    molecule=calc_mol,
                    mode=self._reorg_mode_dd.value,
                    method=self.method_dd.value,
                    basis=self.basis_dd.value,
                    fmax=self.fmax_fi.value,
                    steps=self.max_steps_si.value,
                    progress_stream=log,  # type: ignore[arg-type]
                    solvent=_solvent,
                    checkpoint=_ckpt,
                    resume=_resume,
                )
                result_html = self._format_reorg_result(result)
                save_spectra = result.to_spectra()
                save_type = "reorganization_energy"
            else:  # Single Point
                self.run_status.value = "Calculating..."
                from quantui import run_in_session

                # MP2 heavy-atom warning
                if self.method_dd.value.upper() == "MP2":
                    _n_heavy = sum(1 for a in calc_mol.atoms if a != "H")
                    if _n_heavy > 20:
                        self.result_output.append_display_data(
                            HTML(
                                f'<div style="background:#fffbe6;border-left:4px solid {_theme.css.ACCENT_WARNING_LIGHT};'
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
                    checkpoint=_ckpt,
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
            _viz_mol = (
                result.molecule
                if ct in ("Geometry Opt", "Reorganization Energy")
                else calc_mol
            )
            if ct == "Geometry Opt":
                self._viz_label.value = (
                    f'<p style="color:{_theme.css.TEXT_SECONDARY};font-size:12px;font-weight:600;'
                    'margin:6px 0 2px">Optimized geometry</p>'
                )
                self._viz_label.layout.display = ""
            elif ct == "Reorganization Energy":
                self._viz_label.value = (
                    f'<p style="color:{_theme.css.TEXT_SECONDARY};font-size:12px;font-weight:600;'
                    'margin:6px 0 2px">Optimized neutral geometry</p>'
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
                f'<span style="color:{_theme.css.TEXT_STRONG};font-size:13px;font-weight:500">'
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
                _run_saved = True
                self._last_result_dir = _saved_dir
                # Result folder is now on disk —
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
                        # Also write
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
                # Persist pre-opt geometry trajectory for Frequency runs.
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
                # Write a Molden-format companion
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
                _was_warm = save_type in self._warm_calc_types
                self._warm_calc_types.add(save_type)
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
                    n_steps=getattr(result, "n_steps", None),
                    source="app",
                    warm=_was_warm,
                    stages=log.stage_timings(),
                    density_fit=getattr(result, "density_fit", None),
                )
                _calc_log.log_event(
                    "calc_done",
                    f"{result.method}/{result.basis} on {result.formula}",
                    elapsed_s=round(_elapsed_for_est, 2),
                    converged=result.converged,
                    gpu_used=bool(getattr(result, "gpu_used", False)),
                    gpu_name=getattr(result, "gpu_name", None),
                )
                # Persist the (predicted, actual) pair to
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
                f'<b style="color:{_theme.css.ACCENT_ERROR}">&#9888; Dependency Not Available</b><br>'
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

        except _CalcCancelled:
            _elapsed = time.perf_counter() - _run_wall_t
            _run_was_cancelled = True
            # Clear the cancel flag FIRST: the next line writes through
            # ``log`` (``_LogCapture.write``), which re-raises _CalcCancelled
            # while the flag is still set — that would propagate out of this
            # handler and skip the cancelled card + "Cancelled." status below
            # (only ``finally`` would run, leaving the status stuck on
            # "Cancelling…"). ``finally`` clears it again (idempotent).
            self._cancel_event.clear()
            log.write("\n── Calculation cancelled by user ──\n")
            self.result_output.append_display_data(
                HTML(
                    '<div style="background:#fffbeb;border:1px solid #fcd34d;'
                    'border-radius:8px;padding:14px;margin:8px 0">'
                    '<b style="color:#92400e">&#9632; Calculation cancelled</b><br>'
                    '<small style="color:#92400e">Stopped at your request — no '
                    "results were saved. Adjust the settings and run again.</small>"
                    "</div>"
                )
            )
            self.run_status.value = "Cancelled."
            try:
                self.step_progress.reset()
            except Exception:
                pass
            try:
                _calc_log.log_event(
                    "calc_cancelled",
                    f"{mol.get_formula()} {self.method_dd.value}/{self.basis_dd.value}",
                    elapsed_s=round(_elapsed, 2),
                )
            except OSError:
                pass

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
                f'<b style="color:{_theme.css.ACCENT_ERROR}">&#9888; Calculation Failed</b><br>'
                f'<code style="color:#7f1d1d">{exc}</code><br><br>'
                '<small style="color:#991b1b">'
                "Tips: try a smaller basis set (STO-3G), use a geometry-optimized "
                "structure first, or check for unusually long/short bonds in your "
                "XYZ input. Full error details are in the <b>Output</b> tab.</small>"
                f"{self._resume_hint_html(_ckpt)}"
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
            # Disarm Cancel, re-enable Clear, clear run-in-flight state.
            self._calc_running = False
            self._cancel_event.clear()
            self.cancel_btn.disabled = True
            self.cancel_btn.description = "Cancel"
            self.log_clear_btn.disabled = False
            self._stop_elapsed_ticker()
            try:
                log.stop_heartbeat()
            except Exception:  # noqa: BLE001 — teardown must not mask a failure
                pass
            _run_log_text = ""
            try:
                _run_log_text = log.getvalue()
            except Exception:  # noqa: BLE001 — teardown must not mask a failure
                pass
            self._finish_run_checkpoint(
                _ckpt,
                run_saved=_run_saved,
                run_cancelled=_run_was_cancelled,
                run_log=_run_log_text,
            )
            self._activity_end(kind="compute")

    # ── Live elapsed ticker ────────────────────────────────────────────────

    def _start_elapsed_ticker(self, start_t: float) -> None:
        """Spin up a daemon thread that updates the elapsed chip every ~1 s.

        Writes only to ``_run_elapsed_lbl`` (never ``run_status``), so it never
        fights the stage labels set by ``_LogCapture`` / the calc modules.
        ``Label``/``HTML`` ``.value`` writes are thread-safe, so no io_loop
        marshaling is needed.
        """
        self._stop_elapsed_ticker()  # ensure no prior ticker is still running
        # Reset the runtime estimate; _do_run fills it in once computed.
        self._run_estimate_s = None
        self._run_estimate_conf = "unknown"
        # Drop any prior run's log so a stale fraction can't leak into the
        # new run's chip before this run's _LogCapture is created.
        self._active_log = None
        stop_event = threading.Event()
        self._elapsed_stop_event = stop_event

        def _tick() -> None:
            while not stop_event.wait(1.0):
                try:
                    self._run_elapsed_lbl.value = self._format_elapsed_chip(
                        time.perf_counter() - start_t
                    )
                except Exception:
                    break

        threading.Thread(
            target=_tick, daemon=True, name="quantui-elapsed-ticker"
        ).start()

    def _format_elapsed_chip(self, elapsed: float) -> str:
        """Compose the live chip: ``⏱ <elapsed>`` + ``· ~<remaining> left``.

        Folds the pre-run total estimate (``_run_estimate_s``,
        set by ``_do_run``) into a remaining-time readout. Degrades to
        elapsed-only when there's no estimate (cold history) and switches to
        "longer than estimated" once elapsed passes the estimate — never shows a
        negative or false-precise number. Low-confidence estimates are marked
        "(rough)" so the readout stays honest.
        """
        from quantui.log_utils import format_elapsed

        base = f"⏱ {format_elapsed(elapsed)}"

        # Prefer a self-correcting fraction-based estimate when a calc module
        # reports real progress (PES points, optimizer fmax trend). Only trust it
        # once past a small floor so early-run noise doesn't spike the estimate.
        log = getattr(self, "_active_log", None)
        frac = getattr(log, "_fraction", None) if log is not None else None
        if frac is not None and frac >= 0.03 and elapsed > 0:
            remaining = elapsed * (1.0 - frac) / frac
            return (
                f'<span style="color:{_theme.css.TEXT_SLATE};font-size:13px">'
                f"{base} · ~{format_elapsed(remaining)} left</span>"
            )

        # Fallback: static total estimate minus elapsed.
        est = getattr(self, "_run_estimate_s", None)
        if est and est > 0:
            remaining = est - elapsed
            if remaining > 0:
                rough = (
                    " (rough)"
                    if getattr(self, "_run_estimate_conf", "") == "low"
                    else ""
                )
                base = f"{base} · ~{format_elapsed(remaining)} left{rough}"
            else:
                base = f"{base} · longer than estimated"
        return (
            f'<span style="color:{_theme.css.TEXT_SLATE};font-size:13px">{base}</span>'
        )

    def _stop_elapsed_ticker(self) -> None:
        """Stop the elapsed ticker and clear the chip."""
        ev = getattr(self, "_elapsed_stop_event", None)
        if ev is not None:
            ev.set()
            self._elapsed_stop_event = None
        try:
            self._run_elapsed_lbl.value = ""
        except Exception:
            pass

    def _update_notes(self, change=None) -> None:
        _run_update_notes(self, change)

    def _begin_run_checkpoint(self, log_stream: Optional[Any] = None) -> Optional[Any]:
        """Open a checkpoint for the run about to start, or return ``None``.

        Returning ``None`` — no molecule, an unavailable checkpoint module, an
        unwritable directory — means the calculation runs without one. A
        checkpoint is an optimisation for the failure case; it must never be
        the reason a calculation doesn't start.
        """
        try:
            from quantui.app_runflow import checkpoint_identity
            from quantui.checkpoint import Checkpoint

            identity = checkpoint_identity(self)
            if identity is None:
                self._checkpoint_resumable = False
                return None
            ckpt = Checkpoint(identity, log_stream=log_stream)
            # Read resumability BEFORE begin() — begin() rewrites the metadata
            # with a fresh "running" status, so asking afterwards would
            # describe the run about to start rather than the one that stopped.
            self._checkpoint_resumable = ckpt.resumable_state() is not None
            extra: dict = {}
            if self.calc_type_dd.value == "PES Scan":
                # Lets the resume offer say "8 of 20" rather than just "8".
                extra["total_points"] = int(self._scan_steps.value)
                # Scan geometry isn't part of the checkpoint identity, so
                # without these a restore would reinstate the molecule and
                # method but silently leave a different scan range — and the
                # stored points, matched by coordinate value, would all miss.
                extra["settings"] = {
                    "scan_type": self._scan_type_dd.value,
                    "scan_atom1": int(self._scan_atom1.value),
                    "scan_atom2": int(self._scan_atom2.value),
                    "scan_atom3": int(self._scan_atom3.value),
                    "scan_atom4": int(self._scan_atom4.value),
                    "scan_start": float(self._scan_start.value),
                    "scan_stop": float(self._scan_stop.value),
                    "scan_steps": int(self._scan_steps.value),
                }
            if not ckpt.begin(**extra):
                return None
            return ckpt
        except Exception as exc:  # noqa: BLE001 — never block a run
            self._checkpoint_resumable = False
            try:
                _calc_log.log_event(
                    "checkpoint_unavailable", f"{type(exc).__name__}: {exc}"[:200]
                )
            except Exception:  # noqa: BLE001 — telemetry self-guard
                pass
            return None

    def _refresh_resume_list(self) -> None:
        """Rebuild the History tab's unfinished-calculations list."""
        try:
            from quantui.app_runflow import refresh_resume_list

            refresh_resume_list(self)
        except Exception:  # noqa: BLE001 — never break the History tab
            pass

    def _on_resume_entry_changed(self, _change=None) -> None:
        from quantui.app_runflow import describe_resume_entry

        describe_resume_entry(self, _change)

    def _on_resume_restore(self, _btn=None) -> None:
        """Load the selected checkpoint's settings and go to Calculate."""
        from quantui.app_runflow import restore_resume_entry

        if not restore_resume_entry(self):
            return
        # Send the user where the settings just landed. Restoring without
        # moving them leaves the effect invisible on a tab they aren't
        # looking at, which reads as the button having done nothing.
        try:
            self.root_tab.selected_index = self._tab_index("calculate")
        except Exception:  # noqa: BLE001 — tab index is cosmetic
            pass

    def _on_resume_discard(self, _btn=None) -> None:
        """Delete the selected checkpoint and refresh the list."""
        try:
            import shutil

            selected = self._resume_list_dd.value
            if selected and selected in (getattr(self, "_resume_entries", None) or {}):
                shutil.rmtree(selected, ignore_errors=True)
        except Exception:  # noqa: BLE001 — discarding is best-effort
            pass
        self._refresh_resume_list()
        try:
            from quantui.app_runflow import refresh_resume_notice

            refresh_resume_notice(self)
        except Exception:  # noqa: BLE001 — a stale notice is not fatal
            pass

    def _resume_hint_html(self, ckpt: Optional[Any]) -> str:
        """Return a "you can resume this" line for the failure card, or ``""``.

        The resume offer itself lives up by the Run button, which is not where
        anyone is looking after a calculation fails. Saying it here, next to
        the error, is the difference between the feature being discovered and
        it quietly never being used.
        """
        try:
            if ckpt is None or ckpt.resumable_state() is None:
                return ""
            points = len(ckpt.completed_points())
            done = f"{points} completed scan point{'s' if points != 1 else ''}"
            if not points:
                done = "the steps completed so far"
            return (
                '<br><br><small style="color:#991b1b">'
                f"&#9851; <b>This run can be resumed.</b> {done} "
                "were saved. Leave the settings as they are and press "
                "<b>Run</b> again — the <b>Resume from checkpoint</b> box "
                "above the Run button is already ticked.</small>"
            )
        except Exception:  # noqa: BLE001 — a hint must never mask the error
            return ""

    def _finish_run_checkpoint(
        self,
        ckpt: Optional[Any],
        *,
        run_saved: bool = False,
        run_cancelled: bool = False,
        run_log: str = "",
    ) -> None:
        """Close out *ckpt* after a run ends, however it ended.

        Runs from the ``finally`` of ``_do_run``, so it is reached on success,
        on failure and on cancel. It deliberately does **not** decide whether
        the run succeeded — the calc modules mark completion themselves, since
        only they know whether "finished" means converged, and a run that
        stopped early must keep its resumable state.

        What happens here is bookkeeping: persist or clear the run log for
        resume continuity (CHK.8b / ISSUE.9), refresh the resume offer, and
        prune old checkpoints so the directory doesn't grow without bound.
        """
        try:
            if ckpt is not None:
                if run_saved:
                    ckpt.clear_run_log()
                elif run_log.strip():
                    ckpt.write_run_log(run_log)
        except Exception:  # noqa: BLE001 — teardown must not mask a failure
            pass
        try:
            if ckpt is not None and run_cancelled:
                ckpt.mark_interrupted()
        except Exception:  # noqa: BLE001 — teardown must not mask a failure
            pass
        try:
            from quantui.checkpoint import prune

            prune()
        except Exception:  # noqa: BLE001 — retention is best-effort
            pass
        try:
            from quantui.app_runflow import refresh_resume_notice

            refresh_resume_notice(self)
        except Exception:  # noqa: BLE001 — a stale notice is not fatal
            pass
        # A run that just failed becomes a new listing entry; one that
        # succeeded removes itself. Either way the list is now stale.
        self._refresh_resume_list()

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
        # The standalone Log tab is
        # gone; the PySCF output log now lives in an Accordion inside
        # the History tab (index 3). Switch tabs + expand the log
        # accordion so the user lands directly on the log content.
        self.root_tab.selected_index = self._tab_index("history")
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
                style = f"color:{_theme.css.TEXT_BODY};font-weight:700"
            elif line.startswith("  ✓"):
                style = f"color:{_theme.css.ACCENT_SUCCESS};font-weight:700"
            elif line.startswith("  ✗"):
                style = "color:#dc2626;font-weight:700"
            elif (
                line.startswith("  Machine:")
                or line.startswith("  GPU:")
                or line.startswith("  Threads:")
            ):
                style = f"color:{_theme.css.TEXT_SLATE_DARK}"
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
                style = f"color:{_theme.css.TEXT_SLATE}"
            elif line.startswith("    ✔") or line.startswith("    ⚠"):
                style = "color:#d97706"
            # ── Geometry optimisation (ASE BFGS) ──────────────────────────────
            elif line.startswith("BFGS:"):
                m = _bfgs_re.match(line)
                if m:
                    fmax = float(m.group(3))
                    # Colour by convergence: green when nearly converged, teal otherwise
                    style = (
                        f"color:{_theme.css.ACCENT_SUCCESS};font-weight:600"
                        if fmax < 0.1
                        else f"color:{_theme.css.ACCENT_TEAL}"
                    )
                else:
                    style = f"color:{_theme.css.ACCENT_TEAL}"
            elif line.strip() == "Step Time Energy fmax":
                style = f"color:{_theme.css.TEXT_BODY};font-weight:700"
            # ── Post-optimisation summary ──────────────────────────────────────
            elif line.startswith("── Final SCF"):
                style = "color:#6d28d9;font-weight:600"
            elif "HOMO-LUMO gap:" in line:
                style = "color:#6d28d9;font-weight:600"
            # ── SCF convergence ────────────────────────────────────────────────
            elif "converged SCF energy" in line or "SCF converged" in line:
                style = f"color:{_theme.css.ACCENT_SUCCESS};font-weight:600"
            elif line.lstrip().startswith("cycle=") and "E=" in line:
                style = f"color:{_theme.css.TEXT_SLATE}"
            # ── MO / orbital info (verbose=4) ──────────────────────────────────
            elif "MO energies" in line or "** MO" in line:
                style = "color:#1d4ed8;font-weight:600"
            elif "HOMO" in line or "LUMO" in line or "All MO energies" in line:
                style = f"color:{_theme.css.ACCENT_INFO}"
            elif line.lstrip().startswith("occupied:") or line.lstrip().startswith(
                "virtual:"
            ):
                style = "color:#3b82f6"
            # ── Thermo / properties ────────────────────────────────────────────
            elif "Mulliken" in line or "mulliken" in line:
                style = f"color:{_theme.css.ACCENT_PURPLE}"
            elif "dipole" in line.lower() or "Dipole" in line:
                style = f"color:{_theme.css.ACCENT_PURPLE}"
            elif "nuclear repulsion" in line.lower() or "Nuclear repulsion" in line:
                style = f"color:{_theme.css.TEXT_SUBTLE}"
            elif "E(MP2)" in line or "MP2 correlation" in line:
                style = "color:#0891b2"
            # ── Warnings / errors ──────────────────────────────────────────────
            elif "Warning" in line or "warning" in line:
                style = "color:#d97706"
            elif "Error" in line or "error" in line or "failed" in line:
                style = "color:#dc2626"
            else:
                style = f"color:{_theme.css.TEXT_STRONG}"
            rows.append(f'<div style="{style}">{esc}</div>')
        self._log_output_html.value = (
            '<div style="font-family:monospace;font-size:12px;line-height:1.4;'
            f"padding:8px 10px;background:{_theme.css.BG_PANEL};border:1px solid {_theme.css.BORDER};"
            'border-radius:4px;overflow-x:auto;max-height:550px;overflow-y:auto">'
            + "".join(rows)
            + "</div>"
        )
        self._log_source_lbl.value = (
            f'<span style="font-size:12px;color:{_theme.css.TEXT_SLATE}">Source: {source_label}</span>'
            if source_label
            else ""
        )

    def _render_help_topic(self, change=None) -> None:
        key = self.help_topic_dd.value
        if key and key in HELP_TOPICS:
            entry = HELP_TOPICS[key]
            self.help_content_html.value = (
                f'<div style="border:1px solid {_theme.css.BORDER};border-radius:6px;'
                f'padding:14px 18px;margin:8px 0;background:{_theme.css.BG_PANEL};max-width:700px">'
                f'<h4 style="margin:0 0 10px;color:{_theme.css.TEXT_STRONG};font-size:15px;font-weight:700">'
                f'{entry["title"]}</h4>'
                f'<div style="font-size:14px;color:{_theme.css.TEXT_BODY};line-height:1.6">'
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
                f'<span style="color:{_theme.css.TEXT_SUBTLE};font-size:13px">'
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
            f'<th style="text-align:left;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Method</th>'
            f'<th style="text-align:left;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Basis</th>'
            f'<th style="text-align:right;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Runs</th>'
            f'<th style="text-align:right;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Avg</th>'
            f'<th style="text-align:right;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Min</th>'
            f'<th style="text-align:right;padding:2px 12px 2px 0;color:{_theme.css.TEXT_SLATE}">Max</th>'
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
                f'<span style="color:{_theme.css.TEXT_SUBTLE};font-size:13px">'
                "No events recorded yet.</span>"
            )
        rows = ""
        for e in reversed(events):
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            evt = e.get("event", "")
            msg = e.get("message", "")
            rows += (
                "<tr>"
                f'<td style="padding:1px 10px 1px 0;color:{_theme.css.TEXT_SUBTLE};font-size:11px;white-space:nowrap">{ts}</td>'
                f'<td style="padding:1px 10px 1px 0;color:{_theme.css.TEXT_SLATE_DARK};font-size:12px">{evt}</td>'
                f'<td style="padding:1px 0;color:{_theme.css.TEXT_BODY};font-size:12px">{msg}</td>'
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

    def _format_reorg_result(self, r) -> str:
        return _fmt_reorg_result(r)

    def _show_pes_scan_result(self, result) -> bool:
        return _viz_show_pes_scan_result(self, result)

    def _format_past_result(self, data: dict, result_dir: Optional[Path] = None) -> str:
        return _fmt_past_result(data, result_dir=result_dir)

    # ══ HELPERS ══════════════════════════════════════════════════════════════

    def _get_results_dir(self) -> Path:
        from quantui.results_storage import _default_results_dir

        return _default_results_dir().resolve()
