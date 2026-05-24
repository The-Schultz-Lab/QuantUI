"""Runflow helpers used by QuantUIApp."""

from __future__ import annotations

import threading
import time
from typing import Any

import ipywidgets as widgets
from IPython.display import HTML, Javascript, display


def _calc_type_badge(calc_type: str) -> str:
    return {
        "single_point": "SP",
        "geometry_opt": "GeoOpt",
        "frequency": "Freq",
        "tddft": "UV-Vis",
        "nmr": "NMR",
        "pes_scan": "PES",
    }.get(calc_type, calc_type or "Unknown")


def on_run_clicked(app: Any, btn: Any) -> None:
    """Reset result panes and start the background run thread."""
    app.run_output.clear_output()
    app.result_output.clear_output()
    app.result_viz_output.clear_output()
    app._analysis_mol_output.clear_output()
    app._viz_label.layout.display = "none"
    app._viz_label.value = ""
    app._deactivate_all_ana_panels()
    app._clear_output_widget(app._pes_plot_html)
    app._result_dir_label.value = ""
    app._result_dir_label.layout.display = "none"
    app._result_log_accordion.layout.display = "none"
    app._result_log_accordion.selected_index = None
    app._result_log_output.clear_output()
    app._completion_banner.layout.display = "none"
    app._to_analysis_btn.layout.display = "none"
    app._analysis_empty_html.layout.display = "none"
    threading.Thread(target=app._do_run, daemon=True).start()


def on_calc_type_changed(app: Any, change: Any, *, layout_fn: Any) -> None:
    """Update extra options panel based on selected calculation type."""
    ct = change["new"]

    # QM pre-optimization is meaningful for all workflows except Geometry Opt,
    # which is itself an optimization workflow.
    if ct == "Geometry Opt":
        app._freq_preopt_cb.value = False
        app._freq_preopt_cb.layout.display = "none"
    else:
        app._freq_preopt_cb.layout.display = ""

    if ct == "Geometry Opt":
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app.fmax_fi, app.max_steps_si],
                layout=layout_fn(gap="8px"),
            ),
        ]
    elif ct == "Frequency":
        app._refresh_freq_seed_options()
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app._freq_seed_dd, app._freq_seed_refresh_btn],
                layout=layout_fn(align_items="center", gap="6px", width="100%"),
            ),
            app._freq_seed_note,
        ]
    elif ct == "UV-Vis (TD-DFT)":
        app._refresh_tddft_seed_options()
        app.calc_extra_opts.children = [
            app.nstates_si,
            widgets.HBox(
                [app._tddft_seed_dd, app._tddft_seed_refresh_btn],
                layout=layout_fn(align_items="center", gap="6px", width="100%"),
            ),
            app._tddft_seed_note,
            widgets.HTML(
                '<span style="color:#b45309;font-size:12px">⚠ Requires a DFT '
                "functional (e.g. B3LYP, PBE0). RHF/UHF will run TDHF (CIS) "
                "instead.</span>"
            ),
        ]
    elif ct == "NMR Shielding":
        app.calc_extra_opts.children = [
            widgets.HTML(
                '<span style="color:#b45309;font-size:12px">'
                "⚠ Recommended: B3LYP/6-31G* or better. "
                "STO-3G and 3-21G give qualitative results only. "
                "Start from an optimised geometry for best accuracy.</span>"
            ),
        ]
    elif ct == "PES Scan":
        app._update_scan_widgets()
        app.calc_extra_opts.children = [
            widgets.HBox(
                [app._scan_type_dd],
                layout=layout_fn(margin="0 0 4px 0"),
            ),
            widgets.HBox(
                [app._scan_atom1, app._scan_atom2],
                layout=layout_fn(gap="4px"),
            ),
            app._scan_atom34_box,
            widgets.HBox(
                [
                    app._scan_start,
                    app._scan_stop,
                    app._scan_steps,
                    app._scan_unit_lbl,
                ],
                layout=layout_fn(gap="4px", align_items="center"),
            ),
        ]
    else:
        app.calc_extra_opts.children = []


def update_scan_widgets(app: Any, _change: Any = None) -> None:
    """Show/hide atom inputs and unit label based on scan type."""
    st = app._scan_type_dd.value
    if st == "Bond":
        app._scan_atom34_box.layout.display = "none"
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">Å</span>'
    elif st == "Angle":
        app._scan_atom4.layout.display = "none"
        app._scan_atom3.layout.display = ""
        app._scan_atom34_box.layout.display = ""
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">°</span>'
    else:  # Dihedral
        app._scan_atom3.layout.display = ""
        app._scan_atom4.layout.display = ""
        app._scan_atom34_box.layout.display = ""
        app._scan_unit_lbl.value = '<span style="font-size:12px;color:#555">°</span>'


# Default RMSD tolerance for the seed-geometry "same molecule" check (HIST.6).
# 0.1 Å is generous enough to admit slight conformational differences (e.g.
# re-importing the same SMILES, which can produce ~0.05 Å float-precision
# drift in RDKit's embedding) but tight enough to reject distinct isomers,
# whose heavy-atom positions typically differ by ≥1 Å.
_SEED_GEOMETRY_RMSD_TOLERANCE: float = 0.1


# Per-result cache of (atoms, starting_coords) parsed from trajectory.json.
# Saved geo-opt results are immutable once written, so a session-lifetime
# cache is safe. Keyed by the resolved absolute path of the result dir.
# ``None`` is cached as a sentinel for "trajectory.json missing or malformed"
# to avoid retrying parse on every dropdown refresh.
_SEED_GEOMETRY_CACHE: dict = {}


def _load_starting_geometry(result_dir: Any):
    """Read the starting-frame atom list + coordinates from a geo-opt result.

    Returns ``(atoms, coords_ndarray)`` where ``coords_ndarray`` has shape
    ``(N, 3)``, or ``None`` if ``trajectory.json`` is missing / malformed.
    Per-session cache avoids re-parsing on every dropdown refresh.
    """
    try:
        key = str(result_dir.resolve())
    except OSError:
        key = str(result_dir)
    if key in _SEED_GEOMETRY_CACHE:
        return _SEED_GEOMETRY_CACHE[key]

    import json as _json

    import numpy as _np

    traj_path = result_dir / "trajectory.json"
    if not traj_path.exists():
        _SEED_GEOMETRY_CACHE[key] = None
        return None
    try:
        data = _json.loads(traj_path.read_text())
        atoms = data.get("atoms")
        steps = data.get("steps", [])
        if not atoms or not steps:
            _SEED_GEOMETRY_CACHE[key] = None
            return None
        coords = _np.array(steps[0]["coords"], dtype=float)
        if coords.shape != (len(atoms), 3):
            _SEED_GEOMETRY_CACHE[key] = None
            return None
        result = (list(atoms), coords)
        _SEED_GEOMETRY_CACHE[key] = result
        return result
    except Exception:
        _SEED_GEOMETRY_CACHE[key] = None
        return None


def _geometries_match(
    atoms_a,
    coords_a,
    atoms_b,
    coords_b,
    *,
    rmsd_tol: float = _SEED_GEOMETRY_RMSD_TOLERANCE,
) -> bool:
    """Strict atom-order + RMSD-based geometry comparison (HIST.6).

    Returns ``True`` iff the atom symbol lists are equal in order AND the
    structures' RMSD (no rigid alignment) is at or below ``rmsd_tol`` Å.

    Design decisions for v1:
    - **Strict atom order** rather than permutation-aware. The latter requires
      O(N!) or a proper graph isomorphism solver and is rarely needed in
      practice — users almost always re-import a molecule in the same atom
      order. If atom order matters in a real-world scenario, the right fix
      is upstream (canonicalize on save) rather than per-compare permutation.
    - **No rigid alignment.** The seed-geometry semantics is "load this exact
      saved geometry to start from". A rotated copy will not match — but the
      saved result and current molecule must come from the same input order
      and similar source (e.g. the same SMILES), so rotation drift is rare.
      Alignment can be added later under the same helper if it becomes a real
      pain point.
    - **RMSD across all atoms** rather than per-atom L₂. Heavy displacements
      in one atom shouldn't swamp matches in the rest; conversely a tiny
      jiggle across all atoms is a clear "same molecule".
    """
    if list(atoms_a) != list(atoms_b):
        return False
    import numpy as _np

    coords_a = _np.asarray(coords_a, dtype=float)
    coords_b = _np.asarray(coords_b, dtype=float)
    if coords_a.shape != coords_b.shape:
        return False
    diff = coords_a - coords_b
    rmsd = float(_np.sqrt(_np.mean(_np.sum(diff * diff, axis=1))))
    return rmsd <= rmsd_tol


def _refresh_seed_options(app: Any, dropdown: Any) -> None:
    """Populate a geo-opt seed dropdown filtered by strict atom+coord match.

    Shared helper used by both Frequency and UV-Vis (TD-DFT) seed dropdowns.
    Filter cascade (HIST.6):

    1. No active molecule → list every geo-opt result (no filter; lets the
       user browse history before loading anything).
    2. Formula mismatch → exclude (cheap pre-filter; avoids disk reads).
    3. Same formula, but the candidate's ``trajectory.json`` starting frame
       has a different atom list (in order) OR an RMSD greater than
       ``_SEED_GEOMETRY_RMSD_TOLERANCE`` against the active molecule's
       coordinates → exclude.
    4. Atoms match AND RMSD within tolerance → include.

    If the active molecule's coordinates can't be read (e.g. fresh app with
    no molecule built yet) or a candidate's trajectory.json is malformed,
    falls back to the formula-only filter for that candidate.
    """
    from quantui.results_storage import list_results, load_result

    current_formula: str | None = None
    current_atoms = None
    current_coords = None
    mol = getattr(app, "_molecule", None)
    if mol is not None:
        try:
            current_formula = mol.get_formula()
        except Exception:
            current_formula = None
        try:
            import numpy as _np

            current_atoms = list(mol.atoms)
            current_coords = _np.array(mol.coordinates, dtype=float)
        except Exception:
            current_atoms = None
            current_coords = None

    options = [("(use current molecule)", "")]
    for d in list_results():
        try:
            data = load_result(d)
            if data.get("calc_type") != "geometry_opt":
                continue
            if current_formula is not None and data.get("formula") != current_formula:
                continue
            traj_file = d / "trajectory.json"
            if not traj_file.exists():
                continue
            # Strict atom + coord match when we have something to compare to.
            if current_atoms is not None and current_coords is not None:
                starting = _load_starting_geometry(d)
                if starting is not None:
                    cand_atoms, cand_coords = starting
                    if not _geometries_match(
                        current_atoms, current_coords, cand_atoms, cand_coords
                    ):
                        continue
                # If starting geometry can't be read, fall through to
                # formula-only match (don't punish the user for a malformed
                # trajectory.json on an otherwise-matching result).
            ts = data.get("timestamp", d.name[:19])
            label = f"{data['formula']}  {data['method']}/{data['basis']}" f"  —  {ts}"
            options.append((label, str(d)))
        except Exception:
            continue
    dropdown.options = options


def refresh_freq_seed_options(app: Any) -> None:
    """Populate frequency seed dropdown with saved geometry optimisations."""
    _refresh_seed_options(app, app._freq_seed_dd)


def refresh_tddft_seed_options(app: Any) -> None:
    """Populate UV-Vis (TD-DFT) seed dropdown with saved geometry optimisations."""
    _refresh_seed_options(app, app._tddft_seed_dd)


def on_freq_seed_changed(app: Any, change: Any) -> None:
    """Enable/disable pre-opt checkbox and update seed note message."""
    path_str = change["new"]
    if path_str:
        app._freq_preopt_cb.value = False
        app._freq_preopt_cb.disabled = True
        app._freq_seed_note.value = (
            '<span style="font-size:12px;color:#16a34a">'
            "✓ Final optimised geometry will be loaded from the selected result."
            "</span>"
        )
    else:
        app._freq_preopt_cb.disabled = False
        app._freq_seed_note.value = ""


def on_tddft_seed_changed(app: Any, change: Any) -> None:
    """Enable/disable pre-opt checkbox and update UV-Vis seed note message.

    Mirrors on_freq_seed_changed: a loaded seed geometry is already optimised,
    so the global pre-opt checkbox is disabled while a seed is selected.
    """
    path_str = change["new"]
    if path_str:
        app._freq_preopt_cb.value = False
        app._freq_preopt_cb.disabled = True
        app._tddft_seed_note.value = (
            '<span style="font-size:12px;color:#16a34a">'
            "✓ Final optimised geometry will be loaded from the selected result."
            "</span>"
        )
    else:
        app._freq_preopt_cb.disabled = False
        app._tddft_seed_note.value = ""


def on_solvent_cb_changed(app: Any, change: Any) -> None:
    """Show or hide solvent dropdown based on checkbox state."""
    app.solvent_dd.layout.display = "" if change["new"] else "none"


def on_clear_log(app: Any, btn: Any) -> None:
    """Clear the live run output panel."""
    app.run_output.clear_output()


def on_accumulate(app: Any, btn: Any) -> None:
    """Add the latest result to the in-session comparison list."""
    r = app._last_result
    if r is None:
        return
    app._results.append(r)
    app._refresh_comparison()


def on_clear(app: Any, btn: Any) -> None:
    """Clear in-session comparison results and rendered output."""
    app._results.clear()
    app.comparison_output.clear_output()


def on_compare_refresh(app: Any, btn: Any) -> None:
    """Refresh Compare selector options from saved results."""
    app._populate_compare_list()


def on_compare(app: Any, btn: Any, *, layout_fn: Any) -> None:
    """Render selected saved results in the Compare tab."""
    from pathlib import Path

    selected = app.compare_select.value
    if not selected or selected == ("",):
        return
    app.compare_output.clear_output(wait=True)
    from quantui import (
        comparison_table_html,
        plot_comparison,
        summary_from_saved_result,
    )
    from quantui.results_storage import load_result

    summaries = []
    valid_dirs: list[Any] = []
    for path_str in selected:
        if not path_str:
            continue
        try:
            data = load_result(Path(path_str))
            summaries.append(summary_from_saved_result(data))
            valid_dirs.append(Path(path_str))
        except Exception as exc:
            with app.compare_output:
                display(
                    HTML(f'<p style="color:#ef4444">Error loading result: {exc}</p>')
                )
    if not summaries:
        return
    with app.compare_output:
        display(HTML(comparison_table_html(summaries)))
        if len(summaries) > 1:
            try:
                import matplotlib.pyplot as plt

                fig = plot_comparison(summaries)
                display(fig)
                plt.close(fig)
            except Exception:
                pass
        if valid_dirs:
            btns = []
            for s, rdir in zip(summaries, valid_dirs):
                short = f"{s.formula} {s.method}/{s.basis}"
                button = widgets.Button(
                    description=f"→ Analyse  {short}"[:48],
                    button_style="info",
                    layout=layout_fn(width="auto", max_width="340px"),
                    tooltip=f"Load {short} into the Analysis tab",
                )
                button.on_click(
                    lambda _, rd=rdir, b=button: app._history_load_analysis(
                        rd, source_btns=(b,)
                    )
                )
                btns.append(button)
            display(
                widgets.HTML(
                    '<p style="margin:12px 0 4px;color:#475569;'
                    'font-size:13px;font-weight:600">Analyse a result:</p>'
                )
            )
            display(widgets.VBox(btns, layout=layout_fn(gap="4px")))


def on_compare_clear(app: Any, btn: Any) -> None:
    """Clear Compare tab selection and output area."""
    app.compare_select.value = ()
    app.compare_output.clear_output()


def on_past_refresh(app: Any, btn: Any) -> None:
    """Refresh History saved-results browser."""
    app._refresh_results_browser()


def on_copy_results_path(app: Any, btn: Any) -> None:
    """Copy results directory path to clipboard and show transient status."""
    p = app._get_results_dir()
    p.mkdir(parents=True, exist_ok=True)
    path_str = str(p).replace("\\", "\\\\").replace("'", "\\'")
    display(Javascript(f"navigator.clipboard.writeText('{path_str}')"))
    app.results_path_lbl.value = (
        f'<span style="color:#22c55e;font-size:13px">Copied: {p}</span>'
    )

    def _reset() -> None:
        time.sleep(3)
        app.results_path_lbl.value = (
            f'<span style="font-size:13px;color:#64748b">{p}</span>'
        )

    threading.Thread(target=_reset, daemon=True).start()


def on_reset_click(app: Any, btn: Any) -> None:
    """Reveal the perf-log reset confirmation controls."""
    app._reset_confirm_box.layout.display = ""


def on_confirm_yes(app: Any, btn: Any, *, reset_perf_log_fn: Any) -> None:
    """Reset performance log after confirmation and refresh summary stats."""
    reset_perf_log_fn()
    app._reset_confirm_box.layout.display = "none"
    app._refresh_perf_stats()


def on_confirm_no(app: Any, btn: Any) -> None:
    """Cancel perf-log reset confirmation prompt."""
    app._reset_confirm_box.layout.display = "none"


def on_log_clear(app: Any, btn: Any) -> None:
    """Clear rendered event-log output widgets in the Log tab."""
    app._log_output_html.value = (
        '<span style="color:#94a3b8;font-size:13px">Log cleared.</span>'
    )
    app._log_source_lbl.value = ""


def on_clear_log_cache(app: Any, _unused: Any = None) -> None:
    """First click handler for event-log cache clear workflow."""
    app._clear_log_cache_confirm_btn.layout.display = ""
    app._clear_log_cache_btn.disabled = True


def on_clear_log_cache_confirm(app: Any, *, calc_log_mod: Any) -> None:
    """Second click handler that clears persisted event log and restores UI."""
    try:
        calc_log_mod.log_event(
            "log_cleared",
            "Session event log cleared by user",
            session_id=app._session_id,
        )
        calc_log_mod.clear_event_log()
    except Exception:
        pass
    app._clear_log_cache_confirm_btn.layout.display = "none"
    app._clear_log_cache_btn.disabled = False


def on_help_toggle(app: Any, _unused: Any = None) -> None:
    """Toggle visibility of the floating Help overlay panel."""
    visible = app.help_tab_panel.layout.display != "none"
    app.help_tab_panel.layout.display = "none" if visible else ""


def on_help_topic_changed(app: Any, change: Any = None) -> None:
    """Refresh help topic content after selector changes."""
    _ = change
    app._render_help_topic()


def on_issue_btn(app: Any, _unused: Any = None) -> None:
    """Open the issue-report overlay and reset transient form status."""
    app._issue_textarea.value = ""
    app._issue_status_html.value = ""
    app._issue_overlay.layout.display = ""


def on_issue_cancel(app: Any, _unused: Any = None) -> None:
    """Dismiss the issue-report overlay without saving."""
    app._issue_overlay.layout.display = "none"


def on_issue_submit(app: Any, *, issue_tracker_mod: Any) -> None:
    """Persist issue text and hide overlay on success."""
    text = app._issue_textarea.value.strip()
    if not text:
        app._issue_status_html.value = (
            '<span style="color:#b91c1c;font-size:12px">'
            "Please describe the issue before submitting.</span>"
        )
        return
    app._issue_submit_btn.disabled = True
    try:
        issue_id = issue_tracker_mod.log_issue(
            description=text,
            context=app._build_issue_context(),
            session_id=app._session_id,
        )
        app._issue_status_html.value = (
            f'<span style="color:#16a34a;font-size:12px">'
            f"&#10003; Issue #{issue_id} saved. Thank you!</span>"
        )
        app._issue_overlay.layout.display = "none"
    except Exception as exc:
        app._issue_status_html.value = (
            f'<span style="color:#b91c1c;font-size:12px">Save failed: {exc}</span>'
        )
    finally:
        app._issue_submit_btn.disabled = False


def on_expand_mol_input(app: Any, btn: Any, *, visualization_available: bool) -> None:
    """Expand molecule input section to show full editor and controls."""
    _ = btn
    children = [app.mol_input_expanded, app.mol_info_html, app.viz_output]
    if app.viz_backend_toggle is not None:
        children.append(app.viz_backend_toggle)
    if visualization_available:
        children.append(app.viz_controls_box)
    app.mol_input_container.children = children


def on_method_help(app: Any, btn: Any) -> None:
    """Open help overlay focused on method guidance."""
    _ = btn
    app._show_help_topic("method")


def on_basis_help(app: Any, btn: Any) -> None:
    """Open help overlay focused on basis-set guidance."""
    _ = btn
    app._show_help_topic("basis_set")


def on_exit_clicked(app: Any, _unused: Any = None) -> None:
    """Update UI and request shutdown of Voilà/Jupyter parent and kernel."""
    import os
    import signal

    app._exit_btn.description = "Exiting…"
    app._exit_btn.disabled = True
    app._welcome_html.value = (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'padding:32px;gap:16px">'
        '<svg width="40" height="40" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="140" cy="140" r="48" fill="rgba(37,99,235,0.15)"/>'
        '<circle cx="140" cy="140" r="14" fill="#2563eb"/>'
        '<circle cx="140" cy="140" r="8" fill="#60a5fa"/>'
        "</svg>"
        '<div style="font-size:20px;color:#475569">'
        "QuantUI has shut down. You may close this tab.</div>"
        "</div>"
    )

    def _do_exit() -> None:
        time.sleep(0.6)
        try:
            # Signal the Voilà/Jupyter server process (our parent) to exit cleanly.
            os.kill(os.getppid(), signal.SIGTERM)
        except Exception:
            pass
        # Terminate the kernel process regardless.
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True).start()


def on_cal_run(
    app: Any,
    btn: Any,
    *,
    benchmark_suite: Any,
    benchmark_suite_long: Any,
) -> None:
    """Start async calibration run and initialize calibration UI state."""
    _ = btn
    mode = app._cal_mode_toggle.value
    suite = benchmark_suite if mode == "short" else benchmark_suite_long
    app._cal_stop_event = threading.Event()
    app._cal_run_btn.disabled = True
    app._cal_mode_toggle.disabled = True
    app._cal_stop_btn.layout.display = ""
    app._cal_progress.max = len(suite)
    app._cal_progress.value = 0
    app._cal_progress.layout.display = ""
    app._cal_step_label.layout.display = ""
    app._cal_step_label.value = (
        '<span style="font-size:12px;color:#475569">Starting…</span>'
    )
    app._cal_results_html.value = ""

    threading.Thread(target=app._do_calibration, daemon=True).start()


def on_cal_stop(app: Any, btn: Any) -> None:
    """Signal any active calibration run to stop at the next safe point."""
    _ = btn
    if hasattr(app, "_cal_stop_event"):
        app._cal_stop_event.set()


def do_calibration(app: Any, *, pyscf_available: bool) -> None:
    """Run calibration suite and render calibration summary table."""
    from quantui.benchmarks import run_calibration

    mode = app._cal_mode_toggle.value

    def _progress(
        step_n: int, total: int, label: str, status: str, elapsed: float
    ) -> None:
        icon = {"ok": "✓", "timed_out": "⏱", "stopped": "⛔", "error": "✗"}.get(
            status, "?"
        )
        app._cal_progress.value = step_n
        app._cal_step_label.value = (
            f'<span style="font-size:12px;color:#475569">'
            f"Step {step_n} / {total} — {label} "
            f"[{icon} {elapsed:.1f} s]</span>"
        )

    result = run_calibration(
        progress_cb=_progress,
        stop_event=app._cal_stop_event,
        timeout_per_step=300.0 if mode == "long" else 120.0,
        mode=mode,
    )

    rows = "".join(
        f"<tr>"
        f'<td style="padding:2px 12px 2px 0;font-size:12px">{s.label}</td>'
        f'<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">'
        f"{s.n_electrons}</td>"
        f'<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">'
        f"{s.n_basis if s.n_basis is not None else '—'}</td>"
        f'<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">'
        f"{s.elapsed_s:.2f} s</td>"
        f'<td style="padding:2px 0;font-size:12px">'
        f'{"✓" if s.status == "ok" else ("⏱ timed out" if s.status == "timed_out" else ("⛔ stopped" if s.status == "stopped" else "✗ error"))}'
        f"</td>"
        f"</tr>"
        for s in result.steps
    )
    summary = f"Completed {result.n_completed} / {result.n_total} steps." + (
        " (stopped early)" if result.stopped_early else ""
    )
    app._cal_results_html.value = (
        f'<div style="margin-top:8px">'
        f'<p style="font-size:13px;color:#374151;margin:0 0 6px">{summary}</p>'
        f'<table style="border-collapse:collapse">'
        f"<tr>"
        f'<th style="padding:2px 12px 2px 0;font-size:12px;text-align:left">Calculation</th>'
        f'<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">e⁻</th>'
        f'<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">Basis fns</th>'
        f'<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">Wall time</th>'
        f'<th style="padding:2px 0;font-size:12px">Status</th>'
        f"</tr>"
        f"{rows}</table></div>"
    )

    app._cal_step_label.value = (
        '<span style="font-size:12px;color:#16a34a"><b>Calibration complete.</b> '
        "Time estimates are now active.</span>"
        if result.n_completed > 0
        else '<span style="font-size:12px;color:#dc2626">No steps completed.</span>'
    )
    app._cal_stop_btn.layout.display = "none"
    app._cal_run_btn.disabled = not pyscf_available
    app._cal_mode_toggle.disabled = False
    app._refresh_perf_stats()


def update_notes(app: Any, change: Any = None) -> None:
    """Refresh educational method notes for the active molecule/method."""
    app.notes_output.clear_output(wait=True)
    if app._molecule is None:
        return
    try:
        from quantui import PySCFCalculation

        calc = PySCFCalculation(
            app._molecule,
            method=app.method_dd.value,
            basis=app.basis_dd.value,
        )
        notes = calc.get_educational_notes()
        if notes:
            safe = (
                notes.replace("**", "<b>", 1)
                .replace("**", "</b>", 1)
                .replace("\n\n", "<br><br>")
            )
            with app.notes_output:
                display(
                    HTML(
                        '<div style="background:#fffbf0;padding:8px 12px;'
                        'border-radius:4px;font-size:13px;margin-top:6px">'
                        + safe
                        + "</div>"
                    )
                )
    except Exception:
        pass


def update_estimate(app: Any, *, calc_log_mod: Any, change: Any = None) -> None:
    """Refresh runtime estimate text from the performance model."""
    if app._molecule is None:
        app.perf_estimate_html.value = ""
        return
    try:
        calc_type = {
            "Single Point": "single_point",
            "Geometry Opt": "geometry_opt",
            "Frequency": "frequency",
            "UV-Vis (TD-DFT)": "tddft",
            "NMR Shielding": "nmr",
            "PES Scan": "pes_scan",
        }.get(app.calc_type_dd.value, "single_point")
        n_basis = calc_log_mod.count_basis_functions(
            app._molecule.atoms, app.basis_dd.value
        )
        est = calc_log_mod.estimate_time(
            n_atoms=len(app._molecule.atoms),
            n_electrons=app._molecule.get_electron_count(),
            method=app.method_dd.value,
            basis=app.basis_dd.value,
            n_basis=n_basis,
            calc_type=calc_type,
        )
        app.perf_estimate_html.value = calc_log_mod.format_estimate(est)
    except Exception:
        app.perf_estimate_html.value = ""


def refresh_results_browser(app: Any) -> None:
    """Refresh the History dropdown with saved result directories."""
    try:
        from quantui import list_results, load_result
    except ImportError:
        return
    app.results_path_lbl.value = (
        f'<span style="font-size:13px;color:#64748b">'
        f"{app._get_results_dir()}</span>"
    )
    dirs = list_results()
    if not dirs:
        app.past_dd.options = [("(no saved results)", "")]
        return
    options = []
    for d in dirs:
        try:
            data = load_result(d)
            ts = data.get("timestamp", d.name)
            calc_badge = _calc_type_badge(data.get("calc_type", ""))
            label = (
                f"{ts}  ·  [{calc_badge}]  "
                f"{data['formula']}  {data['method']}/{data['basis']}"
            )
            options.append((label, str(d)))
        except Exception:
            pass
    app.past_dd.options = options if options else [("(no saved results)", "")]
    if app.calc_type_dd.value == "Frequency":
        app._refresh_freq_seed_options()


def refresh_comparison(app: Any) -> None:
    """Refresh in-session comparison output from accumulated results."""
    from quantui import comparison_table_html, summary_from_session_result

    app.comparison_output.clear_output(wait=True)
    if not app._results:
        return
    summaries = [summary_from_session_result(r) for r in app._results]
    with app.comparison_output:
        display(HTML(comparison_table_html(summaries)))
        if len(summaries) > 1:
            try:
                from quantui import plot_comparison

                plot_comparison(summaries)
            except Exception:
                pass


def populate_compare_list(app: Any) -> None:
    """Populate the Compare tab selector with saved result entries."""
    from quantui.results_storage import list_results, load_result

    dirs = list_results()
    if not dirs:
        app.compare_select.options = [("(no saved results)", "")]
        app.compare_btn.disabled = True
        return
    options = []
    for d in dirs:
        try:
            data = load_result(d)
            ts = data.get("timestamp", d.name[:19])
            calc_badge = _calc_type_badge(data.get("calc_type", ""))
            label = (
                f"{ts}  [{calc_badge}]  "
                f"{data['formula']}  {data['method']}/{data['basis']}"
            )
            options.append((label, str(d)))
        except Exception:
            options.append((d.name, str(d)))
    app.compare_select.options = options
    app.compare_btn.disabled = False
