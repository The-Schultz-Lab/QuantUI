"""Runflow helpers used by QuantUIApp."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

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
        "reorganization_energy": "Reorg",
    }.get(calc_type, calc_type or "Unknown")


# Calc-type dropdown label → canonical schema key (used for the header banner).
_CALC_TYPE_CANON = {
    "Geometry Opt": "geometry_opt",
    "Frequency": "frequency",
    "UV-Vis (TD-DFT)": "tddft",
    "NMR Shielding": "nmr",
    "PES Scan": "pes_scan",
    "Reorganization Energy": "reorganization_energy",
}


def _write_run_header(app: Any) -> None:
    """Write the full run header to the live log — synchronously, atomically.

    UXP.6 + bug fix (2026-07-18): the header used to be written two ways, both
    unreliable in Voilà — a provisional line via ``clear_output()`` +
    ``append_stdout()`` (the non-atomic combo ``_set_html_output`` exists to
    avoid) and the structured banner via ``append_stdout`` from the *background*
    ``_do_run`` thread (a bg-thread ``.outputs`` mutation, which this app
    marshals through the io_loop everywhere else). For a large molecule the long
    gap before the first optimizer/SCF step exposed the race: the pre-step-1
    stream (both headers) was dropped and the log jumped straight to ``BFGS: 0``.

    Fix: build the whole banner and assign it in a single atomic
    ``run_output.outputs = (…)`` on the main thread (the click handler). No
    ``clear_output``, no bg-thread write, no intermediate empty state.
    ``get_system_info()`` is ``lru_cache``d (warmed at startup) so this stays
    instant. Later PySCF / optimizer output appends onto this header as before.
    """
    mol = getattr(app, "_molecule", None)
    if mol is None:
        # Nothing will run; just clear the previous log atomically.
        _set_run_output(app, ())
        return
    try:
        from quantui.log_utils import format_log_header

        calc_type = _CALC_TYPE_CANON.get(app.calc_type_dd.value, "single_point")
        try:
            _n_atoms = len(mol.atoms)
        except Exception:
            _n_atoms = None
        _solvent = app.solvent_dd.value if app.solvent_cb.value else None
        try:
            _out_dir = str(app._get_results_dir())
        except Exception:
            _out_dir = None
        banner = format_log_header(
            formula=mol.get_formula(),
            method=app.method_dd.value,
            basis=app.basis_dd.value,
            calc_type=calc_type,
            n_atoms=_n_atoms,
            multiplicity=int(app.mult_si.value),
            solvent=_solvent,
            output_dir=_out_dir,
        )
    except Exception:
        # Fallback: a minimal one-liner still beats a blank window.
        try:
            banner = (
                f"▶ Starting {app.calc_type_dd.value} — {mol.get_formula()} · "
                f"{app.method_dd.value}/{app.basis_dd.value} …\n"
            )
        except Exception:
            banner = "▶ Starting calculation …\n"
    _set_run_output(
        app,
        ({"output_type": "stream", "name": "stdout", "text": banner},),
    )


def _set_run_output(app: Any, outputs: tuple) -> None:
    """Atomically set ``run_output.outputs`` (fallback to clear_output)."""
    try:
        app.run_output.outputs = outputs
    except Exception:
        try:
            app.run_output.clear_output()
            for o in outputs:
                app.run_output.append_stdout(o.get("text", ""))
        except Exception:
            pass


def on_run_clicked(app: Any, btn: Any) -> None:
    """Reset result panes and start the background run thread."""
    # Write the header FIRST (atomic, main thread) — this also clears the
    # previous run's log via the single ``outputs`` assignment.
    _write_run_header(app)
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

    # The "geometry optimization before this calc" checkbox is meaningful
    # for all workflows except Geometry Opt itself (which IS the geom-opt
    # workflow). POLISH.9: this was called "pre-optimisation" pre-2026-05-25;
    # the underlying operation is a full DFT geom-opt — distinct from the
    # LJ classical pre-opt in quantui/preopt.py.
    # Reorganization Energy runs its own neutral + ion optimizations, so the
    # standalone "geometry optimization before this calc" checkbox is
    # meaningless there too (as with Geometry Opt itself).
    if ct in ("Geometry Opt", "Reorganization Energy"):
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
    elif ct == "Reorganization Energy":
        app.calc_extra_opts.children = [
            app._reorg_mode_dd,
            app._reorg_note,
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
            # Every entry here is a geometry_opt result; prefix the label so the
            # user can see the seed is an optimized geometry, not a raw input.
            label = (
                f"⚙ Geom-opt · {data['formula']}  "
                f"{data['method']}/{data['basis']}  —  {ts}"
            )
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
    """Clear the live run output panel — but never mid-run.

    The button is disabled while a calc runs (``_do_run``), but guard here too:
    clearing mid-run wipes the header + in-progress output while the background
    thread keeps appending, leaving a confusing headerless log. Use Cancel to
    stop a run, then Clear.
    """
    if getattr(app, "_calc_running", False):
        app.run_status.value = "Can't clear while a calculation is running."
        return
    app.run_output.clear_output()


def _preopt_small(text: str, color: str = "#555") -> str:
    return f'<small style="color:{color}">{text}</small>'


def on_preopt_preview(app: Any, btn: Any = None) -> None:
    """Run the classical pre-opt on demand and animate the relaxation in-place.

    M-PREOPT PREOPT.2: instead of the pre-opt being a silent step inside the
    run, the user previews it here — watches the bonded-FF relaxation animate
    in the Calculate tab — then Keeps or Reverts it (PREOPT.3). The pre-opt runs
    on a background thread; UI updates are marshalled back to the main thread.
    """
    mol = getattr(app, "_molecule", None)
    if mol is None:
        app.preopt_preview_box.layout.display = ""
        app.preopt_preview_status.value = _preopt_small("Load a molecule first.")
        return
    app.preopt_preview_btn.disabled = True
    app.preopt_accept_btn.disabled = True
    app.preopt_reset_btn.disabled = True
    app.preopt_preview_box.layout.display = ""
    app.preopt_preview_status.value = _preopt_small(
        "⏳ Pre-optimizing — watch it relax below…"
    )
    try:
        app._activity_begin("Previewing pre-optimization…", kind="ui")
    except Exception:
        pass
    threading.Thread(
        target=_preopt_preview_worker, args=(app, mol), daemon=True
    ).start()


def _preopt_preview_worker(app: Any, mol: Any) -> None:
    """Background: run the trajectory-capturing pre-opt, then update UI on main."""
    from quantui.preopt import preoptimize_with_trajectory

    try:
        relaxed, rmsd, frames = preoptimize_with_trajectory(mol)
    except Exception as exc:  # noqa: BLE001 — surface as an inline message
        app._queue_main_thread_callback(_preopt_preview_failed, app, str(exc))
        return
    app._queue_main_thread_callback(_preopt_preview_done, app, relaxed, rmsd, frames)


def _preopt_preview_done(app: Any, relaxed: Any, rmsd: float, frames: Any) -> None:
    """Main thread: animate the relaxation + reveal Keep/Revert."""
    from quantui.app_visualization import build_preopt_preview_html

    app._preopt_relaxed_mol = relaxed
    app.preopt_preview_box.layout.display = ""  # ensure visible when results land
    try:
        bg = app._plotly_theme_colors()["scene_bgcolor"]
    except Exception:
        bg = "white"
    try:
        html = build_preopt_preview_html(list(relaxed.atoms), frames, bgcolor=bg)
        app._set_html_output(app.preopt_preview_output, html)
    except Exception as exc:  # noqa: BLE001
        app.preopt_preview_output.clear_output()
        with app.preopt_preview_output:
            display(HTML(_preopt_small(f"Preview render failed: {exc}", "#b91c1c")))

    if rmsd > 1e-3:
        note = f"moved <b>{rmsd:.3f} Å</b> (RMSD) from your input"
    else:
        note = "barely changed — your geometry was already reasonable"
    app.preopt_preview_status.value = _preopt_small(
        f"Relaxed (MMFF94/UFF): {note}. Use ⇄ or the slider below to compare "
        "input vs relaxed, then Keep it or revert.",
        "#444",
    )
    app.preopt_preview_btn.disabled = False
    app.preopt_accept_btn.disabled = False
    app.preopt_reset_btn.disabled = False
    try:
        app._activity_end(kind="ui")
    except Exception:
        pass


def _preopt_preview_failed(app: Any, msg: str) -> None:
    app.preopt_preview_status.value = _preopt_small(f"Preview failed: {msg}", "#b91c1c")
    app.preopt_preview_btn.disabled = False
    try:
        app._activity_end(kind="ui")
    except Exception:
        pass


def on_preopt_accept(app: Any, btn: Any = None) -> None:
    """Make the previewed relaxed geometry the active molecule (PREOPT.3)."""
    relaxed = getattr(app, "_preopt_relaxed_mol", None)
    if relaxed is None:
        return
    app._set_molecule(relaxed, "Pre-optimized (MMFF94/UFF — accepted from preview)")
    # The active geometry IS the relaxed one now; the run uses it as-is (there is
    # no silent pre-opt step to disable — classical pre-opt is Preview-only).
    app._preopt_relaxed_mol = None
    app.preopt_preview_box.layout.display = "none"
    app.preopt_preview_output.clear_output()
    app.preopt_preview_status.value = ""
    app.run_status.value = "Pre-optimized geometry accepted."


def on_preopt_reset(app: Any, btn: Any = None) -> None:
    """Discard the preview; the original active geometry is untouched."""
    app._preopt_relaxed_mol = None
    app.preopt_preview_box.layout.display = "none"
    app.preopt_preview_output.clear_output()
    app.preopt_preview_status.value = ""
    # Drop any stale "Pre-optimized geometry accepted." left from a prior accept.
    if not app._calc_running:
        app.run_status.value = ""


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


def on_calc_type_help(app: Any, btn: Any) -> None:
    """Open help overlay focused on calculation-type guidance."""
    _ = btn
    app._show_help_topic("calc_type")


def on_exit_clicked(app: Any, _unused: Any = None) -> None:
    """Update UI and request shutdown of Voilà/Jupyter parent and kernel."""
    import os
    import signal

    app._exit_btn.description = "Exiting…"
    app._exit_btn.disabled = True
    # POLISH.1 retry-2 (2026-05-25): the welcome logo now lives in its
    # own ``widgets.Image`` next to the text. At shutdown hide the logo
    # so the centered "QuantUI has shut down" message isn't off-center.
    if hasattr(app, "_welcome_logo"):
        try:
            app._welcome_logo.layout.display = "none"
        except Exception:  # noqa: BLE001 — best-effort UI tweak
            pass
    app._welcome_html.value = (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'padding:32px;gap:16px;width:100%">'
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
    # session 55 hotfix: the old ``"short" else "long"`` two-tier dispatch
    # silently routed tier 3 / tier 4 (and tier 1!) to the tier-2 suite,
    # which set ``progress_bar.max = 20`` while tier 1 only ran 8 steps
    # — the bar froze at 40% on completion. Use the 4-tier lookup so
    # ``max`` matches the actual step count.
    from quantui.benchmarks import _MODE_TO_SUITE

    suite = _MODE_TO_SUITE.get(mode, benchmark_suite)
    app._cal_stop_event = threading.Event()
    # session 55 user request: skip-current-step event, separate from
    # the whole-run stop event. Replaces the hard per-step timeout.
    app._cal_skip_event = threading.Event()
    app._cal_run_btn.disabled = True
    app._cal_mode_toggle.disabled = True
    app._cal_stop_btn.layout.display = ""
    app._cal_skip_btn.layout.display = ""
    app._cal_progress.max = len(suite)
    app._cal_progress.value = 0
    app._cal_progress.layout.display = ""
    app._cal_step_label.layout.display = ""
    app._cal_step_label.value = (
        '<span style="font-size:12px;color:#475569">Starting…</span>'
        # Reserve a second invisible line so the live-message ticker
        # doesn't jump the accordion height (session 55 user report).
        '<br><span style="font-size:11px;color:transparent">.</span>'
    )
    app._cal_results_html.value = ""

    threading.Thread(target=app._do_calibration, daemon=True).start()


def on_cal_stop(app: Any, btn: Any) -> None:
    """Signal any active calibration run to stop at the next safe point."""
    _ = btn
    if hasattr(app, "_cal_stop_event"):
        app._cal_stop_event.set()


def on_cal_skip(app: Any, btn: Any) -> None:
    """Signal the active calibration to skip the CURRENT step + continue.

    Replaces the per-step timeout (session 55 user request after a
    near-finishing benzene B3LYP/6-31G* freq calc got cut off at the
    1800 s tier-4 cap). The worker is killed, the step is marked
    ``skipped``, the event is cleared inside ``run_calibration``, and
    the loop moves on to the next step.
    """
    _ = btn
    if hasattr(app, "_cal_skip_event"):
        app._cal_skip_event.set()


def _cal_status_text(status: str) -> str:
    """Render a benchmark-step status code as a glanceable HTML cell."""
    return {
        "ok": "✓",
        "timed_out": "⏱ timed out",
        "stopped": "⛔ stopped",
        "skipped": "⏭ skipped",
        "error": "✗ error",
        "running": "▶ running",
    }.get(status, status)


def _cal_table_html(steps_so_far, total: int, *, in_flight_step=None) -> str:
    """Render the calibration results table.

    Called incrementally — after every completed step — so the user sees
    rows accumulate in real time instead of waiting for the whole tier
    to finish (session 55 user request). ``steps_so_far`` is the list of
    ``BenchmarkStep`` objects completed; ``in_flight_step`` (optional)
    is a dict ``{label, n_electrons, n_basis, status, elapsed_s}`` that
    appends a "running" row at the bottom while a step is mid-execution.

    For failed steps (error / timeout / skipped) we render an inline
    italic line below the status cell with a truncated ``error_msg``,
    so the user can see WHY a step failed without having to open
    ``calibration.json`` (session 55 user request after MP2/CCSD on
    H₂O/cc-pVDZ silently 'errored' with no on-screen explanation).
    """
    import html as _html_mod

    row_tpl = (
        "<tr>"
        '<td style="padding:2px 12px 2px 0;font-size:12px">{label}</td>'
        '<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">{ne}</td>'
        '<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">{nb}</td>'
        '<td style="padding:2px 8px 2px 0;font-size:12px;text-align:right">{t:.2f} s</td>'
        '<td style="padding:2px 0;font-size:12px">{status}{detail}</td>'
        "</tr>"
    )

    def _err_detail(s) -> str:
        # Show err_msg inline only for non-ok terminal statuses.
        msg = getattr(s, "error_msg", "") or ""
        if not msg or s.status in ("ok", "running"):
            return ""
        # Truncate hard so a verbose PySCF traceback can't blow up the row.
        if len(msg) > 140:
            msg = msg[:137] + "…"
        return (
            '<br><span style="color:#94a3b8;font-style:italic;font-size:11px">'
            f"{_html_mod.escape(msg)}</span>"
        )

    rows = "".join(
        row_tpl.format(
            label=s.label,
            ne=s.n_electrons,
            nb=s.n_basis if s.n_basis is not None else "—",
            t=s.elapsed_s,
            status=_cal_status_text(s.status),
            detail=_err_detail(s),
        )
        for s in steps_so_far
    )
    if in_flight_step is not None:
        rows += row_tpl.format(
            label=in_flight_step["label"],
            ne=in_flight_step.get("n_electrons", "—"),
            nb=in_flight_step.get("n_basis", "—") or "—",
            t=in_flight_step.get("elapsed_s", 0.0),
            status=_cal_status_text("running"),
            detail="",
        )

    n_done = sum(1 for s in steps_so_far if s.status == "ok")
    summary = f"Completed {n_done} / {total} steps."
    return (
        '<div style="margin-top:8px">'
        f'<p style="font-size:13px;color:#374151;margin:0 0 6px">{summary}</p>'
        '<table style="border-collapse:collapse">'
        "<tr>"
        '<th style="padding:2px 12px 2px 0;font-size:12px;text-align:left">Calculation</th>'
        '<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">e⁻</th>'
        '<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">Basis fns</th>'
        '<th style="padding:2px 8px 2px 0;font-size:12px;text-align:right">Wall time</th>'
        '<th style="padding:2px 0;font-size:12px">Status</th>'
        "</tr>"
        f"{rows}</table></div>"
    )


def do_calibration(app: Any, *, pyscf_available: bool) -> None:
    """Run calibration suite and render calibration summary table.

    Fixes shipped 2026-05-25 (session 55 user reports):

    - Wraps the whole run in ``_activity_begin/_end`` so the toolbar
      activity badge stops reading "Idle" while calibration is busy.
    - Per-step ``progress_cb`` writes a multi-line status block (live
      tail of the per-step PySCF / SCF log) so the user can see where
      a slow step is rather than guess whether it froze.
    - Table rows render incrementally (after each step completes)
      instead of all at once at end-of-run.
    - The live-message line is ALWAYS present (transparent placeholder
      when there's no message yet) so the accordion height doesn't
      flicker between one-line and two-line states.
    """
    from quantui.benchmarks import run_calibration

    mode = app._cal_mode_toggle.value
    # Total-step count comes via the ``total`` arg of the ``_progress``
    # callback; no need to compute it locally. (The earlier draft pulled
    # it from ``_MODE_TO_SUITE`` but never used it — ruff F841.)

    # session 55 user request (after a near-finishing benzene
    # B3LYP/6-31G* freq got cut off at the old 1800 s tier-4 cap):
    # no automatic timeout — the user controls long-running steps via
    # the Skip button. If they walk away from a runaway calc, the
    # Stop button is still available. Headless callers that genuinely
    # want a wall-clock cap can pass timeout_per_step explicitly.
    timeout_per_step: Optional[float] = None

    # M-EST follow-up: keep the toolbar activity badge red for the
    # duration of the calibration so the user knows the kernel is busy.
    app._activity_begin(f"Calibrating ({mode})…", kind="compute")

    # Per-step buffer of completed steps for incremental table rendering.
    # Steps accumulate here as soon as each one finishes.
    _completed_steps: list = []
    # Buffer for the currently-running step so we can show a "running"
    # row at the bottom of the table while it's in-flight.
    _in_flight: dict = {}

    def _progress(
        step_n: int,
        total: int,
        label: str,
        status: str,
        elapsed: float,
        *,
        live_message: Optional[str] = None,
        step: Any = None,
    ) -> None:
        """Per-step progress callback.

        Three call modes:
        - Live-tick: status is "running"; ``step`` is None. Updates
          the step label and shows an "in flight" row at the bottom
          of the table.
        - Step-finish: status is one of ok/timed_out/stopped/error;
          ``step`` is the completed ``BenchmarkStep``. Appends to the
          completed-steps buffer + re-renders the table.
        """
        icon = {
            "ok": "✓",
            "timed_out": "⏱",
            "stopped": "⛔",
            "error": "✗",
            "running": "▶",
        }.get(status, "?")
        if status != "running":
            app._cal_progress.value = step_n
            if step is not None:
                _completed_steps.append(step)
        # ALWAYS render two lines so the accordion height doesn't
        # flip-flop. Empty live-message becomes a transparent dot to
        # preserve the line-height.
        live_line_text = live_message if live_message else "."
        live_line_color = "#64748b" if live_message else "transparent"
        app._cal_step_label.value = (
            f'<span style="font-size:12px;color:#475569">'
            f"Step {step_n} / {total} — {label} "
            f"[{icon} {elapsed:.1f} s]</span>"
            f'<br><span style="font-size:11px;color:{live_line_color}">'
            f"{live_line_text}</span>"
        )

        # Refresh in-flight buffer + the table snapshot.
        if status == "running":
            # Pull electron-count / basis from the active suite entry so
            # the in-flight row has the same columns as completed rows.
            _in_flight.update(label=label, elapsed_s=elapsed)
            app._cal_results_html.value = _cal_table_html(
                _completed_steps, total, in_flight_step=_in_flight or None
            )
        else:
            _in_flight.clear()
            app._cal_results_html.value = _cal_table_html(_completed_steps, total)

    try:
        result = run_calibration(
            progress_cb=_progress,
            stop_event=app._cal_stop_event,
            timeout_per_step=timeout_per_step,
            mode=mode,
            skip_event=app._cal_skip_event,
        )
        # Belt-and-suspenders: re-render the table from the canonical
        # ``result.steps`` in case any per-step callback was dropped
        # (e.g. transient widget-update exception). The progress
        # callback should have already kept _completed_steps in sync.
        app._cal_results_html.value = _cal_table_html(
            list(result.steps), result.n_total
        )
    finally:
        app._activity_end(kind="compute")

    app._cal_step_label.value = (
        '<span style="font-size:12px;color:#16a34a"><b>Calibration complete.</b> '
        "Time estimates are now active.</span>"
        '<br><span style="font-size:11px;color:transparent">.</span>'
        if result.n_completed > 0
        else (
            '<span style="font-size:12px;color:#dc2626">No steps completed.</span>'
            '<br><span style="font-size:11px;color:transparent">.</span>'
        )
    )
    app._cal_stop_btn.layout.display = "none"
    app._cal_skip_btn.layout.display = "none"
    app._cal_run_btn.disabled = not pyscf_available
    app._cal_mode_toggle.disabled = False
    app._refresh_perf_stats()


def update_notes(app: Any, change: Any = None) -> None:
    """Refresh the method / basis descriptor cards + open-shell hint.

    Replaces the old inline educational-notes text block (UXP.7). The cards
    describe the *method* and *basis* themselves, so — unlike the old notes —
    they refresh independently of whether a molecule is loaded. The open-shell
    hint (restored from the old notes) appears only when multiplicity > 1.
    """
    try:
        from quantui.descriptor_cards import basis_card_html, method_card_html

        app._method_card_html.value = method_card_html(app.method_dd.value)
        app._basis_card_html.value = basis_card_html(app.basis_dd.value)
    except Exception:
        pass
    _update_open_shell_hint(app)


def _update_open_shell_hint(app: Any) -> None:
    """Show/hide the open-shell (multiplicity > 1 → UHF) guidance hint."""
    try:
        mult = int(app.mult_si.value)
    except Exception:
        mult = 1
    if mult <= 1:
        app._open_shell_hint.value = ""
        app._open_shell_hint.layout.display = "none"
        return
    n_unpaired = mult - 1
    plural = "s" if n_unpaired != 1 else ""
    if app.method_dd.value.upper() == "RHF":
        # Actionable: RHF is the one method that will misbehave for open-shell.
        app._open_shell_hint.value = (
            '<span style="font-size:12px;color:#b45309">'
            f"⚠ Open-shell: {n_unpaired} unpaired electron{plural} "
            f"(multiplicity {mult}). RHF assumes all electrons are paired — "
            "switch to <b>UHF</b> (or a DFT method) for this system.</span>"
        )
    else:
        # Informational: UHF / DFT already handle open-shell correctly.
        app._open_shell_hint.value = (
            '<span style="font-size:12px;color:#64748b">'
            f"Open-shell: {n_unpaired} unpaired electron{plural} "
            f"(multiplicity {mult}) — running unrestricted.</span>"
        )
    app._open_shell_hint.layout.display = ""


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
        # M-EST / EST.1: predict the device the upcoming run will use so
        # the estimator can partition history by GPU vs CPU. The method
        # also matters — gpu4pyscf doesn't support CCSD(T), so even on a
        # GPU machine that calc will run CPU-side.
        _predicted_gpu_used: Optional[bool] = None
        try:
            from quantui.gpu_offload import (
                _GPU_UNSUPPORTED_METHODS as _GPU_NO,
            )
            from quantui.gpu_offload import (
                is_gpu_available,
            )

            _gpu_avail, _ = is_gpu_available()
            if _gpu_avail and app.method_dd.value.upper() not in _GPU_NO:
                _predicted_gpu_used = True
            else:
                _predicted_gpu_used = False
        except Exception:  # noqa: BLE001 — fall back to device-agnostic prediction
            _predicted_gpu_used = None

        est = calc_log_mod.estimate_time(
            n_atoms=len(app._molecule.atoms),
            n_electrons=app._molecule.get_electron_count(),
            method=app.method_dd.value,
            basis=app.basis_dd.value,
            n_basis=n_basis,
            calc_type=calc_type,
            gpu_used=_predicted_gpu_used,
        )
        app.perf_estimate_html.value = calc_log_mod.format_estimate(est)
    except Exception:
        app.perf_estimate_html.value = ""


def refresh_results_browser(app: Any) -> None:
    """Refresh the History dropdown with saved result directories.

    POLISH.6 (M-POLISH, 2026-05-25): prepends a
    ``"(select a calculation to view)"`` placeholder so the dropdown
    opens in an explicit "no calc loaded yet" state. Without the
    placeholder, ipywidgets auto-selected the most-recent entry as the
    dropdown's ``value`` — visually implying the calc was loaded when
    actually the user still has to click "View Results" / "View
    Analysis" to populate the rest of the UI. The ``value`` observer
    fires when options are reassigned (the result card *is* shown),
    but no calc state is loaded into the app until the explicit
    button-click, which mismatched user expectation.

    The placeholder is always at index 0 of ``options`` so the
    Dropdown widget's value-preservation behaviour kicks in: a
    previously-picked real result survives a refresh, but the initial
    render shows the placeholder.
    """
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
    placeholder = ("(select a calculation to view)", "")
    options = [placeholder]
    for d in dirs:
        try:
            data = load_result(d)
            ts = data.get("timestamp", d.name)
            calc_badge = _calc_type_badge(data.get("calc_type", ""))
            # M-EST follow-up (2026-05-25): calibration-produced results
            # get a 🔧 marker so the user can tell them apart from
            # user-initiated calcs. The marker comes from result.json's
            # ``calibration_run_id`` extras field written by the worker.
            calib_marker = "🔧 " if data.get("calibration_run_id") else ""
            label = (
                f"{ts}  ·  [{calc_badge}]  "
                f"{calib_marker}{data.get('formula', '?')}  "
                f"{data.get('method', '?')}/{data.get('basis', '?')}"
            )
            options.append((label, str(d)))
        except Exception:
            pass
    # If the only entry is the placeholder, fall back to the empty-list
    # message — the loop above silently swallowed every load_result call.
    if len(options) == 1:
        app.past_dd.options = [("(no saved results)", "")]
        return
    app.past_dd.options = options
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
                f"{data.get('formula', '?')}  "
                f"{data.get('method', '?')}/{data.get('basis', '?')}"
            )
            options.append((label, str(d)))
        except Exception:
            options.append((d.name, str(d)))
    app.compare_select.options = options
    app.compare_btn.disabled = False
