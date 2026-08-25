"""Visualization and rendering helpers used by QuantUIApp."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, cast

import ipywidgets as widgets
from IPython.display import HTML, display

from quantui import theme as _theme
from quantui.app_builders import _ORB_PNG_INBOX_CLASS
from quantui.orbital_visualization import _png_capture_controls

logger = logging.getLogger(__name__)


@contextmanager
def _viz_render_event(app: Any, task: Any, backend: Any, **extras: Any):
    """Lifecycle telemetry context manager for one render-path execution.

    Emits ``viz_render_start`` on entry and ``viz_render_done`` /
    ``viz_render_error`` on exit, each with ``elapsed_ms``. Wraps the
    actual backend render work; the router decision itself is logged
    separately by ``app._resolve_backend`` via ``viz_route_decision``.

    Extra kwargs are appended as ``key=value`` pairs to the event body
    (e.g. ``mode=3`` for vib renders, ``idx=12`` for trajectory frames).
    All log writes are best-effort — failures never propagate.
    """
    from quantui import calc_log as _clog_evt

    t0 = time.perf_counter()
    pref = getattr(app, "_viz_backend_preference", "auto")
    extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
    base = f"task={task} pref={pref} backend={backend}"
    fields = f"{base} {extras_str}".strip()
    try:
        _clog_evt.log_event("viz_render_start", fields)
    except Exception:
        pass
    try:
        yield
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        try:
            _clog_evt.log_event(
                "viz_render_error",
                f"{fields} elapsed_ms={elapsed_ms} "
                f"err={type(exc).__name__}:{exc}"[:300],
            )
        except Exception:
            pass
        raise
    else:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        try:
            _clog_evt.log_event(
                "viz_render_done",
                f"{fields} elapsed_ms={elapsed_ms}",
            )
        except Exception:
            pass


def show_result_3d(
    app: Any,
    molecule: Any,
    extra_output: Any = None,
    *,
    render_html_fn: Any,
) -> None:
    """Render molecule 3D structure in result and optional extra output panels.

    Backend selection goes through ``app._resolve_backend(task)`` per-output:

    - ``result_viz_output`` uses ``VizTask.STRUCTURE_VIEW_RESULTS``.
    - ``extra_output == _analysis_mol_output`` uses ``ANALYSIS_STRUCTURE_VIEW``.
    - Any other extra_output uses ``STRUCTURE_VIEW_RESULTS`` as a safe default.

    ``render_html_fn`` must return self-contained HTML (e.g.
    ``visualization_py3dmol.render_molecule_html``); the HTML is routed through
    ``app._set_html_output`` so the viewer is replaced as a single atomic
    ``Output.outputs`` swap. This avoids the nested-Output + ``display(viz)``
    pattern that caused a trajectory regression and an Analysis-tab top
    viewer rendering blank with 🙁 on history replay.
    """
    if render_html_fn is None or molecule is None:
        return
    from quantui.viz_backend_router import VizTask as _VT

    is_analysis_output = extra_output is not None and extra_output is getattr(
        app, "_analysis_mol_output", None
    )

    # Results-tab viewer.
    if app.result_viz_output is not None:
        chosen = app._resolve_backend(_VT.STRUCTURE_VIEW_RESULTS)
        if chosen is not None:
            with _viz_render_event(
                app,
                task="structure_view_results",
                backend=str(chosen),
            ):
                html = render_html_fn(
                    molecule,
                    backend=str(chosen),
                    style=app._viz_style,
                    lighting=app._viz_lighting,
                    bgcolor=app._plotly_theme_colors()["scene_bgcolor"],
                )
                app._set_html_output(app.result_viz_output, html)

    # Optional second viewer (typically the Analysis tab).
    if extra_output is not None:
        task = (
            _VT.ANALYSIS_STRUCTURE_VIEW
            if is_analysis_output
            else _VT.STRUCTURE_VIEW_RESULTS
        )
        chosen = app._resolve_backend(task)
        if chosen is not None:
            task_label = (
                "analysis_structure_view"
                if is_analysis_output
                else "structure_view_results"
            )
            with _viz_render_event(app, task=task_label, backend=str(chosen)):
                html = render_html_fn(
                    molecule,
                    backend=str(chosen),
                    style=app._viz_style,
                    lighting=app._viz_lighting,
                    bgcolor=app._plotly_theme_colors()["scene_bgcolor"],
                )
                if is_analysis_output:
                    # M-MEASURE: wires click-to-measure into the freshly
                    # rendered HTML (py3Dmol only) and resets any stale picks
                    # from whatever was shown before. Also injects the
                    # Mulliken/dipole overlay bridge.
                    from quantui.app_measurement import finalize_analysis_html

                    html = finalize_analysis_html(app, html, chosen)
                app._set_html_output(extra_output, html)
            if is_analysis_output:
                app._update_analysis_backend_label(chosen)

    # Track the molecule currently shown in the Analysis-tab viewer so the
    # preference-change re-render path can find it. Set before the overlay
    # push so COM / atom colours have coordinates to read.
    if is_analysis_output:
        app._analysis_displayed_molecule = molecule
        if getattr(app, "_last_mulliken_charges", None):
            try:
                from quantui.populations_overlay import push_populations_overlay

                push_populations_overlay(app)
            except Exception:  # noqa: BLE001 — never block the render
                pass


def on_traj_expand(app: Any, change: dict[str, Any]) -> None:
    """Lazily generate trajectory animation when accordion first opens."""
    if change["new"] != 0:
        return
    result = app._pending_traj_result

    # Safety net: if _pending_traj_result was already consumed by a prior
    # auto-select render but traj_output is now empty, recover by rendering
    # from the cached _last_traj_result.
    # NOTE: traj_output is now a widgets.VBox — check `children` (not
    # `outputs`) for the populated-state heuristic.
    recovery_used = False
    if result is None:
        last = getattr(app, "_last_traj_result", None)
        children = getattr(app.traj_output, "children", ())
        if last is not None and len(children) == 0:
            result = last
            recovery_used = True
            try:
                from quantui import calc_log as _clog_recovery

                _clog_recovery.log_event(
                    "traj_render_recovery",
                    "children=0, rendering from _last_traj_result",
                )
            except Exception:
                pass

    try:
        from quantui import calc_log as _clog_te

        _clog_te.log_event(
            "traj_expand",
            f"pending={app._pending_traj_result is not None} "
            f"recovery={recovery_used} "
            f"children_n={len(getattr(app.traj_output, 'children', ()))}",
        )
    except Exception:
        pass
    if result is None:
        return
    app._pending_traj_result = None
    app._traj_render_token = int(getattr(app, "_traj_render_token", 0)) + 1
    render_token = app._traj_render_token

    # Placeholder: replace traj_output's children with a Loading message.
    app.traj_output.children = (
        widgets.HTML(
            value=(
                '<p style="color:#555;font-style:italic;padding:8px">'
                "Loading trajectory viewer…</p>"
            )
        ),
    )

    try:
        app._show_opt_trajectory(result, render_token=render_token)
    except Exception as exc:
        try:
            from quantui import calc_log as _clog_te2

            _clog_te2.log_event(
                "traj_expand_error",
                f"{type(exc).__name__}: {exc}"[:300],
            )
        except Exception:
            pass
        if render_token != int(getattr(app, "_traj_render_token", 0)):
            return
        app.traj_output.children = (
            widgets.HTML(
                value=(
                    f'<p style="color:#b91c1c;padding:8px">'
                    f"⚠ Trajectory rendering failed: {exc}</p>"
                )
            ),
        )


def show_opt_trajectory(
    app: Any,
    opt_result: Any,
    *,
    layout_fn: Any,
    render_token: int | None = None,
) -> None:
    """Build the trajectory viewer + energy chart in the trajectory panel.

    All optimization steps are loaded once into ONE py3Dmol viewer
    (``addModelsAsFrames``) and navigated client-side with ``setFrame`` via an
    in-HTML stepper (prev/next, play/pause, scrub slider, start↔final flip,
    per-step energy label). Because the viewer instance never changes, the
    camera (rotation/zoom) stays put across steps and there is no per-frame HTML
    rebuild — the previous carousel rebuilt a fresh viewer each frame, which
    reset the camera and flickered. py3Dmol-only per the viz routing policy; the
    energy-convergence chart and Export button are unchanged.
    """

    def _is_stale() -> bool:
        return render_token is not None and render_token != int(
            getattr(app, "_traj_render_token", 0)
        )

    # Support both OptimizationResult (.trajectory) and PESScanResult
    # (.coordinates_list).
    traj = getattr(opt_result, "trajectory", None) or getattr(
        opt_result, "coordinates_list", []
    )
    energies = opt_result.energies_hartree
    n = len(traj)
    if n < 2:
        app.traj_output.children = (
            widgets.HTML(
                value=(
                    '<p style="color:#666;padding:8px">'
                    "No trajectory data available (single-frame result).</p>"
                )
            ),
        )
        return

    hartree_to_kcal = 627.5094740631
    e0 = energies[0] if energies else 0.0
    rel_e = [(e - e0) * hartree_to_kcal for e in energies] if energies else []

    # --- Energy convergence chart ---
    has_plotly = False
    try:
        import plotly.graph_objects as go

        energy_fig = go.Figure(
            go.Scatter(
                x=list(range(n)),
                y=rel_e,
                mode="lines+markers",
                name="ΔE",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=6),
            )
        )
        energy_fig.update_layout(
            title="Energy Convergence",
            xaxis_title="Step",
            yaxis_title="ΔE (kcal/mol)",
            height=220,
            margin=dict(l=60, r=20, t=40, b=40),
        )
        has_plotly = True
    except ImportError:
        pass

    # --- Pre-build XYZ blocks (reused by the viewer and the export) ---
    # (charge was only needed by the retired plotlymol export path)
    xyzblocks = [
        f"{len(m.atoms)}\n{m.get_formula()}\n{m.to_xyz_string()}" for m in traj
    ]
    formula = traj[0].get_formula()
    try:
        bgcolor = app._plotly_theme_colors()["scene_bgcolor"]
    except Exception:
        bgcolor = "white"

    if _is_stale():
        return

    # --- Single-viewer trajectory stepper (all frames preloaded) ---
    # min_height, not height: build_trajectory_viewer_html returns a framed
    # fragment, and a fixed height would clip its bottom border off.
    viewer_output = widgets.Output(
        layout=layout_fn(
            min_height="420px", width="100%", max_width="500px", overflow="hidden"
        )
    )
    try:
        with _viz_render_event(app, task="trajectory", backend="py3dmol", n_frames=n):
            html = build_trajectory_viewer_html(
                xyzblocks,
                formula=formula,
                energies=list(energies) if energies else None,
                rel_e=rel_e or None,
                bgcolor=bgcolor,
            )
        app._set_html_output(viewer_output, html)
    except Exception as exc:  # noqa: BLE001 — surface inline, never crash the tab
        viewer_output.outputs = (
            {
                "output_type": "display_data",
                "data": {
                    "text/html": (
                        '<p style="color:#b91c1c;padding:8px">'
                        f"Trajectory viewer failed: {exc}</p>"
                    )
                },
                "metadata": {},
            },
        )

    # --- Export button (standalone HTML animation; plotlymol3d) ---
    export_btn = widgets.Button(
        description="Export Animation",
        icon="download",
        layout=layout_fn(width="160px"),
        tooltip="Generate a standalone HTML animation file (may take a minute)",
    )
    export_status = widgets.HTML()

    def _on_export(_btn) -> None:
        _btn.disabled = True
        export_status.value = (
            f'<span style="color:#555;font-style:italic">'
            f"Generating {n}-frame animation, please wait…</span>"
        )

        def _do_export() -> None:
            try:
                # py3Dmol, matching what is on screen (2026-08-04, same change
                # as VIB_EXPORT). This reuses build_trajectory_viewer_html —
                # the exact builder that produced the viewer above — so the
                # exported file has the same stepper, the same per-step energy
                # labels and the same geometry, rather than a second renderer's
                # interpretation of them. standalone_html embeds the vendored
                # 3Dmol.js so the file opens offline, with no CDN.
                from quantui.viz_assets import standalone_html

                fragment = build_trajectory_viewer_html(
                    xyzblocks,
                    formula=opt_result.formula,
                    energies=list(energies) if energies else None,
                    rel_e=rel_e or None,
                    bgcolor="white",  # exported file has no app theme around it
                )
                result_dir = getattr(app, "_last_result_dir", None)
                out_path = (
                    result_dir / "trajectory_animation.html"
                    if result_dir is not None
                    else Path.home() / f"{opt_result.formula}_trajectory.html"
                )
                out_path.write_text(
                    standalone_html(fragment, title=f"Geo Opt: {opt_result.formula}"),
                    encoding="utf-8",
                )
                app._queue_main_thread_callback(
                    setattr,
                    export_status,
                    "value",
                    (
                        f'<span style="color:#16a34a;font-size:12px">'
                        f"✓ Saved: {out_path}</span>"
                    ),
                )
            except Exception as exc:
                app._queue_main_thread_callback(
                    setattr,
                    export_status,
                    "value",
                    f'<span style="color:#b91c1c">Export failed: {exc}</span>',
                )
            finally:
                app._queue_main_thread_callback(setattr, _btn, "disabled", False)

        threading.Thread(target=_do_export, daemon=True).start()

    export_btn.on_click(_on_export)

    # --- Assemble: energy chart (HTML in an Output so RequireJS runs) +
    # viewer + export row, set atomically as traj_output's children. ---
    if _is_stale():
        return
    new_children: list[Any] = []
    if has_plotly and rel_e:
        import plotly.io as _pio_e

        energy_html = _pio_e.to_html(
            energy_fig,
            full_html=False,
            include_plotlyjs="require",
            config={"responsive": True},
        )
        energy_holder = widgets.Output()
        energy_holder.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/html": energy_html},
                "metadata": {},
            },
        )
        new_children.append(energy_holder)
    new_children.append(viewer_output)
    new_children.append(
        widgets.HBox(
            [export_btn, export_status],
            layout=layout_fn(align_items="center", margin="4px 0"),
        )
    )
    app.traj_output.children = tuple(new_children)

    try:
        from quantui import calc_log as _clog_sp

        _clog_sp.log_event("traj_show_panel", f"n={n} single_viewer=1")
    except Exception:
        pass


def traj_step_html(
    app: Any, step: int, traj: list[Any], energies: list[Any], rel_e: list[Any]
) -> str:
    """One-line info label for a trajectory step index."""
    n = len(traj)
    mol = traj[step]
    e_abs = f"{energies[step]:.8f} Ha" if energies and step < len(energies) else "—"
    delta = (
        f" &nbsp;·&nbsp; ΔE = {rel_e[step]:+.3f} kcal/mol"
        if rel_e and step < len(rel_e)
        else ""
    )
    return (
        f'<span style="font-size:12px;color:#666">'
        f"Step {step} / {n - 1} &nbsp;·&nbsp; {mol.get_formula()}"
        f" &nbsp;·&nbsp; E = {e_abs}{delta}</span>"
    )


def render_traj_frame(app: Any, molecule: Any, output_widget: Any) -> None:
    """Render one trajectory frame into output widget."""
    try:
        from quantui.visualization_py3dmol import visualize_molecule_plotlymol

        fig = visualize_molecule_plotlymol(
            molecule, mode="ball+stick", resolution=8, width=460, height=340
        )
        scene_bg = app._plotly_theme_colors()["scene_bgcolor"]
        fig.update_layout(paper_bgcolor="white", scene=dict(bgcolor=scene_bg))
        output_widget.clear_output()
        with output_widget:
            display(fig)
        return
    except Exception:  # noqa: BLE001 — MET.3: any PlotlyMol failure (missing
        # backend, or the RDKit valence error a transition metal raises) must
        # fall through to the py3Dmol renderer below, never crash the frame.
        pass

    # Fallback: py3Dmol
    try:
        from quantui.viz_assets import make_view

        xyz = (
            f"{len(molecule.atoms)}\n"
            f"{molecule.get_formula()}\n"
            f"{molecule.to_xyz_string()}"
        )
        view = make_view(width=460, height=340)
        view.addModel(xyz, "xyz")
        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
        view.setBackgroundColor("white")
        view.zoomTo()
        output_widget.clear_output()
        with output_widget:
            display(view)
    except Exception as exc:
        output_widget.clear_output()
        with output_widget:
            display(
                HTML(
                    f'<p style="color:#b91c1c;padding:8px">Frame render failed: {exc}</p>'
                )
            )


def build_vib_data_from_freq_result(app: Any, freq_result: Any, molecule: Any) -> Any:
    """Construct plotlymol3d VibrationalData from a frequency result."""
    try:
        import numpy as np
        from plotlymol3d import VibrationalData, VibrationalMode
    except ImportError:
        return None

    try:
        return app._build_vib_data_inner(
            freq_result, molecule, np, VibrationalData, VibrationalMode
        )
    except Exception as exc:
        try:
            from quantui import calc_log as _clog

            _clog.log_event("vib_data_error", f"{type(exc).__name__}: {exc}"[:300])
        except Exception:
            pass
        return None


def build_vib_data_inner(
    app: Any,
    freq_result: Any,
    molecule: Any,
    np: Any,
    VibrationalData: Any,
    VibrationalMode: Any,
) -> Any:
    """Internal constructor for VibrationalData with dependency injection."""
    displacements = getattr(freq_result, "displacements", None)
    if displacements is None:
        return None

    freqs = freq_result.frequencies_cm1
    intensities = freq_result.ir_intensities
    n_modes = len(freqs)

    coords = np.array(molecule.coordinates, dtype=float)

    # Map element symbols to atomic numbers using a common-elements table.
    z_map = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Br": 35,
        "I": 53,
    }
    atomic_numbers: List[int] = [z_map.get(sym, 0) for sym in molecule.atoms]

    modes = []
    for i in range(n_modes):
        freq = freqs[i]
        ir_inten = intensities[i] if i < len(intensities) else None
        displ = np.array(displacements[i], dtype=float)
        modes.append(
            VibrationalMode(
                mode_number=i + 1,
                frequency=float(freq),
                ir_intensity=ir_inten,
                displacement_vectors=displ,
                is_imaginary=freq < 0,
            )
        )

    return VibrationalData(
        coordinates=coords,
        atomic_numbers=atomic_numbers,
        modes=modes,
        source_file="quantui_freq_calc",
        program="pyscf",
    )


def show_vib_animation(app: Any, freq_result: Any, molecule: Any) -> bool:
    """Populate vibrational animation accordion after a Frequency result.

    Dropdown options are built from raw ``freq_result.frequencies_cm1`` so the
    panel populates regardless of plotlymol3d availability. The plotlymol3d
    `VibrationalData` wrapper is built optionally — required only when the
    plotlymol render path is selected; the py3Dmol render path reads
    displacements directly from ``freq_result``.
    """
    freqs = freq_result.frequencies_cm1
    if not freqs:
        return False

    # Optional plotlymol3d data — may be None if plotlymol3d isn't installed.
    # The py3Dmol render path doesn't need this; only the plotlymol path does.
    vib_data = app._build_vib_data_from_freq_result(freq_result, molecule)

    # Build dropdown options from raw freq_result; skip near-zero translation
    # / rotation modes. Mode numbers are 1-indexed positions in
    # frequencies_cm1.
    options = []
    for i, freq_val in enumerate(freqs, start=1):
        if abs(freq_val) < 10:
            continue
        label = (
            f"Mode {i}: {freq_val:.1f} cm⁻¹"
            if freq_val >= 0
            else f"Mode {i}: {freq_val:.1f} cm⁻¹ (imaginary, TS?)"
        )
        options.append((label, i))

    if not options:
        return False

    app._last_vib_data = vib_data  # may be None — plotlymol3d optional
    app._last_vib_molecule = molecule
    app._last_vib_freq_result = freq_result

    first_label, first_mode = options[0]

    # Decide the render path BEFORE assigning vib_mode_dd.value (which fires
    # on_vib_mode_changed): set the single-viewer flag first so that observer
    # takes the client-side-switch branch instead of spawning a redundant
    # legacy per-mode render. The preferred path is ONE persistent py3Dmol
    # viewer holding every mode, with client-side mode switching, so the camera
    # (rotation/zoom) is preserved across modes (matches pre-opt/trajectory).
    # Falls back to the legacy renderer when py3Dmol isn't selected or
    # displacements are unavailable (e.g. some history replays).
    use_single = _vib_single_viewer_supported(app, freq_result)
    app._vib_single_viewer_active = use_single

    app.vib_mode_dd.options = options
    app.vib_mode_dd.value = first_mode  # fires on_vib_mode_changed

    if use_single:
        if _render_vib_single_viewer(
            app, freq_result, molecule, first_mode, [m for _, m in options]
        ):
            return True
        app._vib_single_viewer_active = False  # build failed → legacy fallback

    # Cache-hit fast path: on history replay the cached HTML for the first
    # mode is on disk, so swap it in synchronously without a placeholder.
    # ``reset_camera=True`` clears any stale camera matrix from a previous
    # freq result so the first mode opens at default zoom-to-fit.
    if _try_vib_cache_hit_sync(app, first_mode, reset_camera=True):
        return True

    # Bump render token so any stale worker thread bails out before stomping
    # this fresh render's output.
    app._vib_render_token = int(getattr(app, "_vib_render_token", 0)) + 1
    token = app._vib_render_token
    _swap_vib_output(
        app,
        _VIB_CAMERA_RESET_JS + f'<p style="color:#555;font-style:italic;padding:8px">'
        f"⏳ Rendering vibrational animation ({first_label})…</p>",
    )
    threading.Thread(
        target=app._render_vib_mode,
        args=(vib_data, molecule, first_mode),
        kwargs={"render_token": token},
        daemon=True,
    ).start()

    return True


def show_ir_spectrum(app: Any, freq_result: Any) -> bool:
    """Populate IR Spectrum accordion after a Frequency result."""
    freqs = list(freq_result.frequencies_cm1 or [])
    ints = list(getattr(freq_result, "ir_intensities", None) or [])
    if not freqs:
        return False

    app._ir_intensities_real = bool(ints)
    if not ints:
        ints = [1.0] * len(freqs)
    app._ir_accordion.set_title(
        0,
        (
            "IR Spectrum"
            if app._ir_intensities_real
            else "IR Spectrum (positions only — intensities unavailable)"
        ),
    )

    app._last_ir_freqs = freqs
    app._last_ir_ints = ints

    app._update_ir_figure("Stick", 20.0)

    # _show_ir_spectrum may run from _do_run background thread.
    app._queue_main_thread_callback(app._wire_ir_controls)

    return True


def wire_ir_controls(app: Any) -> None:
    """Rebind IR controls and reset defaults on the main thread."""
    # Observers are wired once in QuantUIApp._wire_callbacks. Avoid unobserve_all()
    # here because it can remove unrelated trait observers in some frontends.
    app._ir_mode_toggle.value = "Stick"
    app._ir_fwhm_slider.value = 20.0
    app._ir_fwhm_slider.layout.display = "none"


def on_ir_mode_changed(app: Any, change: dict[str, Any]) -> None:
    """Handle Stick/Broadened mode changes for IR panel."""
    mode = change["new"]
    try:
        import quantui.calc_log as _calc_log

        _calc_log.log_event(
            "ir_mode_change",
            mode,
            mode=mode,
            session_id=app._session_id,
        )
    except Exception:
        pass
    app._ir_fwhm_slider.layout.display = "" if mode == "Broadened" else "none"
    app._update_ir_figure(mode, app._ir_fwhm_slider.value)


def on_ir_fwhm_changed(app: Any, change: dict[str, Any]) -> None:
    """Re-render broadened IR trace when line width slider changes."""
    if app._ir_mode_toggle.value == "Broadened":
        app._update_ir_figure("Broadened", change["new"])


def update_ir_figure(app: Any, mode: str, fwhm: float) -> None:
    """Re-render IR spectrum chart for mode and FWHM settings."""
    try:
        import plotly.io as _pio

        from quantui.ir_plot import plot_ir_spectrum

        y_title = (
            "IR Intensity (km/mol)"
            if getattr(app, "_ir_intensities_real", True)
            else "Relative intensity (a.u.)"
        )
        fig = plot_ir_spectrum(
            app._last_ir_freqs,
            app._last_ir_ints,
            mode=mode.lower(),
            fwhm=fwhm,
            yaxis_title=y_title,
        )
        app._apply_plotly_theme(fig)
        app._last_ir_fig = fig
        app._set_html_output(
            app._ir_fig,
            _pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception as exc:
        app._last_ir_fig = None
        try:
            from quantui import calc_log as _clog

            _clog.log_event("ir_fig_error", f"{type(exc).__name__}: {exc}"[:300])
        except Exception:
            pass


def show_uv_vis_spectrum(
    app: Any,
    energies_ev: List[float],
    oscillator_strengths: List[float],
    wavelengths_nm: List[float],
) -> bool:
    """Populate UV-Vis spectrum data and render the default stick plot."""
    wl = list(wavelengths_nm or [])
    if not wl:
        wl = [1240.0 / e for e in energies_ev if e and e > 0]

    peaks: list[tuple[float, float]] = []
    for x0, amp in zip(wl, oscillator_strengths):
        try:
            x_val = float(x0)
            a_val = float(amp)
        except Exception:
            continue
        if x_val <= 0:
            continue
        peaks.append((x_val, max(a_val, 0.0)))

    if not peaks:
        return False

    peaks.sort(key=lambda p: p[0])
    app._last_uv_wavelengths_nm = [p[0] for p in peaks]
    app._last_uv_oscillator_strengths = [p[1] for p in peaks]

    app._update_uv_vis_figure("Stick", 20.0)

    # _show_uv_vis_spectrum may run from _do_run background thread.
    app._queue_main_thread_callback(app._wire_uv_controls)
    return True


def wire_uv_controls(app: Any) -> None:
    """Rebind UV-Vis controls and reset defaults on the main thread."""
    # Observers are wired once in QuantUIApp._wire_callbacks. Avoid unobserve_all()
    # here because it can remove unrelated trait observers in some frontends.
    app._uv_mode_toggle.value = "Stick"
    app._uv_fwhm_slider.value = 20.0
    app._uv_fwhm_slider.layout.display = "none"
    wl = list(getattr(app, "_last_uv_wavelengths_nm", []) or [])
    if wl:
        gamma = max(float(app._uv_fwhm_slider.value), 1.0) / 2.0
        pad = max(80.0, 3.0 * gamma)
        app._uv_xmin_input.value = round(max(100.0, min(wl) - pad), 1)
        app._uv_xmax_input.value = round(max(wl) + pad, 1)


def on_uv_range_changed(app: Any, _change: dict[str, Any] | None = None) -> None:
    """Re-render UV-Vis when λ min/max inputs change."""
    if getattr(app, "_last_uv_wavelengths_nm", None):
        app._update_uv_vis_figure(
            app._uv_mode_toggle.value,
            app._uv_fwhm_slider.value,
        )


def on_uv_mode_changed(app: Any, change: dict[str, Any]) -> None:
    """Handle Stick/Broadened mode changes for UV-Vis panel."""
    mode = change["new"]
    app._uv_fwhm_slider.layout.display = "" if mode == "Broadened" else "none"
    app._update_uv_vis_figure(mode, app._uv_fwhm_slider.value)


def on_uv_fwhm_changed(app: Any, change: dict[str, Any]) -> None:
    """Re-render broadened UV-Vis trace when line width slider changes."""
    if app._uv_mode_toggle.value == "Broadened":
        app._update_uv_vis_figure("Broadened", change["new"])


def update_uv_vis_figure(app: Any, mode: str, fwhm: float) -> None:
    """Re-render UV-Vis spectrum chart for mode and FWHM settings."""
    wl = list(getattr(app, "_last_uv_wavelengths_nm", []) or [])
    osc = list(getattr(app, "_last_uv_oscillator_strengths", []) or [])
    if not wl or not osc:
        return

    try:
        import numpy as _np
        import plotly.graph_objects as _go
        import plotly.io as _pio

        mode_name = str(mode or "Stick")
        mode_norm = mode_name.strip().lower()
        fig = _go.Figure()

        # Use one stable x-range across modes so toggling Stick/Broadened
        # doesn't visibly shift the axis. The Broadened wings need ~3*gamma
        # of headroom to show the full Lorentzian tail; padding by the same
        # amount in Stick keeps the layout identical.
        gamma = max(float(fwhm), 1.0) / 2.0
        pad = max(80.0, 3.0 * gamma)
        x_min_default = max(100.0, min(wl) - pad)
        x_max_default = max(wl) + pad
        xmin_w = getattr(app, "_uv_xmin_input", None)
        xmax_w = getattr(app, "_uv_xmax_input", None)
        x_min = float(xmin_w.value if xmin_w is not None else x_min_default)
        x_max = float(xmax_w.value if xmax_w is not None else x_max_default)
        if x_min >= x_max:
            x_min, x_max = x_min_default, x_max_default

        if mode_norm == "broadened":
            n_points = max(600, int((x_max - x_min) * 2.0))
            x_grid = _np.linspace(x_min, x_max, n_points)
            y_grid = _np.zeros_like(x_grid)
            for x0, amp in zip(wl, osc):
                y_grid += amp * (gamma**2 / ((x_grid - x0) ** 2 + gamma**2))
            fig.add_trace(
                _go.Scatter(
                    x=x_grid.tolist(),
                    y=y_grid.tolist(),
                    mode="lines",
                    line=dict(color="#2563eb", width=2),
                    name="Broadened",
                )
            )
        else:
            stick_x: list[float | None] = []
            stick_y: list[float | None] = []
            for x0, amp in zip(wl, osc):
                stick_x.extend([x0, x0, None])
                stick_y.extend([0.0, amp, None])
            fig.add_trace(
                _go.Scatter(
                    x=stick_x,
                    y=stick_y,
                    mode="lines",
                    line=dict(color="#2563eb", width=2),
                    name="Stick",
                )
            )
            fig.add_trace(
                _go.Scatter(
                    x=wl,
                    y=osc,
                    mode="markers",
                    marker=dict(color="#1d4ed8", size=6),
                    showlegend=False,
                    hovertemplate=(
                        "Wavelength: %{x:.2f} nm"
                        "<br>Oscillator strength: %{y:.3f}<extra></extra>"
                    ),
                )
            )

        tc = app._plotly_theme_colors()
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Oscillator strength",
            height=320,
            margin=dict(l=60, r=20, t=30, b=50),
            showlegend=False,
            plot_bgcolor=tc["plot_bgcolor"],
            paper_bgcolor=tc["paper_bgcolor"],
            font=dict(color=tc["font_color"]),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor=tc["grid_color"],
            zeroline=False,
            range=[x_min, x_max],
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=tc["grid_color"],
            rangemode="tozero",
        )

        app._apply_plotly_theme(fig)
        app._last_uv_fig = fig
        app._set_html_output(
            app._tddft_fig,
            _pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception as exc:
        app._last_uv_fig = None
        try:
            from quantui import calc_log as _clog

            _clog.log_event("uv_fig_error", f"{type(exc).__name__}: {exc}"[:300])
        except Exception:
            pass


def show_nmr_spectrum(
    app: Any,
    *,
    atom_symbols: List[str],
    shielding_iso_ppm: List[float],
    h_shifts: List[tuple[int, float]],
    c_shifts: List[tuple[int, float]],
    reference: str = "TMS",
) -> bool:
    """Populate NMR accordion with a stick spectrum and shielding tables."""
    if not atom_symbols:
        return False

    app._last_nmr_atom_symbols = list(atom_symbols)
    app._last_nmr_shielding = list(shielding_iso_ppm)
    app._last_nmr_h_shifts = list(h_shifts)
    app._last_nmr_c_shifts = list(c_shifts)
    app._last_nmr_reference = reference

    options: List[str] = []
    if h_shifts:
        options.append("¹H")
    if c_shifts:
        options.append("¹³C")
    if not options:
        # Shielding-only fallback: table without a stick plot.
        app._last_nmr_fig = None
        app._nmr_summary.value = _nmr_summary_html(
            atom_symbols, shielding_iso_ppm, h_shifts, c_shifts, reference
        )
        app._set_html_output(app._nmr_fig, "")
        return True

    app._nmr_nucleus_toggle.options = options
    app._nmr_nucleus_toggle.value = options[0]
    app._nmr_summary.value = _nmr_summary_html(
        atom_symbols, shielding_iso_ppm, h_shifts, c_shifts, reference
    )
    app._update_nmr_figure(app._nmr_nucleus_toggle.value)
    app._queue_main_thread_callback(app._wire_nmr_controls)
    return True


def _nmr_summary_html(
    atom_symbols: List[str],
    shielding: List[float],
    h_shifts: List[tuple[int, float]],
    c_shifts: List[tuple[int, float]],
    reference: str,
) -> str:
    from . import theme as _theme

    def _shift_table(label: str, shifts: list, sym: str) -> str:
        if not shifts:
            return ""
        rows = "".join(
            f'<tr><td style="padding:2px 14px 2px 0;color:{_theme.TEXT_SECONDARY}">{sym}-{n}</td>'
            f'<td style="color:{_theme.TEXT_HEADING}">{d:.2f} ppm</td></tr>'
            for n, (_i, d) in enumerate(sorted(shifts, key=lambda x: x[0]), 1)
        )
        return (
            f'<tr><td colspan="2" style="padding:8px 0 2px;font-weight:600">'
            f"{label} shifts (vs. {reference}):</td></tr>"
            f'<tr><th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 14px 2px 0">Atom</th>'
            f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">δ (ppm)</th></tr>'
            + rows
        )

    shielding_rows = "".join(
        f'<tr><td style="padding:2px 10px 2px 0;color:{_theme.TEXT_SECONDARY}">{sym}{i + 1}</td>'
        f'<td style="color:{_theme.TEXT_HEADING}">{s:.2f}</td></tr>'
        for i, (sym, s) in enumerate(zip(atom_symbols, shielding))
    )
    return (
        f'<div style="font-size:13px;margin-top:10px">'
        f'<table style="border-collapse:collapse;margin-bottom:8px">'
        f'<tr><th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px;padding:2px 10px 2px 0">Atom</th>'
        f'<th style="text-align:left;color:{_theme.TEXT_SECONDARY};font-size:12px">σ (ppm)</th></tr>'
        f"{shielding_rows}</table>"
        f'<table style="border-collapse:collapse">'
        f"{_shift_table('¹H', h_shifts, 'H')}"
        f"{_shift_table('¹³C', c_shifts, 'C')}"
        f"</table></div>"
    )


def wire_nmr_controls(app: Any) -> None:
    """Reset NMR nucleus toggle default on the main thread after a new result."""
    options = list(
        getattr(app, "_nmr_nucleus_toggle", None)
        and app._nmr_nucleus_toggle.options
        or []
    )
    if options:
        app._nmr_nucleus_toggle.value = options[0]


def on_nmr_nucleus_changed(app: Any, change: dict[str, Any]) -> None:
    """Switch stick plot between ¹H and ¹³C peaks."""
    app._update_nmr_figure(change["new"])


def update_nmr_figure(app: Any, nucleus: str) -> None:
    """Re-render the NMR stick spectrum for the selected nucleus."""
    symbols = list(getattr(app, "_last_nmr_atom_symbols", None) or [])
    if nucleus == "¹³C":
        shifts = list(getattr(app, "_last_nmr_c_shifts", None) or [])
    else:
        shifts = list(getattr(app, "_last_nmr_h_shifts", None) or [])
    if not symbols or not shifts:
        app._last_nmr_fig = None
        app._set_html_output(app._nmr_fig, "")
        return
    try:
        import plotly.io as _pio

        from quantui.nmr_plot import plot_nmr_spectrum

        fig = plot_nmr_spectrum(
            shifts,
            symbols,
            nucleus_label=nucleus,
        )
        app._apply_plotly_theme(fig)
        app._last_nmr_fig = fig
        app._set_html_output(
            app._nmr_fig,
            _pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception as exc:
        app._last_nmr_fig = None
        try:
            from quantui import calc_log as _clog

            _clog.log_event("nmr_fig_error", f"{type(exc).__name__}: {exc}"[:300])
        except Exception:
            pass


def show_orbital_diagram(app: Any, result: Any) -> bool:
    """Build and reveal interactive orbital diagram accordion."""
    mo_energy = getattr(result, "mo_energy_hartree", None)
    mo_occ = getattr(result, "mo_occ", None)
    if mo_energy is None or mo_occ is None:
        return False

    try:
        from quantui.orbital_visualization import orbital_info_from_arrays

        info = orbital_info_from_arrays(mo_energy, mo_occ, formula=result.formula)
    except Exception:
        return False

    app._last_orb_info = info
    app._last_orb_mo_coeff = getattr(result, "mo_coeff", None)
    app._last_orb_mo_occ = mo_occ
    app._last_orb_mol_atom = getattr(result, "pyscf_mol_atom", None)
    app._last_orb_mol_basis = getattr(result, "pyscf_mol_basis", None)

    plotly_rendered = False
    try:
        import plotly.io as _pio

        from quantui.orbital_visualization import plot_orbital_diagram_plotly

        fig = plot_orbital_diagram_plotly(info, max_orbitals=app._orb_n_orb_input.value)
        yr = fig.layout.yaxis.range
        if yr is not None:
            app._orb_ymin_input.value = round(float(yr[0]), 2)
            app._orb_ymax_input.value = round(float(yr[1]), 2)
        app._apply_plotly_theme(fig)
        app._last_orb_fig = fig
        html_str = _pio.to_html(
            fig,
            include_plotlyjs="require",
            full_html=False,
            config={"responsive": True},
        )
        app._set_html_output(app._orb_diagram_html, html_str)
        plotly_rendered = True
    except Exception:
        pass

    if not plotly_rendered:
        app._last_orb_fig = None
        import base64
        import io as _io

        try:
            from matplotlib.backends.backend_agg import (
                FigureCanvasAgg as _AggCanvas,
            )

            from quantui.orbital_visualization import plot_orbital_diagram

            mpl_fig = plot_orbital_diagram(info)
            _AggCanvas(mpl_fig)
            buf = _io.BytesIO()
            mpl_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            app._set_html_output(
                app._orb_diagram_html,
                (
                    f'<img src="data:image/png;base64,{img_b64}" '
                    'style="max-width:100%;height:auto" />'
                ),
            )
        except Exception:
            pass

    if (
        app._last_orb_mo_coeff is not None
        and app._last_orb_mol_atom is not None
        and app._last_orb_mol_basis is not None
    ):
        app._orb_toggle.value = "HOMO"
        app._orb_iso_controls.layout.display = ""
        app._iso_generate_btn.disabled = False
        # (2) Show the molecule immediately, so the panel is never empty and the
        # first isosurface fades in over an existing viewer instead of appearing
        # in a collapsed one. It also lets the user orient the structure BEFORE
        # generating — the camera carries across, same scene key.
        _show_iso_placeholder(app)
    else:
        app._orb_iso_controls.layout.display = "none"
        app._iso_generate_btn.disabled = True

    return True


def on_iso_generate(app: Any, btn: Any) -> None:
    """Generate orbital isosurface for currently selected orbital."""
    orbital_label = app._orb_toggle.value
    # "By index" mode renders an arbitrary 0-based MO index. Encode it
    # into the label as "MO <n>"; render_orbital_isosurface parses it back.
    if orbital_label == "By index":
        orbital_label = f"MO {int(app._orb_index_input.value)}"
    app._iso_render_token = int(getattr(app, "_iso_render_token", 0)) + 1
    render_token = app._iso_render_token
    btn.disabled = True
    btn.description = "Generating…"
    # Reveal the inline spinner + light the toolbar activity indicator
    # so the (slow) cube generation reads as busy, not hung.
    _spinner = getattr(app, "_iso_spinner", None)
    if _spinner is not None:
        _spinner.layout.display = ""
    try:
        app._activity_begin("Generating orbital isosurface…", kind="compute")
    except Exception:
        pass
    try:
        from quantui import calc_log as _clog

        _clog.log_event("iso_render_start", orbital_label)
    except Exception:
        pass
    _show_cancel(app, True)
    # (7) Do NOT swap the output here. Replacing a rendered viewer with a text
    # placeholder collapsed the panel from ~620px to nothing, so the accordion
    # jumped and the page scrolled — then jumped back when the new surface
    # arrived. Dimming the existing viewer in place keeps the layout identical.
    # With no viewer yet (first generate) there is nothing to dim, so fall back
    # to the message; an empty panel cannot collapse further.
    # A viewer is always present now — the molecule placeholder fills the panel
    # before the first cube exists — so this can always dim rather than swap.
    iso_bridge_busy(app, True)

    done = threading.Event()
    # Balance the single _activity_begin above exactly once, across
    # both the normal-completion and timeout paths (idempotent).
    _finished = threading.Event()

    def _finish_activity() -> None:
        if _finished.is_set():
            return
        _finished.set()
        try:
            app._activity_end(kind="compute")
        except Exception:
            pass
        # Only hide the spinner if no newer generation superseded this one
        # (a newer render still wants the spinner visible).
        if render_token == int(getattr(app, "_iso_render_token", 0)):
            _sp = getattr(app, "_iso_spinner", None)
            if _sp is not None:
                _sp.layout.display = "none"

    def _reset_button() -> None:
        _finish_activity()
        if render_token != int(getattr(app, "_iso_render_token", 0)):
            return
        btn.disabled = False
        btn.description = "Generate Isosurface"
        _show_cancel(app, False)
        iso_bridge_busy(app, False)

    def _run() -> None:
        try:
            app._render_orbital_isosurface(orbital_label, render_token=render_token)
        finally:
            done.set()
            app._queue_main_thread_callback(_reset_button)

    def _watchdog() -> None:
        if done.wait(timeout=180):
            return

        def _show_timeout() -> None:
            _finish_activity()
            if render_token != int(getattr(app, "_iso_render_token", 0)):
                return
            try:
                from quantui import calc_log as _clog

                _clog.log_event("iso_render_timeout", orbital_label)
            except Exception:
                pass
            btn.disabled = False
            btn.description = "Generate Isosurface"
            app._set_html_output(
                app._orb_iso_output,
                '<p style="color:#b91c1c;padding:8px">'
                "⚠ Orbital isosurface timed out after 180 s. "
                "Try a smaller basis set or a smaller molecule.</p>",
            )

        app._queue_main_thread_callback(_show_timeout)

    threading.Thread(target=_run, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()


def on_orb_range_changed(app: Any, _change: Any = None) -> None:
    """Live-update orbital diagram for axis limits or orbital count changes."""
    info = getattr(app, "_last_orb_info", None)
    if info is None:
        return
    ymin = app._orb_ymin_input.value
    ymax = app._orb_ymax_input.value
    if ymin >= ymax:
        return
    try:
        import plotly.io as _pio

        from quantui.orbital_visualization import plot_orbital_diagram_plotly

        fig = plot_orbital_diagram_plotly(
            info,
            max_orbitals=app._orb_n_orb_input.value,
            yrange=(ymin, ymax),
        )
        app._apply_plotly_theme(fig)
        app._last_orb_fig = fig
        app._set_html_output(
            app._orb_diagram_html,
            _pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception:
        app._last_orb_fig = None
        pass


def render_orbital_isosurface(
    app: Any, orbital_label: str, render_token: int | None = None
) -> None:
    """Generate cube file and render orbital isosurface (Linux/WSL only)."""
    import re as _re
    from datetime import datetime as _dt

    def _is_stale() -> bool:
        return render_token is not None and render_token != int(
            getattr(app, "_iso_render_token", 0)
        )

    orb_info = getattr(app, "_last_orb_info", None)
    if orb_info is None:
        return

    n_occ = orb_info.n_occupied
    n_total = len(orb_info.mo_energies_ev)
    idx_map = {
        "HOMO-1": n_occ - 2,
        "HOMO": n_occ - 1,
        "LUMO": n_occ,
        "LUMO+1": n_occ + 1,
    }
    orb_idx = idx_map.get(orbital_label)
    # "MO <n>" labels carry an explicit 0-based index from By-index mode.
    if orb_idx is None:
        _m = _re.match(r"MO\s+(\d+)$", orbital_label)
        if _m:
            orb_idx = int(_m.group(1))
    if orb_idx is None or orb_idx < 0 or orb_idx >= n_total:
        # Out-of-range (now user-reachable via free index entry) — surface it
        # instead of silently leaving the "Generating…" placeholder in place.
        def _show_range_err() -> None:
            if _is_stale():
                return
            app._set_html_output(
                app._orb_iso_output,
                '<p style="color:#b91c1c;padding:8px">'
                f"⚠ Orbital index out of range. This calculation has "
                f"{n_total} molecular orbitals (valid indices 0–"
                f"{n_total - 1}; HOMO = {n_occ - 1}).</p>",
            )

        app._queue_main_thread_callback(_show_range_err)
        return

    mo_coeff = getattr(app, "_last_orb_mo_coeff", None)
    mol_atom = getattr(app, "_last_orb_mol_atom", None)
    mol_basis = getattr(app, "_last_orb_mol_basis", None)
    mo_occ_for_charge = getattr(app, "_last_orb_mo_occ", None)
    if mo_coeff is None or mol_atom is None or mol_basis is None:
        return

    try:
        import plotly.io as _pio

        from quantui.orbital_visualization import (
            generate_cube_from_arrays,
            infer_charge_and_spin,
            plot_cube_isosurface,
            render_orbital_isosurface_py3dmol,
        )
        from quantui.viz_backend_router import VizTask as _VT

        result_dir = getattr(app, "_last_result_dir", None)
        if not isinstance(result_dir, Path):
            try:
                result_dir = app._get_results_dir()
            except Exception:
                result_dir = Path.cwd()

        cube_dir = Path(result_dir) / "isosurfaces"
        cube_dir.mkdir(parents=True, exist_ok=True)

        formula = str(getattr(orb_info, "formula", "") or "molecule")
        safe_formula = _re.sub(r"[^A-Za-z0-9_.-]+", "_", formula).strip("._")
        if not safe_formula:
            safe_formula = "molecule"
        safe_orb = _re.sub(r"[^A-Za-z0-9_.-]+", "_", orbital_label).strip("._")
        if not safe_orb:
            safe_orb = "orbital"
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cube_path = cube_dir / f"{safe_formula}_{safe_orb}_{ts}.cube"

        # Charge/spin aren't carried on the app's orbital-state attributes —
        # infer them from the MO occupations so charged/open-shell molecules
        # (H3O+, OH-, radicals, ...) don't fail to build in PySCF.
        _charge, _spin = infer_charge_and_spin(mol_atom, mo_occ_for_charge)

        # ORBX.2: the user-chosen cubegen grid. Read at generate time rather
        # than cached, so changing the dropdown affects the next Generate
        # without any extra wiring. Falls back to the default if the widget is
        # absent (older layouts, tests constructing a partial app).
        from quantui.orbital_visualization import (
            DEFAULT_ISO_RESOLUTION,
            ISO_RESOLUTION_PRESETS,
            max_render_points,
        )

        _res_key = getattr(
            getattr(app, "_iso_resolution_dd", None), "value", DEFAULT_ISO_RESOLUTION
        )
        _grid = ISO_RESOLUTION_PRESETS.get(
            _res_key, ISO_RESOLUTION_PRESETS[DEFAULT_ISO_RESOLUTION]
        )
        # M-EXPORT2 EXP2.4 / M-ORBEXPORT ORBX.4: best-effort provenance, not a
        # re-verified guarantee — the live method dropdown, not necessarily
        # what actually produced the stored mo_coeff (e.g. after a History
        # replay of a differently-computed result).
        _method_for_provenance = str(
            getattr(getattr(app, "method_dd", None), "value", "") or ""
        )
        generate_cube_from_arrays(
            mol_atom,
            mol_basis,
            mo_coeff,
            orb_idx,
            cube_path,
            nx=_grid,
            ny=_grid,
            nz=_grid,
            charge=_charge,
            spin=_spin,
            method=_method_for_provenance,
        )
        scene_bgcolor = app._plotly_theme_colors()["scene_bgcolor"]

        # Route the render: py3Dmol does native, full-resolution in-browser
        # isosurfacing (primary); the Plotly path is the fallback (downsampled).
        # Both consume the same full-resolution cube on disk. Plotly is the
        # universal fallback whenever py3Dmol is not the chosen backend.
        chosen = app._resolve_backend(_VT.ORBITAL_ISOSURFACE)
        use_py3dmol = str(chosen) == "py3dmol"
        backend_label = "py3dmol" if use_py3dmol else "plotlymol"

        with _viz_render_event(app, task=_VT.ORBITAL_ISOSURFACE, backend=backend_label):
            if use_py3dmol:
                # Same options the sliders and the theme path use, so a fresh
                # render cannot disagree with a redraw of the same orbital.
                html_str = render_orbital_isosurface_py3dmol(
                    cube_path, **iso_render_options(app)
                )
            else:
                is_dark = app.theme_btn.value == "Dark"
                axis_color = "#dbeafe" if is_dark else "#1f2937"
                bond_color = "#cbd5e1" if is_dark else "#4b5563"
                title_color = app._plotly_theme_colors()["font_color"]
                fig = plot_cube_isosurface(
                    cube_path,
                    # The second knob. Without this the Plotly fallback would
                    # stride a finer grid straight back down to the 60³ cap and
                    # the extra wait would buy nothing. py3Dmol above needs no
                    # equivalent — it isosurfaces at full resolution in-browser.
                    max_points=max_render_points(_grid),
                    title=f"{orbital_label} Isosurface",
                    show_molecule=True,
                    show_grid=False,
                    scene_bgcolor=scene_bgcolor,
                    axis_color=axis_color,
                    title_color=title_color,
                    bond_color=bond_color,
                )
                html_str = _pio.to_html(
                    fig,
                    include_plotlyjs="require",
                    full_html=False,
                    config={"responsive": True},
                )
    except Exception as exc:
        if _is_stale():
            return
        err_msg = f"{type(exc).__name__}: {exc}"
        try:
            from quantui import calc_log as _clog

            _clog.log_event(
                "iso_render_error",
                f"{orbital_label}: {err_msg}"[:300],
            )
        except Exception:
            pass

        def _show_err(msg: str = err_msg) -> None:
            app._set_html_output(
                app._orb_iso_output,
                f'<p style="color:#b91c1c;padding:8px">'
                f"⚠ Orbital isosurface failed: {msg}</p>",
            )

        app._queue_main_thread_callback(_show_err)
        return
    if _is_stale():
        return
    # Track the last-generated cube + its orbital
    # label so the "Export cube" button can copy it to the top-level
    # result dir with a friendly name without re-deriving the path.
    app._last_cube_path = cube_path
    # The enclosed-density readout depends on the cube that was just written,
    # so it can only be filled in now — not when the slider was last moved.
    update_iso_enclosed_label(app)
    app._last_cube_orbital = orbital_label
    try:
        from quantui import calc_log as _clog

        _clog.log_event(
            "iso_cube_saved",
            cube_path.name,
            cube_path=str(cube_path),
            orbital=orbital_label,
            session_id=app._session_id,
        )
        _clog.log_event("iso_render_done", orbital_label)
    except Exception:
        pass

    app._queue_main_thread_callback(
        app._set_html_output,
        app._orb_iso_output,
        html_str,
    )

    # Now that ``_last_cube_path`` is populated, the
    # "Export cube" button has something to copy. Enable it on the main
    # thread alongside the iso render swap.
    def _enable_cube_btn() -> None:
        try:
            app._iso_export_cube_btn.disabled = False
        except Exception:
            pass

    app._queue_main_thread_callback(_enable_cube_btn)


def _swap_vib_output(app: Any, html_str: str) -> None:
    """Atomically replace ``app.vib_output``'s content with one HTML payload.

    Single widget-state assignment → single browser update message → no
    transient empty state → no layout reflow → no page-scroll jump on
    every mode switch. Matches the atomic-swap pattern proven for
    ``frame_out`` in the trajectory carousel.
    """
    app.vib_output.outputs = (
        {
            "output_type": "display_data",
            "data": {"text/html": html_str},
            "metadata": {},
        },
    )


def _vib_err(app: Any, msg: str) -> None:
    """Show an error message in the vibrational animation output panel."""
    _swap_vib_output(app, f'<p style="color:#b91c1c;padding:8px">⚠ {msg}</p>')


# JS snippet appended to every py3Dmol vib HTML payload. Patches
# ``$3Dmol.createViewer`` once per page so each new viewer:
#   1. Hijacks its own ``zoomTo`` — if ``window._quantuiVibCamera`` is set
#      (from a previous mode's pan/rotate state) apply it via ``setView``
#      instead of recomputing the default fit. Falls back to the original
#      zoomTo when no camera is saved.
#   2. Starts a periodic interval that writes ``viewer.getView()`` into
#      ``window._quantuiVibCamera``, so the user's interactive pan/zoom
#      survives mode switches.
# The script is idempotent (``_quantuiVibCameraHookInstalled`` guard), so
# it can ship inside every cached HTML blob without doubling the hook.
_VIB_CAMERA_PERSISTENCE_JS = """
<script>
(function() {
    if (window._quantuiVibCameraHookInstalled) return;
    function install() {
        if (!window.$3Dmol || !window.$3Dmol.createViewer) {
            setTimeout(install, 50);
            return;
        }
        if (window._quantuiVibCameraHookInstalled) return;
        window._quantuiVibCameraHookInstalled = true;
        var origCreate = window.$3Dmol.createViewer;
        window.$3Dmol.createViewer = function(element, config) {
            var v = origCreate.call(this, element, config);
            try {
                var origZoomTo = v.zoomTo ? v.zoomTo.bind(v) : null;
                v.zoomTo = function() {
                    if (window._quantuiVibCamera) {
                        try { return v.setView(window._quantuiVibCamera); }
                        catch (e) {}
                    }
                    if (origZoomTo) return origZoomTo.apply(this, arguments);
                };
                var iv = setInterval(function() {
                    try {
                        var el = (element && (element[0] || element));
                        if (el && !document.body.contains(el)) {
                            clearInterval(iv);
                            return;
                        }
                        var view = v.getView();
                        if (view) window._quantuiVibCamera = view;
                    } catch (e) {}
                }, 350);
            } catch (e) {}
            return v;
        };
    }
    install();
})();
</script>
"""

# Tiny one-shot reset injected when a NEW freq result begins, so the
# first mode of a new molecule opens at default zoom-to-fit instead of a
# stale camera from a previous molecule. Mode switches WITHIN the same
# freq result do not reset.
_VIB_CAMERA_RESET_JS = "<script>window._quantuiVibCamera = null;</script>"


def _ensure_vib_camera_hook(html: str) -> str:
    """Prepend ``_VIB_CAMERA_PERSISTENCE_JS`` to ``html`` only when not
    already present. Lets us serve both new cache entries (which embed
    the hook) and old cache entries (written before this feature) with
    a single code path — avoids redundant script tags and keeps cache
    HTML byte-stable across miss/hit cycles."""
    if "_quantuiVibCameraHookInstalled" in html:
        return html
    return _VIB_CAMERA_PERSISTENCE_JS + html


def _is_vib_stale(app: Any, render_token: int | None) -> bool:
    """True when a newer vib render has started; used to bail out of an
    older background render thread before it stomps the newer one's
    output. Returns False when no token was supplied (call-site opt-out)."""
    if render_token is None:
        return False
    return render_token != int(getattr(app, "_vib_render_token", 0))


def _try_vib_cache_hit_sync(
    app: Any, mode_number: int, *, reset_camera: bool = False
) -> bool:
    """Synchronous cache lookup + swap. Returns True iff a cached HTML blob
    matching the current render params was found and injected directly
    into ``app.vib_output``.

    Why: without this, every mode switch sets a "Rendering…" placeholder,
    spawns a thread, the thread checks the cache, then writes the cached
    HTML — visible as a brief placeholder flash even when the result is
    on disk. Doing the cache check on the main thread before the
    placeholder avoids the flash entirely for cache hits.

    When ``reset_camera`` is True a tiny inline script clears
    ``window._quantuiVibCamera`` before the cached HTML runs, so the
    first mode of a new molecule opens at default zoom rather than at a
    stale camera from a previous molecule.

    Bumps ``_vib_render_token`` so any in-flight render thread bails out
    before stomping the swapped cached output.
    """
    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is None:
        return False

    try:
        from quantui.viz_backend_router import VizBackend as _VB
        from quantui.viz_backend_router import VizTask as _VT

        chosen = app._resolve_backend(_VT.VIB_INTERACTIVE)
        if chosen != _VB.PY3DMOL:
            # Plotlymol path doesn't write to the disk cache.
            return False

        viz_settings = getattr(getattr(app, "_user_settings", None), "viz", None)
        fps = int(getattr(viz_settings, "vib_framerate_fps", 10))
        fps = max(1, fps)

        from quantui import vib_cache

        cached_html = vib_cache.get_cached_html(
            Path(result_dir),
            mode_number,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=fps,
        )
    except Exception:
        return False

    if cached_html is None:
        return False

    payload = _ensure_vib_camera_hook(cached_html)
    if reset_camera:
        payload = _VIB_CAMERA_RESET_JS + payload

    app._vib_render_token = int(getattr(app, "_vib_render_token", 0)) + 1
    with _viz_render_event(
        app,
        task="vib_interactive",
        backend="py3dmol",
        mode=mode_number,
        source="cache_sync",
    ):
        _swap_vib_output(app, payload)
        try:
            from quantui import calc_log as _clog_sync_hit

            _clog_sync_hit.log_event(
                "vib_cache_hit",
                f"mode {mode_number} backend=py3dmol fps={fps} path=sync",
            )
        except Exception:
            pass
    return True


def _render_vib_mode_py3dmol(
    app: Any,
    molecule: Any,
    mode_number: int,
    *,
    n_frames: int = 24,
    amplitude: float = 0.4,
    fps: int | None = None,
    render_token: int | None = None,
) -> None:
    """Render vibrational animation via py3Dmol multi-frame XYZ.

    Pure-numpy frame generation (no plotlymol3d
    dependency); 24 sinusoidal-phase frames over one full oscillation;
    py3Dmol view with ``addModelsAsFrames`` + ``animate``; serialized to
    HTML and atomically swapped into ``app.vib_output``.

    ``fps`` controls the playback rate of ``view.animate`` (interval =
    1000 / fps ms). When ``None``, reads from
    ``app._user_settings.viz.vib_framerate_fps``.

    ``render_token`` is checked at every output-write site so a stale
    background render thread can bail out before stomping a newer
    render's output. See ``_is_vib_stale``.
    """
    import numpy as np

    if fps is None:
        fps = int(
            getattr(
                getattr(app, "_user_settings", None) and app._user_settings.viz,
                "vib_framerate_fps",
                10,
            )
        )
    fps = max(1, int(fps))

    try:
        import py3Dmol  # noqa: F401 — probe for a friendly error; make_view imports it
    except ImportError as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, f"py3Dmol unavailable: {exc}")
        return

    freq_result = getattr(app, "_last_vib_freq_result", None)
    if freq_result is None:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, "No frequency result cached for vibrational animation.")
        return

    # Cache hit short-circuit. The cache key now includes ``fps``
    # so a user who changes the framerate will rebuild rather than play back
    # a mismatched-interval HTML blob.
    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is not None:
        try:
            from quantui import vib_cache

            cached_html = vib_cache.get_cached_html(
                Path(result_dir),
                mode_number,
                n_frames=n_frames,
                amplitude=amplitude,
                renderer="py3dmol",
                fps=fps,
            )
        except Exception:
            cached_html = None
        if cached_html is not None:
            if _is_vib_stale(app, render_token):
                return
            _swap_vib_output(app, _ensure_vib_camera_hook(cached_html))
            try:
                from quantui import calc_log as _clog_cache_hit

                _clog_cache_hit.log_event(
                    "vib_cache_hit",
                    f"mode {mode_number} backend=py3dmol fps={fps}",
                )
            except Exception:
                pass
            return

    try:
        from quantui import calc_log as _clog_cache_miss

        _clog_cache_miss.log_event(
            "vib_cache_miss",
            f"mode {mode_number} backend=py3dmol fps={fps}",
        )
    except Exception:
        pass

    try:
        displ = np.array(freq_result.displacements[mode_number - 1], dtype=float)
    except (AttributeError, IndexError, ValueError, TypeError) as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(
                app,
                f"Could not read displacements for mode {mode_number}: {exc}",
            )
        return

    atoms = list(molecule.atoms)
    base_coords = np.array(molecule.coordinates, dtype=float)
    if base_coords.shape != displ.shape:
        if not _is_vib_stale(app, render_token):
            _vib_err(
                app,
                f"Shape mismatch: base coords {base_coords.shape} vs "
                f"displacements {displ.shape}",
            )
        return

    # One full oscillation: n_frames evenly-spaced phases over [0, 2π).
    phases = np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    n_atoms = len(atoms)
    xyz_lines: list[str] = []
    for phase in phases:
        coords = base_coords + amplitude * float(phase) * displ
        xyz_lines.append(f"{n_atoms}")
        xyz_lines.append(f"mode {mode_number} phase {float(phase):+.3f}")
        for sym, xyz in zip(atoms, coords):
            xyz_lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    xyz_string = "\n".join(xyz_lines) + "\n"

    try:
        from quantui.viz_assets import make_view

        interval_ms = max(1, int(round(1000.0 / fps)))
        vib_width = 460
        view = make_view(width=vib_width, height=420)
        view.addModelsAsFrames(xyz_string, "xyz")
        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
        bg = "white" if app.theme_btn.value == "Light" else "#1e1e1e"
        view.setBackgroundColor(bg)
        view.zoomTo()
        view.animate({"loop": "forward", "interval": interval_ms, "reps": 0})
        # Prepend the camera-persistence hook so the user's interactive
        # pan/rotate state survives mode switches. The hook is idempotent
        # (guarded by ``_quantuiVibCameraHookInstalled``) and ships inside
        # the cached HTML too — so disk-cache hits also persist the camera.
        html_str = _theme.frame_viewer_html(
            _VIB_CAMERA_PERSISTENCE_JS + view._make_html(), width=vib_width
        )
    except Exception as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, f"Vibrational animation render failed: {exc}")
        raise

    if _is_vib_stale(app, render_token):
        # A newer render has superseded this one; do NOT write to vib_output
        # (would stomp the newer render's content).
        return

    _swap_vib_output(app, html_str)

    # Persist to disk cache so future visits and history replay can hit
    # this mode instantly. Non-fatal on failure — render still
    # succeeded, cache is purely an optimization.
    if result_dir is not None:
        try:
            freqs = getattr(freq_result, "frequencies_cm1", None) or []
            freq_cm1 = (
                float(freqs[mode_number - 1]) if 0 < mode_number <= len(freqs) else None
            )
            from quantui import vib_cache

            vib_cache.save_cached_html(
                Path(result_dir),
                mode_number,
                html_str,
                freq_cm1=freq_cm1,
                n_frames=n_frames,
                amplitude=amplitude,
                renderer="py3dmol",
                fps=fps,
            )
        except Exception as exc:
            try:
                from quantui import calc_log as _clog_cache_err

                _clog_cache_err.log_event(
                    "vib_cache_write_error",
                    f"mode {mode_number}: {type(exc).__name__}: {exc}"[:300],
                )
            except Exception:
                pass


def _render_vib_mode_plotlymol(
    app: Any,
    vib_data: Any,
    molecule: Any,
    mode_number: int,
    *,
    render_token: int | None = None,
) -> None:
    """Render vibrational animation via the plotlymol3d path (PlotlyMol +
    RDKit bond perception + Plotly figure). Used when user explicitly
    prefers plotlymol or when py3Dmol is unavailable."""
    if vib_data is None:
        if not _is_vib_stale(app, render_token):
            _vib_err(
                app,
                "PlotlyMol vibrational animation requires plotlymol3d, "
                "which is not installed.",
            )
        return

    try:
        from plotlymol3d import create_vibration_animation, xyzblock_to_rdkitmol
    except ImportError as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(
                app,
                f"PlotlyMol vibrational animation requires plotlymol3d "
                f"(<code>pip install plotlymol3d</code>): {exc}",
            )
        return

    xyzblock = (
        f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
        f"{molecule.to_xyz_string()}"
    )
    try:
        rdmol = xyzblock_to_rdkitmol(xyzblock, charge=molecule.charge)
    except Exception as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, f"Could not parse molecule for bond connectivity: {exc}")
        return

    try:
        anim_fig = create_vibration_animation(
            vib_data=vib_data,
            mode_number=mode_number,
            mol=rdmol,
            amplitude=0.4,
            n_frames=20,
            mode="ball+stick",
            resolution=12,
        )
        # Explicit width, not just height: the frame is sized in pixels, and a
        # responsive-width figure would otherwise render this mode at a
        # different size than the py3Dmol path. 460 matches the viewer in
        # _render_vib_mode_py3dmol so the two backends look the same.
        anim_fig.update_layout(width=460, height=420)
    except Exception as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, f"Animation generation failed: {exc}")
        raise

    import plotly.io as _pio

    anim_html = _pio.to_html(
        anim_fig,
        full_html=False,
        include_plotlyjs="require",
        config={"responsive": True},
    )
    if _is_vib_stale(app, render_token):
        return
    _swap_vib_output(app, _theme.frame_viewer_html(anim_html, width=460))


def render_vib_mode(
    app: Any,
    vib_data: Any,
    molecule: Any,
    mode_number: int,
    *,
    render_token: int | None = None,
) -> None:
    """Render vibrational animation for mode_number into ``app.vib_output``.

    Backend dispatch goes through the router (``VizTask.VIB_INTERACTIVE``):
    py3Dmol primary, plotlymol3d fallback. The py3Dmol path is preferred for
    speed and doesn't require plotlymol3d to be installed.

    ``render_token`` lets a caller (e.g. ``on_vib_mode_changed``) bump
    ``app._vib_render_token`` before spawning a worker thread, so any
    stale worker can bail out before stomping the newer render's output.
    """
    from quantui.viz_backend_router import VizBackend as _VB
    from quantui.viz_backend_router import VizTask as _VT

    chosen = app._resolve_backend(_VT.VIB_INTERACTIVE)
    if chosen == _VB.PY3DMOL:
        try:
            with _viz_render_event(
                app, task="vib_interactive", backend="py3dmol", mode=mode_number
            ):
                _render_vib_mode_py3dmol(
                    app, molecule, mode_number, render_token=render_token
                )
        except Exception:
            # viz_render_error was already logged by the context manager;
            # swallow here so a worker-thread render failure doesn't crash
            # the thread. The inner function already wrote a user-facing
            # error message via ``_vib_err``.
            pass
    elif chosen == _VB.PLOTLYMOL:
        try:
            with _viz_render_event(
                app, task="vib_interactive", backend="plotlymol", mode=mode_number
            ):
                _render_vib_mode_plotlymol(
                    app, vib_data, molecule, mode_number, render_token=render_token
                )
        except Exception:
            pass
    else:
        if not _is_vib_stale(app, render_token):
            _vib_err(
                app,
                "No vibrational animation backend available "
                "(neither py3Dmol nor plotlymol3d installed).",
            )


def iso_render_options(app: Any) -> dict:
    """Current isosurface settings, read straight off the widgets.

    One place, so the first render and every live update agree.

    ``transparent_bg`` is deliberately absent: transparency is an EXPORT
    property, applied by ``__quantuiIsoCapture`` at the moment of capture and
    undone immediately after. Putting it here would change the live viewer,
    which is exactly what the user asked it not to do.
    """

    def _val(name: str, default):
        """Widget value coerced to the default's type, or the default.

        The coercion is the point. A widget that exists but holds something
        uncoercible — a partially-built app, a stub — would otherwise raise out
        of here and take the whole render with it.
        """
        raw = getattr(getattr(app, name, None), "value", default)
        try:
            return type(default)(raw)
        except (TypeError, ValueError):
            return default

    return {
        "isovalue": _val("_iso_isovalue_slider", 0.02),
        "opacity": _val("_iso_opacity_slider", 0.85),
        "wireframe": _val("_iso_wireframe_cb", False),
        "color_scheme": _val("_iso_colors_dd", "blue-red"),
        "bgcolor": app._plotly_theme_colors()["scene_bgcolor"],
        "capture_class": (
            _ORB_PNG_INBOX_CLASS
            if getattr(app, "_orb_png_inbox", None) is not None
            else ""
        ),
    }


def _show_iso_placeholder(app: Any) -> None:
    """Molecule-only viewer in the isosurface panel, before any cube exists."""
    # On the Analysis tab app._molecule is usually None — the result came from
    # history, not the Calculate-tab molecule — so this silently did nothing and
    # the panel stayed empty. Prefer what the analysis viewer is already showing.
    mol = (
        getattr(app, "_analysis_displayed_molecule", None)
        or getattr(app, "_last_vib_molecule", None)
        or getattr(app, "_molecule", None)
    )
    if mol is None:
        app._orb_iso_output.clear_output()
        return
    try:
        from quantui.orbital_visualization import render_molecule_placeholder_py3dmol

        app._set_html_output(
            app._orb_iso_output,
            render_molecule_placeholder_py3dmol(
                mol, bgcolor=app._plotly_theme_colors()["scene_bgcolor"]
            ),
        )
    except Exception as exc:  # noqa: BLE001 — a placeholder must never block
        logger.debug("iso placeholder failed: %s", exc)
        app._orb_iso_output.clear_output()


def _show_cancel(app: Any, visible: bool) -> None:
    btn = getattr(app, "_iso_cancel_btn", None)
    if btn is not None:
        btn.layout.display = "" if visible else "none"


def iso_bridge_busy(app: Any, busy: bool) -> None:
    """Dim (or restore) the live viewer while a new cube is computed."""
    bridge = getattr(app, "_iso_js_bridge", None)
    if bridge is None:
        return
    from IPython.display import Javascript, display

    js = (
        "(function(){"  # noqa: UP031 — JS is brace-dense
        "var seq=(window.__quantuiIsoBusySeq||0)+1;window.__quantuiIsoBusySeq=seq;"
        "var n=0;function go(){n++;"
        # Abandon if a newer busy call has been issued. The retry loop waits up
        # to 2 s for the hook to exist, so a busy(true) issued while no viewer
        # was present kept polling and then fired against the NEXT viewer —
        # after busy(false) had already run. That is the overlay that appeared
        # right after the surface and never went away (reported 2026-08-04).
        "if(window.__quantuiIsoBusySeq!==seq){return;}"
        "if(window.__quantuiIsoBusy){window.__quantuiIsoBusy(%s);}"
        "else if(n<40){setTimeout(go,50);}}go();})();" % ("true" if busy else "false")
    )
    try:
        bridge.clear_output(wait=True)
        with bridge:
            display(Javascript(js))
    except Exception as exc:  # noqa: BLE001
        logger.debug("iso busy update failed: %s", exc)


def on_iso_cancel(app: Any, btn: Any = None) -> None:
    """Abandon the in-flight isosurface generation.

    cubegen is a blocking PySCF call that cannot be interrupted, so this does
    not stop the computation — it bumps ``_iso_render_token``, which every
    completion path already checks, so the worker's result is discarded when it
    arrives. From the user's side that is what cancel means: the controls come
    back now and the stale surface never appears.
    """
    app._iso_render_token = int(getattr(app, "_iso_render_token", 0)) + 1
    _show_cancel(app, False)
    iso_bridge_busy(app, False)
    gen = getattr(app, "_iso_generate_btn", None)
    if gen is not None:
        gen.disabled = False
        gen.description = "Generate Isosurface"
    sp = getattr(app, "_iso_spinner", None)
    if sp is not None:
        sp.layout.display = "none"
    try:
        app._activity_end(kind="compute")
    except Exception:  # noqa: BLE001 — cancelling must never raise
        pass
    status = getattr(app, "_iso_export_status", None)
    if status is not None:
        status.value = '<span style="color:#555">Cancelled.</span>'


def iso_bridge_update(app: Any, **opts: Any) -> None:
    """Push appearance changes to the LIVE viewer — no Python re-render.

    This is what keeps the camera. A Python re-render replaces the viewer
    wholesale, so the orientation the user rotated into is lost (GOTCHAS:
    "Camera state does NOT persist across atomic HTML swaps"); the JS side
    instead saves getView()/setView() around a surface rebuild.

    It also stops the panel collapsing and the page jumping on every slider
    tweak, because no output is swapped at all.
    """
    bridge = getattr(app, "_iso_js_bridge", None)
    if bridge is None:
        return
    import json

    from IPython.display import Javascript, display

    payload = json.dumps(opts)
    # Retry-until-present mirrors _vib_bridge_set_mode: at the moment a slider
    # fires, the viewer may still be loading its async 3Dmol bundle.
    js = (
        "(function(){var n=0;function go(){n++;"  # noqa: UP031 — JS is brace-dense
        "if(window.__quantuiIsoUpdate){window.__quantuiIsoUpdate(%s);}"
        "else if(n<40){setTimeout(go,50);}}go();})();" % payload
    )
    try:
        bridge.clear_output(wait=True)
        with bridge:
            display(Javascript(js))
    except Exception as exc:  # noqa: BLE001 — a slider must never raise
        logger.debug("iso bridge update failed: %s", exc)


def update_iso_enclosed_label(app: Any) -> None:
    """Show what fraction of the orbital's density the current isovalue holds."""
    label = getattr(app, "_iso_enclosed_label", None)
    if label is None:
        return
    cube = getattr(app, "_last_cube_path", None)
    iso = float(getattr(getattr(app, "_iso_isovalue_slider", None), "value", 0.02))
    if cube is None or not Path(cube).exists():
        label.value = ""
        return
    from quantui.orbital_visualization import enclosed_density_fraction

    frac = enclosed_density_fraction(Path(cube), iso)
    label.value = (
        ""
        if frac is None
        else (
            f'<span style="font-size:12px;color:#555">encloses '
            f"<b>{frac * 100:.1f}%</b> of the density</span>"
        )
    )


def on_iso_appearance_changed(app: Any, change: dict | None = None) -> None:
    """Isovalue / opacity / colours changed — update the live viewer in place."""
    from quantui.orbital_visualization import orbital_colors

    opts = iso_render_options(app)
    pos, neg = orbital_colors(str(opts["color_scheme"]))
    iso_bridge_update(
        app,
        iso=opts["isovalue"],
        op=opts["opacity"],
        wf=opts["wireframe"],
        pos=pos,
        neg=neg,
    )
    update_iso_enclosed_label(app)


def rerender_3d_scenes_for_theme(app: Any) -> None:
    """Re-render the isosurface and vibrational viewers after a theme change.

    Both bake their background into the generated HTML at render time — the
    colour comes from ``_plotly_theme_colors()["scene_bgcolor"]``, and py3Dmol
    paints it into the WebGL scene rather than reading it from CSS. Nothing
    re-reads it later, so switching Light/Dark left the old background in place
    until the user happened to regenerate. Reported 2026-08-04: *"the background
    ... is sticky to the theme ... but will [change] if I calculate a new
    isosurface."*

    Both re-renders are CHEAP by construction and must stay that way:

    - The isosurface re-reads the cube already on disk. It does NOT re-run
      cubegen, which is the slow part (15-30 s, and up to 4.6x that at the
      finest grid). Re-generating on a theme toggle would be unusable.
    - The vibrational viewer rebuilds from cached displacements, not from a new
      frequency calculation.
    - The trajectory viewer rebuilds from ``_last_traj_result``, not from a new
      optimization.

    The two molecule viewers (Calculate and Analysis tabs) are NOT handled here
    — ``_rerender_plotly_theme`` routes those through ``_rerender_3d_views``,
    which already knew how to redraw them. It simply was not being called on a
    theme change, which is why the Analysis-tab viewer stayed dark while the
    Calculate-tab one updated.

    ⚠️ The vib path needs a full rebuild, not ``__quantuiVibSetMode``. That
    bridge switches frames on the existing viewer client-side — which is exactly
    why the camera survives a mode change — but it never touches the scene
    background, so it cannot fix this.
    """
    # (the isosurface reads its colour via iso_render_options below)

    # ── Orbital isosurface ──────────────────────────────────────────────
    # A background colour is not geometry, so this is one JS call on the live
    # viewer rather than a re-render. Re-rendering was measurably slow — the
    # viewer HTML is megabytes at the finer grids — and it also threw away the
    # camera on every theme toggle.
    try:
        if getattr(app, "_last_cube_path", None) is not None:
            iso_bridge_update(app, bg=app._plotly_theme_colors()["scene_bgcolor"])
    except Exception as exc:  # noqa: BLE001 — a theme toggle must never raise
        logger.warning("isosurface theme update failed: %s", exc)

    # ── Reorganization-energy geometries ────────────────────────────────
    # Same bake-in as every other py3Dmol viewer: bgcolor is painted into the
    # scene at render time. Redrawing is cheap here — the geometries are a few
    # dozen coordinates already in memory, no file read and no recomputation.
    try:
        if getattr(app, "_reorg_geometries", None):
            from quantui.app_analysis import render_reorg_geometries

            render_reorg_geometries(app)
    except Exception as exc:  # noqa: BLE001 — a theme toggle must never raise
        logger.warning("reorg geometry theme re-render failed: %s", exc)

    # ── Optimization trajectory ─────────────────────────────────────────
    # Only when the panel is actually populated: re-rendering an empty
    # trajectory viewer would build one on a tab the user has never opened.
    # The populated-check is INSIDE the try. It reads app.traj_output.children,
    # and this function promises never to raise on a theme toggle — a guard that
    # can itself throw is not a guard. (Caught by a test whose app stub had no
    # traj_output: len() on the auto-Mock raised straight through.)
    try:
        traj = getattr(app, "_last_traj_result", None)
        children = getattr(getattr(app, "traj_output", None), "children", ())
        if traj is not None and len(children) > 0:
            app._show_opt_trajectory(traj)
    except Exception as exc:  # noqa: BLE001 — a theme toggle must never raise
        logger.warning("trajectory theme re-render failed: %s", exc)

    # ── Vibrational animation ───────────────────────────────────────────
    molecule = getattr(app, "_last_vib_molecule", None)
    freq_result = getattr(app, "_last_vib_freq_result", None)
    mode_dd = getattr(app, "vib_mode_dd", None)
    if molecule is not None and freq_result is not None and mode_dd is not None:
        try:
            render_vib_mode(
                app,
                getattr(app, "_last_vib_data", None),
                molecule,
                int(mode_dd.value),
            )
        except Exception as exc:  # noqa: BLE001 — same
            logger.warning("vib theme re-render failed: %s", exc)


def on_vib_mode_changed(app: Any, change: dict[str, Any]) -> None:
    """Re-render vibrational animation when mode dropdown changes."""
    mode_number = change["new"]
    vib_data = getattr(app, "_last_vib_data", None)
    molecule = getattr(app, "_last_vib_molecule", None)
    freq_result = getattr(app, "_last_vib_freq_result", None)
    # vib_data may be None when plotlymol3d is unavailable — the py3Dmol
    # render path doesn't need it. Bail only if we can't render at all.
    if molecule is None or freq_result is None:
        return

    # Single-viewer path: switch modes client-side on the one persistent viewer
    # (camera preserved, no rebuild). Export + prev/next still drive vib_mode_dd,
    # so they keep working unchanged.
    if getattr(app, "_vib_single_viewer_active", False):
        _vib_bridge_set_mode(app, mode_number)
        return

    # Cache-hit fast path: swap cached HTML synchronously, no placeholder,
    # no thread. Bumps the render token internally to invalidate any
    # in-flight render.
    if _try_vib_cache_hit_sync(app, mode_number):
        return

    label = next(
        (lbl for lbl, num in app.vib_mode_dd.options if num == mode_number),
        f"mode {mode_number}",
    )
    # Bump render token so older in-flight render threads bail before they
    # stomp the newer render's output. Eliminates the intermittent
    # missing-render symptom from rapid mode switching.
    app._vib_render_token = int(getattr(app, "_vib_render_token", 0)) + 1
    token = app._vib_render_token
    _swap_vib_output(
        app,
        f'<p style="color:#555;font-style:italic;padding:8px">'
        f"⏳ Rendering vibrational animation ({label})…</p>",
    )
    threading.Thread(
        target=app._render_vib_mode,
        args=(vib_data, molecule, mode_number),
        kwargs={"render_token": token},
        daemon=True,
    ).start()


_STEPPER_BTN_STYLE = (
    f"padding:2px 9px;border:1px solid {_theme.BORDER};border-radius:4px;"
    "background:#f8fafc;color:#334155;cursor:pointer;font-size:13px;line-height:1.4;"
)

# Shared single-viewer stepper logic. Drives ``viewer.setFrame()`` on a viewer
# whose frames are ALL already loaded client-side via ``addModelsAsFrames`` — so
# navigation never rebuilds the viewer, the camera (rotation/zoom) stays put
# across frames, and there is no per-frame HTML/network round-trip. Play/pause is
# a self-managed setInterval (not 3Dmol's animate(), so manual stepping and play
# never fight). Tokens are substituted in :func:`_frame_stepper_controls`.
_STEPPER_JS = """
(function(){
  var UID="__UID__", N=__N__, IV=__IV__, LOOP=__LOOP__;
  var AB_START=__AB_START__, AB_OTHER=__AB_OTHER__;
  __EXTRA__
  function g(p){return document.getElementById(p+UID);}
  var slider=g("st_slider_"), lbl=g("st_lbl_"), prevB=g("st_prev_"),
      nextB=g("st_next_"), playB=g("st_play_"), abB=g("st_ab_");
  var cur=__START__, timer=null;
  function vw(){return window["viewer_"+UID];}
  function label(i){ __LABEL_BODY__ }
  function draw(i){
    i=Math.max(0,Math.min(N-1,i)); cur=i;
    var v=vw();
    if(v){ try{ var p=v.setFrame(i);
      if(p&&p.then){ p.then(function(){v.render();}); } else { v.render(); }
    }catch(e){ try{ v.render(); }catch(_){} } }
    if(slider) slider.value=i;
    if(lbl) lbl.innerHTML=label(i);
    if(prevB) prevB.disabled=(i<=0);
    if(nextB) nextB.disabled=(i>=N-1);
    if(abB) abB.innerHTML=(i===0)?AB_START:AB_OTHER;
  }
  function stop(){ if(timer){clearInterval(timer);timer=null;}
    if(playB) playB.innerHTML="\\u25b6 Play"; }
  function play(){ if(N<=1) return;
    if(cur>=N-1) draw(0);  // at the end → replay from the first frame
    if(playB) playB.innerHTML="\\u23f8 Pause";
    timer=setInterval(function(){
      if(cur>=N-1){ if(LOOP){ draw(0); return; } stop(); return; }
      draw(cur+1);
    }, IV); }
  if(prevB) prevB.onclick=function(){stop();draw(cur-1);};
  if(nextB) nextB.onclick=function(){stop();draw(cur+1);};
  if(playB) playB.onclick=function(){ timer?stop():play(); };
  if(abB)   abB.onclick  =function(){ stop();draw(cur===0?N-1:0); };
  if(slider)slider.oninput=function(){ stop();draw(parseInt(slider.value,10)); };
  var t=0, poll=setInterval(function(){ t++;
    if(vw()){ clearInterval(poll); draw(cur); }
    else if(t>200){ clearInterval(poll);
      if(lbl) lbl.innerHTML="3D viewer failed to load"; }
  },50);
})();
"""


def _frame_stepper_controls(
    uid: str,
    n: int,
    interval_ms: int,
    *,
    label_js: str,
    initial_label: str,
    loop: bool,
    ab_at_start: str | None = None,
    ab_other: str | None = None,
    scrub_title: str = "Scrub frames",
    start_index: int | None = None,
    extra_decls: str = "",
) -> str:
    """Build in-HTML stepper controls (prev/play/next, scrub slider, optional
    A/B flip, live label) for a single multi-frame py3Dmol viewer.

    Element ids are namespaced with the viewer's ``uid`` so multiple viewers on
    one page never collide. The script polls for the global ``viewer_<uid>``
    (py3Dmol creates it after the async 3Dmol.js load resolves) before wiring up.

    ``label_js`` is the JS body of ``function label(i){…}`` returning the label
    HTML for frame ``i``; ``extra_decls`` is JS injected at the top of the IIFE
    (e.g. per-frame energy arrays). ``loop`` makes Play cycle forever, else it
    runs once and stops on the last frame.
    """
    import json

    btn = _STEPPER_BTN_STYLE
    start = (n - 1) if start_index is None else start_index
    ab_html = ""
    if ab_at_start is not None:
        ab_html = (
            f'<button id="st_ab_{uid}" type="button" '
            'title="Jump between the first and last frame" '
            # start index is the last frame, so the button initially offers the
            # "other" (first-frame) action.
            f'style="{btn}">{ab_other}</button>'
        )
    bar = (
        '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;'
        'margin:4px 0 2px;font-size:13px;">'
        f'<button id="st_prev_{uid}" type="button" title="Previous frame" '
        f'style="{btn}">&#9664;</button>'
        f'<button id="st_play_{uid}" type="button" '
        f'style="{btn}">&#9654; Play</button>'
        f'<button id="st_next_{uid}" type="button" title="Next frame" '
        f'style="{btn}">&#9654;</button>'
        f'<input id="st_slider_{uid}" type="range" min="0" max="{n - 1}" '
        f'value="{start}" step="1" title="{scrub_title}" '
        'style="flex:1;min-width:110px;vertical-align:middle;">'
        f"{ab_html}"
        "</div>"
        f'<div id="st_lbl_{uid}" '
        'style="font-size:12px;color:#64748b;margin:0 0 4px 2px;">'
        f"{initial_label}</div>"
    )
    js = (
        _STEPPER_JS.replace("__UID__", uid)
        .replace("__N__", str(n))
        .replace("__IV__", str(interval_ms))
        .replace("__LOOP__", "1" if loop else "0")
        .replace("__START__", str(start))
        .replace("__AB_START__", json.dumps(ab_at_start or ""))
        .replace("__AB_OTHER__", json.dumps(ab_other or ""))
        .replace("__EXTRA__", extra_decls)
        .replace("__LABEL_BODY__", label_js)
    )
    return bar + f"<script>{js}</script>"


# UID-scoped capture function for the reorg-geometry viewer (M-EXPORT2 EXP2.2).
# Every render of build_reorg_geometry_viewer_html gets a fresh uid (full
# atomic HTML swap — see app_analysis.render_reorg_geometries), so, unlike the
# isosurface viewer's single bare window.__quantuiIsoCapture global, this
# defines a per-render global named after the uid to avoid one render's button
# capturing a different render's (possibly already-detached) viewer.
_REORG_CAPTURE_JS = """
(function(){
  var UID="__UID__";
  window["__CAPFN__"] = function(transparent){
    var vw = window["viewer_"+UID];
    if(!vw || !vw.pngURI){ return null; }
    if(!transparent){ return vw.pngURI(); }
    var uri=null;
    try{
      vw.setBackgroundColor(__BG__, 0.0); vw.render();
      uri=vw.pngURI();
    } finally {
      vw.setBackgroundColor(__BG__, 1.0); vw.render();
    }
    return uri;
  };
})();
"""


def build_reorg_geometry_viewer_html(
    geometries: list[dict],
    *,
    bgcolor: str = "white",
    width: int = 560,
    height: int = 420,
    capture_class: str = "",
) -> str:
    """Step through the DISTINCT geometries behind a Marcus 4-point run.

    ``geometries`` is ``[{"label", "atoms", "coordinates", "note"}, ...]``.

    Not four steps. The scheme evaluates four energies on **two** geometries per
    channel (three across both channels) — E_ion(R_neutral) shares its geometry
    with E_neutral(R_neutral), and likewise for R_ion. A four-step control would
    show each geometry twice and invite the reading that all four differ, which
    is why each step is labelled by geometry with the energies computed on it
    listed underneath.

    Deliberately NOT animated (user request): λ is a comparison between two
    states, not a trajectory through them, and looping would imply a path that
    was never computed. Built on ``_frame_stepper_controls`` so the camera
    behaviour, offline loading and control styling match the trajectory viewer
    rather than being reinvented.

    ``capture_class`` wires a "Save PNG" button (M-EXPORT2 EXP2.2), mirroring
    the isosurface viewer's capture bridge (ORBX.1) but with a uid-scoped
    capture function — see ``_REORG_CAPTURE_JS``. Empty omits the button.
    """
    import json
    import re

    from quantui.viz_assets import make_view

    if not geometries:
        return '<p style="color:#555;padding:8px">No geometries available.</p>'

    blocks = []
    for g in geometries:
        atoms, coords = g["atoms"], g["coordinates"]
        lines = [str(len(atoms)), g.get("label", "")]
        for sym, xyz in zip(atoms, coords):
            lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
        blocks.append("\n".join(lines))

    view = make_view(width=width, height=height)
    view.addModelsAsFrames("\n".join(blocks) + "\n", "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.setBackgroundColor(bgcolor)
    view.zoomTo()
    view_html = view._make_html()

    m = re.search(r"3dmolviewer_(\w+)", view_html)
    if m is None:
        return _theme.frame_viewer_html(view_html, width=width)
    uid = m.group(1)

    if capture_class:
        capture_fn = f"__quantuiReorgCapture_{uid}"
        capture_js = (
            _REORG_CAPTURE_JS.replace("__UID__", uid)
            .replace("__CAPFN__", capture_fn)
            .replace("__BG__", json.dumps(bgcolor))
        )
        view_html = (
            view_html
            + f"<script>{capture_js}</script>"
            + _png_capture_controls(uid, capture_class, capture_fn=capture_fn)
        )

    if len(geometries) < 2:
        return _theme.frame_viewer_html(view_html, width=width)

    labels = json.dumps(
        [g.get("label", f"Geometry {i + 1}") for i, g in enumerate(geometries)]
    )
    notes = json.dumps([g.get("note", "") for g in geometries])
    controls = _frame_stepper_controls(
        uid,
        len(geometries),
        1000,  # unused: loop=False and Play is not the point here
        label_js=(
            'var s="<b>"+LBL[i]+"</b>";'
            'if(NOTE[i]) s+="<br><span style=\'color:#555\'>"+NOTE[i]+"</span>";'
            "return s;"
        ),
        initial_label=geometries[0].get("label", "Geometry 1"),
        loop=False,
        ab_at_start="⇄ Compare first/last",
        ab_other="⇄ Back to first",
        scrub_title="Step between geometries",
        start_index=0,
        extra_decls=f"var LBL={labels}; var NOTE={notes};",
    )
    return _theme.frame_viewer_html(view_html, width=width, controls=controls)


def build_reorg_overlay_html(
    reference: dict,
    other: dict,
    *,
    bgcolor: str = "white",
    width: int = 560,
    height: int = 420,
    exaggerate: float = 1.0,
) -> str:
    """Both geometries superimposed, with displacement arrows.

    First attempt drew two full ball-and-stick models in solid colours. For a
    molecule whose geometries nearly coincide — which is most of them, since λ
    is usually a small relaxation — that produces two interpenetrating solids
    and no legible answer to "what moved". Reported 2026-08-05.

    So the visual hierarchy now matches the question:

    - the **reference** is a thin grey wireframe — context, not content;
    - the **relaxed** geometry is a thin coloured wireframe;
    - **arrows** run from each reference atom to its relaxed position, and they
      are the actual signal. An arrow has direction and length, which is what a
      displacement is; two overlapping blobs have neither.

    ``exaggerate`` scales the arrows (not the structures) for relaxations too
    small to see at 1:1 — the same convention as vibrational-mode arrows. The
    structures always show true positions, so nothing displayed is fictional;
    only the arrows are amplified, and the legend says by how much.

    No alignment is performed. Both geometries came from optimizations seeded
    from the same structure with the same atom ordering, so the displacement is
    physical; superimposing (Kabsch) would rotate away part of what λ measures.
    """
    import re

    from quantui.viz_assets import make_view

    def _xyz(g: dict) -> str:
        lines = [str(len(g["atoms"])), g.get("label", "")]
        for sym, c in zip(g["atoms"], g["coordinates"]):
            lines.append(f"{sym} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        return "\n".join(lines) + "\n"

    view = make_view(width=width, height=height)
    # Thin lines, not spheres and thick sticks: the structures are here to give
    # the arrows something to hang on, and bulk is exactly what obscured them.
    view.addModel(_xyz(reference), "xyz")
    view.setStyle({"model": 0}, {"stick": {"radius": 0.05, "color": "#94a3b8"}})
    view.addModel(_xyz(other), "xyz")
    view.setStyle({"model": 1}, {"stick": {"radius": 0.05, "color": "#2166ac"}})

    ref_c = reference["coordinates"]
    oth_c = other["coordinates"]
    n_drawn = 0
    max_d = 0.0
    if len(ref_c) == len(oth_c):
        for a, b in zip(ref_c, oth_c):
            d = sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)) ** 0.5
            max_d = max(max_d, d)
            # Skip atoms that did not move: a zero-length arrow renders as a
            # dot and reads as noise.
            if d < 1e-4:
                continue
            tip = [
                float(a[i]) + (float(b[i]) - float(a[i])) * exaggerate for i in range(3)
            ]
            view.addArrow(
                {
                    "start": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
                    "end": {"x": tip[0], "y": tip[1], "z": tip[2]},
                    "radius": 0.06,
                    "radiusRatio": 2.5,
                    "mid": 0.75,
                    "color": "#b2182b",
                }
            )
            n_drawn += 1

    view.setBackgroundColor(bgcolor)
    view.zoomTo()
    # py3Dmol has no type stubs (ignore_missing_imports); _make_html()
    # genuinely returns str.
    view_html = cast(str, view._make_html())
    if re.search(r"3dmolviewer_(\w+)", view_html) is None:
        return view_html

    scale_note = (
        "" if exaggerate == 1.0 else f" &mdash; arrows scaled &times;{exaggerate:g}"
    )
    moved = (
        "no atom moved measurably"
        if n_drawn == 0
        else f"largest shift {max_d:.3f} &Aring;"
    )
    legend = (
        '<div style="margin:4px 0 2px;font-size:12px;padding:0 2px;line-height:1.5">'
        '<span style="color:#94a3b8;font-weight:700">&#9473;</span> '
        f'{reference.get("label", "reference").split(" — ")[0]}'
        '&ensp;<span style="color:#2166ac;font-weight:700">&#9473;</span> '
        f'{other.get("label", "other").split(" — ")[0]}'
        '&ensp;<span style="color:#b2182b;font-weight:700">&#10230;</span> '
        f"displacement ({moved}){scale_note}"
        '<br><span style="color:#555">Shown as computed, not superimposed &mdash; '
        "the displacement is the quantity of interest.</span>"
        "</div>"
    )
    return _theme.frame_viewer_html(view_html, width=width, controls=legend)


def _preopt_controls_html(uid: str, n: int, interval_ms: int) -> str:
    """Stepper controls for the pre-opt preview (input → relaxed)."""
    label_js = (
        'if(i===0) return "Frame 1/"+N+" \\u2022 Input (your geometry)";'
        'if(i===N-1) return "Frame "+N+"/"+N+" \\u2022 Relaxed (final)";'
        'return "Frame "+(i+1)+"/"+N+" \\u2022 relaxing\\u2026";'
    )
    return _frame_stepper_controls(
        uid,
        n,
        interval_ms,
        label_js=label_js,
        initial_label=f"Frame {n}/{n} &bull; Relaxed (final)",
        loop=False,  # one-shot: stop on the relaxed frame (no lingering "relaxing…")
        ab_at_start="⇄ Show relaxed",
        ab_other="⇄ Show input",
        scrub_title="Scrub the relaxation",
    )


def build_preopt_preview_html(
    atoms: list[str],
    frames: list[list[list[float]]],
    *,
    bgcolor: str = "white",
    fps: int = 8,
) -> str:
    """Build an interactive py3Dmol view of a classical pre-opt relaxation.

    ``frames`` is a list of per-iteration coordinate snapshots (from
    ``preopt.preoptimize_with_trajectory``); ``atoms`` is the element list.
    Returns self-contained, offline-safe HTML (3Dmol.js loaded from the vendored
    bundle via ``make_view``). All frames are loaded client-side via
    ``addModelsAsFrames``, and a stepper UI (prev/next, play/pause, scrub
    slider, and an input&hairsp;&#8644;&hairsp;relaxed A/B flip) drives
    ``viewer.setFrame`` so the user can compare geometries without re-rendering.
    A single-frame trajectory (no relaxation / FF fallback) renders as a static
    structure with no controls. Used by the interactive "Preview
    pre-optimization" flow.
    """
    import re

    from quantui.viz_assets import make_view

    n = len(atoms)
    lines: list[str] = []
    for coords in frames:
        lines.append(str(n))
        lines.append("preopt")
        for sym, xyz in zip(atoms, coords):
            lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    xyz_string = "\n".join(lines) + "\n"

    width = 460
    view = make_view(width=width, height=290)
    view.addModelsAsFrames(xyz_string, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.setBackgroundColor(bgcolor)
    view.zoomTo()
    view_html = view._make_html()

    n_frames = len(frames)
    # Single frame (FF no-op / RDKit absent): nothing to step through.
    if n_frames <= 1:
        return _theme.frame_viewer_html(view_html, width=width)

    m = re.search(r"3dmolviewer_(\w+)", view_html)
    if m is None:
        # Couldn't find the viewer id to wire controls to — fall back to a
        # plain auto-loop animation so the relaxation is still visible.
        interval_ms = max(1, int(round(1000.0 / max(1, fps))))
        view.animate({"loop": "forward", "interval": interval_ms, "reps": 0})
        return _theme.frame_viewer_html(view._make_html(), width=width)

    interval_ms = max(1, int(round(1000.0 / max(1, fps))))
    controls = _preopt_controls_html(m.group(1), n_frames, interval_ms)
    return _theme.frame_viewer_html(view_html, width=width, controls=controls)


def build_trajectory_viewer_html(
    xyzblocks: list[str],
    *,
    formula: str = "",
    energies: list[float] | None = None,
    rel_e: list[float] | None = None,
    bgcolor: str = "white",
    width: int = 460,
    height: int = 340,
    fps: int = 8,
) -> str:
    """Build an interactive py3Dmol view of a geometry-optimization trajectory.

    Loads every step as a frame of ONE viewer (``addModelsAsFrames``) and wires
    an in-HTML stepper (prev/next, play/pause, scrub slider, start↔final flip,
    per-step energy label) that navigates with ``viewer.setFrame`` — so the
    camera stays put across steps and there is no per-frame HTML rebuild
    (the previous carousel rebuilt a fresh viewer each step, resetting the
    rotation/zoom and flickering). Offline-safe via the vendored 3Dmol loader
    (``make_view``). ``energies`` (Hartree) and ``rel_e`` (kcal/mol) are optional
    per-step annotations; a <2-frame trajectory renders as a static structure.
    """
    import json
    import re

    from quantui.viz_assets import make_view

    n = len(xyzblocks)
    xyz_string = "\n".join(b.rstrip("\n") for b in xyzblocks) + "\n"

    view = make_view(width=width, height=height)
    view.addModelsAsFrames(xyz_string, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.setBackgroundColor(bgcolor)
    view.zoomTo()
    view_html = view._make_html()

    if n <= 1:
        return _theme.frame_viewer_html(view_html, width=width)

    m = re.search(r"3dmolviewer_(\w+)", view_html)
    if m is None:
        # can't wire controls without the viewer id
        return _theme.frame_viewer_html(view_html, width=width)

    interval_ms = max(1, int(round(1000.0 / max(1, fps))))
    eabs = json.dumps([float(e) for e in energies]) if energies else "null"
    erel = json.dumps([float(e) for e in rel_e]) if rel_e else "null"
    fjs = json.dumps(formula or "")
    label_js = (
        'var s="Step "+i+" / "+(N-1);'
        f'if({fjs}) s+=" \\u00b7 "+{fjs};'
        'if(EABS&&EABS[i]!=null) s+=" \\u00b7 E = "+EABS[i].toFixed(8)+" Ha";'
        "if(EREL&&EREL[i]!=null) s+="
        '" \\u00b7 \\u0394E = "+(EREL[i]>=0?"+":"")+EREL[i].toFixed(3)+" kcal/mol";'
        "return s;"
    )
    controls = _frame_stepper_controls(
        m.group(1),
        n,
        interval_ms,
        label_js=label_js,
        initial_label=f"Step {n - 1} / {n - 1}",
        loop=True,  # optimization animation: loop continuously
        ab_at_start="⇄ Final geometry",
        ab_other="⇄ First step (input)",
        scrub_title="Scrub the optimization steps",
        extra_decls=f"var EABS={eabs}; var EREL={erel};",
    )
    return _theme.frame_viewer_html(view_html, width=width, controls=controls)


# Single-viewer vibrational animation. ONE py3Dmol viewer holds every mode; the
# per-mode oscillation frames are computed client-side from the embedded
# displacement vectors (tiny: n_atoms×3 per mode) on demand, and a mode switch
# calls ``window.__quantuiVibSetMode`` to swap frames on the SAME viewer instance
# — so the camera (rotation/zoom) is preserved exactly across modes, with no
# rebuild/flash. ``fit`` is true only for the initial mode (zoom-to-fit); switches
# never re-fit. Replaces the old per-mode rebuild + fragile getView/setView hook.
_VIB_VIEWER_JS = """
(function(){
  var UID="__UID__";
  var SYM=__SYM__, BASE=__BASE__, DISPL=__DISPL__;
  var NAT=__NAT__, NF=__NF__, AMP=__AMP__, IV=__IV__, BG=__BG__, INIT=__INIT__;
  function vw(){ return window["viewer_"+UID]; }
  function frames(m){
    var d=DISPL[m]; if(!d) return null;
    var out="";
    for(var f=0; f<NF; f++){
      var ph=Math.sin(2*Math.PI*f/NF);
      out += NAT+"\\nmode "+m+"\\n";
      for(var a=0; a<NAT; a++){
        out += SYM[a]+" "+(BASE[a][0]+AMP*ph*d[a][0]).toFixed(5)+" "+
               (BASE[a][1]+AMP*ph*d[a][1]).toFixed(5)+" "+
               (BASE[a][2]+AMP*ph*d[a][2]).toFixed(5)+"\\n";
      }
    }
    return out;
  }
  window.__quantuiVibSetMode=function(m, fit){
    var v=vw(); if(!v) return false;
    var xyz=frames(m); if(xyz===null) return false;
    try{
      // stopAnimate FIRST: setMode may be called more than once for the same
      // viewer (initial render + the dropdown observer + each mode switch).
      // Without stopping the running loop, animate() stacks additional loops,
      // advancing frames several times per tick — glitchy, too-fast playback
      // that ignores the fps interval. Stopping guarantees exactly one loop.
      if(v.stopAnimate) v.stopAnimate();
      v.removeAllModels();
      v.addModelsAsFrames(xyz,"xyz");
      v.setStyle({"stick":{},"sphere":{"scale":0.3}});
      v.setBackgroundColor(BG);
      if(fit) v.zoomTo();   // fit only on first mode; switches keep the camera
      v.animate({"loop":"forward","interval":IV,"reps":0});
      v.render();
    }catch(e){ return false; }
    return true;
  };
  // Live framerate change (custom fps setting): update the interval and restart
  // the loop on the current frames — no rebuild, so the camera is preserved.
  window.__quantuiVibSetFps=function(fps){
    IV=Math.max(1,Math.round(1000/Math.max(1,fps)));
    var v=vw(); if(!v) return;
    try{ if(v.stopAnimate) v.stopAnimate();
      v.animate({"loop":"forward","interval":IV,"reps":0}); v.render();
    }catch(e){}
  };
  var t=0, poll=setInterval(function(){ t++;
    if(vw()){ clearInterval(poll); window.__quantuiVibSetMode(INIT, true); }
    else if(t>200){ clearInterval(poll); }
  },50);
})();
"""


def build_vib_viewer_html(
    molecule: Any,
    freq_result: Any,
    mode_numbers: list[int],
    initial_mode: int,
    *,
    amplitude: float = 0.4,
    n_frames: int = 24,
    fps: int = 10,
    bgcolor: str = "white",
    width: int = 460,
    height: int = 420,
) -> str:
    """Build a single py3Dmol viewer that holds every vibrational mode.

    All modes share ONE viewer instance; oscillation frames are generated
    client-side from the embedded per-mode displacement vectors, and switching
    modes (``window.__quantuiVibSetMode``) swaps frames on that same viewer so
    the camera is preserved across modes. Offline-safe via the vendored 3Dmol
    loader (``make_view``). Raises if displacements are missing/misshaped so the
    caller can fall back to the legacy per-mode renderer.
    """
    import json
    import re

    import numpy as np

    from quantui.viz_assets import make_view

    displacements = getattr(freq_result, "displacements", None)
    if not displacements:
        raise ValueError("freq_result has no displacements for single-viewer vib")

    atoms = list(molecule.atoms)
    base = np.asarray(molecule.coordinates, dtype=float)
    n_atoms = len(atoms)
    displ_map: dict[int, list] = {}
    for m in mode_numbers:
        d = np.asarray(displacements[m - 1], dtype=float)
        if d.shape != base.shape:
            raise ValueError(
                f"mode {m} displacement shape {d.shape} != coords {base.shape}"
            )
        displ_map[int(m)] = d.tolist()

    view = make_view(width=width, height=height)
    view.setBackgroundColor(bgcolor)
    view_html = view._make_html()  # empty viewer; JS populates the initial mode
    m_uid = re.search(r"3dmolviewer_(\w+)", view_html)
    if m_uid is None:
        raise ValueError("could not find py3Dmol viewer id")

    interval_ms = max(1, int(round(1000.0 / max(1, fps))))
    js = (
        _VIB_VIEWER_JS.replace("__UID__", m_uid.group(1))
        .replace("__SYM__", json.dumps(atoms))
        .replace("__BASE__", json.dumps(base.tolist()))
        .replace("__DISPL__", json.dumps(displ_map))
        .replace("__NAT__", str(n_atoms))
        .replace("__NF__", str(int(n_frames)))
        .replace("__AMP__", repr(float(amplitude)))
        .replace("__IV__", str(interval_ms))
        .replace("__BG__", json.dumps(bgcolor))
        .replace("__INIT__", str(int(initial_mode)))
    )
    return _theme.frame_viewer_html(f"{view_html}<script>{js}</script>", width=width)


def _vib_single_viewer_supported(app: Any, freq_result: Any) -> bool:
    """True when the single-persistent-viewer vib path applies: py3Dmol backend
    is selected and the result carries per-mode displacement vectors."""
    try:
        from quantui.viz_backend_router import VizBackend as _VB
        from quantui.viz_backend_router import VizTask as _VT

        if app._resolve_backend(_VT.VIB_INTERACTIVE) != _VB.PY3DMOL:
            return False
        return bool(getattr(freq_result, "displacements", None))
    except Exception:
        return False


def _render_vib_single_viewer(
    app: Any,
    freq_result: Any,
    molecule: Any,
    initial_mode: int,
    mode_numbers: list[int],
) -> bool:
    """Build + swap in the single-viewer vib animation. Returns True on success
    (sets ``app._vib_single_viewer_active``); False to fall back to legacy."""
    try:
        viz_settings = getattr(getattr(app, "_user_settings", None), "viz", None)
        fps = max(1, int(getattr(viz_settings, "vib_framerate_fps", 10)))
        bg = "white" if app.theme_btn.value == "Light" else "#1e1e1e"
        with _viz_render_event(
            app, task="vib_interactive", backend="py3dmol", source="single_viewer"
        ):
            html = build_vib_viewer_html(
                molecule, freq_result, mode_numbers, initial_mode, fps=fps, bgcolor=bg
            )
    except Exception as exc:  # noqa: BLE001 — fall back to the legacy renderer
        try:
            from quantui import calc_log as _clog_sv

            _clog_sv.log_event(
                "vib_single_viewer_fallback", f"{type(exc).__name__}: {exc}"[:200]
            )
        except Exception:
            pass
        app._vib_single_viewer_active = False
        return False
    _swap_vib_output(app, html)
    app._vib_single_viewer_active = True
    return True


def _vib_bridge_set_mode(app: Any, mode_number: int) -> None:
    """Switch the live single-viewer to ``mode_number`` client-side (camera kept).

    Emits a one-shot JS call to ``window.__quantuiVibSetMode`` via a hidden
    bridge Output; retries briefly in case the viewer is still loading."""
    bridge = getattr(app, "_vib_js_bridge", None)
    if bridge is None:
        return
    from IPython.display import Javascript, display

    # %-formatting is deliberate here: the payload is JavaScript, which is dense
    # with braces, so an f-string or .format() would require doubling every one.
    js = (
        "(function(){var n=0;function go(){n++;"  # noqa: UP031 — see above
        "if(window.__quantuiVibSetMode){window.__quantuiVibSetMode(%d,false);}"
        "else if(n<40){setTimeout(go,50);}}go();})();" % int(mode_number)
    )
    try:
        bridge.clear_output(wait=True)
        with bridge:
            display(Javascript(js))
    except Exception:
        pass


def _vib_bridge_set_fps(app: Any, fps: int) -> None:
    """Update the live single-viewer's animation framerate client-side (no
    rebuild, camera preserved) via ``window.__quantuiVibSetFps``."""
    bridge = getattr(app, "_vib_js_bridge", None)
    if bridge is None:
        return
    from IPython.display import Javascript, display

    # %-formatting is deliberate here — same JavaScript brace-density reason as
    # ``_vib_bridge_set_mode`` above.
    js = (
        "(function(){var n=0;function go(){n++;"  # noqa: UP031 — see above
        "if(window.__quantuiVibSetFps){window.__quantuiVibSetFps(%d);}"
        "else if(n<40){setTimeout(go,50);}}go();})();" % int(fps)
    )
    try:
        bridge.clear_output(wait=True)
        with bridge:
            display(Javascript(js))
    except Exception:
        pass


def show_pes_scan_result(app: Any, result: Any) -> bool:
    """Render PES energy profile chart and stash latest PES result."""
    app._last_pes_result = result
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        e_rel = result.energies_relative_kcal
        x_vals = result.scan_parameter_values

        hover_text = [
            f"{result.scan_coordinate_label}: {x:.4f}<br>"
            f"ΔE = {de:.3f} kcal/mol<br>"
            f"E = {e:.8f} Ha"
            for x, de, e in zip(x_vals, e_rel, result.energies_hartree)
        ]

        fig = go.Figure(
            go.Scatter(
                x=x_vals,
                y=e_rel,
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=8, color="#2563eb"),
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
        tc = app._plotly_theme_colors()
        fig.update_layout(
            xaxis_title=result.scan_coordinate_label,
            yaxis_title="Relative energy / kcal mol⁻¹",
            height=380,
            margin=dict(l=60, r=20, t=30, b=50),
            plot_bgcolor=tc["plot_bgcolor"],
            paper_bgcolor=tc["paper_bgcolor"],
            font=dict(color=tc["font_color"]),
            xaxis=dict(showgrid=True, gridcolor=tc["grid_color"]),
            yaxis=dict(showgrid=True, gridcolor=tc["grid_color"]),
            hovermode="closest",
        )
        app._last_pes_fig = fig
        app._set_html_output(
            app._pes_plot_html,
            pio.to_html(
                fig,
                include_plotlyjs="require",
                full_html=False,
                config={"responsive": True},
            ),
        )
    except Exception:
        app._last_pes_fig = None
        pass

    return True


def build_vib_export_html(app: Any, mode_number: int) -> tuple[str, str]:
    """Build a self-contained HTML string for the given vibrational mode.

    **py3Dmol only, as of 2026-08-04** (user decision). This used to prefer
    plotlymol3d on the reasoning that a self-contained Plotly animation with
    embedded controls is the canonical "export quality" artifact. That reasoning
    optimised the wrong thing: the file someone exports should be the animation
    they were just watching. py3Dmol renders the live vibrational viewer, so
    py3Dmol is what gets exported — same frame construction, same amplitude,
    same frame count, same fps as the on-screen viewer.

    Backend resolution stays preference-independent, and is now trivially so:
    ``VizTask.VIB_EXPORT`` is single-backend in the router. The Plotly export
    branch is removed rather than left unreachable, because unlike the Plotly
    isosurface path (kept, tested, one line from being restored) it duplicated
    animation-building logic that would rot silently behind a flag.

    Returns ``(backend_name, html_string)``.

    Raises ``ValueError`` when vib state is missing or py3Dmol is unavailable.
    """
    freq_result = getattr(app, "_last_vib_freq_result", None)
    molecule = getattr(app, "_last_vib_molecule", None)
    if freq_result is None or molecule is None:
        raise ValueError(
            "No vibrational data available — run a Frequency calculation "
            "and open the Vibrational panel first."
        )

    availability = getattr(app, "_viz_availability", None)
    if availability is None:
        raise ValueError("Visualization availability not initialised.")

    # The one and only export path. Mirrors _render_vib_mode_py3dmol's frame
    # construction but emits stand-alone HTML rather than swapping into
    # vib_output, so the file matches what was on screen.
    if availability.py3dmol:
        try:
            import numpy as np
            import py3Dmol  # noqa: F401 — probe; make_view imports it for the export
        except ImportError as exc:
            raise ValueError(f"py3Dmol unavailable for fallback export: {exc}") from exc

        try:
            displ = np.array(freq_result.displacements[mode_number - 1], dtype=float)
        except (AttributeError, IndexError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not read displacements for mode {mode_number}: {exc}"
            ) from exc

        atoms = list(molecule.atoms)
        base_coords = np.array(molecule.coordinates, dtype=float)
        if base_coords.shape != displ.shape:
            raise ValueError(
                f"Shape mismatch: base coords {base_coords.shape} vs "
                f"displacements {displ.shape}"
            )

        n_frames = 24
        amplitude = 0.4
        fps = int(
            getattr(
                getattr(app, "_user_settings", None) and app._user_settings.viz,
                "vib_framerate_fps",
                10,
            )
        )
        fps = max(1, fps)
        interval_ms = max(1, int(round(1000.0 / fps)))

        phases = np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
        n_atoms = len(atoms)
        xyz_lines: list[str] = []
        for phase in phases:
            coords = base_coords + amplitude * float(phase) * displ
            xyz_lines.append(f"{n_atoms}")
            xyz_lines.append(f"mode {mode_number} phase {float(phase):+.3f}")
            for sym, xyz in zip(atoms, coords):
                xyz_lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
        xyz_string = "\n".join(xyz_lines) + "\n"

        from quantui.viz_assets import make_view, standalone_html

        view = make_view(width=640, height=520)
        view.addModelsAsFrames(xyz_string, "xyz")
        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
        view.setBackgroundColor("white")
        view.zoomTo()
        view.animate({"loop": "forward", "interval": interval_ms, "reps": 0})
        # Exported HTML is opened outside the app (no page bootstrap), so embed
        # the offline 3Dmol.js loader inline to make the file self-contained.
        return ("py3dmol", standalone_html(view._make_html()))

    raise ValueError(
        "py3Dmol is unavailable, so the vibrational animation cannot be "
        "exported. py3Dmol is a required QuantUI dependency — reinstall with "
        "pip install --force-reinstall 'py3Dmol>=2,<3'."
    )
