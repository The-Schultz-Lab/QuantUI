"""Visualization and rendering helpers used by QuantUIApp."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List

import ipywidgets as widgets
from IPython.display import HTML, display


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
    display_molecule_fn: Any,
) -> None:
    """Render molecule 3D structure in result and optional extra output panels.

    Backend selection goes through ``app._resolve_backend(task)`` per-output:

    - ``result_viz_output`` uses ``VizTask.STRUCTURE_VIEW_RESULTS``.
    - ``extra_output == _analysis_mol_output`` uses ``ANALYSIS_STRUCTURE_VIEW``.
    - Any other extra_output uses ``STRUCTURE_VIEW_RESULTS`` as a safe default.
    """
    if display_molecule_fn is None or molecule is None:
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
                app.result_viz_output.clear_output()
                with app.result_viz_output:
                    display_molecule_fn(
                        molecule,
                        backend=str(chosen),
                        style=app._viz_style,
                        lighting=app._viz_lighting,
                        bgcolor=app._plotly_theme_colors()["scene_bgcolor"],
                    )

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
                extra_output.clear_output()
                with extra_output:
                    display_molecule_fn(
                        molecule,
                        backend=str(chosen),
                        style=app._viz_style,
                        lighting=app._viz_lighting,
                        bgcolor=app._plotly_theme_colors()["scene_bgcolor"],
                    )
            if is_analysis_output:
                app._update_analysis_backend_label(chosen)

    # Track the molecule currently shown in the Analysis-tab viewer so the
    # preference-change re-render path can find it.
    if is_analysis_output:
        app._analysis_displayed_molecule = molecule


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
    """Build trajectory carousel and energy chart in trajectory panel."""
    import concurrent.futures

    def _is_stale() -> bool:
        return render_token is not None and render_token != int(
            getattr(app, "_traj_render_token", 0)
        )

    def _set_cache_label(value: str) -> None:
        if _is_stale():
            return
        cache_label.value = value

    def _swap_frame_out(html_str: str) -> None:
        """Atomically replace frame_out's content in a single widget-state
        update so the browser never sees an intermediate empty state.
        Combined with the fixed `height` on frame_out, this prevents the
        layout-flash that otherwise happens between clear+append on every
        frame switch (visible as a page-scroll jump in the previous build)."""
        frame_out.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/html": html_str},
                "metadata": {},
            },
        )

    def _show_frame_error(message: str) -> None:
        if _is_stale():
            return
        _swap_frame_out(
            f'<p style="color:#b91c1c;padding:8px">Frame render failed: {message}</p>'
        )

    # Support both OptimizationResult (.trajectory) and PESScanResult (.coordinates_list)
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

    # --- Pre-build XYZ blocks (reused by carousel, fast path, and export) ---
    charge = traj[0].charge
    xyzblocks = [
        f"{len(m.atoms)}\n{m.get_formula()}\n{m.to_xyz_string()}" for m in traj
    ]
    frame_w, frame_h, frame_res = 460, 340, 8

    # --- Attempt fast-path: bond perception once on frame 0 ---
    ref_mol = None
    plotlymol_fast = False
    try:
        from plotlymol3d import (
            draw_3D_mol as _draw_3D_mol,
        )
        from plotlymol3d import (
            format_figure as _fmt_fig,
        )
        from plotlymol3d import (
            format_lighting as _fmt_light,
        )
        from plotlymol3d import (
            make_subplots as _make_subplots,
        )
        from plotlymol3d import (
            xyzblock_to_rdkitmol as _xyz_to_rdkit,
        )
        from rdkit import Chem as _Chem

        from quantui.visualization_py3dmol import LIGHTING_PRESETS as _LP

        ref_mol = _xyz_to_rdkit(xyzblocks[0], charge=charge)
        plotlymol_fast = ref_mol is not None
    except Exception:
        pass

    def _build_fig_fast(idx: int):
        """Reuse frame-0 bond topology; only swap in new atom positions."""
        mol_xyz = _Chem.MolFromXYZBlock(xyzblocks[idx] + "\n")
        if mol_xyz is None:
            return None
        rw = _Chem.RWMol(ref_mol)
        conf_src = mol_xyz.GetConformer()
        conf_dst = rw.GetConformer()
        for atom_idx in range(rw.GetNumAtoms()):
            conf_dst.SetAtomPosition(atom_idx, conf_src.GetAtomPosition(atom_idx))
        fig = _make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
        _draw_3D_mol(fig, rw.GetMol(), frame_res, "ball+stick")
        fig = _fmt_fig(fig)
        fig = _fmt_light(fig, **_LP.get("soft", _LP["soft"]))
        scene_bg = app._plotly_theme_colors()["scene_bgcolor"]
        fig.update_layout(
            width=frame_w,
            height=frame_h,
            paper_bgcolor="white",
            scene=dict(bgcolor=scene_bg),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    def _try_py3dmol(idx: int):
        """Build frame idx with py3Dmol. Returns (kind, obj) or None."""
        try:
            import py3Dmol as _p3d

            view = _p3d.view(width=frame_w, height=frame_h)
            view.addModel(xyzblocks[idx], "xyz")
            view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
            view.setBackgroundColor(
                "white" if app.theme_btn.value == "Light" else "#1e1e1e"
            )
            view.zoomTo()
            return ("py3dmol", view)
        except Exception:
            return None

    def _try_plotlymol(idx: int):
        """Build frame idx with plotlymol3d. Tries fast bond-cached path
        first, falls back to slow path. Returns (kind, obj) or None."""
        if plotlymol_fast:
            try:
                fig = _build_fig_fast(idx)
                if fig is not None:
                    return ("plotly", fig)
            except Exception:
                pass
        try:
            from quantui.visualization_py3dmol import visualize_molecule_plotlymol

            fig = visualize_molecule_plotlymol(
                traj[idx],
                mode="ball+stick",
                resolution=frame_res,
                width=frame_w,
                height=frame_h,
            )
            scene_bg = app._plotly_theme_colors()["scene_bgcolor"]
            fig.update_layout(paper_bgcolor="white", scene=dict(bgcolor=scene_bg))
            return ("plotly", fig)
        except ImportError:
            return None

    def _build_fig(idx: int):
        """Return (kind, obj) for frame idx. Trajectory frame rendering is
        py3Dmol-only per the routing policy: plotlymol is blocked from
        real-time trajectory use to avoid its RequireJS flicker pattern.
        If py3Dmol is unavailable on this host, returns an error frame
        rather than silently falling back to plotlymol."""
        from quantui.viz_backend_router import VizBackend as _VB
        from quantui.viz_backend_router import VizTask as _VT

        chosen = app._resolve_backend(_VT.TRAJECTORY_FRAME)
        if chosen != _VB.PY3DMOL:
            return (
                "error",
                "Trajectory rendering requires py3Dmol (plotlymol blocked "
                "for real-time use to avoid flicker). py3Dmol is unavailable.",
            )
        with _viz_render_event(
            app, task="trajectory_frame", backend="py3dmol", idx=idx
        ):
            result = _try_py3dmol(idx)
        if result is not None:
            return result
        return ("error", "py3Dmol failed to build trajectory frame")

    frame_cache: dict[int, Any] = {}

    # --- Carousel controls ---
    step_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n - 1,
        description="Step:",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=layout_fn(width="360px"),
    )
    step_info = widgets.HTML(value=app._traj_step_html(0, traj, energies, rel_e))
    # Fixed height (not just min_height) so the container box never resizes
    # between frame swaps — eliminates the layout flash / page-scroll jump
    # the user reported on each arrow/slider click.
    frame_out = widgets.Output(
        layout=layout_fn(height=f"{frame_h}px", width=f"{frame_w}px")
    )
    cache_label = widgets.HTML(
        value=f'<span style="color:#888;font-size:11px;font-style:italic">'
        f"Pre-rendering frames… 0 / {n}</span>"
    )

    def _display_frame(idx: int) -> None:
        if _is_stale():
            return
        kind, obj = frame_cache[idx]
        try:
            from quantui import calc_log as _clog_df

            _clog_df.log_event("traj_frame_display", f"idx={idx} kind={kind}")
        except Exception:
            pass
        if kind == "error":
            _swap_frame_out(
                f'<p style="color:#b91c1c;padding:8px">'
                f"Frame render failed: {obj}</p>"
            )
            return
        if kind == "plotly":
            # Render via Plotly HTML serialization. The atomic outputs swap
            # avoids the brief empty state between clear+append, eliminating
            # the layout-flash visible on rapid frame switches.
            import plotly.io as _pio

            _swap_frame_out(
                _pio.to_html(
                    obj,
                    full_html=False,
                    include_plotlyjs="require",
                    config={"responsive": True},
                )
            )
            return
        # py3Dmol view object — convert to its HTML repr and atomic-swap.
        make_html = getattr(obj, "_make_html", None)
        if callable(make_html):
            try:
                _swap_frame_out(obj._make_html())
                return
            except Exception as exc:
                _swap_frame_out(
                    f'<p style="color:#b91c1c;padding:8px">'
                    f"py3Dmol render failed: {exc}</p>"
                )
                return
        _swap_frame_out(
            '<p style="color:#b91c1c;padding:8px">'
            "Frame object missing HTML representation</p>"
        )

    def _update_frame(change: dict[str, Any]) -> None:
        if _is_stale():
            return
        idx = change["new"]
        step_info.value = app._traj_step_html(idx, traj, energies, rel_e)
        if idx in frame_cache:
            _display_frame(idx)
            return
        _swap_frame_out(
            '<p style="color:#555;font-style:italic;padding:8px">Rendering…</p>'
        )

        def _on_demand() -> None:
            try:
                frame_cache[idx] = _build_fig(idx)
                app._queue_main_thread_callback(_display_frame, idx)
            except Exception as exc:
                if _is_stale():
                    return
                app._queue_main_thread_callback(_show_frame_error, str(exc))

        threading.Thread(target=_on_demand, daemon=True).start()

    step_slider.observe(app._safe_cb(_update_frame), names="value")

    # --- Prev/next arrow buttons for one-step navigation ---
    prev_btn = widgets.Button(
        icon="arrow-left",
        tooltip="Previous frame",
        layout=layout_fn(width="40px", margin="0 4px 0 0"),
        disabled=True,  # starts at frame 0
    )
    next_btn = widgets.Button(
        icon="arrow-right",
        tooltip="Next frame",
        layout=layout_fn(width="40px", margin="0 8px 0 4px"),
        disabled=(n <= 1),
    )

    def _on_prev_clicked(_btn) -> None:
        if step_slider.value > 0:
            step_slider.value -= 1

    def _on_next_clicked(_btn) -> None:
        if step_slider.value < n - 1:
            step_slider.value += 1

    prev_btn.on_click(_on_prev_clicked)
    next_btn.on_click(_on_next_clicked)

    def _update_nav_buttons(change: dict[str, Any]) -> None:
        idx = change["new"]
        prev_btn.disabled = idx <= 0
        next_btn.disabled = idx >= n - 1

    step_slider.observe(app._safe_cb(_update_nav_buttons), names="value")

    # --- Export button ---
    export_btn = widgets.Button(
        description="Export Animation",
        icon="download",
        layout=layout_fn(width="160px", margin="0 0 0 12px"),
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
                from plotlymol3d import create_trajectory_animation

                anim_fig = create_trajectory_animation(
                    xyzblocks=xyzblocks,
                    energies_hartree=energies if energies else None,
                    charge=charge,
                    mode="ball+stick",
                    resolution=12,
                    title=f"Geo Opt: {opt_result.formula}",
                )
                result_dir = getattr(app, "_last_result_dir", None)
                out_path = (
                    result_dir / "trajectory_animation.html"
                    if result_dir is not None
                    else Path.home() / f"{opt_result.formula}_trajectory.html"
                )
                anim_fig.write_html(str(out_path))
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

    # --- Assemble layout ---
    header = widgets.HBox(
        [prev_btn, step_slider, next_btn, export_btn],
        layout=layout_fn(align_items="center", margin="4px 0"),
    )
    panel = widgets.VBox([header, step_info, cache_label, frame_out, export_status])

    # Build and render frame 0 SYNCHRONOUSLY on the main thread before
    # displaying the panel, so the Output widget arrives at the browser with
    # frame 0 already in its outputs list. This avoids the io_loop-callback
    # latency that left frame 0 invisible until the first slider click.
    if _is_stale():
        return
    try:
        frame_cache[0] = _build_fig(0)
        _display_frame(0)
        sync_frame0_ok = True
    except Exception as _f0_exc:
        sync_frame0_ok = False
        try:
            from quantui import calc_log as _clog_f0

            _clog_f0.log_event(
                "traj_frame0_sync_error",
                f"{type(_f0_exc).__name__}: {_f0_exc}"[:300],
            )
        except Exception:
            pass
        _swap_frame_out(
            '<p style="color:#555;font-style:italic;padding:8px">'
            "Rendering frame 0…</p>"
        )

    # Display panel.
    if _is_stale():
        return
    # Build the energy figure as HTML inside an Output widget so RequireJS
    # / Plotly scripts execute, and put the panel widget directly as a
    # sibling child of traj_output. Setting traj_output.children atomically
    # avoids the deferred-display-via-Output issue that was emptying the
    # accordion in BUG-FRESH-TRAJ.
    new_children = []
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
    new_children.append(panel)
    app.traj_output.children = tuple(new_children)
    try:
        from quantui import calc_log as _clog_sp

        _clog_sp.log_event(
            "traj_show_panel",
            f"n={n} plotlymol_fast={plotlymol_fast} "
            f"sync_frame0_ok={sync_frame0_ok} "
            f"traj_children_n={len(getattr(app.traj_output, 'children', ()))}",
        )
    except Exception:
        pass

    def _prerender_all() -> None:
        """Render remaining frames in a background thread (frame 0 already
        built+displayed synchronously above when sync_frame0_ok)."""
        if _is_stale():
            return
        try:
            if 0 not in frame_cache:
                frame_cache[0] = _build_fig(0)
                app._queue_main_thread_callback(_display_frame, 0)
            app._queue_main_thread_callback(
                _set_cache_label,
                f'<span style="color:#888;font-size:11px;font-style:italic">'
                f"Pre-rendering frames… 1 / {n}</span>",
            )
            if n > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {pool.submit(_build_fig, i): i for i in range(1, n)}
                    done = 1
                    for fut in concurrent.futures.as_completed(futures):
                        if _is_stale():
                            return
                        i = futures[fut]
                        try:
                            frame_cache[i] = fut.result()
                        except Exception:
                            pass
                        done += 1
                        app._queue_main_thread_callback(
                            _set_cache_label,
                            f'<span style="color:#888;font-size:11px;font-style:italic">'
                            f"Pre-rendering frames… {done} / {n}</span>",
                        )
        except Exception:
            pass
        app._queue_main_thread_callback(
            _set_cache_label,
            f'<span style="color:#16a34a;font-size:11px">'
            f"✓ All {n} frames ready</span>",
        )
        try:
            from quantui import calc_log as _clog_pre

            _clog_pre.log_event(
                "traj_prerender_complete", f"n={n} cached={len(frame_cache)}"
            )
        except Exception:
            pass

    threading.Thread(target=_prerender_all, daemon=True).start()


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
    except ImportError:
        pass

    # Fallback: py3Dmol
    try:
        import py3Dmol as _p3d

        xyz = (
            f"{len(molecule.atoms)}\n"
            f"{molecule.get_formula()}\n"
            f"{molecule.to_xyz_string()}"
        )
        view = _p3d.view(width=460, height=340)
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
    displacements directly from ``freq_result`` (VIZBACK.8).
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

    app.vib_mode_dd.options = options
    app.vib_mode_dd.value = options[0][1]

    app._last_vib_data = vib_data  # may be None — plotlymol3d optional
    app._last_vib_molecule = molecule
    app._last_vib_freq_result = freq_result

    first_label, first_mode = options[0]
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

        if mode_norm == "broadened":
            gamma = max(float(fwhm), 1.0) / 2.0
            x_min = max(100.0, min(wl) - 80.0)
            x_max = max(wl) + 80.0
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
        fig.update_xaxes(showgrid=True, gridcolor=tc["grid_color"], zeroline=False)
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
        app._orb_iso_output.clear_output()
        app._orb_toggle.value = "HOMO"
        app._orb_iso_controls.layout.display = ""
        app._iso_generate_btn.disabled = False
    else:
        app._orb_iso_controls.layout.display = "none"
        app._iso_generate_btn.disabled = True

    return True


def on_iso_generate(app: Any, btn: Any) -> None:
    """Generate orbital isosurface for currently selected orbital."""
    orbital_label = app._orb_toggle.value
    app._iso_render_token = int(getattr(app, "_iso_render_token", 0)) + 1
    render_token = app._iso_render_token
    btn.disabled = True
    btn.description = "Generating…"
    try:
        from quantui import calc_log as _clog

        _clog.log_event("iso_render_start", orbital_label)
    except Exception:
        pass
    app._orb_iso_output.clear_output()
    with app._orb_iso_output:
        display(
            HTML(
                f'<p style="color:#555;font-style:italic;padding:4px 0">'
                f"⏳ Generating {orbital_label} cube file and rendering isosurface"
                f" — this may take 15–30 s…</p>"
            )
        )

    done = threading.Event()

    def _reset_button() -> None:
        if render_token != int(getattr(app, "_iso_render_token", 0)):
            return
        btn.disabled = False
        btn.description = "Generate Isosurface"

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
            if render_token != int(getattr(app, "_iso_render_token", 0)):
                return
            try:
                from quantui import calc_log as _clog

                _clog.log_event("iso_render_timeout", orbital_label)
            except Exception:
                pass
            btn.disabled = False
            btn.description = "Generate Isosurface"
            app._orb_iso_output.clear_output()
            with app._orb_iso_output:
                display(
                    HTML(
                        '<p style="color:#b91c1c;padding:8px">'
                        "⚠ Orbital isosurface timed out after 180 s. "
                        "Try a smaller basis set or a smaller molecule.</p>"
                    )
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
    if orb_idx is None or orb_idx < 0 or orb_idx >= n_total:
        return

    mo_coeff = getattr(app, "_last_orb_mo_coeff", None)
    mol_atom = getattr(app, "_last_orb_mol_atom", None)
    mol_basis = getattr(app, "_last_orb_mol_basis", None)
    if mo_coeff is None or mol_atom is None or mol_basis is None:
        return

    try:
        import plotly.io as _pio

        from quantui.orbital_visualization import (
            generate_cube_from_arrays,
            plot_cube_isosurface,
        )

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

        generate_cube_from_arrays(mol_atom, mol_basis, mo_coeff, orb_idx, cube_path)
        is_dark = app.theme_btn.value == "Dark"
        axis_color = "#dbeafe" if is_dark else "#1f2937"
        bond_color = "#cbd5e1" if is_dark else "#4b5563"
        title_color = app._plotly_theme_colors()["font_color"]
        fig = plot_cube_isosurface(
            cube_path,
            title=f"{orbital_label} Isosurface",
            show_molecule=True,
            show_grid=False,
            scene_bgcolor=app._plotly_theme_colors()["scene_bgcolor"],
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
            app._orb_iso_output.clear_output()
            with app._orb_iso_output:
                display(
                    HTML(
                        f'<p style="color:#b91c1c;padding:8px">'
                        f"⚠ Orbital isosurface failed: {msg}</p>"
                    )
                )

        app._queue_main_thread_callback(_show_err)
        return
    if _is_stale():
        return
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

    Per VIZBACK.8 spec: pure-numpy frame generation (no plotlymol3d
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
        import py3Dmol
    except ImportError as exc:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, f"py3Dmol unavailable: {exc}")
        return

    freq_result = getattr(app, "_last_vib_freq_result", None)
    if freq_result is None:
        if not _is_vib_stale(app, render_token):
            _vib_err(app, "No frequency result cached for vibrational animation.")
        return

    # Cache hit short-circuit (VIZBACK.9). The cache key now includes ``fps``
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
        interval_ms = max(1, int(round(1000.0 / fps)))
        view = py3Dmol.view(width=460, height=420)
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
        html_str = _VIB_CAMERA_PERSISTENCE_JS + view._make_html()
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
    # this mode instantly (VIZBACK.9). Non-fatal on failure — render still
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
        anim_fig.update_layout(height=420)
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
    _swap_vib_output(app, anim_html)


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
