"""History-loading helpers used by QuantUIApp."""

from __future__ import annotations

import json as _json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import HTML, display


class _LoadTimer:
    """Per-stage timing collector for a history-load operation (HIST.2).

    Used as: open one ``_LoadTimer`` at the top of each loader, wrap each
    interesting sub-stage in ``with timer.stage("name"):``, then call
    ``timer.emit(status=...)`` exactly once (from the loader's ``finally``
    block). One ``history_load_timing`` event is appended to
    ``event_log.jsonl`` per load with the total elapsed time and a per-stage
    breakdown. The data drives the HIST.2 latency-optimization pass — until
    we know which stage dominates, we don't know which to optimize.

    Failures inside ``calc_log.log_event`` are swallowed: telemetry must
    never block the actual load.
    """

    def __init__(self, op_name: str, result_dir: Path) -> None:
        self.op_name = op_name
        self.result_dir = result_dir
        self._t0 = time.perf_counter()
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        s0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = round((time.perf_counter() - s0) * 1000.0, 2)

    def emit(self, status: str = "ok") -> None:
        total_ms = round((time.perf_counter() - self._t0) * 1000.0, 2)
        try:
            from quantui import calc_log as _clog

            stage_msg = " ".join(f"{k}={v}ms" for k, v in self._stages.items())
            _clog.log_event(
                "history_load_timing",
                f"{self.op_name} {self.result_dir.name} "
                f"total={total_ms}ms status={status} {stage_msg}".rstrip(),
                op=self.op_name,
                result_dir=self.result_dir.name,
                total_ms=total_ms,
                status=status,
                **{f"{k}_ms": v for k, v in self._stages.items()},
            )
        except Exception:
            pass


def on_past_dd_changed(app: Any, change: dict[str, Any], *, layout_fn: Any) -> None:
    """Handle history dropdown selection changes."""
    path_str = change["new"]
    # Hide result-specific panels whenever the selection changes so stale
    # content from a previous "View log" click doesn't persist.
    app._deactivate_all_ana_panels()
    app._pending_traj_result = None
    app._result_log_accordion.layout.display = "none"
    app._result_dir_label.layout.display = "none"
    app._iso_generate_btn.disabled = True
    if not path_str:
        app.past_output.clear_output()
        return
    app.past_output.clear_output()
    with app.past_output:
        try:
            from quantui import load_result

            result_dir = Path(path_str)
            data = load_result(result_dir)
            display(HTML(app._format_past_result(data, result_dir=result_dir)))
            btn_results = widgets.Button(
                description="-> View Results",
                button_style="success",
                layout=layout_fn(width="130px"),
                tooltip="Show this result in the Results tab",
            )
            btn_analysis = widgets.Button(
                description="-> View Analysis",
                button_style="info",
                layout=layout_fn(width="140px"),
                tooltip="Load analysis panels and navigate to the Analysis tab",
            )
            btn_results.on_click(
                lambda _, d=data, rd=result_dir, br=btn_results, ba=btn_analysis: (
                    app._history_load_results(d, rd, source_btns=(br, ba))
                )
            )
            btn_analysis.on_click(
                lambda _, rd=result_dir, br=btn_results, ba=btn_analysis: (
                    app._history_load_analysis(rd, source_btns=(br, ba))
                )
            )
            display(
                widgets.HBox(
                    [btn_results, btn_analysis],
                    layout=layout_fn(gap="8px", margin="6px 0 0"),
                )
            )
        except Exception as exc:
            print(f"Could not load result: {exc}")


def on_view_log(app: Any, btn: Any) -> None:
    """Handle View Log action for a selected history result."""
    path_str = app.past_dd.value
    if not path_str:
        return
    result_dir = Path(path_str)
    app._last_result_dir = result_dir
    try:
        import quantui.calc_log as _calc_log

        _calc_log.log_event(
            "history_view",
            result_dir.name,
            result_dir=result_dir.name,
            session_id=app._session_id,
        )
    except Exception:
        pass

    # Read log text and populate log panel
    log_path = result_dir / "pyscf.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        label = result_dir.name
    else:
        text = "(No pyscf.log found for this result.)"
        label = ""
    app._update_log_panel(text, label)
    app._show_result_log(result_dir, text)

    # Build analysis context from disk and apply via registry
    ctx = app._build_history_context(result_dir)
    if ctx is not None:
        data_stub = {"calc_type": ctx.calc_type, "spectra": ctx.spectra_data}
        try:
            mol = app._mol_from_result_dir(result_dir, data_stub)
            if mol is not None:
                app._show_result_3d(mol, extra_output=app._analysis_mol_output)
            else:
                app._analysis_mol_output.clear_output()
        except Exception:
            pass
        app._apply_analysis_context(ctx)

    app._goto_output_tab()


def mol_from_result_dir(result_dir: Path, data: dict[str, Any]) -> Any:
    """Try to reconstruct a displayable Molecule from a saved result directory.

    Returns a Molecule or None if geometry data is not available.
    Tries sources in order: frequency spectra -> orbitals_meta -> trajectory.
    """
    from quantui.molecule import Molecule

    calc_type = data.get("calc_type", "")

    # Frequency: geometry stored inside spectra.molecule
    if calc_type == "frequency":
        mol_data = data.get("spectra", {}).get("molecule", {})
        if mol_data.get("atoms") and mol_data.get("coords"):
            try:
                return Molecule(
                    atoms=mol_data["atoms"],
                    coordinates=mol_data["coords"],
                    charge=mol_data.get("charge", 0),
                    multiplicity=mol_data.get("multiplicity", 1),
                )
            except Exception:
                pass

    # Single point / Geo opt: atom list from orbitals_meta.json
    meta_path = result_dir / "orbitals_meta.json"
    if meta_path.exists():
        try:
            meta = _json.loads(meta_path.read_text())
            mol_atom = meta.get("mol_atom")
            if mol_atom:
                atoms = [sym for sym, _ in mol_atom]
                coords = [coords for _, coords in mol_atom]
                return Molecule(atoms=atoms, coordinates=coords)
        except Exception:
            pass

    # Geo opt fallback: last step of trajectory.json
    if calc_type == "geometry_opt":
        traj_path = result_dir / "trajectory.json"
        if traj_path.exists():
            try:
                traj_data = _json.loads(traj_path.read_text())
                steps = traj_data.get("steps", [])
                if steps:
                    return Molecule(
                        atoms=traj_data["atoms"],
                        coordinates=steps[-1]["coords"],
                        charge=traj_data.get("charge", 0),
                        multiplicity=traj_data.get("multiplicity", 1),
                    )
            except Exception:
                pass

    return None


def _begin_history_load(app: Any, message: str, source_btns: tuple) -> None:
    """Show immediate feedback when a history-load action starts (HIST.1).

    Lights the toolbar activity indicator and disables the source buttons so
    a second click can't fire a parallel load. Both actions are best-effort —
    failure to update a button (e.g. it was already destroyed) must not block
    the actual load.
    """
    for btn in source_btns:
        try:
            btn.disabled = True
        except Exception:
            pass
    try:
        app._activity_begin(message, kind="ui")
    except Exception:
        pass


def _end_history_load(app: Any, source_btns: tuple) -> None:
    """Restore UI state after a history-load action finishes (HIST.1).

    Mirrors :func:`_begin_history_load`. Called from the loader's ``finally``
    block so the activity indicator + buttons are always restored, even when
    the load raises.
    """
    try:
        app._activity_end(kind="ui")
    except Exception:
        pass
    for btn in source_btns:
        try:
            btn.disabled = False
        except Exception:
            pass


def history_load_results(
    app: Any,
    data: dict[str, Any],
    result_dir: Path,
    *,
    source_btns: tuple = (),
) -> None:
    """Display a history result card in the Results tab and navigate there.

    ``source_btns`` is an optional tuple of button widgets to disable while
    the load is in flight (HIST.1 immediate-loading-feedback contract). Tests
    and callers that don't have a button reference can omit it.

    Stage timings are emitted as a single ``history_load_timing`` event on
    completion (HIST.2 — drives latency-optimization decisions).
    """
    _begin_history_load(app, "Loading history result…", source_btns)
    timer = _LoadTimer("history_load_results", result_dir)
    status = "ok"
    try:
        app._last_result_dir = result_dir
        with timer.stage("format_result_html"):
            app.result_output.clear_output()
            with app.result_output:
                display(HTML(app._format_past_result(data, result_dir=result_dir)))
            app._result_dir_label.layout.display = "none"
        with timer.stage("mol_reconstruction"):
            mol = app._mol_from_result_dir(result_dir, data)
        if mol is not None:
            with timer.stage("show_result_3d"):
                app._show_result_3d(mol)
        with timer.stage("nav_tab"):
            app.root_tab.selected_index = 1
    except Exception:
        status = "error"
        raise
    finally:
        timer.emit(status=status)
        _end_history_load(app, source_btns)


def history_load_analysis(
    app: Any,
    result_dir: Path,
    *,
    source_btns: tuple = (),
) -> None:
    """Load analysis panels for a history result and navigate to Analysis tab.

    ``source_btns`` is an optional tuple of button widgets to disable while
    the load is in flight (HIST.1 immediate-loading-feedback contract). Tests
    and callers that don't have a button reference can omit it.

    Stage timings are emitted as a single ``history_load_timing`` event on
    completion (HIST.2 — drives latency-optimization decisions). Stages cover
    the four expected hotspots: pyscf.log read, context build, molecule
    reconstruction, 3D viewer render, and the analysis-context registry walk.
    """
    _begin_history_load(app, "Loading analysis from history…", source_btns)
    timer = _LoadTimer("history_load_analysis", result_dir)
    status = "ok"
    try:
        app._last_result_dir = result_dir
        with timer.stage("read_pyscf_log"):
            log_path = result_dir / "pyscf.log"
            text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else "(No pyscf.log found for this result.)"
            )
        with timer.stage("update_log_panel"):
            app._update_log_panel(result_dir.name if log_path.exists() else "", text)
            app._show_result_log(result_dir, text)

        with timer.stage("build_context"):
            ctx = app._build_history_context(result_dir)
        if ctx is not None:
            data_stub = {"calc_type": ctx.calc_type, "spectra": ctx.spectra_data}
            with timer.stage("mol_reconstruction"):
                try:
                    mol = app._mol_from_result_dir(result_dir, data_stub)
                except Exception:
                    mol = None
            with timer.stage("show_result_3d"):
                try:
                    if mol is not None:
                        app._show_result_3d(mol, extra_output=app._analysis_mol_output)
                    else:
                        app._analysis_mol_output.clear_output()
                except Exception:
                    pass
            with timer.stage("apply_analysis_context"):
                app._apply_analysis_context(ctx)

        with timer.stage("nav_tab"):
            app.root_tab.selected_index = 2
    except Exception:
        status = "error"
        raise
    finally:
        timer.emit(status=status)
        _end_history_load(app, source_btns)


def build_history_context(result_dir: Path, *, context_cls: Any) -> Optional[Any]:
    """Load result.json from result_dir and return an analysis context."""
    try:
        from quantui import load_result

        data = load_result(result_dir)
    except Exception:
        return None
    return context_cls(
        calc_type=data.get("calc_type", ""),
        formula=data.get("formula", result_dir.name),
        method=data.get("method", ""),
        basis=data.get("basis", ""),
        result_dir=result_dir,
        spectra_data=data.get("spectra", {}),
        timestamp=data.get("timestamp", ""),
        source="history",
    )
