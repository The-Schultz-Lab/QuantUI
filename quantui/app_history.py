"""History-loading helpers used by QuantUIApp."""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import HTML, display


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
                lambda _, d=data, rd=result_dir: app._history_load_results(d, rd)
            )
            btn_analysis.on_click(
                lambda _, rd=result_dir: app._history_load_analysis(rd)
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


def history_load_results(app: Any, data: dict[str, Any], result_dir: Path) -> None:
    """Display a history result card in the Results tab and navigate there."""
    app._last_result_dir = result_dir
    app.result_output.clear_output()
    with app.result_output:
        display(HTML(app._format_past_result(data, result_dir=result_dir)))
    app._result_dir_label.layout.display = "none"
    # Also show 3D structure if geometry is recoverable
    mol = app._mol_from_result_dir(result_dir, data)
    if mol is not None:
        app._show_result_3d(mol)
    app.root_tab.selected_index = 1


def history_load_analysis(app: Any, result_dir: Path) -> None:
    """Load analysis panels for a history result and navigate to Analysis tab."""
    app._last_result_dir = result_dir
    log_path = result_dir / "pyscf.log"
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else "(No pyscf.log found for this result.)"
    )
    app._update_log_panel(result_dir.name if log_path.exists() else "", text)
    app._show_result_log(result_dir, text)

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

    app.root_tab.selected_index = 2


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
