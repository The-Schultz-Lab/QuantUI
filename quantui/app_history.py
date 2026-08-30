"""History-loading helpers used by QuantUIApp."""

from __future__ import annotations

import datetime as _dt
import json as _json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import HTML, display

from . import theme as _theme

# ══ HISTORY SEARCH / FACETED FILTER (HIST.7) ══════════════════════════════
#
# The History browser caches the parsed ``result.json`` of every saved calc as
# a list of entry dicts on ``app._history_entries`` (built in
# ``refresh_results_browser``). Filtering re-narrows that in-memory list — no
# per-keystroke disk access — and repopulates ``past_dd`` client-side.

# Calc-type facet chips, in display order: (badge label, canonical calc_type key).
# Mirrors ``_calc_type_badge`` in app_runflow so chips read like the dropdown labels.
HISTORY_CALC_TYPE_FACETS = [
    ("SP", "single_point"),
    ("GeoOpt", "geometry_opt"),
    ("Freq", "frequency"),
    ("UV-Vis", "tddft"),
    ("NMR", "nmr"),
    ("PES", "pes_scan"),
    ("Reorg", "reorganization_energy"),
]

# Status facet chips: (label, key).
HISTORY_STATUS_FACETS = [
    ("Converged", "converged"),
    ("Not converged", "not_converged"),
]


def entry_date(timestamp: Any) -> Optional[_dt.date]:
    """Parse a result timestamp (``YYYY-MM-DD_...``) into a ``date``, or None."""
    try:
        return _dt.date.fromisoformat(str(timestamp)[:10])
    except (ValueError, TypeError):
        return None


def filter_history_entries(
    entries: list[dict[str, Any]],
    *,
    text: str = "",
    name_formulas: Optional[Any] = None,
    calc_types: Optional[Any] = None,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    date_from: Optional[_dt.date] = None,
    date_to: Optional[_dt.date] = None,
    statuses: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Return the subset of *entries* matching every active facet.

    Pure function — no widget or disk access, so it is unit-testable in
    isolation. *entries* is the list of dicts built by
    ``refresh_results_browser`` (keys: ``formula``, ``name``, ``calc_type``,
    ``method``, ``basis``, ``timestamp``, ``date``, ``converged`` ...).

    An empty / falsy facet means "no constraint" for that facet:

    - ``text``: case-insensitive substring match against formula + molecule name.
    - ``name_formulas``: formulas resolved from *text* via the molecule library
      (so a chemical-name query like "benzene" also matches ``C6H6`` results).
      An entry passes the text facet if it matches *either* the substring *or*
      one of these formulas. The DB lookup lives in the caller
      (``apply_history_filter``) to keep this function pure.
    - ``calc_types``: iterable of canonical ``calc_type`` keys; entry passes if
      its ``calc_type`` is in the set.
    - ``method`` / ``basis``: exact match when truthy.
    - ``date_from`` / ``date_to``: inclusive ``date`` bounds on the entry's
      parsed timestamp date (entries with an unparseable date are excluded once
      any bound is set).
    - ``statuses``: subset of ``{"converged", "not_converged"}``.
    """
    needle = (text or "").strip().lower()
    formula_set = {str(f).lower() for f in (name_formulas or [])}
    ct_set = set(calc_types) if calc_types else None
    status_set = set(statuses) if statuses else None
    out: list[dict[str, Any]] = []
    for e in entries:
        if needle:
            hay = f"{e.get('formula', '')} {e.get('name', '')}".lower()
            matched = needle in hay
            if not matched and formula_set:
                matched = str(e.get("formula", "")).lower() in formula_set
            if not matched:
                continue
        if ct_set is not None and e.get("calc_type") not in ct_set:
            continue
        if method and e.get("method") != method:
            continue
        if basis and e.get("basis") != basis:
            continue
        if date_from or date_to:
            d = e.get("date") or entry_date(e.get("timestamp", ""))
            if d is None:
                continue
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
        if status_set is not None:
            key = "converged" if e.get("converged") else "not_converged"
            if key not in status_set:
                continue
        out.append(e)
    return out


def resolve_query_formulas(query: str) -> set[str]:
    """Map a free-text chemical-name query to the set of formulas it names, so
    a search like "benzene" also finds ``C6H6`` results.

    Two precise sources, unioned. Neither matches on synonyms — which would
    wrongly pull "methylbenzene"=toluene into a "benzene" search:

    1. The curated ``config.COMMON_NAME_TO_FORMULA`` map — covers the simple
       molecules the bundled library names by formula (benzene, water, …).
    2. Exact library-*name* matches — covers named organics the library carries
       properly (toluene, aspirin, caffeine). Exact (not substring) so
       "benzene" can't drag in "ethylbenzene"/"nitrobenzene" etc.

    Returns an empty set for a blank query, or if nothing resolves. Failures
    (missing library) are swallowed so search never breaks.
    """
    q = (query or "").strip().lower()
    if not q:
        return set()
    formulas: set[str] = set()
    try:
        from quantui import config

        # Curated names use substring so type-to-narrow works within the map
        # (e.g. "hydrogen" surfaces H2 + the hydrogen-X molecules as you type).
        for name, formula in config.COMMON_NAME_TO_FORMULA.items():
            if q in name:
                formulas.add(formula)
    except Exception:
        pass
    try:
        from quantui import molecule_library as _ml

        # Library: exact entry-*name* match only. Substring/synonym matching
        # would wrongly pull derivatives ("ethylbenzene", "methylbenzene")
        # into a "benzene" search.
        for r in _ml.search(q, limit=200):
            if r.get("formula") and str(r.get("name", "")).lower() == q:
                formulas.add(r["formula"])
    except Exception:
        pass
    return formulas


def refresh_history_facet_options(app: Any, entries: list[dict[str, Any]]) -> None:
    """Repopulate the Method / Basis facet dropdowns from the distinct values
    present in *entries*, preserving the current selection when it survives."""
    methods = sorted(
        {e["method"] for e in entries if e.get("method") and e["method"] != "?"}
    )
    bases = sorted(
        {e["basis"] for e in entries if e.get("basis") and e["basis"] != "?"}
    )
    method_dd = getattr(app, "history_method_dd", None)
    if method_dd is not None:
        cur = method_dd.value
        method_dd.options = [("Any method", "")] + [(m, m) for m in methods]
        method_dd.value = cur if cur in methods else ""
    basis_dd = getattr(app, "history_basis_dd", None)
    if basis_dd is not None:
        cur = basis_dd.value
        basis_dd.options = [("Any basis", "")] + [(b, b) for b in bases]
        basis_dd.value = cur if cur in bases else ""


def apply_history_filter(app: Any) -> None:
    """Re-narrow the cached history entries by the current facet-widget state
    and repopulate ``past_dd`` — no disk access.

    Preserves the index-0 placeholder and (via ipywidgets value-preservation)
    the current selection when it survives the filter. Shows an explicit
    "no matches" option when every entry is filtered out.
    """
    if getattr(app, "_history_filter_suspend", False):
        return
    entries = getattr(app, "_history_entries", None)
    if not entries:
        # Nothing cached yet (pre-scan) — leave whatever placeholder is set.
        return
    calc_types = [
        key for key, btn in getattr(app, "_history_calc_chips", {}).items() if btn.value
    ]
    statuses = [
        key
        for key, btn in getattr(app, "_history_status_chips", {}).items()
        if btn.value
    ]
    text = getattr(getattr(app, "history_search", None), "value", "") or ""
    matches = filter_history_entries(
        entries,
        text=text,
        name_formulas=resolve_query_formulas(text),
        calc_types=calc_types or None,
        method=getattr(getattr(app, "history_method_dd", None), "value", "") or None,
        basis=getattr(getattr(app, "history_basis_dd", None), "value", "") or None,
        date_from=getattr(getattr(app, "history_date_from", None), "value", None),
        date_to=getattr(getattr(app, "history_date_to", None), "value", None),
        statuses=statuses or None,
    )
    placeholder = ("(select a calculation to view)", "")
    if matches:
        app.past_dd.options = [placeholder] + [(e["label"], e["path"]) for e in matches]
    else:
        app.past_dd.options = [placeholder, ("(no matches for current filters)", "")]
    count_lbl = getattr(app, "history_count_lbl", None)
    if count_lbl is not None:
        count_lbl.value = (
            f'<span style="color:{_theme.css.TEXT_FAINT};font-size:12px">'
            f"{len(matches)} of {len(entries)} shown</span>"
        )


class _LoadTimer:
    """Per-stage timing collector for a history-load operation.

    Used as: open one ``_LoadTimer`` at the top of each loader, wrap each
    interesting sub-stage in ``with timer.stage("name"):``, then call
    ``timer.emit(status=...)`` exactly once (from the loader's ``finally``
    block). One ``history_load_timing`` event is appended to
    ``event_log.jsonl`` per load with the total elapsed time and a per-stage
    breakdown. The data drives the latency-optimization pass — until
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
        app._export_bundle_btn.disabled = False
    except Exception:
        pass
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
    Tries sources in order: result.json geometry -> frequency spectra ->
    orbitals_meta -> trajectory.
    """
    from quantui.molecule import Molecule
    from quantui.results_storage import molecule_from_geometry_payload

    calc_type = data.get("calc_type", "")

    geom = data.get("geometry")
    if isinstance(geom, dict) and geom.get("atoms") and geom.get("coordinates"):
        try:
            return molecule_from_geometry_payload(geom)
        except Exception:
            pass

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
    """Show immediate feedback when a history-load action starts.

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
    """Restore UI state after a history-load action finishes.

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
    the load is in flight (immediate-loading-feedback contract). Tests
    and callers that don't have a button reference can omit it.

    Stage timings are emitted as a single ``history_load_timing`` event on
    completion (drives latency-optimization decisions).
    """
    _begin_history_load(app, "Loading history result…", source_btns)
    timer = _LoadTimer("history_load_results", result_dir)
    status = "ok"
    try:
        app._last_result_dir = result_dir
        try:
            app._export_bundle_btn.disabled = False
        except Exception:
            pass
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
        # Also populate the Analysis tab so the two tabs stay in sync.
        # Without this, clicking "View Results" left Analysis showing the
        # previously-loaded calc (or empty panels), which surprised users
        # who expected loading a history item to refresh both views.
        with timer.stage("build_context"):
            ctx = app._build_history_context(result_dir)
        if ctx is not None:
            with timer.stage("analysis_mol_render"):
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
    the load is in flight (immediate-loading-feedback contract). Tests
    and callers that don't have a button reference can omit it.

    Stage timings are emitted as a single ``history_load_timing`` event on
    completion (drives latency-optimization decisions). Stages cover
    the four expected hotspots: pyscf.log read, context build, molecule
    reconstruction, 3D viewer render, and the analysis-context registry walk.
    """
    _begin_history_load(app, "Loading analysis from history…", source_btns)
    timer = _LoadTimer("history_load_analysis", result_dir)
    status = "ok"
    try:
        app._last_result_dir = result_dir
        try:
            app._export_bundle_btn.disabled = False
        except Exception:
            pass
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
    molecule = mol_from_result_dir(result_dir, data)
    return context_cls(
        calc_type=data.get("calc_type", ""),
        formula=data.get("formula", result_dir.name),
        method=data.get("method", ""),
        basis=data.get("basis", ""),
        result_dir=result_dir,
        spectra_data=data.get("spectra", {}),
        timestamp=data.get("timestamp", ""),
        source="history",
        molecule=molecule,
    )
