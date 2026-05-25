"""Self-contained analytics dashboard for QuantUI usage data.

Reads ``~/.quantui/logs/perf_log.jsonl`` (override with
``QUANTUI_LOG_DIR``) and writes a standalone HTML report with charts that
work offline — Plotly's JS is inlined into the file so the user can open
it directly in a browser (no Voilà, no Jupyter).

What the dashboard shows
------------------------

1. **Overview cards** — total runs, total compute time, GPU vs CPU run
   counts, unique molecules / methods / basis sets.
2. **GPU vs CPU speedup table** — for every (method, basis, formula) that
   has runs on BOTH devices, the median CPU time, median GPU time, and
   the resulting speedup factor. Sortable / readable in one glance.
3. **Method usage** — bar chart of run counts per method.
4. **Calc-type distribution** — bar chart of run counts per calc_type.
5. **Recent timeline** — scatter of ``elapsed_s`` over time coloured by
   compute device (CPU grey, GPU green), so a user can spot regressions
   or speedups visually as they run more calcs.

Older perf-log records that pre-date the M-GPU follow-up don't have
``gpu_used`` set — those are treated as "device unknown" and counted in
their own bucket rather than guessed CPU.

Output is a single ``.html`` file (default ``~/.quantui/dashboard.html``)
the user can pin to their browser or email to a collaborator.
"""

from __future__ import annotations

import html as _html
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from quantui.calc_log import _log_dir, get_perf_history, get_prediction_history

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_device(record: dict) -> str:
    """Return ``"GPU"``, ``"CPU"``, or ``"Unknown"`` for one perf record.

    Records written before the M-GPU follow-up (2026-05-25) don't have
    ``gpu_used`` at all — we don't backfill those as CPU because they
    pre-date GPU support entirely, so calling them "CPU" would muddy any
    speedup comparison. ``"Unknown"`` is the honest bucket.
    """
    if "gpu_used" not in record:
        return "Unknown"
    return "GPU" if record["gpu_used"] else "CPU"


def _summary_metrics(records: list[dict]) -> dict:
    """Compute headline counters for the overview cards."""
    total_runs = len(records)
    total_seconds = sum(float(r.get("elapsed_s", 0.0)) for r in records)
    gpu_runs = sum(1 for r in records if _classify_device(r) == "GPU")
    cpu_runs = sum(1 for r in records if _classify_device(r) == "CPU")
    unknown_runs = sum(1 for r in records if _classify_device(r) == "Unknown")
    converged = sum(1 for r in records if r.get("converged"))
    unique_formulas = len({r.get("formula", "") for r in records if r.get("formula")})
    unique_methods = len({r.get("method", "") for r in records if r.get("method")})
    unique_basis = len({r.get("basis", "") for r in records if r.get("basis")})
    return {
        "total_runs": total_runs,
        "total_seconds": total_seconds,
        "gpu_runs": gpu_runs,
        "cpu_runs": cpu_runs,
        "unknown_runs": unknown_runs,
        "converged_runs": converged,
        "unique_formulas": unique_formulas,
        "unique_methods": unique_methods,
        "unique_basis": unique_basis,
    }


def _speedup_rows(records: list[dict]) -> list[dict]:
    """For each (method, basis, formula) with both CPU and GPU runs, return
    a row with median times and the speedup factor.

    Only tuples that have at least one CPU run AND at least one GPU run
    show up. ``Unknown`` device records are ignored for this comparison.
    Sorted by speedup descending (best speedups at the top).
    """
    bucket: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"CPU": [], "GPU": []}
    )
    for r in records:
        dev = _classify_device(r)
        if dev not in ("CPU", "GPU"):
            continue
        key = (
            r.get("method", "?"),
            r.get("basis", "?"),
            r.get("formula", "?"),
        )
        try:
            bucket[key][dev].append(float(r["elapsed_s"]))
        except (KeyError, TypeError, ValueError):
            continue

    rows: list[dict] = []
    for (method, basis, formula), times in bucket.items():
        if not times["CPU"] or not times["GPU"]:
            continue
        cpu_med = statistics.median(times["CPU"])
        gpu_med = statistics.median(times["GPU"])
        if gpu_med <= 0:
            continue
        rows.append(
            {
                "method": method,
                "basis": basis,
                "formula": formula,
                "cpu_runs": len(times["CPU"]),
                "gpu_runs": len(times["GPU"]),
                "cpu_median_s": cpu_med,
                "gpu_median_s": gpu_med,
                "speedup": cpu_med / gpu_med,
            }
        )
    rows.sort(key=lambda r: r["speedup"], reverse=True)
    return rows


def _counts_by(records: list[dict], field: str) -> dict[str, int]:
    """Tally ``records`` by ``field``, dropping empty/missing values."""
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        v = r.get(field)
        if v:
            counts[str(v)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_DASHBOARD_CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       sans-serif; margin: 24px; color: #1f2937; background: #f9fafb; }
h1 { margin: 0 0 4px; }
.sub { color: #6b7280; margin: 0 0 24px; font-size: 14px; }
.card-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 24px; }
.card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 14px 18px; min-width: 160px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.card .label { color: #6b7280; font-size: 12px; text-transform: uppercase;
               letter-spacing: 0.05em; }
.card .value { font-size: 24px; font-weight: 600; margin-top: 4px; color: #111827; }
.card.gpu .value { color: #059669; }
.card.cpu .value { color: #4b5563; }
section { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
          padding: 18px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
section h2 { margin: 0 0 12px; font-size: 18px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #f3f4f6; }
th { background: #f9fafb; color: #374151; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.speedup-good { color: #059669; font-weight: 600; }
td.speedup-flat { color: #6b7280; }
.empty { color: #9ca3af; font-style: italic; padding: 20px 0; }
footer { color: #9ca3af; font-size: 12px; margin-top: 32px; text-align: center; }
</style>
"""


def _card(label: str, value: str, css_class: str = "") -> str:
    cls = f"card {css_class}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="label">{_html.escape(label)}</div>'
        f'<div class="value">{_html.escape(value)}</div></div>'
    )


def _format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _overview_section(summary: dict) -> str:
    cards = [
        _card("Total runs", str(summary["total_runs"])),
        _card("Total compute", _format_seconds(summary["total_seconds"])),
        _card("GPU runs", str(summary["gpu_runs"]), css_class="gpu"),
        _card("CPU runs", str(summary["cpu_runs"]), css_class="cpu"),
    ]
    if summary["unknown_runs"]:
        cards.append(_card("Device unknown", str(summary["unknown_runs"])))
    cards.extend(
        [
            _card("Unique molecules", str(summary["unique_formulas"])),
            _card("Methods used", str(summary["unique_methods"])),
            _card("Basis sets used", str(summary["unique_basis"])),
        ]
    )
    return (
        "<section><h2>Overview</h2>"
        f'<div class="card-row">{"".join(cards)}</div></section>'
    )


def _speedup_section(rows: list[dict]) -> str:
    if not rows:
        return (
            "<section><h2>GPU vs CPU speedup</h2>"
            '<p class="empty">No (method, basis, formula) tuple has runs on '
            "both devices yet. Re-run any prior CPU calc on the GPU to populate "
            "this table.</p></section>"
        )
    body_rows = []
    for r in rows:
        speedup_cls = "speedup-good" if r["speedup"] >= 1.5 else "speedup-flat"
        body_rows.append(
            "<tr>"
            f"<td>{_html.escape(r['method'])}</td>"
            f"<td>{_html.escape(r['basis'])}</td>"
            f"<td>{_html.escape(r['formula'])}</td>"
            f'<td class="num">{r["cpu_runs"]}</td>'
            f'<td class="num">{r["gpu_runs"]}</td>'
            f'<td class="num">{r["cpu_median_s"]:.2f}</td>'
            f'<td class="num">{r["gpu_median_s"]:.2f}</td>'
            f'<td class="num {speedup_cls}">{r["speedup"]:.2f}×</td>'
            "</tr>"
        )
    return (
        "<section><h2>GPU vs CPU speedup</h2>"
        "<table><thead><tr>"
        "<th>Method</th><th>Basis</th><th>Formula</th>"
        "<th>CPU n</th><th>GPU n</th>"
        "<th>CPU median (s)</th><th>GPU median (s)</th>"
        "<th>Speedup</th>"
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></section>"
    )


def _figure_section(title: str, fig_html: Optional[str], empty_msg: str) -> str:
    if fig_html is None:
        return f'<section><h2>{_html.escape(title)}</h2><p class="empty">{empty_msg}</p></section>'
    return f"<section><h2>{_html.escape(title)}</h2>{fig_html}</section>"


def _bar_chart_html(
    counts: dict[str, int], *, title: str, include_plotlyjs: bool
) -> Optional[str]:
    if not counts:
        return None
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return None
    keys = sorted(counts, key=lambda k: counts[k], reverse=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=keys,
                y=[counts[k] for k in keys],
                marker_color="#6366f1",
            )
        ]
    )
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Runs",
        height=320,
        margin=dict(l=40, r=20, t=10, b=40),
        plot_bgcolor="#ffffff",
    )
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False},
    )


def _timeline_html(records: list[dict], *, include_plotlyjs: bool) -> Optional[str]:
    """Scatter of elapsed_s vs timestamp, coloured by device."""
    if not records:
        return None
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return None

    grouped: dict[str, list[tuple[datetime, float, str]]] = {
        "GPU": [],
        "CPU": [],
        "Unknown": [],
    }
    for r in records:
        try:
            ts = datetime.fromisoformat(str(r["timestamp"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        elapsed = float(r.get("elapsed_s", 0.0))
        label = (
            f"{r.get('method', '?')}/{r.get('basis', '?')} on "
            f"{r.get('formula', '?')}"
        )
        grouped[_classify_device(r)].append((ts, elapsed, label))

    color_map = {"GPU": "#059669", "CPU": "#6b7280", "Unknown": "#d1d5db"}
    traces = []
    for dev, points in grouped.items():
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        traces.append(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="markers",
                name=dev,
                text=[p[2] for p in points],
                marker=dict(size=8, color=color_map[dev], opacity=0.8),
                hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} s<extra></extra>",
            )
        )
    if not traces:
        return None
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=380,
        yaxis_title="Elapsed (s)",
        margin=dict(l=50, r=20, t=10, b=50),
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", x=0, y=1.05),
    )
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False},
    )


# ---------------------------------------------------------------------------
# Prediction-accuracy section (M-EST / EST.6, 2026-05-25)
# ---------------------------------------------------------------------------


def _prediction_accuracy_metrics(records: list[dict]) -> dict:
    """Compute headline accuracy metrics from prediction-log records.

    Records with ``predicted_s=None`` are "no-estimate" runs and counted
    separately. For the median-error calculation we use absolute
    percentage error (``|actual - predicted| / predicted * 100``), so
    over- and under-predictions weigh the same; the dashboard shows
    both the signed median (bias) and the absolute median (magnitude).
    """
    have_pred = [
        r
        for r in records
        if r.get("predicted_s") is not None and r.get("error_pct") is not None
    ]
    no_pred = [r for r in records if r.get("predicted_s") is None]
    abs_errs = [abs(float(r["error_pct"])) for r in have_pred]
    signed_errs = [float(r["error_pct"]) for r in have_pred]
    return {
        "n_total": len(records),
        "n_with_estimate": len(have_pred),
        "n_no_estimate": len(no_pred),
        "median_abs_error_pct": (statistics.median(abs_errs) if abs_errs else None),
        "median_signed_error_pct": (
            statistics.median(signed_errs) if signed_errs else None
        ),
        # "Within 25%" — a useful headline metric ("how often is the
        # estimator usefully close?"). Roadmap target: ≥ 70% after a
        # tier-4 calibration.
        "pct_within_25": (
            round(100.0 * sum(1 for e in abs_errs if e <= 25.0) / len(abs_errs), 1)
            if abs_errs
            else None
        ),
    }


def _prediction_scatter_html(
    records: list[dict], *, include_plotlyjs: bool
) -> Optional[str]:
    """Scatter of predicted_s vs actual_s with a y=x reference line."""
    have_pred = [
        r
        for r in records
        if r.get("predicted_s") is not None and r.get("actual_s") is not None
    ]
    if len(have_pred) < 2:
        return None
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return None

    # Hover labels show the calc spec so the user can identify outliers.
    text_labels = [
        f"{r.get('method', '?')}/{r.get('basis', '?')} on {r.get('formula', '?')}"
        for r in have_pred
    ]
    predicted = [float(r["predicted_s"]) for r in have_pred]
    actual = [float(r["actual_s"]) for r in have_pred]
    max_val = max(max(predicted), max(actual), 1.0) * 1.1

    fig = go.Figure()
    # y=x reference line (perfect prediction).
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="perfect (y=x)",
            line=dict(color="#94a3b8", dash="dash", width=1),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=actual,
            mode="markers",
            name="run",
            text=text_labels,
            marker=dict(size=9, color="#6366f1", opacity=0.75),
            hovertemplate=(
                "%{text}<br>predicted: %{x:.2f} s<br>actual: %{y:.2f} s<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=420,
        xaxis=dict(title="Predicted (s)", range=[0, max_val]),
        yaxis=dict(title="Actual (s)", range=[0, max_val]),
        margin=dict(l=60, r=20, t=10, b=50),
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", x=0, y=1.05),
    )
    return pio.to_html(
        fig,
        include_plotlyjs="inline" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False},
    )


def _prediction_accuracy_section(
    records: list[dict], scatter_html: Optional[str]
) -> str:
    """Render the "Prediction accuracy" section of the dashboard."""
    if not records:
        return (
            "<section><h2>Prediction accuracy</h2>"
            '<p class="empty">No predictions logged yet — run a few '
            "calculations and the estimator's track record will appear here.</p>"
            "</section>"
        )

    m = _prediction_accuracy_metrics(records)
    median_abs = m["median_abs_error_pct"]
    median_signed = m["median_signed_error_pct"]
    within_25 = m["pct_within_25"]

    # Banner when median absolute error exceeds 50%: estimator is in
    # rough shape; re-running calibration usually helps.
    banner = ""
    if median_abs is not None and median_abs > 50.0:
        banner = (
            '<p style="background:#fef3c7;color:#78350f;border-left:4px solid #f59e0b;'
            'padding:8px 12px;margin:8px 0;border-radius:4px;font-size:13px">'
            f"⚠ Median absolute prediction error is {median_abs:.0f}%. "
            "Re-running a deeper calibration tier (System Settings → Calibrate "
            "time estimates) typically tightens this within ±25%."
            "</p>"
        )

    cards = [
        _card("Predictions logged", str(m["n_total"])),
        _card(
            "With estimate",
            f"{m['n_with_estimate']} / {m['n_total']}",
        ),
    ]
    if median_abs is not None:
        cards.append(_card("Median |error|", f"{median_abs:.1f}%"))
    if median_signed is not None:
        sign = "+" if median_signed >= 0 else ""
        cards.append(_card("Median bias", f"{sign}{median_signed:.1f}%"))
    if within_25 is not None:
        cards.append(_card("Within ±25%", f"{within_25:.0f}%"))
    if m["n_no_estimate"]:
        cards.append(_card("No estimate", str(m["n_no_estimate"])))

    chart_block = (
        scatter_html
        if scatter_html
        else (
            '<p class="empty">Need at least 2 predictions with an estimate '
            "before plotting accuracy.</p>"
        )
    )
    return (
        "<section><h2>Prediction accuracy</h2>"
        + banner
        + f'<div class="card-row">{"".join(cards)}</div>'
        + chart_block
        + "</section>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dashboard(out_path: Optional[Path] = None) -> Optional[Path]:
    """Generate the QuantUI analytics dashboard as a self-contained HTML.

    Reads ``perf_log.jsonl`` from the active log directory (honouring
    ``QUANTUI_LOG_DIR``) and writes the dashboard to ``out_path``. If
    ``out_path`` is ``None``, defaults to ``<log_dir>/../dashboard.html``
    (one level up so it lives next to ``~/.quantui/`` rather than buried
    in the logs folder).

    Returns the path to the written dashboard on success, or ``None`` if
    there are zero records in the perf log (nothing to report — the
    caller should surface that as an empty-state message).
    """
    records = get_perf_history()
    if not records:
        return None

    if out_path is None:
        out_path = _log_dir().parent / "dashboard.html"
    out_path = Path(out_path)

    summary = _summary_metrics(records)
    speedup_rows = _speedup_rows(records)
    method_counts = _counts_by(records, "method")
    calc_type_counts = _counts_by(records, "calc_type")

    # M-EST / EST.6: prediction-accuracy data lives in its own log file.
    # Best-effort read — older installs without the file produce an
    # empty list and the section degrades to an empty-state message.
    try:
        prediction_records = get_prediction_history()
    except Exception:  # noqa: BLE001 — best-effort
        prediction_records = []

    # Inline plotly.js exactly once (in the first figure that renders).
    # Subsequent figures pass include_plotlyjs=False so we don't ship
    # the ~3 MB bundle three times.
    method_bar = _bar_chart_html(
        method_counts, title="Method usage", include_plotlyjs=True
    )
    calctype_bar = _bar_chart_html(
        calc_type_counts, title="Calc-type distribution", include_plotlyjs=False
    )
    timeline = _timeline_html(records, include_plotlyjs=False)
    prediction_scatter = _prediction_scatter_html(
        prediction_records, include_plotlyjs=False
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>QuantUI analytics</title>" + _DASHBOARD_CSS + "</head><body>"
        "<h1>QuantUI analytics</h1>"
        f'<p class="sub">Generated {generated} — {summary["total_runs"]} runs in perf log</p>'
        + _overview_section(summary)
        + _speedup_section(speedup_rows)
        + _prediction_accuracy_section(prediction_records, prediction_scatter)
        + _figure_section(
            "Method usage",
            method_bar,
            "No method-tagged records found.",
        )
        + _figure_section(
            "Calc-type distribution",
            calctype_bar,
            "No calc-type-tagged records found.",
        )
        + _figure_section(
            "Recent timeline",
            timeline,
            "No timestamped records to plot.",
        )
        + "<footer>QuantUI analytics dashboard — open with any browser.</footer>"
        + "</body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return out_path
