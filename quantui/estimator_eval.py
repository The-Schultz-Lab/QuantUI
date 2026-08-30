"""
Offline evaluation of QuantUI's runtime estimator.

The pre-run time estimate is only actionable if a change to the model can be
*scored*, so this module replays the recorded performance history
through :func:`quantui.calc_log.estimate_time_from_records` and reports how
well the predictor would have done.

The replay is strictly causal: to score record *i*, the predictor is given
only records that were written **before** it. That mirrors what the app
actually knows at the moment the user is looking at the estimate, and it is
the difference between an honest score and one inflated by hindsight.

Two numbers matter together, and neither is meaningful alone:

* **accuracy** — median |error| and the fraction landing within ±25 %, the
  band M-EST set as the target;
* **coverage** — how often the predictor produced any estimate at all. A
  model that answers only when it is sure scores beautifully on accuracy
  while telling the user nothing, so ``n_no_estimate`` is reported next to
  the error figures rather than filtered out of them.

Typical use::

    python -m quantui.estimator_eval               # score the real log
    python -m quantui.estimator_eval --by-source   # split app vs calibration
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import calc_log

__all__ = [
    "ReplayStats",
    "ReplayReport",
    "replay",
    "format_report",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ReplayStats:
    """Accuracy + coverage for one slice of the replay."""

    label: str
    n_scored: int = 0
    n_no_estimate: int = 0
    errors_pct: list[float] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_scored + self.n_no_estimate

    @property
    def coverage_pct(self) -> float:
        """Percentage of runs for which the estimator produced a number."""
        return 100.0 * self.n_scored / self.n_total if self.n_total else 0.0

    @property
    def median_abs_error_pct(self) -> Optional[float]:
        if not self.errors_pct:
            return None
        return statistics.median(abs(e) for e in self.errors_pct)

    @property
    def median_signed_error_pct(self) -> Optional[float]:
        """Median signed error — the *bias*, as distinct from the spread.

        Reported alongside the absolute error because the two failure modes
        need different fixes: a biased predictor has the wrong constant, a
        high-spread one is being fed inconsistent measurements.
        """
        if not self.errors_pct:
            return None
        return statistics.median(self.errors_pct)

    def within_pct(self, band: float) -> Optional[float]:
        """Percentage of scored runs whose error falls within ±*band* %."""
        if not self.errors_pct:
            return None
        hits = sum(1 for e in self.errors_pct if abs(e) <= band)
        return 100.0 * hits / len(self.errors_pct)


@dataclass
class ReplayReport:
    """Overall + per-slice replay results."""

    overall: ReplayStats
    slices: dict[str, ReplayStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _sort_key(record: dict) -> str:
    """Chronological sort key, tolerant of missing timestamps.

    A record with no timestamp sorts first: it is almost certainly older
    than the tagged ones, and putting it early makes it history for later
    records rather than an unscoreable orphan.
    """
    return str(record.get("timestamp") or "")


def _is_scoreable(record: dict) -> bool:
    """True when *record* can serve as ground truth for one prediction."""
    if not record.get("converged"):
        return False
    try:
        # record is untyped JSON; float() may reject a missing/non-numeric
        # value at runtime, which the except below already handles.
        elapsed = float(record.get("elapsed_s"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    # A zero/negative elapsed can't produce a meaningful error percentage.
    return elapsed > 0


def replay(
    records: Optional[Iterable[dict]] = None,
    *,
    slice_by: str = "calc_type",
    use_source: bool = True,
) -> ReplayReport:
    """Score the estimator against *records* (default: the real perf log).

    Args:
        records: Performance records to replay. Defaults to the contents of
            ``perf_log.jsonl``.
        slice_by: Record key to group the report by — ``"calc_type"`` by
            default, ``"source"`` to compare app runs against calibration
            runs.
        use_source: Pass each record's own ``source`` to the predictor, so
            the replay exercises the provenance partitioning. Set ``False``
            to score the pre-Phase-C behaviour for comparison.

    Returns:
        A :class:`ReplayReport`. Records that cannot serve as ground truth
        (unconverged, or missing a usable ``elapsed_s``) are skipped
        entirely — they are not counted as either a hit or a miss, since
        the estimator was never asked about them.
    """
    if records is None:
        records = calc_log.get_perf_history()
    ordered = sorted(records, key=_sort_key)

    overall = ReplayStats(label="overall")
    slices: dict[str, ReplayStats] = {}

    history: list[dict] = []
    for record in ordered:
        # Every record becomes history for the ones after it, whether or
        # not it is itself scoreable — the app's estimator sees them all.
        # Appended in place, after being used as "past" below, rather than
        # concatenated into a fresh list each iteration — the same effect
        # (a prediction never sees its own ground-truth record) in O(n)
        # instead of O(n^2) for a large perf_log.jsonl.
        if not _is_scoreable(record):
            history.append(record)
            continue

        key = str(record.get(slice_by) or "(unset)")
        bucket = slices.setdefault(key, ReplayStats(label=key))

        predicted = calc_log.estimate_time_from_records(
            history,
            n_atoms=int(record.get("n_atoms") or 0),
            n_electrons=int(record.get("n_electrons") or 0),
            method=str(record.get("method") or ""),
            basis=str(record.get("basis") or ""),
            n_basis=record.get("n_basis"),
            n_cores=record.get("n_cores"),
            calc_type=record.get("calc_type"),
            gpu_used=record.get("gpu_used"),
            source=record.get("source") if use_source else None,
        )
        history.append(record)
        if predicted is None or float(predicted["seconds"]) <= 0:
            bucket.n_no_estimate += 1
            overall.n_no_estimate += 1
            continue

        actual = float(record["elapsed_s"])
        pred_s = float(predicted["seconds"])
        error_pct = 100.0 * (actual - pred_s) / pred_s
        bucket.n_scored += 1
        bucket.errors_pct.append(error_pct)
        overall.n_scored += 1
        overall.errors_pct.append(error_pct)

    return ReplayReport(overall=overall, slices=slices)


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], suffix: str = "") -> str:
    return "—" if value is None else f"{value:.1f}{suffix}"


def format_report(report: ReplayReport, *, title: str = "Estimator replay") -> str:
    """Return a fixed-width table summarising *report*."""
    header = (
        f"{'slice':24} {'runs':>6} {'cover':>7} "
        f"{'|err|':>8} {'bias':>8} {'±25%':>7} {'±50%':>7}"
    )
    lines = [title, "=" * len(header), header, "-" * len(header)]

    rows = sorted(report.slices.items(), key=lambda kv: -kv[1].n_total)
    for _, stats in rows + [("overall", report.overall)]:
        lines.append(
            f"{stats.label[:24]:24} {stats.n_total:6d} "
            f"{_fmt(stats.coverage_pct, '%'):>7} "
            f"{_fmt(stats.median_abs_error_pct, '%'):>8} "
            f"{_fmt(stats.median_signed_error_pct, '%'):>8} "
            f"{_fmt(stats.within_pct(25.0), '%'):>7} "
            f"{_fmt(stats.within_pct(50.0), '%'):>7}"
        )
    lines.append("")
    lines.append(
        "|err| = median absolute error   bias = median signed error   "
        "cover = share of runs given any estimate"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for ``python -m quantui.estimator_eval``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m quantui.estimator_eval",
        description="Replay the recorded performance history through the "
        "runtime estimator and report accuracy + coverage.",
    )
    parser.add_argument(
        "--by-source",
        action="store_true",
        help="group results by record provenance instead of calc type",
    )
    parser.add_argument(
        "--ignore-source",
        action="store_true",
        help="score the estimator without provenance partitioning "
        "(the pre-Phase-C behaviour), for before/after comparison",
    )
    args = parser.parse_args(argv)

    report = replay(
        slice_by="source" if args.by_source else "calc_type",
        use_source=not args.ignore_source,
    )
    if report.overall.n_total == 0:
        print("No performance history to replay yet.")
        return 0
    mode = "without" if args.ignore_source else "with"
    print(format_report(report, title=f"Estimator replay ({mode} provenance)"))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI shim
    raise SystemExit(main())
