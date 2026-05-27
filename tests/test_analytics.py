"""Tests for ``quantui.analytics.build_dashboard`` and helpers.

Pure-Python module, platform-independent. Uses ``QUANTUI_LOG_DIR`` to
redirect the perf log to a tmp path so we never touch the user's real
logs.
"""

from __future__ import annotations

import json

import pytest

from quantui import analytics


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


def _write_perf_log(log_dir, records):
    path = log_dir / "perf_log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _rec(
    *,
    method="B3LYP",
    basis="STO-3G",
    formula="H2O",
    elapsed_s=1.0,
    gpu_used=None,
    calc_type="Single Point",
    timestamp="2026-05-25T12:00:00+00:00",
    converged=True,
):
    r = {
        "timestamp": timestamp,
        "formula": formula,
        "n_atoms": 3,
        "n_electrons": 10,
        "method": method,
        "basis": basis,
        "n_iterations": 12,
        "elapsed_s": elapsed_s,
        "converged": converged,
        "calc_type": calc_type,
    }
    if gpu_used is not None:
        r["gpu_used"] = gpu_used
    return r


class TestClassifyDevice:
    def test_gpu_used_true_returns_gpu(self):
        assert analytics._classify_device({"gpu_used": True}) == "GPU"

    def test_gpu_used_false_returns_cpu(self):
        assert analytics._classify_device({"gpu_used": False}) == "CPU"

    def test_missing_field_returns_unknown(self):
        # Pre-M-GPU records have no gpu_used key.
        assert analytics._classify_device({}) == "Unknown"


class TestSummaryMetrics:
    def test_counts_runs_by_device(self):
        records = [
            _rec(gpu_used=True),
            _rec(gpu_used=True),
            _rec(gpu_used=False),
            _rec(),  # Unknown
        ]
        s = analytics._summary_metrics(records)
        assert s["total_runs"] == 4
        assert s["gpu_runs"] == 2
        assert s["cpu_runs"] == 1
        assert s["unknown_runs"] == 1

    def test_total_seconds_sums(self):
        records = [_rec(elapsed_s=10.0), _rec(elapsed_s=2.5)]
        s = analytics._summary_metrics(records)
        assert s["total_seconds"] == 12.5

    def test_unique_counts(self):
        records = [
            _rec(method="B3LYP", basis="STO-3G", formula="H2O"),
            _rec(method="B3LYP", basis="6-31G", formula="H2O"),
            _rec(method="MP2", basis="STO-3G", formula="CH4"),
        ]
        s = analytics._summary_metrics(records)
        assert s["unique_formulas"] == 2
        assert s["unique_methods"] == 2
        assert s["unique_basis"] == 2


class TestSpeedupRows:
    def test_empty_input_returns_empty(self):
        assert analytics._speedup_rows([]) == []

    def test_only_cpu_runs_no_pairs(self):
        # No GPU runs → no speedup data.
        records = [_rec(gpu_used=False, elapsed_s=5.0)]
        assert analytics._speedup_rows(records) == []

    def test_only_gpu_runs_no_pairs(self):
        records = [_rec(gpu_used=True, elapsed_s=1.0)]
        assert analytics._speedup_rows(records) == []

    def test_one_cpu_one_gpu_produces_row(self):
        records = [
            _rec(gpu_used=False, elapsed_s=10.0),
            _rec(gpu_used=True, elapsed_s=2.0),
        ]
        rows = analytics._speedup_rows(records)
        assert len(rows) == 1
        r = rows[0]
        assert r["cpu_median_s"] == 10.0
        assert r["gpu_median_s"] == 2.0
        assert r["speedup"] == 5.0
        assert r["cpu_runs"] == 1
        assert r["gpu_runs"] == 1

    def test_median_used_not_mean(self):
        # CPU times: 10, 100, 1000 → median 100 (not mean 370).
        records = [
            _rec(gpu_used=False, elapsed_s=10.0),
            _rec(gpu_used=False, elapsed_s=100.0),
            _rec(gpu_used=False, elapsed_s=1000.0),
            _rec(gpu_used=True, elapsed_s=2.0),
        ]
        rows = analytics._speedup_rows(records)
        assert rows[0]["cpu_median_s"] == 100.0

    def test_sorted_by_speedup_desc(self):
        # Two tuples, very different speedups.
        records = [
            _rec(method="MP2", gpu_used=False, elapsed_s=100.0),
            _rec(method="MP2", gpu_used=True, elapsed_s=10.0),
            _rec(method="RHF", gpu_used=False, elapsed_s=2.0),
            _rec(method="RHF", gpu_used=True, elapsed_s=1.5),
        ]
        rows = analytics._speedup_rows(records)
        assert len(rows) == 2
        # MP2 has 10x speedup, RHF has ~1.33x → MP2 first.
        assert rows[0]["method"] == "MP2"
        assert rows[1]["method"] == "RHF"

    def test_unknown_device_excluded_from_pair_match(self):
        # An Unknown record can't be paired against CPU or GPU.
        records = [
            _rec(gpu_used=True, elapsed_s=2.0),
            _rec(elapsed_s=10.0),  # Unknown
        ]
        assert analytics._speedup_rows(records) == []


class TestBuildDashboard:
    def test_empty_perf_log_returns_none(self, isolated_log_dir):
        assert analytics.build_dashboard() is None

    def test_writes_file_with_default_path(self, isolated_log_dir):
        _write_perf_log(isolated_log_dir, [_rec(gpu_used=True)])
        out = analytics.build_dashboard()
        assert out is not None
        assert out.exists()
        # Default path lives one level up from logs dir.
        assert out == isolated_log_dir.parent / "dashboard.html"

    def test_writes_to_explicit_path(self, isolated_log_dir, tmp_path):
        _write_perf_log(isolated_log_dir, [_rec()])
        target = tmp_path / "elsewhere" / "report.html"
        out = analytics.build_dashboard(target)
        assert out == target
        assert target.exists()

    def test_dashboard_html_contains_overview_cards(self, isolated_log_dir):
        records = [
            _rec(method="B3LYP", gpu_used=True, elapsed_s=2.0),
            _rec(method="MP2", gpu_used=False, elapsed_s=20.0),
        ]
        _write_perf_log(isolated_log_dir, records)
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        assert "QuantUI analytics" in html
        assert "Total runs" in html
        assert "GPU runs" in html
        assert "CPU runs" in html
        # 2 total runs.
        assert ">2<" in html

    def test_dashboard_includes_speedup_section_when_pairs_exist(
        self, isolated_log_dir
    ):
        records = [
            _rec(method="MP2", gpu_used=False, elapsed_s=100.0),
            _rec(method="MP2", gpu_used=True, elapsed_s=10.0),
        ]
        _write_perf_log(isolated_log_dir, records)
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        assert "GPU vs CPU speedup" in html
        # 10x speedup formatted as 10.00x with a multiplication sign.
        assert "10.00×" in html
        # Empty-state banner should NOT be present when we have data.
        assert "Re-run any prior CPU calc" not in html

    def test_dashboard_shows_empty_state_when_no_pairs(self, isolated_log_dir):
        # Only GPU runs, no CPU pairs → empty-state msg in speedup table.
        records = [_rec(gpu_used=True, elapsed_s=2.0)]
        _write_perf_log(isolated_log_dir, records)
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        assert "Re-run any prior CPU calc" in html

    def test_dashboard_inlines_plotly_js(self, isolated_log_dir):
        # Only one figure should embed the full plotly bundle — verifying
        # we don't accidentally ship 3x by passing include_plotlyjs=True
        # to every figure helper.
        records = [_rec(method="B3LYP"), _rec(method="MP2")]
        _write_perf_log(isolated_log_dir, records)
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        # plotly.js inline mode wraps everything in <script>...</script>
        # that contains "Plotly". We expect exactly one such inline bundle.
        assert "Plotly" in html
        # Sanity: file is non-trivial size (plotly inline is ~3MB).
        assert len(html) > 100_000

    def test_dashboard_resilient_to_partial_records(self, isolated_log_dir):
        # Records missing fields (early app version, partial writes) must
        # not crash the dashboard build.
        records = [
            {"timestamp": "2026-05-25T12:00:00+00:00"},  # bare minimum
            _rec(),  # full
        ]
        _write_perf_log(isolated_log_dir, records)
        out = analytics.build_dashboard()
        assert out is not None
        assert out.exists()


class TestFormatHelpers:
    def test_format_seconds_under_minute(self):
        assert analytics._format_seconds(45.0) == "45.0 s"

    def test_format_seconds_minutes(self):
        assert analytics._format_seconds(90.0) == "1.5 min"

    def test_format_seconds_hours(self):
        assert analytics._format_seconds(7200.0) == "2.0 h"

    def test_counts_by_drops_missing(self):
        records = [{"method": "B3LYP"}, {"method": ""}, {"method": "MP2"}, {}]
        counts = analytics._counts_by(records, "method")
        assert counts == {"B3LYP": 1, "MP2": 1}
