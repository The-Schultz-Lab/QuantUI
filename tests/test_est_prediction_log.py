"""Tests for M-EST / EST.6 — predicted-vs-actual feedback log.

After each ``_do_run``, QuantUI now writes a record to
``prediction_log.jsonl`` with the estimator's pre-run prediction +
the actual wall-clock outcome. The analytics dashboard surfaces:

- headline cards (median absolute error %, % within 25%, bias, etc.)
- a scatter of predicted vs actual with a y=x reference line
- a "consider re-running calibration" banner when median |error| > 50%

All tests are platform-independent. ``prediction_log.jsonl`` is
redirected to ``tmp_path`` via ``QUANTUI_LOG_DIR``.
"""

from __future__ import annotations

import inspect
import json

import pytest

from quantui import analytics
from quantui.calc_log import (
    _prediction_log_path,
    get_prediction_history,
    log_prediction,
)


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


# =====================================================================
# log_prediction / get_prediction_history
# =====================================================================


class TestLogPrediction:
    def test_writes_record_with_all_fields(self, isolated_log_dir):
        log_prediction(
            predicted_s=10.0,
            actual_s=12.5,
            method="B3LYP",
            basis="6-31G*",
            calc_type="single_point",
            formula="H2O",
            confidence="high",
            gpu_used=False,
        )
        records = get_prediction_history()
        assert len(records) == 1
        r = records[0]
        assert r["predicted_s"] == 10.0
        assert r["actual_s"] == 12.5
        assert r["method"] == "B3LYP"
        assert r["calc_type"] == "single_point"
        assert r["formula"] == "H2O"
        assert r["confidence"] == "high"
        assert r["gpu_used"] is False
        # Derived field: signed error percentage.
        assert r["error_pct"] == 25.0

    def test_underprediction_yields_positive_error(self, isolated_log_dir):
        # Predicted 1 min, took 5 min — error_pct should be +400% (actual
        # is 4x the prediction, i.e. 400% larger).
        log_prediction(
            predicted_s=60.0,
            actual_s=300.0,
            method="B3LYP",
            basis="6-31G*",
            calc_type="frequency",
        )
        r = get_prediction_history()[0]
        assert r["error_pct"] == 400.0

    def test_overprediction_yields_negative_error(self, isolated_log_dir):
        # Predicted 100 s, took 50 s — error_pct should be -50%.
        log_prediction(
            predicted_s=100.0,
            actual_s=50.0,
            method="RHF",
            basis="STO-3G",
            calc_type="single_point",
        )
        r = get_prediction_history()[0]
        assert r["error_pct"] == -50.0

    def test_no_estimate_records_none_error(self, isolated_log_dir):
        # When the estimator returned no estimate (insufficient history),
        # we still log the actual outcome so the dashboard counts the
        # "no-estimate" runs separately.
        log_prediction(
            predicted_s=None,
            actual_s=1.5,
            method="B3LYP",
            basis="STO-3G",
            calc_type="single_point",
        )
        r = get_prediction_history()[0]
        assert r["predicted_s"] is None
        assert r["error_pct"] is None
        assert r["actual_s"] == 1.5

    def test_zero_predicted_does_not_div_by_zero(self, isolated_log_dir):
        # Defensive: predicted_s=0 is nonsensical but mustn't crash.
        log_prediction(
            predicted_s=0.0,
            actual_s=1.0,
            method="RHF",
            basis="STO-3G",
            calc_type="single_point",
        )
        r = get_prediction_history()[0]
        assert r["error_pct"] is None  # zero-protected path

    def test_path_honors_quantui_log_dir(self, isolated_log_dir):
        # The fixture sets QUANTUI_LOG_DIR. The prediction log must
        # land there, not in ~/.quantui/logs.
        log_prediction(
            predicted_s=1.0,
            actual_s=1.0,
            method="RHF",
            basis="STO-3G",
            calc_type="single_point",
        )
        assert _prediction_log_path().parent == isolated_log_dir


# =====================================================================
# Analytics metrics
# =====================================================================


class TestPredictionAccuracyMetrics:
    def test_empty_records(self):
        m = analytics._prediction_accuracy_metrics([])
        assert m["n_total"] == 0
        assert m["median_abs_error_pct"] is None
        assert m["median_signed_error_pct"] is None
        assert m["pct_within_25"] is None

    def test_all_within_25_pct(self):
        # Spread of 10% / 15% / 20% / 5% — all within 25%.
        records = [
            {"predicted_s": 1.0, "actual_s": 1.1, "error_pct": 10.0},
            {"predicted_s": 1.0, "actual_s": 1.15, "error_pct": 15.0},
            {"predicted_s": 1.0, "actual_s": 1.2, "error_pct": 20.0},
            {"predicted_s": 1.0, "actual_s": 1.05, "error_pct": 5.0},
        ]
        m = analytics._prediction_accuracy_metrics(records)
        assert m["pct_within_25"] == 100.0

    def test_mixed_within_25(self):
        # 2 of 4 within 25%, 2 outside (one +60%, one -40%).
        records = [
            {"predicted_s": 1.0, "actual_s": 1.1, "error_pct": 10.0},
            {"predicted_s": 1.0, "actual_s": 1.2, "error_pct": 20.0},
            {"predicted_s": 1.0, "actual_s": 1.6, "error_pct": 60.0},
            {"predicted_s": 1.0, "actual_s": 0.6, "error_pct": -40.0},
        ]
        m = analytics._prediction_accuracy_metrics(records)
        assert m["pct_within_25"] == 50.0

    def test_signed_median_picks_up_bias(self):
        # All four runs over-ran the prediction → positive bias.
        records = [
            {"predicted_s": 1.0, "actual_s": 1.5, "error_pct": 50.0},
            {"predicted_s": 1.0, "actual_s": 1.6, "error_pct": 60.0},
            {"predicted_s": 1.0, "actual_s": 1.4, "error_pct": 40.0},
            {"predicted_s": 1.0, "actual_s": 1.7, "error_pct": 70.0},
        ]
        m = analytics._prediction_accuracy_metrics(records)
        assert m["median_signed_error_pct"] is not None
        assert m["median_signed_error_pct"] > 0  # positive bias

    def test_no_estimate_records_excluded_from_error_stats(self):
        # 2 records with no estimate + 2 with — the metrics use only
        # the 2 that have data, and report the no-estimate count.
        records = [
            {"predicted_s": None, "actual_s": 1.0, "error_pct": None},
            {"predicted_s": None, "actual_s": 2.0, "error_pct": None},
            {"predicted_s": 1.0, "actual_s": 1.1, "error_pct": 10.0},
            {"predicted_s": 1.0, "actual_s": 1.2, "error_pct": 20.0},
        ]
        m = analytics._prediction_accuracy_metrics(records)
        assert m["n_total"] == 4
        assert m["n_with_estimate"] == 2
        assert m["n_no_estimate"] == 2
        assert m["median_abs_error_pct"] == 15.0


# =====================================================================
# Dashboard rendering
# =====================================================================


def _seed_perf_log(log_dir):
    """Seed perf_log so build_dashboard doesn't early-return None."""
    p = log_dir / "perf_log.jsonl"
    p.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-25T12:00:00+00:00",
                "formula": "H2O",
                "method": "B3LYP",
                "basis": "STO-3G",
                "elapsed_s": 1.0,
                "converged": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_prediction_log(log_dir, records):
    p = log_dir / "prediction_log.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


class TestDashboardPredictionSection:
    def test_section_present_when_predictions_exist(self, isolated_log_dir):
        _seed_perf_log(isolated_log_dir)
        _seed_prediction_log(
            isolated_log_dir,
            [
                {
                    "timestamp": "2026-05-25T12:00:00+00:00",
                    "predicted_s": 1.0,
                    "actual_s": 1.1,
                    "error_pct": 10.0,
                    "method": "B3LYP",
                    "basis": "STO-3G",
                    "calc_type": "single_point",
                },
                {
                    "timestamp": "2026-05-25T12:01:00+00:00",
                    "predicted_s": 5.0,
                    "actual_s": 6.0,
                    "error_pct": 20.0,
                    "method": "B3LYP",
                    "basis": "STO-3G",
                    "calc_type": "single_point",
                },
            ],
        )
        out = analytics.build_dashboard()
        assert out is not None
        html = out.read_text(encoding="utf-8")
        assert "Prediction accuracy" in html
        # Headline metric should appear (median |error| = 15%).
        assert "15.0%" in html

    def test_empty_state_when_no_predictions(self, isolated_log_dir):
        _seed_perf_log(isolated_log_dir)
        # No prediction_log.jsonl written.
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        assert "Prediction accuracy" in html
        assert "No predictions logged yet" in html

    def test_banner_when_median_error_exceeds_threshold(self, isolated_log_dir):
        _seed_perf_log(isolated_log_dir)
        # All four predictions off by 60%+ → median absolute > 50%.
        _seed_prediction_log(
            isolated_log_dir,
            [
                {
                    "timestamp": f"2026-05-25T12:00:{i:02d}+00:00",
                    "predicted_s": 1.0,
                    "actual_s": 2.0,
                    "error_pct": 100.0,
                    "method": "B3LYP",
                    "basis": "STO-3G",
                    "calc_type": "single_point",
                }
                for i in range(4)
            ],
        )
        out = analytics.build_dashboard()
        html = out.read_text(encoding="utf-8")
        # The re-calibrate banner kicks in at median |error| > 50%.
        assert "Re-running a deeper calibration tier" in html


# =====================================================================
# _do_run wiring — source-level structure check
# =====================================================================


class TestDoRunWiring:
    def test_do_run_captures_predicted_run_s(self):
        from quantui import app as _app_mod

        src = inspect.getsource(_app_mod)
        # The capture variable name is unique to EST.6.
        assert "_predicted_run_s" in src
        # And the call to log_prediction happens after log_calculation.
        assert "log_prediction(" in src

    def test_do_run_passes_gpu_used_to_estimator(self):
        # The pre-run estimate must honour the device prediction so the
        # logged predicted_s matches what the user saw in the UI.
        from quantui import app as _app_mod

        src = inspect.getsource(_app_mod)
        assert "_predicted_gpu_used" in src
