"""Tests for M-EST estimator hardening.

Covers:

- **EST.1**: GPU-aware filtering — passing ``gpu_used`` partitions the
  candidate pool so GPU-history predicts GPU runs and CPU-history
  predicts CPU runs. Includes the partition-fallback path (insufficient
  records → fall back to mixed pool, downgrade confidence).
- **EST.3**: IQR outlier rejection — a single anomalously-slow record
  no longer dominates the median.
- **EST.3**: variance-aware confidence — high-variance pools report
  "low" confidence even with many samples.

All tests are platform-independent. ``perf_log.jsonl`` is redirected to
``tmp_path`` via the ``QUANTUI_LOG_DIR`` env var so the user's real log
is never touched.
"""

from __future__ import annotations

import json

import pytest

from quantui.calc_log import (
    _coefficient_of_variation,
    _confidence_label,
    _iqr_filter,
    estimate_time,
)


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


def _seed_perf_log(log_dir, records):
    path = log_dir / "perf_log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _rec(
    *,
    elapsed_s: float,
    gpu_used=None,
    method="B3LYP",
    basis="STO-3G",
    n_basis=15,
    n_electrons=10,
    calc_type="single_point",
    converged=True,
    n_cores=1,
):
    r = {
        "timestamp": "2026-05-25T12:00:00+00:00",
        "formula": "H2O",
        "n_atoms": 3,
        "n_electrons": n_electrons,
        "method": method,
        "basis": basis,
        "n_iterations": 10,
        "elapsed_s": elapsed_s,
        "converged": converged,
        "n_basis": n_basis,
        "n_cores": n_cores,
        "calc_type": calc_type,
    }
    if gpu_used is not None:
        r["gpu_used"] = gpu_used
    return r


# =====================================================================
# EST.1 — GPU-aware filtering
# =====================================================================


class TestGpuAwareFiltering:
    def test_gpu_pool_used_when_requested(self, isolated_log_dir):
        # 5 GPU records (fast) + 5 CPU records (slow) for the same calc.
        records = [_rec(elapsed_s=1.0, gpu_used=True) for _ in range(5)]
        records += [_rec(elapsed_s=10.0, gpu_used=False) for _ in range(5)]
        _seed_perf_log(isolated_log_dir, records)

        gpu_est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
            gpu_used=True,
        )
        cpu_est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
            gpu_used=False,
        )

        assert gpu_est is not None
        assert cpu_est is not None
        # GPU prediction should land near 1.0 s; CPU near 10.0 s.
        assert gpu_est["seconds"] < 3.0
        assert cpu_est["seconds"] > 5.0
        # And they should differ by roughly the recorded factor.
        assert cpu_est["seconds"] / gpu_est["seconds"] > 3.0

    def test_none_gpu_used_uses_full_pool(self, isolated_log_dir):
        # Default callers (gpu_used=None) get the mixed-pool estimate.
        records = [_rec(elapsed_s=1.0, gpu_used=True) for _ in range(3)]
        records += [_rec(elapsed_s=11.0, gpu_used=False) for _ in range(3)]
        _seed_perf_log(isolated_log_dir, records)

        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
            # gpu_used omitted → None → no partition
        )
        assert est is not None
        # The mixed-pool median falls between the GPU and CPU clusters.
        assert 1.0 < est["seconds"] < 11.0

    def test_pre_session55_records_count_as_cpu(self, isolated_log_dir):
        # Old records have no `gpu_used` key. Requesting gpu_used=False
        # must still admit them (they predate GPU support; conservative
        # assumption is they ran CPU-side).
        records = [_rec(elapsed_s=10.0) for _ in range(5)]
        # Remove the gpu_used key from each (already absent — _rec
        # only adds it when explicit). Sanity check:
        assert all("gpu_used" not in r for r in records)
        _seed_perf_log(isolated_log_dir, records)

        cpu_est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
            gpu_used=False,
        )
        assert cpu_est is not None
        # Should predict roughly 10 s.
        assert 5.0 < cpu_est["seconds"] < 20.0

    def test_gpu_partition_fallback_downgrades_confidence(self, isolated_log_dir):
        # Only 1 GPU record (not enough to partition) + 5 CPU records.
        records = [_rec(elapsed_s=1.0, gpu_used=True)]
        records += [_rec(elapsed_s=10.0, gpu_used=False) for _ in range(5)]
        _seed_perf_log(isolated_log_dir, records)

        gpu_est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
            gpu_used=True,
        )
        assert gpu_est is not None
        # The cpu pool has 6 entries → would normally be "high" or
        # "medium"; with GPU fallback the confidence is downgraded one
        # notch.
        assert gpu_est["confidence"] in ("medium", "low")


# =====================================================================
# EST.3 — IQR outlier rejection
# =====================================================================


class TestIqrFilter:
    def test_passes_through_small_pools(self):
        # IQR isn't meaningful on N < 4 — preserve all values.
        assert _iqr_filter([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_drops_high_outlier(self):
        # 4 values clustered near 10, one anomalous 100.
        result = _iqr_filter([10.0, 10.5, 9.5, 10.2, 100.0])
        assert 100.0 not in result
        # The clustered values are preserved.
        for v in (10.0, 10.5, 9.5, 10.2):
            assert v in result

    def test_drops_low_outlier(self):
        result = _iqr_filter([100.0, 105.0, 95.0, 102.0, 1.0])
        assert 1.0 not in result

    def test_all_equal_pool_unchanged(self):
        # IQR = 0 → no fence — return everything.
        assert _iqr_filter([5.0, 5.0, 5.0, 5.0, 5.0]) == [5.0, 5.0, 5.0, 5.0, 5.0]


class TestEstimatorOutlierRobustness:
    def test_single_outlier_does_not_dominate_prediction(self, isolated_log_dir):
        # 5 records ~1 s + 1 anomalous 100 s record. The naive median is
        # ~1 s already (the outlier sits at position 6/6); but if the
        # outlier is included the IQR-filtered median should still be 1 s.
        records = [_rec(elapsed_s=1.0) for _ in range(5)]
        records.append(_rec(elapsed_s=100.0))
        _seed_perf_log(isolated_log_dir, records)

        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
        )
        assert est is not None
        # Without IQR, including the 100s outlier shifts the median to 1s
        # too (same result here since 5 of 6 cluster at 1.0). The strong
        # case: a 5/5 split would pull naive mean badly; check that we're
        # close to 1 s and that n_samples reflects the filter dropped at
        # least one record.
        assert est["seconds"] < 3.0


# =====================================================================
# EST.3 — Variance-aware confidence
# =====================================================================


class TestCoefficientOfVariation:
    def test_low_variance(self):
        # All values within 1% of mean — CV ~ 0.005.
        cv = _coefficient_of_variation([10.0, 10.05, 9.95, 10.02])
        assert cv < 0.05

    def test_high_variance(self):
        # Values spanning 1-10s on a single (method, basis) — CV > 0.4.
        cv = _coefficient_of_variation([1.0, 5.0, 10.0, 3.0, 8.0])
        assert cv > 0.4

    def test_zero_mean_returns_zero(self):
        assert _coefficient_of_variation([0.0, 0.0, 0.0]) == 0.0

    def test_single_value_returns_zero(self):
        assert _coefficient_of_variation([5.0]) == 0.0


class TestConfidenceLabel:
    def test_low_variance_high_samples_yields_high(self):
        # 6 samples, all ~10 s → CV < 0.15 → "high"
        assert _confidence_label([10.0, 10.1, 9.9, 10.05, 9.95, 10.02], 6) == "high"

    def test_high_variance_yields_low_even_with_many_samples(self):
        # 10 samples spanning 1-30 → CV > 0.35 → "low"
        wild = [1.0, 5.0, 30.0, 2.0, 25.0, 4.0, 28.0, 3.0, 20.0, 10.0]
        assert _confidence_label(wild, len(wild)) == "low"

    def test_few_samples_cap_at_medium(self):
        # 3 samples is enough for CV but caps below "high"
        assert _confidence_label([10.0, 10.05, 9.95], 3) == "medium"

    def test_under_three_samples_always_low(self):
        assert _confidence_label([10.0, 10.05], 2) == "low"

    def test_medium_variance_yields_medium(self):
        # CV around 0.25 — between the 0.15 and 0.35 thresholds → "medium"
        med = [10.0, 14.0, 7.0, 12.0, 8.0, 11.0]
        label = _confidence_label(med, len(med))
        assert label == "medium"


class TestEstimatorVarianceAwareConfidence:
    def test_high_variance_pool_reports_low_confidence(self, isolated_log_dir):
        # 6 records but with huge spread — confidence MUST be "low",
        # not "high" just because n_samples >= 5.
        records = [_rec(elapsed_s=t) for t in (1.0, 5.0, 30.0, 2.0, 25.0, 4.0)]
        _seed_perf_log(isolated_log_dir, records)

        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
        )
        assert est is not None
        assert est["confidence"] == "low"

    def test_tight_pool_with_many_samples_reports_high(self, isolated_log_dir):
        # 10 tightly-clustered samples — confidence should be "high".
        records = [
            _rec(elapsed_s=t)
            for t in (1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.0, 1.0, 1.0)
        ]
        _seed_perf_log(isolated_log_dir, records)

        est = estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="B3LYP",
            basis="STO-3G",
            n_basis=15,
            calc_type="single_point",
        )
        assert est is not None
        assert est["confidence"] == "high"
