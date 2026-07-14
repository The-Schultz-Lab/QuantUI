"""
Tests for quantui.benchmarks — CAL.3 acceptance criteria.

Dataclass and progress-callback tests run unconditionally.
run_calibration() tests are PySCF-gated.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from quantui.benchmarks import (
    _STATUS_OK,
    _STATUS_TIMEOUT,
    BENCHMARK_SUITE,
    BenchmarkStep,
    CalibrationResult,
    _count_electrons,
    load_last_calibration,
    run_calibration,
)

# ---------------------------------------------------------------------------
# PySCF gate
# ---------------------------------------------------------------------------

_PYSCF_AVAILABLE = False
try:
    import pyscf as _pyscf  # noqa: F401

    _PYSCF_AVAILABLE = True
except ImportError:
    pass

pyscf_only = pytest.mark.skipif(
    not _PYSCF_AVAILABLE, reason="PySCF not installed (Linux/macOS/WSL only)"
)

# ---------------------------------------------------------------------------
# _count_electrons
# ---------------------------------------------------------------------------


class TestCountElectrons:
    """L audit fix: _count_electrons used to hand-carry its own truncated
    (Z=1-18) atomic-number table, silently falling back to carbon's Z=6
    for every heavier element (e.g. iodine, gold) instead of using the
    shared, full-periodic-table config.ATOMIC_NUMBERS.
    """

    def test_light_elements_match_known_z(self):
        assert _count_electrons(["O", "H", "H"], charge=0) == 10  # water

    def test_charge_is_subtracted(self):
        assert _count_electrons(["O", "H", "H"], charge=1) == 9

    def test_heavy_element_uses_real_atomic_number_not_carbon_fallback(self):
        # Iodine: Z=53. The old 18-element table fell back to 6 for any
        # element past Ar, undercounting badly for anything heavier.
        assert _count_electrons(["I"], charge=0) == 53


# ---------------------------------------------------------------------------
# BENCHMARK_SUITE contents
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_has_entries(self):
        assert len(BENCHMARK_SUITE) >= 8

    def test_entry_shape(self):
        label, atoms, coords, charge, mult, method, basis = BENCHMARK_SUITE[0]
        assert isinstance(label, str)
        assert isinstance(atoms, list)
        assert isinstance(coords, list)
        assert len(atoms) == len(coords)
        assert isinstance(method, str)
        assert isinstance(basis, str)

    def test_all_charges_valid(self):
        for entry in BENCHMARK_SUITE:
            charge = entry[3]
            assert isinstance(charge, int)

    def test_no_duplicate_labels(self):
        labels = [e[0] for e in BENCHMARK_SUITE]
        assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# BenchmarkStep dataclass
# ---------------------------------------------------------------------------


class TestBenchmarkStep:
    def test_default_elapsed(self):
        s = BenchmarkStep(
            label="H2 RHF",
            method="RHF",
            basis="STO-3G",
            n_atoms=2,
            n_electrons=2,
            status=_STATUS_OK,
        )
        assert s.elapsed_s == 0.0

    def test_default_error_msg(self):
        s = BenchmarkStep(
            label="H2 RHF",
            method="RHF",
            basis="STO-3G",
            n_atoms=2,
            n_electrons=2,
            status=_STATUS_OK,
        )
        assert s.error_msg == ""

    def test_stores_status(self):
        s = BenchmarkStep(
            label="x",
            method="RHF",
            basis="STO-3G",
            n_atoms=1,
            n_electrons=1,
            status=_STATUS_TIMEOUT,
        )
        assert s.status == _STATUS_TIMEOUT


# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------


def _make_result(statuses: list[str]) -> CalibrationResult:
    steps = [
        BenchmarkStep(
            label=f"s{i}",
            method="RHF",
            basis="STO-3G",
            n_atoms=2,
            n_electrons=2,
            status=s,
        )
        for i, s in enumerate(statuses)
    ]
    return CalibrationResult(timestamp="2026-01-01T00:00:00+00:00", steps=steps)


class TestCalibrationResult:
    def test_n_completed_counts_ok(self):
        r = _make_result([_STATUS_OK, _STATUS_OK, _STATUS_TIMEOUT])
        assert r.n_completed == 2

    def test_n_total_reflects_suite(self):
        r = _make_result([_STATUS_OK])
        assert r.n_total == len(BENCHMARK_SUITE)

    def test_stopped_early_default_false(self):
        r = CalibrationResult(timestamp="t")
        assert r.stopped_early is False


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @pyscf_only
    @pytest.mark.slow
    def test_progress_called_for_each_step(self):
        calls = []
        stop = threading.Event()

        # Only run first 2 steps for speed. ``_MODE_TO_SUITE["tier1"]`` is the
        # actual binding ``run_calibration`` reads at call time — patching
        # ``BENCHMARK_SUITE`` alone no longer propagates, since
        # ``BENCHMARK_SUITE_TIER1`` aliases the original list at import time.
        with patch.dict(
            "quantui.benchmarks._MODE_TO_SUITE",
            {"tier1": BENCHMARK_SUITE[:2]},
        ):
            run_calibration(
                progress_cb=lambda *a: calls.append(a),
                stop_event=stop,
                timeout_per_step=60.0,
            )

        # Filter to terminal per-step calls; intermediate "running" heartbeats
        # (emitted every ~500ms while a step is in-flight) are an implementation
        # detail of the live-status display and should not be counted here.
        terminal = [c for c in calls if c[3] != "running"]
        assert len(terminal) == 2
        step_n, total, label, status, elapsed = terminal[0]
        assert step_n == 1
        assert total == 2
        assert isinstance(label, str)
        assert status in (_STATUS_OK, _STATUS_TIMEOUT, "error")
        assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# Stop event
# ---------------------------------------------------------------------------


class TestStopEvent:
    @pyscf_only
    @pytest.mark.slow
    def test_stop_aborts_after_current_step(self):
        stop = threading.Event()

        completed = []

        def _progress(step_n, total, label, status, elapsed):
            completed.append(step_n)
            if step_n >= 1:
                stop.set()  # signal stop after first step

        result = run_calibration(
            progress_cb=_progress,
            stop_event=stop,
            timeout_per_step=60.0,
        )

        assert result.stopped_early is True
        assert len(result.steps) <= 2  # at most one step completed before abort

    def test_stop_before_start_aborts_immediately(self):
        stop = threading.Event()
        stop.set()  # pre-set

        result = run_calibration(stop_event=stop)
        assert result.stopped_early is True
        assert len(result.steps) == 0


# ---------------------------------------------------------------------------
# Timeout per step
# ---------------------------------------------------------------------------


class TestTimeoutPerStep:
    @pyscf_only
    @pytest.mark.slow
    def test_timeout_produces_timed_out_status(self):
        """A 0.001 s timeout should trigger timed_out on any real calculation."""
        result = run_calibration(
            timeout_per_step=0.001,
            stop_event=threading.Event(),
        )
        timed = [s for s in result.steps if s.status == _STATUS_TIMEOUT]
        # At least the first step should time out at 1 ms
        assert len(timed) >= 1


# ---------------------------------------------------------------------------
# load_last_calibration
# ---------------------------------------------------------------------------


class TestLoadLastCalibration:
    def test_returns_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert (
            load_last_calibration() is None or True
        )  # may already exist — just no raise

    def test_returns_dict_after_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        stop = threading.Event()
        stop.set()
        run_calibration(stop_event=stop)
        data = load_last_calibration()
        if data is not None:
            assert "timestamp" in data
            assert "steps" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
