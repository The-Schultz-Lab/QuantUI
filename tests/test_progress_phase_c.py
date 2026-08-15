"""Estimator overhaul — M-PROGRESS Phase C.

Phase C was opened because M-EST's tier-4 validation showed the pre-run time
estimate sitting outside its ±25 % band. The assumption was that the cost
model needed work. Measuring first (2026-08-05) said otherwise:

* the estimator's median *signed* error was ≈ 0 for every calc type — no bias;
* its median *absolute* error was 37 % overall and 52 % for Frequency — huge
  spread;
* the recorded history contained runs of identical chemistry (H2O RHF/6-31G
  Frequency) with wall times from 0.34 s to 143 s.

No cost model can fit ground truth that disagrees with itself by 400×. The
cause was that the test suite wrote its own runs into the developer's real
``~/.quantui/logs/perf_log.jsonl`` — with a *mocked* calculation, so
``elapsed_s`` was pytest-xdist scheduling noise rather than chemistry. Roughly
four fifths of the 2 773 records were test artifacts.

So Phase C is mostly about **what gets measured and how it is labelled**:

1. tests no longer write to the real log (``conftest._isolate_log_dir``);
2. every record says how it was produced (``source``), so populations that
   measure different things are never averaged together;
3. calibration times the calculation instead of the subprocess;
4. stage wall-times are recorded, so a future model can be stage-aware;
5. :mod:`quantui.estimator_eval` scores the predictor, so the next change to
   it is an experiment rather than an opinion.

Platform-independent: no PySCF, no widgets front-end.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import quantui.app as A
from quantui import calc_log, estimator_eval

# ══ Fixtures ═════════════════════════════════════════════════════════════════


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point the perf/event logs at an isolated directory."""
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    calc_log._READ_ALL_CACHE.clear()
    yield tmp_path
    calc_log._READ_ALL_CACHE.clear()


def _record(**overrides) -> dict:
    """A converged perf record with sane defaults, overridable per test."""
    base = {
        "timestamp": "2026-08-01T00:00:00+00:00",
        "formula": "H2O",
        "n_atoms": 3,
        "n_electrons": 10,
        "method": "RHF",
        "basis": "6-31G",
        "n_iterations": 8,
        "elapsed_s": 10.0,
        "converged": True,
        "n_basis": 13,
        "n_cores": 1,
        "calc_type": "single_point",
    }
    base.update(overrides)
    return base


def _estimate(records, **kwargs):
    params = {
        "n_atoms": 3,
        "n_electrons": 10,
        "method": "RHF",
        "basis": "6-31G",
        "n_basis": 13,
        "calc_type": "single_point",
    }
    params.update(kwargs)
    return calc_log.estimate_time_from_records(records, **params)


class _FakeWidget:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def append_stdout(self, text: str) -> None:
        self.chunks.append(text)


class _FakeLabel:
    def __init__(self, value: str = "") -> None:
        self.value = value


@pytest.fixture(scope="module")
def worker_src() -> str:
    """Source of ``benchmarks._calibration_worker``, for ordering assertions."""
    import quantui.benchmarks as B

    src = Path(B.__file__).read_text(encoding="utf-8")
    start = src.index("def _calibration_worker")
    end = src.index("def _tail_last_status_line")
    return src[start:end]


# ══ The leak that caused Phase C ═════════════════════════════════════════════


class TestSuiteDoesNotWriteToTheRealLog:
    """The regression that mattered most: tests polluting the user's history.

    Without the session fixture, every end-to-end ``_do_run`` test appended a
    fabricated timing to the file the estimator trains on. These tests guard
    the fixture rather than the symptom, because the symptom (a skewed
    estimate) only shows up weeks later in someone's real session.
    """

    def test_log_dir_env_var_is_set_for_the_suite(self):
        assert os.environ.get("QUANTUI_LOG_DIR"), (
            "conftest._isolate_log_dir must set QUANTUI_LOG_DIR for the whole "
            "session; without it the suite appends to ~/.quantui/logs"
        )

    def test_log_dir_is_not_the_users_home(self):
        configured = Path(os.environ["QUANTUI_LOG_DIR"]).resolve()
        real = (Path.home() / ".quantui" / "logs").resolve()
        assert configured != real

    def test_writing_a_record_lands_in_the_isolated_dir(self, log_dir):
        calc_log.log_calculation(
            formula="H2O",
            n_atoms=3,
            n_electrons=10,
            method="RHF",
            basis="6-31G",
            n_iterations=8,
            elapsed_s=1.0,
            converged=True,
        )
        assert (log_dir / "perf_log.jsonl").exists()


# ══ Provenance on the record ═════════════════════════════════════════════════


class TestProvenanceFields:
    def _write(self, **kwargs):
        calc_log.log_calculation(
            formula="H2O",
            n_atoms=3,
            n_electrons=10,
            method="RHF",
            basis="6-31G",
            n_iterations=8,
            elapsed_s=12.0,
            converged=True,
            **kwargs,
        )
        return calc_log.get_perf_history()[-1]

    def test_source_is_recorded(self, log_dir):
        assert self._write(source="app")["source"] == "app"

    def test_warm_is_recorded(self, log_dir):
        assert self._write(warm=True)["warm"] is True

    def test_warm_false_is_recorded_not_dropped(self, log_dir):
        """``warm=False`` is a real measurement, not a missing value."""
        assert self._write(warm=False)["warm"] is False

    def test_import_seconds_recorded(self, log_dir):
        assert self._write(import_s=3.25)["import_s"] == 3.25

    def test_stages_recorded(self, log_dir):
        rec = self._write(stages={"running scf": 4.0, "building hessian": 8.0})
        assert rec["stages"] == {"running scf": 4.0, "building hessian": 8.0}

    def test_absent_fields_are_omitted_entirely(self, log_dir):
        """Additive schema: an untagged record must not gain default keys.

        Readers distinguish "unknown provenance" from "known to be an app
        run", so writing a default here would silently mislabel history.
        """
        rec = self._write()
        for key in ("source", "warm", "import_s", "stages"):
            assert key not in rec

    def test_empty_stages_dict_is_omitted(self, log_dir):
        assert "stages" not in self._write(stages={})


# ══ Pool partitioning by provenance ══════════════════════════════════════════


class TestSourcePartitioning:
    def test_prefers_matching_source(self):
        """An app-sourced prediction ignores calibration timings.

        The two populations measure different things; averaging them is what
        produced Phase C's spread.
        """
        records = [
            _record(source="app", elapsed_s=10.0),
            _record(source="app", elapsed_s=10.0),
            _record(source="calibration", elapsed_s=1000.0),
            _record(source="calibration", elapsed_s=1000.0),
        ]
        est = _estimate(records, source="app")
        assert est is not None
        assert est["seconds"] == pytest.approx(10.0, rel=0.01)

    def test_falls_back_when_matching_pool_is_too_thin(self):
        """One matching record isn't a pool — better a rough answer than none."""
        records = [
            _record(source="app", elapsed_s=10.0),
            _record(source="calibration", elapsed_s=40.0),
            _record(source="calibration", elapsed_s=40.0),
        ]
        est = _estimate(records, source="app")
        assert est is not None
        assert est["seconds"] > 10.0

    def test_fallback_downgrades_confidence(self):
        records = [_record(source="calibration", elapsed_s=10.0) for _ in range(8)]
        matched = _estimate(records, source="calibration")
        fell_back = _estimate(records, source="app")
        assert matched["confidence"] == "high"
        assert fell_back["confidence"] == "medium"

    def test_untagged_history_is_used_only_on_the_fallback_path(self):
        """Legacy records predate ``source`` and must not masquerade as app runs."""
        records = [_record(elapsed_s=10.0) for _ in range(8)]
        est = _estimate(records, source="app")
        assert est is not None
        assert est["confidence"] != "high"

    def test_no_source_requested_ignores_the_axis(self):
        """Back-compat: callers that don't know the provenance still get an answer."""
        records = [
            _record(source="app", elapsed_s=10.0),
            _record(source="calibration", elapsed_s=10.0),
        ]
        est = _estimate(records, source=None)
        assert est is not None
        assert est["confidence"] == "low"  # 2 samples caps at low

    def test_device_and_source_downgrades_compose(self):
        """Two fall-backs should not read as merely one notch of doubt."""
        records = [
            _record(source="calibration", gpu_used=False, elapsed_s=10.0)
            for _ in range(8)
        ]
        est = _estimate(records, source="app", gpu_used=True)
        assert est is not None
        assert est["confidence"] == "low"


class TestEstimateTimeWrapper:
    """``estimate_time`` must stay a thin shim over the records-based core."""

    def test_reads_the_perf_log(self, log_dir):
        path = log_dir / "perf_log.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for _ in range(4):
                fh.write(json.dumps(_record(source="app", elapsed_s=7.0)) + "\n")
        calc_log._READ_ALL_CACHE.clear()
        est = calc_log.estimate_time(
            n_atoms=3,
            n_electrons=10,
            method="RHF",
            basis="6-31G",
            n_basis=13,
            calc_type="single_point",
            source="app",
        )
        assert est is not None
        assert est["seconds"] == pytest.approx(7.0, rel=0.01)

    def test_returns_none_on_empty_history(self, log_dir):
        assert (
            calc_log.estimate_time(
                n_atoms=3,
                n_electrons=10,
                method="RHF",
                basis="6-31G",
                calc_type="single_point",
            )
            is None
        )

    def test_frequency_cost_model_still_fires_without_freq_history(self):
        """The SP-anchored fallback must survive the records refactor."""
        records = [_record(calc_type="single_point", elapsed_s=2.0) for _ in range(4)]
        est = calc_log.estimate_time_from_records(
            records,
            n_atoms=3,
            n_electrons=10,
            method="RHF",
            basis="6-31G",
            n_basis=13,
            calc_type="frequency",
        )
        assert est is not None
        # scf + 2×scf (Hessian) + 6N×scf (IR) = 21 × the 2 s anchor.
        assert est["seconds"] == pytest.approx(42.0, rel=0.05)


# ══ Stage timing ═════════════════════════════════════════════════════════════


class TestStageKey:
    """Status text is written for humans; the timing key has to be stable."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Running SCF…", "running scf"),
            ("Opt step 7 — SCF…", "opt step — scf"),
            ("Opt step 12 — SCF…", "opt step — scf"),
            ("Solving TD-DFT excited states (10)…", "solving td-dft excited states"),
            (
                "Computing NMR shielding tensors (GIAO)…",
                "computing nmr shielding tensors giao",
            ),
            ("Scan point 3/12…", "scan point"),
        ],
    )
    def test_normalisation(self, message, expected):
        assert A._LogCapture._stage_key(message) == expected

    def test_step_counters_collapse_to_one_key(self):
        keys = {
            A._LogCapture._stage_key(f"Opt step {i} — gradient…") for i in range(20)
        }
        assert len(keys) == 1


class TestStageTimings:
    def _capture(self):
        return A._LogCapture(_FakeWidget(), _FakeLabel())

    def test_status_marker_opens_a_stage(self):
        log = self._capture()
        log.write("[QuantUI_STATUS] Running SCF…\n")
        assert "running scf" in log.stage_timings()

    def test_set_status_opens_a_stage(self):
        """Silent (verbose=0) calc types report stages this way, not via the stream."""
        log = self._capture()
        log.set_status("Building Hessian…")
        assert "building hessian" in log.stage_timings()

    def test_open_stage_is_included_not_dropped(self):
        log = self._capture()
        log.write("[QuantUI_STATUS] Running SCF…\n")
        timings = log.stage_timings()
        assert timings["running scf"] >= 0.0

    def test_two_stages_are_tracked_separately(self):
        log = self._capture()
        log.write("[QuantUI_STATUS] Running SCF…\n")
        log.write("[QuantUI_STATUS] Building Hessian…\n")
        timings = log.stage_timings()
        assert set(timings) == {"running scf", "building hessian"}

    def test_repeated_same_stage_stays_one_entry(self):
        """Per-step re-announcements must not shatter the breakdown."""
        log = self._capture()
        for i in range(10):
            log.set_status(f"Opt step {i} — SCF…")
        assert list(log.stage_timings()) == ["opt step — scf"]

    def test_revisited_stage_accumulates(self):
        """SCF runs once per optimizer step; its total is the sum, not the last."""
        log = self._capture()
        log.set_status("Running SCF…")
        log.set_status("Computing gradient…")
        log.set_status("Running SCF…")
        assert set(log.stage_timings()) == {"running scf", "computing gradient"}

    def test_no_stages_before_any_status(self):
        assert self._capture().stage_timings() == {}

    def test_scf_cycle_lines_do_not_create_stages(self):
        """Cycle chatter drives the label but is far too fine-grained to time."""
        log = self._capture()
        log.write("[QuantUI_STATUS] Running SCF…\n")
        log.write("cycle= 1 E= -76.0  delta_E= -1.2e-03\n")
        log.write("cycle= 2 E= -76.1  delta_E= -3.4e-05\n")
        assert list(log.stage_timings()) == ["running scf"]

    def test_status_marker_without_a_label_still_times(self):
        """Stage timing must not depend on a status widget being present."""
        log = A._LogCapture(_FakeWidget(), None)
        log.write("[QuantUI_STATUS] Running SCF…\n")
        assert "running scf" in log.stage_timings()


# ══ Calibration measures the calculation, not the subprocess ═════════════════


class TestCalibrationTiming:
    """``elapsed_s`` must exclude the fresh subprocess's import cost.

    Source-level assertions: spawning a real calibration worker needs PySCF
    and a subprocess, neither of which belongs in the fast suite. What's
    being guarded is the ordering of two statements, which the source shows
    directly.
    """

    def test_compute_clock_starts_after_the_calc_module_import(self, worker_src):
        for module in ("optimizer", "freq_calc", "session_calc"):
            import_at = worker_src.index(f"from quantui.{module} import")
            after = worker_src[import_at:]
            clock_at = after.index("t_compute0 = _t.perf_counter()")
            call_at = after.index("res = _")
            assert clock_at < call_at, (
                f"the {module} branch must start its compute clock between the "
                "import and the call"
            )

    def test_elapsed_is_measured_from_the_compute_clock(self, worker_src):
        assert re.search(r"elapsed\s*=\s*_t_done\s*-\s*t_compute0", worker_src)

    def test_import_cost_is_reported_separately(self, worker_src):
        assert re.search(r"import_s\s*=\s*t_compute0\s*-\s*t0", worker_src)
        assert '"import_s": import_s' in worker_src

    def test_calibration_records_are_tagged(self):
        import quantui.benchmarks as B

        src = Path(B.__file__).read_text(encoding="utf-8")
        assert 'source="calibration"' in src


# ══ Replay harness ═══════════════════════════════════════════════════════════


class TestReplay:
    def test_empty_history_reports_nothing(self):
        report = estimator_eval.replay([])
        assert report.overall.n_total == 0

    def test_prediction_uses_only_earlier_records(self):
        """Causality is the whole point — a hindsight score is worthless.

        The last record is 100× the earlier ones. If the replay leaked it
        into its own history the error would be small; it must be large.
        """
        records = [
            _record(timestamp=f"2026-08-0{i}T00:00:00+00:00", elapsed_s=10.0)
            for i in range(1, 5)
        ] + [_record(timestamp="2026-08-09T00:00:00+00:00", elapsed_s=1000.0)]
        report = estimator_eval.replay(records)
        assert report.overall.errors_pct[-1] > 500

    def test_records_are_replayed_in_timestamp_order(self):
        """Input order must not change the score."""
        records = [
            _record(timestamp=f"2026-08-0{i}T00:00:00+00:00", elapsed_s=10.0 * i)
            for i in range(1, 6)
        ]
        forward = estimator_eval.replay(records).overall.errors_pct
        backward = estimator_eval.replay(list(reversed(records))).overall.errors_pct
        assert forward == backward

    def test_runs_without_an_estimate_are_counted_not_hidden(self):
        """Coverage sits beside accuracy so a silent model can't look good."""
        records = [_record(timestamp="2026-08-01T00:00:00+00:00", elapsed_s=10.0)]
        report = estimator_eval.replay(records)
        assert report.overall.n_no_estimate == 1
        assert report.overall.n_scored == 0
        assert report.overall.coverage_pct == 0.0

    def test_unconverged_records_are_skipped_entirely(self):
        """The estimator was never asked about them; they're neither hit nor miss."""
        records = [_record(converged=False) for _ in range(4)]
        report = estimator_eval.replay(records)
        assert report.overall.n_total == 0

    def test_zero_elapsed_cannot_be_scored(self):
        records = [_record(elapsed_s=0.0) for _ in range(4)]
        assert estimator_eval.replay(records).overall.n_total == 0

    def test_error_is_signed_relative_to_the_prediction(self):
        """A run twice as long as predicted is +100 %, not −50 %."""
        records = [
            _record(timestamp=f"2026-08-0{i}T00:00:00+00:00", elapsed_s=10.0)
            for i in range(1, 5)
        ] + [_record(timestamp="2026-08-05T00:00:00+00:00", elapsed_s=20.0)]
        report = estimator_eval.replay(records)
        assert report.overall.errors_pct[-1] == pytest.approx(100.0, rel=0.02)

    def test_slices_group_by_calc_type_by_default(self):
        records = [
            _record(
                timestamp=f"2026-08-0{i}T00:00:00+00:00",
                calc_type="frequency" if i % 2 else "single_point",
            )
            for i in range(1, 7)
        ]
        report = estimator_eval.replay(records)
        assert set(report.slices) == {"frequency", "single_point"}

    def test_slices_can_group_by_source(self):
        records = [
            _record(
                timestamp=f"2026-08-0{i}T00:00:00+00:00",
                source="app" if i % 2 else "calibration",
            )
            for i in range(1, 7)
        ]
        report = estimator_eval.replay(records, slice_by="source")
        assert set(report.slices) == {"app", "calibration"}

    def test_untagged_records_slice_as_unset_not_dropped(self):
        records = [
            _record(timestamp=f"2026-08-0{i}T00:00:00+00:00") for i in range(1, 5)
        ]
        report = estimator_eval.replay(records, slice_by="source")
        assert "(unset)" in report.slices

    def test_use_source_false_scores_the_pre_phase_c_behaviour(self):
        """Before/after comparison needs the old behaviour to stay reachable.

        With partitioning on, the app-sourced run is predicted from app
        history alone and lands on the mark; with it off, the calibration
        records drag the prediction away.
        """
        records = (
            [
                _record(
                    timestamp=f"2026-08-0{i}T00:00:00+00:00",
                    source="app",
                    elapsed_s=10.0,
                )
                for i in range(1, 5)
            ]
            + [
                _record(
                    timestamp=f"2026-08-0{i}T00:00:00+00:00",
                    source="calibration",
                    elapsed_s=500.0,
                )
                for i in range(5, 9)
            ]
            + [
                _record(
                    timestamp="2026-08-09T00:00:00+00:00", source="app", elapsed_s=10.0
                )
            ]
        )
        partitioned = estimator_eval.replay(records).overall.errors_pct[-1]
        mixed = estimator_eval.replay(records, use_source=False).overall.errors_pct[-1]
        assert abs(partitioned) < abs(mixed)


class TestReplayStats:
    def test_within_band_is_a_percentage_of_scored_runs(self):
        stats = estimator_eval.ReplayStats(
            label="x", n_scored=4, errors_pct=[0, 10, 30, 60]
        )
        assert stats.within_pct(25.0) == pytest.approx(50.0)

    def test_bias_and_spread_are_reported_separately(self):
        """±50 % errors average to zero bias — the distinction Phase C turned on."""
        stats = estimator_eval.ReplayStats(
            label="x", n_scored=4, errors_pct=[-50, -50, 50, 50]
        )
        assert stats.median_signed_error_pct == pytest.approx(0.0)
        assert stats.median_abs_error_pct == pytest.approx(50.0)

    def test_empty_stats_report_none_rather_than_zero(self):
        """Zero error and no data must not look alike."""
        stats = estimator_eval.ReplayStats(label="x")
        assert stats.median_abs_error_pct is None
        assert stats.within_pct(25.0) is None

    def test_coverage_counts_both_outcomes(self):
        stats = estimator_eval.ReplayStats(label="x", n_scored=3, n_no_estimate=1)
        assert stats.n_total == 4
        assert stats.coverage_pct == pytest.approx(75.0)


class TestFormatReport:
    def test_renders_a_row_per_slice_plus_overall(self):
        records = [
            _record(timestamp=f"2026-08-0{i}T00:00:00+00:00") for i in range(1, 6)
        ]
        text = estimator_eval.format_report(estimator_eval.replay(records))
        assert "single_point" in text
        assert "overall" in text

    def test_missing_values_render_as_a_dash(self):
        report = estimator_eval.ReplayReport(
            overall=estimator_eval.ReplayStats(label="overall", n_no_estimate=2)
        )
        assert "—" in estimator_eval.format_report(report)


class TestCli:
    def test_reports_cleanly_with_no_history(self, log_dir, capsys):
        assert estimator_eval.main([]) == 0
        assert "No performance history" in capsys.readouterr().out

    def test_prints_a_table_when_history_exists(self, log_dir, capsys):
        path = log_dir / "perf_log.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(1, 6):
                fh.write(
                    json.dumps(
                        _record(timestamp=f"2026-08-0{i}T00:00:00+00:00", source="app")
                    )
                    + "\n"
                )
        calc_log._READ_ALL_CACHE.clear()
        assert estimator_eval.main([]) == 0
        assert "overall" in capsys.readouterr().out

    def test_by_source_flag_switches_the_grouping(self, log_dir, capsys):
        path = log_dir / "perf_log.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(1, 6):
                fh.write(
                    json.dumps(
                        _record(timestamp=f"2026-08-0{i}T00:00:00+00:00", source="app")
                    )
                    + "\n"
                )
        calc_log._READ_ALL_CACHE.clear()
        assert estimator_eval.main(["--by-source"]) == 0
        assert "app" in capsys.readouterr().out
