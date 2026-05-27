"""Tests for the calibration resilience fixes (session 55 user report).

User-reported issues these tests guard against:

1. Status indicator stayed "Idle" during calibration — covered by the
   ``_activity_begin/_end`` wrapper in ``app_runflow.do_calibration``.
   Not directly testable here (UI side); covered by the wrapper's
   presence-in-source check below.
2. No per-step progress visibility — ``_tail_last_status_line``
   returns the most recent meaningful log line; tested directly.
3. ``calibration.json`` dropped state on interrupt —
   ``_save_calibration_json`` is now called after every step (not just
   end-of-loop). Verified by reading source markers + a unit test on
   the helper itself.
4. Stop button didn't work mid-calc — ``run_calibration`` now uses
   ``multiprocessing.Process`` so ``worker.terminate()`` cleanly
   interrupts an in-flight step. The poll-loop logic is tested via
   structure check; the actual termination is exercised by the
   PySCF-gated integration test in ``test_benchmarks.py``.
5. Calibration log file — ``_calibration_log_path`` returns a path
   under ``QUANTUI_LOG_DIR``; tested directly.

All tests are platform-independent.
"""

from __future__ import annotations

import inspect
import json

import pytest

from quantui import benchmarks
from quantui.benchmarks import (
    BenchmarkStep,
    CalibrationResult,
    _calibration_log_path,
    _save_calibration_json,
    _tail_last_status_line,
)


@pytest.fixture
def isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    return tmp_path


# =====================================================================
# _calibration_log_path
# =====================================================================


class TestCalibrationLogPath:
    def test_respects_quantui_log_dir(self, isolated_log_dir):
        path = _calibration_log_path("2026-05-25T12:00:00+00:00")
        # Lives under QUANTUI_LOG_DIR exactly.
        assert path.parent == isolated_log_dir

    def test_filename_includes_timestamp(self, isolated_log_dir):
        path = _calibration_log_path("2026-05-25T12:34:56+00:00")
        assert path.name.startswith("calibration_")
        assert path.name.endswith(".log")
        # The timestamp is in the filename (sanitized — no colons since
        # Windows file systems reject them).
        assert ":" not in path.name
        assert "2026-05-25" in path.name


# =====================================================================
# _tail_last_status_line
# =====================================================================


class TestTailLastStatusLine:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _tail_last_status_line(tmp_path / "nope.log") == ""

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("", encoding="utf-8")
        assert _tail_last_status_line(p) == ""

    def test_prefers_quantui_status_marker(self, tmp_path):
        p = tmp_path / "log.log"
        p.write_text(
            "some random PySCF output\n"
            "[QuantUI_STATUS] Computing Hessian (3/12)\n"
            "more PySCF noise after the marker\n",
            encoding="utf-8",
        )
        out = _tail_last_status_line(p)
        # The QuantUI_STATUS line wins even though it's not the last.
        assert "[QuantUI_STATUS]" in out
        assert "Hessian" in out

    def test_falls_back_to_last_non_blank(self, tmp_path):
        p = tmp_path / "log.log"
        p.write_text(
            "SCF iter 1  E=-1.0\n" "SCF iter 2  E=-1.5\n" "SCF converged\n" "\n",
            encoding="utf-8",
        )
        # No status marker → return the last non-blank line.
        assert _tail_last_status_line(p) == "SCF converged"

    def test_truncates_long_lines(self, tmp_path):
        p = tmp_path / "log.log"
        long_line = "A" * 500
        p.write_text(long_line + "\n", encoding="utf-8")
        out = _tail_last_status_line(p)
        # Hard cap is 120 chars in the helper.
        assert len(out) <= 120


# =====================================================================
# _save_calibration_json
# =====================================================================


class TestSaveCalibrationJson:
    def test_writes_to_user_home(self, monkeypatch, tmp_path):
        # Redirect HOME so the helper writes into tmp_path, not
        # ~/.quantui (which would clobber a real user setup).
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        # On some platforms Path.home() caches; patch directly too.
        from pathlib import Path as _Path

        monkeypatch.setattr(_Path, "home", lambda: tmp_path)

        result = CalibrationResult(timestamp="2026-05-25T12:00:00+00:00", mode="tier1")
        result.steps.append(
            BenchmarkStep(
                label="H2 RHF/STO-3G",
                method="RHF",
                basis="STO-3G",
                n_atoms=2,
                n_electrons=2,
                status="ok",
                elapsed_s=0.5,
                n_basis=2,
                calc_type="single_point",
            )
        )
        log_path = tmp_path / "fake.log"

        _save_calibration_json(result, log_path)
        cal_path = tmp_path / ".quantui" / "calibration.json"
        assert cal_path.exists()
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        assert data["mode"] == "tier1"
        assert data["n_completed"] == 1
        assert data["steps"][0]["label"] == "H2 RHF/STO-3G"
        assert data["log_path"] == str(log_path)

    def test_partial_state_persisted_on_interrupt(self, monkeypatch, tmp_path):
        # Simulates the user's scenario: tier 4 stopped at step 25/30.
        # After the partial save, the on-disk record should show
        # n_completed=24 (or however many ran) + stopped_early=True.
        from pathlib import Path as _Path

        monkeypatch.setattr(_Path, "home", lambda: tmp_path)

        result = CalibrationResult(
            timestamp="2026-05-25T12:00:00+00:00",
            mode="tier4",
            stopped_early=True,
        )
        # Add 24 ok steps + 1 stopped step.
        for i in range(24):
            result.steps.append(
                BenchmarkStep(
                    label=f"step-{i}",
                    method="RHF",
                    basis="STO-3G",
                    n_atoms=2,
                    n_electrons=2,
                    status="ok",
                    elapsed_s=1.0,
                    n_basis=2,
                    calc_type="single_point",
                )
            )
        result.steps.append(
            BenchmarkStep(
                label="step-stop",
                method="B3LYP",
                basis="6-31G*",
                n_atoms=12,
                n_electrons=42,
                status="stopped",
                elapsed_s=300.0,
                n_basis=96,
                calc_type="frequency",
            )
        )

        _save_calibration_json(result, tmp_path / "fake.log")
        cal_path = tmp_path / ".quantui" / "calibration.json"
        data = json.loads(cal_path.read_text(encoding="utf-8"))

        # User's actual complaint was that this dropped to None on
        # interrupt. After the fix, the 24 completed runs must be on
        # disk.
        assert data["n_completed"] == 24
        assert data["stopped_early"] is True
        assert len(data["steps"]) == 25
        # The stopped step is the last one.
        assert data["steps"][-1]["status"] == "stopped"


# =====================================================================
# Source-level structure checks (defend against regression)
# =====================================================================


class TestRunCalibrationStructure:
    """The fix touches ``run_calibration`` heavily. These tests assert
    that key invariants of the new design are still present in the
    source — so a future refactor that drops them fails loudly.
    """

    def test_uses_multiprocessing_process_not_thread_executor(self):
        src = inspect.getsource(benchmarks.run_calibration)
        # The Stop-button-mid-calc fix requires a process, not a
        # ThreadPoolExecutor — threads can't be terminated externally.
        assert "_mp.Process" not in src  # we use _ctx.Process from a context
        assert "Process" in src
        assert "ThreadPoolExecutor" not in src

    def test_poll_loop_checks_stop_event(self):
        src = inspect.getsource(benchmarks.run_calibration)
        # The poll loop must check ``stop_event.is_set()`` so the stop
        # button reaches the worker within poll_interval (500 ms).
        assert "stop_event" in src
        assert "is_set()" in src
        assert ".terminate()" in src

    def test_saves_calibration_after_every_step(self):
        src = inspect.getsource(benchmarks.run_calibration)
        # Count _save_calibration_json invocations inside the loop.
        # Should be at least 2: one inside the PySCF-unavailable
        # branch, one after the main step completes. Plus the final
        # idempotent write outside the loop.
        n = src.count("_save_calibration_json")
        assert n >= 3

    def test_opens_log_file_at_start(self):
        src = inspect.getsource(benchmarks.run_calibration)
        # The per-run log file (the user requested this for tier 4)
        # is opened with "w" mode at the top of the run.
        assert "_calibration_log_path" in src
        assert '"w"' in src or "'w'" in src


class TestDoCalibrationStructure:
    """``app_runflow.do_calibration`` got the ``_activity_begin/_end``
    wrap so the toolbar badge stops reading 'Idle' during calibration.
    """

    def test_wraps_calibration_in_activity_markers(self):
        from quantui import app_runflow

        src = inspect.getsource(app_runflow.do_calibration)
        # The Status-indicator-says-Idle fix (user's first complaint).
        assert "_activity_begin" in src
        assert "_activity_end" in src
        # Must be in a try/finally so a calibration crash still flips
        # the badge back.
        assert "finally" in src
