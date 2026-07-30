"""Silent-phase heartbeat for the live log (M-PROGRESS Phase D).

Motivating measurement (user, 2026-07-30): an aspirin B3LYP/6-31G* UV-Vis run
printed **nothing for 120 s** after "converged SCF energy" while the TD-DFT
solve ran. The status label advanced the whole time — Phase A covers that — but
the output log, which is what a user actually watches, looked frozen.

Phase A solved this for the status *label*; this is the same problem for the
*log*. Tests shrink the intervals so behaviour is observable in under a second.

Platform-independent: no PySCF, no widgets front-end.
"""

from __future__ import annotations

import threading
import time

import pytest

import quantui.app as A


class _FakeWidget:
    """Stand-in for the live log, recording what was appended."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self._lock = threading.Lock()

    def append_stdout(self, text: str) -> None:
        with self._lock:
            self.chunks.append(text)

    def beats(self) -> list[str]:
        with self._lock:
            return [c for c in self.chunks if "still working" in c]


class _FakeLabel:
    def __init__(self, value: str = "") -> None:
        self.value = value


@pytest.fixture
def fast_heartbeat(monkeypatch):
    """Compress the real 25 s / 2 s cadence into something testable."""
    monkeypatch.setattr(A, "_HEARTBEAT_AFTER_S", 0.2)
    monkeypatch.setattr(A, "_HEARTBEAT_POLL_S", 0.02)


@pytest.fixture
def log_and_widget(fast_heartbeat):
    w = _FakeWidget()
    log = A._LogCapture(w, _FakeLabel("Solving TD-DFT excited states (10)…"))
    yield log, w
    log.stop_heartbeat()


class TestHeartbeatFires:
    def test_appears_during_a_silent_stretch(self, log_and_widget):
        log, w = log_and_widget
        log.start_heartbeat()
        time.sleep(0.7)
        assert w.beats(), "no heartbeat during a silent stretch"

    def test_repeats_at_roughly_the_configured_interval(self, log_and_widget):
        # One beat then silence would be almost as bad as none — the point is a
        # continuing sign of life.
        log, w = log_and_widget
        log.start_heartbeat()
        time.sleep(0.75)
        assert len(w.beats()) >= 2

    def test_includes_the_current_stage(self, log_and_widget):
        # "still working" alone doesn't say *what* is working; the stage label is
        # the difference between reassurance and noise.
        log, w = log_and_widget
        log.start_heartbeat()
        time.sleep(0.4)
        assert any("TD-DFT" in b for b in w.beats())

    def test_includes_elapsed_time(self, log_and_widget):
        log, w = log_and_widget
        log.start_heartbeat()
        time.sleep(0.4)
        assert any("elapsed" in b for b in w.beats())

    def test_tolerates_a_missing_status_label(self, fast_heartbeat):
        # _LogCapture accepts status_label=None; the heartbeat must not assume it.
        w = _FakeWidget()
        log = A._LogCapture(w, None)
        log.start_heartbeat()
        time.sleep(0.4)
        log.stop_heartbeat()
        assert w.beats()


class TestHeartbeatStaysQuiet:
    def test_real_output_resets_the_timer(self, log_and_widget):
        # A run that is printing steadily must never show a heartbeat.
        log, w = log_and_widget
        log.start_heartbeat()
        for _ in range(8):
            log.write("cycle= 1 E= -1.0 delta_E= -1e-5\n")
            time.sleep(0.05)
        assert not w.beats()

    def test_no_beats_before_start(self, log_and_widget):
        log, w = log_and_widget
        time.sleep(0.4)
        assert not w.beats()

    def test_stop_halts_further_beats(self, log_and_widget):
        log, w = log_and_widget
        log.start_heartbeat()
        time.sleep(0.4)
        log.stop_heartbeat()
        n = len(w.beats())
        time.sleep(0.4)
        assert len(w.beats()) == n

    def test_stop_is_safe_without_start(self, log_and_widget):
        log, _w = log_and_widget
        log.stop_heartbeat()  # must not raise

    def test_start_is_idempotent(self, log_and_widget):
        # Two threads would double the beat rate.
        log, w = log_and_widget
        log.start_heartbeat()
        log.start_heartbeat()
        time.sleep(0.45)
        assert len(w.beats()) <= 3


class TestHeartbeatDoesNotCorruptTheRecord:
    def test_beats_are_absent_from_the_captured_buffer(self, log_and_widget):
        # getvalue() becomes the result directory's pyscf.log, which should stay
        # a faithful record of PySCF's own output. Heartbeats are UI chrome.
        log, w = log_and_widget
        log.start_heartbeat()
        log.write("real output\n")
        time.sleep(0.5)
        assert w.beats()  # shown live
        assert "still working" not in log.getvalue()  # not archived
        assert "real output" in log.getvalue()

    def test_beat_does_not_raise_cancellation_on_its_own_thread(self, fast_heartbeat):
        # write() raises _CalcCancelled when cancellation is requested. If the
        # heartbeat went through write(), it would raise on the watchdog thread
        # where nothing can catch it, killing the beat silently.
        w = _FakeWidget()
        log = A._LogCapture(w, _FakeLabel("stage"), cancel_check=lambda: True)
        log.start_heartbeat()
        time.sleep(0.4)
        log.stop_heartbeat()
        assert w.beats(), "heartbeat suppressed by the cancel check"
