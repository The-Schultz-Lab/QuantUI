"""Tests for graceful calculation cancellation + the Clear-while-running guard.

Background (2026-06-16 user report): clicking **Clear** mid-run wiped the live
log header while the background calc kept appending steps, leaving a confusing
headerless output. Two fixes: (1) Clear is disabled + guarded while a calc runs;
(2) a **Cancel** button cooperatively stops a run at the next output line via
``_LogCapture`` raising ``_CalcCancelled``.

Platform-independent (no PySCF): exercises the mechanism, the handler, and the
guard directly.
"""

from __future__ import annotations

import ipywidgets as widgets
import pytest

from quantui.app import _CalcCancelled, _LogCapture
from quantui.app_runflow import on_clear_log

# ── _LogCapture cancellation mechanism ──────────────────────────────────────


class TestLogCaptureCancel:
    def test_write_raises_when_cancel_flag_set(self):
        out = widgets.Output()
        cap = _LogCapture(out, cancel_check=lambda: True)
        with pytest.raises(_CalcCancelled):
            cap.write("cycle= 1 E= -76.0\n")

    def test_write_appends_when_cancel_flag_clear(self):
        out = widgets.Output()
        cap = _LogCapture(out, cancel_check=lambda: False)
        cap.write("hello\n")
        joined = "".join(o.get("text", "") for o in out.outputs)
        assert "hello" in joined

    def test_write_appends_when_no_cancel_check(self):
        out = widgets.Output()
        cap = _LogCapture(out)  # cancel_check defaults to None
        cap.write("world\n")
        joined = "".join(o.get("text", "") for o in out.outputs)
        assert "world" in joined

    def test_cancel_check_is_consulted_live(self):
        """Flips mid-stream: writes before the flag succeed, after raise."""
        out = widgets.Output()
        flag = {"v": False}
        cap = _LogCapture(out, cancel_check=lambda: flag["v"])
        cap.write("step 0\n")  # ok
        flag["v"] = True
        with pytest.raises(_CalcCancelled):
            cap.write("step 1\n")


# ── App-level: Cancel handler + Clear guard ─────────────────────────────────


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Isolate settings (reflections/10 Rule 6) so construction never touches the
    # real ~/.quantui under pytest-xdist.
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


class TestCancelHandler:
    def test_cancel_while_running_sets_event_and_disables_button(self, app):
        app._calc_running = True
        app.cancel_btn.disabled = False
        app._cancel_event.clear()

        app._on_cancel()

        assert app._cancel_event.is_set()
        assert app.cancel_btn.disabled is True
        assert "ancel" in app.run_status.value  # "Cancelling…"

    def test_cancel_when_not_running_is_noop(self, app):
        app._calc_running = False
        app._cancel_event.clear()

        app._on_cancel()

        assert not app._cancel_event.is_set()

    def test_cancel_button_disabled_by_default(self, app):
        # Nothing running at construction → Cancel is inert.
        assert app.cancel_btn.disabled is True

    def test_cancel_cleanup_write_does_not_reraise(self, app):
        # Regression: _do_run's ``except _CalcCancelled`` writes a footer line
        # through the same _LogCapture. With the cancel flag STILL set, that
        # write re-raises _CalcCancelled, propagates out of the handler, and
        # skips the cancelled card + ``run_status = "Cancelled."`` (only
        # ``finally`` runs → status stuck on "Cancelling…"). The fix clears the
        # flag before writing; lock in that order.
        cap = _LogCapture(
            app.run_output, app.run_status, cancel_check=app._cancel_event.is_set
        )
        app._cancel_event.set()
        # Buggy order (write while still set) would raise — that was the bug.
        with pytest.raises(_CalcCancelled):
            cap.write("step\n")
        # The fix: clear first, then the cleanup write must NOT raise.
        app._cancel_event.clear()
        cap.write("\n── Calculation cancelled by user ──\n")  # no exception


class TestClearGuard:
    """Spy on ``clear_output`` rather than asserting an emptied ``.outputs`` —
    ipywidgets' ``clear_output`` doesn't synchronously empty a headless widget,
    so the guard *logic* (was clear called?) is the reliable contract to test."""

    def test_clear_blocked_while_running(self, app, monkeypatch):
        calls = []
        monkeypatch.setattr(
            app.run_output, "clear_output", lambda *a, **k: calls.append(1)
        )
        app._calc_running = True

        on_clear_log(app, None)

        assert calls == []  # clear NOT invoked mid-run
        assert "running" in app.run_status.value.lower()

    def test_clear_works_when_idle(self, app, monkeypatch):
        calls = []
        monkeypatch.setattr(
            app.run_output, "clear_output", lambda *a, **k: calls.append(1)
        )
        app._calc_running = False

        on_clear_log(app, None)

        assert calls == [1]  # clear invoked once when idle
