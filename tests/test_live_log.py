"""Tests for the QuantUI-owned live log (M-LOGSCROLL route C).

The scroll behaviour itself is browser-side and can only be verified in a live
Voilà session (LOGSCROLL.0 was done that way). What is testable here is the part
that must not regress silently: the drop-in contract with ``widgets.Output``, the
authoritative Python-side text, and the thread-safe buffering — because
``append_stdout`` is called from the background calc thread.
"""

from __future__ import annotations

import threading

import ipywidgets as widgets

from quantui.live_log import LiveLog


def _log() -> LiveLog:
    return LiveLog(uid="test")


class TestOutputCompatibility:
    """Existing call sites use these three; they must keep working."""

    def test_is_a_widget_so_it_can_sit_in_a_box(self):
        lg = _log()
        assert isinstance(lg, widgets.Widget)
        widgets.VBox([lg])  # must not raise

    def test_append_stdout_accumulates(self):
        lg = _log()
        lg.append_stdout("a\n")
        lg.append_stdout("b\n")
        assert lg.text == "a\nb\n"

    def test_outputs_setter_replaces_atomically(self):
        # The run-header write is a single .outputs assignment specifically to
        # avoid a clear-then-append race (the pre-step-1 blank-window bug).
        lg = _log()
        lg.append_stdout("stale\n")
        lg.outputs = ({"output_type": "stream", "name": "stdout", "text": "HDR\n"},)
        assert lg.text == "HDR\n"

    def test_outputs_setter_joins_multiple_entries(self):
        lg = _log()
        lg.outputs = (
            {"output_type": "stream", "name": "stdout", "text": "one\n"},
            {"output_type": "stream", "name": "stdout", "text": "two\n"},
        )
        assert lg.text == "one\ntwo\n"

    def test_outputs_setter_tolerates_empty(self):
        lg = _log()
        lg.append_stdout("x")
        lg.outputs = ()
        assert lg.text == ""

    def test_outputs_getter_roundtrips(self):
        lg = _log()
        lg.append_stdout("hello\n")
        assert lg.outputs == (
            {"output_type": "stream", "name": "stdout", "text": "hello\n"},
        )

    def test_outputs_getter_empty_when_blank(self):
        assert _log().outputs == ()

    def test_clear_output_accepts_output_kwargs(self):
        # Call sites may pass wait=True as they would to widgets.Output.
        lg = _log()
        lg.append_stdout("x")
        lg.clear_output(wait=True)
        assert lg.text == ""

    def test_add_class_still_available(self):
        _log().add_class("quantui-run-output")  # must not raise


class TestBuffering:
    def test_text_includes_unflushed_pending(self):
        # A reader must never see a partial log just because a flush is pending.
        lg = _log()
        lg.append_stdout("buffered")
        assert "buffered" in lg.text

    def test_flush_is_idempotent(self):
        lg = _log()
        lg.append_stdout("x\n")
        lg.flush()
        lg.flush()
        assert lg.text == "x\n"

    def test_empty_append_is_a_noop(self):
        lg = _log()
        lg.append_stdout("")
        assert lg.text == ""

    def test_set_text_discards_pending(self):
        # Otherwise buffered output could land *after* a header write and
        # reappear above it.
        lg = _log()
        lg.append_stdout("pending")
        lg.set_text("REPLACED")
        assert lg.text == "REPLACED"

    def test_concurrent_appends_do_not_lose_text(self):
        # append_stdout is called from the background calc thread.
        lg = _log()
        n_threads, per_thread = 8, 50

        def worker(tag: int) -> None:
            for i in range(per_thread):
                lg.append_stdout(f"{tag}-{i}\n")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lg.flush()

        lines = [ln for ln in lg.text.split("\n") if ln]
        assert len(lines) == n_threads * per_thread
        assert len(set(lines)) == n_threads * per_thread  # none lost or doubled


class TestBackgroundThreadEmission:
    """Regression guard for the 2026-07-30 "header prints, then nothing" bug.

    ``widgets.Output.__enter__`` captures output by recording the *current parent
    message id*. A background thread has no parent-message context, so a JS
    payload emitted directly from the calc thread never reaches the frontend —
    which is exactly what happened: the run header appeared (written from the
    click handler, i.e. the main thread) and every streaming line after it was
    silently dropped. Emission must therefore be marshalled to the main thread.
    """

    def test_appends_from_a_worker_thread_are_marshalled(self):
        calls: list[str] = []
        lg = LiveLog(uid="sched", schedule=lambda fn, *a: calls.append("scheduled"))

        def worker() -> None:
            lg.append_stdout("from the calc thread\n")
            lg.flush()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert calls, "append from a worker thread did not go through the marshaller"

    def test_header_write_is_also_marshalled(self):
        # set_text runs on the main thread today, but routing it too means the
        # contract does not depend on which thread the caller happens to be on.
        calls: list[str] = []
        lg = LiveLog(uid="sched2", schedule=lambda fn, *a: calls.append("scheduled"))
        lg.set_text("HDR\n")
        assert calls

    def test_scheduler_receives_a_callable_and_payload(self):
        received: list[tuple] = []
        lg = LiveLog(uid="sched3", schedule=lambda fn, *a: received.append((fn, a)))
        lg.set_text("x")
        fn, args = received[-1]
        assert callable(fn)
        assert len(args) == 1 and isinstance(args[0], str)
        assert lg._cls in args[0]  # the JS targets this instance's container

    def test_without_a_scheduler_it_still_does_not_raise(self):
        # Non-app callers (tests, headless) construct LiveLog with no marshaller.
        lg = LiveLog(uid="nosched")
        lg.append_stdout("x\n")
        lg.flush()
        assert lg.text == "x\n"


class TestResync:
    def test_resync_preserves_text(self):
        # resync repaints the DOM from Python state after a frontend re-render;
        # it must not disturb the authoritative copy.
        lg = _log()
        lg.append_stdout("kept\n")
        lg.flush()
        lg.resync()
        assert lg.text == "kept\n"

    def test_resync_on_empty_log_is_safe(self):
        lg = _log()
        lg.resync()
        assert lg.text == ""


class TestIsolation:
    def test_uid_makes_the_container_class_unique(self):
        # Two apps in one kernel must not write into each other's log.
        a, b = LiveLog(uid="a"), LiveLog(uid="b")
        assert a._cls != b._cls

    def test_no_stdout_leak_outside_a_kernel(self, capsys):
        # Without the kernel guard, clear_output writes escape codes to the real
        # stdout in headless contexts (pytest, the CLI).
        lg = _log()
        lg.append_stdout("x\n")
        lg.flush()
        lg.set_text("y")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
