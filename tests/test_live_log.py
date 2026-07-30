"""Tests for the QuantUI-owned live log (M-LOGSCROLL route C).

The scroll behaviour itself is browser-side and can only be verified in a live
Voilà session (LOGSCROLL.0 was done that way). What is testable here is the part
that must not regress silently: the drop-in contract with ``widgets.Output``, the
authoritative Python-side text, and the thread-safe buffering — because
``append_stdout`` is called from the background calc thread.
"""

from __future__ import annotations

import re
import threading
import xml.etree.ElementTree as ET

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


class TestTraitletTransport:
    """Regression guard for the 2026-07-30 "header prints, then nothing" bug.

    The first implementation pushed each chunk as ``display(Javascript(...))``
    into a hidden Output. That routes by *parent message id*, so it only worked
    while a message was being processed — the run header (written from the click
    handler) appeared and every streaming line from the calc thread was silently
    dropped. Marshalling to the main thread did not help: the constraint is the
    message context, not the thread.

    Text now travels over a traitlet, which syncs via the widget comm from any
    thread with no message context. These tests pin that down.
    """

    def test_append_updates_the_mailbox_traitlet(self):
        lg = _log()
        lg.append_stdout("streamed\n")
        lg.flush()
        assert "streamed" in lg._mailbox.value

    def test_append_from_a_worker_thread_reaches_the_mailbox(self):
        # The exact path that was broken: no message context on this thread.
        lg = _log()
        lg.set_text("HDR\n")
        before = lg._mailbox.value

        def worker() -> None:
            lg.append_stdout("from the calc thread\n")
            lg.flush()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert lg._mailbox.value != before
        assert "from the calc thread" in lg._mailbox.value

    def test_sequence_numbers_increase(self):
        # The observer discards replays and warns on gaps using these.
        lg = _log()
        seqs = []
        for i in range(3):
            lg.append_stdout(f"{i}\n")
            lg.flush()
            seqs.append(
                int(re.search(r'data-qseq="(\d+)"', lg._mailbox.value).group(1))
            )
        assert seqs == sorted(seqs) and len(set(seqs)) == 3

    def test_ops_are_labelled(self):
        lg = _log()
        lg.set_text("x")
        assert 'data-qop="set"' in lg._mailbox.value
        lg.append_stdout("y")
        lg.flush()
        assert 'data-qop="append"' in lg._mailbox.value

    def test_payload_is_html_escaped(self):
        # Log lines contain <, > and & (PySCF warnings, file paths, HTML dumps).
        # Unescaped, they would corrupt the mailbox markup.
        lg = _log()
        lg.append_stdout("a <b> & 'c'\n")
        lg.flush()
        raw = lg._mailbox.value
        assert "&lt;b&gt;" in raw
        decoded = ET.fromstring(raw).text
        assert decoded == "a <b> & 'c'\n"

    def test_first_output_replaces_the_placeholder(self):
        # Otherwise the first line would land after "No calculation run yet."
        lg = _log()
        lg.append_stdout("first\n")
        lg.flush()
        assert 'data-qop="set"' in lg._mailbox.value

    def test_no_display_on_the_streaming_path(self, capsys):
        # A stdout leak here means display() crept back in.
        lg = _log()
        lg.append_stdout("x\n")
        lg.flush()
        lg.set_text("y")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


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
