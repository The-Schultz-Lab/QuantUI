"""Tests for the M-STDERR / STDERR.1 fd-level stderr capture helper."""

from __future__ import annotations

import io
import os

import pytest

from quantui.c_stderr import capture_c_stderr

_POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="capture_c_stderr is POSIX-only (fd dup/dup2); no-op on Windows",
)


class TestWindowsNoOp:
    """On Windows the context manager is a no-op and must not touch fds."""

    def test_yields_without_raising_on_windows(self):
        if os.name == "posix":
            pytest.skip("Windows-specific behavior test")
        relay = io.StringIO()
        with capture_c_stderr(relay):
            pass
        # On Windows the relay must remain empty — capture_c_stderr did
        # nothing.
        assert relay.getvalue() == ""

    def test_relay_none_works_on_windows(self):
        if os.name == "posix":
            pytest.skip("Windows-specific behavior test")
        with capture_c_stderr(None):
            pass  # must not raise


@_POSIX_ONLY
class TestPosixCaptureBehavior:
    """The interesting fd-manipulation behavior — only runnable on POSIX."""

    def test_captures_fd_writes_into_relay_stream(self):
        relay = io.StringIO()
        with capture_c_stderr(relay):
            os.write(2, b"hello from c code\n")
        # After exit the captured bytes must be in the relay stream.
        assert "hello from c code" in relay.getvalue()

    def test_restores_original_stderr_fd_on_exit(self):
        # Sanity: after the wrapped block, writes to fd 2 must NOT go to
        # the temp file anymore. We check by writing one captured byte
        # inside, then writing a byte outside — the relay must contain
        # only the first.
        relay = io.StringIO()
        with capture_c_stderr(relay):
            os.write(2, b"inside\n")
        # If the fd weren't restored, this write would still hit the
        # (now-closed) tempfile and fail with OSError. Just confirm it
        # succeeds — we can't easily intercept it for content check.
        os.write(2, b"")  # zero-byte write must succeed on a valid fd
        # And relay still has only what was captured during the block.
        assert "inside" in relay.getvalue()
        assert "outside" not in relay.getvalue()

    def test_restores_fd_even_when_block_raises(self):
        # try/finally contract: descriptor must be restored on exception.
        with pytest.raises(RuntimeError):
            with capture_c_stderr(None):
                os.write(2, b"before raise\n")
                raise RuntimeError("simulated")
        # If the fd weren't restored, this would fail. Confirm fd 2 is
        # still valid by writing zero bytes.
        os.write(2, b"")

    def test_no_relay_stream_drops_captured_output(self):
        # capture_c_stderr(None) must accept writes silently.
        with capture_c_stderr(None):
            os.write(2, b"this disappears\n")
        # Nothing to assert about content — just that it didn't raise.

    def test_captured_bytes_decoded_replace_on_bad_bytes(self):
        # If PySCF C code writes non-UTF8 bytes (e.g. binary garbage on
        # crash), the relay must not raise — replace_errors must absorb.
        relay = io.StringIO()
        with capture_c_stderr(relay):
            os.write(2, b"\xff\xfe valid text after \n")
        # The relay must have something (replaced bytes + the valid text).
        relayed = relay.getvalue()
        assert "valid text after" in relayed

    def test_empty_capture_does_not_write_to_relay(self):
        # If nothing was written to fd 2 inside the block, relay must
        # stay untouched (don't emit a blank line).
        relay = io.StringIO()
        relay.write("previous content\n")
        with capture_c_stderr(relay):
            pass
        # No new content appended.
        assert relay.getvalue() == "previous content\n"

    def test_nested_contexts_restore_correctly(self):
        # Two levels deep: each must restore to the parent's state on
        # exit. Inner write must go to inner relay; outer write to outer.
        outer = io.StringIO()
        inner = io.StringIO()
        with capture_c_stderr(outer):
            os.write(2, b"outer-before\n")
            with capture_c_stderr(inner):
                os.write(2, b"inner-only\n")
            os.write(2, b"outer-after\n")
        assert "inner-only" in inner.getvalue()
        assert "inner-only" not in outer.getvalue()
        assert "outer-before" in outer.getvalue()
        assert "outer-after" in outer.getvalue()

    def test_relay_write_failure_is_swallowed(self):
        # If the relay stream itself raises on write, capture_c_stderr
        # must not propagate — telemetry must never block the caller.
        class _BadStream:
            def write(self, _s):
                raise RuntimeError("relay broken")

        with capture_c_stderr(_BadStream()):
            os.write(2, b"some content\n")
        # If we got here without raising, contract holds.
        os.write(2, b"")  # fd still valid
