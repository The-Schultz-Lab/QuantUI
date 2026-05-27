"""POSIX file-descriptor stderr capture (M-STDERR / STDERR.1).

PySCF and its C-extension dependencies (libcint, BLAS/LAPACK, dftd3) write
diagnostic messages directly to file-descriptor 2 (the OS-level stderr),
bypassing Python's ``sys.stderr`` and PySCF's own ``mol.stdout`` routing.
In a Voilà notebook those bytes surface as red error text above the cell
output even when the calculation succeeded — visually alarming, and
indistinguishable at a glance from a real failure.

This module provides ``capture_c_stderr(relay_stream=...)``, a context
manager that redirects fd 2 to a private temp file for the duration of
the block, then drains the captured bytes into the supplied relay stream
on exit. The end result: C-level diagnostics still reach the user (no
information loss), but through the normal live-log channel rather than
the red-text channel.

The implementation is POSIX-only (uses ``os.dup`` / ``os.dup2`` on fd 2).
On Windows the context is a no-op and yields immediately — safe to use
unconditionally since PySCF is Linux/macOS/WSL only and the rest of the
app's runtime gates on platform separately.

Thread-safety note: fd 2 is a process-global resource. QuantUI runs at
most one calculation at a time (the Run button is disabled during a run
and the work happens on a single background thread), so the standard
guidance is "use this only when no other code in the process is writing
to fd 2 concurrently". Nested contexts work correctly — each push/pop
saves and restores the previous fd 2 binding.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from typing import IO, Optional


@contextlib.contextmanager
def capture_c_stderr(relay_stream: Optional[IO[str]] = None):
    """Capture fd-level stderr to a temp file, relay on exit.

    Parameters
    ----------
    relay_stream:
        Optional writable text stream that receives the captured bytes
        (decoded UTF-8, replace on bad bytes) when the context exits. When
        ``None``, captured output is silently dropped — useful when the
        caller only wants the noise gone, not surfaced anywhere.

    Notes
    -----
    Output is buffered to a temp file during the block and flushed to
    ``relay_stream`` exactly once at exit. For long-running calculations
    that emit periodic warnings (e.g. an iterative SCF that prints one
    warning per cycle), the user sees the warnings as a single batch at
    the end rather than streamed in real time. This is a conscious trade-
    off: real-time streaming would require a pipe + drainer thread, which
    isn't worth the complexity for the typical "occasional libcint /
    BLAS warning" use case.

    The temp file is unlinked automatically by ``TemporaryFile``; no
    cleanup is required from the caller.

    On non-POSIX platforms the context manager yields immediately and
    relay_stream is never written to.
    """
    if os.name != "posix":
        yield
        return

    # Flush any Python-level stderr first so it doesn't get mixed in
    # with what we're about to capture.
    try:
        sys.stderr.flush()
    except Exception:
        pass

    # Binary temp file: C-level writes are bytes, not text.
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_fd: Optional[int] = None
    try:
        saved_fd = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            # Flush stderr (Python-level) before we tear the fd back so
            # any pending writes land in the temp file rather than getting
            # routed to the restored fd.
            try:
                sys.stderr.flush()
            except Exception:
                pass
            # Restore fd 2 before reading the temp file — otherwise any
            # write to stderr during the read (e.g. by relay_stream itself)
            # would loop back into the still-redirected fd.
            os.dup2(saved_fd, 2)
    finally:
        if saved_fd is not None:
            try:
                os.close(saved_fd)
            except OSError:
                pass

        # Drain captured bytes (best-effort) and relay.
        captured = b""
        try:
            tmp.flush()
            tmp.seek(0)
            captured = tmp.read()
        except Exception:
            pass
        finally:
            try:
                tmp.close()
            except Exception:
                pass

        if captured and relay_stream is not None:
            try:
                relay_stream.write(captured.decode("utf-8", errors="replace"))
            except Exception:
                pass
