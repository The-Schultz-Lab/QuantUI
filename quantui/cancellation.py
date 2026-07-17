"""Shared cancellation primitive for in-flight calculations (UXP.5).

The Cancel button sets a ``threading.Event`` on the app; the run's
``_LogCapture`` checks it on every output line and raises
:class:`CalcCancelled` at the next line — a cooperative stop that needs no
(unsafe) thread-kill. UXP.5 tightens that latency by also attaching the same
check to PySCF SCF callbacks and ASE optimizer observers, so cancellation
fires between SCF cycles / optimizer steps even when the calc is running
silently (verbose=0) and no output line is triggering the stream check.

The cancellation exception inherits :class:`BaseException`, **not**
``Exception`` — PySCF / ASE / ``session_calc`` wrap their kernels in
``except Exception``, so a plain-``Exception`` cancel would be swallowed and
re-reported as "Calculation failed". A ``BaseException`` (like
``KeyboardInterrupt``) sails through and reaches ``_do_run``'s
``except _CalcCancelled`` cleanly. See reflection 02 Rule 8.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

CancelCheck = Callable[[], bool]


class CalcCancelled(BaseException):
    """Raised to abort an in-flight calculation at a cooperative checkpoint."""


def raise_if_cancelled(cancel_check: Optional[CancelCheck]) -> None:
    """Raise :class:`CalcCancelled` if *cancel_check* is set and returns True."""
    if cancel_check is not None and cancel_check():
        raise CalcCancelled()


def cancel_check_from_stream(stream: Any) -> Optional[CancelCheck]:
    """Extract the cancel-check predicate carried on a progress stream.

    The run's ``_LogCapture`` exposes its cancel predicate as a public
    ``cancel_check`` attribute. Any calc module that receives the stream can
    duck-type it out with this helper — no import of the app layer needed.
    Returns ``None`` when the stream carries no (callable) predicate.
    """
    cc = getattr(stream, "cancel_check", None)
    return cc if callable(cc) else None


def attach_scf_cancel_callback(mf: Any, cancel_check: Optional[CancelCheck]) -> None:
    """Attach a cancel check to a PySCF SCF object's per-cycle ``callback``.

    PySCF invokes ``mf.callback(envs)`` once per SCF macro-iteration. Raising
    :class:`CalcCancelled` there stops the run between cycles regardless of
    whether output is being streamed. No-op when *cancel_check* is ``None`` or
    the object doesn't accept a callback (best-effort — cooperative
    output-line cancellation still applies).
    """
    if cancel_check is None or mf is None:
        return

    def _cb(_envs: Any, _cc: CancelCheck = cancel_check) -> None:
        if _cc():
            raise CalcCancelled()

    try:
        mf.callback = _cb
    except Exception:  # noqa: BLE001 — best-effort; not all backends allow it
        pass
