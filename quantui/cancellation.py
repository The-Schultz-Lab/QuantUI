"""Shared cancellation primitive for in-flight calculations.

The Cancel button sets a ``threading.Event`` on the app; the run's
``_LogCapture`` checks it on every output line and raises
:class:`CalcCancelled` at the next line — a cooperative stop that needs no
(unsafe) thread-kill. This module tightens that latency by also attaching the same
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


def attach_scf_cancel_callback(
    mf: Any,
    cancel_check: Optional[CancelCheck],
    *,
    progress_cb: Optional[Callable[[Any], None]] = None,
) -> None:
    """Attach cancel + progress hooks to a PySCF SCF object's ``callback``.

    PySCF invokes ``mf.callback(envs)`` once per SCF macro-iteration (``envs``
    is the kernel's locals dict — cycle index, ``e_tot``, etc.). We use that
    single hook for two things:

    - **Cancel**: raise :class:`CalcCancelled` between cycles so a
      Cancel click stops the run regardless of whether output is streamed.
    - **Progress**: call *progress_cb(envs)* each cycle so the
      caller can surface a live "SCF cycle N" status even when the SCF runs at
      ``verbose=0`` (the optimizer/PES per-step SCF, which streams nothing).

    No-op when both hooks are ``None`` or the object doesn't accept a callback
    (best-effort — cooperative output-line cancellation still applies).
    """
    if mf is None or (cancel_check is None and progress_cb is None):
        return

    def _cb(
        envs: Any,
        _cc: Optional[CancelCheck] = cancel_check,
        _pc: Optional[Callable[[Any], None]] = progress_cb,
    ) -> None:
        if _pc is not None:
            try:
                _pc(envs)
            except Exception:  # noqa: BLE001 — progress is best-effort
                pass
        if _cc is not None and _cc():
            raise CalcCancelled()

    try:
        mf.callback = _cb
    except Exception:  # noqa: BLE001 — best-effort; not all backends allow it
        pass
