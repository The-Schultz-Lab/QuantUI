"""Live calculation log that QuantUI owns end-to-end (M-LOGSCROLL route C).

Why this exists
---------------
The Calculate-tab log used to be a plain ``widgets.Output``. ipywidgets rebuilds
that widget's DOM subtree on every appended line and resets ``scrollTop`` to 0,
so scrolling up during a run was impossible — the view snapped away within a
frame. The app worked around it with a ``requestAnimationFrame`` guard that
re-pinned the box to the bottom every frame, which fixed "jumps to the top" by
permanently forcing "stuck at the bottom".

Two cheaper fixes were tried and empirically ruled out in a live Voilà session
(2026-07-30, LOGSCROLL.0):

- **Native anchoring alone** (``overflow-anchor: auto``, no pinning) — the view
  still jumped to the top on every line. ``overflow-anchor`` protects against
  *content insertion*; it cannot undo an explicit ``scrollTop`` assignment.
- **An outer scroll container** wrapping a non-scrolling Output — also jumped to
  the top. The Output's subtree teardown collapses the ancestor's
  ``scrollHeight``, so the browser clamps the ancestor's ``scrollTop`` too.

So the fix has to remove the re-render, not out-manoeuvre it. This module owns a
single ``<div>`` created once, appends **text nodes** to it, and never re-renders
it. With a stable node, ``overflow-anchor: auto`` does the right thing for free:
the browser holds the user's position when content is appended below, and a
one-line "was I at the bottom?" check handles follow-the-tail.

Drop-in contract
----------------
:class:`LiveLog` deliberately mimics the small slice of the ``widgets.Output``
API the app already used — ``append_stdout()``, ``clear_output()`` and assigning
``.outputs`` — so existing call sites (``_LogCapture.write``, the atomic run
header write, the Clear button) work unchanged.

Threading
---------
``append_stdout`` is called from the background calc thread. Writes are buffered
under a lock and flushed at most every ``_FLUSH_INTERVAL_S``; a daemon timer
flushes the tail so the last lines of a run are never stranded. This mirrors the
existing elapsed-ticker pattern (see reflections/02).

Known limitation
----------------
The text lives in the DOM, not in widget state, so a **frontend re-render loses
it** (kernel reconnect, or a view rebuilt from scratch). The full text is kept in
``self.text``; call :meth:`resync` to repaint the container from it. M-RECONNECT
should call ``resync`` when it restores a live view.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import Javascript, display

_LOG = logging.getLogger(__name__)

# Coalescing window for appends. PySCF can emit many lines per second and each
# flush is a comm round-trip, so batching keeps the channel quiet; short enough
# that the log still reads as live.
_FLUSH_INTERVAL_S = 0.12

_BODY_CLASS = "quantui-live-log"

# The container is the scroll box AND the text node parent — deliberately no
# inner <span>. _APP_CSS's system-font rule targets bare `span` with !important,
# so a wrapper span would be forced back to a proportional font and re-break the
# ASCII header the same way .jp-OutputArea-output did (see GOTCHAS).
_CONTAINER_STYLE = (
    "height:300px;overflow-y:auto;overflow-anchor:auto;"
    "border:1px solid #c0ccd8;border-radius:2px;padding:8px;"
    "font-family:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,"
    "'Liberation Mono','Courier New',monospace;"
    "font-size:12.5px;line-height:1.35;white-space:pre-wrap;"
    "word-break:break-word;"
)

_PLACEHOLDER = "No calculation run yet. PySCF output and any errors will appear here."


class LiveLog(widgets.VBox):
    """An append-only, scroll-stable log surface.

    Parameters
    ----------
    uid:
        Suffix making the container class unique per app instance, so two apps
        in one kernel don't write into each other's log.
    """

    def __init__(
        self, uid: str = "main", schedule: Optional[Any] = None, **kwargs: Any
    ) -> None:
        # Marshaller that runs a callable on the kernel/main thread. REQUIRED for
        # streaming to appear: ``widgets.Output.__enter__`` captures output by
        # recording the *current parent message id*, and a background thread has
        # no parent-message context — so a JS payload emitted from the calc
        # thread never reaches the frontend. The run header worked without this
        # only because it is written from the click handler, i.e. already on the
        # main thread. Pass ``app._queue_main_thread_callback``.
        self._schedule = schedule
        self._cls = f"{_BODY_CLASS}-{uid}"
        self._container = widgets.HTML(
            f'<div class="{self._cls}" style="{_CONTAINER_STYLE}">{_PLACEHOLDER}</div>'
        )
        # Hidden channel for the JS append calls. Kept at zero height so it never
        # affects layout; cleared before each display so it cannot grow without
        # bound over a long run.
        self._sink = widgets.Output(
            layout=widgets.Layout(height="0px", overflow="hidden", margin="0")
        )
        self._lock = threading.RLock()
        self._pending = ""
        self._text = ""
        self._timer: Optional[threading.Timer] = None
        self._placeholder_showing = True
        # Emission counters — cheap, and they turn 'is the channel alive?'
        # from a guess into an observable fact (see diagnostics()).
        self._emit_count = 0
        self._emit_errors = 0
        super().__init__([self._container, self._sink], **kwargs)

    # ── public state ────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """Everything written so far — the authoritative copy, not the DOM's."""
        with self._lock:
            return self._text + self._pending

    # ── widgets.Output-compatible surface ───────────────────────────────────

    def append_stdout(self, text: str) -> None:
        """Append *text*. Safe to call from a background thread."""
        if not text:
            return
        with self._lock:
            self._pending += text
            self._schedule_locked()

    def clear_output(self, *_args: Any, **_kwargs: Any) -> None:
        """Reset to the placeholder. Signature tolerates Output's kwargs."""
        self.set_text("")

    @property
    def outputs(self) -> tuple:
        """Mimic ``widgets.Output.outputs`` as a single stream entry."""
        text = self.text
        if not text:
            return ()
        return ({"output_type": "stream", "name": "stdout", "text": text},)

    @outputs.setter
    def outputs(self, value: tuple) -> None:
        """Atomic replace — used by the on-click run-header write.

        Preserving atomicity matters: the header write was moved to a single
        assignment specifically to fix the pre-step-1 blank-window bug, so this
        must not become clear-then-append.
        """
        text = "".join(
            item.get("text", "") for item in (value or ()) if isinstance(item, dict)
        )
        self.set_text(text)

    # ── core operations ─────────────────────────────────────────────────────

    def set_text(self, text: str) -> None:
        """Replace the entire log body with *text* (empty → placeholder)."""
        with self._lock:
            self._cancel_timer_locked()
            self._pending = ""
            self._text = text
            self._placeholder_showing = not text
            body = text if text else _PLACEHOLDER
            # Scroll to the bottom after a replace so follow-the-tail is active;
            # the user can then scroll up and — the whole point of route C —
            # stay there.
            self._run_js(
                f"el.textContent = {json.dumps(body)};"
                "el.scrollTop = el.scrollHeight;"
            )

    def resync(self) -> None:
        """Repaint the container from :attr:`text`.

        For use after a frontend re-render (kernel reconnect) has emptied the
        DOM while Python still holds the log.
        """
        with self._lock:
            body = self._text + self._pending or _PLACEHOLDER
            self._run_js(
                f"el.textContent = {json.dumps(body)};"
                "el.scrollTop = el.scrollHeight;"
            )

    # ── internals ───────────────────────────────────────────────────────────

    def _schedule_locked(self) -> None:
        """Ensure a flush happens soon. Caller must hold the lock."""
        if self._timer is not None:
            return
        self._timer = threading.Timer(_FLUSH_INTERVAL_S, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:  # noqa: BLE001 — best-effort
                pass
            self._timer = None

    def flush(self) -> None:
        """Force any buffered text out immediately."""
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            self._timer = None
            chunk, self._pending = self._pending, ""
            if not chunk:
                return
            replacing_placeholder = self._placeholder_showing
            self._placeholder_showing = False
            self._text += chunk
            if replacing_placeholder:
                # First real output: drop the placeholder rather than appending
                # after it.
                self._run_js(
                    f"el.textContent = {json.dumps(self._text)};"
                    "el.scrollTop = el.scrollHeight;"
                )
                return
            # Append as a text node so the existing DOM is untouched: no
            # re-render means no scrollTop reset, which is the entire fix.
            self._run_js(
                "var atBottom = "
                "(el.scrollHeight - el.scrollTop - el.clientHeight) < 8;"
                f"el.appendChild(document.createTextNode({json.dumps(chunk)}));"
                "if (atBottom) { el.scrollTop = el.scrollHeight; }"
            )

    @staticmethod
    def _in_kernel() -> bool:
        """True only inside a live IPython kernel.

        Without this, ``with self._sink: clear_output()`` writes terminal escape
        codes to real stdout in headless contexts (pytest, the CLI), because
        there is no frontend to capture them. The Python-side text state is
        authoritative either way, so skipping the JS costs nothing off-frontend.
        """
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return ip is not None and getattr(ip, "kernel", None) is not None
        except Exception:  # noqa: BLE001 — absence of IPython is a valid answer
            return False

    def _run_js(self, body: str) -> None:
        """Run *body* with ``el`` bound to the container, if it is present."""
        # The console marker is deliberate and permanent: it is the only way to
        # tell "the JS never arrived" from "it arrived but the selector missed",
        # and distinguishing those took a live debugging round. debug level, so
        # it is silent unless someone opens the console with verbose enabled.
        js = (
            "(function(){var el=document.querySelector('."
            + self._cls
            + "');console.debug('[quantui-live-log] payload',"
            + str(len(body))
            + ",'target',!!el);if(!el){return;}"
            + body
            + "})();"
        )
        # Must emit on the main/kernel thread — see the note on ``schedule`` in
        # __init__. The marshaller runs inline when already on the main thread,
        # and callbacks queued on the io_loop preserve order, so appends stay
        # sequenced.
        if self._schedule is not None:
            self._schedule(self._emit_js, js)
        else:
            self._emit_js(js)

    def _emit_js(self, js: str) -> None:
        # Kernel check lives here, not in _run_js, so the marshalling path is
        # still exercised (and therefore testable) off-frontend.
        if not self._in_kernel():
            return
        try:
            # Exactly the shape proven to work for repeated post-render JS
            # pushes in this app (``app_visualization._vib_bridge_set_mode``):
            # the Output widget's own ``clear_output`` method, called OUTSIDE
            # the context, then a bare ``display`` inside it. The free
            # ``IPython.display.clear_output`` used inside the context instead
            # publishes a clear message through the display pipeline, which is
            # not the same thing. Don't "simplify" this back.
            self._sink.clear_output(wait=True)
            with self._sink:
                display(Javascript(js))
            self._emit_count += 1
        except Exception as exc:  # noqa: BLE001 — never break a run over a log
            # Logged, not swallowed silently: a silent except here is what hid
            # the background-thread failure earlier.
            self._emit_errors += 1
            _LOG.warning("live-log JS emit failed: %s", exc)

    def diagnostics(self) -> dict:
        """Snapshot of the JS channel's health.

        Exists because when the log is blank there is no way to tell, from the
        outside, whether Python never emitted, emitted and raised, or emitted
        fine and the browser dropped it. Pair with the ``[quantui-live-log]``
        console marker: emits climbing + no console marker means the payload is
        not reaching the frontend.
        """
        with self._lock:
            return {
                "emits": self._emit_count,
                "errors": self._emit_errors,
                "chars_buffered": len(self._text) + len(self._pending),
                "pending": len(self._pending),
                "has_scheduler": self._schedule is not None,
                "in_kernel": self._in_kernel(),
                "container_class": self._cls,
            }
