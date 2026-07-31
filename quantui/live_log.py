"""Live calculation log that QuantUI owns end-to-end (M-LOGSCROLL route C).

Why this exists
---------------
The Calculate-tab log used to be a plain ``widgets.Output``. ipywidgets rebuilds
that widget's DOM subtree on every appended line and resets ``scrollTop`` to 0,
so scrolling up during a run was impossible — the view snapped away within a
frame. The old workaround re-pinned the box to the bottom every animation frame,
which "fixed" jumps-to-top by permanently forcing stuck-at-bottom.

Ruled out in live Voilà sessions (2026-07-30):

- **Native anchoring alone** — still jumped to the top. ``overflow-anchor``
  protects against *content insertion*; it cannot undo an explicit ``scrollTop``
  assignment.
- **An outer scroll container** around a non-scrolling Output — also jumped to
  the top: the Output's subtree teardown collapses the ancestor's
  ``scrollHeight``, so the browser clamps the ancestor's ``scrollTop`` too.

So the re-render has to go, not be out-raced. This module owns one ``<div>``
created once and appends **text nodes** to it. With a stable node,
``overflow-anchor: auto`` holds the user's position for free, and a single
"was I at the bottom?" check gives follow-the-tail.

How text reaches the browser — and why NOT ``display()``
--------------------------------------------------------
First implementation pushed each chunk as ``display(Javascript(...))`` into a
hidden Output. **That silently dropped every streaming line.** ``display()``
inside an Output routes by *parent message id*: the frontend captures iopub
messages whose parent matches the one ``Output.__enter__`` recorded. The run
header survived only because it is written while the Run click's comm message is
being processed. Output produced from the calc thread — or from an io_loop
callback — has no message being processed, so there is no parent to route by and
the payload never reaches the browser. Marshalling to the main thread did not
help, because the constraint is the *message context*, not the thread.

The old ``Output.append_stdout`` never had this problem because it mutates the
``outputs`` **traitlet**, which syncs over the widget comm and is completely
independent of message parentage.

So this module uses the two mechanisms already proven in this app:

1. **Traitlet sync carries the data** — setting a hidden widget's ``value``
   works from any thread, no message context required.
2. **JS installed once at render carries the behaviour** — a ``MutationObserver``
   set up the same way the vib camera hook is (reflections/01 Rule 7), which
   copies each chunk into the log container.

No ``display()`` on the streaming path at all. The observer reads chunks out of
mutation *records* rather than re-reading current DOM state, so nothing is lost
when several chunks land in the same frame.

Drop-in contract
----------------
:class:`LiveLog` mimics the slice of ``widgets.Output`` the app already used —
``append_stdout()``, ``clear_output()`` and assigning ``.outputs`` — so
``_LogCapture.write``, the atomic run-header write and Clear are unchanged.

Known limitation
----------------
Text lives in the DOM, not widget state, so a frontend re-render (kernel
reconnect) loses it. Python keeps the authoritative copy in :attr:`text`; call
:meth:`resync` to repaint. M-RECONNECT should call it when restoring a view.
"""

from __future__ import annotations

import html as _html
import logging
import threading
from typing import Any, Optional

import ipywidgets as widgets
from IPython.display import Javascript, display

from quantui import theme as _theme

_LOG = logging.getLogger(__name__)

# Coalescing window for appends. PySCF emits many lines per second; batching
# keeps the comm quiet while staying short enough to read as live.
_FLUSH_INTERVAL_S = 0.12

_LOG_CLASS = "quantui-live-log"
_MAIL_CLASS = "quantui-live-mail"

# The container is the scroll box AND the text-node parent — deliberately no
# inner <span>. _APP_CSS's system-font rule targets bare `span` with !important,
# so a wrapper span would be forced back to a proportional font and re-break the
# ASCII header exactly as .jp-OutputArea-output did (see GOTCHAS).
_CONTAINER_STYLE = (
    "height:300px;overflow-y:auto;overflow-anchor:auto;"
    f"border:1px solid {_theme.BORDER};border-radius:2px;padding:8px;"
    "font-family:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,"
    "'Liberation Mono','Courier New',monospace;"
    "font-size:12.5px;line-height:1.35;white-space:pre-wrap;"
    "word-break:break-word;"
)

_PLACEHOLDER = "No calculation run yet. PySCF output and any errors will appear here."


def _bridge_js(log_cls: str, mail_cls: str) -> str:
    """JS installed once at render: mailbox mutations → log container.

    Retry-until-present mirrors ``_vib_bridge_set_mode``: at install time the
    widgets may not be in the DOM yet. The watchdog re-attaches if ipywidgets
    replaces the mailbox node, which it may do on any re-render.
    """
    return """
(function(){
  var LOG = "__LOG_CLS__", MAIL = "__MAIL_CLS__";
  var lastSeq = 0, observed = null;

  function apply(node){
    var seq = parseInt(node.getAttribute('data-qseq') || '0', 10);
    if (!seq || seq <= lastSeq) { return; }          // replay / out-of-order
    if (seq > lastSeq + 1) {
      console.warn('[quantui-live-log] gap', lastSeq, '->', seq);
    }
    lastSeq = seq;
    var box = document.querySelector('.' + LOG);
    if (!box) { return; }
    var op = node.getAttribute('data-qop') || 'append';
    var txt = node.textContent || '';
    if (op === 'set') {
      box.textContent = txt;
      box.scrollTop = box.scrollHeight;
      return;
    }
    // Append a text node: the existing DOM is untouched, so there is no
    // re-render and therefore no scrollTop reset. That is the whole fix.
    var atBottom = (box.scrollHeight - box.scrollTop - box.clientHeight) < 8;
    box.appendChild(document.createTextNode(txt));
    if (atBottom) { box.scrollTop = box.scrollHeight; }
  }

  function scan(nodes){
    for (var i = 0; i < nodes.length; i++){
      var n = nodes[i];
      if (n.nodeType !== 1) { continue; }
      if (n.hasAttribute && n.hasAttribute('data-qseq')) { apply(n); }
      else if (n.querySelector) {
        var inner = n.querySelector('[data-qseq]');
        if (inner) { apply(inner); }
      }
    }
  }

  var obs = new MutationObserver(function(recs){
    // Read from the RECORDS, not from current DOM state: several chunks can
    // land in one frame and only the records preserve every one.
    for (var i = 0; i < recs.length; i++){ scan(recs[i].addedNodes); }
  });

  function attach(){
    var host = document.querySelector('.' + MAIL);
    if (!host) { return false; }
    if (host === observed) { return true; }
    try { obs.disconnect(); } catch (e) {}
    obs.observe(host, {childList: true, subtree: true});
    observed = host;
    // Catch anything delivered before the observer was live.
    var pending = host.querySelector('[data-qseq]');
    if (pending) { apply(pending); }
    console.debug('[quantui-live-log] bridge attached');
    return true;
  }

  var tries = 0;
  (function boot(){
    if (attach()) { return; }
    if (++tries < 60) { setTimeout(boot, 50); }
    else { console.warn('[quantui-live-log] mailbox never appeared'); }
  })();

  // ipywidgets may replace the mailbox node on a re-render, which silently
  // detaches the observer. Cheap watchdog re-attaches.
  setInterval(function(){
    if (!observed || !document.contains(observed)) { attach(); }
  }, 2000);
})();
""".replace("__LOG_CLS__", log_cls).replace("__MAIL_CLS__", mail_cls)


class LiveLog(widgets.VBox):
    """An append-only, scroll-stable log surface.

    Parameters
    ----------
    uid:
        Suffix making the container/mailbox classes unique per app instance, so
        two apps in one kernel cannot write into each other's log.
    """

    def __init__(self, uid: str = "main", **kwargs: Any) -> None:
        self._cls = f"{_LOG_CLASS}-{uid}"
        self._mail_cls = f"{_MAIL_CLASS}-{uid}"

        self._container = widgets.HTML(
            f'<div class="{self._cls}" style="{_CONTAINER_STYLE}">{_PLACEHOLDER}</div>'
        )
        # Transport. Setting .value is a traitlet sync over the widget comm —
        # thread-safe and independent of message parentage, which is exactly
        # what display() was not. Zero height so it never affects layout.
        self._mailbox = widgets.HTML(
            "", layout=widgets.Layout(height="0px", overflow="hidden", margin="0")
        )
        self._mailbox.add_class(self._mail_cls)
        # Carries the one-time observer install. A Javascript output stored in
        # an Output widget executes when that widget renders, which is how the
        # old scroll guard and the vib camera hook both work.
        self._bridge = widgets.Output(
            layout=widgets.Layout(height="0px", overflow="hidden", margin="0")
        )

        self._lock = threading.RLock()
        self._pending = ""
        self._text = ""
        self._seq = 0
        self._timer: Optional[threading.Timer] = None
        self._placeholder_showing = True
        self._posts = 0
        self._post_errors = 0

        super().__init__([self._container, self._mailbox, self._bridge], **kwargs)
        self._install_bridge()

    # ── public state ────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """Everything written so far — the authoritative copy, not the DOM's."""
        with self._lock:
            return self._text + self._pending

    def diagnostics(self) -> dict:
        """Snapshot of transport health.

        When the log is blank there is otherwise no way to tell whether Python
        never sent, sent and raised, or sent fine and the browser dropped it.
        Pair with the ``[quantui-live-log]`` console markers.
        """
        with self._lock:
            return {
                "posts": self._posts,
                "errors": self._post_errors,
                "seq": self._seq,
                "chars": len(self._text) + len(self._pending),
                "pending": len(self._pending),
                "log_class": self._cls,
                "mail_class": self._mail_cls,
            }

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

        Atomicity matters: the header write became a single assignment to fix
        the pre-step-1 blank-window bug, so this must not become
        clear-then-append.
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
            self._post("set", text if text else _PLACEHOLDER)

    def resync(self) -> None:
        """Repaint the container from :attr:`text` after a frontend re-render."""
        with self._lock:
            self._post("set", (self._text + self._pending) or _PLACEHOLDER)

    def flush(self) -> None:
        """Force any buffered text out immediately."""
        self._flush()

    # ── internals ───────────────────────────────────────────────────────────

    def _schedule_locked(self) -> None:
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

    def _flush(self) -> None:
        with self._lock:
            self._timer = None
            chunk, self._pending = self._pending, ""
            if not chunk:
                return
            replacing_placeholder = self._placeholder_showing
            self._placeholder_showing = False
            self._text += chunk
            # First real output replaces the placeholder rather than appending
            # after it.
            if replacing_placeholder:
                self._post("set", self._text)
            else:
                self._post("append", chunk)

    def _post(self, op: str, payload: str) -> None:
        """Hand one chunk to the browser via the mailbox traitlet.

        NOT named ``_send``: that is ``widgets.Widget._send``, the comm
        transport itself. Overriding it breaks ``add_class`` and every state
        sync — which is exactly what happened, and what
        ``test_add_class_still_available`` now catches.

        Sequence numbers let the observer discard replays and warn on gaps —
        cheap insurance, since a lost chunk would otherwise be invisible.
        """
        self._seq += 1
        try:
            self._mailbox.value = (
                f'<span data-qseq="{self._seq}" data-qop="{op}">'
                f"{_html.escape(payload)}</span>"
            )
            self._posts += 1
        except Exception as exc:  # noqa: BLE001 — never break a run over a log
            # Logged, not silently swallowed: a bare `except: pass` here is what
            # hid the original transport failure for two live test rounds.
            self._post_errors += 1
            _LOG.warning("live-log post failed (op=%s): %s", op, exc)

    @staticmethod
    def _in_kernel() -> bool:
        """True only inside a live IPython kernel.

        Off-frontend (pytest, the CLI) there is nothing to render the installer
        into, and displaying anyway writes to real stdout.
        """
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return ip is not None and getattr(ip, "kernel", None) is not None
        except Exception:  # noqa: BLE001 — absence of IPython is a valid answer
            return False

    def _install_bridge(self) -> None:
        if not self._in_kernel():
            return
        try:
            with self._bridge:
                display(Javascript(_bridge_js(self._cls, self._mail_cls)))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("live-log bridge install failed: %s", exc)
