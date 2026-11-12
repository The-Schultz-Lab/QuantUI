"""Load plotly.js once per page; in-app figures only reference it.

Why (audit VIZ-2, amends DEC-010)
---------------------------------
DEC-010 had the app embed figures in plotly's "require" mode. Plotly 6+ has
no such mode: the string is merely truthy, so every figure inlined the full
~4.8 MB plotly.js bundle -- every slider re-render shipped megabytes.

How
---
- :func:`bootstrap_html` is one ``<script>`` holding the plotly.js bundle that
  ships inside the installed ``plotly`` package (offline-first: never a CDN).
  It is guarded by ``window.__quantuiPlotly``, so re-rendering it (a page
  refresh re-renders widget outputs, a second app on the page, ...) does not
  re-evaluate the bundle. ``module``/``exports``/``define`` are shadowed while
  it runs, so the bundle cannot register itself with a CommonJS/AMD shim left
  on the page (Voila's RequireJS, py3Dmol's loader) instead of ``window``.
- The app displays one small :class:`ipywidgets.Output` from
  ``QuantUIApp.display`` and registers it here
  (:func:`register_bootstrap_output`). It stays empty until the first figure
  is rendered (:func:`figure_html` fills it), so nothing runs during page
  start-up -- the same reason 3Dmol.js is not bootstrapped at start-up
  (see ``viz_assets``).
- :func:`figure_html` renders a figure with ``include_plotlyjs=False`` and
  wraps its plotting script so it waits for ``window.__quantuiPlotly``: a
  figure whose output reaches the browser before the bootstrap still draws.
- With no app bootstrap registered (library use in a plain notebook, tests),
  :func:`figure_html` falls back to a self-contained inline figure, so the
  public plotting helpers keep working on their own.

Standalone exported HTML files do not use this module: they keep
``include_plotlyjs=True`` (inline) so the file works on its own.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: The page-global that holds QuantUI's copy of plotly.js.
PLOTLY_GLOBAL = "__quantuiPlotly"

# Give up waiting after ~60 s (1200 x 50 ms) and say so in the figure's div.
_WAIT_TRIES = 1200

_lock = threading.Lock()
_outputs: weakref.WeakSet[Any] = weakref.WeakSet()

_SCRIPT_RE = re.compile(r'<script(?: type="text/javascript")?>(.*?)</script>', re.S)


@lru_cache(maxsize=1)
def bootstrap_html() -> str:
    """The one-per-page ``<script>`` that defines ``window.__quantuiPlotly``."""
    from plotly.offline import get_plotlyjs

    bundle = get_plotlyjs()
    return (
        "<script>"
        f"if(!window.{PLOTLY_GLOBAL}){{"
        "window.PlotlyConfig=window.PlotlyConfig||{MathJaxConfig:'local'};"
        "var __qPrevPlotly=window.Plotly;"
        "(function(module,exports,define){\n"
        f"{bundle}\n"
        "})();"
        f"window.{PLOTLY_GLOBAL}=window.Plotly;"
        "if(__qPrevPlotly){window.Plotly=__qPrevPlotly;}"
        "}"
        "</script>"
    )


def register_bootstrap_output(output: Any) -> None:
    """Register an ``Output`` widget that is on the page to carry the bootstrap.

    Called by ``QuantUIApp.display``. The output stays empty until the first
    :func:`figure_html` call; after that it is never cleared.
    """
    with _lock:
        _outputs.add(output)


def has_bootstrap() -> bool:
    """True when a displayed app can provide plotly.js to the page."""
    return len(_outputs) > 0


def ensure_bootstrap() -> None:
    """Fill every registered, still-empty bootstrap output (idempotent)."""
    with _lock:
        pending = [o for o in list(_outputs) if not getattr(o, "outputs", ())]
        if not pending:
            return
        payload = (
            {
                "output_type": "display_data",
                "data": {"text/html": bootstrap_html()},
                "metadata": {},
            },
        )
        for out in pending:
            try:
                out.outputs = payload
            except Exception as exc:  # noqa: BLE001 — a figure must still render
                logger.warning("plotly bootstrap output not filled: %s", exc)


def _defer(body: str, div_note: str) -> str:
    return (
        "(function(){var n=0;function go(){"
        f"var P=window.{PLOTLY_GLOBAL};"
        f"if(!P){{if(++n<{_WAIT_TRIES}){{setTimeout(go,50);}}else{{{div_note}}}return;}}"
        f"(function(Plotly){{{body}}})(P);"
        "}go();})();"
    )


def defer_until_plotly(html: str) -> str:
    """Make the plotting scripts in a ``include_plotlyjs=False`` fragment wait
    for ``window.__quantuiPlotly`` (and call it as ``Plotly``)."""
    note = (
        "var d=document.querySelectorAll('.plotly-graph-div:empty');"
        "for(var i=0;i<d.length;i++){d[i].textContent="
        "'Plot library not loaded - reload the page.';}"
    )

    def _wrap(m: re.Match[str]) -> str:
        body = m.group(1)
        if "Plotly." not in body:
            return m.group(0)
        return "<script>" + _defer(body, note) + "</script>"

    return _SCRIPT_RE.sub(_wrap, html)


def figure_html(fig: Any, **kwargs: Any) -> str:
    """HTML fragment for an in-app figure, without the plotly.js bundle.

    Accepts :func:`plotly.io.to_html` keyword arguments (``full_html`` defaults
    to False; ``include_plotlyjs`` is chosen here and must not be passed).
    """
    import plotly.io as pio

    kwargs.setdefault("full_html", False)
    if not has_bootstrap():
        # No app on the page to provide plotly.js: stay self-contained.
        return str(pio.to_html(fig, include_plotlyjs=True, **kwargs))
    html = str(pio.to_html(fig, include_plotlyjs=False, **kwargs))
    ensure_bootstrap()
    return defer_until_plotly(html)
