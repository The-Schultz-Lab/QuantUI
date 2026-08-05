"""Offline-safe py3Dmol asset loading.

py3Dmol's ``view()`` constructor defaults to
``js='https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js'`` and its
``_make_html()`` emits a ``loadScriptAsync('<that URL>')`` call. On any host
with no network (offline classroom — QuantUI's primary target) or a restrictive
CSP, that fetch fails silently and the viewer renders blank. This is the
py3Dmol analogue of the Plotly CDN trap in
``reflections/01-voila-rendering-and-display.md`` Rule 1.

Approach (no new dependency)
----------------------------
The 3Dmol.js bundle is vendored as package data (``data/js/3Dmol-min.js``, the
exact 2.5.4 build py3Dmol targets). :func:`make_view` builds every viewer with
``js=<data: URI of the vendored bytes>`` instead of the CDN URL. This reuses
**py3Dmol's own per-view loader verbatim** — only the source URL changes from a
remote CDN to a local ``data:`` URI — so the viewer loads 3Dmol.js offline with
no network.

Why per-view and NOT a one-time page bootstrap: an earlier version injected the
loader once at app startup (the first display output). Running py3Dmol's
``exports``/``module``-juggling loader during Voilà's own RequireJS/AMD
bootstrap polluted the global module system at the worst moment and broke widget
startup offline. The per-view approach runs the identical loader **after** the
page is up (when a viewer renders) — exactly when py3Dmol normally runs it — so
it never interferes with startup. py3Dmol's ``$3Dmolpromise`` global guard means
only the first viewer on a page actually loads 3Dmol; later views reuse it.

Trade-off: the (cached, ~0.7 MB base64) data: URI rides in each viewer's HTML
payload. Fine for the molecule preview / isosurface / result viewer; heavier for
rapid trajectory/vib frame swaps (the bytes ship per payload even though only
the first triggers a load). Correctness and offline support take priority over
that payload size; optimizing the rapid-swap path is a possible follow-up.

Author: Jonathan Schultz, NCCU
Created: 2026-06-15
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Vendored 3Dmol.js — the exact build py3Dmol 2.x's constructor default points
# at, so the API the py3Dmol-generated JS calls (createViewer, addModel,
# addVolumetricData, addModelsAsFrames, animate, ...) is guaranteed present.
# Provenance + license: data/js/PROVENANCE.md, data/js/3Dmol-min.js.LICENSE.txt.
THREEDMOL_VERSION = "2.5.4"
_JS_PATH = Path(__file__).parent / "data" / "js" / "3Dmol-min.js"

# The CDN URL we replace — kept so a test can assert it never appears in any
# emitted HTML.
CDN_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"


@lru_cache(maxsize=1)
def _js_data_uri() -> str:
    """Return the vendored 3Dmol.js as a base64 ``data:`` URI (cached).

    Empty string if the bundle is missing, in which case :func:`make_view`
    falls back to py3Dmol's default (CDN) ``js`` rather than handing the viewer
    an unusable source.
    """
    try:
        raw = _JS_PATH.read_bytes()
    except OSError as exc:
        logger.warning("Vendored 3Dmol.js unreadable (%s); falling back to CDN", exc)
        return ""
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:text/javascript;base64,{b64}"


def make_view(**kwargs):
    """Build a ``py3Dmol.view`` that loads 3Dmol.js offline (no CDN).

    Identical to ``py3Dmol.view(**kwargs)`` except ``js`` defaults to a ``data:``
    URI of the vendored 3Dmol.js, so the viewer never reaches the network. Use
    this in place of ``py3Dmol.view(...)`` everywhere in the app. If the bundle
    is missing we leave py3Dmol's default ``js`` (CDN) in place.
    """
    import py3Dmol

    data_uri = _js_data_uri()
    if data_uri and "js" not in kwargs:
        kwargs["js"] = data_uri
    return py3Dmol.view(**kwargs)


def standalone_html(view_html: str, *, title: str = "QuantUI animation") -> str:
    """Wrap viewer HTML as a complete, standalone document.

    Views built via :func:`make_view` already embed the vendored 3Dmol.js
    loader (``js=<data: URI>``), so the *content* is self-contained and plays
    offline. This adds the document around it.

    It used to be a no-op pass-through, which wrote a bare fragment to disk.
    Browsers render that in quirks mode, so it mostly worked — but without a
    ``<meta charset>`` a file opened from disk is not reliably decoded as
    UTF-8, and these exports carry non-ASCII: the stepper's ⇄ compare button
    and the → in frame labels. Mojibake in a file someone puts in a talk is a
    poor way to find that out.
    """
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{title}</title>\n"
        "<style>body{margin:0;padding:16px;background:#fff;"
        "font-family:system-ui,-apple-system,'Segoe UI',sans-serif}</style>\n"
        "</head>\n<body>\n" + view_html + "\n</body>\n</html>\n"
    )
