"""Offline-safe py3Dmol asset loading.

py3Dmol's ``view()`` constructor defaults to
``js='https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js'`` and its
``_make_html()`` emits a ``loadScriptAsync('<that URL>')`` call. On any host
with a Content Security Policy, captive portal, or **no network at all**
(offline classroom deployment — QuantUI's primary target), that fetch fails
silently and the viewer renders as a blank rectangle with no error anywhere
in Python. This is the py3Dmol analogue of the Plotly CDN trap documented in
``reflections/01-voila-rendering-and-display.md`` Rule 1 (never load plotly.js
from a CDN) — and it is why molecule, trajectory, vibration, and
orbital-isosurface views all blanked offline (manual finding 2026-06-15).

This module makes py3Dmol load fully offline with **zero new dependencies**:

1. The 3Dmol.js bundle is vendored as package data
   (``data/js/3Dmol-min.js``, the exact 2.5.4 build py3Dmol targets).
2. :func:`offline_bootstrap_html` emits a one-time ``<script>`` that defines
   the page-global ``$3Dmolpromise`` by loading the vendored bytes from a
   ``data:`` URI. It is **byte-for-byte py3Dmol's own loader** with only the
   URL swapped — so whatever subtle scope juggling makes py3Dmol resolve the
   ``$3Dmol`` global online is preserved exactly. Inject it once per page,
   before any viewer renders (see ``QuantUIApp.display``).
3. :func:`make_view` builds every in-app viewer with ``js=_INAPP_SENTINEL``
   (a short, never-fetched string) instead of the CDN URL. Because the
   bootstrap has already defined ``$3Dmolpromise``, py3Dmol's
   ``if (typeof $3Dmolpromise === 'undefined')`` guard is false, the sentinel
   is never loaded, and the viewer reuses the bootstrap's promise. Net effect:
   the heavy bytes ship **once** per page, not once per render — important for
   rapid trajectory/vib frame swaps.
4. :func:`standalone_html` prepends the full inline bootstrap to a viewer's
   HTML so **exported** files (vib / trajectory animation downloads) are
   self-contained and play offline outside the app.

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

# Passed as ``js=`` to in-app views. It is never fetched: the page bootstrap
# defines $3Dmolpromise first, so py3Dmol's guard skips loading entirely. If
# the bootstrap is somehow absent the browser will try (and fail) to load this
# string and surface py3Dmol's built-in "3Dmol.js failed to load" warning <p>
# — a visible failure, never a silent CDN dependency.
_INAPP_SENTINEL = "quantui-3dmol-bundled-offline"

# The CDN URL we are replacing — kept here so a test can assert it never
# appears in any emitted HTML.
CDN_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"


@lru_cache(maxsize=1)
def _js_data_uri() -> str:
    """Return the vendored 3Dmol.js as a base64 ``data:`` URI (cached)."""
    raw = _JS_PATH.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:text/javascript;base64,{b64}"


# py3Dmol's loader, verbatim (py3Dmol/__init__.py ``view.__init__`` ``startjs``),
# with the ``%s`` URL slot filled by our data: URI instead of the CDN. Defining
# ``$3Dmolpromise`` synchronously here means every later view reuses it.
_BOOTSTRAP_TEMPLATE = """<script>
var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {}
    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
        exports = savedexports;
        module = savedmodule;
        resolve();
    };
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
});
};

if(typeof $3Dmolpromise === 'undefined') {
$3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync('%s');
}
</script>
"""


@lru_cache(maxsize=1)
def offline_bootstrap_html() -> str:
    """One-time ``<script>`` that loads vendored 3Dmol.js offline (no CDN).

    Display this exactly once per page, *before* any py3Dmol viewer renders
    (``QuantUIApp.display`` injects it ahead of the app body). It defines the
    page-global ``$3Dmolpromise`` synchronously, so every viewer built via
    :func:`make_view` skips its own load and reuses this promise.
    """
    return _BOOTSTRAP_TEMPLATE % _js_data_uri()


def make_view(**kwargs):
    """Build a ``py3Dmol.view`` that never reaches the CDN.

    Identical to ``py3Dmol.view(**kwargs)`` except ``js`` is forced to the
    in-app sentinel. Relies on :func:`offline_bootstrap_html` having run on the
    page. Use this in place of ``py3Dmol.view(...)`` everywhere in the app.
    """
    import py3Dmol

    kwargs.setdefault("js", _INAPP_SENTINEL)
    return py3Dmol.view(**kwargs)


def standalone_html(view_html: str) -> str:
    """Wrap viewer HTML so it plays offline as a standalone file.

    Exported animation files are opened outside the app, where the page
    bootstrap is absent. Prepending the full inline bootstrap makes the file
    self-contained (the vendored 3Dmol.js travels with it, ~0.5 MB). The
    ``$3Dmolpromise`` guard keeps it from double-loading if combined with other
    py3Dmol output.
    """
    return offline_bootstrap_html() + view_html
