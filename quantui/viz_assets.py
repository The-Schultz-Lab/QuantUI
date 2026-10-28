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
from typing import Any, Optional, Tuple

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


def principal_view_rotation(
    coords: Any,
) -> Optional[Tuple[float, Tuple[float, float, float]]]:
    """Axis–angle that turns a molecule to face the camera (M-UIPOLISH P1).

    3Dmol looks down −z, so a molecule whose plane happens to contain z is
    seen edge-on: water looks linear and a dipole along z is invisible. This
    rotates the principal axes of the (unweighted) atom positions onto the
    screen axes — longest extent horizontal, second vertical, flattest
    direction toward the viewer. Returns ``(angle_deg, axis)`` for 3Dmol's
    ``rotate``, or ``None`` when no rotation is needed or possible.
    """
    import numpy as np

    try:
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    except Exception:  # noqa: BLE001 — orientation is cosmetic
        return None
    if len(xyz) < 2:
        return None
    centered = xyz - xyz.mean(axis=0)
    _, vecs = np.linalg.eigh(centered.T @ centered)
    axes = vecs[:, ::-1].T  # rows: largest, middle, smallest spread
    for k in range(3):  # deterministic signs
        if axes[k][np.argmax(np.abs(axes[k]))] < 0:
            axes[k] = -axes[k]
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    # ``axes`` maps molecule coordinates to screen coordinates (x, y, z).
    rot = axes
    cos_a = float(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    if angle < 1e-3:
        return None
    if abs(angle - 180.0) < 1e-3:
        # Axis from the symmetric part: R = 2 n nᵀ − I.
        n = np.sqrt(np.clip((np.diag(rot) + 1.0) / 2.0, 0.0, None))
        i = int(np.argmax(n))
        for j in range(3):
            if j != i and (rot[i, j] + rot[j, i]) < 0:
                n[j] = -n[j]
    else:
        n = np.array(
            [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
        )
        n = n / np.linalg.norm(n)
    return angle, (float(n[0]), float(n[1]), float(n[2]))


def orient_view(view: Any, coords: Any) -> None:
    """Apply :func:`principal_view_rotation` to a ``py3Dmol`` view (after zoomTo)."""
    rot = principal_view_rotation(coords)
    if rot is not None:
        angle, (x, y, z) = rot
        view.rotate(angle, {"x": x, "y": y, "z": z})


def orient_js(var: str, coords: Any) -> str:
    """Same as :func:`orient_view`, as a JS statement for hand-built viewers."""
    rot = principal_view_rotation(coords)
    if rot is None:
        return ""
    angle, (x, y, z) = rot
    return f"{var}.rotate({angle:.4f},{{x:{x:.6f},y:{y:.6f},z:{z:.6f}}});"


def coords_from_model_text(text: str, fmt: str) -> list:
    """Atom positions from XYZ or Gaussian-cube text (units as in the file).

    Orientation only needs directions, so Bohr (cube) vs Å (XYZ) is moot.
    """
    lines = (text or "").splitlines()
    coords = []
    try:
        if fmt == "cube":
            n_atoms = abs(int(lines[2].split()[0]))
            rows = lines[6 : 6 + n_atoms]
            coords = [[float(v) for v in r.split()[2:5]] for r in rows]
        else:
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    coords.append([float(v) for v in parts[1:4]])
    except (IndexError, ValueError):
        return []
    return coords


def plotly_camera(coords: Any, distance: float = 1.8) -> Optional[dict]:
    """Plotly ``scene.camera`` with the same framing as :func:`orient_view`."""
    import numpy as np

    rot = principal_view_rotation(coords)
    if rot is None:
        return None
    try:
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
        centered = xyz - xyz.mean(axis=0)
        _, vecs = np.linalg.eigh(centered.T @ centered)
    except Exception:  # noqa: BLE001
        return None
    up, toward = vecs[:, 1], vecs[:, 0]  # middle spread up, flattest at camera
    return {
        "eye": {k: float(v) * distance for k, v in zip("xyz", toward)},
        "up": {k: float(v) for k, v in zip("xyz", up)},
    }
