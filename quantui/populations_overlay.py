"""3D Mulliken color + dipole-arrow overlays for the Mulliken panel viewer.

Mirrors the M-MEASURE highlight bridge: a one-shot ``Javascript`` push into a
hidden ``Output`` restyles the *live* py3Dmol viewer without a Python HTML
re-render (which would discard the camera orientation the user rotated into).

Injected by :func:`inject_populations_js` into the dedicated Mulliken-panel
viewer only (decoupled from the top Analysis-tab structure viewer).
Plotlymol has no equivalent live API — overlays are a silent no-op there.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Sequence, Tuple, cast

logger = logging.getLogger(__name__)

# Å per Debye for half-arrow length from the centre of mass. Water (~1.85 D)
# gets a ~1.3 Å half-length — readable next to a typical bond without dominating.
_DIPOLE_SCALE_A_PER_D = 0.70

_POPULATIONS_JS = """
(function(){
  var UID="__UID__";
  function v(){ return window["viewer_"+UID]; }
  var arrowShapes=[];
  var colored=false;

  function clearArrow(){
    var vw=v(); if(!vw){ return; }
    for(var i=0;i<arrowShapes.length;i++){
      try{ vw.removeShape(arrowShapes[i]); }catch(e){}
    }
    arrowShapes=[];
  }

  function resetAtomColors(){
    var vw=v(); if(!vw){ return; }
    if(!colored){ return; }
    try{
      vw.setStyle({}, {stick:{}, sphere:{scale:0.3}});
    }catch(e){}
    colored=false;
  }

  window["__quantuiPopulationsUpdate_"+UID] = function(payload){
    var vw=v(); if(!vw){ return false; }
    clearArrow();
    resetAtomColors();

    var colors = payload.colors || null;
    if(colors && colors.length){
      for(var i=0;i<colors.length;i++){
        if(!colors[i]){ continue; }
        try{
          vw.setStyle(
            {index: i},
            {stick:{color: colors[i]}, sphere:{color: colors[i], scale:0.35}}
          );
        }catch(e){}
      }
      colored=true;
    }

    var arrow = payload.arrow || null;
    if(arrow && arrow.start && arrow.end){
      try{
        arrowShapes.push(vw.addArrow({
          start: arrow.start,
          end: arrow.end,
          radius: 0.08,
          radiusRatio: 3,
          mid: 0.72,
          color: arrow.color || "#16a34a"
        }));
      }catch(e){}
    }
    vw.render();
    return true;
  };
  window.__quantuiPopulationsUpdate = window["__quantuiPopulationsUpdate_"+UID];
})();
"""


def charge_colors(
    charges: Sequence[float],
    *,
    vividness: float = 1.0,
) -> List[str]:
    """Map Mulliken charges to hex colours (blue = +, red = −).

    Normalised to the largest |q| in the molecule so water and a larger
    zwitterion both span the full palette. ``vividness`` (0–1) blends each
    colour toward neutral grey — useful when charges are subtle.
    """
    if not charges:
        return []
    vivid = max(0.0, min(1.0, float(vividness)))
    peak = max(abs(float(c)) for c in charges) or 1.0
    out: List[str] = []
    for c in charges:
        t = max(-1.0, min(1.0, float(c) / peak))
        # t>0 → blue (electron-deficient); t<0 → red (electron-rich).
        if t >= 0:
            color = _lerp_hex("#e5e7eb", "#2563eb", t)
        else:
            color = _lerp_hex("#e5e7eb", "#dc2626", -t)
        if vivid < 1.0:
            color = _lerp_hex("#e5e7eb", color, vivid)
        out.append(color)
    return out


def _lerp_hex(a: str, b: str, t: float) -> str:
    def _rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    ar, ag, ab = _rgb(a)
    br, bg, bb = _rgb(b)
    r = int(round(ar + (br - ar) * t))
    g = int(round(ag + (bg - ag) * t))
    b_ = int(round(ab + (bb - ab) * t))
    return f"#{r:02x}{g:02x}{b_:02x}"


def center_of_mass(
    symbols: Sequence[str], coords: Sequence[Sequence[float]]
) -> List[float]:
    """Approximate COM using atomic numbers as masses (sufficient for arrow origin)."""
    from .molecule import ATOMIC_NUMBERS

    total_m = 0.0
    cx = cy = cz = 0.0
    for sym, xyz in zip(symbols, coords):
        m = float(ATOMIC_NUMBERS.get(sym, 12) or 12)
        total_m += m
        cx += m * float(xyz[0])
        cy += m * float(xyz[1])
        cz += m * float(xyz[2])
    if total_m <= 0:
        return [0.0, 0.0, 0.0]
    return [cx / total_m, cy / total_m, cz / total_m]


def dipole_arrow_endpoints(
    com: Sequence[float],
    dipole_vector_debye: Sequence[float],
    *,
    scale_a_per_d: float = _DIPOLE_SCALE_A_PER_D,
) -> Optional[dict]:
    """Return ``{start, end, color}`` for py3Dmol ``addArrow``, or None if ~zero."""
    if len(dipole_vector_debye) < 3:
        return None
    vx, vy, vz = (float(dipole_vector_debye[i]) for i in range(3))
    mag = (vx * vx + vy * vy + vz * vz) ** 0.5
    if mag < 1e-6:
        return None
    half = scale_a_per_d * mag
    ux, uy, uz = vx / mag, vy / mag, vz / mag
    cx, cy, cz = (float(com[0]), float(com[1]), float(com[2]))
    return {
        "start": {
            "x": cx - half * ux,
            "y": cy - half * uy,
            "z": cz - half * uz,
        },
        "end": {
            "x": cx + half * ux,
            "y": cy + half * uy,
            "z": cz + half * uz,
        },
        "color": "#16a34a",
    }


def inject_populations_js(html: str) -> str:
    """Append the populations overlay bridge to py3Dmol Analysis HTML."""
    m = re.search(r"3dmolviewer_(\w+)", html)
    if m is None:
        logger.warning(
            "could not find py3Dmol viewer id; populations overlay unavailable"
        )
        return html
    uid = m.group(1)
    js = _POPULATIONS_JS.replace("__UID__", uid)
    return f"{html}<script>{js}</script>"


def build_overlay_payload(
    *,
    charges: Optional[Sequence[float]],
    color_enabled: bool,
    dipole_vector: Optional[Sequence[float]],
    dipole_enabled: bool,
    symbols: Optional[Sequence[str]],
    coordinates: Optional[Sequence[Sequence[float]]],
    vividness: float = 1.0,
) -> dict:
    """Assemble the JSON payload for ``__quantuiPopulationsUpdate``."""
    payload: dict = {"colors": None, "arrow": None}
    if color_enabled and charges:
        payload["colors"] = charge_colors(charges, vividness=vividness)
    if (
        dipole_enabled
        and dipole_vector is not None
        and len(dipole_vector) >= 3
        and symbols
        and coordinates
        and len(symbols) == len(coordinates)
    ):
        com = center_of_mass(symbols, coordinates)
        payload["arrow"] = dipole_arrow_endpoints(com, dipole_vector)
    return payload


def _mulliken_vividness(app: Any) -> float:
    slider = getattr(app, "_mulliken_vividness_slider", None)
    if slider is None:
        return 1.0
    return float(getattr(slider, "value", 1.0))


def render_mulliken_viewer_html(app: Any, molecule: Any, *, render_html_fn: Any) -> str:
    """Return self-contained HTML for the Mulliken panel's py3Dmol viewer."""
    if render_html_fn is None or molecule is None:
        return ""
    from quantui.visualization_py3dmol import PY3DMOL_AVAILABLE

    bgcolor = app._plotly_theme_colors()["scene_bgcolor"]
    if PY3DMOL_AVAILABLE:
        html = render_html_fn(
            molecule,
            backend="py3dmol",
            style=app._viz_style,
            lighting=app._viz_lighting,
            bgcolor=bgcolor,
        )
        return inject_populations_js(html)
    html = render_html_fn(
        molecule,
        backend="plotlymol",
        style=app._viz_style,
        lighting=app._viz_lighting,
        bgcolor=bgcolor,
    )
    return cast(str, html)


def show_mulliken_viewer(
    app: Any, molecule: Any = None, *, render_html_fn: Any
) -> None:
    """Render the molecule into the Mulliken panel's dedicated viewer."""
    out = getattr(app, "_mulliken_mol_output", None)
    if out is None:
        return
    mol = (
        molecule
        or getattr(app, "_mulliken_displayed_molecule", None)
        or getattr(app, "_analysis_displayed_molecule", None)
        or getattr(app, "_molecule", None)
    )
    if mol is None:
        out.clear_output()
        app._mulliken_displayed_molecule = None
        return
    try:
        html = render_mulliken_viewer_html(app, mol, render_html_fn=render_html_fn)
        if not html:
            out.clear_output()
            return
        app._set_html_output(out, html)
        app._mulliken_displayed_molecule = mol
        push_populations_overlay(app)
    except Exception as exc:  # noqa: BLE001 — viewer must never block the panel
        logger.debug("mulliken viewer render failed: %s", exc)
        out.clear_output()


def push_populations_overlay(app: Any) -> None:
    """Push Mulliken/dipole overlay state to the Mulliken panel viewer."""
    bridge = getattr(app, "_populations_js_bridge", None)
    if bridge is None:
        return
    mol = getattr(app, "_mulliken_displayed_molecule", None)
    symbols = list(getattr(app, "_last_mulliken_symbols", None) or [])
    charges = list(getattr(app, "_last_mulliken_charges", None) or [])
    dip_vec = getattr(app, "_last_mulliken_dipole_vector", None)
    color_on = bool(getattr(getattr(app, "_mulliken_color_cb", None), "value", False))
    dip_on = bool(getattr(getattr(app, "_mulliken_dipole_cb", None), "value", False))
    coords = list(getattr(mol, "coordinates", None) or []) if mol is not None else []
    if mol is not None and not symbols:
        symbols = list(getattr(mol, "atoms", None) or [])
    payload = build_overlay_payload(
        charges=charges or None,
        color_enabled=color_on,
        dipole_vector=dip_vec,
        dipole_enabled=dip_on,
        symbols=symbols or None,
        coordinates=coords or None,
        vividness=_mulliken_vividness(app),
    )
    try:
        from IPython.display import Javascript, display

        js = (
            "(function(){var fn=window.__quantuiPopulationsUpdate;"
            f"if(typeof fn==='function'){{fn({json.dumps(payload)});}}"
            "})();"
        )
        with bridge:
            bridge.clear_output(wait=True)
            display(Javascript(js))
    except Exception as exc:  # noqa: BLE001 — overlay must never crash the panel
        logger.debug("populations overlay push failed: %s", exc)
