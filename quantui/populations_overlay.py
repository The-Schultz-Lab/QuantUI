"""3D charge colours, dipole arrow and bond-order labels for the Populations
panel viewer.

Mirrors the click-to-measure highlight bridge (:mod:`quantui.app_measurement`):
a one-shot ``Javascript`` push into a hidden ``Output`` restyles the *live*
py3Dmol viewer without a Python HTML re-render (which would discard the camera
orientation the user rotated into).

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

# Bond-order labels are drawn only for pairs at or above this index, so the
# weak non-bonded pairs listed in the table (≥ BOND_ORDER_MIN) do not clutter
# the 3D view with labels floating between unbonded atoms.
BOND_LABEL_MIN = 0.30

_POPULATIONS_JS = """
(function(){
  var UID="__UID__";
  function v(){ return window["viewer_"+UID]; }
  var arrowShapes=[];
  var bondLabels=[];
  var colored=false;
  // 3Dmol creates the viewer asynchronously, so an update can arrive before
  // it exists. Keep the latest payload and retry instead of dropping it
  // (dropping it left the first render uncoloured).
  var pending=null, retries=0;

  function clearArrow(){
    var vw=v(); if(!vw){ return; }
    for(var i=0;i<arrowShapes.length;i++){
      try{ vw.removeShape(arrowShapes[i]); }catch(e){}
    }
    arrowShapes=[];
  }

  function clearLabels(){
    var vw=v(); if(!vw){ return; }
    for(var i=0;i<bondLabels.length;i++){
      try{ vw.removeLabel(bondLabels[i]); }catch(e){}
    }
    bondLabels=[];
  }

  function resetAtomColors(){
    var vw=v(); if(!vw){ return; }
    if(!colored){ return; }
    try{
      vw.setStyle({}, {stick:{}, sphere:{scale:0.3}});
    }catch(e){}
    colored=false;
  }

  function update(payload){
    var vw=v();
    if(!vw){
      var first=(pending===null);
      pending=payload;
      if(first){ retries=0; setTimeout(retry, 100); }
      return false;
    }
    pending=null;
    clearArrow();
    clearLabels();
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
    var labels = payload.labels || null;
    if(labels && labels.length){
      for(var k=0;k<labels.length;k++){
        try{
          bondLabels.push(vw.addLabel(labels[k].text, {
            position: labels[k].position,
            fontSize: 11,
            fontColor: "#111827",
            backgroundColor: "#ffffff",
            backgroundOpacity: 0.75,
            borderThickness: 0,
            alignment: "center",
            inFront: true
          }));
        }catch(e){}
      }
    }
    vw.render();
    return true;
  }

  function retry(){
    if(pending===null){ return; }
    if(v()){ update(pending); return; }
    if(++retries < 100){ setTimeout(retry, 100); } else { pending=null; }
  }

  window["__quantuiPopulationsUpdate_"+UID] = update;
  window.__quantuiPopulationsUpdate = update;
  var initial = __INITIAL__;
  if(initial){ update(initial); }
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


def bond_order_labels(
    bond_orders: Optional[Sequence[dict]],
    coordinates: Optional[Sequence[Sequence[float]]],
    index: str,
    *,
    min_value: float = BOND_LABEL_MIN,
) -> List[dict]:
    """``[{text, position}]`` at bond midpoints for py3Dmol ``addLabel``.

    *index* is ``"mayer"`` or ``"wiberg"``; anything else → no labels.
    Pairs below *min_value* or with out-of-range atom indices are skipped.
    """
    if index not in ("mayer", "wiberg") or not bond_orders or not coordinates:
        return []
    n = len(coordinates)
    out: List[dict] = []
    for bo in bond_orders:
        try:
            i, j = int(bo["i"]), int(bo["j"])
            value = bo.get(index)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if value is None or not (0 <= i < n and 0 <= j < n):
            continue
        value = float(value)
        if value < min_value:
            continue
        a, b = coordinates[i], coordinates[j]
        out.append(
            {
                "text": f"{value:.2f}",
                "position": {
                    "x": (float(a[0]) + float(b[0])) / 2.0,
                    "y": (float(a[1]) + float(b[1])) / 2.0,
                    "z": (float(a[2]) + float(b[2])) / 2.0,
                },
            }
        )
    return out


def inject_populations_js(html: str, initial_payload: Optional[dict] = None) -> str:
    """Append the populations overlay bridge to py3Dmol Analysis HTML.

    *initial_payload* (from :func:`overlay_payload`) is applied as soon as the
    viewer exists, so the first render already matches the checkboxes rather
    than depending on a later bridge push arriving after the viewer is built.
    """
    m = re.search(r"3dmolviewer_(\w+)", html)
    if m is None:
        logger.warning(
            "could not find py3Dmol viewer id; populations overlay unavailable"
        )
        return html
    uid = m.group(1)
    js = _POPULATIONS_JS.replace("__UID__", uid).replace(
        "__INITIAL__", json.dumps(initial_payload)
    )
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
    arrow_convention: str = "physics",
    bond_orders: Optional[Sequence[dict]] = None,
    bond_label_index: str = "off",
) -> dict:
    """Assemble the JSON payload for ``__quantuiPopulationsUpdate``.

    *arrow_convention* ``"physics"`` points the arrow head at the positive
    end (the IUPAC dipole vector, − → +); ``"chemistry"`` reverses it to the
    crossed-arrow convention (+ → −). Only the drawing changes.
    *bond_label_index* ``"mayer"``/``"wiberg"`` labels bonds with that index.
    """
    payload: dict = {"colors": None, "arrow": None, "labels": None}
    labels = bond_order_labels(bond_orders, coordinates, bond_label_index)
    if labels:
        payload["labels"] = labels
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
        vec = [float(v) for v in dipole_vector[:3]]
        if arrow_convention == "chemistry":
            vec = [-v for v in vec]
        payload["arrow"] = dipole_arrow_endpoints(com, vec)
    return payload


def _mulliken_vividness(app: Any) -> float:
    slider = getattr(app, "_mulliken_vividness_slider", None)
    if slider is None:
        return 1.0
    return float(getattr(slider, "value", 1.0))


def _mulliken_viewer_unavailable_html() -> str:
    style = "padding:12px 16px;color:#6b7280;font-size:13px;font-style:italic"
    return (
        f'<div style="{style}">'
        "3D structure unavailable: this result has no saved coordinates. "
        "Re-run the calculation to enable the charge-coloured viewer "
        "(table and chart above are still valid).</div>"
    )


def _resolve_mulliken_molecule(app: Any, molecule: Any = None) -> Any:
    """Return the best molecule for the Mulliken viewer slot."""
    return (
        molecule
        or getattr(app, "_mulliken_pending_molecule", None)
        or getattr(app, "_mulliken_displayed_molecule", None)
        or getattr(app, "_analysis_displayed_molecule", None)
        or getattr(app, "_molecule", None)
    )


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
            show_info=False,
        )
        return inject_populations_js(html, overlay_payload(app, molecule))
    html = render_html_fn(
        molecule,
        backend="plotlymol",
        style=app._viz_style,
        lighting=app._viz_lighting,
        bgcolor=bgcolor,
        show_info=False,
    )
    return cast(str, html)


def show_mulliken_viewer(
    app: Any, molecule: Any = None, *, render_html_fn: Any
) -> None:
    """Render the molecule into the Mulliken panel's dedicated viewer."""
    out = getattr(app, "_mulliken_mol_output", None)
    if out is None:
        return
    mol = _resolve_mulliken_molecule(app, molecule)
    if mol is None:
        app._set_html_output(out, _mulliken_viewer_unavailable_html())
        app._mulliken_displayed_molecule = None
        return
    try:
        html = render_mulliken_viewer_html(app, mol, render_html_fn=render_html_fn)
        if not html:
            app._set_html_output(out, _mulliken_viewer_unavailable_html())
            return
        app._set_html_output(out, html)
        app._mulliken_displayed_molecule = mol
        push_populations_overlay(app)
    except Exception as exc:  # noqa: BLE001 — viewer must never block the panel
        logger.debug("mulliken viewer render failed: %s", exc)
        app._set_html_output(out, _mulliken_viewer_unavailable_html())


def active_charge_scheme(app: Any) -> str:
    """``"npa"`` when the Mulliken/NPA toggle says NPA *and* NPA charges
    exist for the shown result; otherwise ``"mulliken"``."""
    value = getattr(getattr(app, "_pop_scheme_tb", None), "value", "mulliken")
    if value == "npa" and getattr(app, "_last_npa_charges", None):
        return "npa"
    return "mulliken"


def displayed_charges(app: Any) -> List[float]:
    """Charges of the active scheme (chart, colours, summary sum)."""
    if active_charge_scheme(app) == "npa":
        return list(getattr(app, "_last_npa_charges", None) or [])
    return list(getattr(app, "_last_mulliken_charges", None) or [])


def overlay_payload(app: Any, mol: Any) -> dict:
    """Overlay payload for *mol* from the app's charges, dipole and checkboxes."""
    symbols = list(getattr(app, "_last_mulliken_symbols", None) or [])
    charges = displayed_charges(app)
    dip_vec = getattr(app, "_last_mulliken_dipole_vector", None)
    color_on = bool(getattr(getattr(app, "_mulliken_color_cb", None), "value", False))
    dip_on = bool(getattr(getattr(app, "_mulliken_dipole_cb", None), "value", False))
    coords = list(getattr(mol, "coordinates", None) or []) if mol is not None else []
    if mol is not None and not symbols:
        symbols = list(getattr(mol, "atoms", None) or [])
    return build_overlay_payload(
        charges=charges or None,
        color_enabled=color_on,
        dipole_vector=dip_vec,
        dipole_enabled=dip_on,
        symbols=symbols or None,
        coordinates=coords or None,
        vividness=_mulliken_vividness(app),
        arrow_convention=dipole_arrow_convention(app),
        bond_orders=getattr(app, "_last_bond_orders", None),
        bond_label_index=str(
            getattr(getattr(app, "_bond_label_dd", None), "value", "off") or "off"
        ),
    )


def dipole_arrow_convention(app: Any) -> str:
    """The Mulliken panel's arrow-direction choice ("physics"/"chemistry")."""
    value = getattr(getattr(app, "_mulliken_dipole_dir_dd", None), "value", None)
    return "chemistry" if value == "chemistry" else "physics"


def dipole_caption_html(convention: str) -> str:
    """One line saying which way the drawn arrow points."""
    if convention == "chemistry":
        text = (
            "Arrow head at the <b>negative</b> end (+ → −, chemistry crossed-arrow "
            "convention); the dipole vector μ itself points the other way."
        )
    else:
        text = (
            "Arrow head at the <b>positive</b> end (− → +): the dipole vector μ "
            "(IUPAC / physics convention). Many chemistry texts draw + → −."
        )
    return f'<span style="font-size:12px">{text}</span>'


def push_populations_overlay(app: Any) -> None:
    """Push Mulliken/dipole overlay state to the Mulliken panel viewer."""
    bridge = getattr(app, "_populations_js_bridge", None)
    if bridge is None:
        return
    payload = overlay_payload(app, getattr(app, "_mulliken_displayed_molecule", None))
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
