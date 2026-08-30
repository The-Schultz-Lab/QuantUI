"""Click-to-measure atom selection for the Analysis-tab viewer (M-MEASURE).

py3Dmol-only for v1 (MEAS.2-.6): the Analysis tab's top viewer
(``app._analysis_mol_output``) is dual-backend (py3Dmol / plotlymol), but
only py3Dmol renders through a live ``$3Dmol.GLViewer`` with a native
``setClickable`` — plotlymol is a static Plotly figure with no equivalent
without restructuring it into a live widget, which is out of scope here. See
roadmap 45 ("Real constraint found: backend split").

Two one-way bridges, same shape as the isosurface panel's:

- **JS -> kernel** (a click): the exact "standard trick" ORBX.1 uses for its
  Save-PNG button — the click handler writes the picked atom's index into a
  hidden ``widgets.Textarea`` and dispatches an ``input`` event, which
  ipywidgets' own view syncs to the kernel. See ``inject_click_js``.
- **kernel -> JS** (a highlight update): mirrors
  ``app_visualization.iso_bridge_update`` — a hidden ``widgets.Output``
  receives one-shot ``IPython.display.Javascript`` that calls a bridge
  function the click JS defined, so the live viewer can be restyled without a
  Python re-render (which would lose the camera orientation the user rotated
  into — GOTCHAS: "Camera state does NOT persist across atomic HTML swaps").
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Sequence

from . import theme as _theme
from .measurement import describe_picks

logger = logging.getLogger(__name__)

MAX_PICKS = 4

# CSS class the hidden inbox Textarea carries — read by inject_click_js's
# generated JS (``.<class> textarea``) and by app_builders.py, which actually
# builds the widget and applies this class to it.
MEASURE_INBOX_CLASS = "quantui-measure-inbox"

# ---------------------------------------------------------------------------
# JS -> kernel: click transport (MEAS.2)
# ---------------------------------------------------------------------------

# Appended after the normal py3Dmol-rendered HTML for the Analysis-tab viewer
# ONLY (a per-VizTask opt-in, matching the router's existing scoping — this
# must never wire into every py3Dmol render). References the live viewer by
# the uid py3Dmol bakes into its own generated HTML
# (``3dmolviewer_<uid>`` / ``viewer_<uid>``), same lookup
# orbital_visualization._build_iso_viewer uses.
_MEASURE_CLICK_JS = """
(function(){
  var UID="__UID__";
  function v(){ return window["viewer_"+UID]; }
  var shapes=[];

  function clearHighlights(){
    var vw=v(); if(!vw){ return; }
    for(var i=0;i<shapes.length;i++){
      try{ vw.removeShape(shapes[i]); }catch(e){}
    }
    shapes=[];
  }

  // Highlight picked atoms with a translucent sphere overlay (MEAS.4). Looks
  // the atom's live coordinates up on the model rather than taking them from
  // Python, so this needs nothing but the index the click already sent.
  //
  // Radius MUST clear the rendered ball: ball+stick uses sphere.scale 0.3
  // (Mulliken tint uses 0.35). A fixed 0.4 Å sphere sat *inside* C/O/Cl/…
  // (VDW×0.3 ≈ 0.51 for carbon) and was only visible on hydrogen.
  function highlightRadius(a){
    var SPHERE_SCALE=0.35;
    var MARGIN=1.45;
    var vdw=1.7;
    try{
      var radii=($3Dmol && $3Dmol.GLModel && $3Dmol.GLModel.vdwRadii) || {};
      var elem=(a.elem || "C").toString();
      if(radii[elem]!=null){ vdw=radii[elem]; }
      else if(elem.length>1){
        var key=elem.charAt(0).toUpperCase()+elem.slice(1).toLowerCase();
        if(radii[key]!=null){ vdw=radii[key]; }
      }
    }catch(e){}
    return Math.max(0.55, vdw*SPHERE_SCALE*MARGIN);
  }

  var MEAS_COLOR="__MEAS_LINE_COLOR__";

  function vSub(a,b){ return {x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}; }
  function vAdd(a,b){ return {x:a.x+b.x,y:a.y+b.y,z:a.z+b.z}; }
  function vScale(v,s){ return {x:v.x*s,y:v.y*s,z:v.z*s}; }
  function vDot(a,b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
  function vCross(a,b){
    return {x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x};
  }
  function vLen(v){ return Math.sqrt(vDot(v,v)); }
  function vUnit(v){
    var n=vLen(v);
    return n<1e-8?{x:0,y:0,z:0}:vScale(v,1/n);
  }
  function vMid(a,b){ return vScale(vAdd(a,b),0.5); }
  function posAt(m, idx){
    var found=m.selectedAtoms({index: idx});
    return found&&found.length?found[0]:null;
  }
  function addSeg(vw, a, b, dashed){
    shapes.push(vw.addLine({
      start:{x:a.x,y:a.y,z:a.z},
      end:{x:b.x,y:b.y,z:b.z},
      color:MEAS_COLOR,
      dashed:!!dashed
    }));
  }
  function addArc(vw, center, u, v, ang, radius){
    if(ang<1e-4||vLen(u)<1e-6||vLen(v)<1e-6){ return; }
    var steps=Math.max(10, Math.round(Math.abs(ang)*180/Math.PI/4));
    var prev=null;
    for(var s=0;s<=steps;s++){
      var t=ang*s/steps;
      var pt=vAdd(center, vAdd(vScale(u,radius*Math.cos(t)), vScale(v,radius*Math.sin(t))));
      if(prev){ addSeg(vw, prev, pt, false); }
      prev=pt;
    }
  }
  function drawAngleArc(vw, m, i, j, k){
    var pa=posAt(m,i), pb=posAt(m,j), pc=posAt(m,k);
    if(!pa||!pb||!pc){ return; }
    var u=vUnit(vSub(pa,pb)), w=vUnit(vSub(pc,pb));
    if(vLen(u)<1e-6||vLen(w)<1e-6){ return; }
    var ang=Math.acos(Math.max(-1, Math.min(1, vDot(u,w))));
    var n=vUnit(vCross(u,w));
    if(vLen(n)<1e-6){ return; }
    var v=vUnit(vCross(n,u));
    addArc(vw, pb, u, v, ang, 0.45);
  }
  function drawDihedralArc(vw, m, i, j, k, l){
    var pi=posAt(m,i), pj=posAt(m,j), pk=posAt(m,k), pl=posAt(m,l);
    if(!pi||!pj||!pk||!pl){ return; }
    var b1=vUnit(vCross(vSub(pj,pi), vSub(pk,pj)));
    var b2=vUnit(vCross(vSub(pk,pj), vSub(pl,pk)));
    var bc=vUnit(vSub(pk,pj));
    if(vLen(b1)<1e-6||vLen(b2)<1e-6||vLen(bc)<1e-6){ return; }
    var mid=vMid(pj,pk);
    var dh=Math.atan2(vDot(vCross(b1,b2), bc), vDot(b1,b2));
    var steps=Math.max(10, Math.round(Math.abs(dh)*180/Math.PI/4));
    var prev=null;
    var r=0.42;
    for(var s=0;s<=steps;s++){
      var t=dh*s/steps;
      var ct=Math.cos(t), st=Math.sin(t);
      var dir=vAdd(vScale(b1,ct), vScale(vCross(bc,b1),st));
      var pt=vAdd(mid, vScale(vUnit(dir), r));
      if(prev){ addSeg(vw, prev, pt, false); }
      prev=pt;
    }
  }
  function drawMeasurement(vw, m, indices){
    var n=indices.length;
    if(n<2){ return; }
    var pts=[];
    for(var k=0;k<n;k++){
      var p=posAt(m, indices[k]);
      if(!p){ return; }
      pts.push(p);
    }
    for(var k=0;k<n-1;k++){ addSeg(vw, pts[k], pts[k+1], false); }
    if(n===3){ drawAngleArc(vw, m, indices[0], indices[1], indices[2]); }
    if(n>=4){ drawDihedralArc(vw, m, indices[0], indices[1], indices[2], indices[3]); }
  }

  window["__quantuiMeasureHighlight_"+UID] = function(indices){
    var vw=v(); if(!vw){ return false; }
    clearHighlights();
    var m=vw.getModel();
    if(m){
      for(var k=0;k<indices.length;k++){
        var found=m.selectedAtoms({index: indices[k]});
        if(found && found.length){
          var a=found[0];
          shapes.push(vw.addSphere({
            center:{x:a.x,y:a.y,z:a.z}, radius:highlightRadius(a),
            color:"__HL_COLOR__", opacity:__HL_OPACITY__
          }));
        }
      }
      if(indices.length>=2){ drawMeasurement(vw, m, indices); }
    }
    vw.render();
    return true;
  };
  // Last-loaded viewer owns the unqualified name (same convention as
  // __quantuiIsoUpdate) so the kernel-side bridge never has to track uids.
  window.__quantuiMeasureHighlight = window["__quantuiMeasureHighlight_"+UID];

  function onClick(atom){
    var box=document.querySelector(".__INBOX_CLASS__ textarea");
    if(!box){ return; }
    box.value=String(atom.index);
    box.dispatchEvent(new Event("input", {bubbles:true}));
  }

  // Retry-until-present: at the moment this script runs, the async 3Dmol
  // bundle may not have finished building the viewer yet (same shape as
  // iso_bridge_update's retry loop).
  function attach(){
    var vw=v();
    if(!vw){ setTimeout(attach,50); return; }
    vw.setClickable({}, true, onClick);
  }
  attach();
})();
"""

# Saturated yellow — named CSS "yellow" (#ffff00) looked washed at low opacity
# against CPK whites/grays; this chrome-yellow reads as a clear selection cue.
_HIGHLIGHT_COLOR = "#FFEA00"
_HIGHLIGHT_OPACITY = "0.80"
_MEAS_LINE_COLOR = "#06b6d4"


def inject_click_js(html: str, *, inbox_class: str) -> str:
    """Append the click-to-measure wiring to already-rendered py3Dmol HTML.

    Finds the viewer's uid the same way ``_build_iso_viewer`` does (py3Dmol
    bakes ``3dmolviewer_<uid>`` into its own output), so this works on
    whatever ``visualization_py3dmol.render_molecule_html`` returns without
    that function needing to know about measurement at all. Returns *html*
    unchanged (logging a warning) if no viewer id is found — a click-to-
    measure feature that silently does nothing is worse than one that is
    visibly absent, but this must never turn a render failure into a crash.
    """
    m = re.search(r"3dmolviewer_(\w+)", html)
    if m is None:
        logger.warning("could not find py3Dmol viewer id; click-to-measure unavailable")
        return html
    uid = m.group(1)
    js = (
        _MEASURE_CLICK_JS.replace("__UID__", uid)
        .replace("__INBOX_CLASS__", inbox_class)
        .replace("__HL_COLOR__", _HIGHLIGHT_COLOR)
        .replace("__HL_OPACITY__", _HIGHLIGHT_OPACITY)
        .replace("__MEAS_LINE_COLOR__", _MEAS_LINE_COLOR)
    )
    return f"{html}<script>{js}</script>"


# ---------------------------------------------------------------------------
# kernel -> JS: push a highlight update to the live viewer (MEAS.4)
# ---------------------------------------------------------------------------


def push_highlight(app: Any, indices: Sequence[int]) -> None:
    """Restyle the LIVE viewer's picked-atom highlights — no Python re-render.

    Same shape as ``app_visualization.iso_bridge_update``: re-rendering would
    replace the viewer wholesale and lose the camera orientation the user
    rotated into.
    """
    bridge = getattr(app, "_measure_js_bridge", None)
    if bridge is None:
        return
    from IPython.display import Javascript, display

    payload = json.dumps(list(indices))
    js = (
        "(function(){var n=0;function go(){n++;"  # noqa: UP031 — JS is brace-dense
        "if(window.__quantuiMeasureHighlight){window.__quantuiMeasureHighlight(%s);}"
        "else if(n<40){setTimeout(go,50);}}go();})();" % payload
    )
    try:
        bridge.clear_output(wait=True)
        with bridge:
            display(Javascript(js))
    except Exception as exc:  # noqa: BLE001 — a highlight push must never raise
        logger.debug("measure highlight push failed: %s", exc)


# ---------------------------------------------------------------------------
# Readout text (MEAS.5)
# ---------------------------------------------------------------------------

_PLACEHOLDER_TEXT = "Click an atom in the viewer to start measuring."


def _readout_html(text: str) -> str:
    return (
        f'<div style="border:1px solid {_theme.css.BORDER};background:{_theme.css.BG_PANEL};'
        f"border-radius:4px;padding:6px 10px;margin:4px 0;font-size:13px;"
        f'color:{_theme.css.TEXT_BODY}">{text}</div>'
    )


def _set_readout(app: Any, text: str) -> None:
    readout = getattr(app, "_measure_readout", None)
    if readout is not None:
        readout.value = _readout_html(text)


# ---------------------------------------------------------------------------
# Python-side pick state machine (MEAS.3)
# ---------------------------------------------------------------------------


def on_measure_inbox_changed(app: Any, change: dict) -> None:
    """Accumulate a picked atom index from a viewer click.

    Fires when the click JS (``inject_click_js``) writes an atom index into
    the hidden inbox Textarea. 1-4 picks accumulate into a running bond /
    angle / dihedral readout; a 5th click starts a new chain. Clicking an
    already-picked atom again mid-chain is ignored rather than accepted,
    since a repeated index makes the geometry degenerate (ASE raises
    ``ZeroDivisionError`` on a zero-length vector) for no benefit to the
    student.
    """
    raw = (change or {}).get("new") or ""
    box = getattr(app, "_measure_inbox", None)

    def _clear_inbox() -> None:
        if box is not None and box.value:
            box.value = ""

    if not raw:
        return
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        _clear_inbox()
        return

    molecule = getattr(app, "_analysis_displayed_molecule", None)
    if molecule is None or not (0 <= idx < len(molecule.atoms)):
        _clear_inbox()
        return

    picks: List[int] = list(getattr(app, "_measure_picks", None) or [])
    if len(picks) >= MAX_PICKS:
        picks = [idx]
    elif idx in picks:
        _clear_inbox()
        return
    else:
        picks.append(idx)
    app._measure_picks = picks

    try:
        _set_readout(app, describe_picks(molecule, picks))
    except Exception as exc:  # noqa: BLE001 — a click must never crash the app
        logger.debug("measurement readout failed: %s", exc)
        _set_readout(app, "Could not compute a measurement for these atoms.")

    push_highlight(app, picks)
    _clear_inbox()


def on_measure_clear(app: Any, btn: Any = None) -> None:
    """Clear button (MEAS.5): empty the pick chain and its highlights."""
    _ = btn
    app._measure_picks = []
    _set_readout(app, _PLACEHOLDER_TEXT)
    push_highlight(app, [])


def reset_picks(app: Any) -> None:
    """Drop any stale picks — called whenever the Analysis viewer re-renders.

    A fresh render is a brand-new ``<canvas>`` with none of the old
    highlights, so there is nothing to push to the browser here; only the
    Python-side state and the readout text need clearing (MEAS.3: "Switching
    molecules or leaving the Analysis tab clears stale picks — no
    measurement is ever silently computed against the wrong structure.").
    """
    app._measure_picks = []
    _set_readout(app, _PLACEHOLDER_TEXT)


# ---------------------------------------------------------------------------
# plotlymol fallback messaging (MEAS.6)
# ---------------------------------------------------------------------------


def update_panel_for_backend(app: Any, backend: Any) -> None:
    """Swap the measurement panel between controls and an explanation.

    The panel itself stays visible either way — only its *content*
    switches, never a silent no-op when the resolved backend can't support
    clicking.
    """
    from .viz_backend_router import VizBackend

    controls = getattr(app, "_measure_controls", None)
    fallback = getattr(app, "_measure_fallback_msg", None)
    if controls is None or fallback is None:
        return
    is_py3dmol = backend == VizBackend.PY3DMOL
    controls.layout.display = "" if is_py3dmol else "none"
    fallback.layout.display = "none" if is_py3dmol else ""


# ---------------------------------------------------------------------------
# One entry point both analysis-viewer render paths call
# ---------------------------------------------------------------------------


def finalize_analysis_html(app: Any, html: str, backend: Any) -> str:
    """Wire click-to-measure into the top Analysis-tab viewer HTML.

    Called at both places that render into ``app._analysis_mol_output``
    (the post-calc render in ``app_visualization.show_result_3d`` and the
    backend-toggle re-render in ``app._rerender_3d_views``), so click-to-
    measure and MEAS.3's stale-pick reset apply identically regardless of
    which path produced the view.

    Mulliken charge colouring lives in the dedicated Mulliken-panel viewer
    (see ``populations_overlay.show_mulliken_viewer``) — not here.
    """
    from .viz_backend_router import VizBackend

    reset_picks(app)
    update_panel_for_backend(app, backend)
    if backend == VizBackend.PY3DMOL:
        html = inject_click_js(html, inbox_class=MEASURE_INBOX_CLASS)
    return html
