"""Click-to-pick atom selection for PES Scan setup (Calculate-tab viewer).

Reuses the py3Dmol click transport and highlight overlay from
:mod:`quantui.app_measurement`, but fills the PES scan atom dropdowns instead
of the measure readout.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

from .app_measurement import inject_click_js, push_highlight
from .measurement import atom_label

logger = logging.getLogger(__name__)

PES_PICK_INBOX_CLASS = "quantui-pes-pick-inbox"


def _needed_picks(scan_type: str) -> int:
    return {"bond": 2, "angle": 3, "dihedral": 4}.get(scan_type.lower(), 2)


def _scan_pick_indices(app: Any) -> List[int]:
    """0-based indices currently selected for the scan coordinate."""
    st = app._scan_type_dd.value.lower()
    nums = [int(app._scan_atom1.value), int(app._scan_atom2.value)]
    if st in ("angle", "dihedral"):
        nums.append(int(app._scan_atom3.value))
    if st == "dihedral":
        nums.append(int(app._scan_atom4.value))
    return [n - 1 for n in nums]


def push_scan_highlight(app: Any, indices: Sequence[int]) -> None:
    """Highlight scan atoms on the live Calculate-tab viewer."""
    bridge = getattr(app, "_scan_pick_bridge", None)
    if bridge is None:
        return
    orig = getattr(app, "_measure_js_bridge", None)
    try:
        app._measure_js_bridge = bridge
        push_highlight(app, indices)
    finally:
        app._measure_js_bridge = orig


def finalize_pes_calc_html(app: Any, html: str, backend: Any) -> str:
    """Wire click-to-pick when PES Scan uses the py3Dmol Calculate viewer."""
    if str(backend) != "py3dmol":
        return html
    if getattr(app, "calc_type_dd", None) is None:
        return html
    if app.calc_type_dd.value != "PES Scan":
        return html
    html = inject_click_js(html, inbox_class=PES_PICK_INBOX_CLASS)
    push_scan_highlight(app, _scan_pick_indices(app))
    return html


def on_pes_pick_inbox_changed(app: Any, change: dict) -> None:
    """Fill scan atom dropdowns from sequential viewer clicks."""
    raw = (change or {}).get("new") or ""
    box = getattr(app, "_scan_pick_inbox", None)

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

    molecule = getattr(app, "_molecule", None)
    if molecule is None or not (0 <= idx < len(molecule.atoms)):
        _clear_inbox()
        return

    needed = _needed_picks(app._scan_type_dd.value)
    picks: List[int] = list(getattr(app, "_scan_pick_buffer", None) or [])

    if len(picks) >= needed:
        picks = [idx]
    elif idx in picks:
        _clear_inbox()
        return
    else:
        picks.append(idx)
    app._scan_pick_buffer = picks

    widgets = [app._scan_atom1, app._scan_atom2, app._scan_atom3, app._scan_atom4]
    for i, pick in enumerate(picks):
        widgets[i].value = pick + 1

    push_scan_highlight(app, picks)

    if len(picks) >= needed:
        app._scan_pick_buffer = []
        from quantui.app_runflow import _update_scan_coord_summary

        _update_scan_coord_summary(app)
        readout = getattr(app, "_scan_pick_readout", None)
        if readout is not None:
            labels = " → ".join(atom_label(molecule, p) for p in picks)
            readout.value = (
                f'<span style="font-size:12px;color:#059669">' f"Picked {labels}</span>"
            )
    else:
        readout = getattr(app, "_scan_pick_readout", None)
        if readout is not None:
            have = " → ".join(atom_label(molecule, p) for p in picks)
            readout.value = (
                f'<span style="font-size:12px;color:#64748b">'
                f"Picked {have} — click {needed - len(picks)} more…</span>"
            )

    _clear_inbox()


def clear_pes_pick(app: Any) -> None:
    """Reset click-to-pick state and highlights."""
    app._scan_pick_buffer = []
    readout = getattr(app, "_scan_pick_readout", None)
    if readout is not None:
        readout.value = (
            '<span style="font-size:12px;color:#64748b">'
            "Click atoms in the viewer above (py3Dmol) to fill the fields.</span>"
        )
    push_scan_highlight(app, _scan_pick_indices(app))
