"""Edit Structure panel on the Calculate tab (M-INTERACT INT.8).

Pick atoms by clicking the Calculate-tab viewer (py3Dmol; same click bridge
as click-to-measure and PES picking) or by typing their numbers, then set a
bond length / angle / dihedral, delete atoms, add a hydrogen, change an
element, or mark atoms frozen for a Geometry Opt. Every edit replaces the
working molecule (charge and multiplicity unchanged) and can be undone.
"""

from __future__ import annotations

import html as _html
import logging
from typing import Any, Callable, List, Sequence, Tuple

import ipywidgets as widgets

from . import theme as _theme
from .app_measurement import inject_click_js, push_highlight

logger = logging.getLogger(__name__)

EDIT_PICK_INBOX_CLASS = "quantui-edit-pick-inbox"
MAX_EDIT_PICKS = 4
_UNDO_DEPTH = 20


def build_edit_widgets(app: Any, *, layout_fn: Any) -> None:
    """Create the Edit Structure accordion (``app._edit_accordion``)."""
    app._edit_atoms_txt = widgets.Text(
        value="",
        placeholder="click atoms in the viewer, or type numbers: 1 2 3",
        description="Atoms:",
        continuous_update=False,
        style={"description_width": "55px"},
        layout=layout_fn(width="380px"),
    )
    app._edit_current_html = widgets.HTML(value="")
    app._edit_value = widgets.FloatText(
        value=0.0,
        description="New value:",
        style={"description_width": "75px"},
        layout=layout_fn(width="200px"),
    )

    def _btn(desc: str, tip: str, icon: str = "", width: str = "150px") -> Any:
        return widgets.Button(
            description=desc, tooltip=tip, icon=icon, layout=layout_fn(width=width)
        )

    app._edit_bond_btn = _btn("Set bond (Å)", "Two atoms: set their distance")
    app._edit_angle_btn = _btn(
        "Set angle (°)", "Three atoms: set the angle at the middle one"
    )
    app._edit_dihedral_btn = _btn(
        "Set dihedral (°)", "Four atoms: set the torsion about the middle bond"
    )
    app._edit_delete_btn = _btn("Delete atoms", "Delete every picked atom", "trash")
    app._edit_addh_btn = _btn("Add H", "Add one hydrogen to each picked atom", "plus")
    app._edit_element_txt = widgets.Text(
        value="", placeholder="N", layout=layout_fn(width="60px")
    )
    app._edit_element_btn = _btn(
        "Change element", "Change the picked atom(s) to this element", width="150px"
    )
    app._edit_freeze_btn = _btn(
        "Freeze for opt",
        "Add the picked atoms to Geometry Opt's 'Freeze atoms' list",
        "lock",
    )
    app._edit_undo_btn = _btn("Undo", "Undo the last structure edit", "undo", "100px")
    app._edit_clear_btn = _btn(
        "Clear picks", "Forget the picked atoms", "times", "130px"
    )
    app._edit_msg = widgets.HTML(value="")
    app._edit_pick_inbox = widgets.Textarea(
        value="", layout=layout_fn(width="1px", height="1px", visibility="hidden")
    )
    app._edit_pick_inbox.add_class(EDIT_PICK_INBOX_CLASS)
    app._edit_pick_bridge = widgets.Output(
        layout=layout_fn(width="1px", height="1px", visibility="hidden")
    )
    app._edit_undo_stack = []
    app._edit_picks = []
    hint = (
        f'<p style="color:{_theme.css.TEXT_SECONDARY};font-size:12px;margin:0 0 6px">'
        "Click atoms in the viewer above (py3Dmol; atom numbers are shown while "
        "this panel is open) or type their numbers. The last picked atom's side "
        "of the molecule moves. Edits keep charge and multiplicity; optimize "
        "afterwards.</p>"
    )
    body = widgets.VBox(
        [
            widgets.HTML(hint),
            widgets.HBox(
                [app._edit_atoms_txt, app._edit_clear_btn], layout=layout_fn(gap="8px")
            ),
            app._edit_current_html,
            widgets.HBox(
                [
                    app._edit_value,
                    app._edit_bond_btn,
                    app._edit_angle_btn,
                    app._edit_dihedral_btn,
                ],
                layout=layout_fn(gap="6px", flex_wrap="wrap"),
            ),
            widgets.HBox(
                [
                    app._edit_delete_btn,
                    app._edit_addh_btn,
                    app._edit_element_txt,
                    app._edit_element_btn,
                    app._edit_freeze_btn,
                    app._edit_undo_btn,
                ],
                layout=layout_fn(gap="6px", flex_wrap="wrap", align_items="center"),
            ),
            app._edit_msg,
            app._edit_pick_inbox,
            app._edit_pick_bridge,
        ],
        layout=layout_fn(padding="6px"),
    )
    app._edit_accordion = widgets.Accordion(
        children=[body], layout=layout_fn(margin="4px 0")
    )
    app._edit_accordion.set_title(0, "Edit Structure")
    app._edit_accordion.selected_index = None


def wire_edit_widgets(app: Any) -> None:
    """Connect the panel's buttons and observers (called once by the app)."""
    safe = getattr(app, "_safe_cb", lambda f: f)
    app._edit_atoms_txt.observe(
        safe(lambda c: on_edit_atoms_changed(app, c)), names="value"
    )
    app._edit_pick_inbox.observe(safe(lambda c: on_edit_pick(app, c)), names="value")
    app._edit_accordion.observe(
        safe(lambda c: app._refresh_calc_mol_viewer()), names="selected_index"
    )
    for btn, fn in (
        (app._edit_bond_btn, on_set_bond),
        (app._edit_angle_btn, on_set_angle),
        (app._edit_dihedral_btn, on_set_dihedral),
        (app._edit_delete_btn, on_delete),
        (app._edit_addh_btn, on_add_h),
        (app._edit_element_btn, on_change_element),
        (app._edit_freeze_btn, on_freeze),
        (app._edit_undo_btn, on_undo),
        (app._edit_clear_btn, on_clear_picks),
    ):
        btn.on_click(safe(lambda _b, _fn=fn: _fn(app)))


def editing_active(app: Any) -> bool:
    acc = getattr(app, "_edit_accordion", None)
    return acc is not None and acc.selected_index == 0


def finalize_edit_calc_html(app: Any, html: str, backend: Any) -> str:
    """Wire click-to-pick on the Calculate viewer while the panel is open."""
    if str(backend) != "py3dmol" or not editing_active(app):
        return html
    html = inject_click_js(html, inbox_class=EDIT_PICK_INBOX_CLASS)
    _push_edit_highlight(app, current_picks(app))
    return html


def _push_edit_highlight(app: Any, indices: Sequence[int]) -> None:
    bridge = getattr(app, "_edit_pick_bridge", None)
    if bridge is None:
        return
    orig = getattr(app, "_measure_js_bridge", None)
    try:
        app._measure_js_bridge = bridge
        push_highlight(app, indices)
    finally:
        app._measure_js_bridge = orig


def _msg(app: Any, text: str, ok: bool = True) -> None:
    color = _theme.css.ACCENT_SUCCESS if ok else _theme.css.ACCENT_ERROR
    app._edit_msg.value = f'<span style="color:{color};font-size:12px">{text}</span>'


def current_picks(app: Any) -> List[int]:
    """0-based picked atoms, in pick order, from the Atoms field."""
    text = str(app._edit_atoms_txt.value or "")
    out: List[int] = []
    for token in text.replace(",", " ").split():
        try:
            n = int(token) - 1
        except ValueError:
            continue
        if n >= 0 and n not in out:
            out.append(n)
    return out


def _set_picks(app: Any, picks: Sequence[int]) -> None:
    app._edit_atoms_txt.value = " ".join(str(p + 1) for p in picks)


def on_edit_pick(app: Any, change: dict) -> None:
    """A viewer click: append the atom (a fifth pick starts over)."""
    raw = (change or {}).get("new") or ""
    box = app._edit_pick_inbox
    if not raw:
        return
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        box.value = ""
        return
    mol = getattr(app, "_molecule", None)
    if mol is not None and 0 <= idx < len(mol.atoms):
        picks = current_picks(app)
        if len(picks) >= MAX_EDIT_PICKS:
            picks = []
        if idx in picks:
            picks.remove(idx)
        else:
            picks.append(idx)
        _set_picks(app, picks)
    box.value = ""


def on_edit_atoms_changed(app: Any, change: Any = None) -> None:
    """Show the current geometry of the picked atoms; prefill the value."""
    mol = getattr(app, "_molecule", None)
    picks = current_picks(app)
    _push_edit_highlight(app, picks)
    if mol is None or not picks:
        app._edit_current_html.value = ""
        return
    bad = [p + 1 for p in picks if p >= len(mol.atoms)]
    if bad:
        app._edit_current_html.value = (
            f'<span style="color:{_theme.css.ACCENT_ERROR}">No atom '
            f"{', '.join(map(str, bad))} (this structure has {len(mol.atoms)}).</span>"
        )
        return
    from .measurement import angle, atom_label, dihedral, distance

    labels = " – ".join(atom_label(mol, p) for p in picks)
    current = None
    try:
        if len(picks) == 2:
            current = distance(mol, *picks)
            what = f"distance {current:.3f} Å"
        elif len(picks) == 3:
            current = angle(mol, *picks)
            what = f"angle {current:.1f}°"
        elif len(picks) == 4:
            current = dihedral(mol, *picks)
            what = f"dihedral {current:.1f}°"
        else:
            what = "picked"
    except Exception:  # noqa: BLE001 — collinear atoms etc.
        what = "geometry undefined (collinear atoms)"
    if current is not None:
        app._edit_value.value = round(float(current), 3)
    app._edit_current_html.value = (
        f'<span style="font-size:12px">{_html.escape(labels)}: {what}</span>'
    )


def _apply(app: Any, op: Callable[..., Tuple[Any, str]], *args: Any) -> None:
    mol = getattr(app, "_molecule", None)
    if mol is None:
        _msg(app, "Load a molecule first.", ok=False)
        return
    try:
        new_mol, note = op(mol, *args)
    except (ValueError, ZeroDivisionError) as exc:
        _msg(app, f"⚠ {_html.escape(str(exc))}", ok=False)
        return
    stack = app._edit_undo_stack
    stack.append(mol)
    del stack[:-_UNDO_DEPTH]
    app._set_molecule(new_mol, "Edited structure", sync_charge_mult=False)
    _msg(app, _html.escape(note))
    on_edit_atoms_changed(app)


def _need(app: Any, n: int, what: str) -> List[int] | None:
    picks = current_picks(app)
    if len(picks) != n:
        _msg(app, f"{what} needs exactly {n} atoms (you have {len(picks)}).", ok=False)
        return None
    return picks


def on_set_bond(app: Any) -> None:
    from .structure_edit import set_bond_length

    picks = _need(app, 2, "A bond length")
    if picks:
        _apply(app, set_bond_length, *picks, float(app._edit_value.value))


def on_set_angle(app: Any) -> None:
    from .structure_edit import set_angle

    picks = _need(app, 3, "An angle")
    if picks:
        _apply(app, set_angle, *picks, float(app._edit_value.value))


def on_set_dihedral(app: Any) -> None:
    from .structure_edit import set_dihedral

    picks = _need(app, 4, "A dihedral")
    if picks:
        _apply(app, set_dihedral, *picks, float(app._edit_value.value))


def on_delete(app: Any) -> None:
    from .structure_edit import delete_atoms

    picks = current_picks(app)
    if not picks:
        _msg(app, "Pick the atoms to delete.", ok=False)
        return
    _set_picks(app, [])
    _apply(app, delete_atoms, picks)


def on_add_h(app: Any) -> None:
    from .structure_edit import add_hydrogen

    picks = current_picks(app)
    if not picks:
        _msg(app, "Pick the atom(s) to add a hydrogen to.", ok=False)
        return

    def _add_all(mol: Any) -> Tuple[Any, str]:
        notes = []
        for p in picks:
            mol, note = add_hydrogen(mol, p)
            notes.append(note)
        return mol, " ".join(notes)

    _apply(app, _add_all)


def on_change_element(app: Any) -> None:
    from .structure_edit import change_element

    picks = current_picks(app)
    symbol = str(app._edit_element_txt.value or "").strip()
    if not picks or not symbol:
        _msg(app, "Pick atom(s) and type the new element symbol.", ok=False)
        return

    def _change_all(mol: Any) -> Tuple[Any, str]:
        notes = []
        for p in picks:
            mol, note = change_element(mol, p, symbol)
            notes.append(note)
        return mol, " ".join(notes)

    _apply(app, _change_all)


def on_freeze(app: Any) -> None:
    """Add picks to Geometry Opt's Freeze atoms field."""
    from .structure_edit import parse_atom_list

    picks = current_picks(app)
    if not picks:
        _msg(app, "Pick the atoms to hold fixed.", ok=False)
        return
    field = getattr(app, "frozen_atoms_txt", None)
    if field is None:
        return
    try:
        existing = parse_atom_list(str(field.value or ""))
    except ValueError:
        existing = []
    merged = sorted(set(existing) | set(picks))
    field.value = ", ".join(str(i + 1) for i in merged)
    _msg(
        app,
        f"Atoms {field.value} will be held fixed in a Geometry Opt "
        "(set Calc. Type to Geometry Opt).",
    )


def on_undo(app: Any) -> None:
    stack = getattr(app, "_edit_undo_stack", [])
    if not stack:
        _msg(app, "Nothing to undo.", ok=False)
        return
    previous = stack.pop()
    app._set_molecule(previous, "Undo structure edit", sync_charge_mult=False)
    _msg(app, "Undid the last edit.")
    on_edit_atoms_changed(app)


def on_clear_picks(app: Any) -> None:
    _set_picks(app, [])
    app._edit_current_html.value = ""
