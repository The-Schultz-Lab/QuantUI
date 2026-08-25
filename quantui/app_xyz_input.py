"""Interactive XYZ input tab: atom table, cleanup preview, and text sync."""

from __future__ import annotations

from typing import Any, List, Tuple

import ipywidgets as widgets

from quantui import theme as _theme
from quantui.molecule import parse_xyz_input
from quantui.xyz_input import (
    format_xyz_body,
    load_molecule_from_xyz_text,
    propose_xyz_cleanup,
)


def build_xyz_interactive_widgets(app: Any, *, layout_fn: Any) -> None:
    """Create the atom-row builder and cleanup preview widgets on ``app``."""
    app._layout_fn = layout_fn
    app._xyz_table_rows: List[Tuple[Any, Any, Any, Any, Any]] = []

    app.xyz_table_header = widgets.HTML(
        f'<div style="font-size:12px;color:{_theme.TEXT_MUTED};margin:6px 0 2px">'
        "<b>Element</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>X (Å)</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Y (Å)</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "<b>Z (Å)</b></div>"
    )
    app.xyz_table_box = widgets.VBox(
        layout=layout_fn(gap="4px", width="440px"),
    )
    app.xyz_add_atom_btn = widgets.Button(
        description="Add atom",
        icon="plus",
        layout=layout_fn(width="110px", height="28px"),
        tooltip="Add a blank row to the atom table",
    )
    app.xyz_fill_table_btn = widgets.Button(
        description="Fill table from text",
        icon="table",
        layout=layout_fn(width="160px", height="28px"),
        tooltip="Parse the textarea above into the atom table",
    )
    app.xyz_apply_table_btn = widgets.Button(
        description="Apply table to text",
        icon="pencil",
        layout=layout_fn(width="160px", height="28px"),
        tooltip="Write the atom table into the textarea above",
    )
    app.xyz_cleanup_btn = widgets.Button(
        description="Clean up coordinates",
        icon="magic",
        button_style="",
        layout=layout_fn(width="180px", height="28px"),
        tooltip="Propose a normalized version of the coordinate text",
    )
    app.xyz_cleanup_notes = widgets.HTML(value="")
    app.xyz_cleanup_preview = widgets.Textarea(
        disabled=True,
        layout=layout_fn(width="440px", height="110px"),
    )
    app.xyz_cleanup_accept_btn = widgets.Button(
        description="Accept cleanup",
        icon="check",
        button_style="success",
        layout=layout_fn(width="150px", height="30px"),
    )
    app.xyz_cleanup_reject_btn = widgets.Button(
        description="Reject",
        icon="times",
        button_style="warning",
        layout=layout_fn(width="100px", height="30px"),
    )
    app.xyz_cleanup_preview_box = widgets.VBox(
        [
            widgets.HTML(
                f'<span style="font-size:13px;color:{_theme.TEXT_BODY}">'
                "Proposed cleanup</span>"
            ),
            app.xyz_cleanup_notes,
            app.xyz_cleanup_preview,
            widgets.HBox(
                [app.xyz_cleanup_accept_btn, app.xyz_cleanup_reject_btn],
                layout=layout_fn(gap="8px", margin="4px 0 0"),
            ),
        ],
        layout=layout_fn(gap="4px", margin="8px 0 0"),
    )
    app.xyz_cleanup_preview_box.layout.display = "none"

    _add_xyz_table_row(app, layout_fn, symbol="O", x=0.0, y=0.0, z=0.0)
    _add_xyz_table_row(app, layout_fn, symbol="H", x=0.757, y=0.587, z=0.0)
    _add_xyz_table_row(app, layout_fn, symbol="H", x=-0.757, y=0.587, z=0.0)


def _add_xyz_table_row(
    app: Any,
    layout_fn: Any,
    *,
    symbol: str = "H",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> None:
    sym_w = widgets.Text(
        value=symbol,
        layout=layout_fn(width="52px"),
        placeholder="H",
    )
    x_w = widgets.FloatText(value=x, layout=layout_fn(width="92px"), step=0.01)
    y_w = widgets.FloatText(value=y, layout=layout_fn(width="92px"), step=0.01)
    z_w = widgets.FloatText(value=z, layout=layout_fn(width="92px"), step=0.01)
    remove_btn = widgets.Button(
        icon="trash",
        layout=layout_fn(width="36px", height="28px"),
        tooltip="Remove this atom row",
    )

    row_box = widgets.HBox(
        [sym_w, x_w, y_w, z_w, remove_btn],
        layout=layout_fn(gap="6px", align_items="center"),
    )
    row_data = (sym_w, x_w, y_w, z_w, remove_btn)
    app._xyz_table_rows.append(row_data)
    app.xyz_table_box.children = (*app.xyz_table_box.children, row_box)

    def _remove(_btn: Any, *, _row=row_data, _box=row_box) -> None:
        if _row in app._xyz_table_rows:
            app._xyz_table_rows.remove(_row)
        children = list(app.xyz_table_box.children)
        if _box in children:
            children.remove(_box)
            app.xyz_table_box.children = tuple(children)

    remove_btn.on_click(_remove)


def on_xyz_add_atom(app: Any, _btn: Any = None) -> None:
    _add_xyz_table_row(app, app._layout_fn)


def table_rows_to_atoms_coords(app: Any) -> Tuple[List[str], List[List[float]]]:
    atoms: List[str] = []
    coords: List[List[float]] = []
    for sym_w, x_w, y_w, z_w, _remove in app._xyz_table_rows:
        sym = sym_w.value.strip()
        if not sym:
            continue
        atoms.append(sym)
        coords.append([float(x_w.value), float(y_w.value), float(z_w.value)])
    if not atoms:
        raise ValueError("Add at least one atom with an element symbol.")
    return atoms, coords


def sync_textarea_from_table(app: Any) -> None:
    atoms, coords = table_rows_to_atoms_coords(app)
    app.xyz_area.value = format_xyz_body(atoms, coords)


def sync_table_from_textarea(app: Any, *, layout_fn: Any) -> None:
    atoms, coords = parse_xyz_input(app.xyz_area.value.strip())
    app.xyz_table_box.children = ()
    app._xyz_table_rows.clear()
    for sym, (x, y, z) in zip(atoms, coords):
        _add_xyz_table_row(app, layout_fn, symbol=sym, x=x, y=y, z=z)


def on_xyz_apply_table(app: Any, _btn: Any = None) -> None:
    try:
        sync_textarea_from_table(app)
        app.xyz_msg.value = "Updated coordinate text from the atom table."
    except Exception as exc:
        app.xyz_msg.value = f"Table error: {exc}"


def on_xyz_fill_table(app: Any, _btn: Any = None) -> None:
    try:
        sync_table_from_textarea(app, layout_fn=app._layout_fn)
        app.xyz_msg.value = "Filled atom table from coordinate text."
    except Exception as exc:
        app.xyz_msg.value = f"Parse error: {exc}"


def on_xyz_cleanup(app: Any, _btn: Any = None) -> None:
    text = app.xyz_area.value.strip()
    if not text:
        app.xyz_msg.value = "Paste or enter coordinates before cleaning up."
        return
    try:
        cleaned, notes = propose_xyz_cleanup(text)
    except Exception as exc:
        app.xyz_msg.value = f"Cleanup error: {exc}"
        return

    if cleaned.strip() == text.strip():
        app.xyz_msg.value = "Coordinates are already in standard format."
        app.xyz_cleanup_preview_box.layout.display = "none"
        return

    note_html = "".join(f'<li style="margin:2px 0">{note}</li>' for note in notes)
    app.xyz_cleanup_notes.value = (
        f'<ul style="margin:4px 0 6px;padding-left:18px;'
        f'color:{_theme.TEXT_MUTED};font-size:12px">{note_html}</ul>'
    )
    app.xyz_cleanup_preview.value = cleaned
    app.xyz_cleanup_preview_box.layout.display = ""
    app.xyz_msg.value = "Review the proposed cleanup, then accept or reject."


def on_xyz_cleanup_accept(app: Any, _btn: Any = None) -> None:
    app.xyz_area.value = app.xyz_cleanup_preview.value
    app.xyz_cleanup_preview_box.layout.display = "none"
    app.xyz_msg.value = "Accepted cleanup — coordinate text updated."
    try:
        sync_table_from_textarea(app, layout_fn=app._layout_fn)
    except Exception:
        pass


def on_xyz_cleanup_reject(app: Any, _btn: Any = None) -> None:
    app.xyz_cleanup_preview_box.layout.display = "none"
    app.xyz_msg.value = "Cleanup rejected — original text unchanged."


def on_load_xyz(app: Any, _btn: Any = None) -> None:
    """Load geometry from the textarea using calc-setup charge/multiplicity."""
    text = app.xyz_area.value.strip()
    if not text:
        app.xyz_msg.value = "Enter or paste coordinates first."
        return
    try:
        mol, spin_note = load_molecule_from_xyz_text(
            text,
            charge=int(app.charge_si.value),
            multiplicity=int(app.mult_si.value),
        )
        app._set_molecule(
            mol,
            "Loaded from XYZ input",
            sync_charge_mult=False,
        )
        if spin_note:
            app.xyz_msg.value = (
                f"Loaded {mol.get_formula()}. Note: {spin_note} "
                "Adjust charge or multiplicity in Calculation Setup before running."
            )
        else:
            app.xyz_msg.value = f"Loaded {mol.get_formula()} ({len(mol.atoms)} atoms)."
    except Exception as exc:
        app.xyz_msg.value = f"Parse error: {exc}"
