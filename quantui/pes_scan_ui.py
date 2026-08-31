"""PES Scan UI helpers — atom labels, range suggestions, and input validation.

Keeps widget wiring in :mod:`quantui.app_runflow` / :mod:`quantui.app_builders`
thin. All atom numbers exposed to the user are **1-based** (matching GaussView
and the rest of QuantUI); conversion to 0-based ``atom_indices`` for
:func:`quantui.pes_scan.run_pes_scan` happens at dispatch time.
"""

from __future__ import annotations

import html
from typing import List, Optional, Sequence, Tuple

from .measurement import angle, atom_label, dihedral, distance
from .molecule import Molecule

__all__ = [
    "atom_dropdown_options",
    "atom_list_html",
    "default_atom_selection",
    "format_scan_atom_summary",
    "scan_range_bounds",
    "suggest_scan_range",
    "validate_pes_scan_inputs",
]

_BOND_MIN = 0.05  # Å — slightly below typical H–H equilibrium
_ANGLE_MIN = -360.0
_ANGLE_MAX = 360.0


def atom_dropdown_options(molecule: Optional[Molecule]) -> List[Tuple[str, int]]:
    """Dropdown ``(label, value)`` pairs for the loaded molecule.

    Labels combine the 1-based index with the element-prefixed name, e.g.
    ``"1 O1"``, ``"2 H2"``.
    """
    if molecule is None or not molecule.atoms:
        return [("(load a molecule)", 1)]
    return [
        (f"{i + 1} {atom_label(molecule, i)}", i + 1)
        for i in range(len(molecule.atoms))
    ]


def atom_list_html(molecule: Optional[Molecule]) -> str:
    """Compact HTML reference table of numbered atoms for the PES panel."""
    if molecule is None or not molecule.atoms:
        return (
            '<span style="font-size:12px;color:#64748b">'
            "Load a molecule to see atom numbers.</span>"
        )
    cells = []
    for i, sym in enumerate(molecule.atoms):
        lbl = atom_label(molecule, i)
        cells.append(
            f'<span style="display:inline-block;margin:0 6px 4px 0;'
            f'padding:2px 7px;border-radius:4px;background:#f1f5f9;'
            f'font-size:12px;font-family:monospace">'
            f"<b>{i + 1}</b>&nbsp;{html.escape(lbl)} ({html.escape(sym)})</span>"
        )
    return (
        '<div style="line-height:1.6;margin:2px 0 4px 0">'
        + "".join(cells)
        + "</div>"
    )


def default_atom_selection(n_atoms: int, scan_type: str) -> List[int]:
    """Sensible default 1-based atom numbers for a molecule with ``n_atoms``."""
    if n_atoms < 1:
        return [1, 1, 1, 1]
    scan_type = scan_type.lower()
    if scan_type == "bond":
        return [1, min(2, n_atoms)]
    if scan_type == "angle":
        if n_atoms >= 3:
            # Vertex at atom 1 (e.g. H–O–H for water → 2–1–3)
            return [2, 1, 3]
        return [1, 1, min(3, n_atoms)]
    # dihedral
    return [1, min(2, n_atoms), min(3, n_atoms), min(4, n_atoms)]


def _clamp_selection(values: Sequence[int], n_atoms: int) -> List[int]:
    if n_atoms < 1:
        return list(values)
    return [max(1, min(int(v), n_atoms)) for v in values]


def scan_range_bounds(scan_type: str) -> Tuple[Optional[float], Optional[float]]:
    """``(min, max)`` for start/stop widgets; ``None`` means unbounded."""
    if scan_type.lower() == "bond":
        return (_BOND_MIN, 1000.0)
    return (_ANGLE_MIN, _ANGLE_MAX)


def suggest_scan_range(
    molecule: Optional[Molecule],
    scan_type: str,
    atom_numbers: Sequence[int],
) -> Tuple[float, float]:
    """Suggest a start/stop range from the current geometry.

    Falls back to textbook defaults when the molecule is missing or the
    coordinate cannot be measured (collinear dihedral, etc.).
    """
    scan_type = scan_type.lower()
    if scan_type == "bond":
        default = (0.8, 2.0)
    elif scan_type == "angle":
        default = (90.0, 120.0)
    else:
        default = (-180.0, 180.0)

    if molecule is None or not molecule.atoms:
        return default

    idx = [int(a) - 1 for a in atom_numbers]
    n_atoms = len(molecule.atoms)
    if any(i < 0 or i >= n_atoms for i in idx):
        return default

    try:
        if scan_type == "bond" and len(idx) >= 2:
            d = distance(molecule, idx[0], idx[1])
            margin = max(0.25, 0.2 * d)
            return (max(_BOND_MIN, d - margin), d + margin)
        if scan_type == "angle" and len(idx) >= 3:
            a = angle(molecule, idx[0], idx[1], idx[2])
            return (a - 20.0, a + 20.0)
        if scan_type == "dihedral" and len(idx) >= 4:
            dh = dihedral(molecule, idx[0], idx[1], idx[2], idx[3])
            return (dh - 60.0, dh + 60.0)
    except (ZeroDivisionError, ValueError):
        pass
    return default


def format_scan_atom_summary(
    molecule: Optional[Molecule], scan_type: str, atom_numbers: Sequence[int]
) -> str:
    """One-line description of the selected scan coordinate."""
    scan_type = scan_type.lower()
    if molecule is None or not molecule.atoms:
        nums = "–".join(str(a) for a in atom_numbers)
        return f"{scan_type.capitalize()} scan: atoms {nums}"

    labels = []
    for num in atom_numbers:
        i = int(num) - 1
        if 0 <= i < len(molecule.atoms):
            labels.append(f"{num} ({atom_label(molecule, i)})")
        else:
            labels.append(str(num))

    if scan_type == "bond":
        kind = "Bond"
    elif scan_type == "angle":
        kind = "Angle (vertex = atom 2)"
    else:
        kind = "Dihedral"
    return f"{kind}: " + " – ".join(labels)


def validate_pes_scan_inputs(
    molecule: Optional[Molecule],
    scan_type: str,
    atom_numbers: Sequence[int],
    start: float,
    stop: float,
    steps: int,
) -> List[str]:
    """Return human-readable problems; empty list means inputs are OK."""
    problems: List[str] = []
    if molecule is None or not molecule.atoms:
        problems.append("Load a molecule before running a PES scan.")
        return problems

    scan_type = scan_type.lower()
    n_atoms = len(molecule.atoms)
    expected = {"bond": 2, "angle": 3, "dihedral": 4}.get(scan_type)
    if expected is None:
        problems.append(f"Unknown scan type {scan_type!r}.")
        return problems

    nums = [int(a) for a in atom_numbers[:expected]]
    if len(nums) != expected:
        problems.append(
            f"A {scan_type} scan needs {expected} atoms; got {len(atom_numbers)}."
        )
        return problems

    for num in nums:
        if num < 1 or num > n_atoms:
            problems.append(
                f"Atom {num} is out of range — this molecule has {n_atoms} atom"
                f"{'s' if n_atoms != 1 else ''} (numbered 1–{n_atoms})."
            )

    if len(set(nums)) != len(nums):
        problems.append("Each atom in the scan coordinate must be unique.")

    if scan_type == "bond":
        if start <= 0 or stop <= 0:
            problems.append("Bond lengths must be positive (in Å).")
        if start >= stop:
            problems.append("Bond scan: Start must be less than Stop.")
    else:
        if start == stop:
            problems.append("Angle/dihedral scan: Start and Stop must differ.")

    if steps < 2:
        problems.append("Number of scan points must be at least 2.")

    return problems
