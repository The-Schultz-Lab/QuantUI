"""Deterministic displacement-id scheme for frequency checkpointing.

A Frequency job's numerical IR-intensity step needs one SCF per Cartesian
displacement of each atom (``+Δ`` and ``-Δ``) — ``6N`` total for an
``N``-atom molecule. Unlike Geometry-Opt's optimizer steps or PES-Scan's scan
points, these displacements have no inherent order once
``freq_ir_workers.py``'s ``QUANTUI_FREQ_PARALLEL`` opt-in runs them
concurrently across worker processes — completion order is
non-deterministic. So frequency checkpointing resumes a *set* of completed
displacement ids rather than a *prefix*, and this module is the single source
of truth for what those ids are — shared
by :mod:`quantui.freq_calc`'s serial loop, its parallel dispatch, and
:mod:`quantui.freq_ir_workers` / :mod:`quantui.freq_raman_workers`'s worker
processes — so none of them can ever disagree about what "required" means.

``atom_index`` is 0-based and follows ``mol.atom_coords()`` ordering — the
same atom-ordering stability :class:`~quantui.checkpoint.CalcIdentity`
already depends on elsewhere in the checkpoint layer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional, Tuple

_AXIS_LABELS = ("x", "y", "z")
_SIGN_LABELS = {1: "+", -1: "-"}
_SIGN_FROM_LABEL = {"+": 1, "-": -1}


def displacement_id(atom_index: int, axis: int, sign: int) -> str:
    """Stable string id for one Cartesian displacement.

    Args:
        atom_index: 0-based atom index.
        axis: 0, 1, or 2 for x, y, or z.
        sign: ``+1`` or ``-1``.

    Returns:
        e.g. ``"d000_x_+"`` for atom 0, x-axis, ``+Δ``.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2 (got {axis!r})")
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1 (got {sign!r})")
    return f"d{atom_index:03d}_{_AXIS_LABELS[axis]}_{_SIGN_LABELS[sign]}"


def parse_displacement_id(item_id: str) -> Tuple[int, int, int]:
    """Inverse of :func:`displacement_id` — returns ``(atom_index, axis, sign)``.

    Raises:
        ValueError: *item_id* is not a well-formed displacement id.
    """
    try:
        if not item_id.startswith("d"):
            raise ValueError
        atom_part, axis_label, sign_label = item_id[1:].split("_")
        atom_index = int(atom_part)
        axis = _AXIS_LABELS.index(axis_label)
        sign = _SIGN_FROM_LABEL[sign_label]
    except (ValueError, IndexError, KeyError) as exc:
        raise ValueError(f"not a displacement id: {item_id!r}") from exc
    return atom_index, axis, sign


def required_displacement_ids(n_atoms: int) -> List[str]:
    """Every displacement id for an *n_atoms*-atom molecule (``6N`` total).

    Deterministic ordering — atom, then axis, then ``+``/``-`` — purely so
    the id list is reproducible for logging/debugging; resume itself treats
    this as a set (see the module docstring), never a prefix.
    """
    ids: List[str] = []
    for atom_index in range(n_atoms):
        for axis in range(3):
            ids.append(displacement_id(atom_index, axis, 1))
            ids.append(displacement_id(atom_index, axis, -1))
    return ids


def displacement_set_name(
    prefix: str,
    mol: Any,
    *,
    xc: Optional[str],
    density_fit: bool,
    solvent: Optional[str],
    delta_bohr: float,
    unrestricted: bool,
) -> str:
    """Checkpoint item-set name for one reference geometry and setup.

    The ids above say *which* displacement a banked result is, not *what it
    was displaced from*. A resumed Frequency run whose reference geometry
    changed (pre-optimization ticked on the second attempt, a different
    seed or mode displacement) or whose SCF setup changed (density fitting,
    functional, solvent) used to splice the old geometry's dipoles into the
    new finite difference, giving wrong IR intensities with no warning. The
    set name now carries a digest of everything a displaced SCF depends on,
    so results banked for anything else are simply not found.

    *mol* is the PySCF ``Mole`` of the reference SCF.
    """
    coords = [
        [round(float(v), 6) + 0.0 for v in row] for row in mol.atom_coords().tolist()
    ]
    basis = mol.basis if isinstance(mol.basis, str) else repr(mol.basis)
    ecp = mol.ecp if isinstance(mol.ecp, str) else repr(sorted(dict(mol.ecp or {})))
    key = {
        "atoms": [mol.atom_symbol(i) for i in range(mol.natm)],
        "coords_bohr": coords,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "basis": basis,
        "ecp": ecp,
        "xc": xc,
        "density_fit": bool(density_fit),
        "solvent": solvent,
        "delta_bohr": float(delta_bohr),
        "unrestricted": bool(unrestricted),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}-{digest}"
