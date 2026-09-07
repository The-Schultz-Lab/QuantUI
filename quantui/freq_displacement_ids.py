"""Deterministic displacement-id scheme for frequency checkpointing (CHK.4.2).

A Frequency job's numerical IR-intensity step needs one SCF per Cartesian
displacement of each atom (``+Δ`` and ``-Δ``) — ``6N`` total for an
``N``-atom molecule. Unlike Geometry-Opt's optimizer steps or PES-Scan's scan
points, these displacements have no inherent order once
``freq_ir_workers.py``'s ``QUANTUI_FREQ_PARALLEL`` opt-in runs them
concurrently across worker processes — completion order is
non-deterministic. So CHK.4 (see
``QuantUI-development-tracking/TODO/roadmaps/34-m-checkpoint-calc-restart-roadmap.md``)
resumes a *set* of completed displacement ids rather than a *prefix*, and
this module is the single source of truth for what those ids are — shared
by :mod:`quantui.freq_calc`'s serial loop, its parallel dispatch, and
:mod:`quantui.freq_ir_workers` / :mod:`quantui.freq_raman_workers`'s worker
processes — so none of them can ever disagree about what "required" means.

``atom_index`` is 0-based and follows ``mol.atom_coords()`` ordering — the
same atom-ordering stability :class:`~quantui.checkpoint.CalcIdentity`
already depends on elsewhere in the checkpoint layer.
"""

from __future__ import annotations

from typing import List, Tuple

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
