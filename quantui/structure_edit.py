"""Structure editing operations (M-INTERACT INT.8, DEC-023 Tier-1 #8).

GaussView-style edits on a :class:`~quantui.molecule.Molecule`, each
returning a NEW molecule (the caller keeps the old one for undo):

- set a bond length, bond angle or dihedral to a value — the group of atoms
  on the moving side of the bond moves rigidly, as in GaussView's Modify
  Bond/Angle/Dihedral;
- delete atoms; change an atom's element;
- add one hydrogen to an atom, pointing away from its existing bonds
  (GaussView "Add Valence", one H at a time).

Atom indices are 0-based here; the UI shows 1-based numbers.

"Moving side": remove the bond between the fixed atom and the moving atom
and take the connected group that holds the moving atom. When the two atoms
are in the same ring there is no such group, so only the moving atom moves,
and the returned note says so.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Set, Tuple, cast

import numpy as np

from .connectivity import covalent_bonds
from .molecule import Molecule

# Typical X–H bond lengths (Å) for "Add H"; others use 1.09.
_XH_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34, "P": 1.42, "B": 1.19}


def _copy(mol: Molecule, atoms: Sequence[str], coords: np.ndarray) -> Molecule:
    return Molecule(
        list(atoms),
        [list(map(float, c)) for c in coords],
        charge=mol.charge,
        multiplicity=mol.multiplicity,
        validate_spin=False,
    )


def _check(mol: Molecule, *indices: int) -> None:
    n = len(mol.atoms)
    for i in indices:
        if not 0 <= i < n:
            raise ValueError(f"Atom {i + 1} does not exist (this structure has {n}).")
    if len(set(indices)) != len(indices):
        raise ValueError("Pick different atoms.")


def moving_side(mol: Molecule, fixed: int, moving: int) -> Tuple[List[int], bool]:
    """Atoms that move with *moving* when the *fixed*–*moving* bond changes.

    Returns ``(indices, in_ring)``. ``in_ring`` is True when *moving* stays
    connected to *fixed* without that bond, in which case only *moving* is
    returned.
    """
    bonds = covalent_bonds(mol.atoms, mol.coordinates)
    adj: dict = {i: set() for i in range(len(mol.atoms))}
    for a, b in bonds:
        if {a, b} == {fixed, moving}:
            continue
        adj[a].add(b)
        adj[b].add(a)
    seen: Set[int] = {moving}
    stack = [moving]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in seen:
                seen.add(b)
                stack.append(b)
    if fixed in seen:
        return [moving], True
    return sorted(seen), False


def _rotate(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray, theta: float
) -> np.ndarray:
    """Rodrigues rotation of *points* by *theta* (rad) about *axis* through *origin*."""
    k = axis / np.linalg.norm(axis)
    p = points - origin
    rotated = (
        p * math.cos(theta)
        + np.cross(k, p) * math.sin(theta)
        + np.outer(p @ k, k) * (1 - math.cos(theta))
    )
    return cast(np.ndarray, rotated + origin)


def _ring_note(in_ring: bool) -> str:
    return (
        " The atoms are in a ring, so only the last picked atom moved; "
        "optimize afterwards."
        if in_ring
        else ""
    )


def set_bond_length(
    mol: Molecule, i: int, j: int, length: float
) -> Tuple[Molecule, str]:
    """Set the i–j distance to *length* Å, moving j's side."""
    _check(mol, i, j)
    if length <= 0.1:
        raise ValueError("Bond length must be more than 0.1 Å.")
    xyz = np.asarray(mol.coordinates, dtype=float)
    vec = xyz[j] - xyz[i]
    d = float(np.linalg.norm(vec))
    if d < 1e-8:
        raise ValueError("The two atoms are on top of each other.")
    side, in_ring = moving_side(mol, i, j)
    xyz[side] += (length - d) * vec / d
    return _copy(mol, mol.atoms, xyz), (
        f"Atoms {i + 1}–{j + 1}: {d:.3f} → {length:.3f} Å." + _ring_note(in_ring)
    )


def set_angle(
    mol: Molecule, i: int, j: int, k: int, degrees: float
) -> Tuple[Molecule, str]:
    """Set the i–j–k angle to *degrees*, rotating k's side about j."""
    _check(mol, i, j, k)
    if not 0.0 < degrees < 180.0:
        raise ValueError("Angle must be between 0° and 180°.")
    xyz = np.asarray(mol.coordinates, dtype=float)
    a, b = xyz[i] - xyz[j], xyz[k] - xyz[j]
    cos0 = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    current = math.degrees(math.acos(max(-1.0, min(1.0, cos0))))
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-6:
        # Linear: any perpendicular works as the bending axis.
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, a)) > 0.9 * np.linalg.norm(a):
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, trial)
    side, in_ring = moving_side(mol, j, k)
    xyz[side] = _rotate(xyz[side], xyz[j], axis, math.radians(degrees - current))
    return _copy(mol, mol.atoms, xyz), (
        f"Angle {i + 1}–{j + 1}–{k + 1}: {current:.1f}° → {degrees:.1f}°."
        + _ring_note(in_ring)
    )


def _measured_dihedral(
    mol: Molecule, xyz: np.ndarray, i: int, j: int, k: int, l: int  # noqa: E741
) -> float:
    """ASE's i–j–k–l dihedral (0–360°): the same number the measure panel shows."""
    from .measurement import dihedral

    return dihedral(_copy(mol, mol.atoms, xyz), i, j, k, l)


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a − b on a circle (degrees)."""
    return (a - b + 180.0) % 360.0 - 180.0


def set_dihedral(
    mol: Molecule, i: int, j: int, k: int, l: int, degrees: float  # noqa: E741
) -> Tuple[Molecule, str]:
    """Set the i–j–k–l dihedral to *degrees*, rotating the k side about j–k."""
    _check(mol, i, j, k, l)
    xyz0 = np.asarray(mol.coordinates, dtype=float)
    target = float(degrees) % 360.0
    current = _measured_dihedral(mol, xyz0, i, j, k, l)
    side, in_ring = moving_side(mol, j, k)
    if in_ring:
        # Rotating about a ring bond only makes sense for the end atom.
        side = [l]
    delta = math.radians(_angle_diff(target, current))
    axis = xyz0[k] - xyz0[j]
    best = None
    # The sense of a right-handed rotation about j→k relative to ASE's
    # dihedral sign is checked, not assumed: keep whichever lands on target.
    for sense in (1.0, -1.0):
        xyz = xyz0.copy()
        xyz[side] = _rotate(xyz[side], xyz0[k], axis, sense * delta)
        err = abs(_angle_diff(_measured_dihedral(mol, xyz, i, j, k, l), target))
        if best is None or err < best[0]:
            best = (err, xyz)
    assert best is not None
    return _copy(mol, mol.atoms, best[1]), (
        f"Dihedral {i + 1}–{j + 1}–{k + 1}–{l + 1}: {current:.1f}° → {target:.1f}°."
        + _ring_note(in_ring)
    )


def delete_atoms(mol: Molecule, indices: Sequence[int]) -> Tuple[Molecule, str]:
    """Remove atoms; at least one must remain."""
    _check(mol, *indices)
    drop = set(indices)
    if len(drop) >= len(mol.atoms):
        raise ValueError("Cannot delete every atom.")
    keep = [n for n in range(len(mol.atoms)) if n not in drop]
    xyz = np.asarray(mol.coordinates, dtype=float)[keep]
    labels = ", ".join(f"{mol.atoms[n]}{n + 1}" for n in sorted(drop))
    return _copy(mol, [mol.atoms[n] for n in keep], xyz), (
        f"Deleted {labels}. Atoms after them are renumbered; check charge "
        "and multiplicity."
    )


def change_element(mol: Molecule, i: int, symbol: str) -> Tuple[Molecule, str]:
    """Replace atom *i*'s element (geometry unchanged)."""
    _check(mol, i)
    from .xyz_input import normalize_element_symbol

    new = normalize_element_symbol(symbol.strip())
    atoms = list(mol.atoms)
    old = atoms[i]
    atoms[i] = new
    trial = _copy(mol, atoms, np.asarray(mol.coordinates, dtype=float))
    return trial, (
        f"Atom {i + 1}: {old} → {new}. Bond lengths are unchanged; optimize "
        "afterwards and check charge and multiplicity."
    )


def add_hydrogen(mol: Molecule, i: int) -> Tuple[Molecule, str]:
    """Add one H to atom *i*, pointing away from its existing bonds."""
    _check(mol, i)
    xyz = np.asarray(mol.coordinates, dtype=float)
    neighbours = [
        b if a == i else a
        for a, b in covalent_bonds(mol.atoms, xyz.tolist())
        if i in (a, b)
    ]
    if neighbours:
        dirs = [(xyz[n] - xyz[i]) / np.linalg.norm(xyz[n] - xyz[i]) for n in neighbours]
        away = -np.sum(dirs, axis=0)
        if np.linalg.norm(away) < 1e-3:
            # Neighbours cancel (linear or planar-symmetric): go perpendicular.
            ref = dirs[0]
            trial = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(trial, ref)) > 0.9:
                trial = np.array([1.0, 0.0, 0.0])
            away = np.cross(ref, trial)
            if len(dirs) >= 2:
                normal = np.cross(dirs[0], dirs[1])
                if np.linalg.norm(normal) > 1e-3:
                    away = normal
        direction = away / np.linalg.norm(away)
    else:
        direction = np.array([0.0, 0.0, 1.0])
    length = _XH_LENGTH.get(mol.atoms[i], 1.09)
    new_pos = xyz[i] + length * direction
    return _copy(mol, list(mol.atoms) + ["H"], np.vstack([xyz, new_pos])), (
        f"Added H{len(mol.atoms) + 1} to {mol.atoms[i]}{i + 1} "
        f"({length:.2f} Å). Optimize afterwards; check multiplicity."
    )


def parse_atom_list(text: str, n_atoms: Optional[int] = None) -> List[int]:
    """'1, 3 5-7' (1-based, ranges allowed) → sorted 0-based indices."""
    out: Set[int] = set()
    for token in text.replace(",", " ").split():
        if "-" in token:
            lo, hi = token.split("-", 1)
            a, b = int(lo), int(hi)
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(token))
    if any(n < 1 for n in out):
        raise ValueError("Atom numbers start at 1.")
    if n_atoms is not None and any(n > n_atoms for n in out):
        raise ValueError(f"This structure has only {n_atoms} atoms.")
    return sorted(n - 1 for n in out)
