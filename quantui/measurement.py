"""Bond length / angle / dihedral measurement from picked atoms (M-MEASURE MEAS.1).

Thin wrappers around ASE's ``Atoms.get_distance`` / ``get_angle`` /
``get_dihedral`` -- the read-direction sibling of the coordinate math
:mod:`quantui.pes_scan` already writes with ``set_distance`` / ``set_angle`` /
``set_dihedral``. Pure geometry, no widget dependency, same separation
``ase_bridge.py`` / ``pes_scan.py`` already keep.

All indices here are 0-based, matching ``Molecule.atoms`` / ``.coordinates``
and ``pes_scan.py``'s ``atom_indices`` convention; ``atom_label`` converts to
the 1-based label used for display.
"""

from __future__ import annotations

from typing import Sequence

from .ase_bridge import molecule_to_atoms
from .molecule import Molecule

__all__ = ["distance", "angle", "dihedral", "atom_label", "describe_picks"]


def distance(molecule: Molecule, i: int, j: int) -> float:
    """Distance between atoms ``i`` and ``j``, in Angstroms."""
    atoms = molecule_to_atoms(molecule)
    return float(atoms.get_distance(i, j))


def angle(molecule: Molecule, i: int, j: int, k: int) -> float:
    """Angle i-j-k in degrees, vertex at ``j``."""
    atoms = molecule_to_atoms(molecule)
    return float(atoms.get_angle(i, j, k))


def dihedral(molecule: Molecule, i: int, j: int, k: int, l: int) -> float:  # noqa: E741
    """Dihedral i-j-k-l in degrees.

    Raises:
        ZeroDivisionError: from ASE, when three consecutive atoms of the
            chain are collinear -- the dihedral plane is then undefined.
            Callers that surface this to a user (the click-to-measure
            picker) should catch it and show a plain-language message
            instead of crashing the click callback.
    """
    atoms = molecule_to_atoms(molecule)
    return float(atoms.get_dihedral(i, j, k, l))


def atom_label(molecule: Molecule, i: int) -> str:
    """1-based, element-prefixed atom label, e.g. ``"O1"``, ``"H2"``."""
    return f"{molecule.atoms[i]}{i + 1}"


def describe_picks(molecule: Molecule, picks: Sequence[int]) -> str:
    """Progressive click-to-measure readout for 1-4 picked atom indices.

    Mirrors GaussView's convention: each atom after the first is annotated
    with the measurement it newly completes -- bond length for the 2nd pick,
    angle (vertex at the 2nd atom) for the 3rd, dihedral for the 4th.
    """
    if not picks:
        return "Click an atom to start measuring."
    parts = [atom_label(molecule, picks[0])]
    if len(picks) >= 2:
        d = distance(molecule, picks[0], picks[1])
        parts.append(f"{atom_label(molecule, picks[1])} ({d:.3f} Å)")
    if len(picks) >= 3:
        a = angle(molecule, picks[0], picks[1], picks[2])
        parts.append(f"{atom_label(molecule, picks[2])} ({a:.1f}°)")
    if len(picks) >= 4:
        try:
            dh = dihedral(molecule, picks[0], picks[1], picks[2], picks[3])
            parts.append(f"{atom_label(molecule, picks[3])} ({dh:.1f}°)")
        except ZeroDivisionError:
            parts.append(
                f"{atom_label(molecule, picks[3])} "
                "(dihedral undefined — atoms are collinear)"
            )
    return "Picked: " + " → ".join(parts)
