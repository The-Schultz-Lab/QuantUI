"""
Fast force-field geometry pre-optimization using RDKit (MMFF94 / UFF).

Optional step before a quantum-chemistry calculation to clean up a student's
starting geometry — removing severe steric clashes or distorted bond lengths —
without the cost of a full QM optimization.

The force field is **bonded**: MMFF94, falling back to UFF for atoms MMFF lacks
parameters for — the same chemistry QuantUI uses to build its curated library.
Unlike the previous Lennard-Jones potential (which models no bonds and relaxed
atoms toward a generic close-packed cluster, *distorting* even good geometries —
the "garbled aspirin" of the 2026-06-08 manual test), a bonded FF preserves
molecular connectivity, so a reasonable geometry stays reasonable.

Non-destructive guarantee
-------------------------
If RDKit is unavailable, bond perception fails, or no force field has parameters
for the molecule, :func:`preoptimize` returns the **original** geometry
unchanged (RMSD 0.0) rather than a mangled one. Pre-opt can only improve or
no-op — never degrade. (M-PREOPT / PREOPT.1.)

Limitation: bond perception is distance-based, so a *wildly* broken input (atoms
so far apart or so clashed that bonds can't be inferred) yields the no-op rather
than a repair. That is the intended trade-off — far safer than the old LJ
behavior, which "fixed" such cases by collapsing everything into a blob.

Platform notes
--------------
Uses RDKit, which ships in the QuantUI container and conda environments and is
already used throughout QuantUI for structure handling (search, library). If
RDKit is absent the step no-ops gracefully (see above). No PySCF or SLURM
dependency, so it runs on Windows, Linux, and WSL.

Typical usage
-------------
>>> from quantui.preopt import preoptimize
>>> optimized_mol, rmsd = preoptimize(molecule)
>>> print(f"Geometry changed by {rmsd:.3f} Å (RMSD)")
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .molecule import Molecule

logger = logging.getLogger(__name__)

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem  # noqa: F401 — availability probe

    _RDKIT_AVAILABLE = True
except ImportError:
    pass


def _copy_molecule(molecule: Molecule) -> Molecule:
    """Return a fresh Molecule with the same data (never mutate the input)."""
    return Molecule(
        atoms=list(molecule.atoms),
        coordinates=[list(c) for c in molecule.coordinates],
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )


# Cap on captured preview frames so the relaxation animation stays light
# (single-iteration steps; reasonable geometries converge well under this).
_PREVIEW_FRAME_CAP = 40


def _conf_coords(conf, n_atoms: int) -> List[List[float]]:
    """Extract an RDKit conformer's coordinates as a plain list of [x, y, z]."""
    return [
        [
            float(conf.GetAtomPosition(i).x),
            float(conf.GetAtomPosition(i).y),
            float(conf.GetAtomPosition(i).z),
        ]
        for i in range(n_atoms)
    ]


def _rdkit_ff_relax(
    molecule: Molecule, steps: int, *, capture_frames: bool = False
) -> Tuple[List[List[float]], str, Optional[List[List[List[float]]]]]:
    """Relax ``molecule`` with a bonded force field.

    Returns ``(final_coords, ff_name, frames)``. ``frames`` is ``None`` unless
    ``capture_frames`` is True, in which case it is a list of per-iteration
    coordinate snapshots (starting geometry first) for animating the relaxation.

    Mirrors the XYZ→bonds→FF pattern QuantUI already uses (``app_exports``,
    ``pubchem``, ``scripts/build_curated_library.py``). Raises on any failure
    (no bonds perceived, no FF parameters, atom-count change) so the caller can
    fall back to the original geometry. Atom order is preserved — RDKit keeps
    the XYZ order through ``MolFromXYZBlock`` + ``DetermineBonds`` — so the
    returned coordinates map 1:1 onto ``molecule.atoms``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    xyz_block = (
        f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
        f"{molecule.to_xyz_string()}\n"
    )
    rdmol = Chem.MolFromXYZBlock(xyz_block)
    if rdmol is None:
        raise ValueError("RDKit could not parse the molecule geometry")
    # Perceive connectivity (with the correct net charge) so a bonded FF applies.
    rdDetermineBonds.DetermineBonds(rdmol, charge=int(molecule.charge))

    n = rdmol.GetNumAtoms()
    conf = rdmol.GetConformer()

    if AllChem.MMFFHasAllMoleculeParams(rdmol):
        ff_name = "MMFF94"
    elif AllChem.UFFHasAllMoleculeParams(rdmol):
        ff_name = "UFF"
    else:
        raise ValueError("no MMFF or UFF parameters for this molecule")

    if not capture_frames:
        # Fast path: one bulk minimize, no per-step snapshots.
        if ff_name == "MMFF94":
            AllChem.MMFFOptimizeMolecule(rdmol, maxIters=int(steps))
        else:
            AllChem.UFFOptimizeMolecule(rdmol, maxIters=int(steps))
        coords = _conf_coords(conf, n)
        if len(coords) != len(molecule.atoms):
            raise ValueError("atom count changed during FF relaxation")
        return coords, ff_name, None

    # Frame-capturing path: build the force field and step it one iteration at a
    # time, snapshotting the geometry, so we can animate the real relaxation.
    if ff_name == "MMFF94":
        ff = AllChem.MMFFGetMoleculeForceField(
            rdmol, AllChem.MMFFGetMoleculeProperties(rdmol)
        )
    else:
        ff = AllChem.UFFGetMoleculeForceField(rdmol)
    if ff is None:
        raise ValueError("could not build force field for frame capture")
    ff.Initialize()

    frames: List[List[List[float]]] = [_conf_coords(conf, n)]
    converged = False
    for _ in range(int(steps)):
        more = ff.Minimize(maxIts=1)  # 0 == converged, 1 == needs more
        frames.append(_conf_coords(conf, n))
        if more == 0:
            converged = True
            break
        if len(frames) >= _PREVIEW_FRAME_CAP:
            break
    if not converged:
        # Finish the relaxation in bulk (don't snapshot each remaining step);
        # append the final geometry so the animation ends at the real minimum.
        ff.Minimize(maxIts=int(steps))
        frames.append(_conf_coords(conf, n))

    coords = _conf_coords(conf, n)
    if len(coords) != len(molecule.atoms):
        raise ValueError("atom count changed during FF relaxation")
    return coords, ff_name, frames


def preoptimize(
    molecule: Molecule,
    fmax: float = 0.05,
    steps: int = 200,
) -> Tuple[Molecule, float]:
    """Run a fast bonded force-field (MMFF94 / UFF) geometry pre-optimization.

    The input ``molecule`` is **never mutated** — a new ``Molecule`` (same
    ``charge`` / ``multiplicity``) is always returned.

    Args:
        molecule: Input molecule. May have a non-ideal starting geometry.
        fmax: Retained for API compatibility. RDKit's force-field optimizer
            uses its own internal gradient tolerance; the iteration budget is
            controlled by ``steps``.
        steps: Maximum force-field iterations (RDKit ``maxIters``). Default 200.

    Returns:
        ``(optimized_molecule, rmsd)`` — ``rmsd`` is the RMS atomic displacement
        (Å) between the input and relaxed geometries. On **any** failure
        (RDKit missing, bond perception fails, no FF parameters) the original
        geometry is returned unchanged with ``rmsd = 0.0`` — pre-opt never
        degrades a geometry.
    """
    import numpy as np

    if not _RDKIT_AVAILABLE:
        logger.warning("RDKit unavailable — pre-opt skipped, geometry unchanged.")
        return _copy_molecule(molecule), 0.0

    original = np.asarray(molecule.coordinates, dtype=float)
    try:
        coords, ff_name, _frames = _rdkit_ff_relax(molecule, steps)
    except Exception as exc:  # noqa: BLE001 — any FF failure → non-destructive no-op
        logger.warning(
            "Bonded-FF pre-opt failed (%s); returning original geometry unchanged.",
            exc,
        )
        return _copy_molecule(molecule), 0.0

    optimized = np.asarray(coords, dtype=float)
    rmsd = float(np.sqrt(np.mean(np.sum((optimized - original) ** 2, axis=1))))

    optimized_molecule = Molecule(
        atoms=list(molecule.atoms),
        coordinates=optimized.tolist(),
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )
    logger.info(
        "%s pre-optimization complete: RMSD=%.4f Å (maxIters=%d)",
        ff_name,
        rmsd,
        steps,
    )
    return optimized_molecule, rmsd


def preoptimize_with_trajectory(
    molecule: Molecule,
    fmax: float = 0.05,
    steps: int = 200,
) -> Tuple[Molecule, float, List[List[List[float]]]]:
    """Bonded-FF pre-opt that also returns the relaxation **trajectory**.

    Like :func:`preoptimize`, but returns ``(optimized_molecule, rmsd, frames)``
    where ``frames`` is a list of per-iteration coordinate snapshots (the
    starting geometry first, the relaxed geometry last) for animating the
    relaxation in the interactive "Preview pre-optimization" flow (M-PREOPT
    PREOPT.2). Same non-destructive guarantee: on any failure the original
    geometry is returned unchanged with ``rmsd = 0.0`` and a single-frame
    trajectory (just the input), so the viewer always has something to show.
    """
    import numpy as np

    original = np.asarray(molecule.coordinates, dtype=float)
    fallback_frames = [original.tolist()]

    if not _RDKIT_AVAILABLE:
        logger.warning(
            "RDKit unavailable — pre-opt preview skipped, geometry unchanged."
        )
        return _copy_molecule(molecule), 0.0, fallback_frames

    try:
        coords, ff_name, frames = _rdkit_ff_relax(molecule, steps, capture_frames=True)
    except Exception as exc:  # noqa: BLE001 — any FF failure → non-destructive no-op
        logger.warning(
            "Bonded-FF pre-opt preview failed (%s); geometry unchanged.", exc
        )
        return _copy_molecule(molecule), 0.0, fallback_frames

    optimized = np.asarray(coords, dtype=float)
    rmsd = float(np.sqrt(np.mean(np.sum((optimized - original) ** 2, axis=1))))
    optimized_molecule = Molecule(
        atoms=list(molecule.atoms),
        coordinates=optimized.tolist(),
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )
    logger.info(
        "%s pre-opt preview: RMSD=%.4f Å, %d frames",
        ff_name,
        rmsd,
        len(frames) if frames else 1,
    )
    return optimized_molecule, rmsd, frames or fallback_frames
