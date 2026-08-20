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
no-op — never degrade.

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

import contextlib
import importlib.util
import logging
import threading
from typing import Iterator, List, Optional, Tuple

from .molecule import Molecule

logger = logging.getLogger(__name__)

_RDKIT_AVAILABLE = False
try:
    from rdkit import Chem  # noqa: F401 — availability probe

    _RDKIT_AVAILABLE = True
except ImportError:
    pass

# Optional GFN-FF (xtb) backend for metal-capable pre-optimization. RDKit's
# organic valence model can't touch a transition metal; GFN-FF (Grimme's general
# force field, via xtb-python + ASE) covers the whole periodic table. Probe with
# find_spec so merely importing quantui doesn't pull in libxtb/ASE — the heavy
# import happens lazily inside _xtb_gfnff_relax. xtb ships Linux pip wheels only;
# on Windows/macOS it comes from conda (see environment.yml).
_XTB_AVAILABLE = (
    importlib.util.find_spec("xtb") is not None
    and importlib.util.find_spec("ase") is not None
)

if _XTB_AVAILABLE:
    # libxtb prints from Fortran and gfortran buffers preconnected units (stdout)
    # in its *own* runtime buffer — not libc's — so a buffer flush would land on
    # the real terminal after our fd redirect is torn down. Ask gfortran to write
    # unbuffered instead, so the redirect in _contained_xtb_run catches it live.
    # Must be set before libxtb's runtime initialises (i.e. before it's imported).
    import os as _os

    _os.environ.setdefault("GFORTRAN_UNBUFFERED_PRECONNECTED", "1")

# Serialises GFN-FF runs: each one chdir's to a temp scratch dir and redirects
# the process stdout/stderr file descriptors (libxtb prints from Fortran, below
# Python's sys.stdout), both of which are process-global.
_XTB_LOCK = threading.Lock()

# ASE force convergence threshold (eV/Å) for the GFN-FF pre-opt — loose, since
# this is a cleanup pass before the real DFT optimization, not a final geometry.
_XTB_FMAX = 0.05


def _copy_molecule(molecule: Molecule) -> Molecule:
    """Return a fresh Molecule with the same data (never mutate the input)."""
    return Molecule(
        atoms=list(molecule.atoms),
        coordinates=[list(c) for c in molecule.coordinates],
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )


def _rdkit_relax_reason(molecule: Molecule) -> Optional[str]:
    """Why RDKit's bonded FF can't relax ``molecule``, or ``None`` if it can.

    Mirrors the perception steps in :func:`_rdkit_ff_relax` (parse →
    ``DetermineBonds`` → MMFF/UFF parameter check) **without minimizing**. A
    transition metal makes ``DetermineBonds`` raise ("Atom … has no valences
    defined"); so does a geometry too distorted for distance-based perception.
    Never raises.
    """
    if not _RDKIT_AVAILABLE:
        return "RDKit is not available"
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    xyz_block = (
        f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
        f"{molecule.to_xyz_string()}\n"
    )
    rdmol = Chem.MolFromXYZBlock(xyz_block)
    if rdmol is None:
        return "RDKit could not parse the geometry"
    try:
        rdDetermineBonds.DetermineBonds(rdmol, charge=int(molecule.charge))
    except Exception as exc:  # noqa: BLE001 — perception failure is the signal
        return f"RDKit could not perceive bonds ({type(exc).__name__})"
    if AllChem.MMFFHasAllMoleculeParams(rdmol) or AllChem.UFFHasAllMoleculeParams(
        rdmol
    ):
        return None
    return "no MMFF or UFF force-field parameters cover these elements"


def preopt_engine_label(molecule: Molecule) -> str:
    """Which pre-opt engine will run on ``molecule``: the RDKit FF, GFN-FF, or none.

    Deterministic and side-effect-free, so a caller (the preview) can label the
    result without changing the relaxation's return signature. Returns
    ``"MMFF94/UFF"`` (RDKit, organics), ``"GFN-FF"`` (xtb, metals / anything
    RDKit can't), or ``""`` when no backend can handle it.
    """
    if _RDKIT_AVAILABLE and _rdkit_relax_reason(molecule) is None:
        return "MMFF94/UFF"
    if _XTB_AVAILABLE:
        return "GFN-FF"
    return ""


def preopt_support(molecule: Molecule) -> Optional[str]:
    """Why no pre-opt backend can handle ``molecule``, or ``None`` if one can.

    Lets a caller tell a genuine *"already optimal, nothing moved"* no-op apart
    from *"nothing can pre-optimize this"* (M-METAL MET.4). RDKit's organic FF
    handles organics; the optional GFN-FF (xtb) backend handles transition-metal
    complexes RDKit's valence model rejects — so a metal is *supported* whenever
    xtb is installed. When neither applies, the reason names the fix (install
    xtb, or run the DFT geometry optimization). Never raises.
    """
    rdkit_reason = _rdkit_relax_reason(molecule)
    if rdkit_reason is None:
        return None
    if _XTB_AVAILABLE:
        return None  # GFN-FF (xtb) covers the whole periodic table
    return (
        f"{rdkit_reason}, and the GFN-FF (xtb) metal backend is not installed. "
        "Install xtb for metal-capable pre-optimization, or skip the classical "
        "pre-opt and run the DFT geometry optimization"
    )


@contextlib.contextmanager
def _contained_xtb_run() -> Iterator[None]:
    """Run a GFN-FF call with its side effects contained.

    libxtb drops ``gfnff_topo`` / ``gfnff_adjacency`` scratch files in the
    current directory and prints an initialization banner from Fortran — below
    Python's ``sys.stdout``, so only a file-descriptor redirect silences it.
    This chdir's into a temp directory and points fds 1/2 at ``os.devnull`` for
    the duration, restoring both afterward. Serialised by ``_XTB_LOCK`` because
    cwd and the fds are process-global.
    """
    import os
    import sys
    import tempfile

    with _XTB_LOCK:
        prev_cwd = os.getcwd()
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        with tempfile.TemporaryDirectory(prefix="quantui_gfnff_") as scratch:
            try:
                os.chdir(scratch)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                yield
            finally:
                # libxtb prints from Fortran and block-buffers at the C level, so
                # flush all C stdio streams *while* the fds still point at
                # devnull — otherwise the banner flushes to the real terminal
                # right after we restore them.
                try:
                    import ctypes

                    ctypes.CDLL(None).fflush(None)
                except Exception:  # noqa: BLE001 — best-effort silencing
                    pass
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(devnull)
                os.close(saved_out)
                os.close(saved_err)
                os.chdir(prev_cwd)


def _xtb_gfnff_relax(
    molecule: Molecule, steps: int, *, capture_frames: bool = False
) -> Tuple[List[List[float]], str, Optional[List[List[List[float]]]]]:
    """Relax ``molecule`` with GFN-FF (xtb) via an ASE L-BFGS optimizer.

    The metal-capable counterpart to :func:`_rdkit_ff_relax`. Returns
    ``(final_coords, "GFN-FF", frames)`` — ``frames`` is ``None`` unless
    ``capture_frames`` is set, in which case ASE's per-step trajectory is
    captured in a single optimization (no fresh restarts) and thinned to even
    RMSD spacing for the preview animation. Raises on any failure so the caller
    can fall back to the non-destructive no-op.
    """
    import numpy as np
    from ase import Atoms
    from ase.optimize import LBFGS
    from xtb.ase.calculator import XTB

    atoms = Atoms(
        symbols=list(molecule.atoms),
        positions=np.asarray(molecule.coordinates, dtype=float),
    )
    waypoints: List[List[List[float]]] = []
    with _contained_xtb_run():
        atoms.calc = XTB(method="GFN-FF", charge=float(molecule.charge), verbosity=0)
        if capture_frames:
            waypoints.append(atoms.get_positions().tolist())
        opt = LBFGS(atoms, logfile=None)
        if capture_frames:
            opt.attach(lambda: waypoints.append(atoms.get_positions().tolist()))
        opt.run(fmax=_XTB_FMAX, steps=int(steps))
        final = atoms.get_positions()

    coords = [[float(x), float(y), float(z)] for x, y, z in final]
    if len(coords) != len(molecule.atoms):
        raise ValueError("atom count changed during GFN-FF relaxation")
    frames = None
    if capture_frames:
        # The last attached waypoint is the final geometry, so playback ends at
        # exactly the geometry "Keep" would adopt.
        frames = _select_even_rmsd_frames(waypoints, _PREVIEW_FRAMES)
    return coords, "GFN-FF", frames


def _relax_best(
    molecule: Molecule, steps: int, *, capture_frames: bool = False
) -> Tuple[List[List[float]], str, Optional[List[List[List[float]]]]]:
    """Relax with the best available backend, metals included.

    RDKit MMFF/UFF for organics (fast, proven); GFN-FF (xtb) for anything RDKit
    can't perceive (transition-metal complexes). When neither applies, the RDKit
    path runs so its specific failure propagates and the caller no-ops exactly as
    before. Raises only if no backend exists at all.
    """
    if _RDKIT_AVAILABLE and _rdkit_relax_reason(molecule) is None:
        return _rdkit_ff_relax(molecule, steps, capture_frames=capture_frames)
    if _XTB_AVAILABLE:
        return _xtb_gfnff_relax(molecule, steps, capture_frames=capture_frames)
    if _RDKIT_AVAILABLE:
        # No metal backend — let RDKit raise its real reason (caller → no-op).
        return _rdkit_ff_relax(molecule, steps, capture_frames=capture_frames)
    raise RuntimeError("no pre-optimization backend available")


# Interactive-preview animation tuning (preoptimize_with_trajectory). The
# trajectory is captured as fresh minimizations from the input at increasing
# iteration budgets (see _rdkit_ff_relax). _PREVIEW_FRAMES is how many are shown
# (selected at even RMSD spacing); _PREVIEW_TIME_BUDGET_S is a wall-clock safety
# valve so a large molecule can't stall the preview thread building waypoints.
_PREVIEW_FRAMES = 20
_PREVIEW_TIME_BUDGET_S = 6.0


def _preview_iter_grid(steps: int) -> List[int]:
    """Iteration budgets to snapshot for the preview animation.

    Fine early, coarser later: small stiff molecules (e.g. water) relax within a
    handful of iterations, while large molecules' BFGS barely moves for the first
    iterations then accelerates over tens-to-hundreds. A single fixed spacing
    serves one regime and misses the other (a coarse step skips a tiny molecule's
    whole relaxation; a fine step is wasteful for a large one). This grid samples
    the active region for both without an excessive number of fresh minimizations
    (budgets past convergence are nearly free — RDKit's Minimize returns early).
    """
    grid: List[int] = []
    k = 0
    while k < steps:
        grid.append(k)
        if k < 16:
            k += 1
        elif k < 64:
            k += 4
        else:
            k += 8
    return grid


def _select_even_rmsd_frames(
    waypoints: List[List[List[float]]], n_frames: int
) -> List[List[List[float]]]:
    """Pick ~``n_frames`` waypoints spaced at even RMSD from the final geometry.

    ``waypoints`` is an ordered list of geometries (input first, relaxed last).
    Returns a sublist (input first, relaxed last) chosen so consecutive frames
    are roughly equidistant in RMSD. Without this the animation looks weighted
    to wherever the optimizer took its largest steps (RDKit's BFGS barely moves
    for the first iterations, then accelerates), playing back as a long static
    stretch followed by a rush.
    """
    import numpy as np

    if len(waypoints) <= 2:
        return list(waypoints)
    final = np.asarray(waypoints[-1], dtype=float)
    to_final = [
        float(
            np.sqrt(np.mean(np.sum((np.asarray(w, dtype=float) - final) ** 2, axis=1)))
        )
        for w in waypoints
    ]
    total = to_final[0]
    if total < 1e-3:
        return [waypoints[-1]]  # no meaningful motion → single static frame
    targets = np.linspace(total, 0.0, max(2, n_frames))
    chosen: List[int] = []
    j = 0
    for t in targets:
        # to_final decreases as the molecule relaxes; advance to the first
        # waypoint at or below this RMSD target.
        while j < len(waypoints) - 1 and to_final[j] > t:
            j += 1
        if not chosen or chosen[-1] != j:
            chosen.append(j)
    return [waypoints[i] for i in chosen]


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

    # Frame-capturing path. RDKit exposes no per-iteration callback, and calling
    # Minimize(maxIts=1) repeatedly *restarts* its BFGS optimizer each call (the
    # inverse-Hessian estimate resets to the identity), so single-step snapshots
    # barely move while one bulk minimize does ~all the work — an animation that
    # looks static then snaps on the last frame. Instead, snapshot a set of
    # fresh minimizations from the input at increasing iteration budgets. BFGS is
    # deterministic, so minimizing for k iterations is a true waypoint on the
    # path to minimizing for 2k, and the budget==steps point is identical to the
    # silent preoptimize() result (so Preview and a silent run agree). Frames are
    # then selected at even RMSD spacing for a smooth playback.
    import time as _time

    def _relax_to(max_its: int) -> List[List[float]]:
        rd = Chem.Mol(rdmol)  # fresh copy at the input geometry
        cf = rd.GetConformer()
        if ff_name == "MMFF94":
            ff = AllChem.MMFFGetMoleculeForceField(
                rd, AllChem.MMFFGetMoleculeProperties(rd)
            )
        else:
            ff = AllChem.UFFGetMoleculeForceField(rd)
        if ff is None:
            raise ValueError("could not build force field for frame capture")
        ff.Initialize()
        if max_its > 0:
            ff.Minimize(maxIts=max_its)
        return _conf_coords(cf, n)

    final_coords = _relax_to(int(steps))  # == silent preoptimize() geometry
    if len(final_coords) != len(molecule.atoms):
        raise ValueError("atom count changed during FF relaxation")

    # Waypoints at increasing iteration budgets (fresh from input each time;
    # budgets past convergence cost almost nothing as Minimize returns early).
    waypoints: List[List[List[float]]] = []
    t0 = _time.monotonic()
    for k in _preview_iter_grid(int(steps)):
        waypoints.append(_relax_to(k))
        if _time.monotonic() - t0 > _PREVIEW_TIME_BUDGET_S:
            break
    waypoints.append(final_coords)

    frames = _select_even_rmsd_frames(waypoints, _PREVIEW_FRAMES)
    return final_coords, ff_name, frames


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

    if not (_RDKIT_AVAILABLE or _XTB_AVAILABLE):
        logger.warning(
            "No pre-opt backend (RDKit / xtb) available — geometry unchanged."
        )
        return _copy_molecule(molecule), 0.0

    original = np.asarray(molecule.coordinates, dtype=float)
    try:
        coords, ff_name, _frames = _relax_best(molecule, steps)
    except Exception as exc:  # noqa: BLE001 — any FF failure → non-destructive no-op
        logger.warning(
            "Pre-opt failed (%s); returning original geometry unchanged.",
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
    relaxation in the interactive "Preview pre-optimization" flow. Same
    non-destructive guarantee: on any failure the original
    geometry is returned unchanged with ``rmsd = 0.0`` and a single-frame
    trajectory (just the input), so the viewer always has something to show.
    """
    import numpy as np

    original = np.asarray(molecule.coordinates, dtype=float)
    fallback_frames = [original.tolist()]

    if not (_RDKIT_AVAILABLE or _XTB_AVAILABLE):
        logger.warning(
            "No pre-opt backend (RDKit / xtb) available — geometry unchanged."
        )
        return _copy_molecule(molecule), 0.0, fallback_frames

    try:
        coords, ff_name, frames = _relax_best(molecule, steps, capture_frames=True)
    except Exception as exc:  # noqa: BLE001 — any FF failure → non-destructive no-op
        logger.warning("Pre-opt preview failed (%s); geometry unchanged.", exc)
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
