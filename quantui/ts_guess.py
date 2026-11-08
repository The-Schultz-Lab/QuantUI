"""Transition-state guess from a reactant and a product (QST2-style).

The user supplies two minima, reactant and product, with the same atoms in
the same order. QuantUI

1. checks the two structures describe the same system (same element list in
   the same order, same charge and multiplicity),
2. removes overall translation and rotation by superimposing the product on
   the reactant (Kabsch, minimum RMSD over proper rotations; the atom
   ordering is used as given, never re-matched),
3. builds ``n_images`` intermediate geometries with IDPP (image-dependent
   pair potential, Smidstrup et al., J. Chem. Phys. 140, 214106 (2014)):
   the interatomic distances of each image are pulled toward a linear
   interpolation of the reactant and product distances, which avoids the
   atom clashes a straight Cartesian interpolation can produce,
4. evaluates a single-point energy at every image (endpoints included) at
   the run's method/basis, and
5. returns the highest-energy *interior* image as the transition-state
   guess, together with the energy profile.

This is "NEB-lite": the images are never relaxed on the real energy surface,
so the highest image is only an estimate of where the barrier top lies. The
saddle-point search (:func:`quantui.ts_search.run_ts_search`) then refines it,
and only the frequency check after that search says whether a transition
state was found (exactly one imaginary frequency).
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass, field
from typing import IO, Any, Callable, List, Optional, Tuple

from .molecule import Molecule

#: Interior images between reactant and product (endpoints excluded).
DEFAULT_N_IMAGES = 7
MIN_N_IMAGES = 1
MAX_N_IMAGES = 15

#: Below this aligned RMSD (Å) reactant and product are the same structure.
SAME_STRUCTURE_RMSD_ANG = 0.05

#: Two atoms of an interpolated image closer than this (Å) mean the path
#: makes atoms collide, almost always because the atom order of the two
#: structures does not correspond.
CLASH_DISTANCE_ANG = 0.5

HARTREE_TO_KCAL = 627.509474

# IDPP relaxation of the images (a purely geometric objective, so cheap).
# Tighter than ASE's default fmax=0.1, which can stop before the images'
# distances approach their targets.
_IDPP_FMAX = 0.01
_IDPP_STEPS = 500

#: Energy evaluator: geometry -> total energy in Hartree (NaN if the SCF
#: failed). Injected in tests; the default runs a silent single point.
EnergyFn = Callable[[Molecule], float]


@dataclass
class TSGuessResult:
    """Interpolated path, its energy profile and the chosen guess."""

    method: str
    basis: str
    images: List[Molecule]  # reactant, n_images interior images, product
    energies_hartree: List[float]  # one per image; NaN where the SCF failed
    guess_index: int  # index into ``images`` (always an interior image)
    path_fraction: List[float]  # 0 (reactant) … 1 (product), by path length
    alignment_rmsd_ang: float  # product vs reactant after superposition
    interpolation: str = "idpp"  # "idpp" or "linear" (IDPP fallback)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        """Interior images (endpoints excluded)."""
        return max(len(self.images) - 2, 0)

    @property
    def guess(self) -> Molecule:
        return self.images[self.guess_index]

    @property
    def guess_energy_hartree(self) -> float:
        return self.energies_hartree[self.guess_index]

    @property
    def barrier_estimate_kcal(self) -> Optional[float]:
        """E(guess) − E(reactant), kcal/mol; None if either is missing.

        A rough estimate only: the images are not relaxed on the energy
        surface, so this usually overestimates the barrier. The barrier
        that counts is E(TS) − E(reactant) from the search that follows.
        """
        e_r, e_g = self.energies_hartree[0], self.guess_energy_hartree
        if not (math.isfinite(e_r) and math.isfinite(e_g)):
            return None
        return (e_g - e_r) * HARTREE_TO_KCAL

    def to_spectra(self) -> dict:
        """The ``ts_guess`` block stored in ``result.json`` spectra."""
        return {
            "method": "idpp-highest-image",
            "interpolation": self.interpolation,
            "n_images": self.n_images,
            "energies_hartree": [
                float(e) if math.isfinite(e) else None for e in self.energies_hartree
            ],
            "path_fraction": [float(x) for x in self.path_fraction],
            "guess_index": int(self.guess_index),
            "alignment_rmsd_ang": float(self.alignment_rmsd_ang),
            "barrier_estimate_kcal": self.barrier_estimate_kcal,
            "warnings": list(self.warnings),
        }


# ── Validation and alignment ────────────────────────────────────────────────


def _norm_symbol(sym: str) -> str:
    s = str(sym).strip()
    return s[:1].upper() + s[1:].lower()


def validate_endpoints(reactant: Molecule, product: Molecule) -> None:
    """Raise ``ValueError`` unless *reactant* and *product* are interpolable.

    They must list the same elements in the same order (atom *i* of the
    reactant becomes atom *i* of the product) and have the same charge and
    spin multiplicity.
    """
    r_atoms = [_norm_symbol(a) for a in reactant.atoms]
    p_atoms = [_norm_symbol(a) for a in product.atoms]
    if len(r_atoms) != len(p_atoms):
        raise ValueError(
            f"Reactant has {len(r_atoms)} atoms but the product has "
            f"{len(p_atoms)}. Both structures must contain the same atoms."
        )
    if sorted(r_atoms) != sorted(p_atoms):
        raise ValueError(
            f"Reactant ({reactant.get_formula()}) and product "
            f"({product.get_formula()}) contain different atoms. A transition "
            "state connects two structures of the same atoms."
        )
    mismatched = [i + 1 for i, (a, b) in enumerate(zip(r_atoms, p_atoms)) if a != b]
    if mismatched:
        shown = ", ".join(str(i) for i in mismatched[:6])
        more = "…" if len(mismatched) > 6 else ""
        raise ValueError(
            "Reactant and product list their atoms in a different order "
            f"(element differs at atom {shown}{more}). Atom i of the reactant "
            "must be the same atom as atom i of the product; reorder one of "
            "the structures."
        )
    if int(reactant.charge) != int(product.charge):
        raise ValueError(
            f"Reactant charge ({reactant.charge}) and product charge "
            f"({product.charge}) differ; both must be the same."
        )
    if int(reactant.multiplicity) != int(product.multiplicity):
        raise ValueError(
            f"Reactant multiplicity ({reactant.multiplicity}) and product "
            f"multiplicity ({product.multiplicity}) differ; the path is "
            "followed on one spin surface."
        )


def align_to_reference(reference: Any, mobile: Any) -> Tuple[Any, float]:
    """Superimpose *mobile* on *reference* (both ``(N, 3)``; same atom order).

    Kabsch algorithm, unweighted, proper rotations only (a reflection would
    turn a molecule into its mirror image). Returns the moved coordinates
    and the RMSD after superposition, Å.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=float)
    mob = np.asarray(mobile, dtype=float)
    ref_c = ref.mean(axis=0)
    mob_c = mob.mean(axis=0)
    p = mob - mob_c
    q = ref - ref_c
    if len(ref) < 2:
        moved = p + ref_c
    else:
        u, _s, vt = np.linalg.svd(p.T @ q)
        d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        moved = p @ rot.T + ref_c
    rmsd = float(np.sqrt(((moved - ref) ** 2).sum(axis=1).mean()))
    return moved, rmsd


# ── Interpolation ───────────────────────────────────────────────────────────


def _idpp_interpolate_fn() -> Any:
    try:
        from ase.mep.neb import idpp_interpolate
    except ImportError:  # ASE < 3.23
        from ase.neb import idpp_interpolate  # type: ignore[no-redef]
    return idpp_interpolate


def _neb_cls() -> Any:
    try:
        from ase.mep import NEB
    except ImportError:  # ASE < 3.23
        from ase.neb import NEB  # type: ignore[no-redef]
    return NEB


def interpolate_path(
    reactant: Molecule,
    product: Molecule,
    n_images: int = DEFAULT_N_IMAGES,
    *,
    method: str = "idpp",
) -> Tuple[List[Molecule], float, str, List[str]]:
    """Reactant, *n_images* interpolated images and the aligned product.

    Returns ``(images, alignment_rmsd, method_used, warnings)``; the images
    list has ``n_images + 2`` entries; the endpoints are in the reactant's
    frame (IDPP may turn the interior images slightly, which changes no
    energy). IDPP falls back to linear interpolation (with a warning) if it
    fails.
    """
    import numpy as np

    validate_endpoints(reactant, product)
    n_images = int(n_images)
    if not MIN_N_IMAGES <= n_images <= MAX_N_IMAGES:
        raise ValueError(
            f"Number of images must be between {MIN_N_IMAGES} and "
            f"{MAX_N_IMAGES} (got {n_images})."
        )
    if method not in ("idpp", "linear"):
        raise ValueError(f"Unknown interpolation method {method!r}.")

    from ase import Atoms

    r_xyz = np.asarray(reactant.coordinates, dtype=float)
    p_xyz, rmsd = align_to_reference(r_xyz, product.coordinates)
    if rmsd < SAME_STRUCTURE_RMSD_ANG:
        raise ValueError(
            "Reactant and product are the same structure (RMSD "
            f"{rmsd:.3f} Å after superposition). Supply two different minima."
        )

    symbols = [_norm_symbol(a) for a in reactant.atoms]
    images = [Atoms(symbols, positions=r_xyz)]
    images += [Atoms(symbols, positions=r_xyz) for _ in range(n_images)]
    images.append(Atoms(symbols, positions=p_xyz))
    # The tangent method only matters for IDPP's projected forces; named so
    # ASE versions with a different default behave the same.
    neb = _neb_cls()(images, method="improvedtangent")
    neb.interpolate()  # linear first; IDPP starts from it
    warnings: List[str] = []
    used = "linear"
    if method == "idpp":
        try:
            # traj/log None: ASE's defaults write idpp.traj / idpp.log into
            # the working directory.
            _idpp_interpolate_fn()(
                neb, traj=None, log=None, fmax=_IDPP_FMAX, steps=_IDPP_STEPS
            )
            used = "idpp"
        except Exception as exc:  # noqa: BLE001 — linear is a usable fallback
            neb.interpolate()
            warnings.append(
                f"IDPP interpolation failed ({type(exc).__name__}); "
                "used a straight-line interpolation instead."
            )

    mols = [
        Molecule(
            list(reactant.atoms),
            img.get_positions().tolist(),
            charge=reactant.charge,
            multiplicity=reactant.multiplicity,
            validate_spin=False,
        )
        for img in images
    ]
    clash = _closest_contact(mols[1:-1])
    if clash is not None and clash[0] < CLASH_DISTANCE_ANG:
        d, k, i, j = clash
        warnings.append(
            f"Atoms {i + 1} ({symbols[i]}) and {j + 1} ({symbols[j]}) come "
            f"within {d:.2f} Å of each other in image {k}. The path passes "
            "atoms through each other, which usually means the two structures "
            "do not list their atoms in corresponding order."
        )
    return mols, rmsd, used, warnings


def _closest_contact(images: List[Molecule]) -> Optional[Tuple[float, int, int, int]]:
    """``(distance, image_number, i, j)`` of the closest atom pair, or None.

    ``image_number`` counts from 1 (the first interior image).
    """
    import numpy as np

    best: Optional[Tuple[float, int, int, int]] = None
    for k, mol in enumerate(images, start=1):
        xyz = np.asarray(mol.coordinates, dtype=float)
        if len(xyz) < 2:
            continue
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        iu = np.triu_indices(len(xyz), k=1)
        m = int(np.argmin(dist[iu]))
        d = float(dist[iu][m])
        if best is None or d < best[0]:
            best = (d, k, int(iu[0][m]), int(iu[1][m]))
    return best


def path_fractions(images: List[Molecule]) -> List[float]:
    """Cumulative Cartesian path length, scaled to run from 0 to 1."""
    import numpy as np

    xyz = [np.asarray(m.coordinates, dtype=float) for m in images]
    steps = [float(np.linalg.norm(b - a)) for a, b in zip(xyz, xyz[1:])]
    total = sum(steps)
    out = [0.0]
    for s in steps:
        out.append(out[-1] + s)
    if total <= 0:
        n = max(len(images) - 1, 1)
        return [i / n for i in range(len(images))]
    return [x / total for x in out]


def select_guess(energies_hartree: List[float]) -> int:
    """Index of the highest-energy interior image (endpoints never chosen).

    Images whose energy is missing (NaN) are skipped. Raises ``ValueError``
    when no interior image has an energy.
    """
    interior = [
        (i, e)
        for i, e in enumerate(energies_hartree)
        if 0 < i < len(energies_hartree) - 1 and e is not None and math.isfinite(e)
    ]
    if not interior:
        raise ValueError(
            "No interpolated image has an energy (every single point failed), "
            "so there is no transition-state guess."
        )
    return max(interior, key=lambda pair: pair[1])[0]


# ── Energies and the full guess ─────────────────────────────────────────────


def _default_energy_fn(
    method: str, basis: str, solvent: Optional[str], scf_rescue: bool
) -> EnergyFn:
    """Silent single point on the normal in-session engine path."""
    from .session_calc import run_in_session

    def _energy(mol: Molecule) -> float:
        res = run_in_session(
            mol,
            method,
            basis,
            verbose=0,
            progress_stream=io.StringIO(),
            solvent=solvent,
            warm_start=False,
            scf_rescue=scf_rescue,
        )
        if not bool(getattr(res, "converged", False)):
            return float("nan")
        return float(res.energy_hartree)

    return _energy


def build_ts_guess(
    reactant: Molecule,
    product: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    *,
    n_images: int = DEFAULT_N_IMAGES,
    interpolation: str = "idpp",
    solvent: Optional[str] = None,
    scf_rescue: bool = True,
    progress_stream: Optional[IO[str]] = None,
    cancel_check: Any = None,
    energy_fn: Optional[EnergyFn] = None,
) -> TSGuessResult:
    """Interpolate reactant → product and pick the highest-energy image.

    Args:
        reactant, product: Minima with the same atoms in the same order and
            the same charge and multiplicity. The guess keeps the
            reactant's orientation; the product is superimposed on it.
        method, basis: Level of theory for the image energies (use the same
            as the transition-state search that follows).
        n_images: Interior images (endpoints excluded); each costs one
            single point, plus two for the endpoints.
        interpolation: ``"idpp"`` (default) or ``"linear"``.
        solvent: PCM solvent for the single points (None = gas phase).
        energy_fn: Geometry → energy (Hartree, NaN on failure). Defaults
            to a silent :func:`~quantui.session_calc.run_in_session` single
            point.
    """
    from .cancellation import raise_if_cancelled
    from .session_calc import resolve_solvent

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    images, rmsd, used, warnings = interpolate_path(
        reactant, product, n_images, method=interpolation
    )
    solvent = resolve_solvent(solvent)
    if energy_fn is None:
        energy_fn = _default_energy_fn(method, basis, solvent, scf_rescue)

    n_int = len(images) - 2
    stream.write(
        "\n── Transition-state guess from reactant + product "
        f"({'IDPP' if used == 'idpp' else 'linear'} interpolation) ──\n"
        f"  {reactant.get_formula()}  {method}/{basis}"
        + (f"  PCM: {solvent}" if solvent else "")
        + f"\n  {n_int} image(s) between reactant and product; product "
        f"superimposed on the reactant (RMSD {rmsd:.3f} Å).\n"
        f"  Single points at {len(images)} geometries…\n"
    )
    energies: List[float] = []
    for k, mol in enumerate(images):
        raise_if_cancelled(cancel_check)
        try:
            e = float(energy_fn(mol))
        except Exception as exc:  # noqa: BLE001 — one failed image is not fatal
            stream.write(f"  image {k}: single point failed ({exc})\n")
            e = float("nan")
        energies.append(e)
    failed = [k for k, e in enumerate(energies) if not math.isfinite(e)]
    if failed:
        warnings.append(
            "The SCF failed at image(s) "
            + ", ".join(str(k) for k in failed)
            + "; those images were skipped."
        )

    guess_index = select_guess(energies)
    frac = path_fractions(images)
    finite = [(k, e) for k, e in enumerate(energies) if math.isfinite(e)]
    top_k = max(finite, key=lambda pair: pair[1])[0]
    if top_k in (0, len(images) - 1):
        side = "reactant" if top_k == 0 else "product"
        warnings.append(
            f"The {side} is the highest point of the interpolated path, so the "
            "path has no barrier top between the endpoints. The guess (the "
            "highest interior image) is probably poor: check that both "
            "structures are optimized minima, or the reaction may have no "
            "barrier at this level of theory."
        )

    e_ref = energies[0] if math.isfinite(energies[0]) else min(e for _, e in finite)
    ref_label = "reactant" if math.isfinite(energies[0]) else "lowest image"
    stream.write(
        f"\n  image   path    E / Ha             ΔE vs {ref_label} / kcal/mol\n"
    )
    for k, (e, x) in enumerate(zip(energies, frac)):
        tag = " (R)" if k == 0 else (" (P)" if k == len(images) - 1 else "    ")
        if math.isfinite(e):
            row = f"{e:16.8f}   {(e - e_ref) * HARTREE_TO_KCAL:+9.2f}"
        else:
            row = f"{'failed':>16}"
        mark = "   ← highest image: TS guess" if k == guess_index else ""
        stream.write(f"  {k:3d}{tag} {x:5.2f}  {row}{mark}\n")
    for w in warnings:
        stream.write(f"  ⚠ {w}\n")
    stream.write(f"  Starting the transition-state search from image {guess_index}.\n")

    return TSGuessResult(
        method=method,
        basis=basis,
        images=images,
        energies_hartree=energies,
        guess_index=guess_index,
        path_fraction=frac,
        alignment_rmsd_ang=rmsd,
        interpolation=used,
        warnings=warnings,
    )


def check_search_possible(method: str) -> None:
    """Raise now if the search after the guess could not run.

    Called before the image single points, so a missing Sella or a method
    without analytic gradients fails at once instead of after them.
    """
    from . import config as _config
    from .ts_search import SELLA_INSTALL_HINT, sella_available

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' has no analytic gradients here, so it cannot drive a "
            "transition-state search. Use RHF/UHF or a DFT functional."
        )
    if not sella_available():
        raise ImportError(SELLA_INSTALL_HINT)


def run_ts_search_from_endpoints(
    reactant: Molecule,
    product: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    *,
    n_images: int = DEFAULT_N_IMAGES,
    interpolation: str = "idpp",
    solvent: Optional[str] = None,
    scf_rescue: bool = True,
    progress_stream: Optional[IO[str]] = None,
    cancel_check: Any = None,
    energy_fn: Optional[EnergyFn] = None,
    **ts_kwargs: Any,
) -> Any:
    """Build the interpolated guess, then run the transition-state search.

    Returns the :class:`~quantui.ts_search.TSResult` with the guess attached
    as ``result.guess`` (a :class:`TSGuessResult`). Extra keyword arguments
    go to :func:`~quantui.ts_search.run_ts_search` (``fmax``, ``steps``, …).
    """
    from .ts_search import run_ts_search

    check_search_possible(method)
    guess = build_ts_guess(
        reactant,
        product,
        method,
        basis,
        n_images=n_images,
        interpolation=interpolation,
        solvent=solvent,
        scf_rescue=scf_rescue,
        progress_stream=progress_stream,
        cancel_check=cancel_check,
        energy_fn=energy_fn,
    )
    result = run_ts_search(
        guess.guess,
        method,
        basis,
        progress_stream=progress_stream,
        solvent=solvent,
        scf_rescue=scf_rescue,
        cancel_check=cancel_check,
        **ts_kwargs,
    )
    result.guess = guess
    return result
