"""
results_storage — Persist and reload QuantUI calculation results.

Each calculation is saved to a timestamped subdirectory::

    <results_dir>/<timestamp>_<formula>_<method>_<basis>/
        result.json   — structured metadata + energy values (versioned)
        pyscf.log     — raw PySCF stdout (may be absent for short runs)

The ``result.json`` schema carries a ``_schema_version`` field so future
fields (geometry, IR/UV-Vis spectra file paths, etc.) can be added without
breaking existing readers.  A ``"spectra"`` key is reserved now as an empty
dict to make the intended extension point obvious.

Results directory
-----------------
Defaults to ``Path("results")`` relative to the working directory, or to
the value of the ``QUANTUI_RESULTS_DIR`` environment variable if set.
The Apptainer container sets this to ``$HOME/.quantui/results`` so that
results survive across kernel restarts and land in the user's home
directory (which is bind-mounted and writable).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass  # result types accepted via duck typing; no hard import needed

_SCHEMA_VERSION = 2


def _default_results_dir() -> Path:
    env = os.environ.get("QUANTUI_RESULTS_DIR")
    return Path(env) if env else Path("results")


def _safe_name(s: str) -> str:
    """Replace characters that are unsafe in directory names with 'x'."""
    return re.sub(r"[^\w\-]", "x", s)


def save_result(
    result: object,
    pyscf_log: str = "",
    results_dir: Optional[Path] = None,
    calc_type: str = "single_point",
    spectra: Optional[dict] = None,
) -> Path:
    """Write *result* to a new timestamped subdirectory of *results_dir*.

    Accepts any result type that exposes ``.formula``, ``.method``,
    ``.basis``, ``.energy_hartree``, and ``.converged`` attributes
    (``SessionResult``, ``OptimizationResult``, ``FreqResult``,
    ``TDDFTResult``).  Missing optional fields (``homo_lumo_gap_ev``,
    ``n_iterations``) are stored as ``null``.

    Parameters
    ----------
    result:
        Any completed calculation result object.
    pyscf_log:
        Raw PySCF stdout captured during the run.  Written to
        ``pyscf.log`` inside the result directory when non-empty.
    results_dir:
        Override the default results directory.
    calc_type:
        Calculation type string stored in ``result.json`` for display
        in the History browser.  One of ``"single_point"``,
        ``"geometry_opt"``, ``"frequency"``, ``"tddft"``.
    spectra:
        Dict of spectra data (IR frequencies, UV-Vis excitations, …)
        stored under the ``"spectra"`` key in ``result.json``.

    Returns
    -------
    Path
        The directory that was created.
    """
    _HARTREE_TO_EV = 27.211386245988  # local fallback

    base = results_dir if results_dir is not None else _default_results_dir()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    dirname = "_".join(
        [
            ts,
            _safe_name(getattr(result, "formula", "unknown")),
            _safe_name(getattr(result, "method", "unknown")),
            _safe_name(getattr(result, "basis", "unknown")),
        ]
    )
    dest = base / dirname
    # Windows timer resolution can produce identical microsecond timestamps for
    # back-to-back calls; append a counter to guarantee a unique directory.
    _collision = 1
    while dest.exists():
        dest = base / f"{dirname}_{_collision}"
        _collision += 1
    dest.mkdir(parents=True)

    _e_ha = getattr(result, "energy_hartree", float("nan"))
    # energy_ev may be a property (SessionResult) or absent (OptimizationResult
    # and new types also define it as a property, so getattr works for all).
    _e_ev = getattr(result, "energy_ev", _e_ha * _HARTREE_TO_EV)

    data: dict = {
        "_schema_version": _SCHEMA_VERSION,
        "timestamp": ts,
        "calc_type": calc_type,
        "formula": getattr(result, "formula", "?"),
        "method": getattr(result, "method", "?"),
        "basis": getattr(result, "basis", "?"),
        "energy_hartree": _e_ha,
        "energy_ev": _e_ev,
        "homo_lumo_gap_ev": getattr(result, "homo_lumo_gap_ev", None),
        "converged": getattr(result, "converged", None),
        "n_iterations": getattr(result, "n_iterations", -1),
        "spectra": spectra if spectra is not None else {},
    }
    (dest / "result.json").write_text(json.dumps(data, indent=2))

    if pyscf_log:
        (dest / "pyscf.log").write_text(pyscf_log)

    return dest


def list_results(results_dir: Optional[Path] = None) -> list:
    """Return result directories sorted newest-first.

    Only directories containing a ``result.json`` file are included.
    """
    base = results_dir if results_dir is not None else _default_results_dir()
    if not base.exists():
        return []
    return sorted(
        (d for d in base.iterdir() if d.is_dir() and (d / "result.json").exists()),
        reverse=True,
    )


def load_result(result_dir: Path) -> dict:
    """Return the parsed ``result.json`` from *result_dir*."""
    data: dict = json.loads((result_dir / "result.json").read_text())
    return data


def save_orbitals(result_dir: Path, result: object) -> None:
    """Persist MO data to *result_dir*/orbitals.npz and orbitals_meta.json.

    Saves ``mo_energy_hartree``, ``mo_occ``, and ``mo_coeff`` as a compressed
    NumPy archive and ``pyscf_mol_atom`` / ``pyscf_mol_basis`` as JSON so the
    orbital diagram and isosurface can be replayed from history.
    """
    import numpy as _np

    mo_e = getattr(result, "mo_energy_hartree", None)
    mo_occ = getattr(result, "mo_occ", None)
    mo_coeff = getattr(result, "mo_coeff", None)
    mol_atom = getattr(result, "pyscf_mol_atom", None)
    mol_basis = getattr(result, "pyscf_mol_basis", None)

    if mo_e is None and mo_occ is None:
        return

    arrays: dict = {}
    if mo_e is not None:
        arrays["mo_energy_hartree"] = _np.asarray(mo_e)
    if mo_occ is not None:
        arrays["mo_occ"] = _np.asarray(mo_occ)
    if mo_coeff is not None:
        arrays["mo_coeff"] = _np.asarray(mo_coeff)
    if arrays:
        _np.savez_compressed(str(result_dir / "orbitals.npz"), **arrays)

    meta: dict = {}
    if mol_atom is not None:
        # Convert list-of-tuples to JSON-safe list-of-lists.
        meta["mol_atom"] = [[sym, list(coords)] for sym, coords in mol_atom]
    if mol_basis is not None:
        meta["mol_basis"] = mol_basis
    if meta:
        (result_dir / "orbitals_meta.json").write_text(json.dumps(meta))


def save_molden(
    result_dir: Path,
    *,
    mo_energy_hartree=None,
    mo_occ=None,
    mo_coeff=None,
    pyscf_mol_atom=None,
    pyscf_mol_basis: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
    frequencies_cm1: Optional[list] = None,
    normal_modes=None,
    filename: str = "result.molden",
) -> Optional[Path]:
    """Write a Molden-format file alongside ``result.json`` (M-EXPORT / EXPORT.1+2).

    Molden is the lingua franca for orbital + vibration interop with
    Avogadro / IQmol / Jmol / Multiwfn. This helper writes whichever data
    is available — both orbitals and vibrations, just orbitals, or just
    the structure + vibrations — using the appropriate pyscf.tools.molden
    entry point.

    Behaviour:

    - ``mo_coeff`` present → ``pyscf.tools.molden.from_mo(mol, ..., mo_coeff,
      ene=mo_energy, occ=mo_occ)`` writes ``[Atoms]`` + ``[GTO]`` + ``[MO]``.
    - ``mo_coeff`` absent but vibrations present → ``pyscf.tools.molden.header``
      writes only the structure header; we append ``[FREQ]`` +
      ``[FR-COORD]`` + ``[FR-NORM-COORD]`` manually so Avogadro can animate.
    - Neither present → returns ``None`` (nothing meaningful to export).

    Best-effort: PySCF / Molden writer failures are caught and the
    function returns ``None`` rather than propagating. Callers should
    log but not fail the calc on a missing Molden file.

    Returns the path to the written file on success, ``None`` otherwise.
    """
    try:
        from pyscf import gto
        from pyscf.tools import molden as _molden
    except Exception:
        return None

    has_mo = (
        mo_coeff is not None and mo_energy_hartree is not None and mo_occ is not None
    )
    has_vib = bool(frequencies_cm1) and bool(normal_modes)
    if not (has_mo or has_vib):
        return None

    if not pyscf_mol_atom or not pyscf_mol_basis:
        return None

    try:
        mol = gto.Mole()
        mol.atom = [(str(sym), list(coords)) for sym, coords in pyscf_mol_atom]
        mol.basis = pyscf_mol_basis
        mol.charge = int(charge)
        mol.spin = max(0, int(multiplicity) - 1)
        mol.verbose = 0
        mol.build()
    except Exception:
        return None

    dest = result_dir / filename
    try:
        if has_mo:
            _molden.from_mo(
                mol,
                str(dest),
                mo_coeff,
                ene=mo_energy_hartree,
                occ=mo_occ,
            )
        else:
            # Structure-only header; vibration blocks appended below.
            with open(dest, "w", encoding="utf-8") as fh:
                _molden.header(mol, fh)
    except Exception:
        return None

    if has_vib:
        try:
            _append_molden_vibrations(
                dest,
                frequencies_cm1=frequencies_cm1,
                normal_modes=normal_modes,
                pyscf_mol_atom=pyscf_mol_atom,
            )
        except Exception:
            pass  # Best-effort: the orbital block (or header) is already written.

    return dest


def _append_molden_vibrations(
    path: Path,
    *,
    frequencies_cm1: list,
    normal_modes,
    pyscf_mol_atom,
) -> None:
    """Append Molden ``[FREQ]`` + ``[FR-COORD]`` + ``[FR-NORM-COORD]`` blocks.

    Used by :func:`save_molden` after the structure (and optionally MO)
    sections are in place. Format follows the Molden spec — Avogadro and
    IQmol both accept this layout for animated normal-mode display.

    ``frequencies_cm1`` is a flat list of N modes (length matches
    ``normal_modes``). ``normal_modes`` is a list of length-N entries,
    each a list of per-atom (x, y, z) displacement triples. The
    ``[FR-COORD]`` block repeats the equilibrium geometry from
    ``pyscf_mol_atom`` so the file is self-contained.
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n[FREQ]\n")
        for freq in frequencies_cm1:
            fh.write(f"{float(freq):.6f}\n")

        fh.write("\n[FR-COORD]\n")
        for sym, coords in pyscf_mol_atom:
            fh.write(
                f"{sym}  {float(coords[0]):.6f} {float(coords[1]):.6f} "
                f"{float(coords[2]):.6f}\n"
            )

        fh.write("\n[FR-NORM-COORD]\n")
        for i, mode in enumerate(normal_modes, start=1):
            fh.write(f"vibration   {i}\n")
            for atom_vec in mode:
                fh.write(
                    f" {float(atom_vec[0]):.6f} {float(atom_vec[1]):.6f} "
                    f"{float(atom_vec[2]):.6f}\n"
                )


def save_trajectory_xyz(
    result_dir: Path,
    *,
    frames: list,
    energies: list,
    filename: str = "trajectory.xyz",
) -> Optional[Path]:
    """Write a multi-frame XYZ trajectory file (M-EXPORT / EXPORT.3).

    Universal format readable by Avogadro, VMD, OVITO, Jmol, Pymol,
    OpenBabel, ASE (``ase.io.read``), and basically any molecular tool
    that handles XYZ. Each frame's comment line carries the energy in
    Hartree when known (parsed by tools that follow the extended-XYZ
    convention).

    Parameters
    ----------
    result_dir:
        Directory returned by :func:`save_result`.
    frames:
        List of :class:`~quantui.molecule.Molecule` objects, one per
        trajectory step.
    energies:
        Parallel list of total energies in Hartree. Missing entries are
        written as plain frame numbers in the comment line.
    filename:
        Output filename inside *result_dir*. Defaults to
        ``trajectory.xyz``.

    Returns the path on success, ``None`` if ``frames`` is empty or the
    write fails. Best-effort: failures don't propagate.
    """
    if not frames:
        return None

    out_path = result_dir / filename
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for i, mol in enumerate(frames):
                atoms = list(mol.atoms)
                coords = mol.coordinates
                fh.write(f"{len(atoms)}\n")
                # Extended-XYZ comment line: include energy when known
                # so downstream parsers (ASE, OVITO) can pick it up.
                if i < len(energies) and energies[i] is not None:
                    fh.write(f"energy={float(energies[i]):.10f} Hartree\n")
                else:
                    fh.write(f"frame {i}\n")
                for sym, xyz in zip(atoms, coords):
                    fh.write(
                        f"{sym} {float(xyz[0]):.6f} "
                        f"{float(xyz[1]):.6f} {float(xyz[2]):.6f}\n"
                    )
    except Exception:
        return None
    return out_path


def save_trajectory_ase(
    result_dir: Path,
    *,
    frames: list,
    energies: list,
    filename: str = "trajectory.traj",
) -> Optional[Path]:
    """Write an ASE Trajectory (.traj) file (M-EXPORT / EXPORT.7).

    Lets users open the result in ``ase gui trajectory.traj``, slice
    frames (``trajectory.traj@0:10:2``), and use ASE-GUI's interactive
    editing tools to modify the structure as a starting point for
    follow-up calcs. Also enables ASE-Python-side post-processing
    (custom analyses, force diagnostics, etc.). Per-frame energies are
    attached via :class:`ase.calculators.singlepoint.SinglePointCalculator`
    so ``ase gui -g "d(0,1),e-E[0]"`` can plot derived quantities.

    Parameters
    ----------
    result_dir, frames, energies:
        Same convention as :func:`save_trajectory_xyz`.
    filename:
        Output filename inside *result_dir*. Defaults to
        ``trajectory.traj``.

    Returns the path on success, ``None`` if ASE is unavailable, frames
    is empty, or the writer raises. Best-effort: failures don't
    propagate.
    """
    if not frames:
        return None
    try:
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io.trajectory import Trajectory
    except Exception:
        return None

    _HARTREE_TO_EV = 27.211386245988  # ASE uses eV for the calculator energy
    out_path = result_dir / filename
    try:
        traj = Trajectory(str(out_path), "w")
        try:
            for i, mol in enumerate(frames):
                atoms = Atoms(
                    symbols=list(mol.atoms),
                    positions=[list(row) for row in mol.coordinates],
                )
                if i < len(energies) and energies[i] is not None:
                    atoms.calc = SinglePointCalculator(
                        atoms,
                        energy=float(energies[i]) * _HARTREE_TO_EV,
                    )
                traj.write(atoms)
        finally:
            traj.close()
    except Exception:
        return None
    return out_path


def load_orbitals(result_dir: Path):
    """Reload MO data saved by :func:`save_orbitals`.

    Returns a ``SimpleNamespace`` with ``mo_energy_hartree``, ``mo_occ``,
    ``mo_coeff``, ``pyscf_mol_atom``, ``pyscf_mol_basis``, and ``formula``
    (empty string if not known).

    Raises
    ------
    FileNotFoundError
        If ``orbitals.npz`` does not exist in *result_dir*.
    """
    import types

    import numpy as _np

    npz_path = result_dir / "orbitals.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = _np.load(str(npz_path))
    stub = types.SimpleNamespace(
        mo_energy_hartree=(
            data["mo_energy_hartree"] if "mo_energy_hartree" in data else None
        ),
        mo_occ=data["mo_occ"] if "mo_occ" in data else None,
        mo_coeff=data["mo_coeff"] if "mo_coeff" in data else None,
        pyscf_mol_atom=None,
        pyscf_mol_basis=None,
        formula="",
    )
    meta_path = result_dir / "orbitals_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        stub.pyscf_mol_atom = meta.get("mol_atom")
        stub.pyscf_mol_basis = meta.get("mol_basis")
    return stub


def save_trajectory(
    result_dir: Path,
    trajectory: list,
    energies: list,
    filename: str = "trajectory.json",
) -> None:
    """Persist geometry-optimisation trajectory to *result_dir*/*filename*.

    Parameters
    ----------
    result_dir:
        Directory returned by :func:`save_result`.
    trajectory:
        List of ``Molecule`` objects (one per optimisation step).
    energies:
        List of total energies in Hartree, parallel to *trajectory*.
    filename:
        Output filename inside *result_dir*. Defaults to ``trajectory.json``.
        Pass ``preopt_trajectory.json`` for pre-optimisation steps.
    """
    if not trajectory:
        return
    mol0 = trajectory[0]
    data = {
        "atoms": list(mol0.atoms),
        "charge": mol0.charge,
        "multiplicity": mol0.multiplicity,
        "steps": [
            {
                "coords": [list(row) for row in mol.coordinates],
                "energy": energies[i] if i < len(energies) else None,
            }
            for i, mol in enumerate(trajectory)
        ],
    }
    (result_dir / filename).write_text(json.dumps(data))


def load_trajectory(result_dir: Path, filename: str = "trajectory.json"):
    """Reload a saved trajectory as (molecules, energies).

    Returns
    -------
    tuple[list, list]
        ``(trajectory, energies_hartree)`` where *trajectory* is a list of
        ``Molecule`` objects and *energies_hartree* is a parallel list of
        floats (``None`` entries are dropped to an empty list if all absent).

    Raises
    ------
    FileNotFoundError
        If ``trajectory.json`` does not exist in *result_dir*.
    """
    from quantui.molecule import Molecule

    raw = json.loads((result_dir / filename).read_text())
    atoms = raw["atoms"]
    charge = raw.get("charge", 0)
    mult = raw.get("multiplicity", 1)
    trajectory = []
    energies = []
    for step in raw["steps"]:
        trajectory.append(
            Molecule(atoms, step["coords"], charge=charge, multiplicity=mult)
        )
        energies.append(step["energy"])
    # If every energy is None the list is meaningless; return empty instead.
    if all(e is None for e in energies):
        energies = []
    return trajectory, energies


def save_thumbnail(result_dir: Path, data: dict) -> None:
    """Generate a compact PNG thumbnail card for the saved result.

    Silently skips if matplotlib is unavailable or any error occurs.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _colors: dict = {
        "single_point": ("#2563eb", "#dbeafe"),
        "geometry_opt": ("#7c3aed", "#ede9fe"),
        "frequency": ("#15803d", "#dcfce7"),
        "tddft": ("#b45309", "#fef3c7"),
        "nmr": ("#0d9488", "#ccfbf1"),
    }
    _ct_labels: dict = {
        "single_point": "Single Point",
        "geometry_opt": "Geometry Opt",
        "frequency": "Frequency",
        "tddft": "TD-DFT",
        "nmr": "NMR",
    }
    ct = data.get("calc_type", "")
    fg, bg = _colors.get(ct, ("#555555", "#f3f4f6"))
    ct_label = _ct_labels.get(ct, ct.replace("_", " ").title())

    fig = plt.figure(figsize=(2.4, 1.5), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colored header strip
    ax.axhspan(0.80, 1.0, color=fg)
    ax.text(
        0.5,
        0.90,
        ct_label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
    )

    # Formula
    ax.text(
        0.5,
        0.65,
        data.get("formula", "?"),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=fg,
        transform=ax.transAxes,
    )

    # Method / basis
    ax.text(
        0.5,
        0.50,
        f'{data.get("method", "?")} / {data.get("basis", "?")}',
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
        transform=ax.transAxes,
    )

    # Energy
    e_ha = data.get("energy_hartree")
    if e_ha is not None and e_ha == e_ha:  # skip NaN
        ax.text(
            0.5,
            0.34,
            f"E = {e_ha:.5f} Ha",
            ha="center",
            va="center",
            fontsize=7,
            color="#333333",
            transform=ax.transAxes,
            family="monospace",
        )

    # Converged indicator
    conv = data.get("converged")
    if conv is not None:
        ax.text(
            0.5,
            0.16,
            "✓ Converged" if conv else "✗ Not converged",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#15803d" if conv else "#c00000",
            transform=ax.transAxes,
        )

    try:
        fig.savefig(
            str(result_dir / "thumbnail.png"),
            dpi=72,
            bbox_inches="tight",
            facecolor=bg,
            pad_inches=0.05,
        )
    finally:
        plt.close(fig)
