"""
1D Potential Energy Surface (PES) scan using constrained QM optimizations.

Drives a single internal coordinate (bond length, bond angle, or dihedral
angle) through a range of values.  At each scan point all other degrees of
freedom are relaxed via a constrained geometry optimization (BFGS + ASE
FixInternals).  The resulting energy profile and set of geometries can be
plotted and animated in the notebook.

Platform notes
--------------
Requires PySCF and ASE — Linux / macOS / WSL only.

Educational value
-----------------
* H–H bond-stretch curve illustrates dissociation and the bond-strength concept.
* H–O–H angle bending shows the shallow vs. steep sides of the energy well.
* Ethane C–C dihedral scan reveals the staggered / eclipsed energy difference.
* All three examples connect directly to thermochemistry and reaction barriers.
"""

from __future__ import annotations

import io
import logging
import math
import sys
from dataclasses import dataclass
from typing import IO, List, Optional

from .ase_bridge import ASE_AVAILABLE, atoms_to_molecule, molecule_to_atoms
from .molecule import Molecule
from .optimizer import _QuantUIPySCFCalc
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

_HARTREE_TO_KCAL: float = 627.509474  # 1 Ha = 627.509474 kcal/mol


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class PESScanResult:
    """Structured output from a completed 1D PES scan.

    Attributes:
        formula: Hill-notation molecular formula of the input molecule.
        method: SCF method used (e.g. ``'RHF'``).
        basis: Basis set used (e.g. ``'STO-3G'``).
        scan_type: One of ``'bond'``, ``'angle'``, ``'dihedral'``.
        atom_indices: 0-based atom indices defining the scanned coordinate.
            Length 2 for bond, 3 for angle, 4 for dihedral.
        scan_parameter_values: Coordinate value at each scan point.
            Angstroms for bond scans; degrees for angle / dihedral scans.
        energies_hartree: SCF energy in Hartrees at each scan point.
            Same length as ``scan_parameter_values``.
        coordinates_list: Geometry (as :class:`~quantui.molecule.Molecule`)
            at each scan point after constrained relaxation.
        converged_all: ``True`` if every constrained geometry optimization
            converged within the force threshold.
    """

    formula: str
    method: str
    basis: str
    scan_type: str
    atom_indices: List[int]
    scan_parameter_values: List[float]
    energies_hartree: List[float]
    coordinates_list: List[Molecule]
    converged_all: bool

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def energy_hartree(self) -> float:
        """Minimum SCF energy across all scan points (Hartrees)."""
        return min(self.energies_hartree) if self.energies_hartree else float("nan")

    @property
    def energy_ev(self) -> float:
        """Minimum SCF energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    @property
    def converged(self) -> bool:
        """``True`` if all constrained optimizations converged."""
        return self.converged_all

    @property
    def n_steps(self) -> int:
        """Number of scan points completed."""
        return len(self.scan_parameter_values)

    @property
    def energies_relative_kcal(self) -> List[float]:
        """Energy relative to the lowest scan point, in kcal/mol."""
        if not self.energies_hartree:
            return []
        e_min = min(self.energies_hartree)
        return [(e - e_min) * _HARTREE_TO_KCAL for e in self.energies_hartree]

    @property
    def scan_unit(self) -> str:
        """Unit label for the scan parameter axis."""
        return "Å" if self.scan_type == "bond" else "°"

    @property
    def scan_coordinate_label(self) -> str:
        """Axis label for the scanned coordinate (1-based atom numbers)."""
        idx = [i + 1 for i in self.atom_indices]
        if self.scan_type == "bond":
            return f"Bond {idx[0]}–{idx[1]} / Å"
        if self.scan_type == "angle":
            return f"Angle {idx[0]}–{idx[1]}–{idx[2]} / °"
        return f"Dihedral {idx[0]}–{idx[1]}–{idx[2]}–{idx[3]} / °"

    def summary(self) -> str:
        """Return a multi-line human-readable result summary."""
        if not self.energies_hartree:
            return "No scan points computed."
        e_min = min(self.energies_hartree)
        e_max = max(self.energies_hartree)
        barrier = (e_max - e_min) * _HARTREE_TO_KCAL
        min_idx = self.energies_hartree.index(e_min)
        lines = [
            "=" * 60,
            "PES Scan Results",
            "=" * 60,
            f"  Molecule       : {self.formula}",
            f"  Method/Basis   : {self.method}/{self.basis}",
            f"  Scan type      : {self.scan_type}",
            f"  Scan range     : {self.scan_parameter_values[0]:.3f}"
            f" → {self.scan_parameter_values[-1]:.3f} {self.scan_unit}",
            f"  Scan points    : {self.n_steps}",
            f"  Min energy     : {e_min:.8f} Ha  (point {min_idx + 1})",
            f"  Barrier height : {barrier:.2f} kcal/mol",
            f"  All converged  : {'Yes' if self.converged_all else 'No'}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# Main function
# ============================================================================


def run_pes_scan(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    scan_type: str = "bond",
    atom_indices: List[int] = (0, 1),  # type: ignore[assignment]
    start: float = 0.5,
    stop: float = 2.0,
    steps: int = 10,
    fmax: float = 0.05,
    max_opt_steps: int = 100,
    progress_stream: Optional[IO[str]] = None,
) -> PESScanResult:
    """Run a 1D PES scan along an internal coordinate.

    At each scan point the target coordinate is set, a constraint is added to
    hold it there, and a BFGS geometry optimization relaxes all remaining
    degrees of freedom.  The geometry and energy from each constrained
    optimization form the potential energy profile.

    Args:
        molecule: Starting geometry.
        method: SCF method — ``'RHF'``, ``'UHF'``, or a DFT functional.
        basis: Basis set (``'STO-3G'``, ``'6-31G*'``, …).
        scan_type: ``'bond'``, ``'angle'``, or ``'dihedral'``.
        atom_indices: 0-based atom indices defining the coordinate.
            Exactly 2 for bond, 3 for angle, 4 for dihedral.
        start: Starting value of the scanned coordinate
            (Å for bond; degrees for angle/dihedral).
        stop: Ending value.
        steps: Number of evenly spaced scan points (including start and stop).
        fmax: Force convergence threshold (eV/Å) for each constrained optimization.
        max_opt_steps: Maximum BFGS steps per scan point.
        progress_stream: Optional writable stream for per-step progress messages.

    Returns:
        :class:`PESScanResult` with the full energy profile and geometries.

    Raises:
        ImportError: If ASE or PySCF is not installed.
        ValueError: If ``atom_indices`` has the wrong length for ``scan_type``,
            or if any index is out of range for the molecule.
        RuntimeError: If the scan fails unexpectedly.
    """

    # --- Dependency checks ---
    if not ASE_AVAILABLE or _QuantUIPySCFCalc is None:
        raise ImportError(
            "ASE is not installed — cannot run PES scan.\n"
            "  pip install 'ase>=3.22.0'"
        )
    try:
        import pyscf as _pyscf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run PES scan.\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    try:
        import contextlib

        from ase.constraints import FixInternals
        from ase.optimize import BFGS
    except ImportError as exc:
        raise ImportError("ase.optimize.BFGS is not available.") from exc

    # --- Validate atom indices ---
    _expected = {"bond": 2, "angle": 3, "dihedral": 4}
    if scan_type not in _expected:
        raise ValueError(
            f"scan_type must be 'bond', 'angle', or 'dihedral', got {scan_type!r}"
        )
    n_required = _expected[scan_type]
    atom_indices = list(atom_indices)
    if len(atom_indices) != n_required:
        raise ValueError(
            f"scan_type={scan_type!r} requires {n_required} atom indices, "
            f"got {len(atom_indices)}"
        )
    n_atoms = len(molecule.atoms)
    for idx in atom_indices:
        if not (0 <= idx < n_atoms):
            raise ValueError(
                f"Atom index {idx} is out of range for molecule with {n_atoms} atoms."
            )
    if len(set(atom_indices)) != len(atom_indices):
        raise ValueError("Atom indices must be unique.")

    if steps < 2:
        raise ValueError("steps must be >= 2.")

    # --- Set up ASE atoms + PySCF calculator ---
    atoms = molecule_to_atoms(molecule)
    atoms.calc = _QuantUIPySCFCalc(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
    )

    _stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    _null = io.StringIO()

    import numpy as np

    scan_values = np.linspace(start, stop, steps).tolist()

    energies_hartree: List[float] = []
    coordinates_list: List[Molecule] = []
    converged_all = True

    i1, i2 = atom_indices[0], atom_indices[1]
    i3 = atom_indices[2] if len(atom_indices) >= 3 else 0
    i4 = atom_indices[3] if len(atom_indices) >= 4 else 0

    for step_num, val in enumerate(scan_values, start=1):
        _stream.write(
            f"\nScan point {step_num}/{steps}: "
            f"{scan_type} = {val:.4f} {('Å' if scan_type == 'bond' else '°')}\n"
        )

        try:
            # Drive the coordinate to the target value
            if scan_type == "bond":
                atoms.set_distance(i1, i2, val, fix=0.5)

            # Diatomic bond scans have zero relaxable DOF — FixInternals
            # has an off-by-one on 2-atom systems, so skip BFGS entirely.
            _diatomic_bond = scan_type == "bond" and n_atoms <= 2

            if _diatomic_bond:
                ok = True
            else:
                if scan_type == "bond":
                    constraint = FixInternals(bonds=[(val, [i1, i2])])
                elif scan_type == "angle":
                    atoms.set_angle(i1, i2, i3, val)
                    constraint = FixInternals(
                        angles=[(math.radians(val), [i1, i2, i3])]
                    )
                else:  # dihedral
                    atoms.set_dihedral(i1, i2, i3, i4, val)
                    constraint = FixInternals(
                        dihedrals=[(math.radians(val), [i1, i2, i3, i4])]
                    )

                atoms.set_constraint(constraint)

                dyn = BFGS(atoms, logfile=_stream)
                # M-STDERR / STDERR.1: capture fd-2 stderr from PySCF C
                # extensions for the duration of this scan-point optimisation.
                from quantui.c_stderr import capture_c_stderr

                with (
                    capture_c_stderr(_stream),
                    contextlib.redirect_stdout(_null),
                ):
                    ok = bool(dyn.run(fmax=fmax, steps=max_opt_steps))

            converged_all = converged_all and ok

            # Record energy (convert eV → Hartree) and geometry
            e_ev = atoms.get_potential_energy()
            e_ha = e_ev / HARTREE_TO_EV
            energies_hartree.append(e_ha)

            mol_at_point = atoms_to_molecule(
                atoms, charge=molecule.charge, multiplicity=molecule.multiplicity
            )
            coordinates_list.append(mol_at_point)

            _stream.write(
                f"  E = {e_ha:.8f} Ha  ({'converged' if ok else 'not converged'})\n"
            )

        except Exception as exc:
            _stream.write(f"  ⚠ Scan point {step_num} failed: {exc}\n")
            energies_hartree.append(float("nan"))
            coordinates_list.append(molecule)
            converged_all = False

        finally:
            # Always clear the constraint before the next scan point
            atoms.set_constraint()

    return PESScanResult(
        formula=molecule.get_formula(),
        method=method,
        basis=basis,
        scan_type=scan_type,
        atom_indices=list(atom_indices),
        scan_parameter_values=scan_values,
        energies_hartree=energies_hartree,
        coordinates_list=coordinates_list,
        converged_all=converged_all,
    )
