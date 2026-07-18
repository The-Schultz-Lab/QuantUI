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

    def _finite_energies(self) -> List[float]:
        """``energies_hartree`` with failed-point NaN placeholders dropped.

        M6 audit fix (2026-07-14): a failed scan point appends
        ``float("nan")`` to ``energies_hartree`` (see :func:`run_pes_scan`).
        Python's ``min``/``max`` are order-dependent with NaN present — a
        NaN as the first element "wins" (everything compares False against
        it) and poisons the result; a NaN later in the list is correctly
        ignored. Filtering NaN out before any min/max call makes the result
        deterministic regardless of *which* scan point failed.
        """
        return [e for e in self.energies_hartree if math.isfinite(e)]

    @property
    def energy_hartree(self) -> float:
        """Minimum SCF energy across all *successful* scan points (Hartrees)."""
        finite = self._finite_energies()
        return min(finite) if finite else float("nan")

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
        """Energy relative to the lowest successful scan point, in kcal/mol.

        Failed points (NaN in ``energies_hartree``) stay NaN here too —
        callers plotting this list should skip non-finite entries.
        """
        finite = self._finite_energies()
        if not finite:
            return []
        e_min = min(finite)
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
        finite = self._finite_energies()
        if not finite:
            return "No scan points computed."
        e_min = min(finite)
        e_max = max(finite)
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

    # Post-HF methods (MP2/CCSD/CCSD(T)) have no special-casing in
    # _QuantUIPySCFCalc (shared with optimizer.py) — without this guard,
    # method='CCSD' silently falls into the DFT branch (sets mf.xc =
    # "CCSD") and fails deep inside PySCF with a cryptic "LibXCFunctional:
    # name 'CCSD' not found" instead of a clear message.
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' is a post-HF method and cannot be used for a PES "
            "scan — QuantUI only has analytical gradients wired up for "
            "HF/DFT methods here. Scan with RHF, UHF, or a DFT functional "
            "instead."
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

    # UXP.5: cooperative cancel — checked between scan points, per BFGS step,
    # and inside each point's SCF (via the shared calculator).
    from .cancellation import cancel_check_from_stream, raise_if_cancelled

    _cancel_check = cancel_check_from_stream(_stream)
    atoms.calc.cancel_check = _cancel_check

    import numpy as np

    scan_values = np.linspace(start, stop, steps).tolist()

    energies_hartree: List[float] = []
    coordinates_list: List[Molecule] = []
    converged_all = True
    # M6 audit fix (2026-07-14): on a failed scan point, fall back to the
    # last successfully-computed geometry rather than the original input
    # molecule. Snapping every failed frame back to the starting geometry
    # produced a bogus discontinuity in the trajectory animation/plot —
    # the last-good geometry is a far more sensible placeholder for "we
    # don't know where this point landed, but it wasn't back at the start."
    _last_good_molecule = molecule

    i1, i2 = atom_indices[0], atom_indices[1]
    i3 = atom_indices[2] if len(atom_indices) >= 3 else 0
    i4 = atom_indices[3] if len(atom_indices) >= 4 else 0

    for step_num, val in enumerate(scan_values, start=1):
        raise_if_cancelled(_cancel_check)
        # M-PROGRESS A2: live per-point status (the per-point SCF is silent).
        from .log_utils import emit_status

        emit_status(
            _stream,
            f"Scan point {step_num}/{steps} — relaxing (SCF + gradient)…",
        )
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
                    constraint = FixInternals(bonds=[[val, [i1, i2]]])
                elif scan_type == "angle":
                    atoms.set_angle(i1, i2, i3, val)
                    # M7 audit fix (2026-07-14): ASE's radian-based `angles=`
                    # kwarg is not just deprecated, it's flat-out broken with
                    # the currently-targeted ASE (>=3.22, verified against
                    # 3.29.0) — internally it does
                    # ``np.asarray(angles); angles[:, 0] = ...`` to convert
                    # to degrees, which raises "setting an array element
                    # with a sequence" for any real angle constraint (the
                    # per-entry [value, [3 indices]] shape isn't
                    # rectangular). Every angle/dihedral PES scan silently
                    # failed at 100% of its points as a result. `angles_deg`
                    # takes the value directly in degrees and skips that
                    # broken reshape entirely.
                    constraint = FixInternals(angles_deg=[[val, [i1, i2, i3]]])
                else:  # dihedral
                    atoms.set_dihedral(i1, i2, i3, i4, val)
                    constraint = FixInternals(dihedrals_deg=[[val, [i1, i2, i3, i4]]])

                atoms.set_constraint(constraint)

                dyn = BFGS(atoms, logfile=_stream)
                if _cancel_check is not None:
                    dyn.attach(lambda: raise_if_cancelled(_cancel_check), interval=1)
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
            _last_good_molecule = mol_at_point

            _stream.write(
                f"  E = {e_ha:.8f} Ha  ({'converged' if ok else 'not converged'})\n"
            )

        except Exception as exc:
            _stream.write(f"  ⚠ Scan point {step_num} failed: {exc}\n")
            energies_hartree.append(float("nan"))
            coordinates_list.append(_last_good_molecule)
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
