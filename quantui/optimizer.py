"""
QM geometry optimization using ASE-BFGS + PySCF gradients.

Performs a full quantum mechanical geometry optimization by coupling the
ASE BFGS optimizer with a thin PySCF wrapper calculator.  Atoms are
moved iteratively until the maximum force on any atom falls below the
convergence threshold (``fmax``).

Returns both the final optimized molecule and the complete list of
intermediate frames as a trajectory — enabling step-by-step
visualization of the relaxation path in the notebook.

Platform notes
--------------
Requires PySCF — **Linux / macOS / WSL only**.  ASE >= 3.22 required.
This module imports PySCF lazily so it can be imported safely on Windows.

Implementation note
-------------------
ASE does not ship an ``ase.calculators.pyscf`` module.  Instead this
module defines ``_QuantUIPySCFCalc``, a minimal ASE Calculator that
calls PySCF's SCF kernel and analytical nuclear-gradient driver directly.

Educational value
-----------------
* Students see the molecule relax step-by-step (trajectory slider in the
  notebook's 3D viewer).
* The energy-vs-step plot shows convergence behaviour.
* RMSD between initial and final geometry quantifies the structural change.
* Teaches that real molecular properties require an optimized geometry.

Typical usage
-------------
>>> from quantui import optimize_geometry
>>> result = optimize_geometry(molecule, method="RHF", basis="STO-3G")
>>> print(result.summary())
>>> # result.trajectory is a list[Molecule] for the step-through viewer
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, List, Optional

from .ase_bridge import ASE_AVAILABLE, atoms_to_molecule, molecule_to_atoms
from .config import BOHR_TO_ANGSTROM as _BOHR_TO_ANG
from .molecule import Molecule
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# Defaults also exposed in config.py for the notebook UI
DEFAULT_FMAX: float = 0.05  # eV/Å — tight enough for educational use
DEFAULT_OPT_STEPS: int = 200  # generous upper limit for small molecules


# ============================================================================
# Minimal ASE Calculator wrapping PySCF
# ============================================================================

# Defined conditionally so the module can be imported on Windows (no ASE).
try:
    from ase.calculators.calculator import (  # type: ignore[import]
        Calculator,
        all_changes,
    )

    class _QuantUIPySCFCalc(Calculator):
        """
        Thin ASE Calculator that drives PySCF SCF + analytical gradients.

        ASE does not provide an ``ase.calculators.pyscf`` module, so this
        class replaces it.  It builds a PySCF ``Mole`` from the current
        ASE ``Atoms`` object at each step, runs the SCF, computes the
        nuclear gradient, and converts both to ASE units (eV and eV/Å).

        All PySCF output is routed to a ``StringIO`` sink so the notebook
        output stays clean; BFGS step progress is handled by ASE.
        """

        implemented_properties: List[str] = ["energy", "forces"]

        def __init__(
            self,
            method: str = "RHF",
            basis: str = "STO-3G",
            charge: int = 0,
            spin: int = 0,
            cancel_check=None,
            progress_stream=None,
            status_label: str = "Optimizing geometry",
            expected_steps=None,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.method = method
            self.basis = basis
            self.charge = charge
            self.spin = spin
            # Cooperative-cancel predicate; checked per step + wired into
            # the per-step SCF callback (the SCF runs silent here, so the
            # stream-based cancel can't see it).
            self.cancel_check = cancel_check
            # Progress stream + label for per-step status
            # heartbeats (the SCF runs at verbose=0, so nothing else surfaces
            # progress during a step). ``_eval_count`` counts force evaluations.
            self.progress_stream = progress_stream
            self.status_label = status_label
            self.expected_steps = expected_steps  # history-based ~N prior
            self._eval_count = 0

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)

            # Bail before starting this step's SCF if cancel was clicked
            # (fires between BFGS force evaluations, independent of ASE output).
            from .cancellation import (
                attach_scf_cancel_callback,
                raise_if_cancelled,
            )

            raise_if_cancelled(self.cancel_check)

            # Heartbeat so the status line advances during the
            # (silent) per-step SCF + gradient.
            self._eval_count += 1
            from .log_utils import emit_status

            _step_of = (
                f"{self._eval_count}/~{int(self.expected_steps)}"
                if self.expected_steps
                else f"{self._eval_count}"
            )
            emit_status(
                self.progress_stream,
                f"{self.status_label} — SCF + gradient (step {_step_of})…",
            )

            import numpy as np
            from pyscf import dft, gto, scf

            _sink = io.StringIO()  # absorb all PySCF output

            if self.atoms is None:
                raise RuntimeError("No Atoms object attached to calculator.")

            # Build PySCF molecule from the current ASE geometry
            _atom_list_for_cube = [
                (sym, pos)
                for sym, pos in zip(
                    self.atoms.get_chemical_symbols(),
                    self.atoms.get_positions().tolist(),
                )
            ]
            mol = gto.Mole()
            mol.atom = _atom_list_for_cube
            mol.basis = self.basis
            mol.charge = self.charge
            mol.spin = self.spin
            mol.unit = "Angstrom"
            mol.verbose = 0
            mol.stdout = _sink
            mol.build()

            # Select SCF method
            method_upper = self.method.upper()
            if method_upper in ("RHF", "HF"):
                mf = scf.RHF(mol)
            elif method_upper == "UHF":
                mf = scf.UHF(mol)
            else:
                # DFT functional. Route through resolve_xc +
                # maybe_apply_d3 so wB97X-D / PBE-D3 work mid-optimization.
                from .session_calc import maybe_apply_d3, resolve_xc

                mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
                mf.xc = resolve_xc(self.method)
                mf = maybe_apply_d3(mf, self.method)

            mf.verbose = 0
            mf.stdout = _sink

            # Per-SCF-cycle heartbeat during the (silent) step,
            # so the status advances mid-SCF, not just per optimizer step.
            _k = self._eval_count

            def _scf_progress(envs, _k=_k) -> None:
                cyc = envs.get("cycle") if hasattr(envs, "get") else None
                if cyc is None:
                    return
                emit_status(
                    self.progress_stream,
                    f"{self.status_label} — step {_k}, SCF cycle {cyc + 1}…",
                )

            attach_scf_cancel_callback(mf, self.cancel_check, progress_cb=_scf_progress)
            mf.kernel()

            # Save final SCF state for orbital visualization
            self._last_mf = mf
            self._last_atom_list = _atom_list_for_cube

            # Analytical nuclear gradient (Hartree/Bohr)
            grad_driver = mf.nuc_grad_method()
            grad_driver.verbose = 0
            grad_driver.stdout = _sink
            g_ha_bohr = grad_driver.kernel()  # shape (n_atoms, 3)

            # Convert to ASE units and store results
            # Force = -gradient;  1 Ha/Bohr = HARTREE_TO_EV / _BOHR_TO_ANG eV/Å
            self.results["energy"] = float(mf.e_tot) * HARTREE_TO_EV
            self.results["forces"] = (
                -np.asarray(g_ha_bohr) * HARTREE_TO_EV / _BOHR_TO_ANG
            )

except ImportError:
    # ASE not installed — _QuantUIPySCFCalc is unavailable.
    # optimize_geometry() will raise a clear ImportError before ever using it.
    _QuantUIPySCFCalc = None  # type: ignore[assignment,misc]


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class OptimizationResult:
    """
    Structured output from a completed QM geometry optimization.

    Attributes:
        molecule: Final optimized :class:`~quantui.molecule.Molecule`.
        trajectory: All frames as a list of Molecule objects, starting from
            the *input* geometry and ending at the optimized geometry.
            Length is ``n_steps + 1``.
        energies_hartree: SCF energy in Hartrees at each trajectory frame.
            Same length as ``trajectory``.
        converged: ``True`` if the maximum atomic force dropped below
            ``fmax`` within the allowed number of steps.
        n_steps: Number of BFGS optimizer steps taken (``len(trajectory) - 1``).
        method: Calculation method used (e.g. ``'RHF'``).
        basis: Basis set used (e.g. ``'STO-3G'``).
        formula: Hill-notation molecular formula of the input molecule.
    """

    molecule: Molecule
    trajectory: List[Molecule]
    energies_hartree: List[float]
    converged: bool
    n_steps: int
    method: str
    basis: str
    formula: str
    mo_energy_hartree: Optional[Any] = None  # from final SCF step
    mo_occ: Optional[Any] = None
    mo_coeff: Optional[Any] = None
    pyscf_mol_atom: Optional[Any] = None  # atom list at final geometry (Angstrom)
    pyscf_mol_basis: Optional[str] = None

    @property
    def energy_hartree(self) -> float:
        """Final energy in Hartrees (last trajectory frame)."""
        return self.energies_hartree[-1] if self.energies_hartree else float("nan")

    @property
    def energy_ev(self) -> float:
        """Final energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    @property
    def energy_change_hartree(self) -> float:
        """Total energy change from the first to the last frame (Ha)."""
        if len(self.energies_hartree) < 2:
            return 0.0
        return self.energies_hartree[-1] - self.energies_hartree[0]

    @property
    def rmsd_angstrom(self) -> float:
        """
        Root-mean-square displacement (Å) between the initial and final geometry.

        Measures how much the structure changed during optimization.
        Uses pure Python so it works without numpy.
        """
        if len(self.trajectory) < 2:
            return 0.0
        initial = self.trajectory[0].coordinates
        final = self.trajectory[-1].coordinates
        n = len(initial)
        if n == 0:
            return 0.0
        total = sum(
            (fx - ix) ** 2 + (fy - iy) ** 2 + (fz - iz) ** 2
            for (ix, iy, iz), (fx, fy, fz) in zip(initial, final)
        )
        return math.sqrt(total / n)

    def summary(self) -> str:
        """Return a multi-line human-readable result summary."""
        lines = [
            "=" * 60,
            "Geometry Optimization Results",
            "=" * 60,
            f"  Molecule       : {self.formula}",
            f"  Method/Basis   : {self.method}/{self.basis}",
            f"  Converged      : {'Yes' if self.converged else '❌ NO — max steps reached'}",
            f"  Steps taken    : {self.n_steps}",
            f"  Final energy   : {self.energy_hartree:.8f} Ha",
            f"  Energy change  : {self.energy_change_hartree:+.6f} Ha",
            f"  Geometry RMSD  : {self.rmsd_angstrom:.4f} Å",
            "=" * 60,
        ]
        if self.converged:
            lines.append("✅ Optimization converged successfully!")
        else:
            lines.append(
                "⚠️  Optimization did not converge.\n"
                "   Try increasing Max Steps, loosening Force Threshold,\n"
                "   or using LJ pre-optimization to improve the starting geometry."
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Main function
# ============================================================================


def _write_stream(stream: Optional[IO[str]], text: str) -> None:
    """Write to *stream*, ignoring a closed or broken one."""
    if stream is None:
        return
    try:
        stream.write(text)
    except Exception:  # noqa: BLE001 — a progress note is never worth raising
        pass


def _unlink_quietly(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _resume_start_geometry(checkpoint: Optional[Any]) -> Optional[Molecule]:
    """Return the geometry to restart from, or ``None`` if there isn't one.

    ``None`` covers no checkpoint, no trajectory, an empty or corrupt
    trajectory, and a checkpoint that already completed. All of them mean
    "nothing to continue", and the caller handles that by starting normally —
    so this deliberately reports absence rather than raising.
    """
    if checkpoint is None:
        return None
    try:
        state = checkpoint.resumable_state()
        if state is None:
            return None
        traj_path = checkpoint.trajectory_path
        if not traj_path.is_file() or traj_path.stat().st_size == 0:
            return None
        from ase.io.trajectory import Trajectory  # type: ignore[import]

        frames = list(Trajectory(str(traj_path)))
        if not frames:
            return None
        return atoms_to_molecule(
            frames[-1],
            charge=int(state.get("charge", 0) or 0),
            multiplicity=int(state.get("multiplicity", 1) or 1),
        )
    except Exception as exc:  # noqa: BLE001 — a bad checkpoint is not an error
        logger.debug("could not resume from checkpoint: %s", exc)
        return None


def optimize_geometry(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    fmax: float = DEFAULT_FMAX,
    steps: int = DEFAULT_OPT_STEPS,
    progress_stream: Optional[IO[str]] = None,
    status_label: str = "Optimizing geometry",
    report_fraction: bool = True,
    expected_steps: Optional[int] = None,
    checkpoint: Optional[Any] = None,
    resume: bool = False,
) -> OptimizationResult:
    """
    Optimize a molecular geometry at the QM level using ASE-BFGS + PySCF.

    Runs a BFGS quasi-Newton geometry optimization.  At each step the
    PySCF mean-field calculator provides the energy and analytical
    nuclear gradients (forces).  The BFGS optimizer moves the atoms
    toward lower energy until convergence.

    The full trajectory (one :class:`~quantui.molecule.Molecule` per
    optimizer step, including the initial geometry) is stored in
    :attr:`OptimizationResult.trajectory` for step-through visualization.

    Args:
        molecule: Starting geometry as a validated
            :class:`~quantui.molecule.Molecule`.
        method: SCF method — ``'RHF'`` or ``'UHF'``.  Default: ``'RHF'``.
            For optimization ``'RHF'`` is recommended unless the molecule
            is an open-shell radical.
        basis: Basis set.  ``'STO-3G'`` is fastest; ``'6-31G'`` or
            ``'6-31G*'`` give more chemically accurate geometries but
            are significantly slower.  Default: ``'STO-3G'``.
        fmax: Force convergence threshold in eV/Å.  Optimization stops
            when the maximum force on any atom is below this value.
            Default: 0.05 eV/Å (a standard tight threshold).
        steps: Maximum number of BFGS optimizer steps.  Default: 200.
        progress_stream: Optional writable text stream.  BFGS step
            progress (step number and maximum force) is written here.
            Pass a widget-backed stream in the notebook for live output;
            leave ``None`` to write to ``sys.stdout``.
        checkpoint: Optional :class:`~quantui.checkpoint.Checkpoint`. When
            given, the ASE trajectory and the BFGS Hessian state are written
            into the checkpoint directory after every step instead of a temp
            directory, so an interrupted optimization can be continued.
        resume: Continue from *checkpoint* rather than from *molecule*. The
            starting geometry becomes the last frame of the stored trajectory
            and BFGS reloads its accumulated Hessian, so the steps already
            taken are not repeated. Ignored (with a note to the progress
            stream) when the checkpoint holds nothing usable.

    Returns:
        :class:`OptimizationResult` containing the optimized molecule,
        full trajectory, per-step energies, convergence status, and
        summary statistics.

    Raises:
        ImportError: If ASE or PySCF is not installed.
        RuntimeError: If the optimization raises an unexpected exception
            (original exception is chained).

    Note:
        PySCF verbose output is suppressed during optimization to keep
        the progress stream clean.  BFGS writes a concise per-step table
        (step number and maximum force) to *progress_stream*.
    """
    # --- Dependency checks ---
    if not ASE_AVAILABLE or _QuantUIPySCFCalc is None:
        raise ImportError(
            "ASE is not installed — cannot run geometry optimization.\n"
            "  pip install 'ase>=3.22.0'\n"
            "  # or: conda install -c conda-forge ase"
        )

    # Post-HF methods (MP2/CCSD/CCSD(T)) have no special-casing in
    # _QuantUIPySCFCalc — without this guard, method='CCSD' silently falls
    # into the DFT branch (sets mf.xc = "CCSD") and fails deep inside PySCF
    # with a cryptic "LibXCFunctional: name 'CCSD' not found" instead of a
    # clear message. No analytical post-HF nuclear gradients are wired up
    # here, so these methods are single-point only (see session_calc.py).
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' is a post-HF method and cannot be used for geometry "
            "optimization — QuantUI only has analytical gradients wired up "
            "for HF/DFT methods here. Optimize with RHF, UHF, or a DFT "
            f"functional, then run a Single Point calculation with '{method}' "
            "on the optimized geometry."
        )

    try:
        import pyscf as _pyscf  # noqa: F401 — presence check
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run geometry optimization.\n"
            "  conda install -c conda-forge pyscf\n"
            "Note: PySCF is Linux / macOS / WSL only."
        ) from exc

    try:
        from ase.optimize import BFGS  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "ase.optimize.BFGS is not available.\n"
            "Ensure ASE >= 3.22.0: pip install 'ase>=3.22.0'"
        ) from exc

    _stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    _null = io.StringIO()

    # --- Set up ASE Atoms + PySCF calculator ---
    from .cancellation import cancel_check_from_stream, raise_if_cancelled

    _cancel_check = cancel_check_from_stream(_stream)

    # --- Resume from a checkpoint (M-CHECKPOINT CHK.2) ---
    # Restart means two things, and only doing one of them wastes most of the
    # saving: continue from the last geometry *and* reload the BFGS Hessian.
    # Without the Hessian, BFGS restarts as steepest descent and spends
    # several steps rebuilding curvature it had already learned.
    _resume_from = _resume_start_geometry(checkpoint) if resume else None
    if resume and _resume_from is None:
        _write_stream(
            _stream,
            "\n⚠  No usable checkpoint to resume — starting from the beginning.\n",
        )
    if _resume_from is not None and checkpoint is not None:
        try:
            _done = (checkpoint.load_state() or {}).get("steps_done")
            _detail = (
                f"continuing from step {_done}; "
                "geometry and optimizer curvature restored"
                if isinstance(_done, int) and _done > 0
                else "continuing from the last saved geometry"
            )
            checkpoint.log_resumed(_detail)
        except Exception:  # noqa: BLE001 — provenance is never worth a crash
            pass
    start_molecule = _resume_from if _resume_from is not None else molecule

    atoms = molecule_to_atoms(start_molecule)
    atoms.calc = _QuantUIPySCFCalc(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        cancel_check=_cancel_check,
        progress_stream=_stream,
        status_label=status_label,
        expected_steps=expected_steps,
    )

    # PySCF gradients (called by ASE-BFGS at every
    # step) emit fd-2 stderr from libcint / BLAS. Wrap the full BFGS run
    # in capture_c_stderr so those bytes go to _stream instead of the red-
    # text channel. POSIX-only; no-op on Windows.
    from quantui.c_stderr import capture_c_stderr

    # --- Run optimization with trajectory file ---
    converged = False
    try:
        with contextlib.ExitStack() as _stack:
            _resuming = _resume_from is not None
            if checkpoint is not None:
                # Durable location: the point of a checkpoint is that it
                # outlives the process that wrote it.
                checkpoint.dir.mkdir(parents=True, exist_ok=True)
                traj_path = checkpoint.trajectory_path
                restart_path: Optional[Path] = checkpoint.optimizer_restart_path
                if not _resuming:
                    # Starting over into an existing checkpoint directory:
                    # clear the stored Hessian first. Leaving it would have
                    # BFGS begin with curvature from a run this one is not
                    # continuing — silently, and with no way for the user to
                    # tell that is what happened.
                    _unlink_quietly(restart_path)
            else:
                tmpdir = _stack.enter_context(tempfile.TemporaryDirectory())
                traj_path = Path(tmpdir) / "opt.traj"
                restart_path = None

            dyn = BFGS(
                atoms,
                trajectory=str(traj_path),
                logfile=_stream,  # BFGS step table → progress_stream
                # Appending keeps the frames already computed, so the result's
                # trajectory covers the whole optimization rather than only
                # the portion after the interruption.
                append_trajectory=_resuming,
                restart=str(restart_path) if restart_path is not None else None,
            )
            # Check cancel after every BFGS step (belt-and-suspenders
            # with the per-step calculator check above).
            if _cancel_check is not None:
                dyn.attach(lambda: raise_if_cancelled(_cancel_check), interval=1)

            # Keep the checkpoint's step count current so a resume prompt can
            # say how much work is already banked. Written per step because
            # the interesting failure is the process dying without warning —
            # anything deferred to the end would never be written at all.
            if checkpoint is not None:

                def _record_step() -> None:
                    checkpoint.update(steps_done=getattr(dyn, "nsteps", None))

                dyn.attach(_record_step, interval=1)

            # Estimate completion from the fmax-convergence trend
            # (log-scale between the first step's fmax and the target). Data-free
            # and self-correcting. Skipped when report_fraction is False (e.g.
            # reorg drives several sub-optimizations, whose 0→1 resets would make
            # an overall remaining-time estimate oscillate).
            if report_fraction:
                from .log_utils import emit_progress

                _fmax0: list = []  # first-step fmax, captured on first callback

                def _report_opt_fraction() -> None:
                    try:
                        forces = atoms.get_forces()
                        fmax_now = float(math.sqrt((forces**2).sum(axis=1).max()))
                    except Exception:  # noqa: BLE001 — progress is best-effort
                        return
                    if fmax_now <= 0:
                        return
                    if not _fmax0:
                        _fmax0.append(fmax_now)
                        return
                    denom = math.log(_fmax0[0] / fmax) if fmax > 0 else 0.0
                    frac = math.log(_fmax0[0] / fmax_now) / denom if denom > 0 else 0.0
                    # Floor with the history-based step prior so early steps
                    # (where the fmax trend is noisy / near 0) still advance.
                    if expected_steps:
                        step = getattr(atoms.calc, "_eval_count", 0)
                        frac = max(frac, min(step / float(expected_steps), 0.9))
                    emit_progress(_stream, max(0.0, min(frac, 0.99)))

                dyn.attach(_report_opt_fraction, interval=1)

            with capture_c_stderr(_stream), contextlib.redirect_stdout(_null):
                converged = bool(dyn.run(fmax=fmax, steps=steps))

            if checkpoint is not None and converged:
                # Only a converged optimization is finished. One that hit the
                # step limit still has work left, so it stays resumable —
                # raising `steps` and continuing is a real workflow.
                checkpoint.mark_complete()

            # --- Read trajectory frames ---
            from ase.io.trajectory import Trajectory  # type: ignore[import]

            traj_frames = list(Trajectory(str(traj_path)))

    except Exception as exc:
        raise RuntimeError(
            f"Geometry optimization failed for {molecule.get_formula()} "
            f"({method}/{basis}): {exc}"
        ) from exc

    # Convert ASE frames → Molecule objects and extract stored energies
    charge = molecule.charge
    mult = molecule.multiplicity

    trajectory: List[Molecule] = []
    energies_hartree: List[float] = []

    for frame in traj_frames:
        mol_frame = atoms_to_molecule(frame, charge=charge, multiplicity=mult)
        trajectory.append(mol_frame)
        # Each frame has a SinglePointCalculator with the stored energy (eV)
        try:
            e_ev = frame.get_potential_energy()
            energies_hartree.append(e_ev / HARTREE_TO_EV)
        except Exception:  # noqa: BLE001 — NaN fallback for missing per-frame energy
            energies_hartree.append(float("nan"))

    if not trajectory:
        # Edge case: no frames written — return the final atoms state
        trajectory = [atoms_to_molecule(atoms, charge=charge, multiplicity=mult)]
        try:
            e_ev = atoms.get_potential_energy()
            energies_hartree = [e_ev / HARTREE_TO_EV]
        except Exception:  # noqa: BLE001 — NaN fallback for missing final energy
            energies_hartree = [float("nan")]

    n_steps = max(0, len(trajectory) - 1)
    formula = molecule.get_formula()

    # Extract MO data from the final SCF step (non-fatal)
    _opt_mo_energy: Optional[Any] = None
    _opt_mo_occ: Optional[Any] = None
    _opt_mo_coeff: Optional[Any] = None
    _opt_mol_atom: Optional[Any] = None
    _opt_mol_basis: Optional[str] = None
    try:
        import numpy as _np_mo

        _last_mf = getattr(atoms.calc, "_last_mf", None)
        _last_atom_list = getattr(atoms.calc, "_last_atom_list", None)
        if _last_mf is not None:
            _opt_mo_energy = _np_mo.array(_last_mf.mo_energy)
            _opt_mo_occ = _np_mo.array(_last_mf.mo_occ)
            _opt_mo_coeff = _np_mo.array(_last_mf.mo_coeff)
            _opt_mol_atom = _last_atom_list
            _opt_mol_basis = basis
    except Exception as exc:
        # Silent failure here ships an OptimizationResult with no MO data,
        # breaking Energies + Isosurface panels on history replay.
        # (Same root-cause class as session_calc.)
        logger.warning(
            "Final-step MO extraction failed in optimizer for %s: %s",
            molecule.get_formula(),
            exc,
        )

    # Write a final MO summary to the progress stream (replaces per-step verbose output
    # which is suppressed to avoid thousands of SCF lines for long optimizations).
    if _opt_mo_energy is not None and _opt_mo_occ is not None:
        try:
            import numpy as _np_summary

            _HARTREE_TO_EV_s = 27.211386245988
            _e_ev_raw = _np_summary.asarray(_opt_mo_energy) * _HARTREE_TO_EV_s
            _occ_raw = _np_summary.asarray(_opt_mo_occ)
            # For UHF the arrays are (2, n_mo); use alpha spin for summary.
            if _e_ev_raw.ndim == 2:
                _e_ev_1d = _e_ev_raw[0]
                _occ_1d = _occ_raw[0]
            else:
                _e_ev_1d = _e_ev_raw
                _occ_1d = _occ_raw
            _homo_idx = (
                int(_np_summary.where(_occ_1d > 0)[0][-1])
                if (_occ_1d > 0).any()
                else -1
            )
            _lumo_idx = (
                int(_np_summary.where(_occ_1d == 0)[0][0])
                if (_occ_1d == 0).any()
                else -1
            )
            _stream.write(
                "\n── Final SCF (optimised geometry) ────────────────────────────────────\n"
            )
            if _homo_idx >= 0:
                _stream.write(
                    f"  HOMO (MO #{_homo_idx}): {_e_ev_1d[_homo_idx]:.4f} eV\n"
                )
            if _lumo_idx >= 0:
                _stream.write(
                    f"  LUMO (MO #{_lumo_idx}): {_e_ev_1d[_lumo_idx]:.4f} eV\n"
                )
            if _homo_idx >= 0 and _lumo_idx >= 0:
                _stream.write(
                    f"  HOMO-LUMO gap: {_e_ev_1d[_lumo_idx] - _e_ev_1d[_homo_idx]:.4f} eV\n"
                )
            _stream.write(
                f"  All MO energies (eV): {' '.join(f'{e:.3f}' for e in _e_ev_1d)}\n"
            )
        except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
            pass

    logger.info(
        "Geometry optimization: %s %s/%s  steps=%d  converged=%s  "
        "E_final=%.8f Ha  RMSD~%.4f Å",
        formula,
        method,
        basis,
        n_steps,
        converged,
        energies_hartree[-1] if energies_hartree else float("nan"),
        _rmsd(molecule, trajectory[-1]) if len(trajectory) > 1 else 0.0,
    )

    return OptimizationResult(
        molecule=trajectory[-1],
        trajectory=trajectory,
        energies_hartree=energies_hartree,
        converged=converged,
        n_steps=n_steps,
        method=method,
        basis=basis,
        formula=formula,
        mo_energy_hartree=_opt_mo_energy,
        mo_occ=_opt_mo_occ,
        mo_coeff=_opt_mo_coeff,
        pyscf_mol_atom=_opt_mol_atom,
        pyscf_mol_basis=_opt_mol_basis,
    )


def _rmsd(mol_a: Molecule, mol_b: Molecule) -> float:
    """Compute RMSD (Å) between two same-sized molecules (no alignment, pure Python)."""
    a = mol_a.coordinates
    b = mol_b.coordinates
    if len(a) != len(b) or not a:
        return float("nan")
    total = sum(
        (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
        for (ax, ay, az), (bx, by, bz) in zip(a, b)
    )
    return math.sqrt(total / len(a))
