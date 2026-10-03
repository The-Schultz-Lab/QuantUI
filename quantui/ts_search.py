"""Transition-state search (M-TS TS.2; engine choice DEC-024).

A first-order saddle point search with Sella, driven by QuantUI's own ASE
calculator (``optimizer._QuantUIPySCFCalc``), so PCM solvent, density
fitting, D3, GPU offload and SCF rescue behave exactly as in a geometry
optimization. Sella starts from PySCF's analytic Hessian, not a guess.

The search is followed by a frequency calculation at the final geometry,
because "converged" only means the forces vanished: a transition state is
a stationary point with exactly ONE imaginary frequency. The result says in
plain language which kind of stationary point was found.

Sella is an optional dependency (``pip install "quantui[ts]"``); without it
:func:`run_ts_search` raises ``ImportError`` with that instruction.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import IO, Any, List, Optional

from .molecule import Molecule

#: Shown wherever the Transition State calc type is unavailable.
SELLA_INSTALL_HINT = (
    "Transition-state search needs the optional Sella package: "
    'pip install "quantui[ts]" (or pip install sella).'
)

#: Default force threshold (eV/Å). Tighter than a minimization's 0.05: near a
#: saddle point the surface is flat in one direction, and a loose threshold
#: stops short of it.
DEFAULT_TS_FMAX = 0.01
DEFAULT_TS_STEPS = 200

# Frequencies whose magnitude is below this (cm⁻¹) are treated as numerical
# noise from residual rotation/translation, not a real imaginary mode.
IMAGINARY_NOISE_CM1 = 50.0


def sella_available() -> bool:
    """True when Sella (and therefore the Transition State calc type) is usable."""
    try:
        import sella  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means unavailable
        return False
    return True


def classify_stationary_point(frequencies_cm1: List[float]) -> tuple:
    """``(n_imaginary, imaginary_list, verdict)`` for a set of frequencies.

    Imaginary frequencies are stored as negative numbers. Small ones
    (|ν| < :data:`IMAGINARY_NOISE_CM1`) are not counted but are mentioned.
    """
    imag = sorted(f for f in frequencies_cm1 if f < -IMAGINARY_NOISE_CM1)
    small = [f for f in frequencies_cm1 if -IMAGINARY_NOISE_CM1 <= f < 0]
    n = len(imag)
    if n == 1:
        verdict = (
            f"Transition state: exactly one imaginary frequency "
            f"({abs(imag[0]):.0f}i cm⁻¹). Animate that mode in the Vibrational "
            "panel to check it is the reaction you intended."
        )
    elif n == 0:
        verdict = (
            "Not a transition state: no imaginary frequency, so the search "
            "ended at a minimum. Start from a geometry closer to the barrier "
            "top (e.g. the highest point of a PES scan)."
        )
    else:
        freqs = ", ".join(f"{abs(f):.0f}i" for f in imag)
        verdict = (
            f"Not a first-order transition state: {n} imaginary frequencies "
            f"({freqs} cm⁻¹), a higher-order saddle point. Displace along the "
            "extra mode(s) and search again."
        )
    if small:
        verdict += (
            f" ({len(small)} small imaginary value(s) below "
            f"{IMAGINARY_NOISE_CM1:.0f} cm⁻¹ treated as numerical noise.)"
        )
    return n, imag, verdict


@dataclass
class TSResult:
    """Transition-state search plus its frequency check.

    Carries the fields of an optimization result (trajectory, energies,
    steps) and of a frequency result (modes, intensities, thermochemistry,
    orbitals, populations), so the existing Trajectory, Vibrational, IR,
    Energies, Isosurface and Populations panels and the History machinery
    work on it unchanged.
    """

    formula: str
    method: str
    basis: str
    molecule: Molecule
    energy_hartree: float
    converged: bool  # search converged AND frequency check ran
    search_converged: bool
    n_steps: int
    trajectory: List[Molecule] = field(default_factory=list)
    energies_hartree: List[float] = field(default_factory=list)
    n_imaginary: Optional[int] = None
    imaginary_cm1: List[float] = field(default_factory=list)
    verdict: str = ""
    solvent: Optional[str] = None
    freq: Any = None  # FreqResult at the final geometry

    @property
    def is_transition_state(self) -> bool:
        return self.n_imaginary == 1

    @property
    def energy_ev(self) -> float:
        from .session_calc import HARTREE_TO_EV

        return self.energy_hartree * HARTREE_TO_EV

    # Frequency-result fields, delegated, for the analysis panels and saving.
    def __getattr__(self, name: str) -> Any:
        if name in _FREQ_FIELDS:
            freq = self.__dict__.get("freq")
            return getattr(freq, name, None) if freq is not None else None
        raise AttributeError(name)

    def to_spectra(self) -> dict:
        """The ``transition_state`` block stored in ``result.json`` spectra."""
        return {
            "search_converged": bool(self.search_converged),
            "n_steps": int(self.n_steps),
            "n_imaginary": self.n_imaginary,
            "imaginary_cm1": [float(f) for f in self.imaginary_cm1],
            "verdict": self.verdict,
        }


_FREQ_FIELDS = frozenset(
    {
        "frequencies_cm1",
        "ir_intensities",
        "raman_activities",
        "zpve_hartree",
        "displacements",
        "thermo",
        "homo_lumo_gap_ev",
        "n_iterations",
        "mo_energy_hartree",
        "mo_occ",
        "mo_coeff",
        "pyscf_mol_atom",
        "pyscf_mol_basis",
        "atom_symbols",
        "mulliken_charges",
        "dipole_moment_debye",
        "dipole_vector_debye",
        "spin_square",
        "scf_variant",
        "density_fit",
    }
)


def _analytic_hessian_fn(calc: Any):
    """Sella ``hessian_function``: PySCF's analytic Hessian in eV/Å².

    Reuses the calculator's last SCF when it is at the requested geometry
    (Sella asks for the Hessian right after an energy/force evaluation),
    so PCM / D3 / density fitting are the same as for the forces.
    """
    import numpy as np

    from .session_calc import HARTREE_TO_EV

    bohr_to_ang = 0.529177210903

    def _hessian(atoms: Any) -> Any:
        atoms.get_forces()  # make sure the calculator is at this geometry
        mf = getattr(calc, "_last_mf", None)
        if mf is None:
            raise RuntimeError("No SCF available for the Hessian.")
        h = mf.Hessian()
        h.verbose = 0
        hess = np.asarray(h.kernel())  # (N, N, 3, 3), Hartree/Bohr²
        n = hess.shape[0]
        return (
            hess.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
            * HARTREE_TO_EV
            / bohr_to_ang**2
        )

    return _hessian


def run_ts_search(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    *,
    fmax: float = DEFAULT_TS_FMAX,
    steps: int = DEFAULT_TS_STEPS,
    progress_stream: Optional[IO[str]] = None,
    solvent: Optional[str] = None,
    scf_rescue: bool = True,
    analytic_hessian: bool = True,
    check_frequencies: bool = True,
    cancel_check: Any = None,
) -> TSResult:
    """Search for a transition state near *molecule*, then check it.

    Args:
        molecule: Starting guess (close to the barrier top).
        method, basis: Level of theory (PySCF; HF or DFT — post-HF methods
            have no analytic gradients here).
        fmax: Force threshold, eV/Å.
        steps: Maximum Sella steps.
        solvent: PCM solvent for the search and the frequency check.
        analytic_hessian: Start Sella from PySCF's analytic Hessian (and use
            it when Sella refreshes its Hessian). Otherwise Sella estimates it.
        check_frequencies: Run the frequency calculation afterwards.
    """
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' has no analytic gradients here, so it cannot drive a "
            "transition-state search. Use RHF/UHF or a DFT functional."
        )
    try:
        from sella import Sella
    except ImportError as exc:
        raise ImportError(SELLA_INSTALL_HINT) from exc
    from ase import Atoms

    from .optimizer import _QuantUIPySCFCalc
    from .session_calc import HARTREE_TO_EV, resolve_solvent

    if _QuantUIPySCFCalc is None:
        raise ImportError("ASE calculator support is unavailable.")
    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    solvent = resolve_solvent(solvent)

    atoms = Atoms(list(molecule.atoms), positions=molecule.coordinates)
    calc = _QuantUIPySCFCalc(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        progress_stream=stream,
        status_label="Transition-state search",
        scf_rescue=scf_rescue,
        solvent=solvent,
        cancel_check=cancel_check,
    )
    atoms.calc = calc

    stream.write(
        "\n── Transition-state search (Sella, first-order saddle point) ──\n"
        f"  {molecule.get_formula()}  {method}/{basis}"
        + (f"  PCM: {solvent}" if solvent else "")
        + f"\n  fmax = {fmax} eV/Å, max {steps} steps"
        + (", analytic starting Hessian" if analytic_hessian else "")
        + "\n"
    )

    trajectory: List[Molecule] = []
    energies: List[float] = []

    def _record() -> None:
        e_ha = float(atoms.get_potential_energy()) / HARTREE_TO_EV
        fmax_now = float((atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5)
        trajectory.append(
            Molecule(
                list(molecule.atoms),
                atoms.get_positions().tolist(),
                charge=molecule.charge,
                multiplicity=molecule.multiplicity,
                validate_spin=False,
            )
        )
        energies.append(e_ha)
        stream.write(
            f"  step {len(energies) - 1:3d}   E = {e_ha:.8f} Ha   "
            f"max force = {fmax_now:.4f} eV/Å\n"
        )

    kwargs: dict = {}
    if analytic_hessian:
        kwargs["hessian_function"] = _analytic_hessian_fn(calc)
    opt = Sella(atoms, order=1, internal=True, logfile=io.StringIO(), **kwargs)
    opt.attach(_record, interval=1)
    search_converged = bool(opt.run(fmax=fmax, steps=steps))
    n_steps = max(len(energies) - 1, 0)
    stream.write(
        f"\nSearch {'converged' if search_converged else 'did NOT converge'} "
        f"after {n_steps} steps.\n"
    )

    final = Molecule(
        list(molecule.atoms),
        atoms.get_positions().tolist(),
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
        validate_spin=False,
    )
    energy = energies[-1] if energies else float("nan")

    freq = None
    n_imag: Optional[int] = None
    imag: List[float] = []
    verdict = (
        "The search did not converge, so this geometry is not a stationary "
        "point. Try more steps or a better starting guess."
        if not search_converged
        else "Frequency check skipped."
    )
    if check_frequencies:
        from .freq_calc import run_freq_calc

        stream.write("\n── Frequency check at the final geometry ──\n")
        freq = run_freq_calc(
            final,
            method,
            basis,
            progress_stream=stream,
            scf_rescue=scf_rescue,
            solvent=solvent,
        )
        energy = float(freq.energy_hartree)
        if freq.frequencies_cm1:
            n_imag, imag, verdict = classify_stationary_point(
                list(freq.frequencies_cm1)
            )
            if not search_converged:
                verdict = (
                    "The search did NOT converge (forces still above the "
                    "threshold); the frequencies below describe an unconverged "
                    "geometry. " + verdict
                )
        stream.write(f"\n{verdict}\n")

    return TSResult(
        formula=molecule.get_formula(),
        method=method,
        basis=basis,
        molecule=final,
        energy_hartree=energy,
        converged=bool(search_converged and (freq is None or freq.converged)),
        search_converged=search_converged,
        n_steps=n_steps,
        trajectory=trajectory,
        energies_hartree=energies,
        n_imaginary=n_imag,
        imaginary_cm1=imag,
        verdict=verdict,
        solvent=getattr(calc, "solvent_applied", None) or solvent,
        freq=freq,
    )
