"""Intrinsic reaction coordinate (IRC) from a transition state.

Follows the steepest-descent path, in mass-weighted coordinates, downhill
from a first-order saddle point in both directions to the two minima it
connects. That answers the question a transition-state search leaves open:
is this the barrier between *my* reactant and product?

Sella's IRC drives QuantUI's own ASE calculator
(``optimizer._QuantUIPySCFCalc``), so PCM solvent, density fitting, D3, GPU
offload and SCF rescue behave as in a geometry optimization. The step
settings were chosen by measurement (HCN ⇌ HNC and H2CO → H2 + CO at
RHF/6-31G): steps of 0.2 Å·amu½ with ``gamma = 0.4`` reach both minima in
35-71 SCF solves per direction, while Sella's defaults stalled or needed
over a thousand. Close to a minimum Sella's inner iterations can fail to
converge (more often with small basis sets); the path is in the right
valley by then, so that direction is finished with a plain minimization
from where it stopped, and the result says so.

Sella is the optional ``[ts]`` extra, as for the transition-state search.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field
from typing import IO, Any, Callable, List, Optional

from .molecule import Molecule
from .ts_search import SELLA_INSTALL_HINT

#: Step along the path (Å·amu½), Sella's ``dx``.
DEFAULT_IRC_DX = 0.2
#: Maximum steps in each direction.
DEFAULT_IRC_STEPS = 150
#: Force threshold (eV/Å) that ends each direction at a minimum.
DEFAULT_IRC_FMAX = 0.05
_IRC_GAMMA = 0.4
_IRC_ETA = 1e-4
_IRC_INNER_ITER = 20

#: Below this (kJ/mol) an end is not counted as lower than the start.
_DESCENT_KJ = 0.5
_HARTREE_TO_KJ = 2625.4996394799


@dataclass
class IRCBranch:
    """One direction of the path, starting at the transition state."""

    direction: str  # "forward" or "reverse" (Sella's sign of the TS mode)
    converged: bool
    trajectory: List[Molecule] = field(default_factory=list)
    energies_hartree: List[float] = field(default_factory=list)
    #: Steps that followed the IRC; any after these come from the
    #: minimization that finished the direction.
    irc_steps: Optional[int] = None

    @property
    def n_steps(self) -> int:
        return max(len(self.energies_hartree) - 1, 0)

    @property
    def end_energy_hartree(self) -> float:
        return self.energies_hartree[-1] if self.energies_hartree else float("nan")


def _mass_weighted_steps(frames: List[Molecule]) -> List[float]:
    """Mass-weighted distance (Å·amu½) between consecutive frames."""
    import numpy as np
    from ase.data import atomic_masses, atomic_numbers

    if not frames:
        return []
    masses = np.array([atomic_masses[atomic_numbers[str(a)]] for a in frames[0].atoms])
    out = []
    for a, b in zip(frames, frames[1:]):
        d = np.asarray(b.coordinates, float) - np.asarray(a.coordinates, float)
        out.append(float(np.sqrt((masses[:, None] * d**2).sum())))
    return out


@dataclass
class IRCResult:
    """Both directions of an IRC, joined into one path.

    The joined path runs from the reverse end through the transition state
    to the forward end, so the Trajectory panel plays it as one reaction.
    Its reaction coordinate is the signed mass-weighted arc length, zero at
    the transition state.
    """

    formula: str
    method: str
    basis: str
    molecule: Molecule  # the transition state the path starts from
    forward: IRCBranch
    reverse: IRCBranch
    solvent: Optional[str] = None

    @property
    def energy_hartree(self) -> float:
        return self.forward.energies_hartree[0]

    @property
    def energy_ev(self) -> float:
        from .session_calc import HARTREE_TO_EV

        return self.energy_hartree * HARTREE_TO_EV

    @property
    def converged(self) -> bool:
        return self.forward.converged and self.reverse.converged

    @property
    def n_steps(self) -> int:
        return self.forward.n_steps + self.reverse.n_steps

    @property
    def trajectory(self) -> List[Molecule]:
        return list(reversed(self.reverse.trajectory)) + self.forward.trajectory[1:]

    @property
    def energies_hartree(self) -> List[float]:
        return (
            list(reversed(self.reverse.energies_hartree))
            + self.forward.energies_hartree[1:]
        )

    @property
    def ts_index(self) -> int:
        return max(len(self.reverse.trajectory) - 1, 0)

    @property
    def reaction_coordinate(self) -> List[float]:
        back = _mass_weighted_steps(self.reverse.trajectory)
        ahead = _mass_weighted_steps(self.forward.trajectory)
        s_back = [0.0]
        for step in back:
            s_back.append(s_back[-1] - step)
        s_ahead = [0.0]
        for step in ahead:
            s_ahead.append(s_ahead[-1] + step)
        return list(reversed(s_back)) + s_ahead[1:]

    def barriers_kj(self) -> tuple:
        """``(from the reverse end, from the forward end)`` in kJ/mol."""
        e_ts = self.energy_hartree
        return (
            (e_ts - self.reverse.end_energy_hartree) * _HARTREE_TO_KJ,
            (e_ts - self.forward.end_energy_hartree) * _HARTREE_TO_KJ,
        )

    @property
    def verdict(self) -> str:
        b_rev, b_fwd = self.barriers_kj()
        lower = [b > _DESCENT_KJ for b in (b_rev, b_fwd)]
        if not all(lower):
            return (
                "The path did not go downhill on both sides, so the start "
                "may not be a transition state. Run Transition State first "
                "and check it has exactly one imaginary frequency."
            )
        text = (
            "The transition state connects two minima. Barrier: "
            f"{b_rev:.1f} kJ/mol from the start of the path, "
            f"{b_fwd:.1f} kJ/mol from its end. Check in the Trajectory panel "
            "that the first and last structures are your reactant and product."
        )
        finished = [
            b.direction for b in (self.reverse, self.forward) if b.irc_steps is not None
        ]
        if finished:
            text += (
                " Near the end the "
                + " and ".join(finished)
                + " path was completed by minimizing, so its last steps are an "
                "optimization rather than the reaction path."
            )
        if not self.converged:
            text = (
                "Not every direction reached a minimum within the step limit; "
                "the end structures are approximate (optimize them to be sure). " + text
            )
        return text

    def to_spectra(self) -> dict:
        """The ``irc`` block stored in ``result.json`` spectra."""
        b_rev, b_fwd = self.barriers_kj()
        return {
            "reaction_coordinate": [float(s) for s in self.reaction_coordinate],
            "energies_hartree": [float(e) for e in self.energies_hartree],
            "ts_index": self.ts_index,
            "forward": {
                "converged": bool(self.forward.converged),
                "n_steps": self.forward.n_steps,
                "irc_steps": self.forward.irc_steps,
            },
            "reverse": {
                "converged": bool(self.reverse.converged),
                "n_steps": self.reverse.n_steps,
                "irc_steps": self.reverse.irc_steps,
            },
            "barrier_from_start_kj": float(b_rev),
            "barrier_from_end_kj": float(b_fwd),
            "verdict": self.verdict,
        }


def run_irc(
    molecule: Molecule,
    method: str = "RHF",
    basis: str = "STO-3G",
    *,
    dx: float = DEFAULT_IRC_DX,
    steps: int = DEFAULT_IRC_STEPS,
    fmax: float = DEFAULT_IRC_FMAX,
    progress_stream: Optional[IO[str]] = None,
    solvent: Optional[str] = None,
    scf_rescue: bool = True,
    cancel_check: Any = None,
    on_step: Optional[Callable[[str, Molecule, float], None]] = None,
) -> IRCResult:
    """Follow the IRC both ways from the transition state *molecule*.

    Args:
        molecule: A first-order saddle point (a Transition State result).
        method, basis: Level of theory; use the one the transition state
            was found at (HF or DFT; post-HF has no analytic gradients here).
        dx: Step length along the path, Å·amu½.
        steps: Maximum steps in each direction.
        fmax: Force threshold (eV/Å) at which a direction stops.
        on_step: Called with ``(direction, geometry, energy_hartree)`` after
            every step, the starting point included.
    """
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' has no analytic gradients here, so it cannot follow "
            "a reaction path. Use RHF/UHF or a DFT functional."
        )
    try:
        from sella import IRC
        from sella.optimize.irc import IRCInnerLoopConvergenceFailure
    except ImportError as exc:
        raise ImportError(SELLA_INSTALL_HINT) from exc
    from ase import Atoms
    from ase.optimize import BFGS

    from .optimizer import _QuantUIPySCFCalc
    from .session_calc import HARTREE_TO_EV, resolve_solvent

    if _QuantUIPySCFCalc is None:
        raise ImportError("ASE calculator support is unavailable.")
    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    solvent = resolve_solvent(solvent)
    stream.write(
        "\n── Intrinsic reaction coordinate (Sella IRC, mass-weighted) ──\n"
        f"  {molecule.get_formula()}  {method}/{basis}"
        + (f"  PCM: {solvent}" if solvent else "")
        + f"\n  step {dx} Å·amu½, fmax = {fmax} eV/Å, max {steps} steps each way\n"
    )

    # One IRC object for both directions: Sella diagonalizes the Hessian at
    # the transition state once and goes back to it (and to that Hessian)
    # for the second direction, so "forward" and "reverse" are the two signs
    # of the same mode. Separate objects can pick opposite signs and send
    # both directions into the same valley.
    atoms = Atoms(list(molecule.atoms), positions=molecule.coordinates)
    calc = _QuantUIPySCFCalc(
        method=method,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        progress_stream=stream,
        status_label="IRC",
        scf_rescue=scf_rescue,
        solvent=solvent,
        cancel_check=cancel_check,
    )
    atoms.calc = calc
    irc = IRC(
        atoms,
        logfile=os.devnull,
        dx=dx,
        eta=_IRC_ETA,
        gamma=_IRC_GAMMA,
        ninner_iter=_IRC_INNER_ITER,
    )
    current: dict = {}

    def _record() -> None:
        e_ha = float(atoms.get_potential_energy()) / HARTREE_TO_EV
        frame = Molecule(
            list(molecule.atoms),
            atoms.get_positions().tolist(),
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            validate_spin=False,
        )
        current["frames"].append(frame)
        current["energies"].append(e_ha)
        if on_step is not None:
            on_step(current["direction"], frame, e_ha)
        stream.write(
            f"  {current['direction']:7s} step {len(current['energies']) - 1:3d}"
            f"   E = {e_ha:.8f} Ha\n"
        )

    irc.attach(_record, interval=1)
    branches = {}
    for direction in ("forward", "reverse"):
        current.update(direction=direction, frames=[], energies=[])
        stream.write(f"\n{direction.capitalize()} direction:\n")
        # ASE records the starting point only when nsteps is 0; Sella puts
        # the transition state back before the second direction, so reset
        # the count and that direction starts its path at the saddle too.
        irc.nsteps = 0
        irc_steps: Optional[int] = None
        try:
            converged = bool(irc.run(fmax=fmax, steps=steps, direction=direction))
        except IRCInnerLoopConvergenceFailure:
            irc_steps = max(len(current["energies"]) - 1, 0)
            stream.write(
                f"  Sella's IRC step did not converge after {irc_steps} steps; "
                "finishing this direction by minimizing from here.\n"
            )
            opt = BFGS(atoms, logfile=io.StringIO())
            opt.attach(_record, interval=1)
            converged = bool(opt.run(fmax=fmax, steps=steps))
            # The minimization's first observer call repeats the last point.
            if len(current["frames"]) > irc_steps + 1:
                del current["frames"][irc_steps + 1]
                del current["energies"][irc_steps + 1]
            if len(current["frames"]) == irc_steps + 1:
                irc_steps = None  # it was already at the minimum
        if not current["frames"]:  # nothing recorded: keep the start at least
            _record()
        branches[direction] = IRCBranch(
            direction,
            converged,
            current["frames"],
            current["energies"],
            irc_steps=irc_steps,
        )
        stream.write(
            f"  {direction}: {'reached a minimum' if converged else 'did NOT converge'}"
            f" after {branches[direction].n_steps} steps\n"
        )
    applied_solvent = getattr(calc, "solvent_applied", None) or solvent

    result = IRCResult(
        formula=molecule.get_formula(),
        method=method,
        basis=basis,
        molecule=molecule,
        forward=branches["forward"],
        reverse=branches["reverse"],
        solvent=applied_solvent,
    )
    stream.write(f"\n{result.verdict}\n")
    return result
