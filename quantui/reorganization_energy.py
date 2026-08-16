"""
QuantUI Reorganization Energy Module

Computes the internal (inner-sphere) reorganization energy ``λ`` from Marcus
theory using the standard **4-point scheme**.

For a charge-transfer event between a neutral molecule and its ion, the
reorganization energy is the energy penalty for relaxing each charge state
from the *other* state's equilibrium geometry to its own:

    λ = λ₁ + λ₂
    λ₁ = E_ion(R_neutral)     − E_ion(R_ion)        # ion relaxation
    λ₂ = E_neutral(R_ion)     − E_neutral(R_neutral) # neutral relaxation

where ``R_x`` is the fully relaxed (optimized) geometry of charge state ``x``.
The "4 points" are the four single-point energies that appear above:

    E_neutral(R_neutral), E_ion(R_ion)      — the two optimized minima
    E_ion(R_neutral), E_neutral(R_ion)      — the two cross evaluations

Two channels are supported:

* **hole** (charge +1) — relevant for hole/p-type charge transport,
* **electron** (charge −1) — relevant for electron/n-type transport.

Running ``mode="both"`` computes both channels while sharing the single
neutral geometry optimization, so it costs three optimizations rather than
four.

The heavy lifting (SCF + gradients) is delegated to the same code paths as
the rest of QuantUI: :func:`quantui.optimizer.optimize_geometry` for the
relaxations and :func:`quantui.session_calc.run_in_session` for the
single-point cross evaluations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import IO, Any, List, Optional

from .molecule import Molecule
from .optimizer import DEFAULT_FMAX, DEFAULT_OPT_STEPS, optimize_geometry
from .session_calc import HARTREE_TO_EV, run_in_session

# 1 Hartree in kcal/mol (CODATA-consistent with HARTREE_TO_EV).
HARTREE_TO_KCAL: float = 627.509474

VALID_MODES = ("hole", "electron", "both")


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass
class ReorgChannelResult:
    """Reorganization energy for a single charge-transfer channel.

    Attributes:
        kind: ``"hole"`` (cation, +1) or ``"electron"`` (anion, −1).
        ion_charge: Total charge of the ion for this channel.
        ion_multiplicity: Spin multiplicity used for the ion.
        e_neutral_at_neutral: E_neutral(R_neutral) in Hartree (shared point).
        e_ion_at_ion: E_ion(R_ion) in Hartree.
        e_ion_at_neutral: E_ion(R_neutral) in Hartree.
        e_neutral_at_ion: E_neutral(R_ion) in Hartree.
        lambda1_hartree: Ion relaxation energy λ₁ (Ha).
        lambda2_hartree: Neutral relaxation energy λ₂ (Ha).
        lambda_hartree: Total reorganization energy λ = λ₁ + λ₂ (Ha).
        converged: True if every SCF/opt feeding this channel converged.
    """

    kind: str
    ion_charge: int
    ion_multiplicity: int
    e_neutral_at_neutral: float
    e_ion_at_ion: float
    e_ion_at_neutral: float
    e_neutral_at_ion: float
    lambda1_hartree: float
    lambda2_hartree: float
    lambda_hartree: float
    converged: bool
    # REORG.2 — the ion-optimized geometry. It was computed and discarded
    # before, which made the four energies impossible to interpret after the
    # fact: lambda measures how far the molecule relaxed on becoming an ion,
    # and without R_ion there is nothing to compare R_neutral against.
    # Optional so results loaded from a pre-REORG.2 save still construct.
    ion_molecule: Optional[Molecule] = None

    @property
    def lambda_ev(self) -> float:
        """Total reorganization energy in electronvolts."""
        return self.lambda_hartree * HARTREE_TO_EV

    @property
    def lambda_mev(self) -> float:
        """Total reorganization energy in millielectronvolts."""
        return self.lambda_ev * 1000.0

    @property
    def lambda_kcal(self) -> float:
        """Total reorganization energy in kcal/mol."""
        return self.lambda_hartree * HARTREE_TO_KCAL

    @property
    def label(self) -> str:
        """Human-readable channel label."""
        return "Hole (cation)" if self.kind == "hole" else "Electron (anion)"


def reorg_geometries(channels: list, neutral_geometry: dict) -> list[dict]:
    """The DISTINCT geometries behind a run, each labelled with its energies.

    Takes the saved-payload shape (plain dicts), so the live and History paths
    build the identical list — the same one-renderer discipline that fixed
    REORG.1.

    Deduplicated by construction: R_neutral is shared by every channel and
    appears once. A hole+electron run therefore yields three entries
    (neutral, cation, anion), not six, and not the four the four-point name
    suggests. Each entry names which of the four energies were evaluated on it,
    which is what connects the picture back to λ.
    """
    out: list[dict] = []
    if neutral_geometry:
        # Every channel evaluates E_neutral and E_ion at R_neutral, so list the
        # neutral energy once and note the ion energies that share this geometry.
        shared = ", ".join(f"E_{c.get('kind', '?')}(R_neutral)" for c in channels)
        out.append(
            {
                "label": "R_neutral — optimized neutral",
                "atoms": list(neutral_geometry["atoms"]),
                "coordinates": [list(c) for c in neutral_geometry["coordinates"]],
                "note": ("E_neutral(R_neutral)" + (f", {shared}" if shared else "")),
                # Carried through for exports (M-EXPORT2 EXP2.1/.4) — the
                # viewer doesn't need these, but an XYZ file with no charge/
                # multiplicity is a structure nobody downstream can rerun.
                "charge": neutral_geometry.get("charge", 0),
                "multiplicity": neutral_geometry.get("multiplicity", 1),
            }
        )
    for ch in channels:
        geom = ch.get("ion_geometry")
        if not geom:
            continue
        kind = ch.get("kind", "ion")
        charge = ch.get("ion_charge", 0)
        out.append(
            {
                "label": f"R_{kind} — optimized {kind} ion (charge {charge:+d})",
                "atoms": list(geom["atoms"]),
                "coordinates": [list(c) for c in geom["coordinates"]],
                "note": f"E_{kind}(R_{kind}), E_neutral(R_{kind})",
                "charge": geom.get("charge", charge),
                "multiplicity": geom.get("multiplicity", 1),
            }
        )
    return out


def geometry_rmsd(a: Molecule, b: Molecule) -> Optional[float]:
    """Mass-independent RMSD in Angstrom between two geometries, atom-for-atom.

    No alignment is performed, and that is deliberate: both geometries come
    from optimizations seeded from the same structure with the same atom
    ordering, so the displacement IS the physical quantity of interest.
    Superimposing first (Kabsch) would rotate away part of what lambda measures.

    None if the two are not comparable — a readout must never be the reason a
    result fails to display.
    """
    try:
        import numpy as np

        pa = np.asarray(a.coordinates, dtype=float)
        pb = np.asarray(b.coordinates, dtype=float)
        if pa.shape != pb.shape or pa.size == 0:
            return None
        return float(np.sqrt(((pa - pb) ** 2).sum(axis=1).mean()))
    except Exception:  # noqa: BLE001 — see docstring
        return None


def max_atom_displacement(a: Molecule, b: Molecule) -> Optional[tuple[int, float]]:
    """``(atom_index, distance)`` of the atom that moved furthest, in Angstrom.

    RMSD averages the relaxation away; a single atom moving 0.4 A in an
    otherwise rigid molecule is the interesting case and the mean hides it.
    """
    try:
        import numpy as np

        pa = np.asarray(a.coordinates, dtype=float)
        pb = np.asarray(b.coordinates, dtype=float)
        if pa.shape != pb.shape or pa.size == 0:
            return None
        d = np.sqrt(((pa - pb) ** 2).sum(axis=1))
        i = int(np.argmax(d))
        return (i, float(d[i]))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class ReorganizationEnergyResult:
    """Structured output from a 4-point reorganization energy calculation.

    Exposes ``formula``/``method``/``basis``/``energy_hartree``/``converged``
    so it can be persisted by :func:`quantui.results_storage.save_result`
    like every other result type.  ``energy_hartree`` reports the optimized
    neutral SCF energy (the physical reference for the run).
    """

    formula: str
    method: str
    basis: str
    mode: str
    molecule: Molecule  # optimized neutral geometry (used for 3D display)
    neutral_charge: int
    neutral_multiplicity: int
    neutral_energy_hartree: float
    channels: List[ReorgChannelResult] = field(default_factory=list)
    converged: bool = True
    n_total_opt_steps: int = 0

    @property
    def energy_hartree(self) -> float:
        """Optimized neutral SCF energy (Ha) — the run's reference energy."""
        return self.neutral_energy_hartree

    @property
    def energy_ev(self) -> float:
        """Optimized neutral SCF energy in electronvolts."""
        return self.neutral_energy_hartree * HARTREE_TO_EV

    def channel(self, kind: str) -> Optional[ReorgChannelResult]:
        """Return the channel result for ``"hole"``/``"electron"`` or None."""
        for ch in self.channels:
            if ch.kind == kind:
                return ch
        return None

    def to_spectra(self) -> dict:
        """Serialisable payload stored under result.json ``spectra`` key."""
        return {
            "reorganization_energy": {
                "mode": self.mode,
                "neutral_charge": self.neutral_charge,
                "neutral_multiplicity": self.neutral_multiplicity,
                "neutral_energy_hartree": self.neutral_energy_hartree,
                "n_total_opt_steps": self.n_total_opt_steps,
                "channels": [
                    {
                        "kind": ch.kind,
                        "ion_charge": ch.ion_charge,
                        "ion_multiplicity": ch.ion_multiplicity,
                        "e_neutral_at_neutral": ch.e_neutral_at_neutral,
                        "e_ion_at_ion": ch.e_ion_at_ion,
                        "e_ion_at_neutral": ch.e_ion_at_neutral,
                        "e_neutral_at_ion": ch.e_neutral_at_ion,
                        "lambda1_hartree": ch.lambda1_hartree,
                        "lambda2_hartree": ch.lambda2_hartree,
                        "lambda_hartree": ch.lambda_hartree,
                        "lambda_ev": ch.lambda_ev,
                        "lambda_kcal": ch.lambda_kcal,
                        "converged": ch.converged,
                    }
                    for ch in self.channels
                ],
            }
        }

    def summary(self) -> str:
        """Return a multi-line human-readable result summary."""
        lines = [
            "=" * 60,
            "Reorganization Energy (Marcus 4-point)",
            "=" * 60,
            f"  Molecule       : {self.formula}",
            f"  Method/Basis   : {self.method}/{self.basis}",
            f"  Neutral state  : charge {self.neutral_charge:+d}, "
            f"mult {self.neutral_multiplicity}",
            f"  All converged  : {'Yes' if self.converged else 'NO'}",
            f"  Total opt steps: {self.n_total_opt_steps}",
            "-" * 60,
        ]
        for ch in self.channels:
            lines.append(
                f"  {ch.label:<18}: λ = {ch.lambda_ev:.4f} eV "
                f"({ch.lambda_kcal:.2f} kcal/mol)"
            )
            lines.append(
                f"     λ₁ (ion relax) = {ch.lambda1_hartree * HARTREE_TO_EV:.4f} eV,"
                f"  λ₂ (neutral relax) = {ch.lambda2_hartree * HARTREE_TO_EV:.4f} eV"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================


def _promote_method(method: str, multiplicity: int) -> str:
    """Return a method suitable for the given spin state.

    PySCF's restricted RHF cannot treat an open-shell ion, and the QuantUI
    optimizer only special-cases ``RHF``/``UHF`` for Hartree-Fock (DFT is
    auto-restricted/unrestricted from the spin).  So promote a closed-shell
    HF request to UHF whenever the species is open-shell.
    """
    if multiplicity > 1 and method.upper() in ("RHF", "HF"):
        return "UHF"
    return method


def _ion_multiplicity(molecule: Molecule, ion_charge: int) -> int:
    """Low-spin multiplicity for an ion at ``ion_charge``.

    Removing/adding one electron flips the electron-count parity, so the
    ground-state multiplicity is 1 (even electrons) or 2 (odd electrons).
    This picks the minimal valid multiplicity; users wanting a high-spin ion
    can build the calculation manually.
    """
    n_electrons = molecule.get_electron_count() - (ion_charge - molecule.charge)
    return 1 if n_electrons % 2 == 0 else 2


def _emit(stream: IO[str], message: str) -> None:
    try:
        stream.write(message)
    except Exception:  # noqa: BLE001 — logging must never kill the run
        pass


# ============================================================================
# Main entry point
# ============================================================================


def run_reorganization_energy(
    molecule: Molecule,
    mode: str = "both",
    method: str = "B3LYP",
    basis: str = "6-31G*",
    fmax: float = DEFAULT_FMAX,
    steps: int = DEFAULT_OPT_STEPS,
    progress_stream: Optional[IO[str]] = None,
    solvent: Optional[str] = None,
    checkpoint: Optional[Any] = None,
    resume: bool = False,
) -> ReorganizationEnergyResult:
    """Compute the 4-point Marcus reorganization energy for a molecule.

    Args:
        molecule: The **neutral** (reference) molecule.  Its charge and
            multiplicity define the reference state; ions are derived from it.
        mode: ``"hole"``, ``"electron"``, or ``"both"``.
        method: SCF method / DFT functional (e.g. ``"B3LYP"``).  Automatically
            promoted to UHF for open-shell HF cases.
        basis: Basis set name recognised by PySCF.
        fmax: Force convergence threshold (eV/Å) for the optimizations.
        steps: Maximum optimizer steps per optimization.
        progress_stream: Writable stream for live log output (Jupyter widget
            stream in the app, ``sys.stdout`` otherwise).
        solvent: Optional PCM solvent name for the single-point evaluations.
        checkpoint: Optional :class:`~quantui.checkpoint.Checkpoint` for the
            whole run (M-CHECKPOINT CHK.7). This run is really 2-3 independent
            geometry optimizations — the neutral reference plus one per ion
            channel — and each needs its own resume identity, or one leg's
            trajectory/Hessian would overwrite another's. So *checkpoint*
            itself is never handed to ``optimize_geometry`` directly; instead
            each leg gets its own nested checkpoint via
            :meth:`~quantui.checkpoint.Checkpoint.sub`. A leg that fails to
            open its own checkpoint (disk full, permissions) simply runs
            uncheckpointed — never the reason a leg doesn't run.
        resume: Continue every leg from its own checkpoint where one exists.
            A leg with nothing usable to resume just starts fresh — this is
            not an error, it is the ordinary case for legs added after the
            interrupted run's progress. Note: a leg that had already
            *completed* before the interruption has no resumable state
            either (by design — see ``Checkpoint.resumable_state``), so it is
            re-optimized rather than instantly reused. In practice this
            re-optimization converges in a step or two, since it starts
            already at the minimum, and — because BFGS is deterministic —
            reproduces the same geometry, so a later leg seeded from it still
            finds its own checkpoint.

    Returns:
        :class:`ReorganizationEnergyResult`.

    Raises:
        ValueError: If ``mode`` is not one of :data:`VALID_MODES`.
        RuntimeError: If a required single-point evaluation fails to converge.
    """
    mode = (mode or "both").lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose one of {', '.join(VALID_MODES)}."
        )

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout
    base_charge = molecule.charge
    base_mult = molecule.multiplicity
    neutral_method = _promote_method(method, base_mult)

    def _single_point(mol: Molecule, mth: str, tag: str) -> float:
        """Run a single point and return its energy, asserting convergence."""
        _emit(stream, f"\n── Single point: {tag} ──────────────────────────\n")
        res = run_in_session(
            molecule=mol,
            method=mth,
            basis=basis,
            progress_stream=stream,  # type: ignore[arg-type]
            solvent=solvent,
        )
        if not bool(getattr(res, "converged", False)):
            raise RuntimeError(f"Single point did not converge: {tag}")
        return float(res.energy_hartree)

    def _leg_checkpoint(
        tag: str, *, charge: int, multiplicity: int, coords: Any
    ) -> Optional[Any]:
        """Nested checkpoint for one geometry-opt leg (CHK.7), or ``None``.

        ``checkpoint`` (the whole-run checkpoint) is never passed straight to
        ``optimize_geometry`` — see the docstring above for why each leg needs
        its own. A leg whose checkpoint fails to open behaves exactly like no
        checkpoint at all: the optimization still runs, it just isn't
        resumable.
        """
        if checkpoint is None:
            return None
        leg = checkpoint.sub(
            tag, charge=charge, multiplicity=multiplicity, coords=coords
        )
        return leg if leg.begin() else None

    _emit(
        stream,
        "\n"
        + "═" * 62
        + f"\n Reorganization energy (4-point) — mode: {mode}\n"
        + f" {molecule.get_formula()} @ {method}/{basis}\n"
        + "═" * 62
        + "\n",
    )

    # ── Step 1: optimize the neutral reference geometry ──────────────────────
    _emit(stream, "\n── Optimizing neutral geometry (R_neutral) ──────────\n")
    neutral_leg = _leg_checkpoint(
        "neutral_opt",
        charge=base_charge,
        multiplicity=base_mult,
        coords=molecule.coordinates,
    )
    neutral_opt = optimize_geometry(
        molecule=molecule,
        method=neutral_method,
        basis=basis,
        fmax=fmax,
        steps=steps,
        progress_stream=stream,  # type: ignore[arg-type]
        status_label="Reorg: optimizing neutral geometry",
        report_fraction=False,  # Don't let sub-opt 0→1 resets oscillate ETA
        checkpoint=neutral_leg,
        resume=resume,
    )
    neutral_mol = neutral_opt.molecule
    n_total_steps = neutral_opt.n_steps
    all_converged = bool(neutral_opt.converged)

    # E_neutral(R_neutral) — shared "point 1" across both channels.
    e_neutral_at_neutral = _single_point(
        neutral_mol, neutral_method, "E_neutral(R_neutral)"
    )

    # ── Step 2: per-channel ion optimization + cross single points ───────────
    targets: List[tuple[str, int]] = []
    if mode in ("hole", "both"):
        targets.append(("hole", base_charge + 1))
    if mode in ("electron", "both"):
        targets.append(("electron", base_charge - 1))

    channels: List[ReorgChannelResult] = []
    for kind, ion_charge in targets:
        ion_mult = _ion_multiplicity(molecule, ion_charge)
        ion_method = _promote_method(method, ion_mult)
        _emit(
            stream,
            f"\n── {kind.capitalize()} channel: optimizing ion geometry "
            f"(charge {ion_charge:+d}, mult {ion_mult}) ──\n",
        )

        # Seed the ion optimization from the relaxed neutral geometry — it is
        # closer to the ion minimum than the raw input geometry.
        ion_seed = Molecule(
            atoms=list(molecule.atoms),
            coordinates=[list(c) for c in neutral_mol.coordinates],
            charge=ion_charge,
            multiplicity=ion_mult,
        )
        ion_leg = _leg_checkpoint(
            f"{kind}_opt",
            charge=ion_charge,
            multiplicity=ion_mult,
            coords=ion_seed.coordinates,
        )
        ion_opt = optimize_geometry(
            molecule=ion_seed,
            method=ion_method,
            basis=basis,
            fmax=fmax,
            steps=steps,
            progress_stream=stream,  # type: ignore[arg-type]
            status_label=f"Reorg: optimizing {kind} ion geometry",
            report_fraction=False,  # See neutral-opt note above
            checkpoint=ion_leg,
            resume=resume,
        )
        ion_mol = ion_opt.molecule
        n_total_steps += ion_opt.n_steps
        all_converged = all_converged and bool(ion_opt.converged)

        # The four energies (two already share R_neutral / R_ion optimizations).
        e_ion_at_ion = _single_point(ion_mol, ion_method, f"E_{kind}(R_{kind})")
        e_ion_at_neutral = _single_point(
            Molecule(
                atoms=list(molecule.atoms),
                coordinates=[list(c) for c in neutral_mol.coordinates],
                charge=ion_charge,
                multiplicity=ion_mult,
            ),
            ion_method,
            f"E_{kind}(R_neutral)",
        )
        e_neutral_at_ion = _single_point(
            Molecule(
                atoms=list(molecule.atoms),
                coordinates=[list(c) for c in ion_mol.coordinates],
                charge=base_charge,
                multiplicity=base_mult,
            ),
            neutral_method,
            f"E_neutral(R_{kind})",
        )

        lambda1 = e_ion_at_neutral - e_ion_at_ion  # ion relaxation
        lambda2 = e_neutral_at_ion - e_neutral_at_neutral  # neutral relaxation
        lambda_total = lambda1 + lambda2

        channels.append(
            ReorgChannelResult(
                kind=kind,
                ion_charge=ion_charge,
                ion_multiplicity=ion_mult,
                e_neutral_at_neutral=e_neutral_at_neutral,
                e_ion_at_ion=e_ion_at_ion,
                e_ion_at_neutral=e_ion_at_neutral,
                e_neutral_at_ion=e_neutral_at_ion,
                lambda1_hartree=lambda1,
                lambda2_hartree=lambda2,
                lambda_hartree=lambda_total,
                converged=bool(ion_opt.converged),
                ion_molecule=ion_mol,
            )
        )
        _emit(
            stream,
            f"\n  → λ_{kind} = {lambda_total * HARTREE_TO_EV:.4f} eV "
            f"({lambda_total * HARTREE_TO_KCAL:.2f} kcal/mol)\n",
        )

    if checkpoint is not None:
        # Reaching here means every required single point converged (a
        # non-convergent one raises and never gets this far) — the run is
        # done, whether or not an individual leg itself fully converged.
        # Mirrors pes_scan.py: "every point was attempted, so there is
        # nothing left to resume." A leg's own checkpoint (see _leg_checkpoint)
        # only marks itself complete on convergence, independently of this —
        # a leg that hit max steps without converging stays resumable at its
        # own level even after the overall run completes.
        checkpoint.mark_complete()

    result = ReorganizationEnergyResult(
        formula=molecule.get_formula(),
        method=method,
        basis=basis,
        mode=mode,
        molecule=neutral_mol,
        neutral_charge=base_charge,
        neutral_multiplicity=base_mult,
        neutral_energy_hartree=e_neutral_at_neutral,
        channels=channels,
        converged=all_converged,
        n_total_opt_steps=n_total_steps,
    )
    _emit(stream, "\n" + result.summary() + "\n")
    return result
