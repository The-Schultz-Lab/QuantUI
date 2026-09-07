"""
TD-DFT excited-state calculation using PySCF.

Computes vertical excitation energies and oscillator strengths using
time-dependent density functional theory (TD-DFT).  For Hartree-Fock
methods (RHF/UHF), falls back to full TDHF (the random-phase
approximation, RPA — ``mf.TDHF()``) and notes this in the output.

AUDIT additional-concerns — TDHF is NOT the same method as CIS. Full
TDHF/RPA includes the excitation/de-excitation (A/B block) coupling that
the Tamm-Dancoff approximation (TDA) drops; CIS is HF's TDA. This module
calls ``mf.TDHF()`` (full RPA), so its labels say TDHF/RPA rather than
CIS. See PySCF's own discussion of the distinction:
https://pyscf.org/user/tddft.html

Platform notes
--------------
Requires PySCF — Linux / macOS / WSL only.

Educational value
-----------------
* Students see which wavelengths a molecule absorbs (UV-Vis spectrum).
* Oscillator strengths indicate which transitions are optically allowed
  (bright, f > ~0.01) versus dark (f ≈ 0).
* Teaches the connection between electronic structure and spectroscopy.
* Comparing TD-DFT results for different functionals shows how the
  choice of functional affects excitation energies.

Typical usage
-------------
>>> from quantui.tddft_calc import run_tddft_calc
>>> result = run_tddft_calc(molecule, method="B3LYP", basis="6-31G")
>>> for e, f in zip(result.excitation_energies_ev, result.oscillator_strengths):
...     print(f"  E = {e:.3f} eV,  f = {f:.4f}")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import IO, Any, List, Optional

from .molecule import Molecule
from .session_calc import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# Planck × speed of light in eV·nm  (h·c = 1239.84 eV·nm)
_EV_TO_NM: float = 1239.84193


# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class TDDFTResult:
    """Structured output from a TD-DFT excited-state calculation.

    Attributes:
        energy_hartree: Ground-state SCF energy in Hartrees.
        homo_lumo_gap_ev: HOMO-LUMO gap in eV from the ground-state SCF,
            or ``None``.
        converged: ``True`` when the ground-state SCF converged, the TD
            solve ran and produced states, and — whenever per-root status is
            available — at least one requested root actually converged
            (AUDIT F08; relaxed in code review from "every root" to "at
            least one root", since it is routine for a Davidson solve's
            higher/harder roots to miss the default iteration budget while
            the lower, physically relevant ones are fine). ``False`` only
            for a solve that raised, returned no states, or (when
            ``td_converged`` is known) converged *none* of them — an
            SCF-only flag is not overall success for a calculation whose
            deliverable is the excited states. A PySCF build that doesn't
            expose ``td.converged`` at all (``td_converged is None``) is
            treated as "no per-root information available", not as
            "unconverged" — it does not by itself flip this to ``False``.
            See ``td_converged``/``n_converged_states`` for the per-root
            detail this folds together.
        n_iterations: Number of ground-state SCF macro-iterations.
        method: DFT functional or HF method used.
        basis: Basis set.
        formula: Hill-notation molecular formula.
        excitation_energies_ev: Vertical excitation energies in eV.
        oscillator_strengths: Oscillator strengths (dimensionless).
            Bright (optically allowed) transitions have f > ~0.01.
        nstates: Number of excited states requested.
    """

    energy_hartree: float
    homo_lumo_gap_ev: Optional[float]
    converged: bool
    n_iterations: int
    method: str
    basis: str
    formula: str
    excitation_energies_ev: List[float] = field(default_factory=list)
    oscillator_strengths: List[float] = field(default_factory=list)
    nstates: int = 10
    density_fit: bool = False
    # AUDIT F08 — per-root Davidson convergence from the TD solver itself,
    # distinct from the ground-state SCF's own converged flag above. None
    # if the TD solve never ran (e.g. it raised before td.kernel()
    # completed) or the installed PySCF doesn't expose td.converged.
    td_converged: Optional[List[bool]] = None
    # Count of roots whose td_converged flag is True — distinguishes
    # "requested" (nstates), "returned" (len(excitation_energies_ev)), and
    # "converged" root counts, since a Davidson solve can return energies
    # for roots it never actually converged.
    n_converged_states: Optional[int] = None
    # M-UX2 UXP2.10 — the actual PySCF class dispatched for the
    # ground-state SCF (e.g. "RHF", "UHF", "RKS", "UKS"); "" for an older
    # saved result.
    scf_variant: str = ""

    @property
    def energy_ev(self) -> float:
        """Ground-state SCF energy in electronvolts."""
        return self.energy_hartree * HARTREE_TO_EV

    def wavelengths_nm(self) -> List[float]:
        """Return excitation wavelengths in nm (λ = 1239.84 / E_eV)."""
        return [
            _EV_TO_NM / e if e > 0 else float("inf")
            for e in self.excitation_energies_ev
        ]


# ============================================================================
# Main function
# ============================================================================


def run_tddft_calc(
    molecule: Molecule,
    method: str = "B3LYP",
    basis: str = "STO-3G",
    nstates: int = 10,
    progress_stream: Optional[IO[str]] = None,
    scf_rescue: bool = True,
) -> TDDFTResult:
    """Run a TD-DFT excited-state calculation to obtain UV-Vis absorption data.

    Converges the ground-state SCF, then runs the time-dependent response
    equations to compute the requested number of vertical excitation energies
    and their oscillator strengths.

    When *method* is ``'RHF'`` or ``'UHF'``, the function uses full TDHF
    (RPA — NOT the CIS/Tamm-Dancoff approximation; see the module
    docstring) rather than TD-DFT, and writes a note to *progress_stream*.
    For a proper
    UV-Vis simulation, a DFT functional such as ``'B3LYP'`` or ``'PBE0'`` is
    strongly recommended.

    Args:
        molecule: Validated :class:`~quantui.molecule.Molecule`.
        method: DFT functional (e.g. ``'B3LYP'``, ``'PBE0'``,
            ``'CAM-B3LYP'``) or ``'RHF'``/``'UHF'`` for TDHF.
            Default: ``'B3LYP'``.
        basis: Basis set name.  Default: ``'STO-3G'``.
        nstates: Number of excited states to compute.  Default: 10.
        progress_stream: Optional writable text stream for live PySCF output.
        scf_rescue: Whether the ground-state SCF automatically retries
            through the shared rescue helper on non-convergence
            (M-SCF-ROBUST, see :mod:`quantui.scf_robust`). Default ``True``.

    Returns:
        :class:`TDDFTResult` with excitation energies and oscillator strengths.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the ground-state SCF calculation fails.  If the
            TD calculation fails, excitation lists are empty and a warning
            is written to progress_stream — no exception is raised.
    """
    # Post-HF methods (MP2/CCSD/CCSD(T)) have no special-casing below —
    # without this guard, method='CCSD' silently falls into the DFT
    # branch (sets mf.xc = "CCSD") and fails deep inside PySCF with a
    # cryptic "LibXCFunctional: name 'CCSD' not found" instead of a clear
    # message. TD-DFT/TDHF is not defined for these methods here.
    from . import config as _config

    if method.strip().upper() in _config.POST_HF_METHODS:
        raise ValueError(
            f"'{method}' is a post-HF method and cannot be used for "
            "TD-DFT/UV-Vis — use RHF/UHF (TDHF) or a DFT functional "
            "instead."
        )

    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run TD-DFT.\n"
            "PySCF requires Linux, macOS, or WSL."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # See quantui/c_stderr.py — captures fd-2 stderr
    # from libcint / BLAS / LAPACK / TDA solver C code and relays to
    # ``stream`` on exit. POSIX-only; no-op on Windows.
    from quantui.c_stderr import capture_c_stderr

    with capture_c_stderr(stream):
        return _run_tddft_calc_body(
            molecule=molecule,
            method=method,
            basis=basis,
            nstates=nstates,
            progress_stream=progress_stream,
            scf_rescue=scf_rescue,
            _dft=dft,
            _gto=gto,
            _scf=scf,
            stream=stream,
        )


def _run_tddft_calc_body(
    *,
    molecule: Molecule,
    method: str,
    basis: str,
    nstates: int,
    progress_stream: Optional[IO[str]],
    scf_rescue: bool = True,
    _dft: Any,
    _gto: Any,
    _scf: Any,
    stream: IO[str],
) -> TDDFTResult:
    """Inner body of :func:`run_tddft_calc` (split out for stderr-capture wrap)."""
    dft, gto, scf = _dft, _gto, _scf

    # ── Build Mole object ────────────────────────────────────────────────────
    from .inorganic_guards import ecp_for_basis

    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
    # Heavy-element ECP (LANL2DZ / def2); empty dict for all-electron bases.
    mol.ecp = ecp_for_basis(basis, molecule.atoms)
    mol.charge = molecule.charge
    mol.spin = molecule.multiplicity - 1
    mol.verbose = 4
    mol.stdout = stream
    mol.build()

    # ── SCF ──────────────────────────────────────────────────────────────────
    method_upper = method.upper()
    using_hf = method_upper in ("RHF", "UHF")

    if method_upper == "RHF":
        mf = scf.RHF(mol)
        scf_variant = type(mf).__name__
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
        scf_variant = type(mf).__name__
    else:
        # Route through resolve_xc + maybe_apply_d3 so
        # methods like wB97X-D (PySCF rejects "wb97x-d") map cleanly.
        from .session_calc import maybe_apply_d3, resolve_xc

        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        # M-UX2 UXP2.10 — capture before maybe_apply_d3 can wrap/rename it.
        scf_variant = type(mf).__name__
        mf.xc = resolve_xc(method)
        mf, _ = maybe_apply_d3(mf, method, progress_stream=progress_stream)

    # Density fitting (RI), opt-in (M-DF). Off by default. TD-DFT is where the
    # measured win is largest (~1.6x on aspirin), so this is the primary target.
    from .density_fitting import try_density_fit as _try_density_fit

    mf, density_fit_used = _try_density_fit(mf)

    if using_hf and progress_stream is not None:
        try:
            progress_stream.write(
                "\nNote: Using TDHF/RPA for excited states — RHF/UHF was selected.\n"
                "This is full TDHF (the random-phase approximation, with\n"
                "excitation/de-excitation coupling), not the CIS/Tamm-Dancoff\n"
                "approximation. For a proper TD-DFT UV-Vis spectrum, use a DFT\n"
                "functional such as B3LYP or PBE0 in the Method dropdown.\n\n"
            )
        except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
            pass

    # Cooperative cancel between SCF cycles.
    from .cancellation import attach_scf_cancel_callback, cancel_check_from_stream
    from .log_utils import emit_status

    attach_scf_cancel_callback(mf, cancel_check_from_stream(stream))

    emit_status(stream, "Running SCF (ground state)…")
    from .scf_robust import run_scf_with_rescue

    try:
        energy_hartree = float(
            run_scf_with_rescue(mf, rescue=scf_rescue, stream=stream)
        )
    except Exception as exc:
        raise RuntimeError(
            f"SCF failed for {molecule.get_formula()} ({method}/{basis}): {exc}"
        ) from exc

    converged = bool(getattr(mf, "converged", False))
    n_iterations = int(getattr(mf, "cycles", -1))

    # ── HOMO-LUMO gap (non-fatal) ────────────────────────────────────────────
    homo_lumo_gap_ev: Optional[float] = None
    try:
        import numpy as _np

        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        if isinstance(mo_energy, (list, _np.ndarray)) and hasattr(
            mo_energy[0], "__len__"
        ):
            mo_e_ref, mo_occ_ref = mo_energy[0], mo_occ[0]
        else:
            mo_e_ref, mo_occ_ref = mo_energy, mo_occ
        n_occ = int((_np.array(mo_occ_ref) > 0).sum())
        if 0 < n_occ < len(mo_e_ref):
            homo_lumo_gap_ev = float(
                (mo_e_ref[n_occ] - mo_e_ref[n_occ - 1]) * HARTREE_TO_EV
            )
    except Exception as exc:
        logger.debug("HOMO-LUMO gap extraction failed in TD-DFT calc: %s", exc)

    # ── TD-DFT / TDHF ────────────────────────────────────────────────────────
    excitation_energies_ev: List[float] = []
    oscillator_strengths: List[float] = []
    # AUDIT F08 — td.kernel() copies energies/oscillator strengths without
    # checking td.converged (a per-root array from the Davidson solve). A
    # real one-iteration-limited TDHF/6-31G water solve had
    # converged=[False, False, False] yet reported success with three
    # excitations. scf_converged is this function's SCF-only flag (the old
    # sole source of `converged` below); td_converged/n_converged_states
    # carry the TD solve's own per-root status.
    scf_converged = converged
    td_converged: Optional[List[bool]] = None
    n_converged_states: Optional[int] = None

    try:
        emit_status(
            stream,
            f"Solving {'TDHF/RPA' if using_hf else 'TD-DFT'} "
            f"excited states ({nstates})…",
        )
        td = mf.TDHF() if using_hf else mf.TDDFT()
        td.nstates = nstates
        # verbose=5 (DEBUG) is what surfaces PySCF's per-root "root %d
        # converged" lines during the Davidson solve — the only progress
        # signal available while it runs. _LogCapture.write in app.py greps
        # for per-root convergence lines to update the live status label.
        td.verbose = 5
        td.stdout = stream
        td.kernel()

        excitation_energies_ev = [float(e) * HARTREE_TO_EV for e in td.e]
        osc = td.oscillator_strength()
        oscillator_strengths = [float(f) for f in osc]

        _raw_td_converged = getattr(td, "converged", None)
        if _raw_td_converged is not None:
            td_converged = [bool(c) for c in _raw_td_converged]
            n_converged_states = sum(td_converged)

    except Exception as exc:
        logger.warning("TD-DFT/TDHF calculation failed: %s", exc)
        if progress_stream is not None:
            try:
                progress_stream.write(f"\n⚠ TD-DFT failed: {exc}\n")
            except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
                pass

    # AUDIT F08 — overall success requires the TD solve to have actually
    # produced converged roots, not just a ground-state SCF. A TD-DFT run
    # whose entire purpose is the excited states is not "converged" if the
    # Davidson solve raised before producing any roots, or if none of the
    # roots it returned ever converged.
    #
    # Code review (2026-09): requiring *every* requested root to converge
    # was too strict — a Davidson solve for nstates > a few routinely leaves
    # the higher/harder roots short of the default iteration budget while
    # the lower ones (usually what a UV-Vis analysis actually cares about)
    # are fine, so a perfectly normal multi-state run was always flagged
    # "treat with caution". Relaxed to "at least one requested root
    # converged" — still catches the original bug this audit fixed (a
    # solve where every root came back unconverged), and n_converged_states
    # below still reports the exact per-root tally for anyone who wants it.
    #
    # Separately, when the installed PySCF doesn't expose ``td.converged``
    # at all, td_converged stays None — that is "no information", not
    # "unconverged", and must not by itself force converged=False (it used
    # to, forcing every TD-DFT/TDHF result on such a build to be flagged).
    converged = (
        scf_converged
        and excitation_energies_ev != []
        and (td_converged is None or any(td_converged))
    )

    return TDDFTResult(
        energy_hartree=energy_hartree,
        homo_lumo_gap_ev=homo_lumo_gap_ev,
        converged=converged,
        n_iterations=n_iterations,
        method=method,
        basis=basis,
        formula=molecule.get_formula(),
        excitation_energies_ev=excitation_energies_ev,
        oscillator_strengths=oscillator_strengths,
        nstates=nstates,
        density_fit=density_fit_used,
        scf_variant=scf_variant,
        td_converged=td_converged,
        n_converged_states=n_converged_states,
    )
