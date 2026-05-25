"""
TD-DFT excited-state calculation using PySCF.

Computes vertical excitation energies and oscillator strengths using
time-dependent density functional theory (TD-DFT).  For Hartree-Fock
methods (RHF/UHF), falls back to TDHF (equivalent to CIS) and notes
this in the output.

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
        converged: ``True`` if the ground-state SCF converged.
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
) -> TDDFTResult:
    """Run a TD-DFT excited-state calculation to obtain UV-Vis absorption data.

    Converges the ground-state SCF, then runs the time-dependent response
    equations to compute the requested number of vertical excitation energies
    and their oscillator strengths.

    When *method* is ``'RHF'`` or ``'UHF'``, the function uses TDHF (CIS)
    rather than TD-DFT and writes a note to *progress_stream*.  For a proper
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

    Returns:
        :class:`TDDFTResult` with excitation energies and oscillator strengths.

    Raises:
        ImportError: If PySCF is not installed.
        RuntimeError: If the ground-state SCF calculation fails.  If the
            TD calculation fails, excitation lists are empty and a warning
            is written to progress_stream — no exception is raised.
    """
    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is not installed — cannot run TD-DFT.\n"
            "PySCF requires Linux, macOS, or WSL."
        ) from exc

    stream: IO[str] = progress_stream if progress_stream is not None else sys.stdout

    # M-STDERR / STDERR.1: see quantui/c_stderr.py — captures fd-2 stderr
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
    _dft: Any,
    _gto: Any,
    _scf: Any,
    stream: IO[str],
) -> TDDFTResult:
    """Inner body of :func:`run_tddft_calc` (split out for STDERR.1 wrap)."""
    dft, gto, scf = _dft, _gto, _scf

    # ── Build Mole object ────────────────────────────────────────────────────
    mol = gto.Mole()
    mol.atom = molecule.to_pyscf_format()
    mol.basis = basis
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
    elif method_upper == "UHF":
        mf = scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = method

    if using_hf and progress_stream is not None:
        try:
            progress_stream.write(
                "\nNote: Using TDHF (CIS) for excited states — RHF/UHF was selected.\n"
                "For a proper TD-DFT UV-Vis spectrum, use a DFT functional\n"
                "such as B3LYP or PBE0 in the Method dropdown.\n\n"
            )
        except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
            pass

    try:
        energy_hartree = float(mf.kernel())
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

    try:
        td = mf.TDHF() if using_hf else mf.TDDFT()
        td.nstates = nstates
        td.verbose = 3
        td.stdout = stream
        td.kernel()

        excitation_energies_ev = [float(e) * HARTREE_TO_EV for e in td.e]
        osc = td.oscillator_strength()
        oscillator_strengths = [float(f) for f in osc]

    except Exception as exc:
        logger.warning("TD-DFT/TDHF calculation failed: %s", exc)
        if progress_stream is not None:
            try:
                progress_stream.write(f"\n⚠ TD-DFT failed: {exc}\n")
            except Exception:  # noqa: BLE001 — cleanup (stream may be closed)
                pass

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
    )
