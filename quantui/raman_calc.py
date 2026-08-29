"""
Static Raman activities from analytical polarizability + geometry FD.

Computes ∂α/∂R via central finite differences on Cartesian coordinates
(±δ displacements), with the polarizability tensor at each geometry from
pyscf-properties CPHF (``pyscf.prop.polarizability``).  Normal-mode Raman
activities use the Placzek invariants

    S = 45 ᾱ² + 7 γ²

with ᾱ and γ² built from the projected ∂α/∂Q tensor (Å⁴/amu).

This is QuantUI's **Approach 3** for M-SPECTRA2 SPC2.1: PySCF core +
pyscf-properties, mirroring the existing IR dipole-derivative loop in
:mod:`quantui.freq_calc` but swapping dipole moments for polarizability
tensors.  GPU4PySCF analytical Raman and PySCFAD are future fast-path
options (see roadmap 47).

Reference: Porezag & Pederson, Phys. Rev. B **54**, 7830 (1996) (FD
framework); Wilson, Decius & Cross (invariants).
"""

from __future__ import annotations

import logging
import os
from typing import IO, Any, Callable, List, cast

import numpy as np

from quantui.config import BOHR_TO_ANGSTROM as _BOHR_TO_ANG

logger = logging.getLogger(__name__)

_DELTA_BOHR = 0.01


def raman_enabled() -> bool:
    """Return whether Raman activities should be computed during Frequency runs.

    Set ``QUANTUI_RAMAN=0`` to skip the expensive 6N polarizability pass.
    Default: enabled (teaching workflow expects IR + Raman on the same run).
    """
    return os.environ.get("QUANTUI_RAMAN", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _raman_invariants(dalpha_dQ: np.ndarray) -> float:
    """Return Raman activity S (Å⁴/amu) from a 3×3 ∂α/∂Q tensor."""
    axx = float(dalpha_dQ[0, 0])
    ayy = float(dalpha_dQ[1, 1])
    azz = float(dalpha_dQ[2, 2])
    axy = float(dalpha_dQ[0, 1])
    axz = float(dalpha_dQ[0, 2])
    ayz = float(dalpha_dQ[1, 2])
    alpha_bar = (axx + ayy + azz) / 3.0
    gamma2 = 0.5 * (
        (axx - ayy) ** 2
        + (ayy - azz) ** 2
        + (azz - axx) ** 2
        + 6.0 * (axy**2 + axz**2 + ayz**2)
    )
    return 45.0 * alpha_bar**2 + 7.0 * gamma2


def _polarizability_module(mf: Any, dm0_is_unrestricted: bool):
    """Return the pyscf.prop.polarizability submodule matching *mf*."""
    if getattr(mf, "xc", None):
        if dm0_is_unrestricted:
            from pyscf.prop.polarizability import uks as pol_mod
        else:
            from pyscf.prop.polarizability import rks as pol_mod
    elif dm0_is_unrestricted:
        from pyscf.prop.polarizability import uhf as pol_mod
    else:
        from pyscf.prop.polarizability import rhf as pol_mod
    return pol_mod


def compute_raman_activities(
    *,
    mf: Any,
    mol: Any,
    scf: Any,
    dft: Any,
    displacements: List,
    frequencies_cm1: List[float],
    dm0: Any,
    dm0_is_unrestricted: bool,
    density_fit_used: bool,
    stream: IO[str],
    status: Callable[[str], None],
) -> List[float]:
    """Compute static Raman activities (Å⁴/amu) per normal mode.

    Returns an empty list on failure (caller keeps frequencies without Raman).
    """
    if not raman_enabled():
        status("Raman activities skipped (QUANTUI_RAMAN=0).")
        return []

    if not displacements or not frequencies_cm1:
        return []

    try:
        pol_mod = _polarizability_module(mf, dm0_is_unrestricted)
    except ImportError as exc:
        logger.warning("pyscf-properties polarizability unavailable: %s", exc)
        status("Raman activities unavailable (pyscf-properties not installed).")
        return []

    from quantui.density_fitting import try_density_fit as _try_density_fit
    from quantui.gpu_offload import try_to_gpu as _try_to_gpu_inner

    _xc = getattr(mf, "xc", None)
    _n_atoms = mol.natm
    _coords0 = mol.atom_coords().copy()
    _total = _n_atoms * 3 * 2
    _done = 0

    status(
        "Numerical Raman activities: "
        f"{_done}/{_total} finite-difference polarizability evaluations "
        f"({_total - _done} remaining)"
    )

    def _displaced_alpha(atom_idx: int, ax: int, sign: int) -> np.ndarray:
        nonlocal _done
        cp = _coords0.copy()
        cp[atom_idx, ax] += sign * _DELTA_BOHR
        mol.set_geom_(cp, unit="Bohr")
        if _xc is not None:
            _mf_d = dft.UKS(mol) if dm0_is_unrestricted else dft.RKS(mol)
            _mf_d.xc = _xc
        else:
            _mf_d = scf.UHF(mol) if dm0_is_unrestricted else scf.RHF(mol)
        _mf_d.verbose = 0
        _mf_d.stdout = stream
        _mf_d, _ = _try_density_fit(_mf_d, enabled=density_fit_used)
        _mf_d, _, _ = _try_to_gpu_inner(_mf_d, "RHF")
        _mf_d.kernel(dm0=dm0)
        alpha = pol_mod.polarizability(pol_mod.Polarizability(_mf_d))
        _done += 1
        status(
            "Numerical Raman activities: "
            f"{_done}/{_total} finite-difference polarizability evaluations "
            f"({_total - _done} remaining)"
        )
        return cast(np.ndarray, np.asarray(alpha, dtype=float))

    dalpha = np.zeros((_n_atoms * 3, 3, 3), dtype=float)
    _mol_v = mol.verbose
    mol.verbose = 0
    try:
        for atom_idx in range(_n_atoms):
            for ax in range(3):
                ap = _displaced_alpha(atom_idx, ax, +1)
                am = _displaced_alpha(atom_idx, ax, -1)
                dalpha[3 * atom_idx + ax] = (ap - am) / (2.0 * _DELTA_BOHR)
    finally:
        mol.set_geom_(_coords0, unit="Bohr")
        mol.verbose = _mol_v

    dalpha_ang = dalpha / _BOHR_TO_ANG
    nm = np.asarray(displacements, dtype=float)
    if nm.ndim == 2:
        nm = nm.reshape(nm.shape[0], _n_atoms, 3)
    nm_flat = nm.reshape(len(frequencies_cm1), -1)

    activities: List[float] = []
    for mode_idx in range(len(frequencies_cm1)):
        da_dQ = np.einsum("k,kij->ij", nm_flat[mode_idx], dalpha_ang)
        activities.append(_raman_invariants(da_dQ))

    if len(activities) == len(frequencies_cm1):
        status("Numerical Raman activities complete.")
        return activities

    logger.warning(
        "Raman activity count mismatch: %d activities vs %d frequencies",
        len(activities),
        len(frequencies_cm1),
    )
    return []
