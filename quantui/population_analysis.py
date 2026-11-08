"""Natural population analysis (NPA) of a converged SCF density.

Pure-NumPy helpers on a PySCF ``Mole`` and an AO density matrix, called
from :func:`quantui.session_calc.ground_state_analysis` next to Mulliken.

**NPA / NAO, not NBO.** PySCF's :mod:`pyscf.lo.nao` builds *natural atomic
orbitals* (Reed, Weinstock & Weinhold, J. Chem. Phys. 83, 735 (1985)): per
atom and angular momentum, the eigenvectors of the atom-block density,
then an occupancy-weighted orthogonalisation (core / valence / Rydberg).
The natural population of an atom is the sum of the occupancies of its
NAOs, and its natural charge is ``Z_eff − population``. This is the NPA
step of NBO analysis only: no natural *bond* orbitals, Lewis structures or
second-order donor–acceptor energies (those need the external NBO program).

Every function is best-effort at the call site: the caller wraps it so a
failure here can never break a calculation.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, List, Optional

import numpy as np


def _f64(x: Any) -> np.ndarray:
    """``np.asarray(x, dtype=float)`` with a concrete type for mypy."""
    out: np.ndarray = np.asarray(x, dtype=float)
    return out


class _DensityHolder:
    """Minimal stand-in for an SCF object: :func:`pyscf.lo.nao.nao` only
    calls ``make_rdm1()``. Passing the (spin-summed) density this way keeps
    the UHF/ROHF/UKS handling here instead of relying on ``nao``'s own
    ``isinstance`` check, which does not cover every SCF class."""

    def __init__(self, dm: np.ndarray) -> None:
        self._dm = dm

    def make_rdm1(self) -> np.ndarray:
        return self._dm


def total_density(dm: Any) -> np.ndarray:
    """Spin-summed AO density: ``P = Pα + Pβ`` for a ``(2, nao, nao)`` input."""
    arr = _f64(dm)
    if arr.ndim == 3:
        return _f64(arr[0] + arr[1])
    return arr


def ao_overlap(mol: Any) -> np.ndarray:
    """AO overlap matrix ``S`` of *mol*."""
    return _f64(mol.intor_symmetric("int1e_ovlp"))


def nao_coefficients(mol: Any, dm: Any, s: Optional[np.ndarray] = None) -> np.ndarray:
    """AO → NAO coefficient matrix ``C`` (``Cᵀ S C = 1``).

    Column *k* is the NAO built in the slot of AO *k*, so NAOs keep the AO
    atom assignment (``mol.aoslice_by_atom``). The total (α+β) density is
    used for an unrestricted or ROHF reference.
    """
    from pyscf.lo import nao as _nao

    if s is None:
        s = ao_overlap(mol)
    return _f64(_nao.nao(mol, _DensityHolder(total_density(dm)), s=s))


def npa_charges(
    mol: Any,
    dm: Any,
    *,
    s: Optional[np.ndarray] = None,
    c_nao: Optional[np.ndarray] = None,
) -> List[float]:
    """Natural (NPA) charge on each atom, in units of *e*.

    ``q_A = Z_A − Σ_{k∈A} (Cᵀ S P S C)_kk`` with ``P`` the total density.
    ``Z_A`` is the effective nuclear charge (``mol.atom_charges()``), so an
    ECP's core electrons are excluded on both sides. The charges sum to the
    molecular charge.
    """
    if s is None:
        s = ao_overlap(mol)
    p = total_density(dm)
    if c_nao is None:
        c_nao = nao_coefficients(mol, p, s)
    occ = np.diag(reduce(np.dot, (c_nao.T, s, p, s, c_nao)))
    charges = _f64(mol.atom_charges()).copy()
    for ia, (_b0, _b1, p0, p1) in enumerate(mol.aoslice_by_atom()):
        charges[ia] -= float(occ[p0:p1].sum())
    return [float(q) for q in charges]


def _host(arr: Any) -> np.ndarray:
    """NumPy copy of *arr* (CuPy arrays from a gpu4pyscf SCF included)."""
    get = getattr(arr, "get", None)
    if callable(get) and type(arr).__module__.startswith("cupy"):
        return _f64(get())
    return _f64(arr)


def scf_npa_charges(mf: Any) -> Optional[List[float]]:
    """NPA charges of a converged PySCF SCF object, or ``None`` on failure.

    Works for RHF/RKS and (via the α+β density) UHF/UKS/ROHF; a gpu4pyscf
    object's density is copied to the host first.
    """
    mol = getattr(mf, "mol", None)
    if mol is None:
        return None
    dm = _host(mf.make_rdm1())
    return npa_charges(mol, dm)
