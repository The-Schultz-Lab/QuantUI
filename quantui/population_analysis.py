"""NPA charges and Mayer / Wiberg bond orders of a converged SCF density.

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

**Bond orders** (CHEM.3), for atoms A ≠ B:

- *Mayer* (I. Mayer, Chem. Phys. Lett. 97, 270 (1983)), closed shell:
  ``B_AB = Σ_{μ∈A} Σ_{ν∈B} (PS)_μν (PS)_νμ``; open shell:
  ``B_AB = 2 Σ_σ Σ_{μ∈A,ν∈B} (P^σ S)_μν (P^σ S)_νμ`` (σ = α, β), which
  reduces to the closed-shell form when ``P^α = P^β = P/2``.
- *Wiberg* (K. B. Wiberg, Tetrahedron 24, 1083 (1968)):
  ``W_AB = Σ_{μ∈A,ν∈B} (P^⊥_μν)²`` in an orthonormal basis, with the same
  ``2 Σ_σ`` open-shell form. QuantUI uses the **NAO basis**
  (``P^⊥ = Cᵀ S P S C``), as NBO programs do. A Löwdin-orthogonalised AO
  basis was tried and rejected: it is badly basis-set dependent (N₂ RHF:
  3.00 STO-3G, 3.52 6-31G*, 3.71 def2-TZVP; NAO basis: 3.00 / 3.03 / 3.03).

Every function is best-effort at the call site: the caller wraps it so a
failure here can never break a calculation.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, Dict, List, Optional

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


# ── Bond orders (CHEM.3) ─────────────────────────────────────────────────────

#: Atom pairs whose Mayer and Wiberg indices are both below this are not
#: stored or listed (non-bonded pairs give ~0.00–0.05; benzene's para C···C
#: is ~0.1). The same threshold is quoted under the panel's table.
BOND_ORDER_MIN = 0.10


def _spin_densities(dm: Any) -> tuple[list[np.ndarray], float]:
    """``([P], 1)`` for a closed-shell density, ``([Pα, Pβ], 2)`` otherwise."""
    arr = _f64(dm)
    if arr.ndim == 3:
        return [arr[0], arr[1]], 2.0
    return [arr], 1.0


def _atom_projector(mol: Any) -> np.ndarray:
    """``(nao, natm)`` 0/1 matrix: AO (and NAO) slot → owning atom."""
    proj = np.zeros((int(mol.nao_nr()), int(mol.natm)))
    for ia, (_b0, _b1, p0, p1) in enumerate(mol.aoslice_by_atom()):
        proj[p0:p1, ia] = 1.0
    return _f64(proj)


def mayer_bond_orders(
    mol: Any, dm: Any, *, s: Optional[np.ndarray] = None
) -> np.ndarray:
    """``(natm, natm)`` Mayer bond-order matrix (diagonal set to 0).

    A ``(2, nao, nao)`` *dm* (UHF/UKS/ROHF α, β) uses the open-shell form.
    """
    if s is None:
        s = ao_overlap(mol)
    proj = _atom_projector(mol)
    spins, fac = _spin_densities(dm)
    out = np.zeros((proj.shape[1], proj.shape[1]))
    for p in spins:
        ps = p @ s
        out += fac * (proj.T @ (ps * ps.T) @ proj)
    np.fill_diagonal(out, 0.0)
    return _f64(out)


def wiberg_bond_indices(
    mol: Any,
    dm: Any,
    *,
    s: Optional[np.ndarray] = None,
    c_nao: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``(natm, natm)`` Wiberg bond-index matrix in the NAO basis (diag 0)."""
    if s is None:
        s = ao_overlap(mol)
    if c_nao is None:
        c_nao = nao_coefficients(mol, dm, s)
    proj = _atom_projector(mol)
    spins, fac = _spin_densities(dm)
    out = np.zeros((proj.shape[1], proj.shape[1]))
    sc = s @ c_nao
    for p in spins:
        pn = sc.T @ p @ sc
        out += fac * (proj.T @ (pn * pn) @ proj)
    np.fill_diagonal(out, 0.0)
    return _f64(out)


def bond_order_pairs(
    mayer: Any,
    wiberg: Any = None,
    *,
    threshold: float = BOND_ORDER_MIN,
) -> List[Dict[str, Any]]:
    """JSON-ready ``[{"i", "j", "mayer", "wiberg"}, …]`` for pairs i < j
    (0-based) where either index reaches *threshold*; strongest first.
    ``"wiberg"`` is ``None`` when the NAO step failed."""
    m = _f64(mayer)
    w = _f64(wiberg) if wiberg is not None else None
    pairs: List[Dict[str, Any]] = []
    n = m.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mv = float(m[i, j])
            wv = float(w[i, j]) if w is not None else None
            if max(abs(mv), wv if wv is not None else 0.0) < threshold:
                continue
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "mayer": round(mv, 4),
                    "wiberg": round(wv, 4) if wv is not None else None,
                }
            )
    pairs.sort(key=lambda d: -max(abs(d["mayer"]), d["wiberg"] or 0.0))
    return pairs


def scf_population_analysis(mf: Any) -> Dict[str, Any]:
    """NPA charges and bond orders of a converged PySCF SCF object.

    ``{"npa_charges": [...] | None, "bond_orders": [...] | None}``. The NAO
    basis is built once and shared; each piece fails independently (Mayer
    needs no NAOs, so it survives an NAO failure with ``wiberg=None``).
    """
    out: Dict[str, Any] = {"npa_charges": None, "bond_orders": None}
    mol = getattr(mf, "mol", None)
    if mol is None:
        return out
    dm = _host(mf.make_rdm1())
    s = ao_overlap(mol)
    c_nao: Optional[np.ndarray] = None
    try:
        c_nao = nao_coefficients(mol, dm, s)
        out["npa_charges"] = npa_charges(mol, dm, s=s, c_nao=c_nao)
    except Exception:  # noqa: BLE001 — NAO failure must not cost Mayer
        c_nao = None
    try:
        mayer = mayer_bond_orders(mol, dm, s=s)
        wiberg = (
            wiberg_bond_indices(mol, dm, s=s, c_nao=c_nao)
            if c_nao is not None
            else None
        )
        out["bond_orders"] = bond_order_pairs(mayer, wiberg)
    except Exception:  # noqa: BLE001 — best-effort, like Mulliken
        out["bond_orders"] = None
    return out
