"""ProcessPoolExecutor workers for the Raman-activity displacement loop.

Mirrors :mod:`quantui.freq_ir_workers` — each worker runs one displaced SCF
and returns the static polarizability tensor (3×3) via pyscf-properties.
The Frequency driver's Raman pass fans ±Δ geometry tasks out across workers
when :func:`quantui.freq_ir_workers.parallel_enabled_for_run` is true.
"""

from __future__ import annotations

from typing import Any, Dict

_RAMAN_WORKER_STATE: Dict[str, Any] = {}


def init_raman_worker(
    atom_str: str,
    basis: str,
    charge: int,
    spin: int,
    xc: str | None,
    dm0_pickle_path: str,
    omp_threads: int,
    dm0_is_unrestricted: bool,
    density_fit_used: bool,
) -> None:
    """Worker initializer — same threading discipline as IR workers."""
    import os
    import pickle

    threads = str(int(omp_threads))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["PYSCF_NUM_THREADS"] = threads

    with open(dm0_pickle_path, "rb") as fh:
        dm0 = pickle.load(fh)

    _RAMAN_WORKER_STATE.update(
        atom_str=atom_str,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        xc=xc,
        dm0=dm0,
        dm0_is_unrestricted=bool(dm0_is_unrestricted),
        density_fit_used=bool(density_fit_used),
    )


def _polarizability_module(mol: Any, dm0_is_unrestricted: bool):
    xc = _RAMAN_WORKER_STATE.get("xc")
    if xc is not None:
        if dm0_is_unrestricted:
            from pyscf.prop.polarizability import uks as pol_mod
        else:
            from pyscf.prop.polarizability import rks as pol_mod
    elif dm0_is_unrestricted:
        from pyscf.prop.polarizability import uhf as pol_mod
    else:
        from pyscf.prop.polarizability import rhf as pol_mod
    return pol_mod


def run_displaced_polarizability(coords_bohr_flat) -> list[list[float]]:
    """Run one SCF at a displaced geometry; return α as a nested 3×3 list."""
    import numpy as np
    from pyscf import dft, gto, scf

    from quantui.density_fitting import try_density_fit as _try_density_fit

    state = _RAMAN_WORKER_STATE
    coords = np.asarray(coords_bohr_flat, dtype=float).reshape(-1, 3)

    mol = gto.Mole()
    mol.atom = state["atom_str"]
    mol.basis = state["basis"]
    mol.charge = state["charge"]
    mol.spin = state["spin"]
    mol.verbose = 0
    mol.build()
    mol.set_geom_(coords, unit="Bohr")

    dm0 = state.get("dm0")
    dm0_is_unrestricted = bool(state.get("dm0_is_unrestricted"))
    if dm0 is not None:
        dm0_is_unrestricted = dm0_is_unrestricted or np.asarray(dm0).ndim == 3

    xc = state.get("xc")
    if xc is not None:
        mf = dft.UKS(mol) if dm0_is_unrestricted else dft.RKS(mol)
        mf.xc = xc
    else:
        mf = scf.UHF(mol) if dm0_is_unrestricted else scf.RHF(mol)
    mf.verbose = 0
    mf, _ = _try_density_fit(mf, enabled=bool(state.get("density_fit_used")))
    mf.kernel(dm0=dm0)

    pol_mod = _polarizability_module(mol, dm0_is_unrestricted)
    alpha = np.asarray(pol_mod.polarizability(pol_mod.Polarizability(mf)), dtype=float)
    reshaped = alpha.reshape(3, 3)
    return [[float(x) for x in row] for row in reshaped.tolist()]
