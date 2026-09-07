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
    checkpoint_items_dir: str | None = None,
    ecp: dict | None = None,
    scf_rescue: bool = True,
) -> None:
    """Worker initializer — same threading discipline as IR workers.

    ``checkpoint_items_dir`` (M-CHECKPOINT CHK.4.4): directory path (plain
    string — see :func:`quantui.freq_ir_workers.init_worker`'s docstring for
    why not a ``Checkpoint`` object) where each completed displacement's
    polarizability is durably recorded via
    :func:`quantui.checkpoint.mark_item_done_at`. ``None`` when the run has
    no checkpoint.

    ``ecp`` (AUDIT F05): the reference molecule's ``mol.ecp`` mapping — see
    :func:`quantui.freq_ir_workers.init_worker`'s docstring. Without it, a
    heavy-element ECP system runs all-electron in this worker instead.

    ``scf_rescue`` (AUDIT F19): whether ``run_scf_with_rescue`` may apply
    its convergence-rescue ladder. Previously hardcoded to the default
    (``True``) regardless of the caller's ``scf_rescue`` choice.
    """
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
        checkpoint_items_dir=checkpoint_items_dir,
        ecp=ecp or {},
        scf_rescue=bool(scf_rescue),
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


def run_displaced_polarizability(item_id: str, coords_bohr_flat) -> list[list[float]]:
    """Run one SCF at a displaced geometry; return α as a nested 3×3 list.

    ``item_id`` (CHK.4.2, e.g. ``"d000_x_+"``) is used only to durably
    record completion (CHK.4.4) when this run has a checkpoint — see
    :func:`quantui.freq_ir_workers.run_displaced_scf`'s docstring for the
    same pattern and its crash-safety rationale.
    """
    import numpy as np
    from pyscf import dft, gto, scf

    from quantui.density_fitting import try_density_fit as _try_density_fit

    state = _RAMAN_WORKER_STATE
    coords = np.asarray(coords_bohr_flat, dtype=float).reshape(-1, 3)

    mol = gto.Mole()
    mol.atom = state["atom_str"]
    mol.basis = state["basis"]
    # AUDIT F05 — without this, a heavy-element ECP system (e.g.
    # NaH/LANL2DZ) silently runs all-electron here.
    mol.ecp = state.get("ecp") or {}
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
    from .scf_robust import run_scf_with_rescue

    # AUDIT F19 — honor the caller's scf_rescue choice instead of always
    # taking run_scf_with_rescue's default (True).
    run_scf_with_rescue(mf, dm0=dm0, rescue=bool(state.get("scf_rescue", True)))

    pol_mod = _polarizability_module(mol, dm0_is_unrestricted)
    alpha = np.asarray(pol_mod.polarizability(pol_mod.Polarizability(mf)), dtype=float)
    reshaped = alpha.reshape(3, 3)
    nested = [[float(x) for x in row] for row in reshaped.tolist()]

    items_dir = state.get("checkpoint_items_dir")
    if items_dir:
        from quantui.checkpoint import mark_item_done_at

        # Best-effort by construction — never raises. See
        # freq_ir_workers.run_displaced_scf for why this write happens here,
        # inside the worker, rather than only after the parent collects the
        # Future.
        mark_item_done_at(items_dir, item_id, {"alpha": nested})

    return nested
