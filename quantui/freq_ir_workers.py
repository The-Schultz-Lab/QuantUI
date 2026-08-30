"""ProcessPoolExecutor workers for the IR-intensity displacement loop.

The Frequency calculation's IR-intensity step requires ``6N`` SCFs over
finite-difference geometries (one per Cartesian displacement of each atom,
+Δ and −Δ). The default path in :mod:`quantui.freq_calc` runs them
serially with each SCF internally parallelized via BLAS + libcint OpenMP.

When the user opts in via ``QUANTUI_FREQ_PARALLEL=1`` AND the host has
``>= 4`` cores AND the molecule has ``>= 2`` atoms (i.e. ``>= 6``
displacements), the freq_calc driver hands this loop off to a
``ProcessPoolExecutor`` whose workers each call :func:`run_displaced_scf`
on one displaced geometry. Workers are **CPU-only** (no gpu4pyscf) even
when the parent run used the GPU for the reference SCF and Hessian — on
HPC nodes with one GPU and many cores, parallel CPU displacements often
beat serial GPU ones. Each worker process re-imports PySCF, rebuilds the
``gto.Mole`` from the same atom string / basis / charge / spin as the
parent, applies the displacement, and runs the SCF. The initial guess
``dm0`` is shared once per worker via a temp pickle file (the path is
passed through ``initargs``) so we don't pay per-task IPC for a 100×100
matrix.

The functions in this module are intentionally top-level (not nested in
``freq_calc.py``) because ``ProcessPoolExecutor`` requires picklable
references for both ``initializer`` and the task callable. Nested
functions cannot be pickled.

POSIX-first design note: on Linux/macOS the parent process has already
imported NumPy + PySCF by the time we spawn workers. We use
``multiprocessing.get_context("spawn")`` so each worker starts with a
fresh Python interpreter, reads the BLAS-thread env vars BEFORE NumPy is
imported, and therefore actually honors the configured thread budget.
Without ``spawn``, on Linux the default ``fork`` would inherit the
parent's NumPy thread pool and ignore any env-var changes the worker
makes.
"""

from __future__ import annotations

import os
from typing import Any, Dict

# Process-global state, populated by :func:`init_worker` once per worker.
# Kept as a module-level dict (not class state) so workers don't need to
# import any container class to access it.
_WORKER_STATE: Dict[str, Any] = {}


def init_worker(
    atom_str: str,
    basis: str,
    charge: int,
    spin: int,
    xc: str | None,
    dm0_pickle_path: str,
    omp_threads: int,
) -> None:
    """ProcessPoolExecutor worker initializer.

    Runs once per worker process. **Sets BLAS-thread env vars BEFORE
    importing NumPy** — this is the whole point of the ``spawn`` start
    method: each worker reads the env vars on its fresh interpreter
    startup, NOT on the parent's already-imported NumPy state. Then loads
    the shared initial-guess density matrix from the parent's tempfile
    into ``_WORKER_STATE`` so per-task IPC stays tiny.

    Parameters
    ----------
    atom_str:
        Pyscf-format atom string ("O 0 0 0; H 0.96 0 0; ..."). Used to
        rebuild the Mole in the worker.
    basis:
        Basis set name (e.g. ``"STO-3G"``).
    charge, spin:
        Molecular charge and 2S (spin) for the Mole.
    xc:
        DFT functional name when running a KS calculation; ``None`` for
        plain HF.
    dm0_pickle_path:
        Path to a tempfile containing the parent's converged density
        matrix as a NumPy array, used as the SCF initial guess in every
        displaced calculation. Read once here, then kept in
        ``_WORKER_STATE`` for all subsequent task calls.
    omp_threads:
        BLAS thread budget for this worker. Set as ``OMP_NUM_THREADS`` /
        ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``PYSCF_NUM_THREADS``.
    """
    # Order matters: set env vars before any NumPy / PySCF import.
    threads = str(int(omp_threads))
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["PYSCF_NUM_THREADS"] = threads

    import pickle

    with open(dm0_pickle_path, "rb") as fh:
        dm0 = pickle.load(fh)

    _WORKER_STATE.update(
        atom_str=atom_str,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        xc=xc,
        dm0=dm0,
    )


def run_displaced_scf(coords_bohr_flat) -> Any:
    """Run one SCF at the displaced geometry; return the dipole as ndarray.

    Called by :class:`concurrent.futures.ProcessPoolExecutor` once per
    submitted displacement task. ``coords_bohr_flat`` is the displaced
    geometry packed as a flat Python list (``[x0, y0, z0, x1, y1, z1, ...]``)
    for cheap pickling — reshaped to ``(N_atoms, 3)`` inside the worker.

    Uses ``_WORKER_STATE`` populated by :func:`init_worker` for the
    invariant inputs (atom string, basis, etc.) + the shared initial-guess
    density matrix.

    Returns
    -------
    np.ndarray
        Three-component dipole moment in Debye.

    Notes
    -----
    Any exception raised here propagates to the parent via the
    ``Future.result()`` call. The freq_calc driver catches such failures
    and falls back to the serial loop so the user's calc still completes.
    """
    import numpy as np
    from pyscf import dft, gto, scf

    state = _WORKER_STATE
    coords = np.asarray(coords_bohr_flat, dtype=float).reshape(-1, 3)

    mol = gto.Mole()
    mol.atom = state["atom_str"]
    mol.basis = state["basis"]
    mol.charge = state["charge"]
    mol.spin = state["spin"]
    mol.verbose = 0
    mol.build()
    mol.set_geom_(coords, unit="Bohr")

    # M5 audit fix (2026-07-14): whether this displaced SCF needs an
    # unrestricted (UHF/UKS) object is determined by the shared dm0's
    # actual shape -- (2, nao, nao) for UHF/UKS/ROHF, (nao, nao) for
    # RHF/RKS -- NOT by mol.spin == 0. Those two signals only agree when
    # the user's method choice matches the molecule's natural spin state.
    # They diverge when a user explicitly selects UHF for a closed-shell
    # molecule (mol.spin == 0 but the parent mf, and therefore dm0, is
    # still UHF-shaped): building RHF from mol.spin == 0 and then feeding
    # it the UHF-shaped dm0 raises a shape-mismatch ValueError inside
    # PySCF. Mirrors the serial-path fix in freq_calc.py.
    dm0 = state.get("dm0")
    dm0_is_unrestricted = dm0 is not None and np.asarray(dm0).ndim == 3

    xc = state.get("xc")
    if xc is not None:
        mf = dft.UKS(mol) if dm0_is_unrestricted else dft.RKS(mol)
        mf.xc = xc
    else:
        mf = scf.UHF(mol) if dm0_is_unrestricted else scf.RHF(mol)
    mf.verbose = 0
    mf.kernel(dm0=dm0)
    return np.array(mf.dip_moment(verbose=0))


def parallel_enabled_for_run(
    cpu_count: int,
    displacement_count: int,
) -> bool:
    """Decide whether the freq_calc IR loop should use the parallel path.

    Centralised in this module so both the driver and the tests can
    consult the same predicate. The current rules:

    - **Opt-in**: ``QUANTUI_FREQ_PARALLEL`` env var must be truthy
      (``"1"`` / ``"true"`` / ``"True"``). Off by default while the
      parallel path matures.
    - **Cores threshold**: at least 4 cores. Below that, the BLAS
      oversubscription tradeoff doesn't pay off.
    - **Displacement threshold**: at least 6 (i.e. ``>= 2`` atoms). For a
      diatomic the serial loop is 12 SCFs at most and parallel overhead
      dominates.

    When this returns ``True``, displaced SCFs run on CPU worker processes
    regardless of whether gpu4pyscf accelerated the reference SCF/Hessian.
    """
    if not _truthy(os.environ.get("QUANTUI_FREQ_PARALLEL", "")):
        return False
    if cpu_count < 4:
        return False
    if displacement_count < 6:
        return False
    return True


def pick_worker_count(cpu_count: int, displacement_count: int) -> int:
    """Pick a worker count that balances parallelism vs BLAS oversubscription.

    Heuristic: use half the available cores, capped by the number of
    displacement tasks. This leaves room for each worker to have ``>= 2``
    BLAS threads on common 4/8/16-core configurations:

    - 4 cores, 18 displacements → 2 workers × 2 threads each.
    - 8 cores, 60 displacements → 4 workers × 2 threads each.
    - 16 cores, 60 displacements → 8 workers × 2 threads each.
    """
    half = max(1, cpu_count // 2)
    return min(half, displacement_count)


def threads_per_worker(cpu_count: int, n_workers: int) -> int:
    """How many BLAS threads each worker process should get.

    Floors to 1 to avoid setting ``OMP_NUM_THREADS=0`` (which BLAS
    interprets as "use the runtime default" — defeating the budgeting).
    """
    if n_workers <= 0:
        return 1
    return max(1, cpu_count // n_workers)


def _truthy(value: str) -> bool:
    """Match the truthy convention used by ``QUANTUI_DISABLE_GPU`` etc."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")
