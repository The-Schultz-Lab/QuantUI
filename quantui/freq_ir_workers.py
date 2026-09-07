"""ProcessPoolExecutor workers for the IR-intensity displacement loop.

The Frequency calculation's IR-intensity step requires ``6N`` SCFs over
finite-difference geometries (one per Cartesian displacement of each atom,
+Δ and −Δ). The default path in :mod:`quantui.freq_calc` runs them
serially with each SCF internally parallelized via BLAS + libcint OpenMP.

When the user opts in via the System Settings checkbox (persisted as
``compute.freq_parallel``) or ``QUANTUI_FREQ_PARALLEL=1`` (env overrides
settings when set) AND the host has ``>= 4`` cores AND the molecule has ``>= 2`` atoms (i.e. ``>= 6``
displacements), the freq_calc driver hands the IR loop off to a
``ProcessPoolExecutor``; the Raman polarizability loop uses the same opt-in
and worker-pool pattern via :mod:`quantui.freq_raman_workers`.
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
    checkpoint_items_dir: str | None = None,
    ecp: dict | None = None,
    density_fit: bool = False,
    scf_rescue: bool = True,
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
    checkpoint_items_dir:
        M-CHECKPOINT CHK.4.4. Directory path (plain string — a
        ``Checkpoint`` object with a live log stream is not safe to pickle
        across this process boundary) where each completed displacement's
        result is durably recorded via
        :func:`quantui.checkpoint.mark_item_done_at`. ``None`` when the run
        has no checkpoint (checkpointing is always optional — see
        :mod:`quantui.checkpoint`'s "never break a calculation" rule).
    ecp:
        The reference molecule's ``mol.ecp`` mapping (AUDIT F05) —
        ``{element: basis}`` for elements whose basis carries an effective
        core potential (e.g. LANL2DZ/def2 on heavy atoms), or ``{}``/``None``
        for an all-electron basis. Without this, a worker rebuilding the
        ``Mole`` from ``atom_str``/``basis``/``charge``/``spin`` alone would
        silently run the ECP atoms all-electron instead — a different
        Hamiltonian (more electrons, no core potential), not just numerical
        noise. See :func:`quantui.inorganic_guards.ecp_for_basis`.
    density_fit:
        AUDIT F19 — whether the *reference* SCF used density fitting
        (M-DF). The serial IR loop matches this via
        ``try_density_fit(mf, enabled=density_fit_used)`` before every
        displaced SCF; this worker previously had no such parameter at
        all, so every parallel displacement ran without density fitting
        even when the reference (and the serial fallback) used it —
        silently changing the numerical approximation, not just its
        speed, whenever the user opted into parallel IR on a fitted
        calculation.
    scf_rescue:
        AUDIT F19 — whether :func:`quantui.scf_robust.run_scf_with_rescue`
        may apply its convergence-rescue ladder for this run. The serial
        loop threads the caller's ``scf_rescue`` flag through
        (``run_scf_with_rescue(_mf_d, dm0=_dm0, rescue=scf_rescue)``); this
        worker previously hardcoded the rescue default (``True``)
        regardless of what the caller requested.
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
        checkpoint_items_dir=checkpoint_items_dir,
        ecp=ecp or {},
        density_fit=bool(density_fit),
        scf_rescue=bool(scf_rescue),
    )


def run_displaced_scf(item_id: str, coords_bohr_flat) -> Any:
    """Run one SCF at the displaced geometry; return the dipole as ndarray.

    Called by :class:`concurrent.futures.ProcessPoolExecutor` once per
    submitted displacement task. ``coords_bohr_flat`` is the displaced
    geometry packed as a flat Python list (``[x0, y0, z0, x1, y1, z1, ...]``)
    for cheap pickling — reshaped to ``(N_atoms, 3)`` inside the worker.
    ``item_id`` is the displacement's stable id (CHK.4.2, e.g. ``"d000_x_+"``)
    — used only to durably record completion (CHK.4.4) when this run has a
    checkpoint; it plays no role in the SCF itself.

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
    The checkpoint write happens *before* this function returns — if this
    worker process is killed immediately after (e.g. the whole SLURM job is
    preempted right as this task finishes), the durable record already
    exists regardless of whether the parent ever collects this Future.
    """
    import numpy as np
    from pyscf import dft, gto, scf

    state = _WORKER_STATE
    coords = np.asarray(coords_bohr_flat, dtype=float).reshape(-1, 3)

    mol = gto.Mole()
    mol.atom = state["atom_str"]
    mol.basis = state["basis"]
    # AUDIT F05 — without this, a heavy-element ECP system (e.g.
    # NaH/LANL2DZ) silently runs all-electron here: a different
    # Hamiltonian than the reference calculation, not just numerical noise.
    mol.ecp = state.get("ecp") or {}
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
    # AUDIT F19 — match the reference SCF's density-fitting choice, exactly
    # like the serial loop's `_try_density_fit(_mf_d, enabled=_density_fit_used)`
    # (freq_calc.py). Without this, enabling parallel IR silently ran every
    # displaced SCF without density fitting whenever the reference was fitted
    # — a different numerical approximation, not merely a speed difference.
    from .density_fitting import try_density_fit

    mf, _ = try_density_fit(mf, enabled=bool(state.get("density_fit", False)))
    from .scf_robust import run_scf_with_rescue

    # AUDIT F19 — honor the caller's scf_rescue choice instead of always
    # taking run_scf_with_rescue's default (True); the serial loop already
    # threads scf_rescue through as `rescue=scf_rescue`.
    run_scf_with_rescue(mf, dm0=dm0, rescue=bool(state.get("scf_rescue", True)))
    dipole = np.array(mf.dip_moment(verbose=0))

    items_dir = state.get("checkpoint_items_dir")
    if items_dir:
        from quantui.checkpoint import mark_item_done_at

        # Best-effort by construction — mark_item_done_at never raises. A
        # failed write here just means this displacement isn't resumable
        # from disk; the calc itself is unaffected either way.
        mark_item_done_at(items_dir, item_id, {"dipole": dipole.tolist()})

    return dipole


def freq_parallel_opt_in() -> bool:
    """Return whether parallel IR displacements are enabled for the next run."""
    return _freq_parallel_opt_in()


def freq_parallel_env_configured() -> bool:
    """Return True when ``QUANTUI_FREQ_PARALLEL`` is set in the environment."""
    return os.environ.get("QUANTUI_FREQ_PARALLEL") is not None


def _freq_parallel_opt_in() -> bool:
    """Whether parallel IR displacements are opted in.

    Precedence:
    1. ``QUANTUI_FREQ_PARALLEL`` env var when set (HPC job-script override).
    2. ``compute.freq_parallel`` in :mod:`quantui.user_settings` (Settings tab).
    """
    env_val = os.environ.get("QUANTUI_FREQ_PARALLEL")
    if env_val is not None:
        return _truthy(env_val)
    try:
        from quantui.user_settings import UserSettings

        return bool(UserSettings.load().compute.freq_parallel)
    except Exception:  # noqa: BLE001 — settings must never gate a calc
        return False


def parallel_enabled_for_run(
    cpu_count: int,
    displacement_count: int,
) -> bool:
    """Decide whether the freq_calc IR loop should use the parallel path.

    Centralised in this module so both the driver and the tests can
    consult the same predicate. The current rules:

    - **Opt-in**: :func:`_freq_parallel_opt_in` must be true — from the
      Settings tab checkbox (persisted) or ``QUANTUI_FREQ_PARALLEL=1`` in
      the environment (overrides settings when set). Off by default.
    - **Cores threshold**: at least 4 cores. Below that, the BLAS
      oversubscription tradeoff doesn't pay off.
    - **Displacement threshold**: at least 6 (i.e. ``>= 2`` atoms). For a
      diatomic the serial loop is 12 SCFs at most and parallel overhead
      dominates.

    When this returns ``True``, displaced SCFs run on CPU worker processes
    regardless of whether gpu4pyscf accelerated the reference SCF/Hessian.
    """
    if not _freq_parallel_opt_in():
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
