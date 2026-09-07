"""
Static Raman activities: GPU4PySCF analytical path or CPU pyscf-properties FD.

When ``is_gpu_available()`` is true and the mean-field object is
closed-shell (RHF/RKS), :func:`compute_raman_activities` calls
``gpu4pyscf.properties.raman.eval_raman_intensity`` — fully analytical
∂α/∂R via CPKS on GPU (no 6N geometry loop).  Otherwise QuantUI falls
back to pyscf-properties analytical polarizability at ±δ displaced
geometries (Approach 3), mirroring the IR dipole-derivative loop.

Normal-mode Raman activities use the Placzek invariants

    S = 45 ᾱ² + 7 γ²

(Å⁴/amu).  PySCFAD remains a future option for frequency-dependent Raman.

Reference: Porezag & Pederson, Phys. Rev. B **54**, 7830 (1996); Wilson,
Decius & Cross (invariants).
"""

from __future__ import annotations

import logging
import os
from typing import IO, Any, Callable, List, Optional, cast

import numpy as np

from quantui.config import BOHR_TO_ANGSTROM as _BOHR_TO_ANG
from quantui.gpu_offload import is_gpu_available

logger = logging.getLogger(__name__)

_DELTA_BOHR = 0.01


def raman_enabled() -> bool:
    """Return whether Raman activities should be computed during Frequency runs.

    Set ``QUANTUI_RAMAN=0`` to skip the expensive polarizability pass.
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


def _activities_from_gpu_output(raw: Any) -> List[float]:
    """Convert gpu4pyscf / CuPy Raman activity array to plain Python floats."""
    arr = np.asarray(raw)
    if np.iscomplexobj(arr):
        arr = arr.real
    return [float(x) for x in arr.reshape(-1)]


def _try_gpu_raman_activities(
    *,
    mf: Any,
    hessian: Any,
    frequencies_cm1: List[float],
    dm0_is_unrestricted: bool,
    status: Callable[[str], None],
) -> Optional[List[float]]:
    """Analytical GPU Raman via gpu4pyscf when a CUDA device is available.

    Returns ``None`` to signal fallback to the CPU pyscf-properties path.
    """
    if dm0_is_unrestricted:
        # gpu4pyscf.properties.raman.eval_raman_intensity requires RHF (closed-shell).
        return None

    available, gpu_name = is_gpu_available()
    if not available:
        return None

    try:
        from gpu4pyscf.properties import raman as gpu_raman
    except ImportError:
        logger.debug("gpu4pyscf Raman module unavailable")
        return None

    try:
        mf_gpu = mf.to_gpu() if hasattr(mf, "to_gpu") else mf
        label = gpu_name or "GPU"
        status(f"GPU Raman activities (gpu4pyscf on {label})…")
        _freqs, activities, _depol = gpu_raman.eval_raman_intensity(
            mf_gpu, hessian=hessian
        )
        out = _activities_from_gpu_output(activities)
        if len(out) == len(frequencies_cm1):
            status(f"GPU Raman activities complete (gpu4pyscf on {label}).")
            return out
        logger.warning(
            "GPU Raman mode count mismatch: %d activities vs %d frequencies",
            len(out),
            len(frequencies_cm1),
        )
    except Exception as exc:
        logger.warning("GPU Raman via gpu4pyscf failed, falling back to CPU: %s", exc)
        status("GPU Raman failed; falling back to CPU polarizability FD.")
    return None


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


def _cpu_raman_activities_fd(
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
    atom_str: str | None = None,
    scf_rescue: bool = True,
    checkpoint: Optional[Any] = None,
    resume: bool = False,
) -> List[float]:
    """CPU Raman via pyscf-properties polarizability + geometry FD.

    ``checkpoint``/``resume`` (M-CHECKPOINT CHK.4.4) mirror
    :func:`quantui.freq_calc.run_freq_calc`'s displacement checkpointing,
    under a separate named item set (``"raman_displacements"``) so this
    loop's per-displacement records can never collide with the IR loop's
    ``"freq_displacements"`` set, even though both are keyed on the exact
    same ``(atom_idx, axis, sign)`` displacements of the same molecule.
    """
    import os

    pol_mod = _polarizability_module(mf, dm0_is_unrestricted)

    from quantui.density_fitting import try_density_fit as _try_density_fit
    from quantui.freq_displacement_ids import (
        displacement_id as _disp_id,
    )
    from quantui.freq_displacement_ids import (
        parse_displacement_id as _parse_disp_id,
    )
    from quantui.gpu_offload import try_to_gpu as _try_to_gpu_inner

    _ITEM_SET_NAME = "raman_displacements"
    _xc = getattr(mf, "xc", None)
    _n_atoms = mol.natm
    _coords0 = mol.atom_coords().copy()
    _total = _n_atoms * 3 * 2

    # --- M-CHECKPOINT CHK.4: resume already-banked displacements ---
    # Single source of truth for every displacement's polarizability,
    # whichever path produced it (resumed, parallel pool, serial fallback) —
    # the assembly step below reads only from this dict, which is what
    # makes it a proper CHK.4.5 gate.
    _alphas: dict = {}
    if checkpoint is not None and resume:
        for _item_id, _payload in checkpoint.completed_items(_ITEM_SET_NAME).items():
            try:
                _key = _parse_disp_id(_item_id)
                _alphas[_key] = np.asarray(_payload["alpha"], dtype=float)
            except (ValueError, KeyError, TypeError):
                continue
    _done = len(_alphas)
    if _done and checkpoint is not None:
        try:
            checkpoint.log_resumed(
                f"{_done}/{_total} finite-difference polarizability "
                "evaluations already banked from an earlier attempt"
            )
        except Exception:  # noqa: BLE001 — provenance is never worth a crash
            pass

    status(
        "Numerical Raman activities (CPU): "
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
        from .scf_robust import run_scf_with_rescue

        run_scf_with_rescue(_mf_d, dm0=dm0, rescue=scf_rescue)
        alpha = pol_mod.polarizability(pol_mod.Polarizability(_mf_d))
        _done += 1
        status(
            "Numerical Raman activities (CPU): "
            f"{_done}/{_total} finite-difference polarizability evaluations "
            f"({_total - _done} remaining)"
        )
        return cast(np.ndarray, np.asarray(alpha, dtype=float))

    dalpha = np.zeros((_n_atoms * 3, 3, 3), dtype=float)
    _mol_v = mol.verbose
    mol.verbose = 0

    from quantui import freq_ir_workers as _ir_par
    from quantui import freq_raman_workers as _ram_par

    _cpu_count = os.cpu_count() or 1
    _use_parallel = _ir_par.parallel_enabled_for_run(
        cpu_count=_cpu_count,
        displacement_count=_total,
    )
    _ckpt_items_dir = (
        str(checkpoint.items_dir(_ITEM_SET_NAME)) if checkpoint is not None else None
    )
    _parallel_failed = False
    try:
        _remaining: list[tuple[int, int, int]] = [
            (atom_idx, ax, sign)
            for atom_idx in range(_n_atoms)
            for ax in range(3)
            for sign in (1, -1)
            if (atom_idx, ax, sign) not in _alphas
        ]
        if _use_parallel and _remaining:
            try:
                import concurrent.futures as _cf
                import multiprocessing as _mp
                import pickle as _pickle
                import tempfile as _tempfile

                _n_workers = _ir_par.pick_worker_count(_cpu_count, len(_remaining))
                _threads_each = _ir_par.threads_per_worker(_cpu_count, _n_workers)

                _tasks: list[tuple[int, int, int, list[float]]] = []
                for atom_idx, ax, sign in _remaining:
                    cp = _coords0.copy()
                    cp[atom_idx, ax] += sign * _DELTA_BOHR
                    _tasks.append((atom_idx, ax, sign, cp.flatten().tolist()))

                _dm0_handle = _tempfile.NamedTemporaryFile(
                    delete=False, suffix=".dm0.pkl"
                )
                try:
                    _pickle.dump(dm0, _dm0_handle)
                    _dm0_handle.close()

                    _atom_str = atom_str
                    if not _atom_str:
                        from pyscf import lib as _pyscf_lib

                        _atom_str = "; ".join(
                            f"{sym} {pos[0] / _pyscf_lib.param.BOHR} "
                            f"{pos[1] / _pyscf_lib.param.BOHR} "
                            f"{pos[2] / _pyscf_lib.param.BOHR}"
                            for sym, pos in mol._atom
                        )

                    _spin = mol.spin
                    _charge = mol.charge
                    _ctx = _mp.get_context("spawn")
                    with _cf.ProcessPoolExecutor(
                        max_workers=_n_workers,
                        mp_context=_ctx,
                        initializer=_ram_par.init_raman_worker,
                        initargs=(
                            _atom_str,
                            mol.basis,
                            _charge,
                            _spin,
                            _xc,
                            _dm0_handle.name,
                            _threads_each,
                            dm0_is_unrestricted,
                            density_fit_used,
                            _ckpt_items_dir,
                        ),
                    ) as _pool:
                        _futs = {
                            _pool.submit(
                                _ram_par.run_displaced_polarizability,
                                _disp_id(task[0], task[1], task[2]),
                                task[3],
                            ): task
                            for task in _tasks
                        }
                        for _fut in _cf.as_completed(_futs):
                            atom_idx, ax, sign, _coords_done = _futs[_fut]
                            _alphas[(atom_idx, ax, sign)] = np.asarray(
                                _fut.result(), dtype=float
                            )
                            _done += 1
                            status(
                                "Numerical Raman activities (parallel ×"
                                f"{_n_workers}): "
                                f"{_done}/{_total} finite-difference "
                                "polarizability evaluations "
                                f"({_total - _done} remaining)"
                            )
                finally:
                    try:
                        os.unlink(_dm0_handle.name)
                    except OSError:
                        pass
            except Exception as _par_exc:
                logger.warning(
                    "Parallel Raman computation failed (%s); falling back to serial.",
                    _par_exc,
                )
                status(
                    "Parallel Raman activities failed; falling back to serial computation."
                )
                _parallel_failed = True
                # Preserve resumed + partially-completed progress rather
                # than discarding it — mirrors freq_calc.py's IR loop.
                _done = len(_alphas)

        if not _use_parallel or _parallel_failed:
            for atom_idx, ax, sign in _remaining:
                if (atom_idx, ax, sign) in _alphas:
                    continue  # resumed, or already done by a partial parallel attempt
                alpha = _displaced_alpha(atom_idx, ax, sign)
                _alphas[(atom_idx, ax, sign)] = alpha
                if checkpoint is not None:
                    checkpoint.mark_item_done(
                        _ITEM_SET_NAME,
                        _disp_id(atom_idx, ax, sign),
                        {"alpha": alpha.tolist()},
                    )

        # --- CHK.4.5: assembly gate --- runs once, from the shared dict.
        for atom_idx in range(_n_atoms):
            for ax in range(3):
                ap = _alphas[(atom_idx, ax, 1)]
                am = _alphas[(atom_idx, ax, -1)]
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
        status("Numerical Raman activities complete (CPU).")
        return activities

    logger.warning(
        "Raman activity count mismatch: %d activities vs %d frequencies",
        len(activities),
        len(frequencies_cm1),
    )
    return []


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
    hessian: Any = None,
    atom_str: str | None = None,
    scf_rescue: bool = True,
    checkpoint: Optional[Any] = None,
    resume: bool = False,
) -> List[float]:
    """Compute static Raman activities (Å⁴/amu) per normal mode.

    Tries gpu4pyscf analytical Raman when GPU offload is available and the
    reference is closed-shell; otherwise uses CPU pyscf-properties FD.

    Returns an empty list on failure (caller keeps frequencies without Raman).
    """
    if not raman_enabled():
        status("Raman activities skipped (QUANTUI_RAMAN=0).")
        return []

    if not displacements or not frequencies_cm1:
        return []

    if hessian is not None:
        gpu_out = _try_gpu_raman_activities(
            mf=mf,
            hessian=hessian,
            frequencies_cm1=frequencies_cm1,
            dm0_is_unrestricted=dm0_is_unrestricted,
            status=status,
        )
        if gpu_out is not None:
            return gpu_out

    try:
        return _cpu_raman_activities_fd(
            mf=mf,
            mol=mol,
            scf=scf,
            dft=dft,
            displacements=displacements,
            frequencies_cm1=frequencies_cm1,
            dm0=dm0,
            dm0_is_unrestricted=dm0_is_unrestricted,
            density_fit_used=density_fit_used,
            stream=stream,
            status=status,
            atom_str=atom_str,
            scf_rescue=scf_rescue,
            checkpoint=checkpoint,
            resume=resume,
        )
    except ImportError as exc:
        logger.warning("pyscf-properties polarizability unavailable: %s", exc)
        status("Raman activities unavailable (pyscf-properties not installed).")
        return []
