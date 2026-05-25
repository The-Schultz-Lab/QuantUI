"""GPU offload helpers (M-GPU / GPU.1).

Wraps the runtime decision "should this SCF object be migrated to GPU?".
Detection probes ``gpu4pyscf`` + ``cupy`` for a CUDA-capable device; if
anything is missing or broken the helpers silently report "no GPU" so the
caller falls back to CPU. This means M-GPU integration is safe to leave
enabled by default on every platform — Windows users without CUDA, WSL
users without gpu4pyscf installed, and remote machines with broken NVIDIA
drivers all converge to the same "CPU" outcome with no exception leakage.

The companion ``log_utils._detect_gpu`` reports system-level GPU info for
the run banner (nvidia-smi name + memory). This module's job is narrower:
"can QuantUI's PySCF dispatcher offload to that GPU right now?".

Method coverage (verified against the gpu4pyscf README 2026-05):

- RHF / UHF / RKS / UKS — fully supported, ``mf.to_gpu()`` is canonical.
- MP2, CCSD — listed as experimental; ``.to_gpu()`` may succeed but the
  post-HF kernel may still fall back to CPU. ``try_to_gpu`` honours the
  user's intent (offload the SCF; let gpu4pyscf decide the rest).
- CCSD(T), double hybrids — explicitly not supported. ``try_to_gpu`` skips
  GPU for these methods so the SCF + (T) step stays on CPU.

User opt-out: set environment variable ``QUANTUI_DISABLE_GPU=1`` to force
CPU even when GPU is available. Useful for benchmarks, regression
debugging, and "first run as student" comparisons.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Methods for which gpu4pyscf has zero or known-broken support. ``CCSD(T)``
# is documented as unsupported in the gpu4pyscf README; double hybrids are
# also listed but QuantUI doesn't expose any double-hybrid methods today.
_GPU_UNSUPPORTED_METHODS: frozenset = frozenset({"CCSD(T)"})


@lru_cache(maxsize=1)
def is_gpu_available() -> Tuple[bool, Optional[str]]:
    """Return ``(available, gpu_name)`` for the current process.

    Cached for the process lifetime — the answer doesn't change once the
    kernel is up. Callers that need to force a re-check (e.g. tests that
    simulate driver loss) can call ``is_gpu_available.cache_clear()``.

    The check sequence:

    1. ``QUANTUI_DISABLE_GPU=1`` → always return ``(False, None)``.
    2. ``import gpu4pyscf`` — if the package is missing, return
       ``(False, None)``. This is the typical "user didn't install the
       optional extra" path.
    3. ``import cupy`` + ``cupy.cuda.runtime.getDeviceCount()`` — if the
       runtime says no devices are present (or cupy itself can't import),
       return ``(False, None)``.
    4. Read the first device's properties for a friendly name.

    Failures at any step are swallowed; the function never raises.
    """
    if os.environ.get("QUANTUI_DISABLE_GPU", "").strip() in ("1", "true", "True"):
        return (False, None)
    try:
        import gpu4pyscf  # noqa: F401
    except ImportError:
        return (False, None)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — fall-back to CPU on any import-chain breakage
        # Any other import-time error (broken cupy → broken gpu4pyscf
        # import-chain, mismatched cuda libs, etc.) is treated as
        # "no GPU available". Log so `quantui log tail` reveals why.
        logger.debug("gpu4pyscf import raised non-ImportError: %s", exc)
        return (False, None)

    try:
        import cupy as _cupy

        n = int(_cupy.cuda.runtime.getDeviceCount())
        if n < 1:
            return (False, None)
        props = _cupy.cuda.runtime.getDeviceProperties(0)
        name_raw = props.get("name", b"GPU")
        if isinstance(name_raw, bytes):
            name = name_raw.decode("utf-8", errors="replace")
        else:
            name = str(name_raw)
        return (True, name)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — fall-back to CPU on any cupy probe failure
        logger.debug("cupy device probe failed: %s", exc)
        return (False, None)


def try_to_gpu(mf: Any, method_upper: str) -> Tuple[Any, bool, Optional[str]]:
    """Attempt to migrate a PySCF SCF object to GPU. Safe CPU fallback.

    Parameters
    ----------
    mf:
        A constructed PySCF mean-field object (``scf.RHF(mol)``,
        ``dft.RKS(mol)``, …) BEFORE ``mf.kernel()`` is called. ``to_gpu``
        on a converged object is undefined behaviour in current gpu4pyscf.
    method_upper:
        Upper-cased method name (e.g. ``"RHF"``, ``"B3LYP"``, ``"CCSD(T)"``).
        Used only to skip GPU for methods that gpu4pyscf doesn't support.

    Returns
    -------
    tuple ``(maybe_gpu_mf, used_gpu, gpu_name)``:
        - ``maybe_gpu_mf`` is the (possibly converted) SCF object the
          caller should use for ``.kernel()``. Always usable — the
          original ``mf`` is returned unchanged on any failure.
        - ``used_gpu`` is ``True`` only when conversion succeeded.
        - ``gpu_name`` is the device name when ``used_gpu`` is True,
          ``None`` otherwise.
    """
    if method_upper in _GPU_UNSUPPORTED_METHODS:
        return (mf, False, None)
    available, gpu_name = is_gpu_available()
    if not available:
        return (mf, False, None)
    try:
        mf_gpu = mf.to_gpu()
        return (mf_gpu, True, gpu_name)
    except Exception as exc:
        # gpu4pyscf migration can fail for many reasons (unsupported method
        # variant, density-fitting requirement, basis-set quirk). On any
        # failure we fall back to CPU — the calc still runs. Log so the
        # user can `quantui log tail` and see why offload didn't happen.
        logger.warning("mf.to_gpu() migration failed, falling back to CPU: %s", exc)
        return (mf, False, None)
