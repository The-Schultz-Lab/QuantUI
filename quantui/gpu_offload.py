"""GPU offload helpers.

Wraps the runtime decision "should this SCF object be migrated to GPU?".
Detection probes ``gpu4pyscf`` + ``cupy`` for a CUDA-capable device; if
anything is missing or broken the helpers silently report "no GPU" so the
caller falls back to CPU. This means GPU integration is safe to leave
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

User opt-out, two ways:

- ``QUANTUI_DISABLE_GPU=1`` in the environment — process-wide, wins over
  everything. Useful for benchmarks, regression debugging, and "first run
  as student" comparisons, and it propagates to subprocess workers.
- ``compute.gpu_enabled = false`` in user settings — the persistent UI
  toggle (Status tab → Settings). Survives restarts.

Not every CUDA device is worth using: double-precision throughput on
consumer cards is a small fraction of their single-precision, and PySCF is
FP64 throughout. See ``is_low_fp64_device`` — QuantUI reports an advisory
but never overrides the user's choice.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Methods for which gpu4pyscf has zero or known-broken support.
#
# - ``CCSD(T)`` is documented as unsupported in the gpu4pyscf README.
# - ``MP2`` and ``CCSD`` are labelled "experimental" by gpu4pyscf and
#   were observed (2026-05-25) to fail
#   immediately after a successful RHF reference on GPU — the failure
#   fingerprint was "step completed in RHF wall time + small delta,
#   then errored", which fits the post-HF code choking on a
#   GPU-migrated mf object. Until the upstream support matures, route
#   these through CPU so calibration data accrues reliably. The RHF
#   reference still benefits from GPU because ``try_to_gpu`` only
#   short-circuits BEFORE the migration.
# - Double-hybrids would belong here too, but QuantUI doesn't expose
#   any double-hybrid methods today.
_GPU_UNSUPPORTED_METHODS: frozenset = frozenset({"MP2", "CCSD", "CCSD(T)"})

# Datacenter GPU families with usable double-precision throughput (FP64 at
# roughly 1/2 of FP32). Everything else — GeForce, RTX/Quadro workstation, the
# inference-oriented T4/L4/L40/A10/A40 line — gates FP64 to about 1/32–1/64.
#
# This matters because PySCF/gpu4pyscf SCF is FP64 throughout, so a consumer
# card can be genuinely *slower* than a many-core CPU. Measured 2026-07-29 on an
# RTX 5060 Ti: 346 GFLOP/s FP64 versus a 20-core CPU's 792 (0.44×), with real
# B3LYP single points landing at 0.44–0.91× of CPU wall time.
#
# Matching is by device-name substring. Anything unrecognised is reported as
# low-FP64 **deliberately**: consumer hardware is the common case for students,
# and a spurious advisory on a brand-new datacenter card costs far less than
# silently halving someone's throughput. This only ever drives an advisory
# string — it never changes whether GPU offload runs.
_STRONG_FP64_MARKERS: tuple = (
    "A100",
    "A30",
    "H100",
    "H200",
    "B100",
    "B200",
    "GB200",
    "GH200",
    "V100",
    "P100",
    "GV100",
    "GP100",
    "TITAN V",
)

# Reason strings for the "GPU not in use" cases, surfaced by `quantui gpu check`
# and the Status tab. Kept here so the CLI and the UI can't drift from the
# probe's actual logic (they previously re-derived it, which is how a broken
# CUDA install came to be reported as "gpu4pyscf not installed").
_REASON_OK = ""
_REASON_ENV_DISABLED = "QUANTUI_DISABLE_GPU is set in the environment"
_REASON_SETTINGS_DISABLED = (
    "GPU offload is switched off in QuantUI settings "
    "(Status tab → Settings → GPU offload)"
)
_REASON_NOT_INSTALLED = (
    "gpu4pyscf is not installed — install the extra matching your driver's "
    "CUDA version, e.g. pip install 'quantui[gpu-cuda13x]' "
    "(see README → 'Optional: GPU acceleration')"
)
_REASON_NO_DEVICE = "cupy reports 0 CUDA devices"


def is_low_fp64_device(name: Optional[str]) -> bool:
    """Return True when *name* is a GPU with crippled double-precision.

    See ``_STRONG_FP64_MARKERS`` for the rationale, the measured numbers, and
    why an unknown device is treated as low-FP64 rather than assumed fast.
    """
    if not name:
        return False
    upper = name.upper()
    return not any(marker in upper for marker in _STRONG_FP64_MARKERS)


def _gpu_enabled_in_settings() -> bool:
    """Read the persistent ``compute.gpu_enabled`` preference.

    Any failure reading settings returns True — a broken or unreadable
    settings file must never be what silently disables the GPU.
    """
    try:
        from quantui.user_settings import UserSettings

        return bool(UserSettings.load().compute.gpu_enabled)
    except Exception as exc:  # noqa: BLE001 — settings must never gate compute
        logger.debug("could not read compute.gpu_enabled, assuming True: %s", exc)
        return True


@lru_cache(maxsize=1)
def _probe_gpu() -> Tuple[bool, Optional[str], str]:
    """Probe for a usable GPU. Returns ``(available, name, reason)``.

    ``reason`` is empty when available and otherwise a human-readable
    explanation suitable for showing a user directly.

    The check sequence:

    1. ``QUANTUI_DISABLE_GPU=1`` → disabled.
    2. ``compute.gpu_enabled = false`` in user settings → disabled.
    3. ``import gpu4pyscf`` — distinguishing "package absent"
       (``ModuleNotFoundError``) from "present but its import chain is broken"
       (any other ``ImportError``, typically missing CUDA math libraries).
       These are very different problems and must not share a message.
    4. ``import cupy`` + ``cupy.cuda.runtime.getDeviceCount()``.
    5. Read device 0's properties for a friendly name.

    Failures at any step are swallowed and logged; this never raises.
    """
    if os.environ.get("QUANTUI_DISABLE_GPU", "").strip() in ("1", "true", "True"):
        return (False, None, _REASON_ENV_DISABLED)
    if not _gpu_enabled_in_settings():
        return (False, None, _REASON_SETTINGS_DISABLED)
    try:
        import gpu4pyscf  # noqa: F401
    except ModuleNotFoundError:
        # The package genuinely isn't installed — the common "user didn't opt
        # into the extra" path. Not an error worth logging at warning level.
        logger.debug("gpu4pyscf is not installed")
        return (False, None, _REASON_NOT_INSTALLED)
    except ImportError as exc:
        # gpu4pyscf IS installed but its import chain is broken — nearly always
        # missing NVIDIA CUDA math libraries (e.g. libnvJitLink.so,
        # libcublasLt.so), which the gpu4pyscf wheels do not declare as
        # dependencies. Surfacing the real exception is the whole point: the old
        # code reported this as "not installed" and sent users back to an
        # install step they had already completed.
        logger.warning("gpu4pyscf is installed but failed to import: %s", exc)
        return (
            False,
            None,
            f"gpu4pyscf is installed but failed to import: {exc}. This usually "
            "means its CUDA libraries are missing — see README → 'Optional: "
            "GPU acceleration'",
        )
    except Exception as exc:  # noqa: BLE001 — any other import-chain breakage
        logger.warning("gpu4pyscf import raised %s: %s", type(exc).__name__, exc)
        return (
            False,
            None,
            f"gpu4pyscf import raised {type(exc).__name__}: {exc}",
        )

    try:
        import cupy as _cupy

        n = int(_cupy.cuda.runtime.getDeviceCount())
        if n < 1:
            logger.debug("cupy reports 0 CUDA devices")
            return (False, None, _REASON_NO_DEVICE)
        props = _cupy.cuda.runtime.getDeviceProperties(0)
        name_raw = props.get("name", b"GPU")
        if isinstance(name_raw, bytes):
            name = name_raw.decode("utf-8", errors="replace")
        else:
            name = str(name_raw)
        return (True, name, _REASON_OK)
    except Exception as exc:  # noqa: BLE001 — fall-back to CPU on probe failure
        logger.warning("cupy device probe failed: %s", exc)
        return (False, None, f"cupy device probe failed: {exc}")


def probe_gpu() -> Tuple[bool, Optional[str], str]:
    """Return ``(available, gpu_name, reason)`` — the full probe result.

    Prefer this over :func:`is_gpu_available` when you need to tell the user
    *why* the GPU isn't being used.
    """
    return _probe_gpu()


def is_gpu_available() -> Tuple[bool, Optional[str]]:
    """Return ``(available, gpu_name)`` for the current process.

    Thin view over :func:`probe_gpu` that drops the reason, kept because it is
    the long-standing call used by the run dispatcher and the Status tab.
    Cached for the process lifetime — the answer doesn't change once the kernel
    is up. Callers that need a re-check (a settings toggle, or a test
    simulating driver loss) call ``is_gpu_available.cache_clear()``.
    """
    available, name, _reason = _probe_gpu()
    return (available, name)


# The cache lives on ``_probe_gpu`` now that two accessors share it. Forward the
# lru_cache surface so the historical ``is_gpu_available.cache_clear()`` /
# ``.cache_info()`` keep working — without this, existing callers would silently
# clear nothing and read a stale result.
is_gpu_available.cache_clear = _probe_gpu.cache_clear  # type: ignore[attr-defined]
is_gpu_available.cache_info = _probe_gpu.cache_info  # type: ignore[attr-defined]


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
