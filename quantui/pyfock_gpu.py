"""Guarded CuPy detection for PyFock GPU acceleration.

PyFock's GPU path is independent of ``gpu4pyscf``: it uses CuPy directly
inside PyFock's Numba/CUDA kernels.  Keep this probe separate from
``gpu_offload`` so installing one backend can never make the other backend
appear usable.

The probe is deliberately conservative.  A PyFock GPU run is enabled only
when CuPy imports and reports at least one usable CUDA device, and it still
honours QuantUI's persistent GPU preference and process-wide opt-out.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REASON_OK = ""
_REASON_ENV_DISABLED = "QUANTUI_DISABLE_GPU is set in the environment"
_REASON_SETTINGS_DISABLED = (
    "GPU acceleration is switched off in QuantUI settings "
    "(Status tab → Settings → GPU offload)"
)
_REASON_NOT_INSTALLED = (
    "CuPy is not installed — install the PyFock GPU extra matching your CUDA "
    "version, e.g. pip install 'quantui[pyfock-gpu-cuda12x]'"
)
_REASON_NO_DEVICE = "CuPy reports 0 CUDA devices"


def _gpu_enabled_in_settings() -> bool:
    """Read the shared persistent GPU preference without raising."""
    try:
        from quantui.user_settings import UserSettings

        return bool(UserSettings.load().compute.gpu_enabled)
    except Exception as exc:  # noqa: BLE001 — settings never gate startup
        logger.debug("could not read GPU preference, assuming enabled: %s", exc)
        return True


def _device_name(cupy_module: Any) -> str:
    properties = cupy_module.cuda.runtime.getDeviceProperties(0)
    raw_name = properties.get("name", b"GPU")
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8", errors="replace")
    return str(raw_name)


@lru_cache(maxsize=1)
def _probe_pyfock_gpu() -> tuple[bool, Optional[str], str]:
    """Return ``(available, device_name, reason)`` for the PyFock GPU path."""
    if os.environ.get("QUANTUI_DISABLE_GPU", "").strip() in ("1", "true", "True"):
        return False, None, _REASON_ENV_DISABLED
    if not _gpu_enabled_in_settings():
        return False, None, _REASON_SETTINGS_DISABLED

    try:
        import cupy as cp
    except ModuleNotFoundError:
        return False, None, _REASON_NOT_INSTALLED
    except ImportError as exc:
        logger.warning("CuPy is installed but failed to import: %s", exc)
        return False, None, f"CuPy is installed but failed to import: {exc}"
    except Exception as exc:  # noqa: BLE001 — driver failures become CPU fallback
        logger.warning("CuPy import raised %s: %s", type(exc).__name__, exc)
        return False, None, f"CuPy import raised {type(exc).__name__}: {exc}"

    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return False, None, _REASON_NO_DEVICE
        return True, _device_name(cp), _REASON_OK
    except Exception as exc:  # noqa: BLE001 — driver failures become CPU fallback
        logger.warning("CuPy device probe failed: %s", exc)
        return False, None, f"CuPy device probe failed: {exc}"


def probe_pyfock_gpu() -> tuple[bool, Optional[str], str]:
    """Return availability, device name, and an actionable failure reason."""
    return _probe_pyfock_gpu()


def resolve_pyfock_gpu(
    enabled: Optional[bool] = None,
) -> tuple[bool, Optional[str], str]:
    """Resolve whether a calculation should request PyFock GPU execution.

    ``enabled=False`` is a per-calculation hard opt-out.  ``None`` follows the
    persistent QuantUI preference.  An explicit ``True`` still falls back to
    CPU when CuPy or CUDA is unavailable; callers receive the reason so they
    can show the user what happened.
    """
    if enabled is False:
        return False, None, "GPU acceleration disabled for this calculation"
    return probe_pyfock_gpu()


def is_pyfock_gpu_available() -> tuple[bool, Optional[str]]:
    """Compatibility-friendly two-value view of :func:`probe_pyfock_gpu`."""
    available, name, _reason = probe_pyfock_gpu()
    return available, name


def clear_pyfock_gpu_probe_cache() -> None:
    """Clear the process-lifetime probe, primarily for settings/tests."""
    _probe_pyfock_gpu.cache_clear()


__all__ = [
    "clear_pyfock_gpu_probe_cache",
    "is_pyfock_gpu_available",
    "probe_pyfock_gpu",
    "resolve_pyfock_gpu",
]
