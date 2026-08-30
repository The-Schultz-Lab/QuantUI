"""
Quantum chemistry engine registry (PySCF, PyFock, …).

Implements the v0.1 contract in ``QuantUI-development-tracking``:
``TODO/QUANTUM-ENGINE-CONTRACT.md``.

PYF.1: lazy probes + ``resolve_engine`` / ``list_engines`` only. The app still
calls ``session_calc`` directly until PYF.3 wires dispatch through this registry.
"""

from __future__ import annotations

import sys
from typing import Literal

from .base import (
    CALC_TYPES,
    EngineCapabilities,
    EngineError,
    EngineRequest,
    EngineResult,
    EngineUnavailableError,
    QuantumEngine,
    UnsupportedCapabilityError,
)
from .pyfock_engine import PyfockEngine, build_pyfock_engine, is_pyfock_available
from .pyscf_engine import PyscfEngine, build_pyscf_engine, is_pyscf_available

EnginePreference = Literal["auto", "pyscf", "pyfock"]
VALID_ENGINE_PREFERENCES: tuple[str, ...] = ("auto", "pyscf", "pyfock")
DEFAULT_ENGINE_PREFERENCE: EnginePreference = "auto"


def _installed_engines() -> list[QuantumEngine]:
    engines: list[QuantumEngine] = []
    pyscf = build_pyscf_engine()
    if pyscf is not None:
        engines.append(pyscf)
    pyfock = build_pyfock_engine()
    if pyfock is not None:
        engines.append(pyfock)
    return engines


def list_engines() -> list[EngineCapabilities]:
    """Return capability handshakes for every engine installed in this kernel."""
    return [engine.capabilities() for engine in _installed_engines()]


def _select_auto(engines: list[QuantumEngine]) -> QuantumEngine:
    by_id = {engine.engine_id: engine for engine in engines}
    if "pyscf" in by_id:
        return by_id["pyscf"]
    if "pyfock" in by_id:
        return by_id["pyfock"]
    raise EngineUnavailableError(
        "No quantum chemistry engine is installed.",
        user_message=(
            "QuantUI needs PySCF (Linux/macOS/WSL) or PyFock "
            "(pip install quantui[pyfock]) to run calculations."
        ),
    )


def _select_windows_auto(engines: list[QuantumEngine]) -> QuantumEngine:
    """Windows-native bootstrap: PyFock when PySCF is absent."""
    by_id = {engine.engine_id: engine for engine in engines}
    if sys.platform == "win32" and "pyscf" not in by_id and "pyfock" in by_id:
        return by_id["pyfock"]
    return _select_auto(engines)


def resolve_engine(preferred: EnginePreference | None = None) -> QuantumEngine:
    """Pick an engine by user preference or the v0.1 default policy.

    Resolution order:
    1. Explicit ``preferred`` of ``pyscf`` or ``pyfock`` when that engine is installed.
    2. ``auto`` (or ``None``): PySCF when both are installed; PyFock-only on
       native Windows when PySCF is missing; otherwise the sole installed engine.
    """
    engines = _installed_engines()
    choice = preferred or DEFAULT_ENGINE_PREFERENCE

    if choice == "auto":
        return _select_windows_auto(engines)

    by_id = {engine.engine_id: engine for engine in engines}
    if choice not in by_id:
        installed = ", ".join(sorted(by_id)) or "none"
        raise EngineUnavailableError(
            f"Engine {choice!r} is not installed (found: {installed}).",
            user_message=(
                f"The {choice} engine is not available in this environment. "
                "Check the Status tab for install guidance."
            ),
        )
    return by_id[choice]


__all__ = [
    "CALC_TYPES",
    "DEFAULT_ENGINE_PREFERENCE",
    "EngineCapabilities",
    "EngineError",
    "EnginePreference",
    "EngineRequest",
    "EngineResult",
    "EngineUnavailableError",
    "PyfockEngine",
    "PyscfEngine",
    "QuantumEngine",
    "UnsupportedCapabilityError",
    "VALID_ENGINE_PREFERENCES",
    "build_pyfock_engine",
    "build_pyscf_engine",
    "is_pyfock_available",
    "is_pyscf_available",
    "list_engines",
    "resolve_engine",
]
