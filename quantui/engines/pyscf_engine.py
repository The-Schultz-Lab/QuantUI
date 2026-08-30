"""
PySCF quantum engine adapter (registry + capability handshake).

PYF.1: probe availability and publish capabilities only. Calculation dispatch
remains on the existing ``session_calc`` path until PYF.3 wires the registry.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from quantui import config

from .base import CALC_TYPES, EngineCapabilities, QuantumEngine

_PYSCF_SPEC = importlib.util.find_spec("pyscf")
_AVAILABLE = _PYSCF_SPEC is not None


def is_pyscf_available() -> bool:
    """True when PySCF is importable in the current environment."""
    return _AVAILABLE


def _pyscf_version() -> str:
    if not _AVAILABLE:
        return ""
    try:
        import importlib.metadata as im

        return im.version("pyscf")
    except Exception:  # noqa: BLE001 — best-effort version probe
        return "unknown"


class PyscfEngine:
    """Canonical QuantUI engine wherever PySCF is installed."""

    @property
    def engine_id(self) -> str:
        return "pyscf"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id="pyscf",
            display_name="PySCF",
            supported_calc_types=CALC_TYPES,
            supported_methods=tuple(config.SUPPORTED_METHODS),
            supported_basis_sets=tuple(config.SUPPORTED_BASIS_SETS),
            supports_solvent=True,
            supports_checkpoint_warm_start=True,
            supports_gpu=True,
            supports_post_hf=True,
            supports_orbital_export=True,
            platform_notes=(
                "Full QuantUI feature set on Linux, macOS, and WSL. "
                "Native Windows requires WSL or Apptainer."
            ),
            version=_pyscf_version(),
        )


def build_pyscf_engine() -> Optional[QuantumEngine]:
    if not _AVAILABLE:
        return None
    return PyscfEngine()
