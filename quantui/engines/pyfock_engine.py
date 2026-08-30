"""
PyFock quantum engine adapter (registry + capability handshake).

PYF.1: probe availability and publish the Phase-1 DFT subset only. The actual
``Mol`` / ``Basis`` / ``DFT`` bridge ships in PYF.2; dispatch in PYF.3.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from .base import EngineCapabilities, QuantumEngine

_PYFOCK_SPEC = importlib.util.find_spec("pyfock")
_AVAILABLE = _PYFOCK_SPEC is not None

# Phase-1 subset — see QUANTUM-ENGINE-CONTRACT.md.
_PYFOCK_METHODS = ("PBE", "B3LYP", "PBE0")
_PYFOCK_BASES = ("def2-SVP", "def2-TZVP", "LANL2DZ")


def is_pyfock_available() -> bool:
    """True when PyFock is importable in the current environment."""
    return _AVAILABLE


def _pyfock_version() -> str:
    if not _AVAILABLE:
        return ""
    try:
        import importlib.metadata as im

        return im.version("pyfock")
    except Exception:  # noqa: BLE001 — best-effort version probe
        return "unknown"


class PyfockEngine:
    """Pure-Python DFT engine for native Windows and teaching single points."""

    @property
    def engine_id(self) -> str:
        return "pyfock"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id="pyfock",
            display_name="PyFock",
            supported_calc_types=("single_point",),
            supported_methods=_PYFOCK_METHODS,
            supported_basis_sets=_PYFOCK_BASES,
            supports_solvent=False,
            supports_checkpoint_warm_start=False,
            supports_gpu=False,
            supports_post_hf=False,
            supports_orbital_export=False,
            platform_notes=(
                "Phase-1 DFT single-point subset. Density fitting is always on. "
                "Install with pip install quantui[pyfock]."
            ),
            recommended_auxbasis="def2-universal-jfit",
            version=_pyfock_version(),
        )


def build_pyfock_engine() -> Optional[QuantumEngine]:
    if not _AVAILABLE:
        return None
    return PyfockEngine()
