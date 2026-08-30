"""
Quantum-engine contract types (v0.1).

See ``QuantUI-development-tracking/TODO/QUANTUM-ENGINE-CONTRACT.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

CALC_TYPES = (
    "single_point",
    "geometry_opt",
    "frequency",
    "tddft",
    "nmr",
    "pes_scan",
    "reorganization_energy",
)


@dataclass
class EngineRequest:
    """Normalized calculation envelope passed to a quantum engine adapter."""

    request_id: str
    calc_type: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    molecule: Dict[str, Any]
    options: Dict[str, Any] = field(default_factory=dict)
    solvent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngineRequest:
        return cls(
            request_id=data["request_id"],
            calc_type=data["calc_type"],
            method=data["method"],
            basis=data["basis"],
            charge=int(data["charge"]),
            multiplicity=int(data["multiplicity"]),
            molecule=dict(data["molecule"]),
            options=dict(data.get("options") or {}),
            solvent=data.get("solvent"),
        )


@dataclass
class EngineResult:
    """Normalized engine output — maps into ``SessionResult`` in PYF.3+."""

    request_id: str
    engine_id: str
    status: str  # success | error
    converged: bool
    energy_hartree: Optional[float] = None
    n_iterations: int = 0
    method: str = ""
    basis: str = ""
    formula: str = ""
    homo_lumo_gap_ev: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineCapabilities:
    """Capability handshake published by each engine for UI gating (PYF.4+)."""

    engine_id: str
    display_name: str
    supported_calc_types: tuple[str, ...]
    supported_methods: tuple[str, ...]
    supported_basis_sets: Optional[tuple[str, ...]]
    supports_solvent: bool
    supports_checkpoint_warm_start: bool
    supports_gpu: bool
    supports_post_hf: bool
    supports_orbital_export: bool
    platform_notes: str = ""
    recommended_auxbasis: str = ""
    max_atoms_recommended: Optional[int] = None
    version: str = ""


class EngineError(Exception):
    """Base class for engine-layer failures."""

    code: str = "ENGINE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.retryable = retryable

    def to_error_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "user_message": self.user_message,
            "retryable": self.retryable,
        }


class EngineUnavailableError(EngineError):
    code = "ENGINE_UNAVAILABLE"


class UnsupportedCapabilityError(EngineError):
    code = "UNSUPPORTED_CAPABILITY"


@runtime_checkable
class QuantumEngine(Protocol):
    """Engine adapter interface — PYF.1 registry only; dispatch lands in PYF.3."""

    @property
    def engine_id(self) -> str: ...

    def capabilities(self) -> EngineCapabilities: ...
