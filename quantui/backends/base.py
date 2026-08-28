"""
Execution-backend contract types (v0.1).

See ``QuantUI-development-tracking/TODO/EXECUTION-BACKEND-CONTRACT.md``.
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
)


@dataclass
class CalculationRequest:
    """Normalized calculation envelope submitted to a backend."""

    request_id: str
    calc_type: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    molecule: Dict[str, Any]
    options: Dict[str, Any] = field(default_factory=dict)
    solvent: Optional[str] = None
    run_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationRequest":
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
            run_context=dict(data.get("run_context") or {}),
        )


@dataclass
class ProgressEvent:
    request_id: str
    backend_id: str
    stage: str
    message: str
    percent: Optional[float]
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalculationResult:
    request_id: str
    backend_id: str
    status: str  # success | error
    save_type: str
    result_payload: Dict[str, Any]
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    supported_calc_types: tuple[str, ...]
    supported_methods: tuple[str, ...]
    supports_solvent: bool
    supports_history_artifacts: bool
    supports_live_progress: bool
    supports_cancellation: bool
    max_atoms_recommended: Optional[int] = None
    notes: str = ""


@runtime_checkable
class ComputeBackend(Protocol):
    """Backend interface consumed by the app layer (future CL2.3 wiring)."""

    @property
    def backend_id(self) -> str: ...

    def capabilities(self) -> BackendCapabilities: ...

    def dispatch(self, request: CalculationRequest) -> str:
        """Submit *request*; return ``request_id`` for registry polling."""
        ...
