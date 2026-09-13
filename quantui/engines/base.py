"""
Quantum-engine contract types (v0.1).

The engine contract is maintained alongside the project's development
documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import IO, Any, Dict, List, Optional, Protocol, runtime_checkable

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
    progress_stream: Optional[IO[str]] = field(default=None, repr=False, compare=False)
    checkpoint: Optional[Any] = field(default=None, repr=False, compare=False)
    warm_start: bool = True

    def to_dict(self) -> Dict[str, Any]:
        # Streams and checkpoint objects belong to the in-process execution
        # context and are deliberately excluded from the portable envelope.
        return {
            "request_id": self.request_id,
            "calc_type": self.calc_type,
            "method": self.method,
            "basis": self.basis,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "molecule": self.molecule,
            "options": self.options,
            "solvent": self.solvent,
            "warm_start": self.warm_start,
        }

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
            warm_start=bool(data.get("warm_start", True)),
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
    gpu_used: bool = False
    gpu_name: Optional[str] = None
    native_result: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        # Avoid dataclasses.asdict(): it deep-copies native_result before it can
        # be removed, which is expensive for PySCF arrays and may fail for live
        # third-party objects.
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "native_result"
        }

    def to_session_result(self) -> Any:
        """Return the rich QuantUI result used by the current app surface."""
        if self.native_result is not None:
            return self.native_result
        if self.status != "success" or self.energy_hartree is None:
            message = (self.error or {}).get(
                "user_message", "Engine calculation failed."
            )
            raise EngineError(str(message), user_message=str(message))

        from quantui.session_calc import SessionResult

        return SessionResult(
            energy_hartree=self.energy_hartree,
            homo_lumo_gap_ev=self.homo_lumo_gap_ev,
            converged=self.converged,
            n_iterations=self.n_iterations,
            method=self.method,
            basis=self.basis,
            formula=self.formula,
            density_fit=self.engine_id == "pyfock",
            gpu_used=self.gpu_used,
            gpu_name=self.gpu_name,
            scf_variant="RKS" if self.engine_id == "pyfock" else "",
            engine_id=self.engine_id,
        )


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
    """Engine adapter interface."""

    @property
    def engine_id(self) -> str: ...

    def capabilities(self) -> EngineCapabilities: ...

    def run(self, request: EngineRequest) -> EngineResult: ...
