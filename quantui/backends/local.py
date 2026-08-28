"""
Local in-kernel execution backend (baseline).

The app still calls ``session_calc.run_in_session`` directly from ``_do_run``.
This backend exposes capabilities and registry helpers for the future
``ComputeBackend`` seam without changing default runtime behaviour (CL2.1).
"""

from __future__ import annotations

from quantui import config

from .base import BackendCapabilities, CalculationRequest
from .registry import JobRegistry


class LocalBackend:
    backend_id = "local_pyscf"

    def __init__(self, registry: JobRegistry | None = None) -> None:
        self.registry = registry or JobRegistry()

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            supported_calc_types=(
                "single_point",
                "geometry_opt",
                "frequency",
                "tddft",
                "nmr",
                "pes_scan",
            ),
            supported_methods=tuple(config.SUPPORTED_METHODS),
            supports_solvent=True,
            supports_history_artifacts=True,
            supports_live_progress=True,
            supports_cancellation=True,
            notes="Runs in the current Jupyter kernel via session_calc.",
        )

    def register_local_run(self, request: CalculationRequest) -> str:
        """
        Record a local run in the registry (optional telemetry for CL2.3).

        Does not execute the calculation — callers still invoke session_calc.
        """
        self.registry.create(request, self.backend_id, status="running")
        return request.request_id

    def dispatch(self, request: CalculationRequest) -> str:
        """Reserved for app-layer wiring in CL2.3."""
        return self.register_local_run(request)
