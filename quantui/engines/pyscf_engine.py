"""
PySCF quantum engine adapter (registry + capability handshake).

PYF.1: probe availability and publish capabilities only. Calculation dispatch
remains on the existing ``session_calc`` path until PYF.3 wires the registry.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from quantui import config

from .base import (
    CALC_TYPES,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    QuantumEngine,
    UnsupportedCapabilityError,
)

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

    def run(self, request: EngineRequest) -> EngineResult:
        """Adapt the established PySCF paths to the shared engine contract."""
        if request.calc_type == "geometry_opt":
            return self._run_geometry_opt(request)
        if request.calc_type != "single_point":
            raise UnsupportedCapabilityError(
                f"Shared engine dispatch does not yet own {request.calc_type!r}.",
                user_message=(
                    "This calculation type still uses QuantUI's established "
                    "PySCF workflow."
                ),
            )

        from quantui.molecule import Molecule
        from quantui.session_calc import run_in_session

        molecule = Molecule(
            atoms=list(request.molecule["atoms"]),
            coordinates=[list(c) for c in request.molecule["coordinates"]],
            charge=request.charge,
            multiplicity=request.multiplicity,
        )
        native = run_in_session(
            molecule=molecule,
            method=request.method,
            basis=request.basis,
            verbose=int(request.options.get("verbose", 4)),
            progress_stream=request.progress_stream,
            solvent=request.solvent,
            checkpoint=request.checkpoint,
            warm_start=request.warm_start,
            scf_rescue=bool(request.options.get("scf_rescue", True)),
        )
        native.engine_id = self.engine_id
        return EngineResult(
            request_id=request.request_id,
            engine_id=self.engine_id,
            status="success",
            converged=native.converged,
            energy_hartree=native.energy_hartree,
            n_iterations=native.n_iterations,
            method=native.method,
            basis=native.basis,
            formula=native.formula,
            homo_lumo_gap_ev=native.homo_lumo_gap_ev,
            native_result=native,
        )

    def _run_geometry_opt(self, request: EngineRequest) -> EngineResult:
        from quantui.molecule import Molecule
        from quantui.optimizer import optimize_geometry

        molecule = Molecule(
            atoms=list(request.molecule["atoms"]),
            coordinates=[list(c) for c in request.molecule["coordinates"]],
            charge=request.charge,
            multiplicity=request.multiplicity,
        )
        native = optimize_geometry(
            molecule=molecule,
            method=request.method,
            basis=request.basis,
            fmax=float(request.options.get("fmax", 0.05)),
            steps=int(request.options.get("steps", 200)),
            progress_stream=request.progress_stream,
            expected_steps=request.options.get("expected_steps"),
            checkpoint=request.checkpoint,
            resume=bool(request.options.get("resume", False)),
            scf_rescue=bool(request.options.get("scf_rescue", True)),
            engine_id=self.engine_id,
        )
        return EngineResult(
            request_id=request.request_id,
            engine_id=self.engine_id,
            status="success",
            converged=native.converged,
            energy_hartree=native.energy_hartree,
            n_iterations=native.n_steps,
            method=native.method,
            basis=native.basis,
            formula=native.formula,
            native_result=native,
        )


def build_pyscf_engine() -> Optional[QuantumEngine]:
    if not _AVAILABLE:
        return None
    return PyscfEngine()
