"""
Headless batch worker entrypoint for SLURM jobs (M-CLUSTER2 CL2.2).

Invoked from a batch script::

    python -m quantui.backends.worker --request /path/to/staging/request.json

Supports ``single_point``, ``geometry_opt``, and ``frequency`` calc types.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .base import CalculationRequest, CalculationResult
from .registry import JobRegistry
from .worker_payload import (
    freq_result_payload,
    molecule_from_request,
    optimization_result_payload,
    session_result_payload,
    write_trajectory_json,
    write_worker_result,
)

logger = logging.getLogger(__name__)

_SUPPORTED_CALC_TYPES = frozenset({"single_point", "geometry_opt", "frequency"})


def _write_progress(
    staging_dir: Path, stage: str, message: str, percent: float
) -> None:
    payload = {
        "stage": stage,
        "message": message,
        "percent": percent,
        "timestamp": time.time(),
    }
    (staging_dir / "progress.json").write_text(json.dumps(payload, indent=2))


def _append_log(staging_dir: Path, line: str) -> None:
    log_path = staging_dir / "live.log"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def _error_result(
    request: CalculationRequest,
    staging_dir: Path,
    *,
    code: str,
    message: str,
    retryable: bool,
    save_type: str,
    elapsed: float = 0.0,
) -> CalculationResult:
    err = {
        "code": code,
        "user_message": message,
        "technical_message": message,
        "retryable": retryable,
    }
    registry = JobRegistry()
    registry.update_status(request.request_id, "error", error=err)
    _write_progress(staging_dir, "error", message, 100.0)
    return CalculationResult(
        request_id=request.request_id,
        backend_id="cluster_slurm",
        status="error",
        save_type=save_type,
        result_payload={},
        artifacts={"live_log": str(staging_dir / "live.log")},
        metrics={"elapsed_s": round(elapsed, 3)},
        error=err,
    )


def _run_single_point(
    request: CalculationRequest, staging_dir: Path, log_stream
) -> Any:
    from quantui.session_calc import run_in_session

    molecule = molecule_from_request(request)
    _write_progress(staging_dir, "running", "Running single-point SCF", 20.0)
    return run_in_session(
        molecule=molecule,
        method=request.method,
        basis=request.basis,
        progress_stream=log_stream,
        solvent=request.solvent,
    )


def _run_geometry_opt(
    request: CalculationRequest, staging_dir: Path, log_stream
) -> Any:
    from quantui.optimizer import optimize_geometry

    molecule = molecule_from_request(request)
    options = request.options or {}
    fmax = float(options.get("fmax", 0.05))
    max_steps = int(options.get("max_steps", 200))
    _write_progress(staging_dir, "running", "Optimizing geometry", 15.0)
    return optimize_geometry(
        molecule=molecule,
        method=request.method,
        basis=request.basis,
        fmax=fmax,
        steps=max_steps,
        progress_stream=log_stream,
        status_label="SLURM geometry optimization",
    )


def _run_frequency(
    request: CalculationRequest, staging_dir: Path, log_stream
) -> tuple[Any, Any]:
    from quantui.freq_calc import run_freq_calc

    molecule = molecule_from_request(request)
    _write_progress(staging_dir, "running", "Running frequency analysis", 15.0)
    result = run_freq_calc(
        molecule=molecule,
        method=request.method,
        basis=request.basis,
        progress_stream=log_stream,
    )
    return result, molecule


def _build_payload(calc_type: str, outcome: Any, staging_dir: Path) -> dict[str, Any]:
    if calc_type == "single_point":
        return session_result_payload(outcome)
    if calc_type == "geometry_opt":
        traj_file = write_trajectory_json(
            staging_dir,
            outcome.trajectory,
            outcome.energies_hartree,
        )
        return optimization_result_payload(outcome, trajectory_file=traj_file)
    if calc_type == "frequency":
        result, molecule = outcome
        return freq_result_payload(result, molecule)
    raise ValueError(f"unsupported calc_type {calc_type!r}")


def run_worker_request(request_path: Path) -> CalculationResult:
    data = json.loads(request_path.read_text(encoding="utf-8"))
    request = CalculationRequest.from_dict(data)
    staging_dir = request_path.parent
    registry = JobRegistry()
    calc_type = request.calc_type

    _append_log(staging_dir, f"Worker starting for {request.request_id} ({calc_type})")
    _write_progress(staging_dir, "running", "Starting calculation", 5.0)

    if calc_type not in _SUPPORTED_CALC_TYPES:
        msg = (
            f"Batch worker does not yet support calc_type={calc_type!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_CALC_TYPES))}."
        )
        _append_log(staging_dir, msg)
        return _error_result(
            request,
            staging_dir,
            code="UNSUPPORTED_CAPABILITY",
            message=msg,
            retryable=False,
            save_type=calc_type,
        )

    runners: dict[str, Callable[..., Any]] = {
        "single_point": _run_single_point,
        "geometry_opt": _run_geometry_opt,
        "frequency": _run_frequency,
    }

    t0 = time.perf_counter()
    log_path = staging_dir / "live.log"
    try:
        with open(log_path, "a", encoding="utf-8") as log_stream:
            outcome = runners[calc_type](request, staging_dir, log_stream)
    except Exception as exc:  # noqa: BLE001 — worker boundary
        elapsed = time.perf_counter() - t0
        msg = str(exc)
        _append_log(staging_dir, msg)
        return _error_result(
            request,
            staging_dir,
            code="EXECUTION_FAILED",
            message=msg,
            retryable=True,
            save_type=calc_type,
            elapsed=elapsed,
        )

    elapsed = time.perf_counter() - t0
    payload = _build_payload(calc_type, outcome, staging_dir)
    write_worker_result(staging_dir, payload)

    registry.update_status(request.request_id, "success", result_dir=str(staging_dir))
    _write_progress(staging_dir, "finalizing", "Calculation complete", 100.0)

    warnings: list[str] = []
    converged = bool(payload.get("converged", False))
    if not converged:
        warnings.append("Calculation did not fully converge")

    return CalculationResult(
        request_id=request.request_id,
        backend_id="cluster_slurm",
        status="success",
        save_type=calc_type,
        result_payload=payload,
        artifacts={
            "live_log": str(staging_dir / "live.log"),
            "result_json": str(staging_dir / "result.json"),
        },
        metrics={"elapsed_s": round(elapsed, 3)},
        warnings=warnings,
        error=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QuantUI SLURM batch worker")
    parser.add_argument(
        "--request",
        required=True,
        type=Path,
        help="Path to request.json inside the job staging directory",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outcome = run_worker_request(args.request.expanduser().resolve())
    return 0 if outcome.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
