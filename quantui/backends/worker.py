"""
Headless batch worker entrypoint for SLURM jobs (M-CLUSTER2 CL2.2 seed).

Invoked from a batch script::

    python -m quantui.backends.worker --request /path/to/staging/request.json

Currently supports ``single_point`` calc types; other types return a structured
error until the worker gains parity with ``session_calc`` / calc modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .base import CalculationRequest, CalculationResult
from .registry import JobRegistry

logger = logging.getLogger(__name__)


def _write_progress(staging_dir: Path, stage: str, message: str, percent: float) -> None:
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


def _session_result_to_payload(result) -> Dict[str, Any]:
    return {
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": result.homo_lumo_gap_ev,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
    }


def run_worker_request(request_path: Path) -> CalculationResult:
    data = json.loads(request_path.read_text(encoding="utf-8"))
    request = CalculationRequest.from_dict(data)
    staging_dir = request_path.parent
    registry = JobRegistry()

    _append_log(staging_dir, f"Worker starting for {request.request_id}")
    _write_progress(staging_dir, "running", "Starting calculation", 5.0)

    if request.calc_type != "single_point":
        msg = (
            f"Batch worker does not yet support calc_type={request.calc_type!r}. "
            "Use the local backend or wait for a future CL2.2 release."
        )
        _append_log(staging_dir, msg)
        registry.update_status(
            request.request_id,
            "error",
            error={
                "code": "UNSUPPORTED_CAPABILITY",
                "user_message": msg,
                "technical_message": msg,
                "retryable": False,
            },
        )
        return CalculationResult(
            request_id=request.request_id,
            backend_id="cluster_slurm",
            status="error",
            save_type=request.calc_type,
            result_payload={},
            artifacts={"live_log": str(staging_dir / "live.log")},
            metrics={"elapsed_s": 0.0},
            error={
                "code": "UNSUPPORTED_CAPABILITY",
                "user_message": msg,
                "technical_message": msg,
                "retryable": False,
            },
        )

    from quantui.molecule import Molecule

    mol_data = request.molecule
    coords = mol_data.get("coords") or mol_data.get("coordinates")
    if not coords:
        raise ValueError("molecule dict must include 'coords' or 'coordinates'")
    molecule = Molecule(
        atoms=mol_data["atoms"],
        coordinates=coords,
        charge=int(mol_data.get("charge", request.charge)),
        multiplicity=int(mol_data.get("multiplicity", request.multiplicity)),
    )

    t0 = time.perf_counter()
    try:
        from quantui.session_calc import run_in_session

        log_path = staging_dir / "live.log"
        with open(log_path, "a", encoding="utf-8") as log_stream:
            result = run_in_session(
                molecule=molecule,
                method=request.method,
                basis=request.basis,
                progress_stream=log_stream,
                solvent=request.solvent,
            )
    except Exception as exc:  # noqa: BLE001 — worker boundary
        elapsed = time.perf_counter() - t0
        err = {
            "code": "EXECUTION_FAILED",
            "user_message": str(exc),
            "technical_message": repr(exc),
            "retryable": True,
        }
        registry.update_status(request.request_id, "error", error=err)
        _write_progress(staging_dir, "error", str(exc), 100.0)
        return CalculationResult(
            request_id=request.request_id,
            backend_id="cluster_slurm",
            status="error",
            save_type="single_point",
            result_payload={},
            artifacts={"live_log": str(staging_dir / "live.log")},
            metrics={"elapsed_s": round(elapsed, 3)},
            error=err,
        )

    elapsed = time.perf_counter() - t0
    payload = _session_result_to_payload(result)
    result_path = staging_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2))

    registry.update_status(request.request_id, "success", result_dir=str(staging_dir))
    _write_progress(staging_dir, "finalizing", "Calculation complete", 100.0)

    return CalculationResult(
        request_id=request.request_id,
        backend_id="cluster_slurm",
        status="success",
        save_type="single_point",
        result_payload=payload,
        artifacts={
            "live_log": str(staging_dir / "live.log"),
            "result_json": str(result_path),
        },
        metrics={"elapsed_s": round(elapsed, 3)},
        warnings=[] if result.converged else ["SCF did not converge"],
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
