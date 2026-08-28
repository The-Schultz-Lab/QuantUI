"""
SLURM batch execution backend (M-CLUSTER2 CL2.2 foundation).

Salvages polling / batch-squeue patterns from the legacy ``SLURMJobManager``
and adapts them to the execution-backend contract + on-disk registry.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from quantui import config

from . import cluster_config as cfg
from .base import BackendCapabilities, CalculationRequest
from .cluster_security import (
    check_concurrent_job_limit,
    validate_email,
    validate_mail_events,
    validate_resources,
)
from .registry import JobRegistry
from .slurm_errors import format_error_for_student
from .slurm_utils import estimate_slurm_resources, parse_slurm_job_id

logger = logging.getLogger(__name__)

_TERMINAL_SLURM = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
    }
)
_TERMINAL_RECORD = frozenset({"success", "error", "cancelled"})


class SlurmBackend:
    backend_id = "cluster_slurm"

    def __init__(
        self,
        registry: JobRegistry | None = None,
        *,
        partition: str | None = None,
        use_apptainer: bool = True,
        apptainer_image: str | None = None,
    ) -> None:
        self.registry = registry or JobRegistry()
        self.partition = partition or cfg.DEFAULT_PARTITION
        self.use_apptainer = use_apptainer
        self.apptainer_image = apptainer_image or cfg.APPTAINER_BATCH_IMAGE
        self._status_cache: Dict[str, Tuple[str, float]] = {}

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            supported_calc_types=("single_point",),  # worker parity grows in CL2.2+
            supported_methods=tuple(config.SUPPORTED_METHODS),
            supports_solvent=True,
            supports_history_artifacts=True,
            supports_live_progress=True,
            supports_cancellation=True,
            max_atoms_recommended=40,
            notes="Submits headless worker via sbatch; poll registry + staging logs.",
        )

    def dispatch(
        self,
        request: CalculationRequest,
        *,
        cores: int | None = None,
        memory_gb: int | None = None,
        walltime: str | None = None,
        depends_on: str | None = None,
        email: str | None = None,
        mail_events: list[str] | None = None,
        job_name: str | None = None,
    ) -> str:
        if not request.request_id:
            request.request_id = uuid.uuid4().hex[:12]

        check_concurrent_job_limit(len(self.registry.list_active()))
        estimates = estimate_slurm_resources(request)
        cores = cores or estimates["cores"]
        memory_gb = memory_gb or estimates["memory_gb"]
        walltime = walltime or estimates["walltime"]
        resources = validate_resources(cores, memory_gb, walltime)

        email = validate_email(email)
        resolved_events: list[str] = []
        if email is not None:
            resolved_events = validate_mail_events(mail_events)

        record = self.registry.create(
            request,
            self.backend_id,
            resources=resources,
            status="queued",
        )
        staging = record.staging_path
        request_path = staging / "request.json"
        request_path.write_text(json.dumps(request.to_dict(), indent=2))

        label = (
            job_name or f"{request.molecule.get('label', 'quantui')}_{request.method}"
        )
        label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]

        slurm_script = staging / "submit.slurm"
        self._write_slurm_script(
            slurm_script,
            job_name=label,
            request_path=request_path,
            staging_dir=staging,
            resources=resources,
            depends_on=depends_on,
            email=email,
            mail_events=resolved_events,
        )

        try:
            slurm_job_id = self._submit_to_slurm(slurm_script)
        except RuntimeError as exc:
            friendly = format_error_for_student(str(exc))
            self.registry.update_status(
                request.request_id,
                "error",
                error={
                    "code": "BACKEND_UNAVAILABLE",
                    "user_message": friendly or str(exc),
                    "technical_message": str(exc),
                    "retryable": True,
                },
            )
            raise

        self.registry.update_status(
            request.request_id,
            "submitted",
            slurm_job_id=slurm_job_id,
        )
        return request.request_id

    def _worker_command(self, request_path: Path, staging_dir: Path) -> str:
        py = sys.executable
        inner = f"{py} -m quantui.backends.worker --request {request_path}"
        if self.use_apptainer:
            image = self.apptainer_image
            return (
                f'apptainer exec --nv --bind "$HOME:$HOME" --pwd "{staging_dir}" '
                f'"{image}" {inner}'
            )
        return inner

    def _write_slurm_script(
        self,
        output_path: Path,
        *,
        job_name: str,
        request_path: Path,
        staging_dir: Path,
        resources: Dict[str, int | str],
        depends_on: str | None,
        email: str | None,
        mail_events: list[str],
    ) -> None:
        extra: list[str] = []
        if depends_on:
            extra.append(f"#SBATCH --dependency=afterok:{depends_on}")
        if email:
            extra.append(f"#SBATCH --mail-user={email}")
            events_str = ",".join(mail_events or cfg.DEFAULT_MAIL_EVENTS)
            extra.append(f"#SBATCH --mail-type={events_str}")

        optional = "\n" + "\n".join(extra) if extra else ""
        worker_command = self._worker_command(request_path, staging_dir)
        results_dir = staging_dir / "results"
        content = cfg.SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            partition=self.partition,
            cores=resources["cores"],
            memory=resources["memory_gb"],
            walltime=resources["walltime"],
            output_file=str(staging_dir / "slurm-%j.out"),
            error_file=str(staging_dir / "slurm-%j.err"),
            optional_directives=optional,
            results_dir=str(results_dir),
            worker_command=worker_command,
        )
        output_path.write_text(content)

    def _submit_to_slurm(self, script_path: Path) -> str:
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("SLURM submission timed out (>30s)") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr or exc.stdout or "sbatch failed") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("sbatch not found — SLURM is unavailable") from exc

        job_id = parse_slurm_job_id(result.stdout)
        if job_id is None:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout!r}")
        return job_id

    def poll_slurm_status(self, slurm_job_id: str) -> str:
        if not slurm_job_id:
            return "UNKNOWN"

        cached = self._status_cache.get(slurm_job_id)
        if cached is not None:
            cached_status, cached_at = cached
            if cached_status in _TERMINAL_SLURM:
                return cached_status
            if time.monotonic() - cached_at < cfg.STATUS_REFRESH_INTERVAL:
                return cached_status

        status = self._poll_single(slurm_job_id)
        self._status_cache[slurm_job_id] = (status, time.monotonic())
        return status

    def _poll_single(self, slurm_job_id: str) -> str:
        try:
            result = subprocess.run(
                ["squeue", "-j", slurm_job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "UNKNOWN"
        except subprocess.CalledProcessError:
            return "UNKNOWN"

        status = result.stdout.strip()
        return status or "COMPLETED"

    def batch_poll_slurm_statuses(self, slurm_job_ids: List[str]) -> Dict[str, str]:
        if not slurm_job_ids:
            return {}

        now = time.monotonic()
        results: Dict[str, str] = {}
        to_poll: list[str] = []

        for jid in slurm_job_ids:
            cached = self._status_cache.get(jid)
            if cached is not None:
                cached_status, cached_at = cached
                if cached_status in _TERMINAL_SLURM:
                    results[jid] = cached_status
                    continue
                if now - cached_at < cfg.STATUS_REFRESH_INTERVAL:
                    results[jid] = cached_status
                    continue
            to_poll.append(jid)

        if to_poll:
            polled = self._batch_squeue(to_poll)
            ts = time.monotonic()
            for jid, status in polled.items():
                self._status_cache[jid] = (status, ts)
            results.update(polled)

        return results

    def _batch_squeue(self, slurm_job_ids: List[str]) -> Dict[str, str]:
        id_csv = ",".join(slurm_job_ids)
        statuses: Dict[str, str] = {}
        try:
            result = subprocess.run(
                ["squeue", "-j", id_csv, "-h", "-o", "%i %T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    statuses[parts[0]] = parts[1]
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return {jid: "UNKNOWN" for jid in slurm_job_ids}

        for jid in slurm_job_ids:
            statuses.setdefault(jid, "COMPLETED")
        return statuses

    def refresh_registry_statuses(self) -> None:
        """Sync SLURM queue state into registry records for active jobs."""
        active = [
            r
            for r in self.registry.list_active()
            if r.backend_id == self.backend_id and r.slurm_job_id
        ]
        if not active:
            return

        slurm_ids = [r.slurm_job_id for r in active if r.slurm_job_id]
        statuses = self.batch_poll_slurm_statuses(slurm_ids)  # type: ignore[arg-type]

        for record in active:
            jid = record.slurm_job_id
            if not jid:
                continue
            slurm_status = statuses.get(jid, "UNKNOWN")
            mapped = _map_slurm_status(slurm_status)
            if mapped != record.status and record.status not in _TERMINAL_RECORD:
                self.registry.update_status(record.request_id, mapped)

    def cancel(self, request_id: str) -> bool:
        record = self.registry.load(request_id)
        if record is None or not record.slurm_job_id:
            return False
        try:
            subprocess.run(
                ["scancel", record.slurm_job_id],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False
        self.registry.update_status(request_id, "cancelled")
        return True


def _map_slurm_status(slurm_status: str) -> str:
    upper = slurm_status.upper()
    if upper in ("PENDING", "CONFIGURING"):
        return "pending"
    if upper == "RUNNING":
        return "running"
    if upper in _TERMINAL_SLURM:
        if upper == "COMPLETED":
            return "success"
        if upper == "CANCELLED":
            return "cancelled"
        return "error"
    return "submitted"
