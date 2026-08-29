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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from quantui import config

from . import cluster_config as cfg
from .base import CALC_TYPES, BackendCapabilities, CalculationRequest
from .cluster_security import (
    check_concurrent_job_limit,
    check_submit_cooldown,
    validate_email,
    validate_mail_events,
    validate_resources,
)
from .registry import JobRegistry
from .slurm_errors import format_error_for_student
from .slurm_utils import (
    SlurmJobAccounting,
    estimate_slurm_resources,
    parse_sacct_accounting,
    parse_sacct_states,
    parse_slurm_job_id,
)

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
_ACTIVE_RECORD = frozenset({"queued", "pending", "running", "submitted"})


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
            supported_calc_types=CALC_TYPES,
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

        active_slurm = sum(
            1
            for record in self.registry.list_active()
            if record.backend_id == self.backend_id
        )
        check_concurrent_job_limit(active_slurm)
        check_submit_cooldown(self.registry.seconds_since_last_slurm_submit())
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
        self.registry.record_slurm_submit()
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
        status = self._squeue_status(slurm_job_id)
        if status:
            return status
        sacct_status = self._sacct_status(slurm_job_id)
        if sacct_status:
            return sacct_status
        return "UNKNOWN"

    def _squeue_status(self, slurm_job_id: str) -> str | None:
        try:
            result = subprocess.run(
                ["squeue", "-j", slurm_job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return None

        status = result.stdout.strip()
        return status.upper() if status else None

    def _sacct_status(self, slurm_job_id: str) -> str | None:
        batch = self._batch_sacct([slurm_job_id])
        return batch.get(slurm_job_id)

    def _run_sacct(self, slurm_job_ids: List[str]) -> str:
        if not slurm_job_ids:
            return ""
        id_csv = ",".join(slurm_job_ids)
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j",
                    id_csv,
                    "-n",
                    "-X",
                    "-P",
                    "--format=JobID,State,ExitCode,Elapsed",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return ""
        return result.stdout

    def _batch_sacct(self, slurm_job_ids: List[str]) -> Dict[str, str]:
        return parse_sacct_states(self._run_sacct(slurm_job_ids))

    def _batch_sacct_accounting(
        self, slurm_job_ids: List[str]
    ) -> Dict[str, SlurmJobAccounting]:
        return parse_sacct_accounting(self._run_sacct(slurm_job_ids))

    def batch_job_accounting(
        self, slurm_job_ids: List[str]
    ) -> Dict[str, SlurmJobAccounting]:
        """Merge live queue state with sacct exit code / elapsed metadata."""
        if not slurm_job_ids:
            return {}

        states = self.batch_poll_slurm_statuses(slurm_job_ids)
        sacct_rows = self._batch_sacct_accounting(slurm_job_ids)
        merged: Dict[str, SlurmJobAccounting] = {}
        for jid in slurm_job_ids:
            sacct_row = sacct_rows.get(jid)
            state = states.get(jid) or (sacct_row.state if sacct_row else "UNKNOWN")
            merged[jid] = SlurmJobAccounting(
                state=state,
                exit_code=sacct_row.exit_code if sacct_row else "",
                elapsed=sacct_row.elapsed if sacct_row else "",
            )
        return merged

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

        missing = [jid for jid in slurm_job_ids if jid not in statuses]
        if missing:
            statuses.update(self._batch_sacct(missing))

        for jid in slurm_job_ids:
            statuses.setdefault(jid, "UNKNOWN")
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
            if slurm_status in _TERMINAL_SLURM:
                mapped, error = self._terminal_registry_update(
                    record, slurm_status, datetime.now(timezone.utc)
                )
                if mapped is None:
                    continue
                if mapped != record.status and record.status not in _TERMINAL_RECORD:
                    self.registry.update_status(record.request_id, mapped, error=error)
                continue
            mapped = _map_slurm_status(slurm_status)
            if mapped != record.status and record.status not in _TERMINAL_RECORD:
                self.registry.update_status(record.request_id, mapped)

    def _terminal_registry_update(
        self, record, slurm_status: str, now: datetime
    ) -> tuple[str | None, dict | None]:
        """Map a terminal SLURM state to registry status + optional error."""
        if slurm_status not in _TERMINAL_SLURM:
            return None, None

        age_s = _record_age_seconds(record.created_at, now)
        if slurm_status == "COMPLETED" and age_s < cfg.stale_min_age_before_completed():
            return None, None

        mapped = _map_slurm_status(slurm_status)
        error: dict | None = None
        if mapped == "success":
            result_path = record.staging_path / "result.json"
            if not result_path.exists():
                mapped = "error"
                error = {
                    "code": "ARTIFACT_MISSING",
                    "user_message": (
                        "SLURM job finished but result.json was not written "
                        "to staging. Check the batch worker log."
                    ),
                    "technical_message": f"Missing {result_path}",
                    "retryable": True,
                }
        elif mapped == "error":
            error = {
                "code": "SLURM_TERMINAL",
                "user_message": f"SLURM reported job state {slurm_status}.",
                "technical_message": slurm_status,
                "retryable": True,
            }
        return mapped, error

    def reconcile_stale_records(self) -> int:
        """Mark orphaned active SLURM registry rows terminal; return count updated."""
        self.refresh_registry_statuses()
        now = datetime.now(timezone.utc)
        reconciled = 0
        stale_no_id = cfg.stale_no_slurm_id_seconds()

        for record in list(self.registry.list_active()):
            if record.backend_id != self.backend_id:
                continue

            age_s = _record_age_seconds(record.created_at, now)

            if not record.slurm_job_id:
                if age_s >= stale_no_id:
                    self._mark_stale_record(
                        record,
                        code="STALE_RECORD",
                        message=(
                            "Registry entry never received a SLURM job ID. "
                            "Remove it from Cluster Jobs, then resubmit."
                        ),
                    )
                    reconciled += 1
                continue

            if record.status.lower() not in _ACTIVE_RECORD:
                continue

            slurm_status = self.poll_slurm_status(record.slurm_job_id)
            mapped, error = self._terminal_registry_update(record, slurm_status, now)
            if mapped is None:
                continue

            self.registry.update_status(record.request_id, mapped, error=error)
            reconciled += 1

        return reconciled

    def _mark_stale_record(self, record, *, code: str, message: str) -> None:
        self.registry.update_status(
            record.request_id,
            "error",
            error={
                "code": code,
                "user_message": message,
                "technical_message": message,
                "retryable": False,
            },
        )

    def cancel(self, request_id: str) -> bool:
        record = self.registry.load(request_id)
        if record is None:
            return False
        if not record.slurm_job_id:
            self._record_cancel_failure(
                record,
                code="CANCEL_NO_SLURM_ID",
                message=(
                    "This registry row has no SLURM job ID. "
                    "Use Remove on Cluster Jobs to clear it, then resubmit."
                ),
            )
            return False

        slurm_job_id = record.slurm_job_id
        current = self.poll_slurm_status(slurm_job_id)
        if current in _TERMINAL_SLURM:
            return self._finalize_cancel_from_slurm_state(record, current)

        try:
            subprocess.run(
                ["scancel", slurm_job_id],
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
            self._record_cancel_failure(
                record,
                code="CANCEL_FAILED",
                message=(
                    f"Could not cancel SLURM job {slurm_job_id}. "
                    "Try Refresh on Cluster Jobs or run scancel manually."
                ),
            )
            return False

        deadline = time.monotonic() + cfg.cancel_confirm_timeout_seconds()
        while time.monotonic() < deadline:
            status = self._sacct_status(slurm_job_id) or self._squeue_status(
                slurm_job_id
            )
            if status in _TERMINAL_SLURM:
                return self._finalize_cancel_from_slurm_state(record, status)
            if status and status not in ("PENDING", "RUNNING", "CONFIGURING"):
                return self._finalize_cancel_from_slurm_state(record, status)
            time.sleep(cfg.CANCEL_POLL_INTERVAL_SECONDS)

        self._record_cancel_failure(
            record,
            code="CANCEL_PENDING",
            message=(
                f"Cancellation was sent for job {slurm_job_id}, but SLURM has not "
                "confirmed it yet. Refresh Cluster Jobs in a few seconds."
            ),
        )
        return False

    def _finalize_cancel_from_slurm_state(self, record, slurm_status: str) -> bool:
        now = datetime.now(timezone.utc)
        mapped, error = self._terminal_registry_update(record, slurm_status, now)
        if mapped == "cancelled":
            updated = self.registry.load(record.request_id)
            if updated is not None:
                updated.status = "cancelled"
                updated.error = None
                self.registry.save(updated)
            if record.slurm_job_id:
                self._status_cache[record.slurm_job_id] = (
                    slurm_status,
                    time.monotonic(),
                )
            return True
        if mapped is not None:
            self.registry.update_status(record.request_id, mapped, error=error)
            self._record_cancel_failure(
                record,
                code="CANCEL_NOT_NEEDED",
                message=(
                    f"SLURM job {record.slurm_job_id} is already finished "
                    f"({slurm_status})."
                ),
            )
            return False
        return False

    def _record_cancel_failure(self, record, *, code: str, message: str) -> None:
        current = self.registry.load(record.request_id)
        if current is None:
            return
        self.registry.update_status(
            record.request_id,
            current.status,
            error={
                "code": code,
                "user_message": message,
                "technical_message": message,
                "retryable": True,
            },
        )


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


def _record_age_seconds(created_at: str, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (now - created).total_seconds())
    except (TypeError, ValueError):
        return 0.0
