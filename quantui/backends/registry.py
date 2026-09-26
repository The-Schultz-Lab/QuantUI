"""
On-disk job registry for execution backends (M-CLUSTER2 CL2.1).

Each submitted calculation gets ``<jobs_root>/<request_id>.json`` plus a
directory under ``staging_root`` for live logs, progress files and results.

Two layouts exist:

* **Job dir (M-JOBDIRS)** — ``staging_root/<job name>/`` holds ``request.json``
  and ``submit.slurm``; every run of that script creates its own
  ``attempt-NN_job<SLURM id>/`` subdirectory, so a retry never overwrites an
  earlier attempt's files. :attr:`JobRecord.staging_path` resolves to the
  attempt dir of the currently tracked SLURM job.
* **Legacy** — ``staging_root/<request_id>/`` holds everything directly.
  Records written before M-JOBDIRS have no ``job_dir`` and keep this behavior.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import cluster_config as cfg
from .base import CalculationRequest
from .cluster_security import safe_join

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"queued", "pending", "running", "submitted"})


_ATTEMPT_DIR_RE = re.compile(r"^attempt-(\d+)_job(.+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_attempt_dir_name(name: str) -> Optional[tuple[int, str]]:
    """Return ``(attempt number, SLURM job id)`` for an attempt dir name, else None."""
    m = _ATTEMPT_DIR_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


@dataclass
class JobRecord:
    request_id: str
    backend_id: str
    status: str
    calc_type: str
    request: Dict[str, Any]
    staging_dir: str
    created_at: str
    updated_at: str
    slurm_job_id: Optional[str] = None
    result_dir: Optional[str] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    # M-JOBDIRS — None on legacy records (staging_dir holds everything).
    job_dir: Optional[str] = None
    # One entry per sbatch submission made through QuantUI:
    # {"slurm_job_id", "submitted_at", "source"}. Hand-run sbatch attempts are
    # not listed here; they are found on disk (see attempt_dirs()).
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    # Names of attempt dirs (or, for legacy records, the staging dir) already
    # saved to History, so a result is never ingested twice.
    ingested_attempts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobRecord:
        return cls(
            request_id=data["request_id"],
            backend_id=data["backend_id"],
            status=data["status"],
            calc_type=data["calc_type"],
            request=dict(data["request"]),
            staging_dir=data["staging_dir"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            slurm_job_id=data.get("slurm_job_id"),
            result_dir=data.get("result_dir"),
            resources=dict(data.get("resources") or {}),
            error=data.get("error"),
            job_dir=data.get("job_dir"),
            attempts=list(data.get("attempts") or []),
            ingested_attempts=list(data.get("ingested_attempts") or []),
        )

    @property
    def request_obj(self) -> CalculationRequest:
        return CalculationRequest.from_dict(self.request)

    @property
    def job_path(self) -> Optional[Path]:
        return Path(self.job_dir) if self.job_dir else None

    def attempt_dirs(self) -> List[Path]:
        """Attempt dirs on disk, oldest first (empty for legacy records)."""
        job_path = self.job_path
        if job_path is None or not job_path.is_dir():
            return []
        found = []
        for child in job_path.iterdir():
            parsed = parse_attempt_dir_name(child.name)
            if parsed is not None and child.is_dir() and not child.is_symlink():
                found.append((parsed[0], child.name, child))
        return [path for _n, _name, path in sorted(found)]

    def attempt_dir_for(self, slurm_job_id: Optional[str]) -> Optional[Path]:
        """The attempt dir created by SLURM job *slurm_job_id*, if it exists yet."""
        if not slurm_job_id:
            return None
        for path in self.attempt_dirs():
            parsed = parse_attempt_dir_name(path.name)
            if parsed is not None and parsed[1] == str(slurm_job_id):
                return path
        return None

    @property
    def staging_path(self) -> Path:
        """Where the tracked run writes ``live.log`` / ``result.json``.

        Legacy records: the staging dir. Job-dir records: the attempt dir of
        the tracked SLURM job, or the job dir itself while that job has not
        started yet (the attempt dir is created by the batch script).
        """
        job_path = self.job_path
        if job_path is None:
            return Path(self.staging_dir)
        attempt = self.attempt_dir_for(self.slurm_job_id)
        return attempt if attempt is not None else job_path

    @property
    def live_log_path(self) -> Path:
        return self.staging_path / "live.log"

    @property
    def progress_path(self) -> Path:
        return self.staging_path / "progress.json"


class JobRegistry:
    """JSON-backed registry of submitted backend jobs."""

    def __init__(
        self,
        jobs_root: Optional[Path] = None,
        staging_root: Optional[Path] = None,
    ) -> None:
        self.jobs_root = (jobs_root or cfg.default_jobs_root()).expanduser()
        self.staging_root = (staging_root or cfg.default_staging_root()).expanduser()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, request_id: str) -> Path:
        return safe_join(self.jobs_root, f"{request_id}.json")

    def staging_dir_for(self, request_id: str) -> Path:
        path = safe_join(self.staging_root, request_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def new_job_dir(self, name: str) -> Path:
        """Create a fresh job dir named *name* (``_2``, ``_3``… on collision).

        Never reuses an existing directory: each new calculation gets its own.
        """
        n = 1
        while True:
            candidate = name if n == 1 else f"{name}_{n}"
            path = safe_join(self.staging_root, candidate)
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                n += 1
                continue
            return path

    def create(
        self,
        request: CalculationRequest,
        backend_id: str,
        *,
        resources: Optional[Dict[str, Any]] = None,
        status: str = "queued",
        job_name: Optional[str] = None,
    ) -> JobRecord:
        """Register *request*.

        With *job_name* the record gets a job dir (M-JOBDIRS layout); without
        it, a legacy ``staging_root/<request_id>/`` dir.
        """
        if job_name:
            staging = self.new_job_dir(job_name)
            job_dir: Optional[str] = str(staging)
        else:
            staging = self.staging_dir_for(request.request_id)
            job_dir = None
        now = _utc_now()
        record = JobRecord(
            request_id=request.request_id,
            backend_id=backend_id,
            status=status,
            calc_type=request.calc_type,
            request=request.to_dict(),
            staging_dir=str(staging),
            created_at=now,
            updated_at=now,
            resources=dict(resources or {}),
            job_dir=job_dir,
        )
        self.save(record)
        return record

    def save(self, record: JobRecord) -> None:
        record.updated_at = _utc_now()
        path = self._record_path(record.request_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2)
        logger.debug("Saved job record %s", record.request_id)

    def load(self, request_id: str) -> Optional[JobRecord]:
        path = self._record_path(request_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return JobRecord.from_dict(data)

    def delete(self, request_id: str) -> bool:
        path = self._record_path(request_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_all(self) -> List[JobRecord]:
        records: List[JobRecord] = []
        for path in sorted(self.jobs_root.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    records.append(JobRecord.from_dict(json.load(fh)))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping corrupt job record %s: %s", path, exc)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def list_active(self) -> List[JobRecord]:
        return [r for r in self.list_all() if r.status.lower() in _ACTIVE_STATUSES]

    def update_status(
        self,
        request_id: str,
        status: str,
        *,
        slurm_job_id: Optional[str] = None,
        result_dir: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Optional[JobRecord]:
        record = self.load(request_id)
        if record is None:
            return None
        record.status = status
        if slurm_job_id is not None:
            record.slurm_job_id = slurm_job_id
        if result_dir is not None:
            record.result_dir = result_dir
        if error is not None:
            record.error = error
        self.save(record)
        return record

    def start_attempt(
        self, request_id: str, slurm_job_id: str, *, source: str = "submit"
    ) -> Optional[JobRecord]:
        """Point *request_id* at a newly submitted SLURM job.

        Used for the first submission and for every Resubmit: the record
        tracks the new job (status ``submitted``, previous error cleared) and
        the submission is appended to :attr:`JobRecord.attempts`.
        """
        record = self.load(request_id)
        if record is None:
            return None
        record.status = "submitted"
        record.slurm_job_id = slurm_job_id
        record.error = None
        record.attempts.append(
            {
                "slurm_job_id": slurm_job_id,
                "submitted_at": _utc_now(),
                "source": source,
            }
        )
        self.save(record)
        return record

    def mark_ingested(
        self, request_id: str, attempt_key: str, *, result_dir: Optional[str] = None
    ) -> Optional[JobRecord]:
        """Record that *attempt_key* has been saved to History."""
        record = self.load(request_id)
        if record is None:
            return None
        if attempt_key not in record.ingested_attempts:
            record.ingested_attempts.append(attempt_key)
        if result_dir is not None:
            record.result_dir = result_dir
        self.save(record)
        return record

    def _submit_meta_path(self) -> Path:
        return self.jobs_root / ".slurm_submit_meta.json"

    def record_slurm_submit(self) -> None:
        """Persist the wall-clock time of the most recent successful SLURM submit."""
        payload = {"last_submit_epoch": time.time()}
        self._submit_meta_path().write_text(json.dumps(payload), encoding="utf-8")

    def seconds_since_last_slurm_submit(self) -> Optional[float]:
        path = self._submit_meta_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            epoch = data.get("last_submit_epoch")
            if epoch is None:
                return None
            return time.time() - float(epoch)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None
