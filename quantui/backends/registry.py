"""
On-disk job registry for execution backends (M-CLUSTER2 CL2.1).

Each submitted calculation gets ``<jobs_root>/<request_id>.json`` plus a
companion staging directory under ``staging_root/<request_id>/`` for live logs
and progress files while the job runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import cluster_config as cfg
from .base import CalculationRequest
from .cluster_security import safe_join

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"queued", "pending", "running", "submitted"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
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
        )

    @property
    def request_obj(self) -> CalculationRequest:
        return CalculationRequest.from_dict(self.request)

    @property
    def staging_path(self) -> Path:
        return Path(self.staging_dir)

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

    def create(
        self,
        request: CalculationRequest,
        backend_id: str,
        *,
        resources: Optional[Dict[str, Any]] = None,
        status: str = "queued",
    ) -> JobRecord:
        staging = self.staging_dir_for(request.request_id)
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
