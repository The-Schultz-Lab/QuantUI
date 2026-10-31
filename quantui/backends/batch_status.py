"""
Where a batch job is, read from its job folder (M-BATCH2 B2.9 / B2.10).

A job submitted from a terminal (``quantui-batch`` on the login node, or
``quantui submit --prepare-only`` plus ``sbatch``) has a registry record with
status ``prepared`` and no SLURM job id: the record hears about the job only
when the worker finishes it. Inside an OnDemand session there is no
``squeue`` either. So the state reported here comes from the job folder:

* the launcher's job log (``.quantui-batch-jobs``, one line per ``sbatch``),
* the ``attempt-NN_job<id>/`` folder the batch script creates when the job
  starts,
* the worker's ``progress.json`` and ``result.json`` in that folder.

When the caller can see the queue (``squeue`` on a cluster node), its
states take precedence for the jobs it lists.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import JobRecord, JobRegistry

#: The launcher's per-job submission log (``quantui_batch.JOB_LOG``).
JOB_LOG = ".quantui-batch-jobs"

#: A running job's ``progress.json`` older than this (seconds) may mean the
#: job stopped (time limit, memory) without the worker writing anything.
STALE_PROGRESS_S = 30 * 60

#: States a job is still in before it finishes.
UNFINISHED_STATES = frozenset({"waiting", "running"})

_QUEUE_STATES = {
    "PENDING": "waiting",
    "CONFIGURING": "waiting",
    "RUNNING": "running",
    "COMPLETING": "running",
    "SUSPENDED": "running",
}


@dataclass
class JobView:
    """One job's state, for ``quantui jobs`` and the History panel."""

    name: str
    request_id: str
    calc_type: str
    method: str
    basis: str
    state: str  # prepared | waiting | running | done | failed
    detail: str
    percent: Optional[float] = None
    job_id: Optional[str] = None
    updated: Optional[float] = None  # epoch seconds of the last progress write

    @property
    def unfinished(self) -> bool:
        return self.state in UNFINISHED_STATES


def logged_job_ids(job_dir: Path) -> List[str]:
    """SLURM job ids in the launcher's job log, oldest first."""
    try:
        text = (job_dir / JOB_LOG).read_text(encoding="utf-8")
    except OSError:
        return []
    ids = []
    for line in text.splitlines():
        jid = line.split("\t", 1)[0].strip()
        if jid:
            ids.append(jid)
    return ids


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _ago(seconds: float) -> str:
    minutes = int(max(seconds, 0) // 60)
    if minutes < 1:
        return "just now"
    if minutes < 120:
        return f"{minutes} min ago"
    return f"{minutes // 60} h ago"


def _latest_job_id(record: JobRecord) -> Optional[str]:
    if record.job_path is not None:
        ids = logged_job_ids(record.job_path)
        if ids:
            return ids[-1]
    if record.slurm_job_id:
        return str(record.slurm_job_id)
    ids = [str(a["slurm_job_id"]) for a in record.attempts if a.get("slurm_job_id")]
    return ids[-1] if ids else None


def _attempt_for(record: JobRecord, job_id: Optional[str]) -> Optional[Path]:
    """The attempt folder of *job_id*, else the newest one (hand-run sbatch)."""
    if record.job_path is None:
        staging = Path(record.staging_dir)
        return staging if staging.is_dir() else None
    attempt = record.attempt_dir_for(job_id)
    if attempt is not None or job_id:
        return attempt
    attempts = record.attempt_dirs()
    return attempts[-1] if attempts else None


def job_view(
    record: JobRecord,
    *,
    queue: Optional[Dict[str, str]] = None,
    now: Optional[float] = None,
) -> JobView:
    """Describe *record*'s newest submission.

    *queue* maps SLURM job ids to their ``squeue`` state, when the caller
    can see the queue; without it the state is read from the job folder.
    """
    now = time.time() if now is None else now
    req = record.request_obj
    name = record.job_path.name if record.job_path is not None else record.request_id
    view = JobView(
        name=name,
        request_id=record.request_id,
        calc_type=record.calc_type,
        method=req.method,
        basis=req.basis,
        state="prepared",
        detail="not submitted",
    )
    job_id = _latest_job_id(record)
    view.job_id = job_id
    attempt = _attempt_for(record, job_id)

    if attempt is not None and (attempt / "result.json").is_file():
        view.state, view.detail, view.percent = "done", "finished", 100.0
        return view

    progress = _read_json(attempt / "progress.json") if attempt is not None else None
    if progress is not None:
        view.updated = float(progress.get("timestamp") or 0) or None
        view.percent = progress.get("percent")
        message = str(progress.get("message") or "")
        if progress.get("stage") == "error":
            view.state, view.detail = "failed", message or "the calculation failed"
            return view

    queued = (queue or {}).get(job_id or "")
    if queued is not None:
        view.state = _QUEUE_STATES.get(queued.upper(), "running")
        if view.state == "waiting":
            view.detail = f"in the queue ({queued.upper()})"
            return view

    if progress is not None:
        message = str(progress.get("message") or "running")
        age = now - view.updated if view.updated else None
        view.state = "running"
        view.detail = message + (f" · updated {_ago(age)}" if age is not None else "")
        if queued is None and age is not None and age > STALE_PROGRESS_S:
            view.detail += (
                " · no progress since: it may have stopped (time or memory "
                "limit); quantui-batch status shows what Slurm says"
            )
        return view

    if attempt is not None:
        view.state, view.detail = "running", "started, no progress yet"
        return view
    if queued is not None:
        view.detail = "running, no progress yet"
        return view
    if job_id:
        view.state = "waiting"
        view.detail = f"submitted as job {job_id}, not started yet"
        return view
    status = record.status.lower()
    if status in ("success", "error", "cancelled"):
        view.state = {"success": "done", "error": "failed"}.get(status, status)
        view.detail = status
    return view


def job_views(
    registry: JobRegistry,
    *,
    queue: Optional[Dict[str, str]] = None,
    unfinished_only: bool = False,
) -> List[JobView]:
    """Views of every SLURM record in *registry*, newest first."""
    records = sorted(
        (r for r in registry.list_all() if r.backend_id == "cluster_slurm"),
        key=lambda r: r.created_at,
        reverse=True,
    )
    views = [job_view(r, queue=queue) for r in records]
    if unfinished_only:
        views = [v for v in views if v.unfinished]
    return views
