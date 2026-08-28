"""
Cluster security helpers for SLURM batch execution.

Salvaged from the legacy QuantUI archive and scoped to backend staging /
registry paths rather than the old per-user calculations layout.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantui.security import SecurityError

from . import cluster_config as cfg

_WALLTIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def validate_user_path(path: Path, allowed_root: Path) -> Path:
    """Resolve *path* and assert it lives under *allowed_root*."""
    path_str = str(path)
    if "\x00" in path_str:
        raise SecurityError("Path contains null bytes")

    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise SecurityError(
            f"Path '{resolved}' is outside allowed directory '{allowed_root}'"
        ) from exc
    return resolved


def safe_join(root: Path, *parts: str) -> Path:
    """Join *parts* onto *root* with traversal checks."""
    for part in parts:
        if ".." in part:
            raise SecurityError(f"Path component contains '..': {part!r}")
        if "\x00" in part:
            raise SecurityError("Path component contains null bytes")
        if part.startswith("/") or part.startswith("\\"):
            raise SecurityError(f"Path component is absolute: {part!r}")
        if len(part) >= 2 and part[1] == ":":
            raise SecurityError(f"Path component contains drive letter: {part!r}")

    candidate = root.joinpath(*parts)
    return validate_user_path(candidate, allowed_root=root)


def _is_valid_walltime(walltime: str) -> bool:
    if not isinstance(walltime, str):
        return False
    if not _WALLTIME_RE.match(walltime):
        return False
    parts = walltime.split(":")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    return 0 <= minutes < 60 and 0 <= seconds < 60 and hours >= 0


def validate_resources(
    cores: int,
    memory_gb: int,
    walltime: str,
) -> Dict[str, Any]:
    """Validate SLURM resource requests against configured limits."""
    errors: list[str] = []

    if not isinstance(cores, int):
        errors.append(f"cores must be an integer, got {type(cores).__name__}")
    elif cores < cfg.MIN_CORES or cores > cfg.MAX_CORES:
        errors.append(f"cores={cores} out of range [{cfg.MIN_CORES}, {cfg.MAX_CORES}]")

    if not isinstance(memory_gb, int):
        errors.append(f"memory_gb must be an integer, got {type(memory_gb).__name__}")
    elif memory_gb < cfg.MIN_MEMORY_GB or memory_gb > cfg.MAX_MEMORY_GB:
        errors.append(
            f"memory_gb={memory_gb} out of range "
            f"[{cfg.MIN_MEMORY_GB}, {cfg.MAX_MEMORY_GB}]"
        )

    if not _is_valid_walltime(walltime):
        errors.append(f"walltime='{walltime}' is not a valid HH:MM:SS string")
    elif walltime not in cfg.WALLTIME_OPTIONS:
        errors.append(
            f"walltime='{walltime}' not in allowed options: {cfg.WALLTIME_OPTIONS}"
        )

    if errors:
        raise SecurityError("Resource validation failed:\n  • " + "\n  • ".join(errors))

    return {"cores": cores, "memory_gb": memory_gb, "walltime": walltime}


def check_concurrent_job_limit(
    active_job_count: int,
    max_jobs: Optional[int] = None,
) -> None:
    limit = max_jobs if max_jobs is not None else cfg.max_concurrent_jobs()
    if active_job_count >= limit:
        raise SecurityError(
            f"Concurrent job limit reached ({active_job_count}/{limit}). "
            "Please wait for a running job to finish or cancel one."
        )


def validate_email(email: Optional[str]) -> Optional[str]:
    if email is None:
        return None
    if not isinstance(email, str) or not _EMAIL_RE.match(email):
        raise SecurityError(
            f"Invalid email address: {email!r}. "
            "Only standard RFC 5321 addresses are accepted."
        )
    return email


def validate_mail_events(events: Optional[List[str]]) -> List[str]:
    if not events:
        return list(cfg.DEFAULT_MAIL_EVENTS)
    unknown = [e for e in events if e not in cfg.ALLOWED_MAIL_EVENTS]
    if unknown:
        raise SecurityError(
            f"Unknown mail event(s): {unknown}. "
            f"Allowed values: {cfg.ALLOWED_MAIL_EVENTS}"
        )
    return list(events)
