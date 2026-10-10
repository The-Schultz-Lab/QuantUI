"""
Start a batch job from another job's final geometry (``quantui-batch --from``).

A chained job (say a frequency calculation after a geometry optimization) is
usually queued before the job it depends on has finished, so its starting
geometry cannot be written into ``request.json`` at submit time. Instead the
request names the source job folder in ``run_context["geometry_from"]``, and
the worker calls :func:`final_geometry` when the job starts, after Slurm's
``afterok`` dependency has let it run.

Only geometries a job actually optimized are offered: the last step of a
``geometry_opt`` trajectory, the last step of a ``preopt_before_run``
trajectory, or the molecule a ``frequency`` result was computed at. A PES
scan's ``trajectory.json`` holds scan points, not a minimum, so it is never
used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import parse_attempt_dir_name


def finished_attempts(job_dir: Path) -> List[Path]:
    """Attempt dirs with a ``result.json``, newest first.

    A job dir from before per-attempt folders holds its outputs directly; it
    counts as one attempt when it has a ``result.json`` of its own.
    """
    found = []
    if job_dir.is_dir():
        for child in job_dir.iterdir():
            parsed = parse_attempt_dir_name(child.name)
            if (
                parsed is not None
                and child.is_dir()
                and not child.is_symlink()
                and (child / "result.json").is_file()
            ):
                found.append((parsed[0], child))
    attempts = [path for _n, path in sorted(found, reverse=True)]
    if not attempts and (job_dir / "result.json").is_file():
        attempts = [job_dir]
    return attempts


def _final_molecule(path: Path) -> Optional[Dict[str, Any]]:
    """``final_molecule`` from a geometry_opt result.json (B2.3), if recorded.

    Results written before B2.3 have none; the caller falls back to the
    trajectory's last step.
    """
    try:
        mol = json.loads(path.read_text(encoding="utf-8"))["final_molecule"]
        return {"atoms": list(mol["atoms"]), "coords": mol["coords"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _last_step(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = data.get("steps") or []
        if not steps:
            return None
        return {"atoms": list(data["atoms"]), "coords": steps[-1]["coords"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def geometry_from_attempt(attempt: Path, calc_type: str) -> Optional[Dict[str, Any]]:
    """The optimized geometry one finished attempt produced, if it has one."""
    if calc_type == "geometry_opt":
        geo = _final_molecule(attempt / "result.json") or _last_step(
            attempt / "trajectory.json"
        )
        if geo:
            return geo
    geo = _last_step(attempt / "preopt_trajectory.json")
    if geo:
        return geo
    if calc_type == "frequency":
        try:
            result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            mol = result["spectra"]["molecule"]
            return {"atoms": list(mol["atoms"]), "coords": mol["coords"]}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


def final_geometry(job_dir: Path) -> Dict[str, Any]:
    """``{"atoms", "coords", "source"}`` from *job_dir*'s newest usable attempt.

    Raises ``ValueError`` with a reason a student can act on when the job has
    no finished attempt, or did not optimize its geometry.
    """
    job_dir = Path(job_dir)
    try:
        request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{job_dir} is not a QuantUI job folder ({exc})") from exc
    calc_type = str(request.get("calc_type", ""))
    attempts = finished_attempts(job_dir)
    if not attempts:
        raise ValueError(f"{job_dir.name} has no finished attempt with a result")
    for attempt in attempts:
        geo = geometry_from_attempt(attempt, calc_type)
        if geo:
            geo["source"] = f"{job_dir.name}/{attempt.name}"
            return geo
    raise ValueError(
        f"{job_dir.name} ({calc_type}) did not optimize its geometry; start from "
        "a geometry_opt job, a frequency job, or a job run with --preopt"
    )
