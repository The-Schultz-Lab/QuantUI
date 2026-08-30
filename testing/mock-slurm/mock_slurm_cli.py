#!/usr/bin/env python3
"""Cross-platform mock SLURM CLI for QuantUI backend tests."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _state_dir() -> Path:
    raw = os.environ.get("MOCK_SLURM_STATE")
    if not raw:
        print("MOCK_SLURM_STATE must be set", file=sys.stderr)
        sys.exit(1)
    return Path(raw)


def _iter_jobs(state_dir: Path, job_ids: list[str] | None = None):
    for path in sorted(state_dir.glob("*.json")):
        if path.name == "config.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        job_id = str(data.get("job_id", path.stem))
        if job_ids and job_id not in job_ids:
            continue
        yield job_id, data, path


def cmd_squeue(argv: list[str]) -> int:
    fmt = "%T"
    job_filter = ""
    i = 0
    while i < len(argv):
        if argv[i] == "-o" and i + 1 < len(argv):
            fmt = argv[i + 1]
            i += 2
            continue
        if argv[i] == "-j" and i + 1 < len(argv):
            job_filter = argv[i + 1]
            i += 2
            continue
        i += 1

    ids = [j.strip() for j in job_filter.split(",") if j.strip()] or None
    state_dir = _state_dir()
    for job_id, data, _path in _iter_jobs(state_dir, ids):
        status = data.get("status", "PENDING")
        if status == "COMPLETED":
            continue
        if "%i" in fmt and "%T" in fmt:
            print(f"{job_id} {status}")
        else:
            print(status)
    return 0


def cmd_sacct(argv: list[str]) -> int:
    job_filter = ""
    i = 0
    while i < len(argv):
        if argv[i] == "-j" and i + 1 < len(argv):
            job_filter = argv[i + 1]
            i += 2
            continue
        i += 1

    ids = [j.strip() for j in job_filter.split(",") if j.strip()] or None
    state_dir = _state_dir()
    for job_id, data, _path in _iter_jobs(state_dir, ids):
        status = str(data.get("status", "PENDING")).upper()
        exit_code = str(data.get("exit_code", "0:0"))
        elapsed = str(data.get("elapsed", "00:00:01"))
        print(f"{job_id}|{status}|{exit_code}|{elapsed}")
    return 0


def cmd_scancel(argv: list[str]) -> int:
    if not argv:
        print("scancel: error: job id required", file=sys.stderr)
        return 1
    job_id = argv[0]
    path = _state_dir() / f"{job_id}.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "CANCELLED"
    path.write_text(json.dumps(data), encoding="utf-8")
    return 0


def cmd_sbatch(argv: list[str]) -> int:
    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    script_path = ""
    for arg in argv:
        if Path(arg).is_file():
            script_path = arg
            break
    if not script_path:
        print("sbatch: error: No job script specified", file=sys.stderr)
        return 1

    job_id = str(10000 + len(list(state_dir.glob("*.json"))))
    content = Path(script_path).read_text(encoding="utf-8")
    cores = memory = walltime = None
    for line in content.splitlines():
        m = re.search(r"#SBATCH\s+--ntasks=(\d+)", line)
        if m:
            cores = int(m.group(1))
        m = re.search(r"#SBATCH\s+--mem=(\d+)G", line)
        if m:
            memory = int(m.group(1))
        m = re.search(r"#SBATCH\s+--time=(\S+)", line)
        if m:
            walltime = m.group(1)

    limits_path = state_dir / "config.json"
    max_cores = 32
    max_mem = 128
    if limits_path.exists():
        cfg = json.loads(limits_path.read_text(encoding="utf-8"))
        max_cores = cfg.get("max_cores", max_cores)
        max_mem = cfg.get("max_memory_gb", max_mem)

    if cores is not None and cores > max_cores:
        print(f"sbatch: error: CPU count {cores} exceeds limit", file=sys.stderr)
        return 1
    if memory is not None and memory > max_mem:
        print(f"sbatch: error: Memory {memory}G exceeds limit", file=sys.stderr)
        return 1

    payload = {
        "job_id": job_id,
        "status": "PENDING",
        "script": script_path,
        "submitted_at": os.environ.get("MOCK_SLURM_NOW", "0"),
    }
    (state_dir / f"{job_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Submitted batch job {job_id}")

    if os.environ.get("MOCK_SLURM_AUTO_COMPLETE", "0") == "1":
        data = json.loads((state_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        data["status"] = "COMPLETED"
        (state_dir / f"{job_id}.json").write_text(json.dumps(data), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mock_slurm_cli")
    parser.add_argument("command", choices=["squeue", "sacct", "scancel", "sbatch"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    argv = ns.args
    if argv and argv[0] == "--":
        argv = argv[1:]

    if ns.command == "squeue":
        return cmd_squeue(argv)
    if ns.command == "sacct":
        return cmd_sacct(argv)
    if ns.command == "scancel":
        return cmd_scancel(argv)
    return cmd_sbatch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
