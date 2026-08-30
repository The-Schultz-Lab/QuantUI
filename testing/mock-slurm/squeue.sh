#!/usr/bin/env bash
# Mock SLURM squeue for QuantUI backend tests.
set -euo pipefail

STATE_DIR="${MOCK_SLURM_STATE:?MOCK_SLURM_STATE must be set}"

format="%T"
job_filter=""

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    -o)
      i=$((i + 1))
      format="${args[$i]}"
      ;;
    -j)
      i=$((i + 1))
      job_filter="${args[$i]}"
      ;;
  esac
  i=$((i + 1))
done

python3 - "$STATE_DIR" "$format" "$job_filter" <<'PY'
import json
import sys
from pathlib import Path

state_dir, fmt, job_filter = sys.argv[1:4]
ids = []
if job_filter:
    ids = [j.strip() for j in job_filter.split(",") if j.strip()]

for path in sorted(Path(state_dir).glob("*.json")):
    if path.name == "config.json":
        continue
    data = json.loads(path.read_text())
    job_id = str(data.get("job_id", path.stem))
    if ids and job_id not in ids:
        continue
    status = data.get("status", "PENDING")
    if status == "COMPLETED":
        continue
    if "%i" in fmt and "%T" in fmt:
        print(f"{job_id} {status}")
    else:
        print(status)
PY
