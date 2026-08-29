#!/usr/bin/env bash
# Mock SLURM sacct for QuantUI backend tests.
set -euo pipefail

STATE_DIR="${MOCK_SLURM_STATE:?MOCK_SLURM_STATE must be set}"

job_filter=""

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    -j)
      i=$((i + 1))
      job_filter="${args[$i]}"
      ;;
  esac
  i=$((i + 1))
done

python3 - "$STATE_DIR" "$job_filter" <<'PY'
import json
import sys
from pathlib import Path

state_dir, job_filter = sys.argv[1:3]
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
    status = str(data.get("status", "PENDING")).upper()
    exit_code = str(data.get("exit_code", "0:0"))
    elapsed = str(data.get("elapsed", "00:00:01"))
    print(f"{job_id}|{status}|{exit_code}|{elapsed}")
PY
