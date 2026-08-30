#!/usr/bin/env bash
# Mock SLURM scancel for QuantUI backend tests.
set -euo pipefail

STATE_DIR="${MOCK_SLURM_STATE:?MOCK_SLURM_STATE must be set}"

job_id="${1:-}"
if [[ -z "$job_id" ]]; then
  echo "scancel: error: job id required" >&2
  exit 1
fi

python3 - "$STATE_DIR" "$job_id" <<'PY'
import json
import sys
from pathlib import Path

state_dir, job_id = sys.argv[1:3]
path = Path(state_dir) / f"{job_id}.json"
if not path.exists():
    sys.exit(0)
data = json.loads(path.read_text())
data["status"] = "CANCELLED"
path.write_text(json.dumps(data))
PY
