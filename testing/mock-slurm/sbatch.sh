#!/usr/bin/env bash
# Mock SLURM sbatch for QuantUI backend tests (Linux CI).
set -euo pipefail

STATE_DIR="${MOCK_SLURM_STATE:?MOCK_SLURM_STATE must be set}"
mkdir -p "$STATE_DIR"

script_path=""
for arg in "$@"; do
  if [[ -f "$arg" ]]; then
    script_path="$arg"
    break
  fi
done

if [[ -z "$script_path" ]]; then
  echo "sbatch: error: No job script specified" >&2
  exit 1
fi

job_id="$((10000 + $(find "$STATE_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)))"
out_file="$STATE_DIR/${job_id}.json"

python3 - "$STATE_DIR" "$job_id" "$script_path" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

state_dir, job_id, script_path = sys.argv[1:4]
content = Path(script_path).read_text()
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

limits_path = Path(state_dir) / "config.json"
max_cores = 32
max_mem = 128
if limits_path.exists():
    cfg = json.loads(limits_path.read_text())
    max_cores = cfg.get("max_cores", max_cores)
    max_mem = cfg.get("max_memory_gb", max_mem)

if cores is not None and cores > max_cores:
    print(f"sbatch: error: CPU count {cores} exceeds limit", file=sys.stderr)
    sys.exit(1)
if memory is not None and memory > max_mem:
    print(f"sbatch: error: Memory {memory}G exceeds limit", file=sys.stderr)
    sys.exit(1)

payload = {
    "job_id": job_id,
    "status": "PENDING",
    "script": script_path,
    "submitted_at": os.environ.get("MOCK_SLURM_NOW", "0"),
}
Path(state_dir, f"{job_id}.json").write_text(json.dumps(payload))
PY

echo "Submitted batch job ${job_id}"

# Optionally transition to RUNNING then COMPLETED for squeue tests.
if [[ "${MOCK_SLURM_AUTO_COMPLETE:-0}" == "1" ]]; then
  python3 - "$STATE_DIR" "$job_id" <<'PY'
import json, sys
from pathlib import Path
state_dir, job_id = sys.argv[1:3]
path = Path(state_dir) / f"{job_id}.json"
data = json.loads(path.read_text())
data["status"] = "COMPLETED"
path.write_text(json.dumps(data))
PY
fi
