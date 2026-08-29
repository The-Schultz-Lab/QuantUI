"""
Cluster / SLURM configuration defaults for batch execution backends.

Kept separate from ``quantui.config`` so the local teaching interface does
not import scheduler settings unless cluster code is used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource defaults and limits (conservative, educational cluster policy)
DEFAULT_CORES = 4
DEFAULT_MEMORY_GB = 8
DEFAULT_WALLTIME = "04:00:00"

MIN_CORES = 1
MAX_CORES = 32
MIN_MEMORY_GB = 1
MAX_MEMORY_GB = 128
MAX_CONCURRENT_JOBS = 2
SUBMIT_COOLDOWN_SECONDS = 30
STALE_NO_SLURM_ID_SECONDS = 600


def max_concurrent_jobs() -> int:
    """Return the active-job cap (operator override via env)."""
    override = os.environ.get("QUANTUI_MAX_CONCURRENT_JOBS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning(
                "Invalid QUANTUI_MAX_CONCURRENT_JOBS=%r; using default %s",
                override,
                MAX_CONCURRENT_JOBS,
            )
    return MAX_CONCURRENT_JOBS


def submit_cooldown_seconds() -> int:
    """Minimum seconds between SLURM submits (0 disables cooldown)."""
    override = os.environ.get("QUANTUI_SLURM_SUBMIT_COOLDOWN_S")
    if override:
        try:
            return max(0, int(override))
        except ValueError:
            logger.warning(
                "Invalid QUANTUI_SLURM_SUBMIT_COOLDOWN_S=%r; using default %s",
                override,
                SUBMIT_COOLDOWN_SECONDS,
            )
    return SUBMIT_COOLDOWN_SECONDS


def stale_no_slurm_id_seconds() -> int:
    """Mark active jobs without a SLURM id stale after this many seconds."""
    override = os.environ.get("QUANTUI_SLURM_STALE_NO_ID_S")
    if override:
        try:
            return max(60, int(override))
        except ValueError:
            logger.warning(
                "Invalid QUANTUI_SLURM_STALE_NO_ID_S=%r; using default %s",
                override,
                STALE_NO_SLURM_ID_SECONDS,
            )
    return STALE_NO_SLURM_ID_SECONDS


WALLTIME_OPTIONS = [
    "00:30:00",
    "01:00:00",
    "02:00:00",
    "04:00:00",
    "08:00:00",
    "12:00:00",
    "24:00:00",
    "48:00:00",
]

# Site-specific — override via env or config.local import in deployment.
DEFAULT_PARTITION = os.environ.get("QUANTUI_SLURM_PARTITION", "common")

ALLOWED_MAIL_EVENTS = ["NONE", "BEGIN", "END", "FAIL", "REQUEUE", "ALL"]
DEFAULT_MAIL_EVENTS = ["END", "FAIL"]

# Status polling (seconds)
STATUS_REFRESH_INTERVAL = 10
CANCEL_CONFIRM_TIMEOUT_SECONDS = 30
CANCEL_POLL_INTERVAL_SECONDS = 1


def cancel_confirm_timeout_seconds() -> float:
    override = os.environ.get("QUANTUI_SLURM_CANCEL_CONFIRM_S")
    if override:
        try:
            return max(5.0, float(override))
        except ValueError:
            logger.warning(
                "Invalid QUANTUI_SLURM_CANCEL_CONFIRM_S=%r; using default %s",
                override,
                CANCEL_CONFIRM_TIMEOUT_SECONDS,
            )
    return float(CANCEL_CONFIRM_TIMEOUT_SECONDS)


# Registry and staging roots
def default_jobs_root() -> Path:
    override = os.environ.get("QUANTUI_JOBS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".quantui" / "jobs"


def default_staging_root() -> Path:
    override = os.environ.get("QUANTUI_STAGING_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".quantui" / "staging"


# Apptainer image for batch workers (NCShare-oriented default path).
APPTAINER_BATCH_IMAGE = os.environ.get(
    "QUANTUI_BATCH_IMAGE",
    os.path.expanduser("~/quantui-gpu.sif"),
)

# SLURM batch script template. ``{worker_command}`` is the full command line
# run inside the allocation (Apptainer-wrapped when configured).
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={cores}
#SBATCH --mem={memory}G
#SBATCH --time={walltime}
#SBATCH --output={output_file}
#SBATCH --error={error_file}{optional_directives}

set -euo pipefail

echo "Job started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Running on node: $(hostname)"
echo "SLURM job ID: ${{SLURM_JOB_ID:-<none>}}"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cores}}}"
export QUANTUI_RESULTS_DIR="{results_dir}"
mkdir -p "$QUANTUI_RESULTS_DIR"

{worker_command}

echo "Job completed at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""
