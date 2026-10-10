"""
Cluster / SLURM configuration defaults for batch execution backends.

Kept separate from ``quantui.config`` so the local teaching interface does
not import scheduler settings unless cluster code is used.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
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

# Optional site directives (M-BATCH2 B2.1), read per job so a changed
# environment applies without restarting. Each set variable adds one
# ``#SBATCH`` line, in this order. NCShare GPU jobs, for example, need
# ``QUANTUI_SLURM_PARTITION=gpu QUANTUI_SLURM_GRES=gpu:h200:1``.
SITE_DIRECTIVE_ENV = (
    ("QUANTUI_SLURM_ACCOUNT", "account"),
    ("QUANTUI_SLURM_QOS", "qos"),
    ("QUANTUI_SLURM_GRES", "gres"),
)
# The values are pasted into the batch script, so only the characters
# Slurm account/QOS/GRES names use are accepted.
_SITE_DIRECTIVE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:,=+-]+$")


def site_directives() -> list[str]:
    """``#SBATCH`` lines for the account/QOS/GRES env settings that are set.

    Raises ``ValueError`` for a value with characters outside
    ``[A-Za-z0-9_.:,=+-]`` rather than dropping it: a silently missing
    ``--gres`` would run a GPU job on CPU with a correct-looking result.
    """
    lines = []
    for var, flag in SITE_DIRECTIVE_ENV:
        value = os.environ.get(var, "").strip()
        if not value:
            continue
        if not _SITE_DIRECTIVE_VALUE_RE.match(value):
            raise ValueError(
                f"{var}={value!r} has characters not allowed in an #SBATCH "
                "value (use letters, digits and _ . : , = + -)"
            )
        lines.append(f"#SBATCH --{flag}={value}")
    return lines


def gpu_requested() -> bool:
    """True when the worker should get ``apptainer exec --nv``.

    ``QUANTUI_SLURM_NV=1``/``0`` decides outright; otherwise (unset or
    ``auto``) it is on when ``QUANTUI_SLURM_GRES`` names a GPU, e.g.
    ``gpu:h200:1``. Without a GPU in the allocation ``--nv`` has nothing to
    expose. A site whose GPU GRES has another name (``shard:1``) sets
    ``QUANTUI_SLURM_NV=1``.
    """
    override = os.environ.get("QUANTUI_SLURM_NV", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False
    return "gpu" in os.environ.get("QUANTUI_SLURM_GRES", "").lower()


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


DEFAULT_STAGING_ROOT = Path("~") / ".quantui" / "staging"


def staging_root_env_configured() -> bool:
    """True when ``QUANTUI_STAGING_DIR`` pins the job root (UI shows it locked)."""
    return bool(os.environ.get("QUANTUI_STAGING_DIR"))


def default_staging_root() -> Path:
    """Root for SLURM job folders.

    Precedence: ``QUANTUI_STAGING_DIR`` > the ``compute.slurm_job_root``
    user setting (M-JOBDIRS JD.1) > ``~/.quantui/staging``.
    """
    override = os.environ.get("QUANTUI_STAGING_DIR")
    if override:
        return Path(override).expanduser()
    try:
        from quantui.user_settings import UserSettings

        configured = UserSettings.load().compute.slurm_job_root
    except Exception:  # noqa: BLE001 — a broken settings file must not block jobs
        configured = ""
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_STAGING_ROOT.expanduser()


# Apptainer image for batch workers (NCShare-oriented default path).
APPTAINER_BATCH_IMAGE = os.environ.get(
    "QUANTUI_BATCH_IMAGE",
    os.path.expanduser("~/quantui-gpu.sif"),
)

# Shell block that gives every run of submit.slurm its own attempt dir
# (M-JOBDIRS). Numbering is max(existing NN) + 1 under a lock, and ``mkdir``
# without ``-p`` makes the script fail rather than write into an existing
# directory — an earlier attempt's files are never overwritten, whether the
# job came from QuantUI's Resubmit or a hand-run ``sbatch submit.slurm``.
# ``$JOB_DIR`` is set by the line build_attempt_setup() prepends.
_ATTEMPT_SETUP_BODY = r"""exec 9>"$JOB_DIR/.attempt.lock"
if command -v flock >/dev/null 2>&1; then flock 9; fi
attempt_n=0
for d in "$JOB_DIR"/attempt-*; do
  [ -d "$d" ] || continue
  k="${d##*/attempt-}"
  k="${k%%_*}"
  case "$k" in
    ''|*[!0-9]*) continue ;;
  esac
  k=$((10#$k))
  if [ "$k" -gt "$attempt_n" ]; then attempt_n=$k; fi
done
attempt_n=$((attempt_n + 1))
ATTEMPT_DIR="$JOB_DIR/$(printf 'attempt-%02d_job%s' "$attempt_n" "${SLURM_JOB_ID:-manual$$}")"
mkdir "$ATTEMPT_DIR"
exec 9>&-
ln -sfn "${ATTEMPT_DIR##*/}" "$JOB_DIR/latest" 2>/dev/null || true
cd "$ATTEMPT_DIR"
echo "Attempt directory: $ATTEMPT_DIR"
"""


def build_attempt_setup(job_dir: str) -> str:
    """Return the attempt-dir shell block for a job dir (shell-quoted)."""
    return f"\nJOB_DIR={shlex.quote(job_dir)}\n{_ATTEMPT_SETUP_BODY}"


# SLURM batch script template. ``{worker_command}`` is the full command line
# run inside the allocation (Apptainer-wrapped when configured).
# ``--comment=quantui`` tags every QuantUI job so an operator can list them
# all (``squeue -o "%i %u %j %T %k"``), whoever submitted them and from where.
# The worker is one process with OpenMP threads, hence one task with
# ``{cores}`` CPUs (B2.1). ``--cpus-per-task`` also makes Slurm set this
# job's own SLURM_CPUS_PER_TASK, so a value inherited from a submitting
# allocation can no longer set the thread count.
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --mem={memory}G
#SBATCH --time={walltime}
#SBATCH --comment=quantui
#SBATCH --output="{output_file}"
#SBATCH --error="{error_file}"{optional_directives}

set -euo pipefail

echo "Job started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Running on node: $(hostname)"
echo "SLURM job ID: ${{SLURM_JOB_ID:-<none>}}"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{cores}}}"
{attempt_setup}
{worker_command}

echo "Job completed at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""
