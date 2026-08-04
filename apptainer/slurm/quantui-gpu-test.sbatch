#!/bin/bash
# quantui-gpu-test.sbatch — verify the QuantUI GPU image on a cluster node.
#
#   sbatch apptainer/slurm/quantui-gpu-test.sbatch
#   sbatch --export=ALL,QUANTUI_IMAGE=/path/to/quantui-gpu.sif apptainer/slurm/...
#
# Sized for NCShare's H200 nodes. Two values below are marked TODO because they
# are site-specific and unconfirmed — a wrong partition fails in seconds with a
# clear message, so leaving them obviously blank beats guessing something that
# looks plausible.

#SBATCH --job-name=quantui-gpu-test
#SBATCH --output=quantui-gpu-test-%j.out
#SBATCH --error=quantui-gpu-test-%j.err

# TODO(NCShare): confirm the partition name with Uthpala before first run.
#SBATCH --partition=CHANGEME

# TODO(NCShare): confirm the exact --gres string. "gpu:h200:1" is the expected
# form given the node type, but the configured gres name is what matters and
# only the site can confirm it. "gpu:1" is the safer fallback if unsure.
#SBATCH --gres=gpu:h200:1

# 4 CPUs per GPU: the GPU nodes are 192 cores / 8 GPUs, so ~12 physical cores
# per GPU are available. 4 leaves plenty of headroom for PySCF's threaded CPU
# portions without oversubscribing a shared node.
#SBATCH --cpus-per-task=4

# 16 GB is generous for the verification molecule (water/STO-3G needs a tiny
# fraction). Raise it for real workloads — the nodes carry 2 TB.
#SBATCH --mem=16G

# Wall time is for human error, not compute: the checks below take under a
# minute. A short ceiling also gets the job scheduled sooner.
#SBATCH --time=00:20:00

set -euo pipefail

IMAGE="${QUANTUI_IMAGE:-$HOME/quantui-gpu.sif}"

echo "=================================================="
echo "job        : ${SLURM_JOB_ID:-<none>} on $(hostname)"
echo "image      : $IMAGE"
echo "gres       : ${SLURM_JOB_GRES:-<unset>}"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "started    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================="

if [ ! -f "$IMAGE" ]; then
  echo "ERROR: image not found: $IMAGE" >&2
  echo "Set QUANTUI_IMAGE, or stage the .sif to \$HOME." >&2
  exit 2
fi

# Match PySCF's CPU threading to the allocation. Left unset, OpenMP grabs every
# core it can see on a shared node — including cores belonging to other jobs,
# which oversubscribes and slows everyone down. This matters even for GPU runs:
# parts of the SCF stay on the host.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Make the run deterministic. $HOME is bind-mounted into the container, and
# QuantUI keeps a persistent compute.gpu_enabled preference at
# ~/.quantui/settings.json — so a preference set on some other machine would
# silently put this job on the CPU. Pointing at a job-scoped path means the
# image's own default (GPU enabled) applies and the result reflects the
# hardware, not a leftover setting. Drop this line if you are deliberately
# testing that your saved preference is honoured.
export QUANTUI_SETTINGS_PATH="${TMPDIR:-/tmp}/quantui-settings-${SLURM_JOB_ID:-$$}.json"

# Keep results with the job rather than in a shared NFS home.
export QUANTUI_RESULTS_DIR="${SLURM_SUBMIT_DIR:-$PWD}/quantui-results-${SLURM_JOB_ID:-$$}"
mkdir -p "$QUANTUI_RESULTS_DIR"

# --nv is what exposes the host driver to the container. Without it every check
# below fails at step 2, and any real calculation quietly runs on the CPU.
# `|| STATUS=$?` rather than a bare call: `set -e` would abort the script the
# moment verify-gpu.sh exits non-zero, so a failing run would lose the summary
# below — exactly the output you need when the job failed.
STATUS=0
srun bash "${SLURM_SUBMIT_DIR:-$PWD}/apptainer/verify-gpu.sh" "$IMAGE" || STATUS=$?

echo "=================================================="
echo "finished   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "exit       : $STATUS"
echo "results    : $QUANTUI_RESULTS_DIR"
echo "=================================================="
exit $STATUS
