#!/bin/bash
# quantui-batch.sbatch — run a QuantUI CalculationRequest via the headless worker.
#
# Generated per-job copies live in ~/.quantui/staging/<request_id>/submit.slurm.
# This file is the reference template for operators (NCShare / Apptainer).
#
#   sbatch --export=ALL,QUANTUI_BATCH_IMAGE=$HOME/quantui-gpu.sif \
#          apptainer/slurm/quantui-batch.sbatch
#
# TODO(NCShare): confirm partition and account/QoS with site admin.

#SBATCH --job-name=quantui-batch
#SBATCH --output=quantui-batch-%j.out
#SBATCH --error=quantui-batch-%j.err
#SBATCH --partition=CHANGEME
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

IMAGE="${QUANTUI_BATCH_IMAGE:-$HOME/quantui-gpu.sif}"
REQUEST="${QUANTUI_REQUEST_PATH:?Set QUANTUI_REQUEST_PATH to staging/request.json}"
STAGING="$(dirname "$REQUEST")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export QUANTUI_RESULTS_DIR="${STAGING}/results"
mkdir -p "$QUANTUI_RESULTS_DIR"

apptainer exec --nv --bind "$HOME:$HOME" --pwd "$STAGING" "$IMAGE" \
  python -m quantui.backends.worker --request "$REQUEST"
