#!/usr/bin/env bash
# Mock SLURM squeue for QuantUI backend tests: same behaviour as the .cmd shim,
# from the one implementation in mock_slurm_cli.py.
set -euo pipefail
exec python3 "$(dirname "$0")/mock_slurm_cli.py" squeue "$@"
