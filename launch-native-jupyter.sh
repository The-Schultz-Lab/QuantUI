#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="logs/native-jupyter.log"
mkdir -p "$(dirname "$LOG_FILE")"

{
    echo
    echo "=== QuantUI native Jupyter launch: $(date -Iseconds) ==="
    echo "PWD: $(pwd)"
} >> "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quantui

echo "Using Python: $(command -v python)"
echo "Using Jupyter: $(command -v jupyter)"

# Reinstall editable package only when pyproject metadata changed, or on first run.
# Must not abort the launch when offline (set -e) — pip fetches build deps from
# PyPI; fail fast + non-fatal. The editable source is live regardless.
if [ ! -f .dev_install_stamp ] || [ pyproject.toml -nt .dev_install_stamp ]; then
    if pip install -e . -q --timeout=5 --retries=0; then
        touch .dev_install_stamp
    else
        # Stamp even on failure so offline launches don't retry pip every time.
        echo "[QuantUI] editable reinstall skipped (offline?) - using live source"
        touch .dev_install_stamp
    fi
fi

# Prevent stale bytecode from WSL2 DrvFs mtime quirks.
rm -rf quantui/__pycache__
export PYTHONDONTWRITEBYTECODE=1

exec jupyter lab notebooks/molecule_computations.ipynb \
    --no-browser \
    --port=8868 \
    --ServerApp.port_retries=0 \
    --ServerApp.root_dir="$(pwd)" \
    --IdentityProvider.token='' \
    --ServerApp.password=''
