#!/usr/bin/env bash
# QuantUI NATIVE MODE — macOS launcher
# Double-click in Finder, or run from a terminal. Equivalent to
# launch-native.bat on Windows but runs natively on macOS (no WSL
# layer, since PySCF installs cleanly on macOS).
#
# Use this when you have edited quantui/*.py and want to test
# immediately — quantui/*.py changes are always live in editable mode.

set -eu

# Resolve script directory so double-click from any location works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "QuantUI NATIVE MODE — Local conda env on macOS, no container"
echo "Use this when you have edited quantui/*.py and want to test immediately."
echo

# Locate conda.sh. Miniconda at ~/miniconda3 is the documented install;
# we try a few common fallback locations before giving up.
CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/opt/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "ERROR: Could not locate conda.sh."
    echo "       Install Miniconda to ~/miniconda3 and re-run."
    echo "       https://docs.conda.io/projects/miniconda/en/latest/"
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate quantui

# pip install -e . is skipped when pyproject.toml has not changed since
# the last install (.dev_install_stamp). quantui/*.py changes are always
# live in editable mode — reinstall is only needed after pyproject.toml
# changes or on first use.
if [ ! -f .dev_install_stamp ] || [ pyproject.toml -nt .dev_install_stamp ]; then
    echo "Installing quantui in editable mode (first run or pyproject.toml changed)..."
    # Must not abort the launch when offline (set -e) — pip fetches build deps
    # from PyPI, which hangs/fails with no network. Fail fast + non-fatal; the
    # editable source is live regardless of whether the reinstall ran.
    if pip install -e . -q --timeout=5 --retries=0; then
        touch .dev_install_stamp
    else
        echo "[QuantUI] editable reinstall skipped (offline?) - using live source"
    fi
fi

# Mirrors the Windows launcher: clear bytecode and disable .pyc writes.
# Not strictly required on macOS (no WSL2 DrvFs mtime issues), but kept
# for parity so the dev experience is identical across platforms.
rm -rf quantui/__pycache__
export PYTHONDONTWRITEBYTECODE=1

# Port 8867 mirrors the Windows native launcher.
PORT=8867
URL="http://localhost:${PORT}"

echo "Starting Voilà on ${URL}..."
echo

# Launch Voilà in the background so we can open the browser, then wait
# on the process so closing this Terminal window stops the server.
voila notebooks/molecule_computations.ipynb \
    --no-browser \
    --port="${PORT}" \
    --ServerApp.disable_check_xsrf=True &
VOILA_PID=$!

# Give Voilà a moment to bind to the port before opening the browser.
sleep 4
open "${URL}"

echo
echo "Native dev server running at ${URL}"
echo "All local quantui/*.py changes are live — no rebuild needed."
echo "Close this Terminal window or press Ctrl-C to stop."
echo

wait "${VOILA_PID}"
