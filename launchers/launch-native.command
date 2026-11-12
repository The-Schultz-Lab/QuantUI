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
# Repo root is one level up from this launchers/ folder.
cd "$SCRIPT_DIR/.."

# Startup banner (shared; see _banner.sh). Non-fatal by design under
# `set -eu` — decoration must never abort a launch.
if [ -f "$SCRIPT_DIR/_banner.sh" ]; then
    # shellcheck source=/dev/null
    . "$SCRIPT_DIR/_banner.sh" || true
    quantui_banner \
        "NATIVE MODE — local conda env on macOS, no container" \
        "quantui/*.py edits are live; no rebuild needed" || true
else
    echo "QuantUI NATIVE MODE — Local conda env on macOS, no container"
    echo
fi

# Locate conda.sh. Detect the install rather than assuming one location:
# miniforge, miniconda, and anaconda are all supported, in home, /opt, and
# Homebrew Caskroom prefixes. Falls back to an already-configured conda via
# $CONDA_EXE.
CONDA_SH=""
for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/miniforge3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" \
    "/usr/local/Caskroom/miniforge/base/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
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
if [ -z "$CONDA_SH" ] && [ -n "${CONDA_EXE:-}" ]; then
    fallback="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    [ -f "$fallback" ] && CONDA_SH="$fallback"
fi

if [ -z "$CONDA_SH" ]; then
    echo "ERROR: Could not locate conda.sh."
    echo "       Install Miniforge to ~/miniforge3 and re-run."
    echo "       https://github.com/conda-forge/miniforge"
    echo
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
fi

# conda's activation scripts read unset variables, which `set -u` turns
# into a fatal error, so nounset is off for these two lines only.
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate quantui
set -u

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
        # Stamp even on failure so offline launches don't retry (and re-delay
        # on) pip every time. Re-run `pip install -e .` manually when online if
        # you add a real dependency.
        echo "[QuantUI] editable reinstall skipped (offline?) - using live source"
        touch .dev_install_stamp
    fi
fi

# Mirrors the Windows launcher: clear bytecode and disable .pyc writes.
# Not strictly required on macOS (no WSL2 DrvFs mtime issues), but kept
# for parity so the dev experience is identical across platforms.
rm -rf quantui/__pycache__
export PYTHONDONTWRITEBYTECODE=1

# Port 8867 mirrors the Windows native launcher.
PORT=8867
# Per-run token (QUANTUI_APP_TOKEN when the caller sets one): without it any
# other local user or process could drive this session. Standalone Voila
# reads no token from the environment, so it is passed as an option.
TOKEN="${QUANTUI_APP_TOKEN:-$(python -c 'import secrets; print(secrets.token_urlsafe(24))')}"
URL="http://localhost:${PORT}/?token=${TOKEN}"

# True when something answers on 127.0.0.1:$1 (python is the conda env's).
port_open() {
    python -c 'import socket, sys
s = socket.socket()
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1"
}

# A server already on the port (an earlier run, another app) would be opened
# in the browser instead of this one.
if port_open "${PORT}"; then
    echo "ERROR: port ${PORT} is already in use - is QuantUI already running?"
    echo "       Stop that server first, then run this again."
    exit 1
fi

echo "Starting Voilà on ${URL}..."
echo

# Launch Voilà in the background so we can open the browser, then wait
# on the process so closing this Terminal window stops the server.
voila notebooks/molecule_computations.ipynb \
    --no-browser \
    --port="${PORT}" \
    --Voila.ip=127.0.0.1 \
    "--Voila.token=${TOKEN}" \
    --ServerApp.disable_check_xsrf=True &
VOILA_PID=$!

# Wait (up to 60 s) until Voilà answers on the port, rather than a fixed
# sleep; stop early if it exited (e.g. a startup error).
waited=0
until port_open "${PORT}"; do
    if ! kill -0 "${VOILA_PID}" 2>/dev/null; then
        echo "ERROR: Voilà exited during startup - see the messages above."
        wait "${VOILA_PID}" || true
        exit 1
    fi
    if [ "${waited}" -ge 60 ]; then
        echo "Voilà has not answered on port ${PORT} after 60 s; opening anyway."
        break
    fi
    sleep 1
    waited=$((waited + 1))
done
# Non-fatal under `set -e`: the URL is printed below regardless.
open "${URL}" || true

echo
echo "Native dev server running at ${URL}"
echo "All local quantui/*.py changes are live — no rebuild needed."
echo "Close this Terminal window or press Ctrl-C to stop."
echo

wait "${VOILA_PID}"
