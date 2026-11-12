#!/usr/bin/env bash
# QuantUI NATIVE MODE — Linux / WSL launcher
# Run from a WSL Ubuntu (or native Linux) terminal:  ./launch-native.sh
# Equivalent to launch-native.command on macOS and launch-native.bat on
# Windows, but for a repo that lives on the native Linux filesystem
# (e.g. a clone at ~/GitHub/QuantUI). Running from the native filesystem
# rather than /mnt/c avoids the 9P "Operation not permitted" errors that
# break editable installs and is dramatically faster for git/builds.
#
# Use this when you have edited quantui/*.py and want to test immediately —
# quantui/*.py changes are always live in editable mode.

set -eu

# Resolve script directory so the launcher works no matter where it is run
# from (native ~/… clone or a /mnt/c checkout).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root is one level up from this launchers/ folder.
cd "$SCRIPT_DIR/.."

# Startup banner (shared with the other shell launchers; see _banner.sh).
# `|| true` so a missing/oddly-permissioned banner file can never abort a
# launch under `set -eu` — it is decoration, not function.
if [ -f "$SCRIPT_DIR/_banner.sh" ]; then
    # shellcheck source=/dev/null
    . "$SCRIPT_DIR/_banner.sh" || true
    quantui_banner \
        "NATIVE MODE — local conda env on Linux/WSL, no container" \
        "quantui/*.py edits are live; no rebuild needed" || true
else
    echo "QuantUI NATIVE MODE — Local conda env on Linux/WSL, no container"
    echo
fi

# Locate conda.sh. Detect the install rather than assuming one location:
# miniforge (recommended for WSL), miniconda, and anaconda are all supported,
# in home and system prefixes. Falls back to an already-configured conda via
# $CONDA_EXE.
CONDA_SH=""
for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniforge3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
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
    exit 1
fi

# conda's activation scripts read unset variables, which `set -u` turns
# into a fatal error, so nounset is off for these two lines only.
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate quantui
set -u

# pip install -e . is skipped when pyproject.toml has not changed since the
# last install (.dev_install_stamp). quantui/*.py changes are always live in
# editable mode — reinstall is only needed after pyproject.toml changes or on
# first use.
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

# Clear bytecode and disable .pyc writes. Required when the repo is on a
# /mnt/c (WSL2 DrvFs) checkout, where Windows-side mtime changes are not
# reliably propagated and Python may load stale pre-edit .pyc.
# Harmless on a native-filesystem clone, so it is kept for parity.
rm -rf quantui/__pycache__
export PYTHONDONTWRITEBYTECODE=1

# Port 8867 mirrors the Windows/macOS native launchers so the same browser tab
# can be reused across platforms.
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

# Launch Voilà in the background so we can open the browser, then wait on the
# process so closing this terminal (or Ctrl-C) stops the server.
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
# Try the common openers in order; under WSL, wslview/cmd.exe hand off to the
# Windows browser. All are non-fatal — the URL is printed regardless.
if command -v wslview >/dev/null 2>&1; then
    wslview "${URL}" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}" >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "${URL}" >/dev/null 2>&1 || true
fi

echo
echo "Native dev server running at ${URL}"
echo "All local quantui/*.py changes are live — no rebuild needed."
echo "Press Ctrl-C or close this terminal to stop."
echo

wait "${VOILA_PID}"
