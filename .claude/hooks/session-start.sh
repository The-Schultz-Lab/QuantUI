#!/bin/bash
# SessionStart hook for QuantUI — Claude Code on the web.
# Installs the package + dev/test deps so pytest and the linters work in a
# fresh, ephemeral cloud container. Mirrors .github/workflows/ci.yml so the
# in-session environment matches what CI enforces.
#
# Runs synchronously: the session waits until deps are installed, so tests and
# linters are guaranteed ready (no race). Web-only and idempotent.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment. Local machines
# use the conda env from local-setup/environment.yml instead.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# Best-effort: a newer pip resolves faster, but some base images ship a
# distro-managed pip that can't self-upgrade — don't let that abort setup.
python -m pip install --upgrade pip || true

# Ensure a modern setuptools/wheel. Some base images ship a distro-patched
# setuptools that breaks source builds of pyscf-properties with
# "AttributeError: install_layout". --ignore-installed sidesteps the
# distro-managed copy (which has no RECORD and so can't be uninstalled).
python -m pip install --upgrade --ignore-installed setuptools wheel || true

# Editable install with the same extras CI uses. Editable + plain `install`
# (not `ci`-style clean installs) so the resolved deps are cached in the
# container image for later sessions.
pip install -e ".[pyscf,ase,dev]"

echo "QuantUI cloud env ready: package + [pyscf,ase,dev] installed." >&2
