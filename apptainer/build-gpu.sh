#!/usr/bin/env bash
# build-gpu.sh — Build the QuantUI GPU Apptainer image.
#
# Usage (from the repo root):
#   bash apptainer/build-gpu.sh                    # build the pinned version
#   bash apptainer/build-gpu.sh --version 0.5.3    # build a different release
#   bash apptainer/build-gpu.sh --clean            # remove the old .sif first
#   bash apptainer/build-gpu.sh --test             # build, then run %test
#   bash apptainer/build-gpu.sh --fakeroot         # build unprivileged (HPC)
#
# Unlike build.sh, this does NOT copy the working tree into the image — it
# installs a published release from PyPI. So it can be run from anywhere, and
# what lands in the image is something anyone else can install by name.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEF="apptainer/quantui-gpu.def"
SIF="quantui-gpu.sif"
APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"

CLEAN=false
RUN_TESTS=false
FAKEROOT=false
VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)    CLEAN=true; shift ;;
    --test)     RUN_TESTS=true; shift ;;
    --fakeroot) FAKEROOT=true; shift ;;
    --version)  VERSION="${2:-}"; shift 2 ;;
    --help|-h)  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $1  (use --help)" >&2; exit 1 ;;
  esac
done

command -v "$APPTAINER_CMD" >/dev/null 2>&1 || {
  echo "ERROR: apptainer not found. Ubuntu/WSL: sudo apt-get install -y apptainer" >&2
  exit 1
}
[[ -f "$DEF" ]] || {
  echo "ERROR: $DEF not found — run this from the repo root." >&2
  exit 1
}

# Resolve the version the def would use, so the preflight below checks the same
# thing the build will actually request.
if [[ -z "$VERSION" ]]; then
  # ^\s* anchors to the %arguments default. Without the anchor this matches the
  # `--build-arg QUANTUI_VERSION=...` EXAMPLE in the comment above it, which
  # sits earlier in the file — so the script would preflight and build a
  # version nobody asked for.
  VERSION="$(grep -oP '^\s*QUANTUI_VERSION=\K[0-9][^\s]*' "$DEF" | head -1)"
  if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not read the QUANTUI_VERSION default from $DEF." >&2
    echo "       Pass one explicitly:  --version <x.y.z>" >&2
    exit 1
  fi
fi

# Preflight: confirm the release exists on PyPI before spending 20+ minutes
# pulling a multi-GB CUDA base image only to fail on the last pip step. This is
# a real trap right after cutting a tag — the GitHub release can be published
# while the PyPI job is still waiting on its environment approval.
echo "Checking PyPI for quantui==${VERSION} ..."
if command -v curl >/dev/null 2>&1; then
  http="$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/quantui/${VERSION}/json")"
  if [[ "$http" != "200" ]]; then
    echo "ERROR: quantui==${VERSION} is not on PyPI (HTTP ${http})." >&2
    echo "       If you just tagged it, the release workflow may still be waiting" >&2
    echo "       on the 'pypi' environment approval. Check:" >&2
    echo "         gh run list --workflow=release.yml --limit 3" >&2
    echo "       Or build a version that is published:  --version <x.y.z>" >&2
    exit 1
  fi
  echo "  found."
else
  echo "  (curl unavailable — skipping preflight)"
fi

[[ "$CLEAN" == true && -f "$SIF" ]] && { echo "Removing $SIF ..."; rm "$SIF"; }

# Apptainer unpacks the base image into $APPTAINER_TMPDIR (default /tmp). On
# many systems — WSL included — /tmp is mounted `nodev`, which can make the
# build fail partway through creating device nodes, and it is often a small
# tmpfs that a multi-GB CUDA base image overflows. Both failures land deep into
# a long build. Default to a work dir next to the output instead, on the same
# filesystem that already has room for the .sif.
if [[ -z "${APPTAINER_TMPDIR:-}" ]] && findmnt -no OPTIONS /tmp 2>/dev/null | grep -q nodev; then
  APPTAINER_TMPDIR="${PWD}/.apptainer-build-tmp"
  mkdir -p "$APPTAINER_TMPDIR"
  export APPTAINER_TMPDIR
  echo "Note: /tmp is nodev — using $APPTAINER_TMPDIR for the build instead."
  echo "      Override by setting APPTAINER_TMPDIR yourself."
fi

BUILD_OPTS=()
[[ "$FAKEROOT" == true ]] && BUILD_OPTS+=(--fakeroot)
BUILD_OPTS+=(--build-arg "QUANTUI_VERSION=${VERSION}")

cat <<EOF
============================================================
Building: $SIF
From:     $DEF
QuantUI:  $VERSION  (from PyPI)
Options:  ${BUILD_OPTS[*]}

The CUDA devel base is several GB — expect ~15-30 min on a
first build, most of it download and extract.
============================================================
EOF

START=$(date +%s)
"$APPTAINER_CMD" build "${BUILD_OPTS[@]}" "$SIF" "$DEF"
ELAPSED=$(( ($(date +%s) - START) / 60 ))

echo
echo "Build complete in ${ELAPSED} minutes."
ls -lh "$SIF"

if [[ "${APPTAINER_TMPDIR:-}" == "${PWD}/.apptainer-build-tmp" ]]; then
  rm -rf "$APPTAINER_TMPDIR"
fi

if [[ "$RUN_TESTS" == true ]]; then
  echo
  echo "Running %test (build-host checks — no GPU required) ..."
  "$APPTAINER_CMD" test "$SIF"
fi

cat <<EOF

Next: verify on a machine with a GPU. On NCShare, get an allocation first:

  salloc --partition=<PARTITION> --gres=gpu:h200:1 --cpus-per-task=4 \\
         --mem=16G --time=00:30:00
  bash apptainer/verify-gpu.sh $SIF

%test above proves the stack imports. It cannot prove the GPU works — it runs
on the build host, which usually has no device. verify-gpu.sh is what does.
EOF
