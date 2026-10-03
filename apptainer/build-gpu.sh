#!/usr/bin/env bash
# build-gpu.sh — Build the QuantUI GPU Apptainer image.
#
# Usage (from the repo root):
#   bash apptainer/build-gpu.sh                    # build the working tree
#   bash apptainer/build-gpu.sh --clean            # remove the old .sif first
#   bash apptainer/build-gpu.sh --test             # build, then run %test
#   bash apptainer/build-gpu.sh --fakeroot         # build unprivileged (HPC)
#
# Like build.sh, this copies the working tree into the image (the def's %files
# allowlist), so run it from the repo root on the code you want to ship. To
# build a release, check out its tag first (git checkout v0.9.0). The commit
# is recorded in the image (label QuantUICommit, /opt/build-info).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEF="apptainer/quantui-gpu.def"
SIF="quantui-gpu.sif"
APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"

CLEAN=false
RUN_TESTS=false
FAKEROOT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)    CLEAN=true; shift ;;
    --test)     RUN_TESTS=true; shift ;;
    --fakeroot) FAKEROOT=true; shift ;;
    --version)
      echo "ERROR: --version was removed: the GPU image now builds from the working" >&2
      echo "       tree, like the CPU image. Check out the release tag instead:" >&2
      echo "         git checkout v<x.y.z> && bash apptainer/build-gpu.sh" >&2
      exit 1 ;;
    --help|-h)  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# What is being built, for the banner and the image's provenance record. The
# version is the tree's own; the commit says which tree (`-dirty` when tracked
# files differ from it, since %files copies the files as they are on disk).
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)"
COMMIT="unknown"
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
  COMMIT="$(git describe --tags --always --dirty --abbrev=12)"
fi

# --clean maps to apptainer's --force below rather than rm-ing the image here:
# a failure after this point must not leave you with no image at all.

# Apptainer unpacks the base image into $APPTAINER_TMPDIR (default /tmp). On
# many systems — WSL included — /tmp is mounted `nodev`, which can make the
# build fail partway through creating device nodes, and it is often a small
# tmpfs that a multi-GB CUDA base image overflows. Both failures land deep into
# a long build. Default to a work dir next to the output instead, on the same
# filesystem that already has room for the .sif.
# Scratch dir, concurrency guard, cleanup trap and space preflight.
# Shared with build.sh — both scripts hit the same failures.
# shellcheck source=apptainer/_build_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_build_env.sh"

BUILD_OPTS=()
[[ "$FAKEROOT" == true ]] && BUILD_OPTS+=(--fakeroot)
[[ "$CLEAN" == true ]] && BUILD_OPTS+=(--force)
BUILD_OPTS+=(--build-arg "QUANTUI_COMMIT=${COMMIT}")

cat <<EOF
============================================================
Building: $SIF
From:     $DEF
QuantUI:  $VERSION  (working tree, $COMMIT)
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
