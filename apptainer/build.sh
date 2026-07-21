#!/usr/bin/env bash
# build.sh — Build the QuantUI Apptainer container
#
# Usage (run from the repo root):
#   bash apptainer/build.sh            # build to repo root
#   bash apptainer/build.sh --clean    # remove old .sif first
#   bash apptainer/build.sh --test     # build + run container tests
#   bash apptainer/build.sh --fakeroot # build without root (HPC systems)
#
# The .def file copies the repo root into the container, so this script
# MUST be run from the repo root, not from apptainer/.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DEF="apptainer/quantui.def"
SIF="quantui.sif"
APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"

# ── Parse flags ───────────────────────────────────────────────────────────────
CLEAN=false
RUN_TESTS=false
FAKEROOT=false

for arg in "$@"; do
  case "$arg" in
    --clean)    CLEAN=true ;;
    --test)     RUN_TESTS=true ;;
    --fakeroot) FAKEROOT=true ;;
    --help|-h)
      sed -n '2,12p' "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg  (use --help)" >&2
      exit 1
      ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v "$APPTAINER_CMD" &>/dev/null; then
  echo "ERROR: apptainer not found."
  echo "Install on Ubuntu/WSL:  sudo apt-get install -y apptainer"
  exit 1
fi

if [[ ! -f "$DEF" ]]; then
  echo "ERROR: Definition file not found: $DEF"
  echo "Run this script from the repo root (not from apptainer/)."
  exit 1
fi

# ── Optional clean ────────────────────────────────────────────────────────────
if [[ "$CLEAN" == true && -f "$SIF" ]]; then
  echo "Removing existing $SIF ..."
  rm "$SIF"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
BUILD_OPTS=""
if [[ "$FAKEROOT" == true ]]; then
  BUILD_OPTS="--fakeroot"
fi

echo "============================================================"
echo "Building: $SIF"
echo "From:     $DEF"
echo "Options:  ${BUILD_OPTS:-none}"
echo "This takes ~15–30 minutes (download + extract dominate; the mamba"
echo "solve is fast). Go make a coffee."
echo "============================================================"

START=$(date +%s)
"$APPTAINER_CMD" build $BUILD_OPTS "$SIF" "$DEF"
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "Build complete in ${ELAPSED} minutes."
ls -lh "$SIF"

# ── Optional tests ────────────────────────────────────────────────────────────
if [[ "$RUN_TESTS" == true ]]; then
  echo ""
  echo "Running container %test section..."
  "$APPTAINER_CMD" test "$SIF"

  echo ""
  echo "Running smoke test (import check + water molecule)..."
  "$APPTAINER_CMD" exec --cleanenv "$SIF" python -c "
import quantui, pyscf, ase, py3Dmol
from quantui import Molecule, parse_xyz_input
atoms, coords = parse_xyz_input('O 0 0 0\nH 0.757 0.587 0\nH -0.757 0.587 0')
mol = Molecule(atoms, coords)
print('Smoke test passed:', mol.get_formula())
"

  echo ""
  echo "Running full notebook workflow tests..."
  "$APPTAINER_CMD" exec --cleanenv --writable-tmpfs "$SIF" bash -c '
    pip install pytest -q 2>/dev/null
    python -m pytest tests/test_notebook_workflows.py -v --tb=short --override-ini="addopts="
  '
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "To run the container:"
echo "  apptainer run $SIF app    # Voilà widget UI (students)"
echo "  apptainer run $SIF        # JupyterLab (development)"
