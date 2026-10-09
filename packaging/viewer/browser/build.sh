#!/usr/bin/env bash
# Build the browser (Voici + Pyodide) QuantUI Viewer as a static site.
#
#   packaging/viewer/browser/build.sh WHEEL SAMPLES_DIR OUT_DIR [PYODIDE_TARBALL]
#
# WHEEL           the quantui-*.whl to bundle
# SAMPLES_DIR     result folders shipped as demo content (may be empty)
# OUT_DIR         where the static site is written (serve it over HTTP; file://
#                 cannot run WebAssembly workers)
# PYODIDE_TARBALL optional pyodide-X.tar.bz2 to self-host (offline-capable,
#                 ~550 MB site). Omitted: Pyodide loads from the jsDelivr CDN.
#
# Pins matter: jupyterlite-pyodide-kernel 0.7.2 is built against a newer
# JupyterLab than voici 0.10.0 ships and fails to load ("Shared module
# @jupyterlab/pluginmanager doesn't exist"); 0.7.0 matches. The xeus addon is
# disabled because the kernel is Pyodide, not xeus-python.
set -euo pipefail
WHEEL=$1 SAMPLES=$2 OUT=$3 TARBALL=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python -m pip install --quiet "voici==0.10.0" "jupyterlite-pyodide-kernel==0.7.0" \
    "jupyterlab_widgets==3.0.17"
mkdir -p "$WORK/content" "$WORK/wheels"
cp "$HERE/viewer.ipynb" "$WORK/content/"
mkdir -p "$WORK/content/sample_results"
cp -r "$SAMPLES"/. "$WORK/content/sample_results/" 2>/dev/null || true
cp "$WHEEL" "$WORK/wheels/"
python -m pip download --quiet --no-deps --only-binary=:all: --dest "$WORK/wheels" \
    --python-version 3.13 --platform any \
    plotly narwhals py3Dmol "ipywidgets==8.1.9" comm "jupyterlab_widgets==3.0.17" \
    "widgetsnbextension==4.0.16"

ARGS=(--contents content --output-dir "$(realpath -m "$OUT")" --piplite-wheels wheels
      --disable-addons jupyterlite-xeus)
if [ -n "$TARBALL" ]; then
    ARGS+=(--PyodideAddon.pyodide_url="$(realpath "$TARBALL")")
fi
(cd "$WORK" && voici build "${ARGS[@]}")
echo "Built $(du -sh "$OUT" | cut -f1) site at $OUT — open voici/render/viewer.html"
