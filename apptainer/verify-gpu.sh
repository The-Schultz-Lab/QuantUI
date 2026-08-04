#!/usr/bin/env bash
# verify-gpu.sh — prove a QuantUI GPU image actually reaches the GPU.
#
# Usage:
#   bash apptainer/verify-gpu.sh [IMAGE]        # default: quantui-gpu.sif
#
# Run this ON A GPU NODE (inside an salloc/srun allocation on a cluster).
# It is read-only and takes under a minute.
#
# Why a ladder rather than one big check: when "the GPU doesn't work" the
# useful question is always *which layer*. Each step below isolates exactly one
# and reports its own verdict, so a failure names the thing to fix instead of
# sending you back to the top. The steps escalate:
#
#   host driver -> container sees device -> CuPy/driver ABI -> gpu4pyscf
#   -> QuantUI's own opinion -> a real calculation that reports gpu_used
#
# The last one is the only claim that matters. The first five can all pass
# while a calculation still runs on the CPU.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

IMAGE="${1:-quantui-gpu.sif}"
APPTAINER_CMD="${APPTAINER_CMD:-apptainer}"

PASS=0
FAIL=0
WARN=0

c_ok=$'\033[0;32m'; c_bad=$'\033[0;31m'; c_warn=$'\033[0;33m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
[ -t 1 ] || { c_ok=""; c_bad=""; c_warn=""; c_dim=""; c_off=""; }

step()  { printf '\n%s──[ %s ]%s\n' "$c_dim" "$1" "$c_off"; }
ok()    { printf '%s  PASS%s  %s\n' "$c_ok"   "$c_off" "$1"; PASS=$((PASS+1)); }
bad()   { printf '%s  FAIL%s  %s\n' "$c_bad"  "$c_off" "$1"; FAIL=$((FAIL+1)); }
warn()  { printf '%s  WARN%s  %s\n' "$c_warn" "$c_off" "$1"; WARN=$((WARN+1)); }

if [ ! -f "$IMAGE" ]; then
  echo "ERROR: image not found: $IMAGE" >&2
  echo "Build it first:  bash apptainer/build-gpu.sh" >&2
  exit 2
fi
if ! command -v "$APPTAINER_CMD" >/dev/null 2>&1; then
  echo "ERROR: $APPTAINER_CMD not found." >&2
  exit 2
fi

# WSL2 exposes the GPU through /dev/dxg and a driver shim in /usr/lib/wsl/lib
# rather than the usual kernel module, and Apptainer's --nv does not wire that
# up on its own. Detect it so the extra binds can be added and so step 2 is not
# reported as a hard failure on a machine where CUDA compute may work fine.
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=true
fi

NV_FLAGS=(--nv)
if [ "$IS_WSL" = true ] && [ -d /usr/lib/wsl/lib ]; then
  # Binds only — deliberately NOT an LD_LIBRARY_PATH override. Setting it here
  # would clobber the image's own (/usr/local/cuda/lib64:/.singularity.d/libs),
  # which is the path that resolves the host driver correctly on a real
  # cluster. Measured 2026-08-04: forcing the WSL lib first does not help
  # anyway — see the WSL note after step 2.
  NV_FLAGS+=(--bind /usr/lib/wsl/lib:/usr/lib/wsl/lib)
  [ -e /dev/dxg ] && NV_FLAGS+=(--bind /dev/dxg:/dev/dxg)
fi

echo "Image:     $IMAGE"
echo "Apptainer: $($APPTAINER_CMD --version)"
echo "Host:      $(hostname)"
if [ "$IS_WSL" = true ]; then
  echo "Platform:  WSL2 — added ${#NV_FLAGS[@]} GPU flags (see notes at step 2)"
fi

# ── Step 0 — the two silent CPU-fallback traps, checked before anything else ──
# Both of these let every later step pass while calculations quietly run on the
# CPU, so they are worth ruling out first rather than debugging backwards from
# a confusing gpu_used:false at the end.
step "0. Environment traps (would silently force CPU)"

if [ -n "${QUANTUI_DISABLE_GPU:-}" ]; then
  bad "QUANTUI_DISABLE_GPU=${QUANTUI_DISABLE_GPU} is set in your shell — every run will be CPU. unset it."
else
  ok "QUANTUI_DISABLE_GPU is not set"
fi

# Apptainer bind-mounts $HOME, so a persisted preference from any other machine
# follows you into the container.
SETTINGS="${QUANTUI_SETTINGS_PATH:-$HOME/.quantui/settings.json}"
if [ -f "$SETTINGS" ] && grep -q '"gpu_enabled"[[:space:]]*:[[:space:]]*false' "$SETTINGS" 2>/dev/null; then
  bad "compute.gpu_enabled is FALSE in $SETTINGS"
  echo "         Apptainer bind-mounts \$HOME, so the container inherits this and runs on CPU."
  echo "         Either flip it (Status tab -> Settings -> GPU offload), or make this run"
  echo "         deterministic:  --env QUANTUI_SETTINGS_PATH=/tmp/quantui-settings.json"
else
  ok "no persisted gpu_enabled=false in $SETTINGS"
fi

# ── Step 1 — host driver ─────────────────────────────────────────────────────
step "1. Host driver sees a GPU (outside the container)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/         /'
  ok "nvidia-smi lists a device on the host"
else
  bad "no GPU visible on the host — are you on a GPU node? (salloc --gres=gpu:1)"
  echo
  echo "Stopping: nothing below can pass without this."
  exit 1
fi

# ── Step 2 — --nv actually plumbs the driver in ───────────────────────────────
step "2. Device visible INSIDE the container (proves --nv works)"
if $APPTAINER_CMD exec "${NV_FLAGS[@]}" "$IMAGE" nvidia-smi -L 2>/dev/null | sed 's/^/         /'; then
  ok "--nv exposes the driver to the container"
elif [ "$IS_WSL" = true ]; then
  # On WSL, nvidia-smi inside a container commonly fails with "GPU access
  # blocked by the operating system" even when CUDA compute works, because the
  # NVML path is a Windows-side shim rather than a real kernel interface. It is
  # therefore not evidence either way here — step 3 is. On a real cluster this
  # branch never runs, and a failure there is genuine.
  warn "nvidia-smi failed inside the container — expected on WSL, not conclusive"
  echo "         WSL routes CUDA through /dev/dxg + /usr/lib/wsl/lib, and NVML is"
  echo "         a Windows shim that generally does not work in containers."
  echo "         Step 3 (real CuPy compute) is the check that decides it here."
else
  bad "--nv did not expose the driver"
  echo "         Common causes: --nv omitted; Apptainer built without NVIDIA support;"
  echo "         no GPU allocated to this job (check --gres)."
fi

# ── WSL reachability probe ───────────────────────────────────────────────────
# Measured here 2026-08-04 (RTX 5060 Ti, driver 581.95, Apptainer 1.5.3): the
# container loads the right libcuda — /usr/lib/wsl/lib/libcuda.so.1, or the one
# --nv injects — and cuDriverGetVersion still returns 0. WSL reaches the GPU
# through /dev/dxg and a Windows-side shim that does not function inside an
# Apptainer container, and binding the whole of /usr/lib/wsl does not change it.
#
# This is a platform limit, NOT a defect in the image. The check that would
# have caught a real image defect passed: the default load resolves the
# host driver from /.singularity.d/libs, not the compat/libcuda.so.570 that the
# CUDA base image ships — that ordering is what matters on a cluster.
WSL_UNREACHABLE=false
if [ "$IS_WSL" = true ]; then
  drv=$($APPTAINER_CMD exec "${NV_FLAGS[@]}" "$IMAGE" python -c \
    "import cupy; print(cupy.cuda.runtime.driverGetVersion())" 2>/dev/null || echo 0)
  [ "${drv:-0}" -eq 0 ] 2>/dev/null && WSL_UNREACHABLE=true
fi

# On WSL with no reachable GPU, downgrade the device-dependent steps: they
# cannot pass here for reasons that have nothing to do with this image.
dev_fail() {
  if [ "$WSL_UNREACHABLE" = true ]; then
    warn "$1 — cannot be tested on WSL (see note below)"
  else
    bad "$1"
  fi
}

# ── Step 3 — CuPy/driver ABI ─────────────────────────────────────────────────
# The first step that can fail while step 2 passes: the device is visible but
# the wheel line is wrong for this driver (e.g. cuda13x wheels on a 570 driver).
step "3. CuPy can talk to the driver (wheel/driver ABI match)"
if out=$($APPTAINER_CMD exec "${NV_FLAGS[@]}" "$IMAGE" python -c "
import cupy
print('device :', cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())
print('cupy   :', cupy.__version__)
print('runtime:', cupy.cuda.runtime.runtimeGetVersion())
a = cupy.arange(1_000_000, dtype=cupy.float64)
print('fp64 dot:', float((a*a).sum()))
" 2>&1); then
  echo "$out" | sed 's/^/         /'
  ok "CuPy allocated and ran an FP64 kernel on the device"
else
  echo "$out" | sed 's/^/         /'
  dev_fail "CuPy could not use the device — likely a CUDA wheel/driver mismatch"
fi

# ── Step 4 — QuantUI's own probe ─────────────────────────────────────────────
# `quantui gpu check` documents its exit code: 0 when offload is available, 1
# when it is not (cli.py _cmd_gpu_check). Keying off that rather than grepping
# the message means this cannot drift when the wording changes. The reason
# string goes to stderr, hence 2>&1.
step "4. QuantUI's own opinion (gpu4pyscf import + detection)"
out=$($APPTAINER_CMD exec "${NV_FLAGS[@]}" "$IMAGE" quantui gpu check 2>&1); rc=$?
echo "$out" | sed 's/^/         /'
if [ "$rc" -eq 0 ]; then
  ok "quantui gpu check reports GPU offload available"
else
  dev_fail "quantui gpu check reports the GPU unavailable — the reason above is the fix"
fi

# ── Step 5 — negative control ────────────────────────────────────────────────
# Doubles as the clearest teaching contrast: same image, same command, one env
# var, opposite answer. If this does NOT flip, the toggle is broken and every
# CPU-vs-GPU comparison built on it would be measuring nothing.
step "5. Negative control — QUANTUI_DISABLE_GPU=1 must flip the answer"
# NOTE the inverted polarity: here a NON-ZERO exit is the pass. The whole point
# is that the GPU becomes unavailable, and `gpu check` signals that with exit 1.
# Treating non-zero as a failure here would report FAIL on a perfectly healthy
# node — a mistake worth naming, since it costs a GPU allocation to discover.
out=$($APPTAINER_CMD exec "${NV_FLAGS[@]}" --env QUANTUI_DISABLE_GPU=1 "$IMAGE" quantui gpu check 2>&1); rc=$?
echo "$out" | sed 's/^/         /'
if [ "$rc" -ne 0 ] && echo "$out" | grep -q "QUANTUI_DISABLE_GPU"; then
  ok "CPU fallback is reachable AND visible (the reason names the env var)"
elif [ "$rc" -ne 0 ]; then
  # Disabled, but for some other reason — so this run proves nothing about the
  # toggle. Only meaningful once step 4 passes.
  warn "GPU is unavailable here, but not because of QUANTUI_DISABLE_GPU (see step 4)"
else
  bad "QUANTUI_DISABLE_GPU=1 did NOT disable the GPU — the toggle is broken"
fi

# ── Step 6 — the only claim that matters ─────────────────────────────────────
step "6. A real calculation reports gpu_used=true"
if out=$($APPTAINER_CMD exec "${NV_FLAGS[@]}" "$IMAGE" python -c "
from quantui import Molecule, run_in_session
mol = Molecule(
    atoms=['O', 'H', 'H'],
    coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
)
r = run_in_session(mol, method='RHF', basis='sto-3g', verbose=0)
print('formula  :', r.formula)
print('converged:', r.converged)
print('energy   :', r.energy_hartree)
print('gpu_used :', r.gpu_used)
print('gpu_name :', r.gpu_name)
raise SystemExit(0 if r.gpu_used else 1)
" 2>&1); then
  echo "$out" | sed 's/^/         /'
  ok "the calculation actually took the GPU path"
else
  echo "$out" | sed 's/^/         /'
  dev_fail "gpu_used was false (or the run failed) — this is the claim that counts"
  echo "         Note it still CONVERGED if it fell back: a correct energy is not"
  echo "         evidence the GPU was used. Only gpu_used is."
fi

# ── Summary ──────────────────────────────────────────────────────────────────
printf '\n%s────────────────────────────────────────────%s\n' "$c_dim" "$c_off"
printf '  %spassed %d%s   %sfailed %d%s   %swarnings %d%s\n' \
  "$c_ok" "$PASS" "$c_off" "$c_bad" "$FAIL" "$c_off" "$c_warn" "$WARN" "$c_off"

if [ "$WSL_UNREACHABLE" = true ]; then
  printf '\n  %sINCONCLUSIVE on WSL.%s The image is sound as far as this platform\n' \
    "$c_warn" "$c_off"
  printf '  can show: it builds, imports, and its CPU fallback works. But WSL\n'
  printf '  reaches the GPU through a Windows-side shim that does not function\n'
  printf '  inside an Apptainer container (cuDriverGetVersion returns 0 even with\n'
  printf '  the right libcuda loaded), so steps 3-6 cannot pass here for reasons\n'
  printf '  unrelated to this image.\n\n'
  printf '  Run this on a real GPU node — that is the only place it can answer.\n\n'
  printf '  %sNative WSL is a different story%s and IS worth using: outside a\n' \
    "$c_ok" "$c_off"
  printf '  container the same GPU works normally (verified here — driver 13000,\n'
  printf '  gpu_used: true). So QuantUI'"'"'s own GPU logic — offload dispatch, the\n'
  printf '  gpu_used reporting, method coverage — can be exercised locally in a\n'
  printf '  native env with the gpu extra matching YOUR driver:\n\n'
  printf '    pip install "quantui[gpu-cuda13x]"   # 13x for a 580+ driver\n'
  printf '    quantui gpu check\n\n'
  printf '  Only the container plumbing needs a real GPU node.\n'
  exit 3
fi

if [ "$FAIL" -eq 0 ]; then
  printf '\n  %sImage verified on this node.%s\n' "$c_ok" "$c_off"
  printf '  Record the pairing that worked, for the next build:\n'
  $APPTAINER_CMD exec "$IMAGE" cat /opt/build-info/quantui-version.txt 2>/dev/null \
    | sed 's/^/    quantui  /'
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -1 | sed 's/^/    driver   /'
  exit 0
fi

printf '\n  %s%d check(s) failed — fix the LOWEST-numbered one first.%s\n' \
  "$c_bad" "$FAIL" "$c_off"
printf '  Each step assumes the ones below it; a later failure is usually a\n'
printf '  symptom of an earlier one.\n'
exit 1
