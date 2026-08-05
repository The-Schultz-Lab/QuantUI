#!/usr/bin/env python3
"""Generate apptainer/ncshare-gpu-diagnostic.ipynb.

The notebook is generated rather than hand-edited so the cell sources stay
readable here (real Python, lintable, with exact geometries computed rather
than pasted) instead of becoming escaped JSON strings nobody wants to touch.

Regenerate after editing:  python apptainer/make_ncshare_diagnostic.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "ncshare-gpu-diagnostic.ipynb"


def benzene_xyz() -> list[tuple[str, float, float, float]]:
    """D6h benzene, C-C 1.39 A, C-H 1.09 A. Computed, not pasted, so the
    geometry is exactly symmetric and the number in the notebook is auditable."""
    r_c, r_h = 1.39, 1.39 + 1.09
    atoms = []
    for i in range(6):
        a = math.radians(60 * i)
        atoms.append(("C", r_c * math.cos(a), r_c * math.sin(a), 0.0))
    for i in range(6):
        a = math.radians(60 * i)
        atoms.append(("H", r_h * math.cos(a), r_h * math.sin(a), 0.0))
    return atoms


BENZENE = ",\n        ".join(
    f'("{s}", {x:.6f}, {y:.6f}, {z:.6f})' for s, x, y, z in benzene_xyz()
)

MD_INTRO = """# QuantUI GPU diagnostic — NCShare

Run this **inside the container**, in an interactive session on a GPU node.

```bash
salloc --partition=<PARTITION> --gres=gpu:h200:1 --cpus-per-task=4 \\
       --mem=32G --time=01:00:00

apptainer exec --nv \\
  --env QUANTUI_SETTINGS_PATH=$TMPDIR/quantui-settings.json \\
  quantui-gpu.sif jupyter lab --ip=0.0.0.0 --no-browser
```

`--nv` is what exposes the driver; without it every GPU check below fails.
`QUANTUI_SETTINGS_PATH` isolates the run from any `compute.gpu_enabled=false`
saved in your `$HOME`, which Apptainer bind-mounts — see cell 2.

**Run all cells top to bottom.** Each is independent and traps its own errors,
so one failure does not stop the rest. The last cell writes a JSON record and
prints a summary block to paste into your notes.

### What each section can and cannot prove

| § | Question |
|---|---|
| 1 | What hardware and allocation is this? (records the driver number) |
| 2 | Is anything silently forcing CPU? |
| 3 | Can CuPy reach the device and do FP64? |
| 4 | Does QuantUI's own probe agree? |
| 5 | Is the H200 correctly treated as a strong-FP64 device? |
| 6 | Negative control — does disabling GPU actually change the answer? |
| 7 | Does a real calculation report `gpu_used`? |
| 8 | Where is the CPU/GPU crossover? |

Sections 1-6 can all pass while calculations still run on CPU. **Section 7 is
the only one that proves offload happened**, and note that a CPU fallback still
converges to the correct energy — a right answer is not evidence of GPU use.
"""

CELL_SETUP = '''# ── Setup ────────────────────────────────────────────────────────────────
# Collects results as it goes; the final cell serialises this.
import json, os, platform, subprocess, sys, time
from datetime import datetime, timezone

REPORT = {"sections": {}, "generated_utc": datetime.now(timezone.utc).isoformat()}

def record(section, **kv):
    REPORT["sections"].setdefault(section, {}).update(kv)
    return kv

def sh(cmd, timeout=60):
    """Run a shell command, returning (ok, output). Never raises."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def banner(t):
    print("=" * 68); print(t); print("=" * 68)

print("python  :", sys.version.split()[0])
print("platform:", platform.platform())
'''

CELL_HW = '''# ── 1. Hardware and allocation ───────────────────────────────────────────
# Also captures the exact driver version, which the planning notes still list
# as unknown (the recorded 570.133.20 is inconsistent with CUDA 13.0).
banner("1. HARDWARE AND ALLOCATION")

import quantui
print("quantui :", quantui.__version__)
try:
    print("image   :", open("/opt/build-info/quantui-version.txt").read().strip(),
          "built", open("/opt/build-info/build-date.txt").read().strip())
except Exception:
    print("image   : (build-info not found — not running inside the GPU image?)")

slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_GRES",
          "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST"):
    print(f"{k:22} {slurm.get(k, '(unset)')}")

ok, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,"
             "compute_cap --format=csv,noheader")
print("\\nnvidia-smi:", out if ok else f"FAILED — {out}")
if ok and out:
    fields = [f.strip() for f in out.splitlines()[0].split(",")]
    record("hardware", gpu_name=fields[0],
           driver_version=fields[1] if len(fields) > 1 else None,
           memory=fields[2] if len(fields) > 2 else None,
           compute_capability=fields[3] if len(fields) > 3 else None)
    print("\\n>>> RECORD THIS DRIVER VERSION — it is the open hardware question.")

ok, out = sh("nvidia-smi -L")
print("\\nvisible devices:\\n" + (out if ok else "none"))

record("hardware", quantui_version=quantui.__version__,
       nvidia_smi_ok=ok, slurm={k: slurm.get(k) for k in slurm})
record("allocation", cpus=os.cpu_count(),
       omp_num_threads=os.environ.get("OMP_NUM_THREADS", "(unset)"))
# ⚠️ The number that matters is the AFFINITY MASK, not cpu_count(). Slurm can
# constrain a job two ways: a cgroup CPU *quota* (you get N cores' worth of
# time, but the mask still shows every core on the node) or a cpuset (the mask
# itself shrinks). Under a quota, OpenMP sees all 192 cores, spawns 192 threads
# and thrashes them across your 6 cores' worth of quota — so the CPU leg runs
# SLOWER than a properly configured run, and every GPU speedup measured against
# it is flattering. Found on NCShare 2026-08-05: 6 cores requested,
# os.cpu_count() == 192, OMP_NUM_THREADS unset.
affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
requested = os.environ.get("SLURM_CPUS_PER_TASK")
print("\\ncpus on node   :", os.cpu_count())
print("affinity mask  :", affinity)
print("slurm requested:", requested or "(not under slurm)")
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS", "(unset)"))
record("allocation", affinity=affinity, slurm_cpus_per_task=requested)

_omp = os.environ.get("OMP_NUM_THREADS")
if not _omp and affinity and os.cpu_count() and affinity < os.cpu_count():
    print("\\n  OK: the affinity mask is smaller than the node, so OpenMP sees")
    print("      only your allocation even with OMP_NUM_THREADS unset.")
elif not _omp:
    print("\\n  \\u26a0 CPU TIMINGS BELOW ARE NOT TRUSTWORTHY.")
    print("      OpenMP can see every core on this node but Slurm has given you")
    print(f"      {requested or 'fewer'}. It will spawn one thread per visible core and")
    print("      thrash them, making the CPU leg slower than it should be — and")
    print("      every GPU speedup correspondingly flattering.")
    print("      Fix, then re-run this notebook:")
    print("        export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK")
'''

CELL_TRAPS = '''# ── 2. Environment traps ─────────────────────────────────────────────────
# Both of these let every other check pass while calculations quietly run on
# the CPU — and neither reports an error, because the job converges and the
# energy is correct.
banner("2. ENVIRONMENT TRAPS")

disable = os.environ.get("QUANTUI_DISABLE_GPU", "")
print(f"QUANTUI_DISABLE_GPU  : {disable or '(unset)'}",
      "  <-- WOULD FORCE CPU" if disable else "  OK")

settings_path = os.environ.get(
    "QUANTUI_SETTINGS_PATH", os.path.expanduser("~/.quantui/settings.json"))
print(f"settings file        : {settings_path}")
gpu_enabled = None
if os.path.exists(settings_path):
    try:
        cfg = json.load(open(settings_path))
        gpu_enabled = cfg.get("compute", {}).get("gpu_enabled")
        print(f"compute.gpu_enabled  : {gpu_enabled}",
              "  <-- WOULD FORCE CPU" if gpu_enabled is False else "  OK")
    except Exception as exc:
        print(f"  (unreadable: {exc} — QuantUI treats this as enabled)")
else:
    print("compute.gpu_enabled  : (no file — defaults to enabled)  OK")
    print("  Good: this run is isolated from whatever is saved in $HOME.")

if gpu_enabled is False:
    print("\\n  FIX: relaunch with --env QUANTUI_SETTINGS_PATH=$TMPDIR/q.json")
    print("       Apptainer bind-mounts $HOME, so a preference saved on any")
    print("       other machine follows you onto this node.")

record("traps", quantui_disable_gpu=disable or None,
       settings_path=settings_path, gpu_enabled=gpu_enabled)
'''

CELL_CUPY = '''# ── 3. CuPy reaches the device ───────────────────────────────────────────
# The first check that can fail while nvidia-smi looks fine: a device is
# visible but the wheel line does not match the driver. Also the first FP64
# arithmetic, which is what quantum chemistry actually runs on.
banner("3. CUPY / DRIVER ABI")

try:
    import cupy
    drv = cupy.cuda.runtime.driverGetVersion()
    rt = cupy.cuda.runtime.runtimeGetVersion()
    n = cupy.cuda.runtime.getDeviceCount()
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"cupy            : {cupy.__version__}")
    print(f"driver version  : {drv}   (0 means the driver is unreachable)")
    print(f"runtime version : {rt}")
    print(f"devices         : {n}")
    print(f"device 0        : {name}")

    a = cupy.arange(4_000_000, dtype=cupy.float64)
    cupy.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    val = float((a * a).sum())
    cupy.cuda.Stream.null.synchronize()
    print(f"fp64 kernel     : {val:.6e}  ({time.perf_counter()-t0:.3f}s)")
    print("\\nPASS — CuPy allocated and ran FP64 on the device.")
    record("cupy", ok=True, version=cupy.__version__, driver=drv, runtime=rt,
           devices=n, device_name=name)
except Exception as exc:
    print(f"FAIL — {type(exc).__name__}: {exc}")
    print("\\n  cudaErrorInsufficientDriver here means the driver is older than")
    print("  the CUDA runtime in the image. This image ships cuda12x wheels,")
    print("  which work on a 570-series driver and anything newer.")
    record("cupy", ok=False, error=f"{type(exc).__name__}: {exc}")
'''

CELL_PROBE = '''# ── 4. QuantUI's own probe ───────────────────────────────────────────────
# Note this is the FIRST place gpu4pyscf gets imported. That import calls
# getDeviceCount() at module level, so it cannot run without a visible device —
# which is why the container's build-time tests deliberately never import it.
banner("4. QUANTUI GPU PROBE")

from quantui.gpu_offload import probe_gpu, is_gpu_available, is_low_fp64_device

is_gpu_available.cache_clear()
available, gpu_name, reason = probe_gpu()
print(f"available : {available}")
print(f"device    : {gpu_name}")
print(f"reason    : {reason or '(none — GPU is usable)'}")

ok, out = sh("quantui gpu check")
print(f"\\n$ quantui gpu check   (exit {'0' if ok else 'non-zero'})")
print("  " + out.replace("\\n", "\\n  "))

record("probe", available=available, device=gpu_name, reason=reason or None,
       cli_ok=ok)
print("\\nPASS — QuantUI can offload." if available
      else "\\nFAIL — the reason above is the thing to fix.")
'''

CELL_FP64 = '''# ── 5. FP64 classification ───────────────────────────────────────────────
# QuantUI warns when a GPU has crippled double precision, because PySCF is FP64
# throughout and a consumer card can be genuinely slower than a many-core CPU.
#
# Unknown devices are treated as low-FP64 BY DESIGN, so the risk on a cluster is
# a datacenter card whose name string is not in _STRONG_FP64_MARKERS: students
# would get a wrong and very visible warning. This asserts the expected target
# specifically rather than "nothing is flagged" — on a consumer card the flag is
# correct, and calling that a failure would be nonsense.
banner("5. FP64 CLASSIFICATION")

# Datacenter families whose FP64 is ~1/2 of FP32. Listed independently of
# quantui's own markers on purpose: comparing that list against itself would
# prove nothing.
DATACENTER = ("H200", "H100", "A100", "A30", "B100", "B200", "GB200", "GH200",
              "V100", "P100")

if gpu_name:
    low = is_low_fp64_device(gpu_name)
    expected_strong = any(m in gpu_name.upper() for m in DATACENTER)
    print(f"device            : {gpu_name}")
    print(f"flagged low-FP64  : {low}")
    print(f"datacenter class  : {expected_strong}")

    if expected_strong and low:
        ok_fp64 = False
        print("\\n  FAIL — a datacenter GPU is being flagged as low-FP64.")
        print("  Its name is missing from _STRONG_FP64_MARKERS in")
        print("  quantui/gpu_offload.py, so users here would see a spurious")
        print("  'offload may be SLOWER than your CPU' advisory. Report it.")
    elif expected_strong:
        ok_fp64 = True
        print("\\n  PASS — datacenter GPU correctly treated as strong-FP64,")
        print("  so no spurious advisory. This is the H200 case.")
    else:
        ok_fp64 = True
        print(f"\\n  {'PASS' if low else 'NOTE'} — this is not a datacenter card, so the")
        print("  low-FP64 flag is CORRECT and the advisory is wanted: PySCF is")
        print("  FP64 throughout and offload here may well be slower than CPU.")
        print("  (Section 8 measures whether it actually is.)")
    record("fp64", device=gpu_name, flagged_low_fp64=low,
           datacenter_class=expected_strong, classification_correct=ok_fp64)
else:
    print("skipped — no device detected in section 4")
    record("fp64", classification_correct=None)
'''

CELL_NEGATIVE = '''# ── 6. Negative control ──────────────────────────────────────────────────
# A SUBPROCESS, not an in-process toggle: QUANTUI_DISABLE_GPU is read when the
# probe first runs and the result is cached, so flipping os.environ here would
# not honestly reproduce a CPU-only start. A fresh interpreter does.
#
# If this does not flip the answer, the toggle is broken — and every CPU-vs-GPU
# comparison built on it (section 8) would be measuring nothing.
banner("6. NEGATIVE CONTROL")

code = ("from quantui.gpu_offload import probe_gpu;"
        "a,n,r = probe_gpu(); print(a); print(r)")
env = dict(os.environ, QUANTUI_DISABLE_GPU="1")
p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                   text=True, env=env, timeout=300)
lines = p.stdout.strip().splitlines()
flipped = bool(lines) and lines[0].strip() == "False"
print("with QUANTUI_DISABLE_GPU=1:")
print("  available :", lines[0] if lines else "(no output)")
print("  reason    :", lines[1] if len(lines) > 1 else "(none)")

names_env = len(lines) > 1 and "QUANTUI_DISABLE_GPU" in lines[1]
if flipped and names_env:
    print("\\nPASS — CPU fallback is reachable AND the reason names the env var.")
elif flipped:
    print("\\nPARTIAL — GPU is off, but for some other reason (see section 4).")
else:
    print("\\nFAIL — the toggle did not disable the GPU.")
record("negative_control", flipped=flipped, reason_names_env_var=names_env)
'''

CELL_REAL = '''# ── 7. A real calculation ────────────────────────────────────────────────
# The only claim that matters. Everything above can pass while calculations
# still run on the CPU.
banner("7. REAL CALCULATION")

from quantui import Molecule, run_in_session

WATER = Molecule(
    atoms=["O", "H", "H"],
    coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
)

t0 = time.perf_counter()
r = run_in_session(WATER, method="RHF", basis="sto-3g", verbose=0)
elapsed = time.perf_counter() - t0

print(f"formula   : {r.formula}")
print(f"converged : {r.converged}  in {r.n_iterations} iterations")
print(f"energy    : {r.energy_hartree:.10f} Ha")
print(f"gap       : {r.homo_lumo_gap_ev:.4f} eV")
print(f"gpu_used  : {r.gpu_used}")
print(f"gpu_name  : {r.gpu_name}")
print(f"wall      : {elapsed:.3f}s")

record("real_calculation", gpu_used=bool(r.gpu_used), gpu_name=r.gpu_name,
       energy=r.energy_hartree, converged=bool(r.converged), seconds=elapsed)

if r.gpu_used:
    print("\\nPASS — this calculation took the GPU path.")
else:
    print("\\nFAIL — it fell back to CPU.")
    print("  Note it still CONVERGED and the energy is correct. That is the")
    print("  point: a right answer is not evidence the GPU was used.")
'''

CELL_CROSSOVER = '''# ── 8. CPU vs GPU crossover ──────────────────────────────────────────────
# Small systems lose on GPU: kernel-launch and host<->device transfer dominate
# the actual arithmetic. As the basis grows, integral/Fock work grows faster
# than that overhead and the GPU wins. Where the crossover sits depends on
# molecule, basis, method AND hardware — which is why it has to be measured
# here rather than assumed.
#
# Each leg runs in a SUBPROCESS so the CPU leg genuinely starts with the GPU
# disabled before any import, rather than toggling a cached probe.
#
# Start with SYSTEMS as-is (~2-5 min total). Uncomment the cc-pVTZ row for a
# wider gap; it is markedly slower on CPU.
banner("8. CPU vs GPU CROSSOVER")

BENZENE_XYZ = [
        __BENZENE__,
]

SYSTEMS = [
    # label,            molecule,  method, basis
    ("H2O / STO-3G",    "water",   "RHF",  "sto-3g"),
    ("H2O / cc-pVDZ",   "water",   "RHF",  "cc-pvdz"),
    ("C6H6 / 6-31G",    "benzene", "RHF",  "6-31g"),
    ("C6H6 / cc-pVDZ",  "benzene", "RHF",  "cc-pvdz"),
    # ("C6H6 / cc-pVTZ",  "benzene", "RHF",  "cc-pvtz"),   # slow on CPU
]

LEG = r"""
import json, sys, time
from quantui import Molecule, run_in_session
name, method, basis = sys.argv[1], sys.argv[2], sys.argv[3]
if name == "water":
    mol = Molecule(atoms=["O","H","H"],
                   coordinates=[[0,0,0],[0.757,0.587,0],[-0.757,0.587,0]])
else:
    xyz = json.loads(sys.argv[4])
    mol = Molecule(atoms=[a[0] for a in xyz],
                   coordinates=[[a[1],a[2],a[3]] for a in xyz])
t0 = time.perf_counter()
r = run_in_session(mol, method=method, basis=basis, verbose=0)
print(json.dumps({"seconds": time.perf_counter()-t0, "energy": r.energy_hartree,
                  "gpu_used": bool(r.gpu_used), "converged": bool(r.converged)}))
"""

def leg(mol_name, method, basis, force_cpu, timeout=1800):
    env = dict(os.environ)
    if force_cpu:
        env["QUANTUI_DISABLE_GPU"] = "1"
    else:
        env.pop("QUANTUI_DISABLE_GPU", None)
    args = [sys.executable, "-c", LEG, mol_name, method, basis,
            json.dumps(BENZENE_XYZ)]
    try:
        p = subprocess.run(args, capture_output=True, text=True, env=env,
                           timeout=timeout)
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"error": (p.stderr or p.stdout).strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}

rows = []
print(f"{'system':<18} {'GPU (s)':>10} {'CPU (s)':>10} {'speedup':>9}  verdict")
print("-" * 68)
for label, mol_name, method, basis in SYSTEMS:
    g = leg(mol_name, method, basis, force_cpu=False)
    c = leg(mol_name, method, basis, force_cpu=True)
    if "error" in g or "error" in c:
        print(f"{label:<18} {'ERROR':>10}  {g.get('error') or c.get('error')}")
        rows.append({"system": label, "error": g.get("error") or c.get("error")})
        continue
    # A GPU leg that silently fell back makes the comparison meaningless.
    fell_back = not g["gpu_used"]
    speed = c["seconds"] / g["seconds"] if g["seconds"] else float("nan")
    verdict = ("GPU leg FELL BACK — not a comparison" if fell_back
               else "GPU wins" if speed > 1.05
               else "CPU wins" if speed < 0.95 else "tie")
    print(f"{label:<18} {g['seconds']:>10.2f} {c['seconds']:>10.2f} "
          f"{speed:>8.2f}x  {verdict}")
    agree = abs(g["energy"] - c["energy"]) < 1e-6
    if not agree:
        print(f"{'':18} WARNING: energies differ by "
              f"{abs(g['energy']-c['energy']):.2e} Ha")
    rows.append({"system": label, "gpu_seconds": g["seconds"],
                 "cpu_seconds": c["seconds"], "speedup": speed,
                 "gpu_used": g["gpu_used"], "energies_agree": agree,
                 "verdict": verdict})

record("crossover", systems=rows)
print("\\nThe crossover is where speedup passes 1.0x. Below it the GPU loses to")
print("launch and transfer overhead; above it the arithmetic dominates.")

wins = [r for r in rows if r.get("speedup", 0) > 1.05]
if not wins:
    best = max((r for r in rows if "speedup" in r),
               key=lambda r: r["speedup"], default=None)
    print("\\nNO CROSSOVER in this range — the GPU lost every case.")
    if best:
        print(f"  Closest: {best['system']} at {best['speedup']:.2f}x.")
    print("  On a datacenter GPU, uncomment the cc-pVTZ row and rerun.")
    print("  On a consumer card this is the expected result, not a fault:")
    print("  FP64 is gated to ~1/32-1/64 of FP32 on those parts.")
else:
    first = wins[0]
    print(f"\\nCROSSOVER: GPU first wins at {first['system']} "
          f"({first['speedup']:.2f}x).")
'''.replace("__BENZENE__", BENZENE)

CELL_SUMMARY = '''# ── Summary and record ───────────────────────────────────────────────────
banner("SUMMARY")

s = REPORT["sections"]
def verdict(cond, yes="PASS", no="FAIL"):
    return yes if cond else no

checks = [
    ("hardware visible",   s.get("hardware", {}).get("nvidia_smi_ok")),
    ("no env trap",        not s.get("traps", {}).get("quantui_disable_gpu")
                           and s.get("traps", {}).get("gpu_enabled") is not False),
    ("cupy reaches device", s.get("cupy", {}).get("ok")),
    ("quantui can offload", s.get("probe", {}).get("available")),
    ("fp64 classification", s.get("fp64", {}).get("classification_correct")),
    ("negative control",   s.get("negative_control", {}).get("flipped")),
    ("gpu_used on a run",  s.get("real_calculation", {}).get("gpu_used")),
]
for label, ok in checks:
    print(f"  [{verdict(bool(ok))}]  {label}")

REPORT["summary"] = {label: bool(ok) for label, ok in checks}
REPORT["all_passed"] = all(bool(ok) for _, ok in checks)

out = os.path.join(os.environ.get("QUANTUI_RESULTS_DIR", os.getcwd()),
                   "ncshare-gpu-diagnostic.json")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as fh:
    json.dump(REPORT, fh, indent=2, default=str)

print("\\n" + "=" * 68)
print("PASTE-ABLE RECORD")
print("=" * 68)
hw = s.get("hardware", {})
print(f"date        : {REPORT['generated_utc']}")
print(f"gpu         : {hw.get('gpu_name')}")
print(f"driver      : {hw.get('driver_version')}      <-- the open question")
print(f"compute cap : {hw.get('compute_capability')}")
print(f"quantui     : {hw.get('quantui_version')}")
print(f"cupy/driver : {s.get('cupy', {}).get('version')} / "
      f"driver API {s.get('cupy', {}).get('driver')}")
print(f"partition   : {hw.get('slurm', {}).get('SLURM_JOB_PARTITION')}")
print(f"gres        : {hw.get('slurm', {}).get('SLURM_JOB_GRES')}")
print(f"all passed  : {REPORT['all_passed']}")
print(f"\\nfull JSON   : {out}")
'''


_ids = iter(f"cell-{i:02d}" for i in range(100))


def code(src: str) -> dict:
    return {
        "id": next(_ids),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.rstrip("\n").splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "id": next(_ids),
        "cell_type": "markdown",
        "metadata": {},
        "source": src.rstrip("\n").splitlines(keepends=True),
    }


nb = {
    "cells": [
        md(MD_INTRO),
        code(CELL_SETUP),
        code(CELL_HW),
        code(CELL_TRAPS),
        code(CELL_CUPY),
        code(CELL_PROBE),
        code(CELL_FP64),
        code(CELL_NEGATIVE),
        code(CELL_REAL),
        code(CELL_CROSSOVER),
        code(CELL_SUMMARY),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
