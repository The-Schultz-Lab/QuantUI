"""
Structured log header/footer for QuantUI calculation output.

Collects machine metadata (CPU, RAM, GPU, OMP threads), formats a banner
header written before each calculation, and a summary footer written after
with wall/CPU timing, convergence status, key energies, and a warnings digest.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

_WIDTH = 80  # total width of === border lines
_SEP = "=" * _WIDTH
_SUB = "-" * _WIDTH  # lighter divider between header sections

# Static list of formats the app can export a result to (EXPORT milestone).
_EXPORT_FORMATS = "XYZ, MOL/SDF, PDB, Molden, .traj, cube, HTML spectra, .zip"

# SCF/correlation methods that are NOT DFT functionals — shown verbatim in the
# "Method" field with no separate functional. Everything else is treated as a
# DFT functional and reported as RKS/UKS + the functional name.
_NON_DFT_METHODS = frozenset({"RHF", "UHF", "MP2", "CCSD", "CCSD(T)"})

# ASCII-art wordmark shown at the top of every run banner. Rendered in the
# monospace stdout stream of the live log, so the alignment holds. Raw string:
# the art contains backslashes. Trailing whitespace is stripped per line so it
# doesn't trip the linter (the art is the left edge; right padding is cosmetic).
_ASCII_LOGO_LINES = [
    r"   ___                    _   _   _ ___",
    r"  / _ \ _   _  __ _ _ __ | |_| | | |_ _|",
    r" | | | | | | |/ _` | '_ \| __| | | || |",
    r" | |_| | |_| | (_| | | | | |_| |_| || |",
    r"  \__\_\\__,_|\__,_|_| |_|\__|\___/|___|",
]


# ============================================================================
# System-info helpers
# ============================================================================


def _read_proc_cpu() -> str:
    """Return CPU model name from /proc/cpuinfo (Linux/WSL)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _read_proc_ram_gb() -> Optional[float]:
    """Return total RAM in GiB from /proc/meminfo (Linux/WSL)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (OSError, ValueError):
        pass
    return None


def _psutil_ram_gb() -> Optional[float]:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        return None


def _detect_gpu() -> Optional[Dict[str, str]]:
    """Try nvidia-smi first, then cupy.  Returns dict or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                return {"name": parts[0], "mem_mb": parts[1], "driver": parts[2]}
            elif len(parts) == 2:
                return {"name": parts[0], "mem_mb": parts[1], "driver": ""}
            elif len(parts) == 1:
                return {"name": parts[0], "mem_mb": "", "driver": ""}
    except Exception:
        pass

    # cupy fallback
    try:
        import cupy

        n = cupy.cuda.runtime.getDeviceCount()
        if n > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"Unknown GPU")
            if isinstance(name, bytes):
                name = name.decode()
            total_mem_mb = props.get("totalGlobalMem", 0) // (1024 * 1024)
            return {"name": name, "mem_mb": str(total_mem_mb), "driver": ""}
    except Exception:
        pass

    return None


def _pyscf_version() -> str:
    """Return the installed PySCF version, or a clear 'not installed' marker.

    PySCF is Linux/macOS/WSL only, so this is expected to be absent on the
    Windows UI-dev path (see constraint #1).
    """
    try:
        import pyscf

        return str(getattr(pyscf, "__version__", "unknown"))
    except Exception:
        return "not installed"


def collect_system_info() -> Dict[str, Any]:
    """Gather CPU, RAM, GPU, thread count, host, and versions.  Safe anywhere."""
    cpu_model = (
        _read_proc_cpu() or platform.processor() or platform.machine() or "Unknown CPU"
    )
    cpu_count = os.cpu_count() or 1

    ram_gb = _read_proc_ram_gb() or _psutil_ram_gb()
    ram_str = f"{ram_gb:.0f} GB" if ram_gb else "Unknown"

    gpu = _detect_gpu()

    omp = os.environ.get("OMP_NUM_THREADS", None)

    try:
        hostname = socket.gethostname() or "unknown"
    except Exception:
        hostname = "unknown"

    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "ram_str": ram_str,
        "gpu": gpu,
        "omp_threads": omp,
        "hostname": hostname,
        "python": platform.python_version(),
        "pyscf": _pyscf_version(),
    }


@lru_cache(maxsize=1)
def get_system_info() -> Dict[str, Any]:
    """Lazy-cached version of collect_system_info().  Populated on first call."""
    return collect_system_info()


# ============================================================================
# Duration formatter
# ============================================================================


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.t"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


# ============================================================================
# Header
# ============================================================================

_CALC_TYPE_LABELS: Dict[str, str] = {
    "single_point": "Single Point Energy",
    "geometry_opt": "Geometry Optimization",
    "frequency": "Frequency Analysis",
    "tddft": "TD-DFT (UV-Vis)",
    "nmr": "NMR Shielding",
    "pes_scan": "PES Scan",
    "reorganization_energy": "Reorganization Energy (4-point)",
}


def _split_method_functional(
    method: str, multiplicity: int
) -> tuple[str, Optional[str]]:
    """Split the user's method selection into (wf-method, functional).

    HF / post-HF selections are shown verbatim with no functional. A DFT
    functional is reported as its Kohn-Sham class (RKS closed-shell / UKS
    open-shell) plus the functional name, matching PySCF's auto-dispatch.
    """
    if method.upper() in _NON_DFT_METHODS:
        return method, None
    return ("RKS" if multiplicity <= 1 else "UKS"), method


def _row(label: str, value: Any) -> str:
    """One aligned ``  Label           : value`` header row."""
    return f"  {label:<16}: {value}"


def format_log_header(
    *,
    formula: str,
    method: str,
    basis: str,
    calc_type: str,
    timestamp: Optional[str] = None,
    n_atoms: Optional[int] = None,
    multiplicity: int = 1,
    solvent: Optional[str] = None,
    output_dir: Optional[str] = None,
    calc_id: Optional[str] = None,
    starting_energy: Optional[float] = None,
) -> str:
    """Return a formatted header string to prepend to calculation log output.

    Core args (formula/method/basis/calc_type) are required; the rest add
    run-provenance rows (host, versions, device, solvation, output dir, …) and
    default to sensible placeholders so callers can pass only what they know at
    header-write time.
    """
    sysinfo = get_system_info()

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if calc_id is None:
        calc_id = uuid.uuid4().hex[:8]

    ct_label = _CALC_TYPE_LABELS.get(calc_type, calc_type.replace("_", " ").title())
    wf_method, functional = _split_method_functional(method, multiplicity)

    gpu = sysinfo["gpu"]
    if gpu:
        mem = f"  |  {gpu['mem_mb']} MB" if gpu.get("mem_mb") else ""
        drv = f"  |  Driver {gpu['driver']}" if gpu.get("driver") else ""
        device = f"GPU — {gpu['name']}{mem}{drv}"
    else:
        device = "CPU"

    omp = sysinfo["omp_threads"]
    omp_str = (
        f"OMP_NUM_THREADS={omp}" if omp else f"not set  (cores: {sysinfo['cpu_count']})"
    )

    structure = formula if n_atoms is None else f"{formula}  ({n_atoms} atoms)"
    start_e = (
        f"{starting_energy:.8f} Ha"
        if starting_energy is not None
        else "— (available after the first SCF)"
    )

    lines = [
        "",
        _SEP,
        "",
        *_ASCII_LOGO_LINES,
        "",
        "  Quantum Chemistry Interface",
        _SEP,
        _row("Calculation ID", calc_id),
        _row("Timestamp", timestamp),
        _row("Host", sysinfo.get("hostname", "unknown")),
        _row(
            "Python",
            f"{sysinfo.get('python', '?')}   |   "
            f"PySCF: {sysinfo.get('pyscf', '?')}",
        ),
        _row("Device", device),
        _row(
            "Machine",
            f"{sysinfo['cpu_model']}  |  {sysinfo['cpu_count']} cores  "
            f"|  RAM: {sysinfo['ram_str']}",
        ),
        _row("Threads", omp_str),
        _SUB,
        _row("Method", wf_method),
        _row("Functional", functional or "—"),
        _row("Basis Set", basis),
        _row("Solvation", f"PCM ({solvent})" if solvent else "None"),
        _row("Job Type", ct_label),
        _SUB,
        _row("Input Structure", structure),
        _row("Starting Energy", start_e),
        _SUB,
        _row("Output directory", output_dir or "—"),
        _row("Export formats", _EXPORT_FORMATS),
        _SEP,
        "",
    ]
    return "\n".join(lines)


# ============================================================================
# Footer
# ============================================================================


def _extract_warnings(log_text: str) -> list[str]:
    """Return list of unique warning/error lines found in log_text."""
    seen: set[str] = set()
    found = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(
            kw in lower
            for kw in ("warn", "error", "failed", "not converge", "imaginary")
        ):
            if stripped not in seen:
                seen.add(stripped)
                found.append(stripped)
    return found


def format_log_footer(
    *,
    result: Any,
    wall_time: float,
    cpu_time: float,
    log_text: str = "",
    success: bool = True,
) -> str:
    """Return a formatted footer string to append to calculation log output.

    Args:
        result: Any result dataclass (SessionResult, OptResult, FreqResult, etc.)
            or None if the calculation failed.
        wall_time: Elapsed wall-clock seconds.
        cpu_time: Elapsed process CPU seconds.
        log_text: The log body text to scan for warnings.
        success: Whether the calculation completed without an exception.
    """
    from .session_calc import HARTREE_TO_EV  # local import — avoids circular deps

    lines: list[str] = ["", _SEP, "  ── Result " + "─" * (_WIDTH - 12)]

    if result is not None:
        converged = getattr(result, "converged", None)
        n_iter = getattr(result, "n_iterations", None)
        energy = getattr(result, "energy_hartree", None)
        gap_ev = getattr(result, "homo_lumo_gap_ev", None)
        zpve = getattr(result, "zpve_hartree", None)
        n_steps = getattr(result, "n_steps", None)  # OptResult

        # Convergence line
        if converged is not None:
            tick = "✓" if converged else "✗"
            conv_word = "converged" if converged else "did NOT converge"
            iter_str = f"  |  Iterations: {n_iter}" if n_iter is not None else ""
            lines.append(f"  {tick} SCF {conv_word}{iter_str}")
        if n_steps is not None:
            lines.append(f"    Geometry optimization: {n_steps} steps")

        # Energy
        if energy is not None:
            ev = energy * HARTREE_TO_EV
            lines.append(f"    Energy:        {energy:.8f} Ha  ({ev:.4f} eV)")

        # HOMO-LUMO gap
        if gap_ev is not None:
            lines.append(f"    HOMO-LUMO gap: {gap_ev:.4f} eV")

        # ZPVE (frequency only)
        if zpve is not None and zpve != 0.0:
            lines.append(
                f"    ZPVE:          {zpve:.6f} Ha  ({zpve * HARTREE_TO_EV:.4f} eV)"
            )

        # Imaginary frequencies (FreqResult)
        n_imag = None
        try:
            n_imag = result.n_imaginary_modes()  # type: ignore[attr-defined]
        except AttributeError:
            pass
        if n_imag is not None and n_imag > 0:
            lines.append(
                f"    ⚠ {n_imag} imaginary frequency mode(s) — geometry may not be a true minimum"
            )

    # Timing section
    lines.append("  ── Timing " + "─" * (_WIDTH - 12))
    wall_str = _fmt_duration(wall_time)
    cpu_str = _fmt_duration(cpu_time)
    ratio = cpu_time / wall_time if wall_time > 0 else 0.0
    lines.append(
        f"    Wall time: {wall_str}    CPU time: {cpu_str}    CPU/Wall: {ratio:.1f}×"
    )

    # Warnings digest
    lines.append("  ── Warnings Digest " + "─" * (_WIDTH - 22))
    warnings = _extract_warnings(log_text)
    if warnings:
        for w in warnings[:10]:  # cap at 10
            # Truncate very long lines
            w_disp = w if len(w) <= 74 else w[:71] + "..."
            lines.append(f"    ⚠ {w_disp}")
        if len(warnings) > 10:
            lines.append(f"    ... and {len(warnings) - 10} more (see full log)")
    else:
        lines.append("    (none)")

    # Final status line
    lines.append(_SEP)
    if success:
        lines.append("  ✓ Calculation completed successfully.")
    else:
        lines.append("  ✗ Calculation ended with errors — see log above.")
    lines.append(_SEP)
    lines.append("")

    return "\n".join(lines)
