"""
SLURM helper utilities salvaged from the legacy QuantUI archive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import cluster_config as cfg
from .base import CalculationRequest


@dataclass(frozen=True)
class SlurmJobAccounting:
    """Normalized SLURM accounting row for one batch job."""

    state: str
    exit_code: str = ""
    elapsed: str = ""


# Short calc-type tags for job dir names.
_CALC_TYPE_TAGS = {
    "single_point": "sp",
    "geometry_opt": "opt",
    "transition_state": "ts",
    "irc": "irc",
    "frequency": "freq",
    "tddft": "tddft",
    "nmr": "nmr",
    "pes_scan": "scan",
    "reorganization_energy": "reorg",
}

_JOB_NAME_MAX_LEN = 80
# SLURM itself accepts longer names, but squeue's default format column is
# narrow; the job dir keeps the full name.
SLURM_JOB_NAME_MAX_LEN = 40


def sanitize_job_name(name: str) -> str:
    """Make *name* safe as a job dir name and SLURM ``--job-name``.

    Characters other than letters, digits, ``-`` and ``_`` become ``_``;
    leading/trailing separators are stripped. Returns ``""`` when nothing
    usable is left.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned[:_JOB_NAME_MAX_LEN].rstrip("_-")


def job_name_basis(basis: str) -> str:
    """A basis set name as it appears in job and folder names.

    Pople polarisation is written the way chemists abbreviate it:
    ``6-31G*`` / ``6-31G(d)`` → ``6-31Gd``, ``6-31G**`` / ``6-31G(d,p)`` →
    ``6-31Gdp``, and ``+`` → ``p`` (``6-31+G*`` → ``6-31pGd``). Before, every
    such character became ``x`` (``6-31Gxdxpx``). Still distinct from
    ``6-31G``. Any other unsafe character becomes ``x``.
    """
    if basis.endswith("**"):
        basis = basis[:-2] + "(d,p)"
    elif basis.endswith("*"):
        basis = basis[:-1] + "(d)"
    basis = re.sub(r"[(),]", "", basis).replace("+", "p")
    return re.sub(r"[^\w\-]", "x", basis)


def default_job_name(request: CalculationRequest) -> str:
    """``<label>_<calc tag>_<method>_<basis>``, e.g. ``H2O_opt_B3LYP_def2-SVP``.

    Unsafe characters become ``x`` as in History result dirs
    (``results_storage._safe_name``); the basis is written by
    :func:`job_name_basis`.
    """
    from quantui.results_storage import _safe_name

    label = str(request.molecule.get("label") or "quantui")
    tag = _CALC_TYPE_TAGS.get(request.calc_type, request.calc_type)
    parts = [
        _safe_name(label),
        _safe_name(tag),
        _safe_name(request.method),
        job_name_basis(request.basis) if request.basis else "",
    ]
    name = "_".join(p for p in parts if p)
    return sanitize_job_name(name) or "quantui"


def format_walltime(hours: float) -> str:
    total_seconds = int(hours * 3600)
    h = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{h:02d}:{minutes:02d}:{seconds:02d}"


def parse_slurm_job_id(sbatch_output: str) -> Optional[str]:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if match:
        return match.group(1)
    return None


def parse_sacct_accounting(sacct_output: str) -> Dict[str, SlurmJobAccounting]:
    """Parse ``sacct -P --format=JobID,State,ExitCode,Elapsed`` output."""
    rows: Dict[str, SlurmJobAccounting] = {}
    for line in sacct_output.splitlines():
        line = line.strip()
        if not line or line.startswith("JobID|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        state = parts[1].strip().upper()
        if not job_id or not state or "." in job_id:
            continue
        exit_code = parts[2].strip() if len(parts) > 2 else ""
        elapsed = parts[3].strip() if len(parts) > 3 else ""
        rows[job_id] = SlurmJobAccounting(
            state=state, exit_code=exit_code, elapsed=elapsed
        )
    return rows


def parse_sacct_states(sacct_output: str) -> Dict[str, str]:
    """Parse ``sacct -P --format=JobID,State`` lines into a job-id → state map."""
    return {
        job_id: row.state
        for job_id, row in parse_sacct_accounting(sacct_output).items()
    }


# Resource-estimate factors. Module-level so ``quantui install-launcher`` can
# copy them into the host-side ``quantui-batch`` launcher, which re-implements
# this estimate without importing QuantUI (tests/test_batch_submit.py checks
# that the two agree).
# Relative memory per basis. Not calibrated against sacct yet; new entries
# are placed by basis size relative to the old ones: def2-SVP has cc-pVDZ's
# functions (C 3s2p1d, H 2s1p); def2-TZVP matches cc-pVTZ on heavy atoms
# but is smaller on H; LANL2DZ is D95V-sized on light atoms (its heavy-atom
# ECP removes core functions).
BASIS_FACTORS = {
    "STO-3G": 1.0,
    "3-21G": 1.2,
    "6-31G": 1.5,
    "6-31G*": 2.0,
    "6-31G**": 2.5,
    "cc-pVDZ": 3.0,
    "cc-pVTZ": 5.0,
    "def2-SVP": 3.0,
    "def2-TZVP": 4.5,
    "LANL2DZ": 1.5,
    # Diffuse sets: more functions than the parent set (on C, aug-cc-pVDZ has
    # 23 vs 14, def2-SVPD 20 vs 14) and less integral screening, so each sits
    # above its parent. def2-SVPD is placed against def2-SVP ~ cc-pVDZ (3.0).
    "6-31+G*": 2.5,
    "6-31++G**": 3.0,
    "aug-cc-pVDZ": 4.0,
    "def2-SVPD": 3.5,
}
DEFAULT_BASIS_FACTOR = 2.0


def basis_factor(basis: str) -> float:
    """BASIS_FACTORS entry for *basis*: case-insensitive, Pople aliases folded.

    ``6-31G(d)`` and ``6-31G(d,p)`` are ``6-31G*`` and ``6-31G**`` written the
    other way, and used to fall back to the generic factor.
    """
    key = str(basis).strip().lower().replace(" ", "")
    if key.endswith("(d,p)"):
        key = key[: -len("(d,p)")] + "**"
    elif key.endswith("(d)"):
        key = key[: -len("(d)")] + "*"
    for name, factor in BASIS_FACTORS.items():
        if name.lower() == key:
            return factor
    return DEFAULT_BASIS_FACTOR


def _walltime_steps_for(count: int, default: int) -> int:
    """One extra walltime step per doubling of *count* above *default*."""
    steps = 0
    while count > default:
        steps += 1
        default *= 2
    return steps


CALC_FACTORS = {
    "single_point": 1.0,
    "geometry_opt": 2.5,
    # A saddle search with analytic Hessians, then a full frequency check.
    "transition_state": 4.0,
    # Gradients along the path both ways, from a Hessian at the start.
    "irc": 3.0,
    "frequency": 4.0,
    "tddft": 2.0,
    "nmr": 2.0,
    "pes_scan": 3.0,
    "reorganization_energy": 6.0,
}


def estimate_slurm_resources(request: CalculationRequest) -> Dict[str, Any]:
    """
    Heuristic cores / memory / walltime for a batch submission.

    Seed logic from legacy ``PySCFCalculation.estimate_resources()``, extended
    with coarse calc-type multipliers. NCShare operators should tune defaults
    against measured ``sacct`` usage.
    """
    mol = request.molecule
    atoms = mol.get("atoms") or []
    num_atoms = len(atoms)
    charge = int(mol.get("charge", 0))
    mult = int(mol.get("multiplicity", 1))
    from quantui.config import electron_count

    # The full periodic table (config.electron_count): the old short list here
    # had no Mn, Co, Ni or Mo, so a transition-metal complex's metal counted
    # as 0 electrons and its memory came out low.
    num_electrons = electron_count(atoms, charge)

    b_factor = basis_factor(request.basis)

    method_upper = request.method.upper()
    method_factor = 1.2 if method_upper == "UHF" else 1.0
    if method_upper in ("MP2", "CCSD", "CCSD(T)"):
        method_factor = max(method_factor, 2.5)
    elif method_upper not in ("RHF", "UHF"):
        method_factor = max(method_factor, 1.3)

    calc_factor = CALC_FACTORS.get(request.calc_type, 1.5)

    base_memory = max(
        4, int(2 * (max(num_electrons, 1) / 10) * b_factor * method_factor)
    )
    memory_gb = min(int(base_memory * calc_factor), cfg.ESTIMATE_MAX_MEMORY_GB)

    if num_atoms < 10:
        cores = 4
    elif num_atoms < 20:
        cores = 8
    else:
        cores = 16
    cores = min(cores, cfg.MAX_CORES)

    if num_atoms < 5:
        walltime = "00:30:00"
    elif num_atoms < 10:
        walltime = "01:00:00"
    elif num_atoms < 20:
        walltime = "02:00:00"
    else:
        walltime = "04:00:00"

    if request.basis in ("cc-pVTZ",):
        time_map = {
            "00:30:00": "01:00:00",
            "01:00:00": "02:00:00",
            "02:00:00": "04:00:00",
            "04:00:00": "08:00:00",
        }
        walltime = time_map.get(walltime, walltime)

    if calc_factor >= 3.0 and walltime in cfg.WALLTIME_OPTIONS:
        idx = cfg.WALLTIME_OPTIONS.index(walltime)
        if idx + 1 < len(cfg.WALLTIME_OPTIONS):
            walltime = cfg.WALLTIME_OPTIONS[idx + 1]

    # Open-shell systems occasionally need extra SCF effort — minor bump.
    if mult > 1 and walltime in cfg.WALLTIME_OPTIONS:
        idx = cfg.WALLTIME_OPTIONS.index(walltime)
        if idx + 1 < len(cfg.WALLTIME_OPTIONS):
            walltime = cfg.WALLTIME_OPTIONS[idx + 1]

    # Work that grows with an option the defaults assume — TD-DFT
    # states (default 10) and PES scan points (default 10, each one an
    # optimization). One walltime step per doubling; memory is unchanged.
    options = request.options or {}
    extra_steps = 0
    try:
        if request.calc_type == "tddft":
            extra_steps = _walltime_steps_for(int(options.get("nstates", 10)), 10)
        elif request.calc_type == "pes_scan":
            extra_steps = _walltime_steps_for(int(options.get("steps", 10)), 10)
    except (TypeError, ValueError):
        extra_steps = 0
    for _ in range(extra_steps):
        if walltime in cfg.WALLTIME_OPTIONS:
            idx = cfg.WALLTIME_OPTIONS.index(walltime)
            if idx + 1 < len(cfg.WALLTIME_OPTIONS):
                walltime = cfg.WALLTIME_OPTIONS[idx + 1]

    # QUANTUI_FREQ_PARALLEL awareness. The base `frequency`
    # estimate above (calc_factor=4.0) already independently validated as
    # correct for the *serial* IR-displacement loop (CHEM-3200 Lab 2: a real
    # 19-atom def2-SVP UKS frequency job needed exactly the 120-128GB this
    # estimator recommends). But that path runs ONE SCF at a time; the
    # QUANTUI_FREQ_PARALLEL=1 opt-in (freq_ir_workers.py) instead fans the
    # per-displacement SCFs out across N worker *processes* running
    # concurrently, each rebuilding its own Mole and holding its own
    # integrals/Fock matrix — memory need scales with N, not with thread
    # count, so the serial estimate alone would silently under-provision.
    # Not yet triggered in the CHEM-3200 case (the env var wasn't set
    # there) — this is a documented latent risk, not (yet) an observed
    # failure. `freq_parallel_memory_multiplier` is always present (1 when
    # the opt-in isn't active, or for any non-frequency calc type) so a
    # caller/log can see whether — and by how much — this fired.
    freq_parallel_multiplier = 1
    # A Transition State ends with the same frequency calculation.
    if request.calc_type in ("frequency", "transition_state"):
        try:
            from quantui.freq_ir_workers import freq_parallel_opt_in, pick_worker_count

            if freq_parallel_opt_in():
                displacement_count = max(num_atoms, 1) * 3 * 2
                freq_parallel_multiplier = max(
                    1, pick_worker_count(cores, displacement_count)
                )
        except Exception:  # noqa: BLE001 — a bad probe must not block sizing
            freq_parallel_multiplier = 1
        if freq_parallel_multiplier > 1:
            memory_gb = min(
                memory_gb * freq_parallel_multiplier, cfg.ESTIMATE_MAX_MEMORY_GB
            )

    return {
        "cores": cores,
        "memory_gb": memory_gb,
        "walltime": walltime,
        "freq_parallel_memory_multiplier": freq_parallel_multiplier,
    }
