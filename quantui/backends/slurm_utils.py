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


def estimate_slurm_resources(request: CalculationRequest) -> Dict[str, Any]:
    """
    Heuristic cores / memory / walltime for a batch submission.

    Seed logic from legacy ``PySCFCalculation.estimate_resources()``, extended
    with coarse calc-type multipliers. NCShare operators should tune defaults
    after the CL2.2 spike.
    """
    mol = request.molecule
    atoms = mol.get("atoms") or []
    num_atoms = len(atoms)
    charge = int(mol.get("charge", 0))
    mult = int(mol.get("multiplicity", 1))
    num_electrons = sum(_ATOMIC_NUMBERS.get(str(a).title(), 0) for a in atoms) - charge

    basis_factors = {
        "STO-3G": 1.0,
        "3-21G": 1.2,
        "6-31G": 1.5,
        "6-31G*": 2.0,
        "6-31G**": 2.5,
        "cc-pVDZ": 3.0,
        "cc-pVTZ": 5.0,
    }
    basis_factor = basis_factors.get(request.basis, 2.0)

    method_upper = request.method.upper()
    method_factor = 1.2 if method_upper == "UHF" else 1.0
    if method_upper in ("MP2", "CCSD", "CCSD(T)"):
        method_factor = max(method_factor, 2.5)
    elif method_upper not in ("RHF", "UHF"):
        method_factor = max(method_factor, 1.3)

    calc_factors = {
        "single_point": 1.0,
        "geometry_opt": 2.5,
        "frequency": 4.0,
        "tddft": 2.0,
        "nmr": 2.0,
        "pes_scan": 3.0,
        "reorganization_energy": 6.0,
    }
    calc_factor = calc_factors.get(request.calc_type, 1.5)

    base_memory = max(
        4, int(2 * (max(num_electrons, 1) / 10) * basis_factor * method_factor)
    )
    memory_gb = min(int(base_memory * calc_factor), cfg.MAX_MEMORY_GB)

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

    # M-CLUSTER2 CL2.9 — QUANTUI_FREQ_PARALLEL awareness. The base `frequency`
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
    if request.calc_type == "frequency":
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
            memory_gb = min(memory_gb * freq_parallel_multiplier, cfg.MAX_MEMORY_GB)

    return {
        "cores": cores,
        "memory_gb": memory_gb,
        "walltime": walltime,
        "freq_parallel_memory_multiplier": freq_parallel_multiplier,
    }


_ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Fe": 26,
    "Cu": 29,
    "Zn": 30,
    "Br": 35,
    "I": 53,
}
