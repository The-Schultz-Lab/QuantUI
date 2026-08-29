"""
Serialize batch-worker results to staging JSON for SLURM ingest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import CalculationRequest


def molecule_from_request(request: CalculationRequest):
    from quantui.molecule import Molecule

    mol_data = request.molecule
    coords = mol_data.get("coords") or mol_data.get("coordinates")
    if not coords:
        raise ValueError("molecule dict must include 'coords' or 'coordinates'")
    return Molecule(
        atoms=mol_data["atoms"],
        coordinates=coords,
        charge=int(mol_data.get("charge", request.charge)),
        multiplicity=int(mol_data.get("multiplicity", request.multiplicity)),
    )


def write_trajectory_json(
    staging_dir: Path,
    trajectory: list,
    energies: list,
    *,
    filename: str = "trajectory.json",
) -> str:
    from quantui.results_storage import save_trajectory

    save_trajectory(staging_dir, trajectory, energies, filename=filename)
    return filename


def session_result_payload(result) -> Dict[str, Any]:
    return {
        "calc_type": "single_point",
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": result.homo_lumo_gap_ev,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
    }


def optimization_result_payload(result, *, trajectory_file: str) -> Dict[str, Any]:
    energy = getattr(result, "energy_hartree", None)
    if energy is None:
        energies = getattr(result, "energies_hartree", None) or []
        energy = energies[-1] if energies else float("nan")
    return {
        "calc_type": "geometry_opt",
        "energy_hartree": energy,
        "homo_lumo_gap_ev": getattr(result, "homo_lumo_gap_ev", None),
        "converged": result.converged,
        "n_iterations": getattr(result, "n_iterations", result.n_steps),
        "n_steps": result.n_steps,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        "trajectory_file": trajectory_file,
    }


def freq_result_payload(result, molecule) -> Dict[str, Any]:
    displacements = None
    if result.displacements is not None:
        try:
            import numpy as np

            displacements = np.asarray(result.displacements).tolist()
        except Exception:
            displacements = None
    return {
        "calc_type": "frequency",
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": result.homo_lumo_gap_ev,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        "spectra": {
            "ir": {
                "frequencies_cm1": list(result.frequencies_cm1),
                "ir_intensities": list(result.ir_intensities),
                "raman_activities": list(getattr(result, "raman_activities", []) or []),
                "zpve_hartree": result.zpve_hartree,
                "displacements": displacements,
            },
            "molecule": {
                "atoms": list(molecule.atoms),
                "coords": [list(map(float, row)) for row in molecule.coordinates],
                "charge": molecule.charge,
                "multiplicity": molecule.multiplicity,
            },
        },
    }


def write_worker_result(staging_dir: Path, payload: Dict[str, Any]) -> Path:
    result_path = staging_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    return result_path


def unsupported_calc_payload(calc_type: str) -> Dict[str, Any]:
    return {
        "calc_type": calc_type,
        "supported": False,
    }
