"""
Serialize batch-worker results to staging JSON for SLURM ingest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import CalculationRequest


def molecule_to_dict(molecule) -> dict[str, Any]:
    return {
        "atoms": list(molecule.atoms),
        "coords": [list(map(float, row)) for row in molecule.coordinates],
        "charge": int(molecule.charge),
        "multiplicity": int(molecule.multiplicity),
    }


def molecule_from_dict(data: dict[str, Any]):
    from quantui.molecule import Molecule

    coords = data.get("coords") or data.get("coordinates")
    if not coords:
        raise ValueError("molecule dict must include 'coords' or 'coordinates'")
    return Molecule(
        atoms=data["atoms"],
        coordinates=coords,
        charge=int(data.get("charge", 0)),
        multiplicity=int(data.get("multiplicity", 1)),
    )


def molecule_from_request(request: CalculationRequest):
    mol_data = dict(request.molecule)
    mol_data.setdefault("charge", request.charge)
    mol_data.setdefault("multiplicity", request.multiplicity)
    return molecule_from_dict(mol_data)


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
        # M-ISSUES ISSUE.10 — session_calc.run_in_session() computes these on
        # every SessionResult, but this function used to drop them before
        # they ever reached result.json: CHEM-3200's Lab 2 In-Lab Part A and
        # Postwork Q2 need the metal's Mulliken charge and the total dipole
        # moment from each single-point batch result, and neither was
        # recoverable as-written. getattr-guarded so an older SessionResult
        # (e.g. the GPU-offload path, where mf.mulliken_pop is
        # NotImplemented and these are already None on the object itself)
        # still serializes cleanly. See GOTCHAS.md.
        "mulliken_charges": getattr(result, "mulliken_charges", None),
        "dipole_moment_debye": getattr(result, "dipole_moment_debye", None),
        "dipole_vector_debye": getattr(result, "dipole_vector_debye", None),
        "atom_symbols": getattr(result, "atom_symbols", None),
        # M-SCF-ROBUST SCFR.5 provenance — "none"/"bootstrap"/"level_shift"/
        # "failed" (quantui.scf_robust.SCF_RESCUE_*). getattr-guarded so an
        # older SessionResult (pre-SCFR) still serializes cleanly.
        "scf_rescue_stage": getattr(result, "scf_rescue_stage", "none"),
        # M-UX2 UXP2.10 — see the other *_result_payload functions' matching
        # field.
        "scf_variant": getattr(result, "scf_variant", "") or None,
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
        # M-UX2 UXP2.10 — see session_result_payload's matching field.
        "scf_variant": getattr(result, "scf_variant", "") or None,
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


def tddft_result_payload(result) -> Dict[str, Any]:
    wavelengths = result.wavelengths_nm()
    return {
        "calc_type": "tddft",
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": result.homo_lumo_gap_ev,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        # M-UX2 UXP2.10 — see session_result_payload's matching field.
        "scf_variant": getattr(result, "scf_variant", "") or None,
        "spectra": {
            "uv_vis": {
                "excitation_energies_ev": list(result.excitation_energies_ev),
                "oscillator_strengths": list(result.oscillator_strengths),
                "wavelengths_nm": list(wavelengths),
            }
        },
    }


def nmr_result_payload(result) -> Dict[str, Any]:
    return {
        "calc_type": "nmr",
        "energy_hartree": float("nan"),
        "homo_lumo_gap_ev": None,
        "converged": result.converged,
        "n_iterations": -1,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        # M-UX2 UXP2.10 — see session_result_payload's matching field.
        "scf_variant": getattr(result, "scf_variant", "") or None,
        "spectra": {
            "nmr": {
                "atom_symbols": list(result.atom_symbols),
                "shielding_iso_ppm": list(result.shielding_iso_ppm),
                "chemical_shifts_ppm": {
                    str(k): v for k, v in result.chemical_shifts_ppm.items()
                },
                "reference_compound": result.reference_compound,
                "reference_key": result.reference_key,
                "is_fallback_reference": result.is_fallback_reference,
            }
        },
    }


def pes_scan_result_payload(result, *, trajectory_file: str) -> Dict[str, Any]:
    return {
        "calc_type": "pes_scan",
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": None,
        "converged": result.converged_all,
        "n_iterations": -1,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        "trajectory_file": trajectory_file,
        "spectra": {
            "pes_scan": {
                "scan_type": result.scan_type,
                "atom_indices": list(result.atom_indices),
                "scan_parameter_values": list(result.scan_parameter_values),
                "energies_hartree": list(result.energies_hartree),
            }
        },
    }


def reorg_result_payload(result) -> Dict[str, Any]:
    neutral = molecule_to_dict(result.molecule)
    channels = []
    for ch in result.channels:
        entry = {
            "kind": ch.kind,
            "ion_charge": ch.ion_charge,
            "ion_multiplicity": ch.ion_multiplicity,
            "e_neutral_at_neutral": ch.e_neutral_at_neutral,
            "e_ion_at_ion": ch.e_ion_at_ion,
            "e_ion_at_neutral": ch.e_ion_at_neutral,
            "e_neutral_at_ion": ch.e_neutral_at_ion,
            "lambda1_hartree": ch.lambda1_hartree,
            "lambda2_hartree": ch.lambda2_hartree,
            "lambda_hartree": ch.lambda_hartree,
            "converged": ch.converged,
        }
        if ch.ion_molecule is not None:
            entry["ion_geometry"] = molecule_to_dict(ch.ion_molecule)
        channels.append(entry)
    return {
        "calc_type": "reorganization_energy",
        "energy_hartree": result.energy_hartree,
        "homo_lumo_gap_ev": None,
        "converged": result.converged,
        "n_iterations": result.n_total_opt_steps,
        "method": result.method,
        "basis": result.basis,
        "formula": result.formula,
        "neutral_geometry": neutral,
        "channels": channels,
        "spectra": result.to_spectra(),
    }


def write_worker_result(staging_dir: Path, payload: Dict[str, Any]) -> Path:
    result_path = staging_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    return result_path


def write_analysis_artifacts(
    staging_dir: Path,
    calc_type: str,
    result,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    trajectory: list | None = None,
    energies: list | None = None,
) -> None:
    """Write Analysis-replay sidecars into *staging_dir* (mirrors ``app._do_run``).

    Persists MO data, Molden, and external trajectory formats so SLURM ingest
    can promote them into History the same way local runs do.
    """
    from quantui.results_storage import (
        save_molden,
        save_orbitals,
        save_trajectory_ase,
        save_trajectory_xyz,
    )

    if calc_type in ("single_point", "geometry_opt", "frequency"):
        save_orbitals(staging_dir, result)
        try:
            save_molden(
                staging_dir,
                mo_energy_hartree=getattr(result, "mo_energy_hartree", None),
                mo_occ=getattr(result, "mo_occ", None),
                mo_coeff=getattr(result, "mo_coeff", None),
                pyscf_mol_atom=getattr(result, "pyscf_mol_atom", None),
                pyscf_mol_basis=getattr(result, "pyscf_mol_basis", None),
                charge=int(charge),
                multiplicity=int(multiplicity),
                frequencies_cm1=getattr(result, "frequencies_cm1", None),
                normal_modes=getattr(result, "displacements", None),
            )
        except Exception:
            pass

    if calc_type in ("geometry_opt", "pes_scan") and trajectory:
        e_list = list(energies or [])
        try:
            save_trajectory_xyz(staging_dir, frames=trajectory, energies=e_list)
            save_trajectory_ase(staging_dir, frames=trajectory, energies=e_list)
        except Exception:
            pass


def unsupported_calc_payload(calc_type: str) -> Dict[str, Any]:
    return {
        "calc_type": calc_type,
        "supported": False,
    }
