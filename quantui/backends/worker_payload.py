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
        # AUDIT F12 — these were computed onto SessionResult but never
        # reached staging JSON at all (a serialization-layer gap distinct
        # from _basic_result's ingest-layer one): post-HF correlation
        # breakdown, solvent/GPU/density-fit provenance, and the AUDIT
        # F04/F07 dispersion/CC-convergence flags.
        "mp2_correlation_hartree": getattr(result, "mp2_correlation_hartree", None),
        "ccsd_correlation_hartree": getattr(result, "ccsd_correlation_hartree", None),
        "ccsd_t_correction_hartree": getattr(result, "ccsd_t_correction_hartree", None),
        "cc_converged": getattr(result, "cc_converged", None),
        "dispersion_applied": getattr(result, "dispersion_applied", None),
        "solvent": getattr(result, "solvent", None),
        "gpu_used": bool(getattr(result, "gpu_used", False)),
        "gpu_name": getattr(result, "gpu_name", None),
        "density_fit": bool(getattr(result, "density_fit", False)),
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
        "gpu_used": bool(getattr(result, "gpu_used", False)),
        "gpu_name": getattr(result, "gpu_name", None),
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
    _thermo = getattr(result, "thermo", None)
    _thermo_payload = (
        {
            "zpve_hartree": _thermo.zpve_hartree,
            "H_hartree": _thermo.H_hartree,
            "S_jmol": _thermo.S_jmol,
            "G_hartree": _thermo.G_hartree,
            "temperature_k": _thermo.temperature_k,
            # AUDIT F18 — pressure and the thermo model itself were never
            # recorded anywhere; both are fixed by the harmonic-oscillator/
            # rigid-rotor/ideal-gas model at 1 atm used in freq_calc.py.
            "pressure_atm": 1.0,
            "approximation": "ideal_gas_rigid_rotor_harmonic_oscillator",
        }
        if _thermo is not None
        else None
    )
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
        # AUDIT F12 — was never serialized, though FreqResult carries it.
        "density_fit": bool(getattr(result, "density_fit", False)),
        "spectra": {
            "ir": {
                "frequencies_cm1": list(result.frequencies_cm1),
                "ir_intensities": list(result.ir_intensities),
                "raman_activities": list(getattr(result, "raman_activities", []) or []),
                "zpve_hartree": result.zpve_hartree,
                "displacements": displacements,
                # AUDIT F18 — thermo (H, S, G) was computed by freq_calc.py
                # but discarded here; the saved JSON had only frequencies,
                # intensities, activities, displacements, and ZPVE.
                "thermo": _thermo_payload,
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
        # AUDIT F12 — was never serialized, though TDDFTResult carries it.
        "density_fit": bool(getattr(result, "density_fit", False)),
        # AUDIT F08 (code review follow-up) — per-root convergence detail
        # never left the worker process; a SLURM-submitted TDDFT run's
        # History card could show only the folded "converged" bool, never
        # the per-root tally format_tddft_result shows for an interactive
        # run.
        "td_converged": (
            [bool(c) for c in result.td_converged]
            if getattr(result, "td_converged", None) is not None
            else None
        ),
        "n_converged_states": getattr(result, "n_converged_states", None),
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
        # AUDIT F12 — was never serialized, though NMRResult carries it.
        "density_fit": bool(getattr(result, "density_fit", False)),
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
