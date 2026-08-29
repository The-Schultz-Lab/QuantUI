"""Shared helpers for SLURM staging → History ingest tests."""

from __future__ import annotations

import json
from pathlib import Path

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry


def patch_results_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "results"
    monkeypatch.setattr(
        "quantui.results_storage._default_results_dir",
        lambda: root,
    )
    return root


def make_staging_record(
    tmp_path: Path,
    payload: dict,
    *,
    calc_type: str = "single_point",
    request_id: str = "job1",
):
    staging = tmp_path / "staging" / request_id
    staging.mkdir(parents=True, exist_ok=True)
    request = CalculationRequest(
        request_id=request_id,
        calc_type=calc_type,
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={
            "atoms": ["H", "H"],
            "coordinates": [[0, 0, 0], [0, 0, 0.74]],
        },
    )
    (staging / "request.json").write_text(json.dumps(request.to_dict()))
    (staging / "result.json").write_text(json.dumps(payload))
    registry = JobRegistry(
        jobs_root=tmp_path / "jobs",
        staging_root=tmp_path / "staging",
    )
    record = registry.create(request, "cluster_slurm", status="submitted")
    record.staging_dir = str(staging)
    return record, staging


def sample_payload(calc_type: str) -> dict:
    """Minimal worker-style payloads for each Calculate-tab calc type."""
    base = {
        "energy_hartree": -1.12,
        "homo_lumo_gap_ev": 10.0,
        "converged": True,
        "n_iterations": 4,
        "method": "RHF",
        "basis": "STO-3G",
        "formula": "H2",
        "calc_type": calc_type,
    }
    if calc_type == "single_point":
        return base
    if calc_type == "geometry_opt":
        return {
            **base,
            "n_steps": 2,
            "trajectory_file": "trajectory.json",
        }
    if calc_type == "frequency":
        return {
            **base,
            "spectra": {
                "ir": {
                    "frequencies_cm1": [4400.0],
                    "ir_intensities": [1.0],
                    "raman_activities": [0.1],
                    "zpve_hartree": 0.01,
                    "displacements": [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
                },
                "molecule": {
                    "atoms": ["H", "H"],
                    "coords": [[0, 0, 0], [0, 0, 0.74]],
                    "charge": 0,
                    "multiplicity": 1,
                },
            },
        }
    if calc_type == "tddft":
        return {
            **base,
            "method": "B3LYP",
            "spectra": {
                "uv_vis": {
                    "excitation_energies_ev": [4.5],
                    "oscillator_strengths": [0.3],
                    "wavelengths_nm": [275.5],
                }
            },
        }
    if calc_type == "nmr":
        return {
            **base,
            "energy_hartree": float("nan"),
            "homo_lumo_gap_ev": None,
            "n_iterations": -1,
            "spectra": {
                "nmr": {
                    "atom_symbols": ["H", "H"],
                    "shielding_iso_ppm": [30.0, 30.0],
                    "chemical_shifts_ppm": {"0": 4.5, "1": 4.5},
                    "reference_compound": "TMS",
                    "reference_key": "tms",
                    "is_fallback_reference": False,
                }
            },
        }
    if calc_type == "pes_scan":
        return {
            **base,
            "homo_lumo_gap_ev": None,
            "n_iterations": -1,
            "trajectory_file": "trajectory.json",
            "spectra": {
                "pes_scan": {
                    "scan_type": "bond",
                    "atom_indices": [0, 1],
                    "scan_parameter_values": [0.7, 0.74, 0.8],
                    "energies_hartree": [-1.11, -1.12, -1.10],
                }
            },
        }
    if calc_type == "reorganization_energy":
        return {
            **base,
            "homo_lumo_gap_ev": None,
            "n_iterations": 6,
            "neutral_geometry": {
                "atoms": ["H", "H"],
                "coords": [[0, 0, 0], [0, 0, 0.74]],
                "charge": 0,
                "multiplicity": 1,
            },
            "channels": [
                {
                    "kind": "hole",
                    "ion_charge": 1,
                    "ion_multiplicity": 2,
                    "e_neutral_at_neutral": -1.12,
                    "e_ion_at_ion": -1.08,
                    "e_ion_at_neutral": -1.07,
                    "e_neutral_at_ion": -1.11,
                    "lambda1_hartree": 0.01,
                    "lambda2_hartree": 0.02,
                    "lambda_hartree": 0.03,
                    "converged": True,
                    "ion_geometry": {
                        "atoms": ["H", "H"],
                        "coords": [[0, 0, 0], [0, 0, 0.80]],
                        "charge": 1,
                        "multiplicity": 2,
                    },
                }
            ],
            "spectra": {
                "reorganization_energy": {
                    "channels": [{"kind": "hole", "lambda_hartree": 0.03}]
                }
            },
        }
    raise ValueError(f"unknown calc_type {calc_type!r}")
