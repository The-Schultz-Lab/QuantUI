"""
Tests for SLURM staging → History ingest helpers.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm_ingest import ingest_staging_success


def _record(tmp_path: Path, payload: dict, *, calc_type: str = "single_point"):
    staging = tmp_path / "staging" / "job1"
    staging.mkdir(parents=True, exist_ok=True)
    request = CalculationRequest(
        request_id="job1",
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
    return record


class TestSlurmIngest:
    def test_ingest_single_point(self, tmp_path, monkeypatch):
        results_root = tmp_path / "results"
        monkeypatch.setattr(
            "quantui.results_storage._default_results_dir",
            lambda: results_root,
        )
        payload = {
            "calc_type": "single_point",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
        }
        record = _record(tmp_path, payload)
        saved = ingest_staging_success(record, "log line\n")
        assert saved.exists()
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "single_point"
        assert (saved / "pyscf.log").read_text() == "log line\n"

    def test_ingest_geometry_opt_copies_trajectory(self, tmp_path, monkeypatch):
        results_root = tmp_path / "results"
        monkeypatch.setattr(
            "quantui.results_storage._default_results_dir",
            lambda: results_root,
        )
        staging = tmp_path / "staging" / "job1"
        staging.mkdir(parents=True, exist_ok=True)
        traj = {
            "atoms": ["H", "H"],
            "charge": 0,
            "multiplicity": 1,
            "steps": [{"coords": [[0, 0, 0], [0, 0, 0.8]], "energy": -1.1}],
        }
        (staging / "trajectory.json").write_text(json.dumps(traj))
        payload = {
            "calc_type": "geometry_opt",
            "energy_hartree": -1.1,
            "converged": True,
            "n_steps": 1,
            "n_iterations": 1,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
            "trajectory_file": "trajectory.json",
        }
        record = _record(tmp_path, payload, calc_type="geometry_opt")
        saved = ingest_staging_success(record)
        assert (saved / "trajectory.json").exists()

    def test_ingest_frequency_with_spectra(self, tmp_path, monkeypatch):
        results_root = tmp_path / "results"
        monkeypatch.setattr(
            "quantui.results_storage._default_results_dir",
            lambda: results_root,
        )
        payload = {
            "calc_type": "frequency",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
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
        record = _record(tmp_path, payload, calc_type="frequency")
        saved = ingest_staging_success(record)
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "frequency"
        assert "ir" in data["spectra"]

    def test_ingest_tddft_with_spectra(self, tmp_path, monkeypatch):
        results_root = tmp_path / "results"
        monkeypatch.setattr(
            "quantui.results_storage._default_results_dir",
            lambda: results_root,
        )
        payload = {
            "calc_type": "tddft",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "B3LYP",
            "basis": "STO-3G",
            "formula": "H2",
            "spectra": {
                "uv_vis": {
                    "excitation_energies_ev": [4.5],
                    "oscillator_strengths": [0.3],
                    "wavelengths_nm": [275.5],
                }
            },
        }
        record = _record(tmp_path, payload, calc_type="tddft")
        saved = ingest_staging_success(record)
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "tddft"
        assert data["spectra"]["uv_vis"]["excitation_energies_ev"] == [4.5]

    def test_ingest_copies_orbitals_and_thumbnail(self, tmp_path, monkeypatch):
        results_root = tmp_path / "results"
        monkeypatch.setattr(
            "quantui.results_storage._default_results_dir",
            lambda: results_root,
        )
        staging = tmp_path / "staging" / "job1"
        staging.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            staging / "orbitals.npz",
            mo_energy_hartree=np.array([-1.0, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
        )
        (staging / "orbitals_meta.json").write_text(
            json.dumps({"mol_atom": [["H", [0, 0, 0]]], "mol_basis": "STO-3G"})
        )
        payload = {
            "calc_type": "single_point",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
        }
        record = _record(tmp_path, payload)
        with patch("quantui.results_storage.save_thumbnail") as mock_thumb:
            saved = ingest_staging_success(record)
            mock_thumb.assert_called_once()
        assert (saved / "orbitals.npz").exists()
        assert (saved / "orbitals_meta.json").exists()
