"""
Tests for SLURM staging → History ingest helpers.
"""

import json
from unittest.mock import patch

import numpy as np

from quantui.backends.slurm_ingest import ingest_staging_success
from tests.slurm_ingest_helpers import make_staging_record, patch_results_root


class TestSlurmIngest:
    def test_ingest_single_point(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        payload = {
            "calc_type": "single_point",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
        }
        record, _staging = make_staging_record(tmp_path, payload)
        saved = ingest_staging_success(record, "log line\n")
        assert saved.exists()
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "single_point"
        assert (saved / "pyscf.log").read_text() == "log line\n"

    def test_ingest_geometry_opt_copies_trajectory(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        traj = {
            "atoms": ["H", "H"],
            "charge": 0,
            "multiplicity": 1,
            "steps": [{"coords": [[0, 0, 0], [0, 0, 0.8]], "energy": -1.1}],
        }
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
        record, staging = make_staging_record(
            tmp_path, payload, calc_type="geometry_opt"
        )
        (staging / "trajectory.json").write_text(json.dumps(traj))
        saved = ingest_staging_success(record)
        assert (saved / "trajectory.json").exists()

    def test_ingest_frequency_with_spectra(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
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
        record, _staging = make_staging_record(tmp_path, payload, calc_type="frequency")
        saved = ingest_staging_success(record)
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "frequency"
        assert "ir" in data["spectra"]

    def test_ingest_tddft_with_spectra(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
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
        record, _staging = make_staging_record(tmp_path, payload, calc_type="tddft")
        saved = ingest_staging_success(record)
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "tddft"
        assert data["spectra"]["uv_vis"]["excitation_energies_ev"] == [4.5]

    def test_ingest_copies_orbitals_and_thumbnail(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        payload = {
            "calc_type": "single_point",
            "energy_hartree": -1.12,
            "converged": True,
            "n_iterations": 4,
            "method": "RHF",
            "basis": "STO-3G",
            "formula": "H2",
        }
        record, staging = make_staging_record(tmp_path, payload)
        np.savez_compressed(
            staging / "orbitals.npz",
            mo_energy_hartree=np.array([-1.0, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
        )
        (staging / "orbitals_meta.json").write_text(
            json.dumps({"mol_atom": [["H", [0, 0, 0]]], "mol_basis": "STO-3G"})
        )
        with patch("quantui.results_storage.save_thumbnail") as mock_thumb:
            saved = ingest_staging_success(record)
            mock_thumb.assert_called_once()
        assert (saved / "orbitals.npz").exists()
        assert (saved / "orbitals_meta.json").exists()
