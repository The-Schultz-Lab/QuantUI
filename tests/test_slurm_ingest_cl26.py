"""CL2.6 — SLURM staging ingest matrix (mock cluster, local History parity).

Maps to the NCShare runbook manual validation matrix (steps 4–7) and
M-CLUSTER2 roadmap CL2.6. No real SLURM cluster required: exercises
worker staging → ingest → History artifacts the same way local ``_do_run``
persists results.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from quantui.backends.slurm_ingest import (
    completion_summary_html,
    ingest_staging_success,
)
from quantui.backends.worker import run_worker_request
from quantui.molecule import Molecule
from quantui.results_storage import load_orbitals, load_result, load_trajectory
from tests.slurm_ingest_helpers import (
    make_staging_record,
    patch_results_root,
    sample_payload,
)

ALL_CALC_TYPES = (
    "single_point",
    "geometry_opt",
    "frequency",
    "tddft",
    "nmr",
    "pes_scan",
    "reorganization_energy",
)


class TestCL26IngestMatrix:
    """Runbook step 4/7: every calc type lands in History with expected schema."""

    @pytest.mark.parametrize("calc_type", ALL_CALC_TYPES)
    def test_ingest_creates_result_json(self, tmp_path, monkeypatch, calc_type):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload(calc_type)
        record, _staging = make_staging_record(tmp_path, payload, calc_type=calc_type)
        saved = ingest_staging_success(record, log_text="batch log\n")
        assert saved.is_dir()
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == calc_type
        assert (saved / "pyscf.log").read_text() == "batch log\n"

    @pytest.mark.parametrize(
        "calc_type,spectra_key",
        [
            ("frequency", "ir"),
            ("tddft", "uv_vis"),
            ("nmr", "nmr"),
            ("pes_scan", "pes_scan"),
            ("reorganization_energy", "reorganization_energy"),
        ],
    )
    def test_spectra_calc_types_persist_spectra_block(
        self, tmp_path, monkeypatch, calc_type, spectra_key
    ):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload(calc_type)
        record, _staging = make_staging_record(tmp_path, payload, calc_type=calc_type)
        saved = ingest_staging_success(record)
        data = load_result(saved)
        assert spectra_key in data.get(
            "spectra", {}
        ), f"{calc_type} ingest must preserve spectra.{spectra_key} for Analysis"

    def test_reorg_ingest_persists_reorg_channels(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("reorganization_energy")
        record, _staging = make_staging_record(
            tmp_path, payload, calc_type="reorganization_energy"
        )
        saved = ingest_staging_success(record)
        data = load_result(saved)
        assert data.get("reorg_channels"), "reorg_channels required for History replay"
        assert data["reorg_channels"][0]["lambda_hartree"] == pytest.approx(0.03)


class TestCL26AnalysisReplayParity:
    """Runbook step 6 + Analysis parity: sidecars survive ingest."""

    def test_geometry_opt_copies_trajectory_and_export_sidecars(
        self, tmp_path, monkeypatch
    ):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("geometry_opt")
        record, staging = make_staging_record(
            tmp_path, payload, calc_type="geometry_opt"
        )
        traj = {
            "atoms": ["H", "H"],
            "charge": 0,
            "multiplicity": 1,
            "steps": [
                {"coords": [[0, 0, 0], [0, 0, 0.8]], "energy": -1.1},
                {"coords": [[0, 0, 0], [0, 0, 0.74]], "energy": -1.12},
            ],
        }
        (staging / "trajectory.json").write_text(json.dumps(traj))
        (staging / "trajectory.xyz").write_text("sidecar xyz\n")
        (staging / "trajectory.traj").write_bytes(b"sidecar traj")

        saved = ingest_staging_success(record)
        assert (saved / "trajectory.json").exists()
        assert (saved / "trajectory.xyz").read_text() == "sidecar xyz\n"
        assert (saved / "trajectory.traj").read_bytes() == b"sidecar traj"
        trajectory, energies = load_trajectory(saved)
        assert len(trajectory) == 2
        assert len(energies) == 2

    def test_orbitals_sidecar_loads_after_ingest(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("single_point")
        record, staging = make_staging_record(tmp_path, payload)
        np.savez_compressed(
            staging / "orbitals.npz",
            mo_energy_hartree=np.array([-1.0, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
            mo_coeff=np.eye(2),
        )
        (staging / "orbitals_meta.json").write_text(
            json.dumps(
                {
                    "mol_atom": [["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
                    "mol_basis": "STO-3G",
                }
            )
        )

        saved = ingest_staging_success(record)
        orbitals = load_orbitals(saved)
        np.testing.assert_array_almost_equal(
            orbitals.mo_energy_hartree, np.array([-1.0, 0.5])
        )
        assert orbitals.pyscf_mol_basis == "STO-3G"

    def test_frequency_ingest_copies_preopt_trajectory(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("frequency")
        record, staging = make_staging_record(tmp_path, payload, calc_type="frequency")
        preopt = {
            "atoms": ["H", "H"],
            "charge": 0,
            "multiplicity": 1,
            "steps": [{"coords": [[0, 0, 0], [0, 0, 0.75]], "energy": -1.11}],
        }
        (staging / "preopt_trajectory.json").write_text(json.dumps(preopt))

        saved = ingest_staging_success(record)
        assert (saved / "preopt_trajectory.json").exists()
        data = json.loads((saved / "preopt_trajectory.json").read_text())
        assert len(data["steps"]) == 1

    def test_frequency_prefers_staging_molden_over_fallback(
        self, tmp_path, monkeypatch
    ):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("frequency")
        record, staging = make_staging_record(tmp_path, payload, calc_type="frequency")
        (staging / "result.molden").write_text("[Atoms] worker-written\n")

        saved = ingest_staging_success(record)
        assert (saved / "result.molden").read_text() == "[Atoms] worker-written\n"

    def test_frequency_fallback_writes_molden_when_staging_absent(
        self, tmp_path, monkeypatch
    ):
        patch_results_root(tmp_path, monkeypatch)
        payload = sample_payload("frequency")
        record, _staging = make_staging_record(tmp_path, payload, calc_type="frequency")

        with patch("quantui.results_storage.save_molden") as mock_molden:
            mock_molden.return_value = None
            ingest_staging_success(record)
            mock_molden.assert_called_once()

    def test_thumbnail_generated_for_every_calc_type(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        for calc_type in ALL_CALC_TYPES:
            payload = sample_payload(calc_type)
            record, _staging = make_staging_record(
                tmp_path, payload, calc_type=calc_type, request_id=f"job-{calc_type}"
            )
            with patch("quantui.results_storage.save_thumbnail") as mock_thumb:
                ingest_staging_success(record)
                mock_thumb.assert_called_once()


class TestCL26WorkerIngestRoundtrip:
    """Worker staging output → ingest without a real cluster."""

    @pytest.fixture
    def worker_staging(self, tmp_path):
        staging_dir = tmp_path / "staging" / "job1"
        staging_dir.mkdir(parents=True)
        from quantui.backends.base import CalculationRequest
        from quantui.backends.registry import JobRegistry

        request = CalculationRequest(
            request_id="job1",
            calc_type="single_point",
            method="RHF",
            basis="STO-3G",
            charge=0,
            multiplicity=1,
            molecule={
                "atoms": ["H", "H"],
                "coordinates": [[0, 0, 0], [0, 0, 0.74]],
            },
        )
        (staging_dir / "request.json").write_text(json.dumps(request.to_dict()))
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        registry.create(request, "cluster_slurm", status="submitted")
        return staging_dir, tmp_path

    @patch("quantui.session_calc.run_in_session")
    def test_worker_then_ingest_single_point(
        self, mock_run, worker_staging, monkeypatch
    ):
        staging_dir, tmp_path = worker_staging
        patch_results_root(tmp_path, monkeypatch)
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            mo_energy_hartree=np.array([-1.0, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
            mo_coeff=np.eye(2),
            pyscf_mol_atom=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.74])],
            pyscf_mol_basis="STO-3G",
        )

        outcome = run_worker_request(staging_dir / "request.json")
        assert outcome.status == "success"
        assert (staging_dir / "orbitals.npz").exists()

        record, _ = make_staging_record(
            tmp_path,
            json.loads((staging_dir / "result.json").read_text()),
            calc_type="single_point",
        )
        record.staging_dir = str(staging_dir)
        saved = ingest_staging_success(record, "worker log\n")
        data = load_result(saved)
        assert data["calc_type"] == "single_point"
        assert (saved / "orbitals.npz").exists()
        load_orbitals(saved)

    @patch("quantui.optimizer.optimize_geometry")
    def test_worker_then_ingest_geometry_opt(
        self, mock_opt, worker_staging, monkeypatch
    ):
        staging_dir, tmp_path = worker_staging
        patch_results_root(tmp_path, monkeypatch)
        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol, mol],
            energies_hartree=[-1.11, -1.12],
            converged=True,
            n_steps=1,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging_dir / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        (staging_dir / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging_dir / "request.json")
        assert outcome.status == "success"
        assert (staging_dir / "trajectory.json").exists()

        record, _ = make_staging_record(
            tmp_path,
            json.loads((staging_dir / "result.json").read_text()),
            calc_type="geometry_opt",
        )
        record.staging_dir = str(staging_dir)
        saved = ingest_staging_success(record)
        assert (saved / "trajectory.json").exists()
        trajectory, _energies = load_trajectory(saved)
        assert len(trajectory) >= 1


class TestCL26CompletionSummary:
    @pytest.mark.parametrize("calc_type", ALL_CALC_TYPES)
    def test_summary_html_includes_calc_type(self, tmp_path, calc_type):
        payload = sample_payload(calc_type)
        html = completion_summary_html(tmp_path / "saved", payload)
        assert "SLURM calculation complete" in html
        assert calc_type.replace("_", " ") in html
        assert str(tmp_path / "saved") in html

    def test_nmr_summary_omits_energy_line(self):
        payload = sample_payload("nmr")
        html = completion_summary_html(Path("/tmp/saved"), payload)
        assert "Energy:" not in html
        assert "Shift entries:" in html


class TestCL26IngestErrors:
    def test_missing_result_json_raises(self, tmp_path, monkeypatch):
        patch_results_root(tmp_path, monkeypatch)
        record, staging = make_staging_record(tmp_path, sample_payload("single_point"))
        (staging / "result.json").unlink()
        with pytest.raises(FileNotFoundError, match="Missing staging result"):
            ingest_staging_success(record)
