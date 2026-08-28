"""
Tests for the headless batch worker (mocked — no PySCF required in CI).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.worker import run_worker_request


@pytest.fixture
def staging(tmp_path):
    staging_dir = tmp_path / "staging" / "job1"
    staging_dir.mkdir(parents=True)
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
    registry.update_status("job1", "submitted", slurm_job_id="1")
    return staging_dir


class TestWorker:
    def test_unsupported_calc_type_returns_error(self, staging):
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"
        assert outcome.error["code"] == "UNSUPPORTED_CAPABILITY"

    @patch("quantui.session_calc.run_in_session")
    def test_single_point_success(self, mock_run, staging):
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        assert outcome.result_payload["energy_hartree"] == -1.12
        assert (staging / "result.json").exists()
        assert (staging / "progress.json").exists()
