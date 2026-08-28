"""
Tests for SlurmBackend submit/poll/cancel using mock SLURM scripts.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm import SlurmBackend

MOCK_SLURM_DIR = Path(__file__).resolve().parents[1] / "testing" / "mock-slurm"


@pytest.fixture
def mock_slurm_env(tmp_path, monkeypatch):
    state = tmp_path / "mock_state"
    state.mkdir()
    monkeypatch.setenv("MOCK_SLURM_STATE", str(state))
    monkeypatch.setenv("MOCK_SLURM_AUTO_COMPLETE", "1")
    path_entries = [str(MOCK_SLURM_DIR), *os.environ.get("PATH", "").split(":")]
    monkeypatch.setenv("PATH", ":".join(path_entries))
    return state


@pytest.fixture
def slurm_backend(tmp_path, mock_slurm_env):
    registry = JobRegistry(
        jobs_root=tmp_path / "jobs",
        staging_root=tmp_path / "staging",
    )
    return SlurmBackend(
        registry=registry,
        partition="test",
        use_apptainer=False,
    )


def _request(rid: str = "slurm001") -> CalculationRequest:
    return CalculationRequest(
        request_id=rid,
        calc_type="single_point",
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={
            "atoms": ["H", "H"],
            "coords": [[0, 0, 0], [0, 0, 0.74]],
        },
    )


class TestSlurmBackendSubmit:
    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_writes_registry_and_scripts(self, mock_run, slurm_backend, tmp_path):
        mock_run.return_value.stdout = "Submitted batch job 555666\n"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0

        rid = slurm_backend.dispatch(_request())
        record = slurm_backend.registry.load(rid)
        assert record is not None
        assert record.slurm_job_id == "555666"
        assert record.status == "submitted"
        assert (record.staging_path / "request.json").exists()
        assert (record.staging_path / "submit.slurm").exists()
        slurm_text = (record.staging_path / "submit.slurm").read_text()
        assert "#SBATCH" in slurm_text
        assert "quantui.backends.worker" in slurm_text

    @patch("quantui.backends.slurm.subprocess.run")
    def test_submit_failure_marks_registry_error(self, mock_run, slurm_backend):
        mock_run.side_effect = __import__("subprocess").CalledProcessError(
            1, cmd=["sbatch"], stderr="QOSMaxSubmitJobPerUserLimit"
        )
        with pytest.raises(RuntimeError):
            slurm_backend.dispatch(_request("fail001"))
        record = slurm_backend.registry.load("fail001")
        assert record.status == "error"
        assert record.error["code"] == "BACKEND_UNAVAILABLE"


class TestSlurmBackendPolling:
    def test_batch_poll_uses_single_squeue(self, slurm_backend, mock_slurm_env):
        # Seed mock state with two pending jobs
        for jid in ("10001", "10002"):
            (mock_slurm_env / f"{jid}.json").write_text(
                json.dumps({"job_id": jid, "status": "RUNNING"})
            )

        with patch("quantui.backends.slurm.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "10001 RUNNING\n10002 PENDING\n"
            mock_run.return_value.returncode = 0
            statuses = slurm_backend.batch_poll_slurm_statuses(["10001", "10002"])

        assert statuses["10001"] == "RUNNING"
        assert statuses["10002"] == "PENDING"
        assert mock_run.call_count == 1

    def test_cache_avoids_repeat_squeue(self, slurm_backend):
        with patch("quantui.backends.slurm.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "RUNNING\n"
            mock_run.return_value.returncode = 0
            slurm_backend.poll_slurm_status("900001")
            slurm_backend.poll_slurm_status("900001")
        assert mock_run.call_count == 1


class TestSlurmBackendCancel:
    @patch("quantui.backends.slurm.subprocess.run")
    def test_cancel_updates_registry(self, mock_run, slurm_backend):
        mock_run.return_value.returncode = 0
        slurm_backend.registry.create(_request("cancelme"), "cluster_slurm")
        slurm_backend.registry.update_status("cancelme", "running", slurm_job_id="777")
        assert slurm_backend.cancel("cancelme") is True
        record = slurm_backend.registry.load("cancelme")
        assert record.status == "cancelled"
