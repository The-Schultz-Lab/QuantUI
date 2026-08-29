"""
Tests for SlurmBackend submit/poll/cancel using mock SLURM scripts.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm import SlurmBackend
from quantui.security import SecurityError

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


def _age_record(registry: JobRegistry, request_id: str, *, seconds: float) -> None:
    record = registry.load(request_id)
    assert record is not None
    old = (
        (datetime.now(timezone.utc) - timedelta(seconds=seconds))
        .replace(microsecond=0)
        .isoformat()
    )
    record.created_at = old
    record.updated_at = old
    registry.save(record)


class TestSlurmBackendSubmit:
    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_writes_registry_and_scripts(
        self, mock_run, slurm_backend, tmp_path
    ):
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

    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_blocks_at_concurrent_limit(self, mock_run, slurm_backend):
        for idx in range(2):
            slurm_backend.registry.create(
                _request(f"active{idx}"),
                "cluster_slurm",
                status="running",
            )

        with pytest.raises(SecurityError, match="Concurrent job limit"):
            slurm_backend.dispatch(_request("blocked"))

        mock_run.assert_not_called()

    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_ignores_non_slurm_active_jobs(self, mock_run, slurm_backend):
        mock_run.return_value.stdout = "Submitted batch job 555666\n"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0

        slurm_backend.registry.create(_request("local-active"), "local_stub")
        slurm_backend.registry.update_status("local-active", "running")

        rid = slurm_backend.dispatch(_request("slurm-new"))
        assert slurm_backend.registry.load(rid) is not None

    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_blocks_during_cooldown(
        self, mock_run, slurm_backend, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", "30")
        slurm_backend.registry.record_slurm_submit()
        time.sleep(0.05)

        with pytest.raises(SecurityError, match="Please wait"):
            slurm_backend.dispatch(_request("cooldown-blocked"))

        mock_run.assert_not_called()

    @patch("quantui.backends.slurm.subprocess.run")
    def test_dispatch_records_submit_timestamp(self, mock_run, slurm_backend):
        mock_run.return_value.stdout = "Submitted batch job 555666\n"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0

        assert slurm_backend.registry.seconds_since_last_slurm_submit() is None
        slurm_backend.dispatch(_request("ts001"))
        since = slurm_backend.registry.seconds_since_last_slurm_submit()
        assert since is not None
        assert since < 5


class TestSlurmBackendReconcile:
    def test_reconcile_stale_record_without_slurm_id(self, slurm_backend, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_STALE_NO_ID_S", "60")
        slurm_backend.registry.create(_request("stale-no-id"), "cluster_slurm")
        _age_record(slurm_backend.registry, "stale-no-id", seconds=120)

        assert slurm_backend.reconcile_stale_records() == 1
        record = slurm_backend.registry.load("stale-no-id")
        assert record.status == "error"
        assert record.error["code"] == "STALE_RECORD"

    def test_reconcile_completed_missing_artifact(self, slurm_backend, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_STALE_NO_ID_S", "60")
        slurm_backend.registry.create(_request("missing-art"), "cluster_slurm")
        slurm_backend.registry.update_status(
            "missing-art", "running", slurm_job_id="888001"
        )
        _age_record(slurm_backend.registry, "missing-art", seconds=200)

        with patch.object(
            slurm_backend,
            "batch_poll_slurm_statuses",
            return_value={"888001": "COMPLETED"},
        ):
            slurm_backend.reconcile_stale_records()

        record = slurm_backend.registry.load("missing-art")
        assert record.status == "error"
        assert record.error["code"] == "ARTIFACT_MISSING"

    def test_reconcile_ignores_young_completed(self, slurm_backend, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_STALE_NO_ID_S", "60")
        slurm_backend.registry.create(_request("young"), "cluster_slurm")
        slurm_backend.registry.update_status("young", "running", slurm_job_id="888002")

        with (
            patch.object(
                slurm_backend,
                "batch_poll_slurm_statuses",
                return_value={"888002": "COMPLETED"},
            ),
            patch.object(
                slurm_backend,
                "poll_slurm_status",
                return_value="COMPLETED",
            ),
        ):
            assert slurm_backend.reconcile_stale_records() == 0

        record = slurm_backend.registry.load("young")
        assert record.status == "running"


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

    def test_poll_uses_sacct_when_not_in_squeue(self, slurm_backend, mock_slurm_env):
        (mock_slurm_env / "888.json").write_text(
            json.dumps({"job_id": "888", "status": "FAILED"})
        )
        assert slurm_backend.poll_slurm_status("888") == "FAILED"

    def test_batch_poll_falls_back_to_sacct(self, slurm_backend, mock_slurm_env):
        (mock_slurm_env / "10001.json").write_text(
            json.dumps({"job_id": "10001", "status": "RUNNING"})
        )
        (mock_slurm_env / "10002.json").write_text(
            json.dumps(
                {
                    "job_id": "10002",
                    "status": "COMPLETED",
                    "exit_code": "0:0",
                    "elapsed": "00:02:00",
                }
            )
        )
        statuses = slurm_backend.batch_poll_slurm_statuses(["10001", "10002"])
        assert statuses["10001"] == "RUNNING"
        assert statuses["10002"] == "COMPLETED"

    def test_batch_job_accounting_includes_sacct_fields(
        self, slurm_backend, mock_slurm_env
    ):
        (mock_slurm_env / "10003.json").write_text(
            json.dumps(
                {
                    "job_id": "10003",
                    "status": "FAILED",
                    "exit_code": "1:0",
                    "elapsed": "00:00:15",
                }
            )
        )
        acct = slurm_backend.batch_job_accounting(["10003"])
        assert acct["10003"].state == "FAILED"
        assert acct["10003"].exit_code == "1:0"
        assert acct["10003"].elapsed == "00:00:15"

    def test_cache_avoids_repeat_squeue(self, slurm_backend):
        with patch("quantui.backends.slurm.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "RUNNING\n"
            mock_run.return_value.returncode = 0
            slurm_backend.poll_slurm_status("900001")
            slurm_backend.poll_slurm_status("900001")
        assert mock_run.call_count == 1


class TestSlurmBackendCancel:
    def test_cancel_without_slurm_id_sets_error(self, slurm_backend):
        slurm_backend.registry.create(_request("no-id"), "cluster_slurm")
        assert slurm_backend.cancel("no-id") is False
        record = slurm_backend.registry.load("no-id")
        assert record.error["code"] == "CANCEL_NO_SLURM_ID"

    def test_cancel_confirms_via_sacct(
        self, slurm_backend, mock_slurm_env, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_SLURM_CANCEL_CONFIRM_S", "5")
        slurm_backend.registry.create(_request("cancelme"), "cluster_slurm")
        slurm_backend.registry.update_status("cancelme", "running", slurm_job_id="777")
        (mock_slurm_env / "777.json").write_text(
            json.dumps({"job_id": "777", "status": "RUNNING"})
        )

        assert slurm_backend.cancel("cancelme") is True
        record = slurm_backend.registry.load("cancelme")
        assert record.status == "cancelled"
        assert record.error is None
        data = json.loads((mock_slurm_env / "777.json").read_text())
        assert data["status"] == "CANCELLED"

    @patch("quantui.backends.slurm.subprocess.run")
    def test_cancel_already_cancelled_is_idempotent(self, mock_run, slurm_backend):
        slurm_backend.registry.create(_request("done"), "cluster_slurm")
        slurm_backend.registry.update_status("done", "running", slurm_job_id="999")

        def _fake_run(args, **_kwargs):
            cmd = args[0] if args else ""
            result = mock_run.return_value
            if cmd == "squeue":
                result.stdout = ""
            elif cmd == "sacct":
                result.stdout = "999|CANCELLED|0:15|00:00:04\n"
            return result

        mock_run.return_value.returncode = 0
        mock_run.side_effect = _fake_run

        assert slurm_backend.cancel("done") is True
        record = slurm_backend.registry.load("done")
        assert record.status == "cancelled"
