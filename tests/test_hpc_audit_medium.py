"""
Regression tests for the Medium HPC findings of the 2026-11-10 audit
(HPC-2 … HPC-7): registry robustness, stale jobs, request-id reuse,
requeued attempts, SLURM ingest geometry and ``quantui-batch submit --from``.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui import app_slurm
from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry


def _request(rid: str = "hpc001", calc_type: str = "single_point"):
    return CalculationRequest(
        request_id=rid,
        calc_type=calc_type,
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.74]]},
    )


MOCK_SLURM_CLI = (
    Path(__file__).resolve().parents[1] / "testing" / "mock-slurm" / "mock_slurm_cli.py"
)


@pytest.fixture
def mock_slurm(tmp_path, monkeypatch):
    """Route squeue/sacct/scancel/sbatch to testing/mock-slurm."""
    state = tmp_path / "mock_state"
    state.mkdir()
    monkeypatch.setenv("MOCK_SLURM_STATE", str(state))
    real_run = subprocess.run

    def _run(args, **kwargs):
        if args and args[0] in ("squeue", "sacct", "scancel", "sbatch"):
            args = [sys.executable, str(MOCK_SLURM_CLI), *args]
        return real_run(args, **kwargs)

    monkeypatch.setattr("quantui.backends.slurm.subprocess.run", _run)
    return state


@pytest.fixture
def registry(tmp_path):
    return JobRegistry(jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging")


# ---------------------------------------------------------------------------
# HPC-3: monitor thread guard, atomic save, tolerant load, submit OSError
# ---------------------------------------------------------------------------


class TestRegistryRobustness:
    def test_save_is_atomic_when_the_write_fails(self, registry):
        record = registry.create(_request(), "cluster_slurm")
        before = registry._record_path(record.request_id).read_text()
        record.status = "running"
        with patch(
            "quantui.backends.registry.json.dump", side_effect=OSError("disk full")
        ):
            with pytest.raises(OSError):
                registry.save(record)
        # The old record is intact and no temp file is left behind.
        assert registry._record_path(record.request_id).read_text() == before
        assert sorted(p.name for p in registry.jobs_root.iterdir()) == ["hpc001.json"]

    @pytest.mark.parametrize("text", ["", '{"request_id": "hpc0', "[]"])
    def test_load_of_a_torn_record_returns_none(self, registry, text):
        registry.create(_request(), "cluster_slurm")
        registry._record_path("hpc001").write_text(text)
        assert registry.load("hpc001") is None
        assert registry.list_all() == []


def _monitor_app(registry):
    calls = []
    app = SimpleNamespace(
        _job_registry=registry,
        _slurm_monitor_stop=None,
        _calc_running=True,
        _queue_main_thread_callback=lambda fn, *a: calls.append(fn),
    )
    return app, calls


class TestMonitorGuard:
    def test_monitor_releases_ui_when_a_poll_keeps_failing(self, registry, monkeypatch):
        registry.create(_request(), "cluster_slurm", status="running")
        backend = SimpleNamespace(
            refresh_registry_statuses=lambda: (_ for _ in ()).throw(
                OSError("squeue exploded")
            )
        )
        monkeypatch.setattr(app_slurm, "slurm_backend_for_app", lambda _a: backend)
        monkeypatch.setattr(app_slurm, "_POLL_INTERVAL_S", 0)
        app, calls = _monitor_app(registry)
        app._slurm_monitor_stop = __import__("threading").Event()

        app_slurm._monitor_slurm_job(app, "hpc001")

        assert calls[-1] is app_slurm._finish_slurm_monitor

    def test_monitor_releases_ui_on_an_unexpected_error(self, registry, monkeypatch):
        registry.create(_request(), "cluster_slurm", status="running")
        monkeypatch.setattr(
            app_slurm,
            "slurm_backend_for_app",
            lambda _a: (_ for _ in ()).throw(ValueError("boom")),
        )
        app, calls = _monitor_app(registry)
        app_slurm._monitor_slurm_job(app, "hpc001")
        assert calls == [app_slurm._finish_slurm_monitor]

    def test_monitor_survives_a_truncated_record(self, registry, monkeypatch):
        registry.create(_request(), "cluster_slurm", status="running")
        registry._record_path("hpc001").write_text('{"request_id": ')
        backend = SimpleNamespace(refresh_registry_statuses=lambda: None)
        monkeypatch.setattr(app_slurm, "slurm_backend_for_app", lambda _a: backend)
        app, calls = _monitor_app(registry)
        app._slurm_monitor_stop = __import__("threading").Event()
        app_slurm._monitor_slurm_job(app, "hpc001")
        assert calls[-1] is app_slurm._finish_slurm_monitor


class TestSubmitReleasesUiOnOSError:
    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    @patch("quantui.app_slurm.slurm_submit_block_reason", return_value=None)
    @patch("quantui.app_slurm.build_calculation_request")
    @patch("quantui.app_slurm.slurm_backend_for_app")
    def test_oserror_resets_run_ui(
        self, mock_backend, mock_build, _block, _avail, registry
    ):
        mock_backend.return_value = SimpleNamespace(
            dispatch=lambda _r, **_k: (_ for _ in ()).throw(
                PermissionError("cannot write job folder")
            )
        )
        mock_build.return_value = _request()
        app = SimpleNamespace(
            _job_registry=registry,
            calc_type_dd=SimpleNamespace(value="Single Point"),
            run_status=SimpleNamespace(value=""),
            run_output=SimpleNamespace(
                append_stdout=lambda *_a, **_k: None,
                append_display_data=lambda *_a, **_k: None,
            ),
            _calc_running=False,
            run_btn=SimpleNamespace(disabled=False),
            cancel_btn=SimpleNamespace(disabled=True),
            log_clear_btn=SimpleNamespace(disabled=False),
        )
        with patch(
            "quantui.app_slurm.calc_type_key_from_app", return_value="single_point"
        ):
            app_slurm.submit_slurm_run(app)
        assert app._calc_running is False
        assert app.run_btn.disabled is False
        assert app.run_status.value == "SLURM submission failed."


# ---------------------------------------------------------------------------
# HPC-6: a requeued job (same SLURM id, new attempt dir) is tracked to the
# newest attempt
# ---------------------------------------------------------------------------


class TestRequeuedAttempt:
    def _requeued(self, registry):
        record = registry.create(_request(), "cluster_slurm", job_name="h2-sp")
        registry.start_attempt(record.request_id, "4242")
        job = record.job_path
        (job / "attempt-01_job4242").mkdir()  # died with the node
        (job / "attempt-02_job4242").mkdir()  # the requeued run
        (job / "attempt-02_job4242" / "result.json").write_text("{}")
        return registry.load(record.request_id)

    def test_attempt_dir_for_returns_the_newest_matching_attempt(self, registry):
        record = self._requeued(registry)
        assert record.attempt_dir_for("4242").name == "attempt-02_job4242"
        assert record.staging_path.name == "attempt-02_job4242"

    def test_completed_requeue_is_success_not_artifact_missing(self, registry):
        from quantui.backends.slurm import SlurmBackend

        record = self._requeued(registry)
        backend = SlurmBackend(registry=registry, use_apptainer=False)
        mapped, error = backend._terminal_registry_update(record, "COMPLETED")
        assert (mapped, error) == ("success", None)

    def test_batch_status_reads_the_newest_attempt(self, registry):
        from quantui.backends.batch_status import job_view

        record = self._requeued(registry)
        view = job_view(record, queue={"4242": "COMPLETED"})
        assert view.state == "done"


# ---------------------------------------------------------------------------
# HPC-2: a job Slurm no longer knows (UNKNOWN) eventually leaves the active set
# ---------------------------------------------------------------------------


class TestJobSlurmNoLongerKnows:
    @pytest.fixture
    def backend(self, registry, mock_slurm, monkeypatch):
        from quantui.backends.slurm import SlurmBackend

        monkeypatch.setenv("QUANTUI_SLURM_STALE_UNKNOWN_S", "60")
        return SlurmBackend(registry=registry, use_apptainer=False)

    def _tracked(self, registry, jid="777001"):
        record = registry.create(_request(), "cluster_slurm", job_name="h2-sp")
        registry.start_attempt(record.request_id, jid)
        registry.update_status(record.request_id, "running")
        return registry.load(record.request_id)

    def _poll(self, backend, times=1):
        for _ in range(times):
            backend._status_cache.clear()  # every poll really asks Slurm
            backend.refresh_registry_statuses()

    def _age_unknown(self, registry, rid="hpc001", seconds=7200):
        record = registry.load(rid)
        record.unknown_since = (
            (datetime.now(timezone.utc) - timedelta(seconds=seconds))
            .replace(microsecond=0)
            .isoformat()
        )
        registry.save(record)

    def test_unknown_job_goes_stale_and_can_be_removed(self, backend, registry):
        self._tracked(registry)
        self._poll(backend, 3)
        # Polls alone are not enough: the time margin has not passed.
        record = registry.load("hpc001")
        assert record.status == "running"
        assert record.unknown_polls == 3

        self._age_unknown(registry)
        self._poll(backend)
        record = registry.load("hpc001")
        assert record.status == "error"
        assert record.error["code"] == "STALE_RECORD"
        assert "777001" in record.error["user_message"]
        assert registry.list_active() == []

        app = SimpleNamespace(_job_registry=registry, _slurm_active_request_id=None)
        with patch("quantui.app_slurm.is_slurm_available", return_value=False):
            assert app_slurm.remove_slurm_job_record(app, "hpc001") is True
        assert registry.load("hpc001") is None

    def test_time_alone_is_not_enough(self, backend, registry):
        self._tracked(registry)
        self._poll(backend)
        self._age_unknown(registry)
        self._poll(backend)  # 2 polls < STALE_UNKNOWN_POLLS
        assert registry.load("hpc001").status == "running"

    def test_result_json_on_disk_means_success(self, backend, registry):
        record = self._tracked(registry)
        attempt = record.job_path / "attempt-01_job777001"
        attempt.mkdir()
        (attempt / "result.json").write_text("{}")
        self._poll(backend, 2)
        self._age_unknown(registry)
        self._poll(backend)
        record = registry.load("hpc001")
        assert (record.status, record.error) == ("success", None)

    def test_progress_error_on_disk_is_reported(self, backend, registry):
        record = self._tracked(registry)
        attempt = record.job_path / "attempt-01_job777001"
        attempt.mkdir()
        (attempt / "progress.json").write_text(
            json.dumps({"stage": "error", "message": "SCF did not converge"})
        )
        self._poll(backend, 2)
        self._age_unknown(registry)
        self._poll(backend)
        record = registry.load("hpc001")
        assert record.status == "error"
        assert record.error["user_message"] == "SCF did not converge"

    def test_squeue_failure_is_not_counted(self, backend, registry, monkeypatch):
        self._tracked(registry)

        def _no_squeue(args, **_kw):
            raise FileNotFoundError(args[0])

        monkeypatch.setattr("quantui.backends.slurm.subprocess.run", _no_squeue)
        self._poll(backend, 5)
        record = registry.load("hpc001")
        assert (record.status, record.unknown_polls) == ("running", 0)

    def test_streak_resets_when_slurm_sees_the_job_again(
        self, backend, registry, mock_slurm
    ):
        self._tracked(registry)
        self._poll(backend, 2)
        (mock_slurm / "777001.json").write_text(
            json.dumps({"job_id": "777001", "status": "RUNNING"})
        )
        self._poll(backend)
        record = registry.load("hpc001")
        assert (record.unknown_since, record.unknown_polls) == (None, 0)
        assert record.status == "running"


# ---------------------------------------------------------------------------
# HPC-4: reusing a request id must not overwrite a live job's record
# ---------------------------------------------------------------------------


class TestRequestIdReuse:
    def test_registry_create_refuses_a_taken_id(self, registry):
        registry.create(_request(), "cluster_slurm", status="running")
        with pytest.raises(FileExistsError):
            registry.create(_request(), "cluster_slurm")
        assert registry.load("hpc001").status == "running"

    def test_dispatching_the_same_request_twice_keeps_both_jobs(
        self, registry, mock_slurm, monkeypatch
    ):
        from quantui.backends.slurm import SlurmBackend

        monkeypatch.setenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", "0")
        backend = SlurmBackend(registry=registry, use_apptainer=False)
        with patch(
            "quantui.backends.slurm.check_submit_cooldown", lambda *_a, **_k: None
        ):
            first = backend.dispatch(_request())
            second = backend.dispatch(_request())

        assert first == "hpc001"
        assert second != first and second.startswith("hpc001-")
        a, b = registry.load(first), registry.load(second)
        assert a.slurm_job_id and b.slurm_job_id and a.slurm_job_id != b.slurm_job_id
        assert a.job_dir != b.job_dir
        assert b.request["run_context"]["source_request_id"] == "hpc001"
        assert (
            json.loads((Path(b.job_dir) / "request.json").read_text())["request_id"]
            == second
        )
