"""
Regression tests for the Medium HPC findings of the 2026-11-10 audit
(HPC-2 … HPC-7): registry robustness, stale jobs, request-id reuse,
requeued attempts, SLURM ingest geometry and ``quantui-batch submit --from``.
"""

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
