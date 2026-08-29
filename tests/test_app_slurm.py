"""
Tests for SLURM dispatch helpers and app integration utilities.
"""

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from quantui.app_slurm import (
    active_slurm_job_count,
    max_concurrent_slurm_jobs,
    slurm_submit_block_reason,
    submit_slurm_run,
    use_slurm_execution,
)
from quantui.backends.dispatch import (
    build_calculation_request,
    calc_type_key_from_app,
    is_slurm_available,
)
from quantui.backends.registry import JobRegistry
from quantui.security import SecurityError


def _fake_app(**overrides):
    mol = SimpleNamespace(
        atoms=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74]],
        charge=0,
        get_formula=lambda: "H2",
    )
    defaults = dict(
        _molecule=mol,
        method_dd=SimpleNamespace(value="RHF"),
        basis_dd=SimpleNamespace(value="STO-3G"),
        mult_si=SimpleNamespace(value=1),
        calc_type_dd=SimpleNamespace(value="Single Point"),
        solvent_dd=SimpleNamespace(value=""),
        solvent_cb=SimpleNamespace(value=False),
        _user_settings=SimpleNamespace(
            compute=SimpleNamespace(execution_backend="local")
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDispatch:
    def test_build_calculation_request(self):
        req = build_calculation_request(_fake_app(), request_id="abc")
        assert req.request_id == "abc"
        assert req.calc_type == "single_point"
        assert req.molecule["atoms"] == ["H", "H"]

    def test_calc_type_key_from_app(self):
        app = _fake_app(calc_type_dd=SimpleNamespace(value="Frequency"))
        assert calc_type_key_from_app(app) == "frequency"

    @patch("quantui.backends.dispatch.shutil.which")
    def test_is_slurm_available(self, mock_which):
        mock_which.return_value = "/usr/bin/sbatch"
        assert is_slurm_available() is True
        mock_which.return_value = None
        assert is_slurm_available() is False


class TestUseSlurmExecution:
    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    def test_true_when_pref_slurm(self, _mock):
        app = _fake_app(
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            )
        )
        assert use_slurm_execution(app) is True

    @patch("quantui.app_slurm.is_slurm_available", return_value=False)
    def test_false_when_no_sbatch(self, _mock):
        app = _fake_app(
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            )
        )
        assert use_slurm_execution(app) is False

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    def test_false_when_pref_local(self, _mock):
        assert use_slurm_execution(_fake_app()) is False


class TestSlurmSubmitGuards:
    def test_block_reason_when_at_limit(self, tmp_path):
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        for idx in range(max_concurrent_slurm_jobs()):
            registry.create(
                build_calculation_request(_fake_app(), request_id=f"job{idx}"),
                "cluster_slurm",
                status="running",
            )
        app = _fake_app(_job_registry=registry)
        reason = slurm_submit_block_reason(app)
        assert reason is not None
        assert "Concurrent job limit" in reason

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    def test_block_reason_during_cooldown(self, _mock_slurm, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_SUBMIT_COOLDOWN_S", "30")
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        registry.record_slurm_submit()
        time.sleep(0.05)
        app = _fake_app(_job_registry=registry)
        reason = slurm_submit_block_reason(app)
        assert reason is not None
        assert "Please wait" in reason

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    @patch("quantui.app_slurm.slurm_backend_for_app")
    def test_block_reason_reconciles_stale_jobs(
        self, mock_backend_for_app, _mock_slurm, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_SLURM_STALE_NO_ID_S", "60")
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        registry.create(
            build_calculation_request(_fake_app(), request_id="stale"),
            "cluster_slurm",
        )
        record = registry.load("stale")
        old = (
            (datetime.now(timezone.utc) - timedelta(seconds=120))
            .replace(microsecond=0)
            .isoformat()
        )
        record.created_at = old
        record.updated_at = old
        registry.save(record)

        backend = SimpleNamespace(reconcile_stale_records=MagicMock(return_value=1))
        mock_backend_for_app.return_value = backend
        app = _fake_app(_job_registry=registry)

        slurm_submit_block_reason(app)

        mock_backend_for_app.assert_called_once_with(app)
        backend.reconcile_stale_records.assert_called_once()

    def test_active_count_ignores_other_backends(self, tmp_path):
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        registry.create(
            build_calculation_request(_fake_app(), request_id="local1"),
            "local_stub",
            status="running",
        )
        registry.create(
            build_calculation_request(_fake_app(), request_id="slurm1"),
            "cluster_slurm",
            status="running",
        )
        app = _fake_app(_job_registry=registry)
        assert active_slurm_job_count(app) == 1

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    @patch("quantui.app_slurm.build_calculation_request")
    @patch("quantui.app_slurm.slurm_backend_for_app")
    def test_submit_blocked_without_locking_ui(
        self, mock_backend_for_app, mock_build_request, _mock_slurm, tmp_path
    ):
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs",
            staging_root=tmp_path / "staging",
        )
        for idx in range(max_concurrent_slurm_jobs()):
            registry.create(
                build_calculation_request(_fake_app(), request_id=f"active{idx}"),
                "cluster_slurm",
                status="running",
            )

        backend = MagicMock()
        backend.reconcile_stale_records.return_value = 0
        mock_backend_for_app.return_value = backend

        app = _fake_app(
            _job_registry=registry,
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            ),
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

        submit_slurm_run(app)

        backend.dispatch.assert_not_called()
        mock_build_request.assert_not_called()
        assert app._calc_running is False
        assert app.run_btn.disabled is False

    @patch("quantui.app_slurm.is_slurm_available", return_value=True)
    @patch("quantui.app_slurm.build_calculation_request")
    @patch("quantui.app_slurm.slurm_backend_for_app")
    def test_security_error_resets_run_ui(
        self, mock_backend_for_app, mock_build_request, _mock_slurm, tmp_path
    ):
        backend = SimpleNamespace(
            dispatch=lambda _req: (_ for _ in ()).throw(
                SecurityError("Concurrent job limit reached (2/2).")
            )
        )
        mock_backend_for_app.return_value = backend
        mock_build_request.return_value = build_calculation_request(
            _fake_app(), request_id="newjob"
        )

        app = _fake_app(
            _job_registry=JobRegistry(
                jobs_root=tmp_path / "jobs",
                staging_root=tmp_path / "staging",
            ),
            _user_settings=SimpleNamespace(
                compute=SimpleNamespace(execution_backend="slurm")
            ),
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

        submit_slurm_run(app)

        assert app._calc_running is False
        assert app.run_btn.disabled is False
        assert "Concurrent job limit" in app.run_status.value
