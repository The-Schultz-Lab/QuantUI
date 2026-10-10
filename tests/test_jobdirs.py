"""
M-JOBDIRS — SLURM job dirs with per-attempt subdirs.

Covers the job dir layout (naming, collisions, ``--job-name``), the attempt
setup block inside ``submit.slurm`` (run through a real ``bash``), Resubmit,
the worker's ``--attempt-dir`` / tracked-record guard, and legacy records.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from quantui.backends import cluster_config as cfg
from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry, parse_attempt_dir_name
from quantui.backends.slurm import SlurmBackend
from quantui.backends.slurm_utils import default_job_name, sanitize_job_name

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="submit.slurm is a bash script",
)


def _request(rid: str = "jd001", **overrides) -> CalculationRequest:
    fields = dict(
        request_id=rid,
        calc_type="geometry_opt",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        molecule={
            "atoms": ["H", "H"],
            "coords": [[0, 0, 0], [0, 0, 0.74]],
            "label": "H2",
        },
    )
    fields.update(overrides)
    return CalculationRequest(**fields)


@pytest.fixture
def registry(tmp_path):
    return JobRegistry(jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging")


@pytest.fixture
def backend(registry):
    return SlurmBackend(registry=registry, partition="test", use_apptainer=False)


def _sbatch_returning(*job_ids):
    """Patch subprocess.run so successive sbatch calls return *job_ids*."""
    outputs = iter(job_ids)

    def _run(args, **kwargs):
        result = subprocess.CompletedProcess(args, 0)
        result.stdout = f"Submitted batch job {next(outputs)}\n"
        result.stderr = ""
        return result

    return patch("quantui.backends.slurm.subprocess.run", side_effect=_run)


def _run_attempt_setup(job_dir: Path, slurm_job_id: str) -> subprocess.CompletedProcess:
    script = "set -euo pipefail\n" + cfg.build_attempt_setup(str(job_dir))
    env = {**os.environ, "SLURM_JOB_ID": slurm_job_id}
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# Naming (JD.2)
# ---------------------------------------------------------------------------


class TestJobNames:
    def test_default_name_is_label_calc_method_basis(self):
        assert default_job_name(_request()) == "H2_opt_B3LYP_def2-SVP"

    def test_default_name_keeps_star_basis_distinct(self):
        name = default_job_name(_request(calc_type="single_point", basis="6-31G*"))
        assert name == "H2_sp_B3LYP_6-31Gx"

    def test_missing_label_falls_back(self):
        req = _request(molecule={"atoms": ["H"], "coords": [[0, 0, 0]]})
        assert default_job_name(req).startswith("quantui_opt_")

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("my water run", "my_water_run"),
            ("../../etc/passwd", "etc_passwd"),
            ("  __weird!!name__ ", "weird_name"),
            ("***", ""),
        ],
    )
    def test_sanitize(self, raw, expected):
        assert sanitize_job_name(raw) == expected

    def test_sanitize_caps_length(self):
        assert len(sanitize_job_name("x" * 500)) <= 80


# ---------------------------------------------------------------------------
# Registry layout (JD.2 / JD.8)
# ---------------------------------------------------------------------------


class TestRegistryLayout:
    def test_job_dir_collision_gets_suffix(self, registry):
        a = registry.create(_request("a"), "cluster_slurm", job_name="run")
        b = registry.create(_request("b"), "cluster_slurm", job_name="run")
        assert Path(a.job_dir).name == "run"
        assert Path(b.job_dir).name == "run_2"

    def test_staging_path_follows_tracked_attempt(self, registry):
        rec = registry.create(_request(), "cluster_slurm", job_name="run")
        job = Path(rec.job_dir)
        rec = registry.start_attempt(rec.request_id, "101")
        # Queued: the attempt dir does not exist yet.
        assert rec.staging_path == job
        (job / "attempt-01_job101").mkdir()
        (job / "attempt-02_job102").mkdir()
        assert rec.staging_path == job / "attempt-01_job101"
        rec = registry.start_attempt(rec.request_id, "102", source="resubmit")
        assert rec.staging_path == job / "attempt-02_job102"
        assert rec.live_log_path == job / "attempt-02_job102" / "live.log"
        assert [a["source"] for a in rec.attempts] == ["submit", "resubmit"]

    def test_attempt_dirs_sorted_numerically_and_skip_latest_link(self, registry):
        rec = registry.create(_request(), "cluster_slurm", job_name="run")
        job = Path(rec.job_dir)
        for name in ("attempt-10_job9", "attempt-02_job5", "notes"):
            (job / name).mkdir()
        if sys.platform != "win32":
            (job / "latest").symlink_to("attempt-10_job9")
        assert [p.name for p in rec.attempt_dirs()] == [
            "attempt-02_job5",
            "attempt-10_job9",
        ]

    def test_parse_attempt_dir_name(self):
        assert parse_attempt_dir_name("attempt-03_job812345") == (3, "812345")
        assert parse_attempt_dir_name("latest") is None

    def test_legacy_record_unchanged(self, registry):
        """Records without job_name keep staging_root/<request_id>/ (JD.8)."""
        rec = registry.create(_request("legacy1"), "cluster_slurm")
        assert rec.job_dir is None
        assert rec.staging_path == registry.staging_root / "legacy1"
        assert rec.attempt_dirs() == []

    def test_old_record_json_loads(self, registry):
        """A record file written before M-JOBDIRS has none of the new keys."""
        rec = registry.create(_request("old1"), "cluster_slurm")
        data = rec.to_dict()
        for key in ("job_dir", "attempts", "ingested_attempts"):
            data.pop(key)
        (registry.jobs_root / "old1.json").write_text(json.dumps(data))
        loaded = registry.load("old1")
        assert loaded.job_dir is None
        assert loaded.attempts == []
        assert loaded.ingested_attempts == []


# ---------------------------------------------------------------------------
# Dispatch (JD.2 / JD.3)
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_dispatch_creates_named_job_dir(self, backend):
        with _sbatch_returning("5001"):
            rid = backend.dispatch(_request())
        rec = backend.registry.load(rid)
        job = Path(rec.job_dir)
        assert job.name == "H2_opt_B3LYP_def2-SVP"
        assert (job / "request.json").exists()
        script = (job / "submit.slurm").read_text()
        assert "#SBATCH --job-name=H2_opt_B3LYP_def2-SVP" in script
        assert f'#SBATCH --output="{job / "slurm-%j.out"}"' in script
        assert "attempt-%02d_job%s" in script
        assert '--attempt-dir "$ATTEMPT_DIR"' in script
        assert rec.slurm_job_id == "5001"
        assert rec.attempts[0]["slurm_job_id"] == "5001"

    def test_same_config_twice_gets_separate_dirs(self, backend, monkeypatch):
        monkeypatch.setattr(
            "quantui.backends.slurm.check_submit_cooldown", lambda _s: None
        )
        with _sbatch_returning("1", "2"):
            a = backend.dispatch(_request("a"))
            b = backend.dispatch(_request("b"))
        names = [Path(backend.registry.load(r).job_dir).name for r in (a, b)]
        assert names == ["H2_opt_B3LYP_def2-SVP", "H2_opt_B3LYP_def2-SVP_2"]

    def test_job_name_override_is_sanitized_and_truncated_for_slurm(self, backend):
        long_name = "my very long descriptive job name " * 3
        with _sbatch_returning("7"):
            rid = backend.dispatch(_request(), job_name=long_name)
        job = Path(backend.registry.load(rid).job_dir)
        assert job.name == sanitize_job_name(long_name)
        script = (job / "submit.slurm").read_text()
        assert f"#SBATCH --job-name={job.name[:40]}\n" in script

    def test_apptainer_binds_job_dir_outside_home(self, registry, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/nonexistent")))
        backend = SlurmBackend(
            registry=registry,
            partition="test",
            use_apptainer=True,
            apptainer_image="/img.sif",
        )
        job = registry.staging_root / "x"
        tokens = shlex.split(backend._worker_command(job / "request.json", job))
        assert f"{job}:{job}" in tokens

    def test_apptainer_skips_extra_bind_under_home(self, registry, monkeypatch):
        monkeypatch.setattr(
            Path, "home", classmethod(lambda cls: registry.staging_root.parent)
        )
        backend = SlurmBackend(
            registry=registry,
            partition="test",
            use_apptainer=True,
            apptainer_image="/img.sif",
        )
        job = registry.staging_root / "x"
        cmd = backend._worker_command(job / "request.json", job)
        assert cmd.count("--bind") == 1


# ---------------------------------------------------------------------------
# Attempt setup block (JD.3) — real bash
# ---------------------------------------------------------------------------


@needs_bash
class TestAttemptSetupScript:
    def test_each_run_gets_a_new_numbered_attempt(self, tmp_path):
        job = tmp_path / "job dir with space"
        job.mkdir()
        assert _run_attempt_setup(job, "101").returncode == 0
        (job / "attempt-01_job101" / "result.json").write_text("first")
        assert _run_attempt_setup(job, "102").returncode == 0
        assert sorted(p.name for p in job.iterdir() if p.name.startswith("att")) == [
            "attempt-01_job101",
            "attempt-02_job102",
        ]
        assert (job / "attempt-01_job101" / "result.json").read_text() == "first"
        assert os.readlink(job / "latest") == "attempt-02_job102"

    def test_numbering_continues_after_gaps(self, tmp_path):
        job = tmp_path / "job"
        (job / "attempt-05_job9").mkdir(parents=True)
        assert _run_attempt_setup(job, "10").returncode == 0
        assert (job / "attempt-06_job10").is_dir()

    def test_mkdir_clash_aborts(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        script = (
            "set -euo pipefail\n"
            + cfg.build_attempt_setup(str(job)).replace(
                'mkdir "$ATTEMPT_DIR"', 'mkdir "$ATTEMPT_DIR"; mkdir "$ATTEMPT_DIR"'
            )
            + "echo SHOULD_NOT_RUN\n"
        )
        result = subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, "SLURM_JOB_ID": "1"},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "SHOULD_NOT_RUN" not in result.stdout


# ---------------------------------------------------------------------------
# Resubmit (JD.4)
# ---------------------------------------------------------------------------


class TestResubmit:
    def test_resubmit_tracks_new_job_and_clears_error(self, backend, monkeypatch):
        monkeypatch.setattr(
            "quantui.backends.slurm.check_submit_cooldown", lambda _s: None
        )
        with _sbatch_returning("100", "200"):
            rid = backend.dispatch(_request())
            backend.registry.update_status(
                rid, "error", error={"code": "SLURM_TERMINAL", "user_message": "x"}
            )
            new_id = backend.resubmit(rid)
        rec = backend.registry.load(rid)
        assert new_id == "200"
        assert rec.slurm_job_id == "200"
        assert rec.status == "submitted"
        assert rec.error is None
        assert [a["slurm_job_id"] for a in rec.attempts] == ["100", "200"]
        assert rec.attempts[1]["source"] == "resubmit"

    def test_resubmit_rejects_active_job(self, backend):
        with _sbatch_returning("100"):
            rid = backend.dispatch(_request())
        with pytest.raises(ValueError, match="still active"):
            backend.resubmit(rid)

    def test_resubmit_rejects_legacy_record(self, backend):
        backend.registry.create(_request("legacy"), "cluster_slurm", status="error")
        with pytest.raises(ValueError, match="before per-job folders"):
            backend.resubmit("legacy")

    def test_resubmit_unknown_job(self, backend):
        with pytest.raises(ValueError, match="not in your registry"):
            backend.resubmit("nope")


# ---------------------------------------------------------------------------
# Worker (JD.3 / JD.11 guard)
# ---------------------------------------------------------------------------


class TestWorkerAttemptDir:
    @pytest.fixture
    def job(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
        monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "staging"))
        registry = JobRegistry()
        rec = registry.create(
            _request("w1", calc_type="not_a_calc"), "cluster_slurm", job_name="w"
        )
        job = Path(rec.job_dir)
        (job / "request.json").write_text(json.dumps(rec.request))
        registry.start_attempt("w1", "300")
        return registry, job

    def test_outputs_go_to_attempt_dir(self, job):
        from quantui.backends.worker import run_worker_request

        _registry, job_dir = job
        attempt = job_dir / "attempt-01_job300"
        attempt.mkdir()
        run_worker_request(job_dir / "request.json", attempt)
        assert (attempt / "live.log").exists()
        assert (attempt / "progress.json").exists()
        assert not (job_dir / "live.log").exists()

    def test_tracked_job_updates_record(self, job, monkeypatch):
        from quantui.backends.worker import run_worker_request

        registry, job_dir = job
        monkeypatch.setenv("SLURM_JOB_ID", "300")
        attempt = job_dir / "attempt-01_job300"
        attempt.mkdir()
        run_worker_request(job_dir / "request.json", attempt)
        assert registry.load("w1").status == "error"

    def test_hand_run_job_leaves_record_alone(self, job, monkeypatch):
        from quantui.backends.worker import run_worker_request

        registry, job_dir = job
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        attempt = job_dir / "attempt-02_job999"
        attempt.mkdir()
        run_worker_request(job_dir / "request.json", attempt)
        rec = registry.load("w1")
        assert rec.status == "submitted"
        assert rec.slurm_job_id == "300"


# ---------------------------------------------------------------------------
# End to end: the real submit.slurm, run twice by hand
# ---------------------------------------------------------------------------


@needs_bash
def test_submit_script_rerun_by_hand_keeps_both_attempts(tmp_path, monkeypatch):
    """Generate a real submit.slurm and run it twice with bash, as a user
    would with ``sbatch submit.slurm``. The first (failed) attempt's files
    must survive the second run untouched."""
    monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "staging"))
    backend = SlurmBackend(registry=JobRegistry(), partition="t", use_apptainer=False)
    # An unsupported calc type fails fast inside the worker, no PySCF needed.
    with _sbatch_returning("41"):
        rid = backend.dispatch(_request("e2e", calc_type="not_a_calc"))
    job = Path(backend.registry.load(rid).job_dir)
    script = job / "submit.slurm"

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    first = subprocess.run(
        ["bash", str(script)],
        env={**env, "SLURM_JOB_ID": "41"},
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert first.returncode != 0  # worker reports the unsupported calc type
    first_log = (job / "attempt-01_job41" / "live.log").read_text(encoding="utf-8")

    second = subprocess.run(
        ["bash", str(script)],
        env={**env, "SLURM_JOB_ID": "42"},
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert second.returncode != 0
    assert (job / "attempt-02_job42" / "live.log").exists()
    assert (job / "attempt-01_job41" / "live.log").read_text(
        encoding="utf-8"
    ) == first_log
    # The hand-run job (42) is not the tracked one (41): record untouched by it.
    assert backend.registry.load(rid).slurm_job_id == "41"


# ---------------------------------------------------------------------------
# History provenance (JD.5 / JD.6)
# ---------------------------------------------------------------------------


class TestHistoryProvenance:
    @pytest.mark.parametrize(
        "calc_type", ["single_point", "frequency", "reorganization_energy"]
    )
    def test_legacy_record_ingest_is_tagged_slurm(
        self, tmp_path, monkeypatch, calc_type
    ):
        """All three save_result paths carry the provenance extras."""
        from quantui.backends.slurm_ingest import ingest_staging_success
        from tests.slurm_ingest_helpers import (
            make_staging_record,
            patch_results_root,
            sample_payload,
        )

        patch_results_root(tmp_path, monkeypatch)
        record, _staging = make_staging_record(
            tmp_path, sample_payload(calc_type), calc_type=calc_type
        )
        record.slurm_job_id = "4242"
        saved = ingest_staging_success(record)
        data = json.loads((saved / "result.json").read_text())
        assert data["execution_backend"] == "slurm"
        assert data["slurm"]["job_id"] == "4242"
        assert data["slurm"]["attempt"] is None
        assert data["slurm"]["request_id"] == record.request_id

    def test_attempt_dir_ingest_records_attempt_and_its_job_id(
        self, tmp_path, monkeypatch, registry
    ):
        from quantui.backends.slurm_ingest import ingest_staging_success
        from tests.slurm_ingest_helpers import patch_results_root, sample_payload

        patch_results_root(tmp_path, monkeypatch)
        rec = registry.create(_request(), "cluster_slurm", job_name="run")
        rec = registry.start_attempt(rec.request_id, "100")
        # A hand-run second attempt the record does not track.
        attempt = Path(rec.job_dir) / "attempt-02_job555"
        attempt.mkdir()
        (attempt / "result.json").write_text(json.dumps(sample_payload("single_point")))
        saved = ingest_staging_success(rec, attempt_dir=attempt)
        info = json.loads((saved / "result.json").read_text())["slurm"]
        assert info["job_id"] == "555"
        assert info["attempt"] == 2
        assert info["attempt_dir"] == str(attempt)
        assert info["job_dir"] == rec.job_dir

    def test_history_marker_and_card_row(self):
        from quantui.app_formatters import format_past_result, slurm_history_marker

        data = {
            "calc_type": "single_point",
            "formula": "H2",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -1.1,
            "energy_ev": -30.0,
            "converged": True,
            "execution_backend": "slurm",
            "slurm": {
                "job_id": "812345",
                "attempt": 2,
                "job_dir": "/scratch/u/run",
                "attempt_dir": "/scratch/u/run/attempt-02_job812345",
            },
        }
        assert slurm_history_marker(data) == "🖥 SLURM 812345·a2 "
        card = format_past_result(data)
        assert "Ran on" in card
        assert "job 812345 &middot; attempt 2" in card
        assert "/scratch/u/run/attempt-02_job812345" in card

    def test_local_result_has_no_marker_or_row(self):
        from quantui.app_formatters import format_past_result, slurm_history_marker

        data = {
            "calc_type": "single_point",
            "formula": "H2",
            "method": "RHF",
            "basis": "STO-3G",
            "energy_hartree": -1.1,
            "energy_ev": -30.0,
            "converged": True,
        }
        assert slurm_history_marker(data) == ""
        assert "Ran on" not in format_past_result(data)


# ---------------------------------------------------------------------------
# Hand-run attempts reach History; nothing is ingested twice (JD.11)
# ---------------------------------------------------------------------------


def _finished_attempt(job_dir: Path, name: str) -> Path:
    from tests.slurm_ingest_helpers import sample_payload

    attempt = job_dir / name
    attempt.mkdir()
    (attempt / "result.json").write_text(json.dumps(sample_payload("single_point")))
    (attempt / "live.log").write_text(f"log of {name}\n")
    return attempt


def _jobs_tab_app(registry, **overrides):
    from types import SimpleNamespace

    fields = dict(
        _job_registry=registry,
        _slurm_jobs_summary_html=SimpleNamespace(value=""),
        _slurm_jobs_table_html=SimpleNamespace(value=""),
        _slurm_jobs_select=SimpleNamespace(options=[], value="", disabled=True),
        _slurm_jobs_status_html=SimpleNamespace(value=""),
        _slurm_active_request_id=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestHandRunAttempts:
    @pytest.fixture
    def job(self, tmp_path, monkeypatch, registry):
        from tests.slurm_ingest_helpers import patch_results_root

        results = patch_results_root(tmp_path, monkeypatch)
        rec = registry.create(_request(), "cluster_slurm", job_name="run")
        registry.start_attempt(rec.request_id, "100")
        registry.update_status(rec.request_id, "error")
        return registry, rec.request_id, Path(rec.job_dir), results

    def test_uningested_attempts_lists_only_finished_new_ones(self, job):
        from quantui.backends.slurm_ingest import uningested_attempts

        registry, rid, job_dir, _ = job
        (job_dir / "attempt-01_job100").mkdir()  # failed: no result.json
        done = _finished_attempt(job_dir, "attempt-02_job555")
        assert uningested_attempts(registry.load(rid)) == [done]

    def test_ingest_attempt_marks_ingested_without_retargeting_record(self, job):
        from quantui.backends.slurm_ingest import ingest_attempt, uningested_attempts

        registry, rid, job_dir, results = job
        done = _finished_attempt(job_dir, "attempt-02_job555")
        saved = ingest_attempt(registry, registry.load(rid), done)
        assert (saved / "pyscf.log").read_text() == "log of attempt-02_job555\n"
        rec = registry.load(rid)
        assert rec.ingested_attempts == ["attempt-02_job555"]
        assert rec.result_dir is None  # hand-run attempt: record not retargeted
        assert uningested_attempts(rec) == []
        assert len(list(results.iterdir())) == 1

    @patch("quantui.app_slurm.is_slurm_available", return_value=False)
    def test_refresh_ingests_hand_run_attempt_once(self, _avail, job):
        from quantui.app_slurm import refresh_slurm_jobs_tab

        registry, _rid, job_dir, results = job
        _finished_attempt(job_dir, "attempt-02_job555")
        app = _jobs_tab_app(registry)
        with patch("quantui.app_runflow.refresh_results_browser") as refresh:
            refresh_slurm_jobs_tab(app)
            refresh_slurm_jobs_tab(app)
        assert refresh.call_count == 1
        assert "Saved 1 finished cluster run" in app._slurm_jobs_status_html.value
        [entry] = list(results.iterdir())
        info = json.loads((entry / "result.json").read_text())["slurm"]
        assert (info["job_id"], info["attempt"]) == ("555", 2)

    @patch("quantui.app_slurm.is_slurm_available", return_value=False)
    def test_refresh_leaves_monitored_attempt_to_the_monitor(self, _avail, job):
        from quantui.app_slurm import ingest_new_slurm_attempts

        registry, rid, job_dir, results = job
        registry.update_status(rid, "success")
        _finished_attempt(job_dir, "attempt-01_job100")  # the tracked job
        app = _jobs_tab_app(registry, _slurm_active_request_id=rid)
        assert ingest_new_slurm_attempts(app) == []
        app._slurm_active_request_id = None
        assert len(ingest_new_slurm_attempts(app)) == 1

    def test_view_on_finished_job_does_not_duplicate_history(self, job):
        """Reconnecting to an ingested job shows it, but saves nothing new."""
        from quantui.app_slurm import _ingest_success

        registry, rid, job_dir, results = job
        registry.update_status(rid, "success")
        _finished_attempt(job_dir, "attempt-01_job100")
        app = _jobs_tab_app(
            registry,
            run_status=_Value(),
            run_output=_Sink(),
            result_output=_Sink(),
        )
        with patch("quantui.app_runflow.refresh_results_browser"):
            _ingest_success(app, registry.load(rid))
            _ingest_success(app, registry.load(rid))
        assert len(list(results.iterdir())) == 1
        rec = registry.load(rid)
        assert rec.result_dir == str(next(results.iterdir()))

    def test_legacy_record_already_ingested_is_detected(self, tmp_path, registry):
        from quantui.backends.slurm_ingest import already_ingested

        rec = registry.create(_request("leg"), "cluster_slurm")
        staging = rec.staging_path
        rec.result_dir = str(staging)  # worker's value: not yet ingested
        assert already_ingested(rec, staging) is False
        saved = tmp_path / "results" / "x"
        saved.mkdir(parents=True)
        (saved / "result.json").write_text("{}")
        rec.result_dir = str(saved)  # app's value after ingest
        assert already_ingested(rec, staging) is True


class _Value:
    value = ""


class _Sink:
    def append_stdout(self, _text):
        pass

    def append_display_data(self, _obj):
        pass


# ---------------------------------------------------------------------------
# Resubmit button (JD.4 UI)
# ---------------------------------------------------------------------------


class TestResubmitButton:
    @pytest.fixture
    def app(self, backend, monkeypatch):
        monkeypatch.setattr(
            "quantui.backends.slurm.check_submit_cooldown", lambda _s: None
        )
        monkeypatch.setattr(
            "quantui.app_slurm.slurm_backend_for_app", lambda _app: backend
        )
        monkeypatch.setattr("quantui.app_slurm.is_slurm_available", lambda: False)
        with _sbatch_returning("100"):
            rid = backend.dispatch(_request())
        backend.registry.update_status(rid, "error")
        app = _jobs_tab_app(backend.registry, _calc_running=False)
        app._slurm_jobs_select.value = rid
        return app, rid

    def test_resubmit_starts_monitoring_new_attempt(self, app, backend):
        from quantui.app_slurm import on_slurm_jobs_resubmit_clicked

        app_ns, rid = app
        with (
            _sbatch_returning("200"),
            patch("quantui.app_slurm.attach_slurm_job") as attach,
        ):
            on_slurm_jobs_resubmit_clicked(app_ns)
        attach.assert_called_once_with(app_ns, rid)
        assert "Resubmitted as SLURM job 200" in app_ns._slurm_jobs_status_html.value
        assert backend.registry.load(rid).slurm_job_id == "200"

    def test_resubmit_blocked_while_running(self, app, backend):
        from quantui.app_slurm import resubmit_slurm_job

        app_ns, rid = app
        app_ns._calc_running = True
        ok, message = resubmit_slurm_job(app_ns, rid)
        assert not ok
        assert "already running" in message
        assert backend.registry.load(rid).slurm_job_id == "100"

    def test_resubmit_legacy_record_explains(self, app, backend):
        from quantui.app_slurm import resubmit_slurm_job

        app_ns, _rid = app
        backend.registry.create(_request("legacy"), "cluster_slurm", status="error")
        ok, message = resubmit_slurm_job(app_ns, "legacy")
        assert not ok
        assert "before per-job folders" in message


# ---------------------------------------------------------------------------
# Configurable job root (JD.1)
# ---------------------------------------------------------------------------


class TestJobRoot:
    @pytest.fixture(autouse=True)
    def _private_settings(self, tmp_path, monkeypatch):
        # Keep writes out of the suite-wide settings file other tests read.
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))

    def test_precedence_env_then_setting_then_default(self, tmp_path, monkeypatch):
        from quantui.user_settings import UserSettings

        monkeypatch.delenv("QUANTUI_STAGING_DIR", raising=False)
        settings = UserSettings.load()
        settings.compute.slurm_job_root = ""
        settings.save()
        assert cfg.default_staging_root() == cfg.DEFAULT_STAGING_ROOT.expanduser()

        settings.compute.slurm_job_root = str(tmp_path / "scratch")
        settings.save()
        assert cfg.default_staging_root() == tmp_path / "scratch"

        monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "env"))
        assert cfg.default_staging_root() == tmp_path / "env"
        assert cfg.staging_root_env_configured()

    def test_setting_round_trips(self):
        from quantui.user_settings import UserSettings

        data = UserSettings().to_dict()
        data["compute"]["slurm_job_root"] = "  /work/me/jobs "
        assert UserSettings._from_dict(data).compute.slurm_job_root == "/work/me/jobs"
        data["compute"]["slurm_job_root"] = 5
        assert UserSettings._from_dict(data).compute.slurm_job_root == ""

    def _app(self):
        from types import SimpleNamespace

        from quantui.user_settings import UserSettings

        return SimpleNamespace(
            _user_settings=UserSettings.load(),
            slurm_job_root_note=SimpleNamespace(value=""),
            _job_registry=object(),
        )

    def test_handler_saves_absolute_path_and_resets_registry(
        self, tmp_path, monkeypatch
    ):
        from quantui.app_slurm import on_slurm_job_root_changed
        from quantui.user_settings import UserSettings

        monkeypatch.delenv("QUANTUI_STAGING_DIR", raising=False)
        app = self._app()
        target = tmp_path / "new root"
        on_slurm_job_root_changed(app, str(target))
        assert target.is_dir()
        assert UserSettings.load().compute.slurm_job_root == str(target)
        assert app._job_registry is None
        assert "New cluster jobs will be created" in app.slurm_job_root_note.value

    def test_handler_rejects_relative_path(self, monkeypatch):
        from quantui.app_slurm import on_slurm_job_root_changed
        from quantui.user_settings import UserSettings

        monkeypatch.delenv("QUANTUI_STAGING_DIR", raising=False)
        before = UserSettings.load().compute.slurm_job_root
        app = self._app()
        on_slurm_job_root_changed(app, "relative/jobs")
        assert "full path" in app.slurm_job_root_note.value
        assert UserSettings.load().compute.slurm_job_root == before


# ---------------------------------------------------------------------------
# Job name field on the Calculate tab (JD.2 UI)
# ---------------------------------------------------------------------------


@patch("quantui.app_slurm.threading.Thread")
@patch("quantui.app_slurm.is_slurm_available", return_value=True)
def test_submit_passes_job_name_and_clears_field(_avail, _thread, backend, monkeypatch):
    from types import SimpleNamespace

    from quantui.app_slurm import submit_slurm_run

    monkeypatch.setattr("quantui.app_slurm.slurm_backend_for_app", lambda _app: backend)
    monkeypatch.setattr(
        "quantui.app_slurm.build_calculation_request", lambda _app: _request("ui1")
    )
    monkeypatch.setattr("quantui.app_slurm.calc_type_key_from_app", lambda _app: "sp")
    monkeypatch.setattr("quantui.app_slurm._SUPPORTED_SLURM_CALC_TYPES", {"sp"})
    monkeypatch.setattr("quantui.app_slurm.slurm_submit_block_reason", lambda _a: None)
    app = _jobs_tab_app(
        backend.registry,
        _slurm_job_name_txt=SimpleNamespace(value="  water scan #1 "),
        _calc_running=False,
        run_status=_Value(),
        run_output=_Sink(),
        run_btn=SimpleNamespace(disabled=False),
        cancel_btn=SimpleNamespace(disabled=True),
        log_clear_btn=SimpleNamespace(disabled=False),
    )
    with _sbatch_returning("321"):
        submit_slurm_run(app)
    rec = backend.registry.load("ui1")
    assert Path(rec.job_dir).name == "water_scan_1"
    assert app._slurm_job_name_txt.value == ""


# ---------------------------------------------------------------------------
# Descriptive names for shareable files (JD.7)
# ---------------------------------------------------------------------------


class TestDescriptiveNames:
    def test_worker_renames_shareable_files(self, tmp_path):
        from quantui.backends.worker import _apply_descriptive_names

        (tmp_path / "result.molden").write_text("m")
        (tmp_path / "trajectory.xyz").write_text("x")
        (tmp_path / "orbitals.npz").write_text("o")
        names = _apply_descriptive_names(tmp_path, "H2_opt_B3LYP_def2-SVP")
        assert names == {
            "result.molden": "H2_opt_B3LYP_def2-SVP.molden",
            "trajectory.xyz": "H2_opt_B3LYP_def2-SVP_trajectory.xyz",
        }
        assert (tmp_path / "H2_opt_B3LYP_def2-SVP.molden").read_text() == "m"
        assert (tmp_path / "orbitals.npz").exists()  # read back by the app

    def test_ingest_restores_fixed_names_in_history(
        self, tmp_path, monkeypatch, registry
    ):
        from quantui.backends.slurm_ingest import ingest_staging_success
        from tests.slurm_ingest_helpers import patch_results_root, sample_payload

        patch_results_root(tmp_path, monkeypatch)
        rec = registry.create(_request(), "cluster_slurm", job_name="run")
        attempt = Path(rec.job_dir) / "attempt-01_job1"
        attempt.mkdir()
        (attempt / "run.molden").write_text("[Molden Format]\n")
        payload = sample_payload("single_point")
        payload["artifact_names"] = {"result.molden": "run.molden"}
        (attempt / "result.json").write_text(json.dumps(payload))
        saved = ingest_staging_success(rec, attempt_dir=attempt)
        assert (saved / "result.molden").read_text() == "[Molden Format]\n"
        assert (attempt / "run.molden").exists()  # the job folder copy stays

    def test_legacy_worker_run_keeps_fixed_names(self, tmp_path, monkeypatch):
        """No --attempt-dir (legacy staging) → no renaming, as before."""
        from quantui.backends.worker import run_worker_request

        monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
        monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "staging"))
        staging = tmp_path / "staging" / "leg"
        staging.mkdir(parents=True)
        (staging / "request.json").write_text(
            json.dumps(_request("leg", calc_type="nope").to_dict())
        )
        (staging / "result.molden").write_text("m")
        run_worker_request(staging / "request.json")
        assert (staging / "result.molden").exists()
