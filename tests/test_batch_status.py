"""Batch job state from the job folder (M-BATCH2 B2.9 / B2.10)."""

from __future__ import annotations

import json
import time

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.batch_status import (
    JOB_LOG,
    STALE_PROGRESS_S,
    job_view,
    job_views,
)
from quantui.backends.registry import JobRegistry


@pytest.fixture
def roots(tmp_path, monkeypatch):
    jobs, staging = tmp_path / "jobs", tmp_path / "staging"
    monkeypatch.setenv("QUANTUI_JOBS_DIR", str(jobs))
    monkeypatch.setenv("QUANTUI_STAGING_DIR", str(staging))
    return JobRegistry(jobs_root=jobs, staging_root=staging)


def _terminal_job(registry, name="water_opt", *, created="2026-10-30T10:00:00+00:00"):
    """A job folder + record as the quantui-batch launcher writes them."""
    job = registry.staging_root / name
    job.mkdir(parents=True)
    request = CalculationRequest(
        request_id=f"{name}-1",
        calc_type="geometry_opt",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        molecule={"atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.74]]},
    )
    (job / "request.json").write_text(json.dumps(request.to_dict()))
    record = registry.create(request, "cluster_slurm", status="prepared")
    record.job_dir = str(job)
    record.created_at = created
    registry.save(record)
    return job, record


def _submit(job, job_id):
    with open(job / JOB_LOG, "a") as fh:
        fh.write(f"{job_id}\t2026-10-30T10:00:00Z\t[]\n")


def _progress(
    attempt, *, stage="running", message="Optimizing geometry", pct=40.0, age=0
):
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "progress.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "message": message,
                "percent": pct,
                "timestamp": time.time() - age,
            }
        )
    )


class TestJobView:
    def test_prepared_but_never_submitted(self, roots):
        _job, record = _terminal_job(roots)
        view = job_view(record)
        assert view.state == "prepared" and not view.unfinished

    def test_submitted_but_not_started(self, roots):
        job, record = _terminal_job(roots)
        _submit(job, "1001")
        view = job_view(record)
        assert view.state == "waiting" and view.job_id == "1001"
        assert "submitted as job 1001" in view.detail

    def test_running_shows_the_workers_progress(self, roots):
        job, record = _terminal_job(roots)
        _submit(job, "1001")
        _progress(job / "attempt-01_job1001", age=120)
        view = job_view(record)
        assert view.state == "running" and view.percent == 40.0
        assert view.detail == "Optimizing geometry · updated 2 min ago"

    def test_stale_progress_says_it_may_have_stopped(self, roots):
        job, record = _terminal_job(roots)
        _submit(job, "1001")
        _progress(job / "attempt-01_job1001", age=STALE_PROGRESS_S + 60)
        view = job_view(record)
        assert view.state == "running" and "may have stopped" in view.detail
        # With the queue in view it is not a guess.
        view = job_view(record, queue={"1001": "RUNNING"})
        assert view.state == "running" and "may have stopped" not in view.detail

    def test_the_newest_submission_counts(self, roots):
        job, record = _terminal_job(roots)
        _submit(job, "1001")
        _progress(job / "attempt-01_job1001", stage="error", message="Out of memory")
        assert job_view(record).state == "failed"
        _submit(job, "1002")  # a rerun
        view = job_view(record, queue={"1002": "PENDING"})
        assert view.state == "waiting" and view.detail == "in the queue (PENDING)"
        assert view.job_id == "1002"

    def test_finished(self, roots):
        job, record = _terminal_job(roots)
        _submit(job, "1001")
        attempt = job / "attempt-01_job1001"
        _progress(attempt)
        (attempt / "result.json").write_text("{}")
        assert job_view(record).state == "done"

    def test_hand_run_sbatch_without_a_job_log(self, roots):
        # quantui submit --prepare-only, then sbatch by hand: no job log.
        job, record = _terminal_job(roots)
        _progress(job / "attempt-01_job77", message="Running frequency analysis")
        view = job_view(record)
        assert view.state == "running" and view.detail.startswith("Running frequency")

    def test_views_newest_first_and_unfinished_only(self, roots):
        old, _ = _terminal_job(roots, "old", created="2026-10-29T10:00:00+00:00")
        new, _ = _terminal_job(roots, "new", created="2026-10-30T10:00:00+00:00")
        _submit(new, "2")
        assert [v.name for v in job_views(roots)] == ["new", "old"]
        assert [v.name for v in job_views(roots, unfinished_only=True)] == ["new"]


class TestJobsCommand:
    def test_lists_jobs_without_squeue(self, roots, monkeypatch, capsys):
        from quantui.cli import main

        monkeypatch.setattr("shutil.which", lambda name: None)
        job, _ = _terminal_job(roots, "water_opt")
        _submit(job, "1001")
        _progress(job / "attempt-01_job1001", pct=40.0, age=60)
        _terminal_job(roots, "never_sent", created="2026-10-29T10:00:00+00:00")
        assert main(["jobs"]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0].split()[:4] == ["JOB", "TYPE", "METHOD/BASIS", "STATE"]
        assert "water_opt" in out[1] and "running 40%" in out[1] and "1001" in out[1]
        assert "never_sent" in out[2] and "prepared" in out[2]
        assert main(["jobs", "--running"]) == 0
        assert "never_sent" not in capsys.readouterr().out

    def test_refreshes_from_the_queue_when_it_can(self, roots, monkeypatch, capsys):
        from quantui.backends import slurm
        from quantui.cli import main

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
        refreshed = []
        monkeypatch.setattr(
            slurm.SlurmBackend,
            "refresh_registry_statuses",
            lambda self: refreshed.append(True),
        )
        monkeypatch.setattr(
            slurm.SlurmBackend,
            "batch_poll_slurm_statuses",
            lambda self, ids: {i: "PENDING" for i in ids},
        )
        job, _ = _terminal_job(roots)
        _submit(job, "1001")
        assert main(["jobs"]) == 0
        assert refreshed and "in the queue (PENDING)" in capsys.readouterr().out


class TestHistoryNote:
    def test_shows_unfinished_terminal_jobs(self, roots, tmp_path, monkeypatch):
        from quantui.app import QuantUIApp
        from quantui.app_slurm import ingest_finished_jobs_on_startup

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path / "results"))
        app = QuantUIApp()
        app._job_registry = roots
        job, _ = _terminal_job(roots, "hcn_ts")
        _submit(job, "1001")
        _progress(job / "attempt-01_job1001", message="Transition-state search, step 3")
        ingest_finished_jobs_on_startup(app)
        assert app._batch_jobs_html.layout.display == ""
        assert "hcn_ts" in app._batch_jobs_html.value
        assert "Transition-state search, step 3" in app._batch_jobs_html.value
        (job / "attempt-01_job1001" / "result.json").write_text("{}")
        with pytest.MonkeyPatch.context() as mp:
            # The finished job is ingested elsewhere; only the note is checked.
            mp.setattr("quantui.app_slurm.ingest_new_slurm_attempts", lambda app: [])
            ingest_finished_jobs_on_startup(app)
        assert app._batch_jobs_html.layout.display == "none"


def test_ingest_marks_a_prepared_record_succeeded(roots, tmp_path, monkeypatch):
    from quantui.backends.slurm_ingest import ingest_attempt
    from tests.slurm_ingest_helpers import sample_payload

    monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path / "results"))
    job, record = _terminal_job(roots)
    attempt = job / "attempt-01_job1001"
    attempt.mkdir()
    (attempt / "result.json").write_text(json.dumps(sample_payload("single_point")))
    ingest_attempt(roots, record, attempt)
    assert roots.load(record.request_id).status == "success"
