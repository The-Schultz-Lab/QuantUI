"""
Terminal batch submission: ``.xyz`` input, ``quantui submit --prepare-only``,
``quantui install-launcher``, and the host-side ``quantui-batch`` launcher.

The launcher tests run the real installed script end to end against fake
``apptainer``/``sbatch``/``squeue``/``sacct``/``srun`` commands on ``PATH``.
The fake ``apptainer exec IMAGE quantui ...`` runs this checkout's CLI with the
test interpreter, so the prepare step is the real one. Nothing here touches a
real cluster; on-cluster behaviour still needs the NCShare check list.
"""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from quantui import cli
from quantui.backends.batch_input import (
    BatchInputError,
    load_request,
    parse_option_pairs,
    request_from_xyz,
)
from quantui.backends.registry import JobRegistry

WATER_XYZ = """3
water
O   0.0000   0.0000   0.1173
H   0.0000   0.7572  -0.4692
H   0.0000  -0.7572  -0.4692
"""

needs_posix = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the launcher and its fake Slurm commands are POSIX scripts",
)


def _capture(argv):
    out, err = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = cli.main(argv)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def water(tmp_path):
    path = tmp_path / "water.xyz"
    path.write_text(WATER_XYZ)
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.delenv("QUANTUI_ENABLE_SLURM", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Building requests from files
# ---------------------------------------------------------------------------


class TestRequestFromXyz:
    def test_builds_request_with_molecule_and_settings(self, water):
        req, warnings = request_from_xyz(
            water,
            calc_type="frequency",
            method="B3LYP",
            basis="def2-SVP",
            preopt=True,
            options={"nstates": 5},
        )
        assert warnings == []
        assert req.calc_type == "frequency"
        assert (req.method, req.basis) == ("B3LYP", "def2-SVP")
        assert req.molecule["atoms"] == ["O", "H", "H"]
        assert req.molecule["coords"][1] == pytest.approx([0.0, 0.7572, -0.4692])
        assert (req.molecule["charge"], req.molecule["multiplicity"]) == (0, 1)
        assert req.options == {"nstates": 5, "preopt_before_run": True}
        assert req.request_id.startswith("water-")
        assert req.run_context == {"source_file": "water.xyz"}

    def test_request_ids_are_unique_per_call(self, water):
        a, _ = request_from_xyz(
            water, calc_type="single_point", method="RHF", basis="STO-3G"
        )
        b, _ = request_from_xyz(
            water, calc_type="single_point", method="RHF", basis="STO-3G"
        )
        assert a.request_id != b.request_id

    def test_requires_calc_type(self, water):
        with pytest.raises(BatchInputError, match="needs --calc"):
            request_from_xyz(water, calc_type=None, method="RHF", basis="STO-3G")

    def test_rejects_unknown_calc_type(self, water):
        with pytest.raises(BatchInputError, match="unknown --calc"):
            request_from_xyz(water, calc_type="md", method="RHF", basis="STO-3G")

    def test_rejects_impossible_spin(self, water):
        # Water has 10 electrons: a doublet is impossible.
        with pytest.raises(BatchInputError, match="do not fit"):
            request_from_xyz(
                water,
                calc_type="single_point",
                method="RHF",
                basis="STO-3G",
                multiplicity=2,
            )

    def test_rejects_unreadable_xyz(self, tmp_path):
        bad = tmp_path / "bad.xyz"
        bad.write_text("not a molecule\n")
        with pytest.raises(BatchInputError, match="could not read XYZ"):
            request_from_xyz(
                bad, calc_type="single_point", method="RHF", basis="STO-3G"
            )

    def test_warns_on_unknown_method_and_useless_preopt(self, water):
        _, warnings = request_from_xyz(
            water,
            calc_type="single_point",
            method="B3LPY",
            basis="STO-3G",
            preopt=True,
        )
        assert any("B3LPY" in w for w in warnings)
        assert any("--preopt has no effect" in w for w in warnings)

    def test_xyz_defaults_follow_the_app(self, water):
        from quantui import config

        req, _ = load_request(water, calc_type="single_point")
        assert (req.method, req.basis) == (config.DEFAULT_METHOD, config.DEFAULT_BASIS)
        assert (req.charge, req.multiplicity) == (0, 1)


class TestJsonOverrides:
    def _json(self, tmp_path):
        path = tmp_path / "req.json"
        path.write_text(
            json.dumps(
                {
                    "request_id": "keep-me",
                    "calc_type": "single_point",
                    "method": "RHF",
                    "basis": "STO-3G",
                    "charge": 0,
                    "multiplicity": 1,
                    "molecule": {
                        "atoms": ["H", "H"],
                        "coords": [[0, 0, 0], [0, 0, 0.74]],
                        "charge": 0,
                        "multiplicity": 1,
                    },
                }
            )
        )
        return path

    def test_unset_flags_keep_file_values(self, tmp_path):
        req, _ = load_request(self._json(tmp_path))
        assert (req.request_id, req.calc_type, req.method) == (
            "keep-me",
            "single_point",
            "RHF",
        )

    def test_flags_override_file_and_keep_molecule_in_step(self, tmp_path):
        req, _ = load_request(
            self._json(tmp_path),
            calc_type="frequency",
            basis="def2-SVP",
            charge=1,
            multiplicity=2,
            preopt=True,
        )
        assert (req.calc_type, req.basis) == ("frequency", "def2-SVP")
        assert (req.charge, req.molecule["charge"]) == (1, 1)
        assert (req.multiplicity, req.molecule["multiplicity"]) == (2, 2)
        assert req.options["preopt_before_run"] is True


class TestOptionPairs:
    def test_values_parse_as_json_when_possible(self):
        assert parse_option_pairs(
            ["nstates=10", "atom_indices=[0, 1]", "scan_type=bond", "x=true"]
        ) == {"nstates": 10, "atom_indices": [0, 1], "scan_type": "bond", "x": True}

    @pytest.mark.parametrize("bad", ["nstates", "=3"])
    def test_rejects_malformed_pairs(self, bad):
        with pytest.raises(BatchInputError):
            parse_option_pairs([bad])


# ---------------------------------------------------------------------------
# quantui submit
# ---------------------------------------------------------------------------


class TestSubmitPrepareOnly:
    def test_writes_job_dir_without_sbatch_or_site_gate(
        self, roots, water, monkeypatch
    ):
        monkeypatch.setattr(
            "quantui.backends.slurm.subprocess.run",
            lambda *a, **k: pytest.fail("sbatch must not run in --prepare-only"),
        )
        rc, out, err = _capture(
            [
                "submit",
                str(water),
                "--prepare-only",
                "--calc",
                "geometry_opt",
                "--method",
                "B3LYP",
                "--basis",
                "def2-SVP",
                "--job-name",
                "water opt",
                "--apptainer-image",
                "/images/q.sif",
            ]
        )
        assert rc == 0, err
        [script_line] = out.splitlines()
        script = Path(script_line)
        assert script.name == "submit.slurm" and script.is_file()
        job_dir = script.parent
        assert job_dir.name == "water_opt"
        assert job_dir.parent == roots / "staging"
        text = script.read_text()
        assert "/images/q.sif" in text
        assert "#SBATCH --job-name=water_opt" in text
        request = json.loads((job_dir / "request.json").read_text())
        assert request["calc_type"] == "geometry_opt"
        assert request["molecule"]["atoms"] == ["O", "H", "H"]
        assert "prepared water_opt" in err

        [record] = JobRegistry().list_all()
        assert record.status == "prepared"
        assert record.slurm_job_id is None
        assert record.job_dir == str(job_dir)

    def test_prepared_jobs_do_not_count_toward_concurrency(self, roots, water):
        for _ in range(3):  # above the default limit of 2
            rc, _out, err = _capture(
                ["submit", str(water), "--prepare-only", "--calc", "single_point"]
            )
            assert rc == 0, err
        assert JobRegistry().list_active() == []

    def test_bad_input_reports_and_continues(self, roots, water, tmp_path):
        rc, out, err = _capture(
            [
                "submit",
                str(tmp_path / "missing.xyz"),
                str(water),
                "--prepare-only",
                "--calc",
                "single_point",
            ]
        )
        assert rc == 1
        assert len(out.splitlines()) == 1  # the good one was still prepared
        assert "missing.xyz" in err

    def test_resource_overrides_are_validated(self, roots, water):
        rc, out, err = _capture(
            [
                "submit",
                str(water),
                "--prepare-only",
                "--calc",
                "single_point",
                "--memory-gb",
                "100000",
            ]
        )
        assert rc == 1
        assert out == ""
        assert "rejected" in err

    def test_job_name_with_several_inputs_gets_suffixes(self, roots, water):
        rc, out, err = _capture(
            [
                "submit",
                str(water),
                str(water),
                "--prepare-only",
                "--calc",
                "single_point",
                "--job-name",
                "x",
            ]
        )
        assert rc == 0, err
        assert [Path(line).parent.name for line in out.splitlines()] == ["x", "x_2"]

    def test_dry_run_and_prepare_only_conflict(self, roots, water):
        rc, _out, err = _capture(["submit", str(water), "--dry-run", "--prepare-only"])
        assert rc == 2
        assert "conflict" in err

    def test_xyz_dry_run_prints_estimate(self, roots, water):
        rc, out, err = _capture(
            ["submit", str(water), "--dry-run", "--calc", "frequency"]
        )
        assert rc == 0, err
        assert "frequency" in out and "cores=" in out and "not submitted" in out

    def test_bad_option_pair_is_a_usage_error(self, roots, water):
        rc, _out, err = _capture(
            ["submit", str(water), "--dry-run", "--calc", "tddft", "--option", "oops"]
        )
        assert rc == 2
        assert "KEY=VALUE" in err


# ---------------------------------------------------------------------------
# quantui install-launcher
# ---------------------------------------------------------------------------


class TestInstallLauncher:
    def test_needs_an_image(self, tmp_path, monkeypatch):
        monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
        monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
        rc, _out, err = _capture(["install-launcher", str(tmp_path / "bin")])
        assert rc == 1
        assert "inside the QuantUI image" in err

    def test_uses_running_image_and_fills_defaults(self, roots, tmp_path, monkeypatch):
        from quantui import __version__

        monkeypatch.setenv("APPTAINER_CONTAINER", "/opt/images/quantui.sif")
        rc, out, _err = _capture(["install-launcher", str(tmp_path / "bin")])
        assert rc == 0
        launcher = tmp_path / "bin" / "quantui-batch"
        text = launcher.read_text()
        assert 'DEFAULT_IMAGE = "/opt/images/quantui.sif"' in text
        assert f'DEFAULT_STAGING = "{roots / "staging"}"' in text
        assert f'INSTALLED_FROM = "{__version__}"' in text
        assert "@QUANTUI_" not in text
        assert os.access(launcher, os.X_OK) or sys.platform == "win32"
        assert "quantui-batch help" in out

    def test_refuses_to_overwrite_without_force(self, roots, tmp_path):
        dest = tmp_path / "bin"
        assert _capture(["install-launcher", str(dest), "--image", "/x.sif"])[0] == 0
        rc, _out, err = _capture(["install-launcher", str(dest), "--image", "/y.sif"])
        assert rc == 1 and "--force" in err
        assert (
            _capture(["install-launcher", str(dest), "--image", "/y.sif", "--force"])[0]
            == 0
        )
        assert "/y.sif" in (dest / "quantui-batch").read_text()

    def test_launcher_template_is_python36_syntax(self):
        import ast
        from importlib import resources

        src = (
            resources.files("quantui")
            .joinpath("data")
            .joinpath("launcher")
            .joinpath("quantui_batch.py")
            .read_text(encoding="utf-8")
        )
        # The launcher runs on the cluster host's own python3, not the image's.
        ast.parse(src, feature_version=(3, 6))


# ---------------------------------------------------------------------------
# quantui-batch, end to end against fake Slurm/Apptainer commands
# ---------------------------------------------------------------------------


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def cluster(tmp_path):
    """Fake cluster: bin dir of mock commands + env + helpers."""
    mock = tmp_path / "mock"
    bindir = mock / "bin"
    bindir.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    image = tmp_path / "quantui.sif"
    image.write_text("fake image")
    py = sys.executable
    repo = str(Path(__file__).resolve().parents[1])

    _write_exe(
        bindir / "apptainer",
        f"""#!{py}
import os, sys
args = sys.argv[1:]
assert args[0] == "exec", args
args = args[1:]
while args and args[0].startswith("--"):
    flag = args.pop(0)
    if flag in ("--bind", "--pwd"):
        args.pop(0)
image, cmd = args[0], args[1:]
with open(os.path.join({str(mock)!r}, "apptainer.log"), "a") as fh:
    fh.write(image + "\\n")
with open(os.path.join({str(mock)!r}, "apptainer_env.log"), "a") as fh:
    fh.write("OPENBLAS_NUM_THREADS=" + os.environ.get("OPENBLAS_NUM_THREADS", "") + "\\n")
assert cmd[0] == "quantui", cmd
os.execv({py!r}, [{py!r}, "-m", "quantui.cli"] + cmd[1:])
""",
    )
    _write_exe(
        bindir / "srun",
        f"""#!{py}
import os, sys
args = sys.argv[1:]
with open(os.path.join({str(mock)!r}, "srun.log"), "a") as fh:
    fh.write(" ".join(args) + "\\n")
with open(os.path.join({str(mock)!r}, "srun_env.log"), "a") as fh:
    fh.write("SLURM_JOB_ID=" + os.environ.get("SLURM_JOB_ID", "") + "\\n")
while args[0].startswith("--"):
    args.pop(0)
os.execvp(args[0], args)
""",
    )
    _write_exe(
        bindir / "sbatch",
        f"""#!/bin/bash
n=$(( $(cat {mock}/next_id 2>/dev/null || echo 1000) + 1 ))
echo $n > {mock}/next_id
printf '%s\\n' "$*" >> {mock}/sbatch.log
env | grep -E '^(SLURM|SBATCH)_' >> {mock}/sbatch_env.log
echo "$n"
""",
    )
    _write_exe(bindir / "squeue", f"#!/bin/bash\ncat {mock}/queue 2>/dev/null\n")
    _write_exe(
        bindir / "sacct",
        f'#!/bin/bash\njid="$2"\ncat "{mock}/sacct_$jid" 2>/dev/null\n',
    )
    _write_exe(bindir / "scancel", f'#!/bin/bash\necho "$*" >> {mock}/scancel.log\n')

    env = dict(os.environ)
    env.update(
        HOME=str(home),
        PATH=f"{bindir}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH=repo + os.pathsep + env.get("PYTHONPATH", ""),
        QUANTUI_JOBS_DIR=str(tmp_path / "jobs"),
        QUANTUI_STAGING_DIR=str(home / ".quantui" / "staging"),
        QUANTUI_LOG_DIR=str(tmp_path / "logs"),
    )
    env.pop("QUANTUI_ENABLE_SLURM", None)
    env.pop("QUANTUI_BATCH_IMAGE", None)
    env.pop("QUANTUI_MAX_CONCURRENT_JOBS", None)

    install = subprocess.run(
        [
            py,
            "-m",
            "quantui.cli",
            "install-launcher",
            str(home / "bin"),
            "--image",
            str(image),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(home),
    )
    assert install.returncode == 0, install.stderr
    launcher = home / "bin" / "quantui-batch"
    (home / "water.xyz").write_text(WATER_XYZ)

    class Cluster:
        staging = home / ".quantui" / "staging"

        def run(self, *args, extra_env=None):
            e = dict(env)
            e.update(extra_env or {})
            return subprocess.run(
                [py, str(launcher), *args],
                env=e,
                capture_output=True,
                text=True,
                cwd=str(home),
            )

        def submit_water(self, *extra, extra_env=None):
            return self.run(
                "submit",
                "water.xyz",
                "--calc",
                "single_point",
                "--method",
                "RHF",
                "--basis",
                "STO-3G",
                *extra,
                extra_env=extra_env,
            )

        def set_queue(self, *rows):
            (mock / "queue").write_text("".join(r + "\n" for r in rows))

        def set_sacct(self, job_id, text):
            (mock / f"sacct_{job_id}").write_text(textwrap.dedent(text).lstrip())

        def log(self, name):
            path = mock / name
            return path.read_text() if path.exists() else ""

    c = Cluster()
    c.image = image
    return c


@needs_posix
class TestLauncherEndToEnd:
    def test_submit_prepares_in_image_and_sbatches_on_host(self, cluster):
        r = cluster.submit_water("--job-name", "h2o")
        assert r.returncode == 0, r.stderr
        assert "submitted h2o  (Slurm job 1001)" in r.stdout
        job_dir = cluster.staging / "h2o"
        assert (job_dir / "submit.slurm").is_file()
        assert (job_dir / ".quantui-batch-jobs").read_text().startswith("1001\t")
        assert (
            cluster.log("sbatch.log").strip()
            == f"--parsable {job_dir / 'submit.slurm'}"
        )
        # The worker inside the job uses the same image the launcher prepared with.
        assert str(cluster.image) in (job_dir / "submit.slurm").read_text()
        assert cluster.log("apptainer.log").strip() == str(cluster.image)

    def test_sbatch_does_not_inherit_the_callers_job_variables(self, cluster):
        # An OnDemand Shell session is itself a Slurm job.
        r = cluster.submit_water(
            extra_env={
                "SLURM_CPUS_PER_TASK": "2",
                "SLURM_JOB_ID": "77",
                "SBATCH_X": "1",
            }
        )
        assert r.returncode == 0, r.stderr
        # SBATCH_* are the user's own sbatch defaults and pass through.
        assert cluster.log("sbatch_env.log") == "SBATCH_X=1\n"

    def test_status_follows_a_job_from_queue_to_done(self, cluster):
        assert cluster.submit_water("--job-name", "h2o").returncode == 0
        script = cluster.staging / "h2o" / "submit.slurm"

        cluster.set_queue(f"1001|PENDING|{script}")
        r = cluster.run("status")
        assert "h2o" in r.stdout and "PENDING" in r.stdout
        assert "1 of 2 allowed" in r.stdout

        attempt = cluster.staging / "h2o" / "attempt-01_job1001"
        attempt.mkdir()
        (attempt / "progress.json").write_text(
            json.dumps({"stage": "running", "message": "SCF", "percent": 40.0})
        )
        cluster.set_queue(f"1001|RUNNING|{script}")
        r = cluster.run("status")
        assert "RUNNING" in r.stdout and "40%  SCF" in r.stdout

        (attempt / "result.json").write_text(json.dumps({"converged": True}))
        cluster.set_queue()
        cluster.set_sacct("1001", "1001|COMPLETED|00:01:00|\n")
        r = cluster.run("status")
        assert "DONE" in r.stdout and "attempt-01_job1001" in r.stdout

    def test_out_of_memory_gets_a_rerun_hint(self, cluster):
        assert cluster.submit_water("--job-name", "big").returncode == 0
        (cluster.staging / "big" / "attempt-01_job1001").mkdir()
        cluster.set_sacct(
            "1001",
            """
            1001|OUT_OF_MEMORY|00:10:00|
            1001.batch|OUT_OF_MEMORY|00:10:00|31.5G
            """,
        )
        r = cluster.run("status")
        assert "OUT_OF_MEMORY" in r.stdout
        assert "used 31.5G" in r.stdout
        assert "quantui-batch rerun big --mem=" in r.stdout

    def test_rerun_passes_sbatch_overrides_and_records_new_id(self, cluster):
        assert cluster.submit_water("--job-name", "big").returncode == 0
        r = cluster.run("rerun", "big", "--mem=64G")
        assert r.returncode == 0, r.stderr
        assert "Slurm job 1002" in r.stdout
        script = cluster.staging / "big" / "submit.slurm"
        assert (
            cluster.log("sbatch.log").splitlines()[-1]
            == f"--parsable --mem=64G {script}"
        )
        ids = (cluster.staging / "big" / ".quantui-batch-jobs").read_text().split()
        assert ids[0] == "1001" and "1002" in ids

    def test_rerun_refuses_a_job_that_is_still_running(self, cluster):
        assert cluster.submit_water("--job-name", "busy").returncode == 0
        cluster.set_queue(f"1001|RUNNING|{cluster.staging / 'busy' / 'submit.slurm'}")
        r = cluster.run("rerun", "busy")
        assert r.returncode == 1
        assert "still running" in r.stderr

    def test_concurrent_job_limit_counts_only_quantui_jobs(self, cluster):
        s = cluster.staging
        s.mkdir(parents=True, exist_ok=True)
        cluster.set_queue(
            f"900|RUNNING|{s / 'a' / 'submit.slurm'}",
            f"901|PENDING|{s / 'b' / 'submit.slurm'}",
            "902|RUNNING|/home/me/other-project/job.sbatch",
        )
        r = cluster.submit_water()
        assert r.returncode == 1
        assert "limit 2" in r.stderr
        assert cluster.log("sbatch.log") == ""
        assert cluster.log("apptainer.log") == ""  # refused before preparing

        r = cluster.submit_water(extra_env={"QUANTUI_MAX_CONCURRENT_JOBS": "5"})
        assert r.returncode == 0, r.stderr

    def test_bad_spin_never_reaches_sbatch(self, cluster):
        r = cluster.submit_water("--mult", "2")
        assert r.returncode != 0
        assert "do not fit" in r.stderr
        assert cluster.log("sbatch.log") == ""

    def test_prepare_runs_through_srun_when_asked(self, cluster):
        r = cluster.submit_water(
            extra_env={"QUANTUI_BATCH_PREPARE": "srun", "SLURM_JOB_ID": "77"}
        )
        assert r.returncode == 0, r.stderr
        assert "--time=00:05:00" in cluster.log("srun.log")
        # srun must make its own allocation, not a step in the caller's job.
        assert "SLURM_JOB_ID=77" not in cluster.log("srun_env.log")

    def test_prepare_step_runs_single_threaded(self, cluster):
        assert cluster.submit_water().returncode == 0
        assert "OPENBLAS_NUM_THREADS=1" in cluster.log("apptainer_env.log")

    def test_failed_prepare_explains_the_fallback(self, cluster):
        r = cluster.run("submit", "missing.xyz", "--calc", "single_point")
        assert r.returncode == 1
        assert "QUANTUI_BATCH_PREPARE=srun" in r.stderr

    def test_log_cancel_and_path(self, cluster):
        assert cluster.submit_water("--job-name", "h2o").returncode == 0
        job_dir = cluster.staging / "h2o"
        assert cluster.run("log", "h2o").stdout.startswith("h2o has not started")
        attempt = job_dir / "attempt-01_job1001"
        attempt.mkdir()
        (attempt / "live.log").write_text("SCF converged\n")
        assert "SCF converged" in cluster.run("log", "h2o").stdout

        cluster.set_queue(f"1001|RUNNING|{job_dir / 'submit.slurm'}")
        assert cluster.run("cancel", "h2o").returncode == 0
        assert cluster.log("scancel.log").strip() == "1001"
        assert cluster.run("path", "h2").stdout.strip() == str(job_dir)

    def test_unknown_job_and_help(self, cluster):
        r = cluster.run("log", "nope")
        assert r.returncode == 1 and "no job folder" in r.stderr
        r = cluster.run("help")
        assert r.returncode == 0
        assert str(cluster.image) in r.stdout


# ---------------------------------------------------------------------------
# History pickup when the app starts
# ---------------------------------------------------------------------------


class TestStartupIngest:
    def _app(self, registry):
        from types import SimpleNamespace

        return SimpleNamespace(_job_registry=registry, _slurm_active_request_id=None)

    def test_prepared_job_attempt_reaches_history_without_slurm(
        self, tmp_path, monkeypatch, water
    ):
        from unittest.mock import patch

        from quantui.app_slurm import ingest_finished_jobs_on_startup
        from quantui.backends.slurm import SlurmBackend
        from tests.slurm_ingest_helpers import patch_results_root, sample_payload

        monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
        results = patch_results_root(tmp_path, monkeypatch)
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging"
        )
        req, _ = request_from_xyz(
            water, calc_type="single_point", method="RHF", basis="STO-3G"
        )
        record = SlurmBackend(registry=registry, use_apptainer=False).prepare(
            req, job_name="h2o"
        )
        attempt = Path(record.job_dir) / "attempt-01_job4242"
        attempt.mkdir()
        (attempt / "result.json").write_text(json.dumps(sample_payload("single_point")))

        app = self._app(registry)
        with (
            patch("quantui.app_slurm.is_slurm_available", return_value=False),
            patch("quantui.app_runflow.refresh_results_browser") as refresh,
        ):
            assert len(ingest_finished_jobs_on_startup(app)) == 1
            assert ingest_finished_jobs_on_startup(app) == []  # never twice
        assert refresh.call_count == 1
        assert len(list(results.iterdir())) == 1

    def test_no_registry_means_nothing_is_created(self, tmp_path, monkeypatch):
        from quantui.app_slurm import ingest_finished_jobs_on_startup

        jobs = tmp_path / "jobs"
        monkeypatch.setenv("QUANTUI_JOBS_DIR", str(jobs))
        assert ingest_finished_jobs_on_startup(object()) == []
        assert not jobs.exists()
