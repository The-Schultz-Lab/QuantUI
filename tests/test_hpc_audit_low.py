"""
Regression tests for the Low HPC findings of the 2026-11-10 audit
(HPC-9 … HPC-22): batch-script injection via the partition, batch input
validation, client-side limits, HTML escaping, the mock Slurm, the
``quantui-batch`` launcher, polling and job-folder permissions.
"""

import io
import sys
from pathlib import Path

import pytest

from quantui import cli
from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.security import SecurityError

WATER_XYZ = """3
water
O   0.0000   0.0000   0.1173
H   0.0000   0.7572  -0.4692
H   0.0000  -0.7572  -0.4692
"""


def _request(rid: str = "low001", calc_type: str = "single_point"):
    return CalculationRequest(
        request_id=rid,
        calc_type=calc_type,
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.74]]},
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
def registry(tmp_path):
    return JobRegistry(jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("QUANTUI_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.delenv("QUANTUI_ENABLE_SLURM", raising=False)
    for var in ("QUANTUI_SLURM_ACCOUNT", "QUANTUI_SLURM_QOS", "QUANTUI_SLURM_GRES"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def water(tmp_path):
    path = tmp_path / "water.xyz"
    path.write_text(WATER_XYZ)
    return path


def _import_launcher(path):
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("quantui_batch_low_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def launcher(tmp_path, roots):
    from quantui.batch_launcher import render_launcher

    image = tmp_path / "quantui.sif"
    image.write_text("fake image")
    path = tmp_path / "quantui_batch_rendered.py"
    path.write_text(
        render_launcher(str(image), sys.executable, "test"), encoding="utf-8"
    )
    mod = _import_launcher(path)
    mod.TEST_IMAGE = str(image)
    return mod


# ---------------------------------------------------------------------------
# HPC-9: the partition is validated like account/QOS/GRES
# ---------------------------------------------------------------------------

_INJECTED = "common\n#SBATCH --account=other\ntouch /tmp/pwned"


class TestPartitionValidation:
    def test_validate_partition(self):
        from quantui.backends.cluster_config import validate_partition

        assert validate_partition(" gpu ") == "gpu"
        for bad in (_INJECTED, "a b", "", "x;y", "$(id)"):
            with pytest.raises(ValueError):
                validate_partition(bad)

    def test_backend_rejects_injected_partition_before_job_dir(self, registry):
        from quantui.backends.slurm import SlurmBackend

        backend = SlurmBackend(registry, partition=_INJECTED, use_apptainer=False)
        with pytest.raises(SecurityError, match="partition"):
            backend.prepare(_request())
        for root in (registry.staging_root, registry.jobs_root):
            assert not root.exists() or not any(root.iterdir())

    def test_backend_accepts_plain_partition(self, registry):
        from quantui.backends.slurm import SlurmBackend

        backend = SlurmBackend(registry, partition="gpu", use_apptainer=False)
        record = backend.prepare(_request())
        script = (Path(record.job_dir) / "submit.slurm").read_text()
        assert "#SBATCH --partition=gpu\n" in script

    def test_cli_prepare_only_rejects_bad_partition(self, roots, water):
        rc, out, err = _capture(
            ["submit", str(water), "--calc", "single_point", "--prepare-only"]
            + ["--partition", "common\n#SBATCH --qos=x"]
        )
        assert rc == 1
        assert "partition" in err
        assert out == ""

    def test_launcher_rejects_env_partition(self, launcher, monkeypatch):
        monkeypatch.setenv("QUANTUI_SLURM_PARTITION", _INJECTED)
        with pytest.raises(launcher.InputError, match="partition"):
            launcher.partition()
        monkeypatch.setenv("QUANTUI_SLURM_PARTITION", "gpu")
        assert launcher.partition() == "gpu"
