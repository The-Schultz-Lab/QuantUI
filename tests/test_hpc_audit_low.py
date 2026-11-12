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


# ---------------------------------------------------------------------------
# HPC-11: batch input is checked at submit, in the package and the launcher
# ---------------------------------------------------------------------------


def _write_json(path, **over):
    import json

    data = {
        "request_id": "req1",
        "calc_type": "single_point",
        "method": "RHF",
        "basis": "STO-3G",
        "charge": 0,
        "multiplicity": 1,
        "molecule": {"atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.74]]},
    }
    data.update(over)
    path.write_text(json.dumps(data))
    return path


def _launcher_args(launcher, argv):
    return launcher.submit_parser("t").parse_args(argv)


class TestBatchInputChecks:
    @pytest.mark.parametrize(
        "over, match",
        [
            ({"request_id": "x/y"}, "request_id"),
            ({"request_id": "a..b"}, "request_id"),
            ({"multiplicity": 2}, "multiplicity 2"),
            ({"calc_type": "nope"}, "calc_type"),
            (
                {"molecule": {"atoms": ["H", "H"], "coords": [[0, 0, 0]]}},
                "one \\[x, y, z\\] per atom",
            ),
            (
                {"molecule": {"atoms": ["H", "Xx"], "coords": [[0, 0, 0]] * 2}},
                "unknown element",
            ),
            (
                {"molecule": {"atoms": ["H", "H"], "coords": [[0, 0], [0, 0, 1]]}},
                "three finite numbers",
            ),
            (
                {
                    "molecule": {
                        "atoms": ["H", "H"],
                        "coords": [[0, 0, 0], [0, 0, 0.74]],
                        "multiplicity": 2,
                    }
                },
                "multiplicity 2",
            ),
        ],
    )
    def test_json_request_is_checked_in_both(self, tmp_path, launcher, over, match):
        from quantui.backends.batch_input import BatchInputError, load_request

        path = _write_json(tmp_path / "r.json", **over)
        with pytest.raises(BatchInputError, match=match):
            load_request(path)
        args = _launcher_args(launcher, [str(path)])
        with pytest.raises(launcher.InputError, match=match):
            launcher.build_request(path, args, {})

    def test_valid_json_still_loads(self, tmp_path, launcher):
        from quantui.backends.batch_input import load_request

        path = _write_json(tmp_path / "r.json")
        request, warnings = load_request(path)
        assert warnings == [] and request.request_id == "req1"
        ours, _ = launcher.build_request(
            path, _launcher_args(launcher, [str(path)]), {}
        )
        assert ours["request_id"] == "req1"

    @pytest.mark.parametrize(
        "raw, expected", [("no", False), ("Off", False), ("yes", True), ("1", True)]
    )
    def test_string_booleans(self, tmp_path, launcher, raw, expected):
        from quantui.backends.batch_input import load_request

        path = _write_json(tmp_path / "r.json")
        request, _ = load_request(path, options={"scf_rescue": raw})
        assert request.options["scf_rescue"] is expected
        req = {"calc_type": "single_point", "options": {"scf_rescue": raw}}
        launcher.check_options(req)
        assert req["options"]["scf_rescue"] is expected

    def test_bad_boolean_is_refused(self, tmp_path, launcher):
        from quantui.backends.batch_input import BatchInputError, load_request

        path = _write_json(tmp_path / "r.json")
        with pytest.raises(BatchInputError, match="true or false"):
            load_request(path, options={"scf_rescue": "maybe"})
        with pytest.raises(launcher.InputError, match="true or false"):
            launcher.check_options(
                {"calc_type": "single_point", "options": {"scf_rescue": "maybe"}}
            )

    def test_ignored_options_warn(self, water, launcher):
        from quantui.backends.batch_input import load_request

        _, warnings = load_request(
            water, calc_type="tddft", options={"nstates": 5, "nstate": 5}
        )
        assert len(warnings) == 1 and "nstate " in warnings[0] + " "
        assert "are ignored by a tddft job" in warnings[0]
        ours = launcher.check_options(
            {"calc_type": "tddft", "options": {"nstates": 5, "nstate": 5}}
        )
        assert ours == warnings

    def test_option_keys_match_launcher(self, launcher):
        from quantui.backends.base import CALC_TYPES
        from quantui.backends.batch_input import option_keys_for

        for calc in CALC_TYPES:
            assert launcher.SITE["calc_option_keys"][calc] == sorted(
                option_keys_for(calc)
            )

    def test_basis_is_normalised_or_warned(self, water, launcher):
        from quantui.backends.batch_input import load_request

        request, warnings = load_request(
            water, calc_type="single_point", basis="def2-svp"
        )
        assert request.basis == "def2-SVP" and warnings == []
        request, warnings = load_request(
            water, calc_type="single_point", basis="6-31G(d)"
        )
        assert request.basis == "6-31G(d)" and warnings == []
        _, warnings = load_request(water, calc_type="single_point", basis="not-a-basis")
        assert len(warnings) == 1 and "not-a-basis" in warnings[0]
        req = {"basis": "not-a-basis"}
        assert launcher.check_basis(req) == warnings
        req = {"basis": "def2-svp"}
        assert launcher.check_basis(req) == [] and req["basis"] == "def2-SVP"

    def test_zero_resources_are_refused_not_estimated(self, registry, launcher, water):
        from quantui.backends.slurm import SlurmBackend

        backend = SlurmBackend(registry, use_apptainer=False)
        with pytest.raises(SecurityError, match="cores=0"):
            backend.prepare(_request(), cores=0)
        with pytest.raises(SecurityError, match="memory_gb=0"):
            backend.prepare(_request("low002"), memory_gb=0)
        args = _launcher_args(launcher, [str(water), "--calc", "single_point"])
        req, _ = launcher.build_request(water, args, {})
        args.cores = 0
        with pytest.raises(launcher.InputError, match="cores=0"):
            launcher.resolve_resources(req, args)

    def test_dotdot_file_stem_is_accepted(self, tmp_path, roots, launcher):
        from quantui.backends.batch_input import load_request

        path = tmp_path / "a..b.xyz"
        path.write_text(WATER_XYZ)
        request, _ = load_request(path, calc_type="single_point")
        assert ".." not in request.request_id
        rc, out, err = _capture(
            ["submit", str(path), "--calc", "single_point", "--prepare-only"]
        )
        assert rc == 0, err
        assert launcher.file_label(path) == "a.b"

    def test_preopt_help_lists_all_preopt_calc_types(self):
        from quantui.backends.batch_input import PREOPT_CALC_TYPES

        text = Path(cli.__file__).read_text(encoding="utf-8")
        start = text.index('"--preopt"')
        snippet = text[start : start + 300]
        for calc in PREOPT_CALC_TYPES:
            assert calc in snippet
