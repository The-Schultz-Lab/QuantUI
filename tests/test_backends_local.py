"""quantui.backends.local.LocalBackend (audit INFRA-22: had no tests)."""

from __future__ import annotations

import pytest

from quantui import config
from quantui.backends.base import CALC_TYPES, CalculationRequest, ComputeBackend
from quantui.backends.local import LocalBackend
from quantui.backends.registry import JobRegistry


@pytest.fixture
def registry(tmp_path):
    return JobRegistry(jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging")


def _request(request_id="req-1"):
    return CalculationRequest(
        request_id=request_id,
        calc_type="single_point",
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.74]]},
    )


def test_capabilities_describe_the_local_backend(registry):
    caps = LocalBackend(registry).capabilities()
    assert caps.backend_id == LocalBackend.backend_id == "local_pyscf"
    assert caps.supported_calc_types == CALC_TYPES
    assert caps.supported_methods == tuple(config.SUPPORTED_METHODS)
    assert caps.supports_solvent and caps.supports_cancellation
    assert caps.supports_live_progress and caps.supports_history_artifacts


def test_register_local_run_records_a_running_job(registry):
    backend = LocalBackend(registry)
    req = _request()
    assert backend.register_local_run(req) == "req-1"
    rec = registry.load("req-1")
    assert rec is not None
    assert rec.status == "running"
    assert rec.backend_id == "local_pyscf"
    assert rec.calc_type == "single_point"
    assert rec.request_obj.method == "RHF"


def test_dispatch_registers_without_running_anything(registry):
    backend = LocalBackend(registry)
    assert backend.dispatch(_request("req-2")) == "req-2"
    assert registry.load("req-2").status == "running"


def test_a_request_id_is_never_registered_twice(registry):
    backend = LocalBackend(registry)
    backend.register_local_run(_request())
    with pytest.raises(FileExistsError):
        backend.register_local_run(_request())


def test_default_registry_follows_the_env_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_JOBS_DIR", str(tmp_path / "j"))
    monkeypatch.setenv("QUANTUI_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("QUANTUI_STAGING_DIR", raising=False)
    backend = LocalBackend()
    assert backend.registry.jobs_root == tmp_path / "j"
    assert str(backend.registry.staging_root).startswith(str(tmp_path / "home"))


def test_satisfies_the_compute_backend_protocol(registry):
    assert isinstance(LocalBackend(registry), ComputeBackend)
