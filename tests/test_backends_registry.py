"""
Tests for job registry and SLURM resource heuristics.
"""

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm_utils import (
    SlurmJobAccounting,
    estimate_slurm_resources,
    parse_sacct_accounting,
    parse_sacct_states,
    parse_slurm_job_id,
)


@pytest.fixture
def registry(tmp_path):
    jobs = tmp_path / "jobs"
    staging = tmp_path / "staging"
    return JobRegistry(jobs_root=jobs, staging_root=staging)


def _sample_request(request_id: str = "req001") -> CalculationRequest:
    return CalculationRequest(
        request_id=request_id,
        calc_type="single_point",
        method="RHF",
        basis="6-31G",
        charge=0,
        multiplicity=1,
        molecule={
            "atoms": ["O", "H", "H"],
            "coords": [[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]],
            "label": "water",
        },
    )


class TestJobRegistry:
    def test_create_and_load(self, registry):
        req = _sample_request()
        registry.create(req, "cluster_slurm", resources={"cores": 4})
        loaded = registry.load("req001")
        assert loaded is not None
        assert loaded.backend_id == "cluster_slurm"
        assert loaded.staging_path.exists()
        assert (loaded.staging_path / "..").exists()

    def test_list_active_excludes_terminal(self, registry):
        req = _sample_request("a")
        registry.create(req, "cluster_slurm")
        registry.update_status("a", "success")
        req2 = _sample_request("b")
        registry.create(req2, "cluster_slurm", status="running")
        active = registry.list_active()
        assert len(active) == 1
        assert active[0].request_id == "b"

    def test_update_status_persists(self, registry):
        registry.create(_sample_request(), "cluster_slurm")
        registry.update_status("req001", "submitted", slurm_job_id="12345")
        loaded = registry.load("req001")
        assert loaded.slurm_job_id == "12345"
        assert loaded.status == "submitted"

    def test_slurm_submit_meta_roundtrip(self, registry):
        assert registry.seconds_since_last_slurm_submit() is None
        registry.record_slurm_submit()
        since = registry.seconds_since_last_slurm_submit()
        assert since is not None
        assert 0 <= since < 2


class TestSlurmUtils:
    def test_parse_slurm_job_id(self):
        assert parse_slurm_job_id("Submitted batch job 123456") == "123456"
        assert parse_slurm_job_id("garbage") is None

    def test_parse_sacct_states(self):
        output = (
            "123456|COMPLETED|0:0|00:01:23\n"
            "123456.batch|COMPLETED|0:0|00:01:23\n"
            "789|CANCELLED|0:15|00:00:04\n"
        )
        states = parse_sacct_states(output)
        assert states == {"123456": "COMPLETED", "789": "CANCELLED"}

    def test_parse_sacct_accounting(self):
        output = "123456|FAILED|1:0|00:00:42\n"
        rows = parse_sacct_accounting(output)
        assert rows["123456"] == SlurmJobAccounting(
            state="FAILED", exit_code="1:0", elapsed="00:00:42"
        )

    def test_estimate_resources_scales_with_atoms(self):
        small = _sample_request()
        large = CalculationRequest(
            request_id="big",
            calc_type="frequency",
            method="B3LYP",
            basis="cc-pVTZ",
            charge=0,
            multiplicity=1,
            molecule={
                "atoms": ["C"] * 25,
                "coords": [[float(i), 0.0, 0.0] for i in range(25)],
            },
        )
        small_est = estimate_slurm_resources(small)
        large_est = estimate_slurm_resources(large)
        assert large_est["memory_gb"] >= small_est["memory_gb"]
        assert large_est["cores"] >= small_est["cores"]
