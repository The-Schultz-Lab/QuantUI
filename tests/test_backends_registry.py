"""
Tests for job registry and SLURM resource heuristics.
"""

import json

import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.slurm_utils import estimate_slurm_resources, parse_slurm_job_id


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
        record = registry.create(req, "cluster_slurm", resources={"cores": 4})
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


class TestSlurmUtils:
    def test_parse_slurm_job_id(self):
        assert parse_slurm_job_id("Submitted batch job 123456") == "123456"
        assert parse_slurm_job_id("garbage") is None

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
