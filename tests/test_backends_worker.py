"""
Tests for the headless batch worker (mocked — no PySCF required in CI).
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from quantui.backends.base import CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.worker import run_worker_request


@pytest.fixture
def staging(tmp_path):
    staging_dir = tmp_path / "staging" / "job1"
    staging_dir.mkdir(parents=True)
    request = CalculationRequest(
        request_id="job1",
        calc_type="single_point",
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={
            "atoms": ["H", "H"],
            "coordinates": [[0, 0, 0], [0, 0, 0.74]],
        },
    )
    (staging_dir / "request.json").write_text(json.dumps(request.to_dict()))
    registry = JobRegistry(
        jobs_root=tmp_path / "jobs",
        staging_root=tmp_path / "staging",
    )
    registry.create(request, "cluster_slurm", status="submitted")
    registry.update_status("job1", "submitted", slurm_job_id="1")
    return staging_dir


class TestWorker:
    def test_unsupported_calc_type_returns_error(self, staging):
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "not_a_real_calc"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"
        assert outcome.error["code"] == "UNSUPPORTED_CAPABILITY"

    @patch("quantui.session_calc.run_in_session")
    def test_single_point_success(self, mock_run, staging):
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        assert outcome.result_payload["energy_hartree"] == -1.12
        assert (staging / "result.json").exists()
        assert (staging / "progress.json").exists()

    @patch("quantui.session_calc.run_in_session")
    def test_single_point_writes_orbitals_sidecar(self, mock_run, staging):
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            mo_energy_hartree=np.array([-1.0, 0.5]),
            mo_occ=np.array([2.0, 0.0]),
            mo_coeff=np.eye(2),
            pyscf_mol_atom=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.74])],
            pyscf_mol_basis="STO-3G",
        )
        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        assert (staging / "orbitals.npz").exists()
        assert (staging / "orbitals_meta.json").exists()

    @patch("quantui.optimizer.optimize_geometry")
    def test_geometry_opt_success(self, mock_opt, staging):
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-1.12],
            converged=True,
            n_steps=0,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        data["options"] = {"fmax": 0.05, "max_steps": 50}
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        assert outcome.save_type == "geometry_opt"
        payload = json.loads((staging / "result.json").read_text())
        assert payload["calc_type"] == "geometry_opt"
        assert (staging / "trajectory.json").exists()

    @patch("quantui.freq_calc.run_freq_calc")
    def test_frequency_success(self, mock_freq, staging):
        mock_freq.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            frequencies_cm1=[4400.0],
            ir_intensities=[1.0],
            raman_activities=[0.2],
            zpve_hartree=0.01,
            displacements=[[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "frequency"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        payload = json.loads((staging / "result.json").read_text())
        assert payload["calc_type"] == "frequency"
        assert payload["spectra"]["ir"]["frequencies_cm1"] == [4400.0]

    @patch("quantui.freq_calc.run_freq_calc")
    @patch("quantui.optimizer.optimize_geometry")
    def test_frequency_preopt_writes_preopt_trajectory(
        self, mock_opt, mock_freq, staging
    ):
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-1.12],
            converged=True,
            n_steps=0,
        )
        mock_freq.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            frequencies_cm1=[4400.0],
            ir_intensities=[1.0],
            raman_activities=[0.2],
            zpve_hartree=0.01,
            displacements=[[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "frequency"
        data["options"] = {"preopt_before_run": True}
        data["run_context"] = {"seed_label": "2026-08-29_H2"}
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        mock_opt.assert_called_once()
        assert (staging / "preopt_trajectory.json").exists()
        assert "Seed geometry loaded from" in (staging / "live.log").read_text()

    @patch("quantui.tddft_calc.run_tddft_calc")
    def test_tddft_success(self, mock_tddft, staging):
        mock_tddft.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=5.0,
            converged=True,
            n_iterations=6,
            method="B3LYP",
            basis="STO-3G",
            formula="H2",
            excitation_energies_ev=[4.0, 5.0],
            oscillator_strengths=[0.1, 0.2],
            nstates=2,
            wavelengths_nm=lambda: [310.0, 248.0],
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "tddft"
        data["options"] = {"nstates": 2}
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        payload = json.loads((staging / "result.json").read_text())
        assert payload["calc_type"] == "tddft"
        assert len(payload["spectra"]["uv_vis"]["excitation_energies_ev"]) == 2


class TestCheckpointWiring:
    """M-CLUSTER2 CL2.8 — geometry_opt/pes_scan (+ preopt) checkpoint/resume.

    Before this, a job killed by OOM/TIMEOUT always restarted from nothing —
    confirmed by grep, zero hits for "checkpoint"/"resume" in worker.py.
    Most of pes_scan's cost is its independent scan points, so this matters
    most for a killed run at, say, point 20 of 25.
    """

    def _identity(self, *, calc_type: str):
        from quantui.checkpoint import CalcIdentity
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        return CalcIdentity.from_molecule(
            mol, calc_type=calc_type, method="RHF", basis="STO-3G"
        )

    @patch("quantui.optimizer.optimize_geometry")
    def test_geometry_opt_first_attempt_gets_a_fresh_checkpoint(
        self, mock_opt, staging
    ):
        from quantui.checkpoint import Checkpoint
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-1.12],
            converged=True,
            n_steps=0,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_opt.call_args
        assert kwargs["resume"] is False
        assert isinstance(kwargs["checkpoint"], Checkpoint)
        # The checkpoint directory lives inside this job's own staging area,
        # not the interactive app's ~/.quantui/checkpoints — self-contained,
        # never collides with another job.
        assert (staging / ".checkpoint").is_dir()

    @patch("quantui.optimizer.optimize_geometry")
    def test_geometry_opt_resumes_when_a_prior_attempt_left_progress(
        self, mock_opt, staging
    ):
        from quantui.checkpoint import Checkpoint
        from quantui.molecule import Molecule

        # Simulate a previous attempt that got killed mid-run: open the same
        # checkpoint identity the worker will compute, and leave a
        # non-empty trajectory file behind (has_progress() reads exactly
        # this) — status stays "running" (never marked complete).
        ckpt = Checkpoint(
            self._identity(calc_type="geometry_opt"), root=staging / ".checkpoint"
        )
        ckpt.begin()
        ckpt.trajectory_path.write_text("dummy trajectory data")

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-1.12],
            converged=True,
            n_steps=3,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_opt.call_args
        assert kwargs["resume"] is True
        assert "Resuming geometry optimization" in (staging / "live.log").read_text()

    @patch("quantui.pes_scan.run_pes_scan")
    def test_pes_scan_first_attempt_gets_a_fresh_checkpoint(self, mock_scan, staging):
        from quantui.checkpoint import Checkpoint
        from quantui.molecule import Molecule

        mock_scan.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            converged_all=True,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            scan_type="bond",
            atom_indices=[0, 1],
            scan_parameter_values=[0.7, 0.8],
            energies_hartree=[-1.1, -1.12],
            coordinates_list=[
                Molecule(
                    atoms=["H", "H"],
                    coordinates=[[0, 0, 0], [0, 0, 0.7]],
                    charge=0,
                    multiplicity=1,
                ),
                Molecule(
                    atoms=["H", "H"],
                    coordinates=[[0, 0, 0], [0, 0, 0.8]],
                    charge=0,
                    multiplicity=1,
                ),
            ],
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "pes_scan"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_scan.call_args
        assert kwargs["resume"] is False
        assert isinstance(kwargs["checkpoint"], Checkpoint)

    @patch("quantui.pes_scan.run_pes_scan")
    def test_pes_scan_resumes_and_reports_points_already_computed(
        self, mock_scan, staging
    ):
        import json as _json

        ckpt = self._identity(calc_type="pes_scan")
        from quantui.checkpoint import Checkpoint

        real_ckpt = Checkpoint(ckpt, root=staging / ".checkpoint")
        real_ckpt.begin()
        # completed_points() reads points.jsonl directly (CHK.3 format).
        real_ckpt.points_path.write_text(
            _json.dumps({"index": 0, "value": 0.7, "energy_hartree": -1.1}) + "\n"
        )

        from quantui.molecule import Molecule

        mock_scan.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            converged_all=True,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            scan_type="bond",
            atom_indices=[0, 1],
            scan_parameter_values=[0.7, 0.8],
            energies_hartree=[-1.1, -1.12],
            coordinates_list=[
                Molecule(
                    atoms=["H", "H"],
                    coordinates=[[0, 0, 0], [0, 0, 0.7]],
                    charge=0,
                    multiplicity=1,
                ),
                Molecule(
                    atoms=["H", "H"],
                    coordinates=[[0, 0, 0], [0, 0, 0.8]],
                    charge=0,
                    multiplicity=1,
                ),
            ],
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "pes_scan"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_scan.call_args
        assert kwargs["resume"] is True
        log_text = (staging / "live.log").read_text()
        assert "Resuming PES scan" in log_text
        assert "1 point(s) already computed" in log_text

    @patch("quantui.optimizer.optimize_geometry")
    @patch("quantui.freq_calc.run_freq_calc")
    def test_preopt_checkpoint_result_is_cached_across_attempts(
        self, mock_freq, mock_opt, staging
    ):
        """A second attempt on the same job reuses the saved preopt geometry
        instead of re-optimizing — the idempotency fix that keeps a
        downstream pes_scan/frequency/tddft checkpoint identity stable
        across independent optimizer runs of the same molecule (BLAS/OpenMP
        reduction order isn't guaranteed bit-identical run to run)."""
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[-1.12],
            converged=True,
            n_steps=3,
        )
        mock_freq.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
            frequencies_cm1=[4400.0],
            ir_intensities=[1.0],
            raman_activities=[],
            zpve_hartree=0.01,
            displacements=None,
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "frequency"
        data["options"] = {"preopt_before_run": True}
        (staging / "request.json").write_text(json.dumps(data))

        outcome1 = run_worker_request(staging / "request.json")
        assert outcome1.status == "success"
        mock_opt.assert_called_once()
        assert (staging / "preopt_geometry_frequency.json").exists()

        # Second attempt on the same staging dir (simulating a resubmission)
        # must NOT call optimize_geometry again.
        outcome2 = run_worker_request(staging / "request.json")
        assert outcome2.status == "success"
        mock_opt.assert_called_once()  # still just the one call from attempt 1
        assert "Reusing saved preopt geometry" in (staging / "live.log").read_text()
