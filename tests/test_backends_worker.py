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


@pytest.fixture(autouse=True)
def _stub_required_single_point():
    """The worker now runs the app's required single point after every
    geometry optimization / pre-opt (audit HPC-14). Stub it so tests that
    mock only the optimizer stay PySCF-free; tests that patch
    ``run_in_session`` themselves override this."""
    with patch(
        "quantui.session_calc.run_in_session",
        return_value=SimpleNamespace(converged=True, energy_hartree=-1.12),
    ):
        yield


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

    @pytest.mark.parametrize("calc_type", ["nmr", "pes_scan"])
    def test_solvent_on_unsupported_calc_type_returns_error(self, staging, calc_type):
        """AUDIT F11 — run_nmr_calc/run_pes_scan don't accept a solvent; a
        solvent set for one of these calc_types must fail the request rather
        than silently run gas-phase. (frequency/tddft gained PCM, DEC-023.)
        """
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = calc_type
        data["solvent"] = "water"
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"
        assert outcome.error["code"] == "UNSUPPORTED_CAPABILITY"
        assert "solvent" in outcome.error["user_message"].lower()

    @patch("quantui.session_calc.run_in_session")
    def test_solvent_on_single_point_is_accepted(self, mock_run, staging):
        """Sanity check: the calc_types that DO support solvent must not be
        rejected by the new guard."""
        data = json.loads((staging / "request.json").read_text())
        data["solvent"] = "water"
        (staging / "request.json").write_text(json.dumps(data))
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
        assert mock_run.call_args.kwargs["solvent"] == "water"

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

    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_solvent_on_geometry_opt_runs_required_final_single_point(
        self, mock_opt, mock_run, staging
    ):
        """Code review (audit follow-up) — app_runflow.py's UI enables the
        solvent checkbox for Geometry Opt (with a label documenting the
        gas-phase-optimization + solvated-final-single-point
        approximation), so a SLURM submission with the same settings must
        actually apply it rather than hard-rejecting the whole job."""
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
            energies_hartree=[-1.10],
            converged=True,
            n_steps=3,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        data["solvent"] = "water"
        data["options"] = {"fmax": 0.05, "max_steps": 50}
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        assert mock_run.call_args.kwargs["solvent"] == "water"
        # The optimization itself is solvated now, not just the final SP.
        assert mock_opt.call_args.kwargs["solvent"] == "water"
        # The solvated single-point energy replaces the optimizer's
        # last-step energy, mirroring app.py's interactive
        # _run_required_final_single_point handling.
        payload = json.loads((staging / "result.json").read_text())
        assert payload["energy_hartree"] == -1.12

    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_solvent_on_geometry_opt_final_single_point_must_converge(
        self, mock_opt, mock_run, staging
    ):
        """An unconverged required solvated single point must fail the job
        rather than silently reporting the gas-phase optimizer's result."""
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
            energies_hartree=[-1.10],
            converged=True,
            n_steps=3,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=False,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "geometry_opt"
        data["solvent"] = "water"
        data["options"] = {"fmax": 0.05, "max_steps": 50}
        (staging / "request.json").write_text(json.dumps(data))

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"

    def test_unknown_solvent_fails_before_running(self, staging):
        data = json.loads((staging / "request.json").read_text())
        data["solvent"] = "seawater"
        (staging / "request.json").write_text(json.dumps(data))
        with patch("quantui.session_calc.run_in_session") as mock_run:
            outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"
        assert outcome.error["code"] == "VALIDATION_ERROR"
        assert "seawater" in outcome.error["user_message"]
        mock_run.assert_not_called()

    @pytest.mark.parametrize(
        "calc_type, target",
        [
            ("frequency", "quantui.freq_calc.run_freq_calc"),
            ("tddft", "quantui.tddft_calc.run_tddft_calc"),
        ],
    )
    def test_solvent_reaches_frequency_and_tddft(self, staging, calc_type, target):
        """DEC-023 / B2.2 — PCM frequency and TD-DFT run in batch too."""
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = calc_type
        data["solvent"] = "Water"
        (staging / "request.json").write_text(json.dumps(data))
        with patch(target, side_effect=RuntimeError("stop after dispatch")) as m:
            outcome = run_worker_request(staging / "request.json")
        assert m.call_args.kwargs["solvent"] == "Water"
        # The stub raised, so the job errors — but not as "unsupported".
        assert outcome.error["code"] != "UNSUPPORTED_CAPABILITY"

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
        assert "Seed geometry loaded from" in (staging / "live.log").read_text(
            encoding="utf-8"
        )
        # B2.5 — frequencies need the tighter pre-opt threshold.
        from quantui.optimizer import FREQ_PREOPT_FMAX

        assert mock_opt.call_args.kwargs["fmax"] == FREQ_PREOPT_FMAX == 0.01
        assert "Force threshold 0.01 eV/Å" in (staging / "live.log").read_text(
            encoding="utf-8"
        )

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

    def _identity(self, *, calc_type: str, extra: tuple = ()):
        from quantui.checkpoint import CalcIdentity
        from quantui.molecule import Molecule

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        return CalcIdentity.from_molecule(
            mol, calc_type=calc_type, method="RHF", basis="STO-3G", extra=extra
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
        assert "Resuming geometry optimization" in (staging / "live.log").read_text(
            encoding="utf-8"
        )

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

        # AUDIT F10 — the request sets no scan_type/atom_indices options, so
        # _run_pes_scan defaults to scan_type="bond", atom_indices=[0, 1];
        # the checkpoint identity here must match that exactly, since
        # resume_key now includes them.
        ckpt = self._identity(calc_type="pes_scan", extra=("bond", "0", "1"))
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
        log_text = (staging / "live.log").read_text(encoding="utf-8")
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
        assert "Reusing saved preopt geometry" in (staging / "live.log").read_text(
            encoding="utf-8"
        )

    @patch("quantui.freq_calc.run_freq_calc")
    def test_frequency_first_attempt_gets_a_fresh_checkpoint(self, mock_freq, staging):
        """M-CHECKPOINT CHK.4 — the calc type roadmap 34's real production
        cost data was about: a killed frequency job used to discard the
        entire 6N-displacement Hessian on every resubmission."""
        from quantui.checkpoint import Checkpoint

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
        _args, kwargs = mock_freq.call_args
        assert kwargs["resume"] is False
        assert isinstance(kwargs["checkpoint"], Checkpoint)
        assert (staging / ".checkpoint").is_dir()

    @patch("quantui.freq_calc.run_freq_calc")
    def test_frequency_resumes_when_a_prior_attempt_left_progress(
        self, mock_freq, staging
    ):
        """A prior attempt that banked some displacements (see
        checkpoint.py's ``items/freq_displacements/`` — CHK.4.1) before
        getting killed must resume, not restart the whole Hessian."""
        from quantui.checkpoint import Checkpoint

        # Simulate a previous attempt that got killed mid-run: open the same
        # checkpoint identity the worker will compute, and bank one
        # displacement — status stays "running" (never marked complete).
        ckpt = Checkpoint(
            self._identity(calc_type="frequency"), root=staging / ".checkpoint"
        )
        ckpt.begin()
        ckpt.mark_item_done("freq_displacements", "d000_x_+", {"dipole": [0, 0, 0]})

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
        _args, kwargs = mock_freq.call_args
        assert kwargs["resume"] is True
        assert "Resuming frequency analysis" in (staging / "live.log").read_text(
            encoding="utf-8"
        )

    # ── reorganization_energy ──────────────────────────────────────────────
    # The library has taken checkpoint/resume since CHK.7 (PR #55), but the
    # batch worker never passed them, so a TIMEOUT threw away all 2-3
    # optimizations. The whole-run checkpoint holds nothing itself; progress
    # lives in nested per-leg checkpoints (Checkpoint.sub).

    @staticmethod
    def _reorg_result():
        from quantui.molecule import Molecule
        from quantui.reorganization_energy import ReorganizationEnergyResult

        mol = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )
        return ReorganizationEnergyResult(
            formula="H2",
            method="RHF",
            basis="STO-3G",
            mode="both",
            molecule=mol,
            neutral_charge=0,
            neutral_multiplicity=1,
            neutral_energy_hartree=-1.12,
            channels=[],
        )

    @staticmethod
    def _set_reorg_request(staging, mode: str = "both") -> None:
        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = "reorganization_energy"
        data["options"] = {"mode": mode}
        (staging / "request.json").write_text(json.dumps(data))

    def _reorg_ckpt_with_leg_progress(self, staging, *, mode: str):
        """A prior attempt killed mid-way through its neutral leg."""
        from quantui.checkpoint import Checkpoint

        ckpt = Checkpoint(
            self._identity(
                calc_type="reorganization_energy",
                extra=(f"mode={mode}", "solvent="),
            ),
            root=staging / ".checkpoint",
        )
        ckpt.begin()
        leg = ckpt.sub(
            "neutral_opt",
            charge=0,
            multiplicity=1,
            coords=[[0, 0, 0], [0, 0, 0.74]],
        )
        leg.begin()
        leg.trajectory_path.write_text("dummy trajectory data")
        return ckpt

    @patch("quantui.reorganization_energy.run_reorganization_energy")
    def test_reorg_first_attempt_gets_a_fresh_checkpoint(self, mock_reorg, staging):
        from quantui.checkpoint import Checkpoint

        mock_reorg.return_value = self._reorg_result()
        self._set_reorg_request(staging)

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_reorg.call_args
        assert kwargs["resume"] is False
        assert isinstance(kwargs["checkpoint"], Checkpoint)
        assert kwargs["mode"] == "both"
        assert (staging / ".checkpoint").is_dir()

    @patch("quantui.reorganization_energy.run_reorganization_energy")
    def test_reorg_resumes_when_a_prior_attempt_left_leg_progress(
        self, mock_reorg, staging
    ):
        self._reorg_ckpt_with_leg_progress(staging, mode="both")
        mock_reorg.return_value = self._reorg_result()
        self._set_reorg_request(staging, mode="both")

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_reorg.call_args
        assert kwargs["resume"] is True
        assert "Resuming reorganization energy" in (staging / "live.log").read_text(
            encoding="utf-8"
        )

    @patch("quantui.reorganization_energy.run_reorganization_energy")
    def test_reorg_does_not_resume_a_different_mode(self, mock_reorg, staging):
        """Mode is part of the identity: a "hole" run's legs are not this
        "both" run's checkpoint."""
        self._reorg_ckpt_with_leg_progress(staging, mode="hole")
        mock_reorg.return_value = self._reorg_result()
        self._set_reorg_request(staging, mode="both")

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        _args, kwargs = mock_reorg.call_args
        assert kwargs["resume"] is False


class TestCheckpointsOutliveAttempts:
    """Since M-JOBDIRS each run writes into its own attempt folder; the
    checkpoint and the cached pre-opt geometry must live in the job folder
    or a resubmission (new attempt folder) can never resume."""

    def _job(self, tmp_path):
        job = tmp_path / "H2_opt"
        job.mkdir()
        (job / "request.json").write_text("{}")
        a1 = job / "attempt-01_job101"
        a2 = job / "attempt-02_job102"
        a1.mkdir()
        a2.mkdir()
        return job, a1, a2

    def test_state_dir_is_the_job_folder_for_attempts(self, tmp_path):
        from quantui.backends.worker import _job_state_dir

        job, a1, _a2 = self._job(tmp_path)
        assert _job_state_dir(a1) == job
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        assert _job_state_dir(legacy) == legacy

    def test_second_attempt_resumes_the_first_attempts_checkpoint(self, tmp_path):
        import io

        from quantui.backends.worker import _begin_worker_checkpoint
        from quantui.molecule import Molecule

        job, a1, a2 = self._job(tmp_path)
        mol = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
        kw = dict(calc_type="geometry_opt", method="RHF", basis="STO-3G")
        ckpt, resumable = _begin_worker_checkpoint(
            mol, staging_dir=a1, log_stream=io.StringIO(), **kw
        )
        assert not resumable
        ckpt.trajectory_path.write_text("progress from the killed attempt")
        _ckpt2, resumable2 = _begin_worker_checkpoint(
            mol, staging_dir=a2, log_stream=io.StringIO(), **kw
        )
        assert resumable2
        assert (job / ".checkpoint").is_dir()
        assert not (a1 / ".checkpoint").exists()


class TestInteractiveParity:
    """Batch worker vs interactive run (audit HPC-14)."""

    @staticmethod
    def _h2():
        from quantui.molecule import Molecule

        return Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            multiplicity=1,
        )

    def _opt(self, energy=-1.10):
        mol = self._h2()
        return SimpleNamespace(
            molecule=mol,
            trajectory=[mol],
            energies_hartree=[energy],
            converged=True,
            n_steps=3,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )

    @staticmethod
    def _set(staging, **fields):
        data = json.loads((staging / "request.json").read_text())
        data.update(fields)
        (staging / "request.json").write_text(json.dumps(data))

    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_gas_phase_geometry_opt_runs_required_final_single_point(
        self, mock_opt, mock_run, staging
    ):
        mock_opt.return_value = self._opt()
        mock_run.return_value = SimpleNamespace(
            converged=True,
            energy_hartree=-1.13,
            atom_symbols=["H", "H"],
            mulliken_charges=[0.0, 0.0],
            npa_charges=[0.01, -0.01],
            bond_orders=[{"i": 0, "j": 1, "order": 1.0}],
            dipole_moment_debye=0.0,
            dipole_vector_debye=[0.0, 0.0, 0.0],
        )
        self._set(staging, calc_type="geometry_opt", options={"fmax": 0.05})

        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "success"
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["solvent"] is None
        payload = json.loads((staging / "result.json").read_text())
        assert payload["energy_hartree"] == -1.13
        assert payload["npa_charges"] == [0.01, -0.01]
        assert payload["bond_orders"] == [{"i": 0, "j": 1, "order": 1.0}]
        assert "Required single-point (after geometry optimisation)" in (
            staging / "live.log"
        ).read_text(encoding="utf-8")

    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_gas_phase_geometry_opt_unconverged_final_sp_fails(
        self, mock_opt, mock_run, staging
    ):
        mock_opt.return_value = self._opt()
        mock_run.return_value = SimpleNamespace(converged=False, energy_hartree=-1.0)
        self._set(staging, calc_type="geometry_opt")
        assert run_worker_request(staging / "request.json").status == "error"

    @patch("quantui.tddft_calc.run_tddft_calc")
    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_preopt_required_sp_failure_falls_back_to_input_geometry(
        self, mock_opt, mock_run, mock_tddft, staging
    ):
        moved = self._opt()
        moved.molecule = type(moved.molecule)(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.80]],
            charge=0,
            multiplicity=1,
        )
        mock_opt.return_value = moved
        mock_run.return_value = SimpleNamespace(converged=False, energy_hartree=-1.0)
        mock_tddft.side_effect = RuntimeError("stop here")
        self._set(staging, calc_type="tddft", options={"preopt_before_run": True})

        run_worker_request(staging / "request.json")
        mock_run.assert_called_once()  # the required SP after the pre-opt
        # Like the app: the pre-opt is abandoned, the calc gets the input
        # geometry, and nothing is cached for a resubmission to reuse.
        used = mock_tddft.call_args.kwargs["molecule"]
        assert [list(map(float, c)) for c in used.coordinates][1] == [0, 0, 0.74]
        assert not list(staging.glob("**/preopt_geometry_*.json"))
        log = (staging / "live.log").read_text(encoding="utf-8")
        assert "Geometry optimization failed" in log

    @patch("quantui.session_calc.run_in_session")
    @patch("quantui.optimizer.optimize_geometry")
    def test_single_point_preopt_has_no_extra_single_point(
        self, mock_opt, mock_run, staging
    ):
        mock_opt.return_value = self._opt()
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        self._set(staging, options={"preopt_before_run": True})
        assert run_worker_request(staging / "request.json").status == "success"
        mock_run.assert_called_once()  # the single point itself

    @patch("quantui.session_calc.run_in_session")
    def test_failure_writes_traceback_to_live_log(self, mock_run, staging):
        mock_run.side_effect = ValueError("boom in the SCF")
        outcome = run_worker_request(staging / "request.json")
        assert outcome.status == "error"
        log = (staging / "live.log").read_text(encoding="utf-8")
        assert "Traceback (most recent call last)" in log
        assert "ValueError: boom in the SCF" in log

    @patch("quantui.session_calc.run_in_session")
    def test_result_json_keeps_elapsed_and_warnings(self, mock_run, staging):
        mock_run.return_value = SimpleNamespace(
            energy_hartree=-1.12,
            homo_lumo_gap_ev=10.0,
            converged=False,
            n_iterations=50,
            method="RHF",
            basis="STO-3G",
            formula="H2",
        )
        outcome = run_worker_request(staging / "request.json")
        payload = json.loads((staging / "result.json").read_text())
        assert payload["elapsed_s"] == outcome.metrics["elapsed_s"]
        assert isinstance(payload["elapsed_s"], float)
        assert payload["warnings"] == ["Calculation did not fully converge"]
        assert outcome.warnings == payload["warnings"]
