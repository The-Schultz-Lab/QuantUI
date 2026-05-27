"""Tests for the M-EST follow-up: calibration results saved as job files.

Session 55 (2026-05-25) user request:

  > Are the calculations run as part of the calibration time estimates
  > saved to job files so users can load the results as usual?

Before this change, calibration steps only wrote to ``perf_log.jsonl``
(for the estimator) and ``calibration.json`` (for the UI summary). The
full result objects were discarded. Tier-4 in particular runs MP2 +
CCSD on H₂O/cc-pVDZ plus benzene B3LYP/6-31G* frequency — those are
real research-quality calcs and the user wanted them saved.

This file tests the new save path WITHOUT running PySCF, by:

1. Unit-testing ``save_result(..., extras={...})`` — the new kwarg that
   embeds ``calibration_run_id`` (and any other extras) in result.json.
2. Unit-testing the ``_TeeStream`` helper used to fan PySCF's
   progress_stream to both the shared calibration log and an in-memory
   buffer (so save_result has the per-calc PySCF log).
3. Unit-testing ``_save_calibration_step`` against a fake result
   object — confirms it writes a result_dir with the calibration tag.
4. Structure-grep tests that the worker passes ``calibration_run_id``
   to the helper and returns ``result_dir`` on the queue, and that
   ``BenchmarkStep`` has the new ``result_dir`` field.

All tests platform-independent. No PySCF required.
"""

from __future__ import annotations

import inspect
import io
import json
from types import SimpleNamespace

# =====================================================================
# save_result(..., extras=...) — new kwarg
# =====================================================================


class TestSaveResultExtras:
    def test_extras_merged_into_result_json(self, tmp_path):
        from quantui.results_storage import save_result

        fake_result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-75.0 * 27.211386245988,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
        )

        out = save_result(
            fake_result,
            pyscf_log="line 1\nline 2\n",
            results_dir=tmp_path,
            calc_type="single_point",
            extras={"calibration_run_id": "2026-05-25T12:00:00+00:00"},
        )
        data = json.loads((out / "result.json").read_text())
        assert data["calibration_run_id"] == "2026-05-25T12:00:00+00:00"
        # Existing fields still present.
        assert data["formula"] == "H2O"
        assert data["calc_type"] == "single_point"

    def test_extras_can_overwrite_builtin_field(self, tmp_path):
        # Documented behaviour: extras takes precedence. This is by
        # design — calibration uses it deliberately and a future caller
        # may want the same affordance.
        from quantui.results_storage import save_result

        fake_result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            converged=True,
            n_iterations=1,
        )
        out = save_result(
            fake_result,
            results_dir=tmp_path,
            extras={"formula": "OVERRIDDEN"},
        )
        data = json.loads((out / "result.json").read_text())
        assert data["formula"] == "OVERRIDDEN"

    def test_extras_none_is_no_op(self, tmp_path):
        # Existing callers that don't pass extras must keep working.
        from quantui.results_storage import save_result

        fake_result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            converged=True,
            n_iterations=1,
        )
        out = save_result(fake_result, results_dir=tmp_path)
        data = json.loads((out / "result.json").read_text())
        # No calibration_run_id when extras wasn't passed.
        assert "calibration_run_id" not in data


# =====================================================================
# _TeeStream — fan progress to two destinations
# =====================================================================


class TestTeeStream:
    def test_writes_to_all_streams(self):
        from quantui.benchmarks import _TeeStream

        a = io.StringIO()
        b = io.StringIO()
        tee = _TeeStream(a, b)
        tee.write("hello\n")
        tee.write("world\n")
        assert a.getvalue() == "hello\nworld\n"
        assert b.getvalue() == "hello\nworld\n"

    def test_returns_len_of_written(self):
        from quantui.benchmarks import _TeeStream

        tee = _TeeStream(io.StringIO())
        assert tee.write("abcde") == 5

    def test_one_broken_stream_doesnt_kill_others(self):
        from quantui.benchmarks import _TeeStream

        class _Broken:
            def write(self, _s):
                raise RuntimeError("simulated")

            def flush(self):
                raise RuntimeError("simulated")

        good = io.StringIO()
        tee = _TeeStream(_Broken(), good)
        tee.write("payload")
        tee.flush()
        # The good stream still got the data.
        assert good.getvalue() == "payload"


# =====================================================================
# _save_calibration_step — the worker's save helper
# =====================================================================


class TestSaveCalibrationStep:
    def test_single_point_creates_result_dir_with_tag(self, tmp_path, monkeypatch):
        # Redirect the default results dir to tmp_path.
        from pathlib import Path as _Path

        monkeypatch.setattr(_Path, "home", lambda: tmp_path)

        from quantui.benchmarks import _save_calibration_step

        fake_result = SimpleNamespace(
            formula="H2O",
            method="B3LYP",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-75.0 * 27.211386245988,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=12,
        )
        fake_mol = SimpleNamespace(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.7, 0.6, 0], [-0.7, 0.6, 0]],
            charge=0,
            multiplicity=1,
        )

        saved = _save_calibration_step(
            fake_result,
            calc_type="single_point",
            pyscf_log="some log",
            calibration_run_id="2026-05-25T12:00:00+00:00",
            mol=fake_mol,
        )
        assert saved is not None
        assert saved.exists()
        data = json.loads((saved / "result.json").read_text())
        assert data["calibration_run_id"] == "2026-05-25T12:00:00+00:00"
        assert data["calc_type"] == "single_point"
        assert data["formula"] == "H2O"
        # pyscf.log should be present from the worker's per-calc tee buffer.
        assert (saved / "pyscf.log").exists()
        assert "some log" in (saved / "pyscf.log").read_text()

    def test_frequency_includes_spectra(self, tmp_path, monkeypatch):
        from pathlib import Path as _Path

        monkeypatch.setattr(_Path, "home", lambda: tmp_path)

        from quantui.benchmarks import _save_calibration_step

        fake_freq = SimpleNamespace(
            formula="H2O",
            method="B3LYP",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-75.0 * 27.211386245988,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=12,
            frequencies_cm1=[1600.0, 3700.0, 3800.0],
            ir_intensities=[80.0, 5.0, 50.0],
            zpve_hartree=0.02,
            displacements=None,
        )
        fake_mol = SimpleNamespace(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.7, 0.6, 0], [-0.7, 0.6, 0]],
            charge=0,
            multiplicity=1,
        )

        saved = _save_calibration_step(
            fake_freq,
            calc_type="frequency",
            pyscf_log="",
            calibration_run_id="tier4-run-1",
            mol=fake_mol,
        )
        assert saved is not None
        data = json.loads((saved / "result.json").read_text())
        # The Analysis tab's IR + Vibrational panels read these keys.
        assert "spectra" in data
        assert "ir" in data["spectra"]
        assert data["spectra"]["ir"]["frequencies_cm1"] == [1600.0, 3700.0, 3800.0]
        assert "molecule" in data["spectra"]
        assert data["spectra"]["molecule"]["atoms"] == ["O", "H", "H"]


# =====================================================================
# Worker + BenchmarkStep structural checks
# =====================================================================


class TestWorkerStructure:
    def test_benchmark_step_has_result_dir_field(self):
        from quantui.benchmarks import BenchmarkStep

        s = BenchmarkStep(
            label="x",
            method="RHF",
            basis="STO-3G",
            n_atoms=2,
            n_electrons=2,
            status="ok",
        )
        # New field — default None.
        assert s.result_dir is None

    def test_calibration_worker_signature_accepts_run_id(self):
        from quantui.benchmarks import _calibration_worker

        sig = inspect.signature(_calibration_worker)
        assert "calibration_run_id" in sig.parameters

    def test_worker_source_calls_save_calibration_step(self):
        from quantui import benchmarks

        src = inspect.getsource(benchmarks._calibration_worker)
        assert "_save_calibration_step" in src
        # And the queue payload now carries result_dir.
        assert "result_dir" in src

    def test_save_calibration_json_includes_result_dir(self):
        # The persisted calibration.json should expose result_dir per
        # step so future tooling can find the saved results.
        from quantui import benchmarks

        src = inspect.getsource(benchmarks._save_calibration_json)
        assert '"result_dir"' in src or "'result_dir'" in src


class TestHistoryLabelMarker:
    def test_refresh_results_browser_emits_calibration_marker(self):
        from quantui import app_runflow

        src = inspect.getsource(app_runflow.refresh_results_browser)
        # The 🔧 marker is rendered when calibration_run_id is present
        # on the saved result.json.
        assert "calibration_run_id" in src
        assert "🔧" in src or "calib_marker" in src
