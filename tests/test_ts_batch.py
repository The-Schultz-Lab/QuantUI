"""Transition State as a batch job (M-TS TS.5).

One real job (HCN → HNC at RHF/STO-3G, a few seconds) through the worker,
History ingest, ``--from`` chaining and the results summary; the resume
paths with the search mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.backends.base import CALC_TYPES, CalculationRequest
from quantui.backends.registry import JobRegistry
from quantui.backends.worker import (
    _TS_PROGRESS,
    _TS_SEARCH_DONE,
    _ts_state_key,
    run_worker_request,
)
from quantui.molecule import Molecule
from quantui.ts_search import TSResult

HCN_ATOMS = ["C", "N", "H"]
HCN_GUESS = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.18], [1.05, 0.0, 0.55]]


def _job(tmp_path, *, method="RHF", options=None):
    """A job folder with request.json and one attempt dir, as SLURM runs it."""
    job = tmp_path / "staging" / "hcn_ts"
    attempt = job / "attempt-01_job101"
    attempt.mkdir(parents=True)
    request = CalculationRequest(
        request_id="hcn_ts",
        calc_type="transition_state",
        method=method,
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": HCN_ATOMS, "coords": HCN_GUESS},
        options=dict(options or {}),
    )
    (job / "request.json").write_text(json.dumps(request.to_dict()))
    return job, attempt, request


def test_it_is_a_batch_calc_type():
    from quantui.backends import slurm_utils
    from quantui.backends.batch_input import SOLVENT_CALC_TYPES

    assert "transition_state" in CALC_TYPES
    assert "transition_state" in SOLVENT_CALC_TYPES
    assert slurm_utils._CALC_TYPE_TAGS["transition_state"] == "ts"


class TestSubmitChecks:
    def test_post_hf_is_refused_before_queueing(self, tmp_path):
        from quantui.backends.batch_input import BatchInputError, load_request

        xyz = tmp_path / "hcn.xyz"
        xyz.write_text(
            "3\nguess\n"
            + "".join(
                f"{a} {x} {y} {z}\n" for a, (x, y, z) in zip(HCN_ATOMS, HCN_GUESS)
            )
        )
        with pytest.raises(BatchInputError, match="analytic gradients"):
            load_request(xyz, calc_type="transition_state", method="MP2")
        request, _ = load_request(
            xyz, calc_type="transition_state", method="B3LYP", solvent="water"
        )
        assert request.solvent == "Water"


@pytest.mark.slow
class TestRealJob:
    def test_worker_ingest_chain_and_summary(self, tmp_path, monkeypatch):
        pytest.importorskip("sella")
        pytest.importorskip("pyscf")
        from quantui import load_result
        from quantui.backends.batch_chain import final_geometry
        from quantui.backends.result_summary import summarize
        from quantui.backends.slurm_ingest import (
            completion_summary_html,
            ingest_staging_success,
        )

        job, attempt, request = _job(tmp_path)
        out = run_worker_request(job / "request.json", attempt_dir=attempt)
        assert out.status == "success", out.error
        payload = json.loads((attempt / "result.json").read_text())
        assert payload["calc_type"] == "transition_state"
        ts = payload["spectra"]["transition_state"]
        assert ts["n_imaginary"] == 1 and ts["search_converged"]
        assert ts["verdict"].startswith("Transition state")
        assert payload["spectra"]["ir"]["frequencies_cm1"][0] < -800
        assert payload["n_steps"] >= 1
        # The saddle point, not the guess, for History and --from.
        assert payload["final_molecule"]["coords"][2] != pytest.approx(HCN_GUESS[2])
        names = payload.get("artifact_names") or {}  # descriptive names (JD.7)
        for name in ("trajectory.json", "orbitals.npz", "result.molden"):
            assert (attempt / names.get(name, name)).exists(), name
        # The finished search is kept for a resubmission; progress is gone.
        assert (job / _TS_SEARCH_DONE).is_file()
        assert not (job / _TS_PROGRESS).exists()

        geo = final_geometry(job)
        assert geo["coords"] == payload["final_molecule"]["coords"]

        text = "\n".join(summarize(payload, job, attempt))
        assert "a transition state has exactly one" in text
        assert "verdict:          Transition state" in text

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path / "results"))
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging"
        )
        record = registry.create(request, "cluster_slurm", status="success")
        saved = ingest_staging_success(record, "", attempt_dir=attempt)
        data = load_result(saved)
        assert data["calc_type"] == "transition_state"
        assert data["spectra"]["transition_state"]["n_imaginary"] == 1
        assert data["geometry"]["coordinates"][2] != pytest.approx(HCN_GUESS[2])
        assert (saved / "trajectory.json").exists()
        html = completion_summary_html(saved, payload)
        assert "Search steps" in html and "Transition state" in html


def _fake_search(n_new):
    """A run_ts_search stand-in that reports *n_new* steps after the start."""

    def _run(molecule, method, basis, **kw):
        frames = [molecule]
        kw["on_step"](molecule, -92.0)  # the starting point, as Sella records it
        for i in range(n_new):
            m = Molecule(
                list(molecule.atoms),
                [[x + 0.01 * (i + 1), y, z] for x, y, z in molecule.coordinates],
                validate_spin=False,
            )
            frames.append(m)
            kw["on_step"](m, -92.0 - 0.001 * (i + 1))
        return TSResult(
            formula="CHN",
            method=method,
            basis=basis,
            molecule=frames[-1],
            energy_hartree=-92.0,
            converged=True,
            search_converged=True,
            n_steps=n_new,
            trajectory=frames,
            energies_hartree=[-92.0] * len(frames),
        )

    return _run


def _no_freq_check(result, **kw):
    result.freq = SimpleNamespace(converged=True)
    return result


@patch("quantui.backends.worker._build_payload", return_value={"converged": True})
@patch("quantui.ts_search.apply_frequency_check", side_effect=_no_freq_check)
class TestResume:
    def test_first_attempt_saves_progress_every_step(self, _f, _b, tmp_path):
        job, attempt, request = _job(tmp_path, options={"max_steps": 50})
        seen = []

        def _run(molecule, method, basis, **kw):
            assert kw["steps"] == 50 and molecule.coordinates[2] == HCN_GUESS[2]
            inner = _fake_search(2)

            def _on_step(mol, e):
                kw_on_step(mol, e)
                seen.append(json.loads((job / _TS_PROGRESS).read_text()))

            kw_on_step = kw["on_step"]
            kw["on_step"] = _on_step
            return inner(molecule, method, basis, **kw)

        with patch("quantui.ts_search.run_ts_search", side_effect=_run):
            assert (
                run_worker_request(job / "request.json", attempt_dir=attempt).status
                == "success"
            )
        assert [len(s["energies"]) for s in seen] == [1, 2, 3]
        done = json.loads((job / _TS_SEARCH_DONE).read_text())
        assert done["key"] == _ts_state_key(request) and len(done["energies"]) == 3

    def test_killed_search_resumes_from_its_last_step(self, _f, _b, tmp_path):
        job, attempt, request = _job(tmp_path, options={"max_steps": 50})
        saved = [HCN_GUESS, [[0.1, 0, 0], [0, 0, 1.2], [0.9, 0, 0.6]]]
        (job / _TS_PROGRESS).write_text(
            json.dumps(
                {
                    "key": _ts_state_key(request),
                    "trajectory": saved,
                    "energies": [-91.9, -91.95],
                }
            )
        )
        calls = []

        def _run(molecule, method, basis, **kw):
            calls.append((molecule.coordinates, kw["steps"]))
            return _fake_search(2)(molecule, method, basis, **kw)

        with patch("quantui.ts_search.run_ts_search", side_effect=_run):
            run_worker_request(job / "request.json", attempt_dir=attempt)
        coords, steps = calls[0]
        assert [list(r) for r in coords] == saved[-1] and steps == 49
        done = json.loads((job / _TS_SEARCH_DONE).read_text())
        # The saved path plus the new steps, the resumed start not repeated.
        assert len(done["trajectory"]) == 4 and done["trajectory"][:2] == saved
        assert (
            "Resuming the transition-state search at step 1"
            in (attempt / "live.log").read_text()
        )

    def test_finished_search_is_not_redone(self, _f, _b, tmp_path):
        job, attempt, request = _job(tmp_path)
        (job / _TS_SEARCH_DONE).write_text(
            json.dumps(
                {
                    "key": _ts_state_key(request),
                    "trajectory": [HCN_GUESS, HCN_GUESS],
                    "energies": [-91.9, -92.0],
                    "search_converged": True,
                    "verdict": "",
                    "solvent": None,
                }
            )
        )
        with patch("quantui.ts_search.run_ts_search") as search:
            run_worker_request(job / "request.json", attempt_dir=attempt)
        search.assert_not_called()
        _f.assert_called_once()
        assert "running only the frequency check" in (attempt / "live.log").read_text()

    def test_state_from_a_different_calculation_is_ignored(self, _f, _b, tmp_path):
        job, attempt, request = _job(tmp_path, method="B3LYP")
        other = dict(_ts_state_key(request), method="RHF")
        (job / _TS_PROGRESS).write_text(
            json.dumps(
                {"key": other, "trajectory": [[[9, 9, 9]] * 3], "energies": [0.0]}
            )
        )
        calls = []

        def _run(molecule, method, basis, **kw):
            calls.append(molecule.coordinates)
            return _fake_search(1)(molecule, method, basis, **kw)

        with patch("quantui.ts_search.run_ts_search", side_effect=_run):
            run_worker_request(job / "request.json", attempt_dir=attempt)
        assert [list(r) for r in calls[0]] == HCN_GUESS
