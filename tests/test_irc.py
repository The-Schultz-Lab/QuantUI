"""IRC calc type (Sella's IRC from a transition state)."""

from __future__ import annotations

import io

import pytest

from quantui.irc import IRCBranch, IRCResult
from quantui.molecule import Molecule

ATOMS = ["C", "N", "H"]
TS_XYZ = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.18], [1.0, 0.0, 0.6]]


def _mol(h):
    return Molecule(ATOMS, [[0, 0, 0], [0, 0, 1.18], h], validate_spin=False)


def _fake(rev_end=-92.80, fwd_end=-92.82, *, irc_steps=None, converged=True):
    ts = _mol(TS_XYZ[2])
    rev = IRCBranch(
        "reverse",
        converged,
        [ts, _mol([1.0, 0, 0.9]), _mol([0.9, 0, 1.6])],
        [-92.70, -92.75, rev_end],
    )
    fwd = IRCBranch(
        "forward",
        converged,
        [ts, _mol([1.0, 0, 0.3]), _mol([0.6, 0, -0.9]), _mol([0.0, 0, -1.05])],
        [-92.70, -92.76, -92.80, fwd_end],
        irc_steps=irc_steps,
    )
    return IRCResult("CHN", "RHF", "STO-3G", ts, fwd, rev)


class TestResult:
    def test_path_runs_reverse_end_to_forward_end(self):
        r = _fake()
        assert len(r.trajectory) == len(r.energies_hartree) == 6
        assert r.ts_index == 2 and r.trajectory[2] is r.molecule
        s = r.reaction_coordinate
        assert s[2] == 0.0 and s[0] < s[1] < 0 < s[3] < s[4] < s[5]
        assert r.energies_hartree[0] == -92.80 and r.energies_hartree[-1] == -92.82
        assert r.energy_hartree == -92.70 and r.n_steps == 5

    def test_barriers_and_verdict(self):
        r = _fake()
        b_start, b_end = r.barriers_kj()
        assert b_start == pytest.approx(0.10 * 2625.4996, rel=1e-6)
        assert b_end == pytest.approx(0.12 * 2625.4996, rel=1e-6)
        assert r.verdict.startswith("The transition state connects two minima")
        block = r.to_spectra()
        assert block["ts_index"] == 2 and len(block["reaction_coordinate"]) == 6
        assert block["forward"] == {"converged": True, "n_steps": 3, "irc_steps": None}

    def test_not_downhill_on_one_side(self):
        r = _fake(rev_end=-92.70)
        assert "may not be a transition state" in r.verdict

    def test_minimized_tail_and_unconverged_are_said(self):
        r = _fake(irc_steps=2, converged=False)
        assert "forward path was completed by minimizing" in r.verdict
        assert r.verdict.startswith("Not every direction reached a minimum")

    def test_post_hf_is_refused(self):
        from quantui.irc import run_irc

        with pytest.raises(ValueError, match="analytic gradients"):
            run_irc(_mol(TS_XYZ[2]), "MP2", "STO-3G")


class TestApp:
    def _saved_ts(self, results):
        """A Transition State result on disk, as the app saves it."""
        from quantui.results_storage import save_result, save_trajectory

        ts = _mol(TS_XYZ[2])
        res = type(
            "R",
            (),
            dict(
                formula="CHN",
                method="RHF",
                basis="STO-3G",
                energy_hartree=-92.70,
                converged=True,
                molecule=ts,
            ),
        )()
        d = save_result(
            res,
            calc_type="transition_state",
            spectra={"transition_state": {"n_imaginary": 1}},
            results_dir=results,
            molecule=ts,
        )
        save_trajectory(d, [_mol([1.0, 0, 0.5]), ts], [-92.69, -92.70])
        return d

    def test_seed_run_card_panels_and_history(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from quantui import load_result
        from quantui.app import QuantUIApp
        from quantui.app_formatters import format_past_result

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        ts_dir = self._saved_ts(tmp_path)
        app = QuantUIApp()
        app._set_molecule(_mol([1.05, 0, 0.55]), "guess")
        app.calc_type_dd.value = "IRC"
        seeds = dict((v, k) for k, v in app._seed_dd.options)
        assert str(ts_dir) in seeds and "⛰ TS" in seeds[str(ts_dir)]
        assert app._freq_preopt_cb.layout.display == "none"
        app._seed_dd.value = str(ts_dir)

        started = {}

        def _run(mol, method, basis, **kw):
            started["coords"] = [list(r) for r in mol.coordinates]
            return _fake()

        with patch("quantui.irc.run_irc", side_effect=_run):
            app._do_run()
        # The run started from the transition state, not the guess.
        assert started["coords"][2] == pytest.approx(TS_XYZ[2])
        assert app._last_calc_type == "irc"
        assert {"PES Scan", "Trajectory"} <= set(app._ana_available)
        saved = app._last_result_dir
        data = load_result(saved)
        assert data["calc_type"] == "irc"
        assert data["spectra"]["irc"]["ts_index"] == 2
        assert (saved / "trajectory.json").exists()
        card = format_past_result(data, saved)
        assert "IRC" in card and "Barrier" in card and "connects two minima" in card

        fresh = QuantUIApp()
        fresh._apply_analysis_context(fresh._build_history_context(saved))
        assert {"PES Scan", "Trajectory"} <= set(fresh._ana_available)


class TestBatch:
    def test_worker_payload_summary_and_history(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import patch

        from quantui import load_result
        from quantui.backends.base import CalculationRequest
        from quantui.backends.batch_input import BatchInputError, check_method
        from quantui.backends.registry import JobRegistry
        from quantui.backends.result_summary import summarize
        from quantui.backends.slurm_ingest import ingest_staging_success
        from quantui.backends.worker import run_worker_request

        job = tmp_path / "staging" / "hcn_irc"
        attempt = job / "attempt-01_job7"
        attempt.mkdir(parents=True)
        request = CalculationRequest(
            request_id="hcn_irc",
            calc_type="irc",
            method="RHF",
            basis="STO-3G",
            charge=0,
            multiplicity=1,
            molecule={"atoms": ATOMS, "coords": TS_XYZ},
            options={"max_steps": 40},
        )
        (job / "request.json").write_text(json.dumps(request.to_dict()))
        seen = {}

        def _run(mol, method, basis, **kw):
            seen.update(steps=kw["steps"], coords=mol.coordinates)
            return _fake()

        with patch("quantui.irc.run_irc", side_effect=_run):
            out = run_worker_request(job / "request.json", attempt_dir=attempt)
        assert out.status == "success", out.error
        assert seen["steps"] == 40
        payload = json.loads((attempt / "result.json").read_text())
        assert payload["calc_type"] == "irc" and payload["n_steps"] == 5
        assert payload["spectra"]["irc"]["ts_index"] == 2
        text = "\n".join(summarize(payload, job, attempt))
        assert "path steps:       5" in text and "kJ/mol from its end" in text

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path / "results"))
        registry = JobRegistry(
            jobs_root=tmp_path / "jobs", staging_root=tmp_path / "staging"
        )
        record = registry.create(request, "cluster_slurm", status="success")
        saved = ingest_staging_success(record, "", attempt_dir=attempt)
        assert load_result(saved)["spectra"]["irc"]["verdict"]
        assert (saved / "trajectory.json").exists()

        request.method = "CCSD"
        with pytest.raises(BatchInputError, match="analytic gradients"):
            check_method(request)


@pytest.mark.slow
def test_hcn_hnc_path_connects_both_minima():
    pytest.importorskip("sella")
    pytest.importorskip("pyscf")
    import numpy as np

    from quantui.irc import run_irc
    from quantui.ts_search import run_ts_search

    guess = Molecule(ATOMS, [[0, 0, 0], [0, 0, 1.18], [1.05, 0, 0.55]])
    ts = run_ts_search(guess, "RHF", "STO-3G", progress_stream=io.StringIO())
    r = run_irc(ts.molecule, "RHF", "STO-3G", progress_stream=io.StringIO())
    assert r.converged

    def h_on(frame):
        x = np.asarray(frame.coordinates)
        return "C" if np.linalg.norm(x[2] - x[0]) < np.linalg.norm(x[2] - x[1]) else "N"

    # One end is HCN (H on C), the other HNC (H on N).
    assert {h_on(r.trajectory[0]), h_on(r.trajectory[-1])} == {"C", "N"}
    b_start, b_end = r.barriers_kj()
    assert b_start > 100 and b_end > 100


@pytest.mark.slow
def test_batch_chain_ts_then_irc(tmp_path):
    # quantui-batch submit guess.xyz --calc transition_state, then
    # submit --from that job --calc irc: the IRC starts at the saddle point.
    pytest.importorskip("sella")
    pytest.importorskip("pyscf")
    import json

    import numpy as np

    from quantui.backends.base import CalculationRequest
    from quantui.backends.worker import run_worker_request

    def _job(name, calc, run_context=None):
        job = tmp_path / name
        attempt = job / "attempt-01_job1"
        attempt.mkdir(parents=True)
        req = CalculationRequest(
            request_id=name,
            calc_type=calc,
            method="RHF",
            basis="STO-3G",
            charge=0,
            multiplicity=1,
            molecule={
                "atoms": ATOMS,
                "coords": [[0, 0, 0], [0, 0, 1.18], [1.05, 0, 0.55]],
            },
            run_context=run_context or {},
        )
        (job / "request.json").write_text(json.dumps(req.to_dict()))
        out = run_worker_request(job / "request.json", attempt_dir=attempt)
        assert out.status == "success", out.error
        return job, json.loads((attempt / "result.json").read_text())

    ts_job, ts = _job("ts", "transition_state")
    _, irc = _job("irc", "irc", {"geometry_from": str(ts_job)})
    block = irc["spectra"]["irc"]
    assert irc["calc_type"] == "irc" and irc["converged"]
    assert block["barrier_from_start_kj"] > 100 and block["barrier_from_end_kj"] > 100
    path = json.loads(
        (tmp_path / "irc" / "attempt-01_job1" / "trajectory.json").read_text()
    )

    def dists(xyz):
        x = np.array(xyz)
        return np.linalg.norm(x[:, None] - x[None], axis=-1)

    # Same structure (Sella may move it rigidly in space).
    start = path["steps"][block["ts_index"]]["coords"]
    assert dists(start) == pytest.approx(
        dists(ts["final_molecule"]["coords"]), abs=1e-3
    )
