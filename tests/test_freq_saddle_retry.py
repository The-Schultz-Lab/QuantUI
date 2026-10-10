"""
M-BATCH2 B2.5 — one retry off a saddle point after a pre-optimized Frequency run.

A geometry optimization can stop where the forces vanish by symmetry, e.g. a
methyl rotor at its eclipsed conformation; the frequencies then show an
imaginary mode. ``retry_frequency_off_saddle`` displaces along it,
re-optimizes and repeats the frequencies once. The engine calls are mocked
here (a real toluene run takes minutes; see the B2.5 commit message).
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.freq_calc import (
    IMAGINARY_RETRY_THRESHOLD_CM1,
    FreqResult,
    imaginary_modes_to_follow,
    retry_frequency_off_saddle,
)
from quantui.molecule import Molecule
from quantui.optimizer import FREQ_PREOPT_FMAX

H2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
# Three modes for two atoms is not physical; only the shapes matter here.
MODES = [
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
]


def _freq(freqs, energy=-1.10, displacements=MODES):
    return FreqResult(
        energy_hartree=energy,
        homo_lumo_gap_ev=None,
        converged=True,
        n_iterations=5,
        method="RHF",
        basis="STO-3G",
        formula="H2",
        frequencies_cm1=list(freqs),
        displacements=displacements,
    )


def _opt(molecule, steps=4):
    return SimpleNamespace(molecule=molecule, n_steps=steps, converged=True)


class TestModesToFollow:
    def test_only_modes_below_the_threshold(self):
        r = _freq([-39.4, IMAGINARY_RETRY_THRESHOLD_CM1 + 1, 500.0])
        assert imaginary_modes_to_follow(r) == [0]

    def test_nothing_without_displacements(self):
        assert imaginary_modes_to_follow(_freq([-80.0, 1, 2], displacements=None)) == []


class TestRetry:
    def test_no_imaginary_mode_does_nothing(self):
        with patch("quantui.optimizer.optimize_geometry") as m_opt:
            assert (
                retry_frequency_off_saddle(
                    H2, _freq([10.0, 500.0, 4400.0]), method="RHF", basis="STO-3G"
                )
                is None
            )
        m_opt.assert_not_called()

    def test_displaces_reoptimizes_and_recomputes(self):
        seen = {}
        reopt = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.73]])

        def fake_opt(**kw):
            seen["start"] = kw["molecule"].coordinates
            seen["fmax"] = kw["fmax"]
            seen["solvent"] = kw["solvent"]
            return _opt(reopt, steps=7)

        second = _freq([60.0, 500.0, 4400.0], energy=-1.12)
        with patch("quantui.optimizer.optimize_geometry", side_effect=fake_opt):
            with patch(
                "quantui.freq_calc.run_freq_calc", return_value=second
            ) as m_freq:
                mol, res = retry_frequency_off_saddle(
                    H2,
                    _freq([-48.3, -112.5, 4400.0]),
                    method="B3LYP",
                    basis="def2-SVP",
                    solvent="water",
                    progress_stream=io.StringIO(),
                )

        # Displaced along both imaginary modes (x and y), 0.3 Å each.
        assert seen["start"][0] == pytest.approx([0.3, 0.3, 0.0])
        assert seen["start"][1] == pytest.approx([-0.3, -0.3, 0.74])
        assert seen["fmax"] == FREQ_PREOPT_FMAX
        assert seen["solvent"] == "water"
        assert m_freq.call_args.kwargs["molecule"] is reopt
        assert m_freq.call_args.kwargs["solvent"] == "water"
        assert mol is reopt and res is second
        assert res.imaginary_mode_retry == {
            "followed_cm1": [-48.3, -112.5],
            "displacement_angstrom": pytest.approx(0.3),
            "reopt_steps": 7,
            "reopt_converged": True,
            "reopt_fmax_ev_per_angstrom": FREQ_PREOPT_FMAX,
            "energy_change_hartree": pytest.approx(-0.02),
            "remaining_imaginary_cm1": [],
        }
        json.dumps(res.imaginary_mode_retry)  # saved in result.json

    def test_reports_what_is_left_after_one_try(self, capsys):
        with patch("quantui.optimizer.optimize_geometry", return_value=_opt(H2)):
            with patch(
                "quantui.freq_calc.run_freq_calc",
                return_value=_freq([-300.0, 500.0, 4400.0]),
            ) as m_freq:
                _mol, res = retry_frequency_off_saddle(
                    H2, _freq([-300.0, 500.0, 4400.0]), method="RHF", basis="STO-3G"
                )
        m_freq.assert_called_once()  # once, never a loop
        assert res.imaginary_mode_retry["remaining_imaginary_cm1"] == [-300.0]
        assert "Still imaginary after one retry" in capsys.readouterr().out


class TestWorker:
    @pytest.fixture
    def request_path(self, tmp_path):
        from quantui.backends.base import CalculationRequest

        staging = tmp_path / "job"
        staging.mkdir()
        req = CalculationRequest(
            request_id="saddle1",
            calc_type="frequency",
            method="RHF",
            basis="STO-3G",
            charge=0,
            multiplicity=1,
            molecule={"atoms": ["H", "H"], "coordinates": [[0, 0, 0], [0, 0, 0.74]]},
            options={"preopt_before_run": True},
        )
        path = staging / "request.json"
        path.write_text(json.dumps(req.to_dict()))
        return path

    def _run(self, path, freqs_in_order):
        from quantui.backends.worker import run_worker_request

        results = [_freq(f) for f in freqs_in_order]
        with patch(
            "quantui.optimizer.optimize_geometry",
            return_value=SimpleNamespace(
                molecule=H2,
                trajectory=[H2],
                energies_hartree=[-1.1],
                converged=True,
                n_steps=2,
            ),
        ) as m_opt:
            with patch("quantui.freq_calc.run_freq_calc", side_effect=results):
                outcome = run_worker_request(path)
        payload = json.loads((path.parent / "result.json").read_text())
        return outcome, payload, m_opt

    def test_imaginary_mode_after_preopt_is_retried(self, request_path):
        outcome, payload, m_opt = self._run(
            request_path, [[-39.4, 500.0, 4400.0], [55.0, 500.0, 4400.0]]
        )
        assert outcome.status == "success"
        assert m_opt.call_count == 2  # the pre-opt, then the re-optimization
        ir = payload["spectra"]["ir"]
        assert ir["frequencies_cm1"] == [55.0, 500.0, 4400.0]
        assert ir["imaginary_mode_retry"]["followed_cm1"] == [-39.4]
        log = (request_path.parent / "live.log").read_text()
        assert "Imaginary mode(s) 39.4i" in log

    def test_no_retry_without_a_preopt(self, request_path):
        data = json.loads(request_path.read_text())
        data["options"] = {}
        request_path.write_text(json.dumps(data))
        _outcome, payload, m_opt = self._run(request_path, [[-39.4, 500.0, 4400.0]])
        m_opt.assert_not_called()
        assert payload["spectra"]["ir"]["imaginary_mode_retry"] is None


def test_frequency_chain_uses_the_retried_geometry(tmp_path):
    """--from a frequency job starts at the geometry its frequencies used."""
    from quantui.backends.batch_chain import geometry_from_attempt

    attempt = tmp_path / "attempt-01_job1"
    attempt.mkdir()
    (attempt / "preopt_trajectory.json").write_text(
        json.dumps(
            {"atoms": ["H", "H"], "steps": [{"coords": [[0, 0, 0], [0, 0, 0.8]]}]}
        )
    )
    final = {"atoms": ["H", "H"], "coords": [[0, 0, 0], [0, 0, 0.73]]}
    (attempt / "result.json").write_text(json.dumps({"spectra": {"molecule": final}}))
    assert geometry_from_attempt(attempt, "frequency") == final


def test_result_card_mentions_the_retry():
    from quantui.app_formatters import format_freq_result

    r = _freq([55.0, 500.0, 4400.0])
    r.imaginary_mode_retry = {"followed_cm1": [-39.4], "reopt_steps": 7}
    html = format_freq_result(r)
    assert "Saddle-point retry" in html and "39.4i" in html and "7 steps" in html
    assert "Saddle-point retry" not in format_freq_result(_freq([55.0, 500.0]))
