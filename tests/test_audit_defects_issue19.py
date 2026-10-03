"""Remaining parity-audit § 5 defects (M-ISSUES ISSUE.19)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quantui.backends.dispatch import build_calculation_request
from quantui.molecule import Molecule
from tests.test_backends_dispatch import _fake_app

H2 = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def staging(tmp_path):
    """A worker staging folder holding an H2 single-point request."""
    from quantui.backends.base import CalculationRequest

    staging_dir = tmp_path / "staging" / "job1"
    staging_dir.mkdir(parents=True)
    request = CalculationRequest(
        request_id="job1",
        calc_type="single_point",
        method="RHF",
        basis="STO-3G",
        charge=0,
        multiplicity=1,
        molecule={"atoms": H2.atoms, "coordinates": H2.coordinates},
    )
    (staging_dir / "request.json").write_text(json.dumps(request.to_dict()))
    return staging_dir


def _freq_result_dir(tmp_path):
    """A saved H2 frequency result with one normal mode."""
    from quantui.results_storage import save_result

    result = SimpleNamespace(
        formula="H2", method="RHF", basis="STO-3G", energy_hartree=-1.1, converged=True
    )
    return save_result(
        result,
        results_dir=tmp_path,
        calc_type="frequency",
        spectra={
            "ir": {
                "frequencies_cm1": [-300.0],
                "ir_intensities": [0.0],
                "displacements": [[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]],
            },
            "molecule": {
                "atoms": H2.atoms,
                "coords": H2.coordinates,
                "charge": 0,
                "multiplicity": 1,
            },
        },
    )


class TestPreoptOnSlurm:
    """#3 — SP/NMR pre-opt was dropped on SLURM; #8 — mode seeds skipped it."""

    @pytest.mark.parametrize(
        "ct,calc_type", [("Single Point", "single_point"), ("NMR Shielding", "nmr")]
    )
    def test_single_point_and_nmr_send_the_preopt_flag(self, ct, calc_type):
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value=ct),
            _freq_preopt_cb=SimpleNamespace(value=True),
        )
        req = build_calculation_request(app, request_id="p")
        assert req.calc_type == calc_type
        assert req.options.get("preopt_before_run") is True

    def test_unticked_sends_nothing(self):
        req = build_calculation_request(_fake_app(), request_id="p")
        assert "preopt_before_run" not in req.options

    def test_mode_displaced_frequency_seed_keeps_the_preopt(self, tmp_path):
        seed = f"freq:{_freq_result_dir(tmp_path)}"
        app = _fake_app(
            calc_type_dd=SimpleNamespace(value="Frequency"),
            _seed_dd=SimpleNamespace(value=seed),
            _freq_preopt_cb=SimpleNamespace(value=True),
            _freq_perturb_mode_dd=SimpleNamespace(value=1),
            _freq_perturb_fraction=SimpleNamespace(value=0.5),
        )
        req = build_calculation_request(app, request_id="m")
        # Displaced along the mode, and re-optimized before the frequencies.
        assert req.molecule["coordinates"][1][2] != pytest.approx(0.74)
        assert req.options.get("preopt_before_run") is True

    @pytest.mark.parametrize("calc_type", ["single_point", "nmr"])
    def test_worker_preoptimizes_before_sp_and_nmr(self, staging, calc_type):
        moved = Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.71]])
        opt = SimpleNamespace(
            molecule=moved,
            trajectory=[moved],
            energies_hartree=[-1.117],
            converged=True,
            n_steps=3,
        )
        seen = {}

        def _fake_calc(**kw):
            seen["z"] = kw["molecule"].coordinates[1][2]
            return SimpleNamespace(
                energy_hartree=-1.117,
                homo_lumo_gap_ev=None,
                converged=True,
                n_iterations=4,
                method="RHF",
                basis="STO-3G",
                formula="H2",
                shielding_iso=[30.0, 30.0],
                atom_symbols=["H", "H"],
                reference_shielding={},
                chemical_shifts={},
            )

        data = json.loads((staging / "request.json").read_text())
        data["calc_type"] = calc_type
        data["options"] = {"preopt_before_run": True}
        (staging / "request.json").write_text(json.dumps(data))
        target = (
            "quantui.session_calc.run_in_session"
            if calc_type == "single_point"
            else "quantui.nmr_calc.run_nmr_calc"
        )
        from quantui.backends.base import CalculationRequest
        from quantui.backends.worker import _run_nmr, _run_single_point

        req = CalculationRequest.from_dict(data)
        runner = _run_single_point if calc_type == "single_point" else _run_nmr
        with patch("quantui.optimizer.optimize_geometry", return_value=opt) as m_opt:
            with patch(target, side_effect=_fake_calc):
                with open(staging / "live.log", "w") as log:
                    runner(req, staging, log)
        m_opt.assert_called_once()
        assert seen["z"] == pytest.approx(0.71)


class TestSeedGating:
    """#8 — the checkbox follows the seed kind."""

    def _app(self, seed):
        return SimpleNamespace(
            _freq_preopt_cb=SimpleNamespace(value=False, disabled=True),
            _seed_dd=SimpleNamespace(value=seed),
        )

    def test_mode_seed_ticks_and_enables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("freq:/x")
        _gate_preopt_for_seed(app, "freq:/x")
        assert app._freq_preopt_cb.value is True
        assert app._freq_preopt_cb.disabled is False

    def test_optimized_seed_disables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("/results/opt")
        app._freq_preopt_cb.value = True
        _gate_preopt_for_seed(app, "/results/opt")
        assert app._freq_preopt_cb.value is False
        assert app._freq_preopt_cb.disabled is True

    def test_no_seed_enables(self):
        from quantui.app_runflow import _gate_preopt_for_seed

        app = self._app("")
        _gate_preopt_for_seed(app, "")
        assert app._freq_preopt_cb.disabled is False
