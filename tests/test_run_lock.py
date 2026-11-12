"""One run at a time, read once (audit APP-2, APP-3, APP-4, APP-6).

- A second Run click while a run is in flight is ignored, and loading a
  molecule mid-run does not re-arm Run.
- The run's inputs are frozen while it runs and restored afterwards.
- Everything a run depends on is read when it starts: a widget change or a
  Charge-field edit mid-run cannot reach its later stages.
- A seed geometry keeps the charge/multiplicity of the run being set up.
- An error before ``_do_run``'s own cleanup still releases the run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule


def _water(charge: int = 0, mult: int = 1) -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
        charge=charge,
        multiplicity=mult,
    )


@pytest.fixture
def app():
    a = QuantUIApp()
    a._set_molecule(_water())
    a.method_dd.value = "RHF"
    a.basis_dd.value = "STO-3G"
    return a


def test_second_click_while_running_is_ignored(app):
    # The worker is a stub that never finishes, so the run stays in flight.
    with patch.object(app, "_run_in_thread") as worker:
        app._on_run_clicked(None)
        app._on_run_clicked(None)
    assert worker.call_count == 1
    assert app._calc_running and app.run_btn.disabled


def test_inputs_frozen_while_running_and_restored_after(app):
    app.solvent_cb.disabled = True  # a control that was already off stays off
    with patch.object(app, "_run_in_thread"):
        app._on_run_clicked(None)
    for w in (app.method_dd, app.basis_dd, app.charge_si, app.mult_si):
        assert w.disabled, w
    assert app.calc_type_dd.disabled
    assert not app.method_help_btn.disabled
    app._end_run_lock()
    for w in (app.method_dd, app.basis_dd, app.charge_si, app.mult_si):
        assert not w.disabled, w
    assert app.solvent_cb.disabled
    assert not app._calc_running and not app.run_btn.disabled


def test_loading_a_molecule_mid_run_keeps_run_disabled(app):
    with patch.object(app, "_run_in_thread"):
        app._on_run_clicked(None)
    app._set_molecule(_water())
    assert app.run_btn.disabled


def test_error_before_the_run_starts_releases_the_lock(app):
    app._begin_run_lock()
    with patch.object(app, "_do_run", side_effect=OSError(28, "No space left")):
        app._run_in_thread()
    assert not app._calc_running
    assert not app.run_btn.disabled
    assert not app.method_dd.disabled
    assert "could not start" in app.run_status.value


def _opt_mocks(final_mol: Molecule):
    opt = MagicMock()
    opt.energy_hartree = -75.0
    opt.converged = True
    opt.n_iterations = 3
    opt.energies_hartree = [-75.0]
    opt.trajectory = []
    opt.formula = "H2O"
    opt.method = "RHF"
    opt.basis = "STO-3G"
    opt.molecule = final_mol
    opt.mo_energy_hartree = opt.mo_occ = opt.mo_coeff = None
    opt.pyscf_mol_atom = opt.pyscf_mol_basis = None
    sp = MagicMock()
    sp.converged = True
    sp.energy_hartree = -75.1
    sp.mo_energy_hartree = [0.0]
    sp.mo_occ = [2.0]
    sp.mo_coeff = [[0.0]]
    sp.pyscf_mol_atom = [("O", [0.0, 0.0, 0.0])]
    sp.pyscf_mol_basis = "STO-3G"
    e_opt, e_sp = MagicMock(), MagicMock()
    e_opt.to_session_result.return_value = opt
    e_sp.to_session_result.return_value = sp
    return e_opt, e_sp


def test_mid_run_edits_do_not_reach_later_stages(app):
    # Geometry Opt, then its required final single point. Between the two,
    # the basis dropdown and the active molecule's charge change (a molecule
    # load or the Charge field). The final single point used to read both.
    app.calc_type_dd.value = "Geometry Opt"
    e_opt, e_sp = _opt_mocks(_water())

    def _engine(request, *a, **kw):
        if request.calc_type == "geometry_opt":
            app.basis_dd.value = "3-21G"
            app._molecule.charge = 1
            return e_opt
        return e_sp

    with patch("quantui.engines.run_calc", side_effect=_engine) as run:
        with patch("quantui.save_result"):
            app._begin_run_lock()
            app._run_in_thread()
    requests = [c.args[0] for c in run.call_args_list]
    assert [r.calc_type for r in requests] == ["geometry_opt", "single_point"]
    assert {r.basis for r in requests} == {"STO-3G"}
    assert {r.charge for r in requests} == {0}


def test_seed_keeps_the_runs_charge_and_multiplicity(app, tmp_path):
    # A neutral optimization's final geometry as the seed for a cation run.
    from quantui.app_runflow import seed_with_run_charge

    seeded = seed_with_run_charge(_water(0, 1), _water(1, 2))
    assert (seeded.charge, seeded.multiplicity) == (1, 2)

    app._set_molecule(_water(1, 2))
    app.charge_si.value, app.mult_si.value = 1, 2
    app.method_dd.value = "UHF"
    app.calc_type_dd.value = "Geometry Opt"
    app._geo_seed_dd.options = [("—", ""), ("neutral opt", str(tmp_path))]
    app._geo_seed_dd.value = str(tmp_path)
    e_opt, e_sp = _opt_mocks(_water(1, 2))
    with patch(
        "quantui.results_storage.load_trajectory",
        return_value=([_water(0, 1)], [-75.0]),
    ):
        with patch("quantui.engines.run_calc", side_effect=[e_opt, e_sp]) as run:
            with patch("quantui.save_result"):
                app._begin_run_lock()
                app._run_in_thread()
    opt_request = run.call_args_list[0].args[0]
    assert (opt_request.charge, opt_request.multiplicity) == (1, 2)
