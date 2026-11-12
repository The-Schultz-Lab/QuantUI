"""Run-flow fixes from the 2026-11 audit (Medium findings).

- APP-5: a failed save is reported, not swallowed; a failed extra file
  (thumbnail, Molden, ...) does not unsave the result.
- APP-10: a Cancel that arrives once the result exists does not discard it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule

pyscf = pytest.importorskip("pyscf")


def _h2() -> Molecule:
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


@pytest.fixture
def app():
    a = QuantUIApp()
    a._set_molecule(_h2())
    a.method_dd.value = "RHF"
    a.basis_dd.value = "STO-3G"
    a.calc_type_dd.value = "Single Point"
    return a


def _run(app: QuantUIApp) -> None:
    app._begin_run_lock()
    app._run_in_thread()


def _result_html(app: QuantUIApp) -> str:
    return "\n".join(
        str(o.get("data", {}).get("text/html", "")) for o in app.result_output.outputs
    )


# ── APP-5 ─────────────────────────────────────────────────────────────────


def test_failed_save_is_reported_and_forgets_the_old_folder(app, tmp_path):
    app._last_result_dir = tmp_path  # a previous run's folder
    with patch(
        "quantui.save_result", side_effect=OSError(28, "No space left on device")
    ):
        _run(app)
    assert "NOT saved" in app.run_status.value
    assert "No space left" in app.run_status.value
    assert not app.run_status.value.startswith("Done")
    assert app._last_result_dir is None
    assert app._export_bundle_btn.disabled
    assert "Result NOT saved" in _result_html(app)
    # The calculation itself still finished and is shown.
    assert app._last_result is not None


def test_failed_extra_file_does_not_unsave_the_result(app, tmp_path):
    saved = tmp_path / "run"
    saved.mkdir()
    with (
        patch("quantui.save_result", return_value=saved),
        patch("quantui.load_result", return_value={"timestamp": "t"}),
        patch("quantui.results_storage.save_thumbnail", side_effect=OSError("boom")),
        patch("quantui.results_storage.save_orbitals") as orbitals,
        patch("quantui.results_storage.save_molden") as molden,
    ):
        _run(app)
    assert app.run_status.value.startswith("Done"), app.run_status.value
    assert app._last_result_dir == saved
    # The extras after the failing thumbnail still ran.
    orbitals.assert_called_once()
    molden.assert_called_once()
    assert "NOT saved" not in _result_html(app)
    assert isinstance(app._last_result_dir, Path)


# ── APP-10 ────────────────────────────────────────────────────────────────


def test_cancel_after_the_result_exists_keeps_it(app, tmp_path):
    # The click lands while the finished result is being saved (the slow,
    # post-calculation part of a run). It used to raise out of the next log
    # write: "Cancelled.", a cancelled card under the result card, nothing
    # saved.
    saved = tmp_path / "run"
    saved.mkdir()

    def _save(*a, **kw):
        assert app.cancel_btn.disabled  # disarmed once the result exists
        app._on_cancel(None)
        return saved

    with (
        patch("quantui.save_result", side_effect=_save),
        patch("quantui.load_result", return_value={"timestamp": "t"}),
    ):
        _run(app)
    assert app.run_status.value.startswith("Done"), app.run_status.value
    assert app._last_result_dir == saved
    assert "Calculation cancelled" not in _result_html(app)
    assert not app._cancel_event.is_set()


def test_cancel_racing_the_end_of_the_calculation_keeps_the_result(app, tmp_path):
    # The click arrived just before the result was in hand (the flag is set,
    # nothing has raised yet): the result is kept, and the log says so.
    saved = tmp_path / "run"
    saved.mkdir()
    real_format = app._format_result

    def _format(result):
        app._on_cancel(None)
        assert app._cancel_event.is_set()
        return real_format(result)

    with (
        patch.object(app, "_format_result", side_effect=_format),
        patch("quantui.save_result", return_value=saved),
        patch("quantui.load_result", return_value={"timestamp": "t"}),
    ):
        _run(app)
    assert app.run_status.value.startswith("Done"), app.run_status.value
    assert "Calculation cancelled" not in _result_html(app)
    assert "the result is kept" in app._active_log.getvalue()


def test_cancel_still_works_before_the_result(app):
    app._begin_run_lock()
    assert not app._run_result_ready
    app._on_cancel(None)
    assert app._cancel_event.is_set()
    app._cancel_event.clear()
    app._end_run_lock()


# ── APP-11 ────────────────────────────────────────────────────────────────


def _moved_h2() -> Molecule:
    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.90]])


def _sp_coords(run_calc_mock) -> list:
    requests = [c.args[0] for c in run_calc_mock.call_args_list]
    sp = [r for r in requests if r.calc_type == "single_point"]
    assert sp, requests
    return sp[-1].molecule["coordinates"]


def test_preopt_failure_after_the_geometry_moved_runs_at_the_input(app):
    # The optimization returns, then something after it fails (here: the
    # summary line, on an empty energy list). The log promises "the
    # user-provided geometry as-is"; calc_mol used to be the optimized one.
    from quantui import engines

    opt = MagicMock()
    opt.molecule = _moved_h2()
    opt.converged = True
    opt.n_steps = 3
    opt.energies_hartree = []
    app._freq_preopt_cb.value = True
    with (
        patch("quantui.optimize_geometry", return_value=opt),
        patch("quantui.engines.run_calc", side_effect=engines.run_calc) as run,
        patch("quantui.save_result", side_effect=OSError("skip")),
    ):
        _run(app)
    assert _sp_coords(run)[1][2] == pytest.approx(0.74)
    html = _result_html(app)
    assert "geometry optimization before Single Point failed" in html


def test_preopt_failure_is_shown_on_the_result_card(app):
    app._freq_preopt_cb.value = True
    with (
        patch("quantui.optimize_geometry", side_effect=RuntimeError("singular matrix")),
        patch("quantui.save_result", side_effect=OSError("skip")),
    ):
        _run(app)
    html = _result_html(app)
    assert "failed (singular matrix)" in html
    assert "starting geometry" in html


def test_nmr_preopt_starts_from_the_seed_and_is_kept(app, tmp_path):
    # NMR with a seed + pre-opt: the pre-opt used to run on the active
    # molecule, then the seed overwrote its result.
    seed = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.80]])
    optimized = _moved_h2()
    app.calc_type_dd.value = "NMR Shielding"
    app._seed_dd.options = [("—", ""), ("seed", str(tmp_path))]
    app._seed_dd.value = str(tmp_path)
    app._freq_preopt_cb.value = True
    opt = MagicMock()
    opt.molecule = optimized
    opt.converged = True
    opt.n_steps = 2
    opt.energies_hartree = [-1.1]
    sp = MagicMock()
    sp.converged = True
    sp_engine = MagicMock()
    sp_engine.to_session_result.return_value = sp
    with (
        patch("quantui.results_storage.load_trajectory", return_value=([seed], [-1.0])),
        patch("quantui.optimize_geometry", return_value=opt) as optimize,
        patch("quantui.engines.run_calc", return_value=sp_engine),
        patch(
            "quantui.nmr_calc.run_nmr_calc", side_effect=RuntimeError("stop here")
        ) as nmr,
    ):
        _run(app)
    start = optimize.call_args.kwargs["molecule"]
    assert start.coordinates[1][2] == pytest.approx(0.80)  # the seed
    ran_at = nmr.call_args.kwargs["molecule"]
    assert ran_at.coordinates[1][2] == pytest.approx(0.90)  # the optimized one


# ── APP-13 ────────────────────────────────────────────────────────────────


def test_calibration_error_resets_the_panel_and_releases_run(app):
    with patch("threading.Thread"):
        app._on_cal_run(None)  # panel set to "Starting…", Run disabled
    assert app._cal_mode_toggle.disabled and app.run_btn.disabled
    with patch(
        "quantui.benchmarks.run_calibration", side_effect=RuntimeError("no pyscf")
    ):
        app._do_calibration()
    assert "Calibration failed" in app._cal_step_label.value
    assert "no pyscf" in app._cal_step_label.value
    assert not app._cal_mode_toggle.disabled
    assert app._cal_stop_btn.layout.display == "none"
    assert not app._cal_running
    assert not app.run_btn.disabled


def test_run_is_refused_while_calibrating(app):
    with patch("threading.Thread") as thread:
        app._on_cal_run(None)
    thread.return_value.start.assert_called_once()
    assert app._cal_running and app.run_btn.disabled
    with patch.object(app, "_run_in_thread") as worker:
        app._on_run_clicked(None)
    worker.assert_not_called()
    assert not app._calc_running
    assert "Calibration is running" in app.run_status.value


def test_calibration_is_refused_while_a_run_is_in_flight(app):
    app._begin_run_lock()
    with patch("threading.Thread") as thread:
        app._on_cal_run(None)
    thread.assert_not_called()
    assert not app._cal_running
    assert "calculation is running" in app._cal_step_label.value
    app._end_run_lock()


# ── APP-8 / UI-5 ──────────────────────────────────────────────────────────


@pytest.fixture
def optimized_h2(app, tmp_path):
    """A Geometry Opt on H2 (RHF/STO-3G) saved to ``tmp_path/run``."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    app.calc_type_dd.value = "Geometry Opt"
    with (
        patch("quantui.save_result", return_value=run_dir),
        patch("quantui.load_result", return_value={"timestamp": "t"}),
    ):
        _run(app)
    assert app._last_result_dir == run_dir, app.run_status.value
    return run_dir


def test_structure_export_follows_the_run_not_a_history_view(
    app, optimized_h2, tmp_path
):
    viewed = tmp_path / "history_view"
    viewed.mkdir()
    app._last_result_dir = viewed  # a History view of another result
    app._on_export_xyz(None)
    files = list(optimized_h2.glob("H2_*.xyz"))
    assert len(files) == 1, app.struct_export_status.value
    assert not list(viewed.glob("*.xyz"))
    # The optimized geometry, not the input (0.74 Å).
    z = float(files[0].read_text().splitlines()[3].split()[3])
    assert z != pytest.approx(0.74, abs=1e-4)


def test_structure_export_after_loading_another_molecule(
    app, optimized_h2, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app._set_molecule(
        Molecule(
            ["N", "H", "H", "H"],
            [
                [0, 0, 0.1],
                [0.94, 0, -0.27],
                [-0.47, 0.81, -0.27],
                [-0.47, -0.81, -0.27],
            ],
        )
    )
    app.method_dd.value = "B3LYP"
    app._on_export_xyz(None)
    assert not list(optimized_h2.glob("H3N*.xyz"))  # not the old run's folder
    files = list(tmp_path.glob("*.xyz"))
    assert [f.name for f in files] == ["H3N_B3LYP_STO-3G.xyz"]
    text = files[0].read_text()
    assert text.splitlines()[0] == "4"
    assert "B3LYP" in text.splitlines()[1]


# ── UI-9 ──────────────────────────────────────────────────────────────────


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
    )


def test_loading_another_molecule_resets_the_edit_state(app):
    from quantui.app_structure_edit import on_set_bond, on_undo

    app._set_molecule(_water())
    app._edit_atoms_txt.value = "1 2"
    app._edit_value.value = 1.0
    on_set_bond(app)
    app.frozen_atoms_txt.value = "1, 3"
    app._edit_atoms_txt.value = "2 3"
    app._set_molecule(_h2(), "library")
    assert app._edit_undo_stack == []
    assert app._edit_atoms_txt.value == ""
    assert app.frozen_atoms_txt.value == ""  # atom 3 does not even exist now
    on_undo(app)  # used to swap the water back in
    assert app._molecule.atoms == ["H", "H"]


def test_same_atoms_reload_keeps_the_freeze_list(app):
    app._set_molecule(_water())
    app.frozen_atoms_txt.value = "1"
    moved = _water()
    moved.coordinates[0][2] = 0.2
    app._set_molecule(moved, "Pre-optimized")
    assert app.frozen_atoms_txt.value == "1"


def test_delete_renumbers_the_freeze_list_and_undo_restores_it(app):
    from quantui.app_structure_edit import on_delete, on_undo

    mol = Molecule(
        ["C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    )
    app._set_molecule(mol)
    app.frozen_atoms_txt.value = "2, 4"
    app._edit_atoms_txt.value = "2 3"
    on_delete(app)
    assert app._molecule.atoms == ["C", "H", "H"]
    # Atom 2 is gone; old atom 4 is now atom 2.
    assert app.frozen_atoms_txt.value == "2"
    assert "renumbered" in app._edit_msg.value
    on_undo(app)
    assert len(app._molecule.atoms) == 5
    assert app.frozen_atoms_txt.value == "2, 4"


# ── UI-14 (export provenance item) ────────────────────────────────────────


def test_reorg_export_names_the_displayed_results_level(app, tmp_path):
    # A History result at B3LYP/6-31G is on display while the setup
    # dropdowns are at RHF/STO-3G: the exported files used to be named and
    # stamped with the dropdowns.
    from quantui.app import _AnalysisContext

    app._apply_analysis_context(
        _AnalysisContext(
            calc_type="single_point",
            formula="H2",
            method="B3LYP",
            basis="6-31G",
            source="history",
        )
    )
    app._last_result_dir = tmp_path
    app._reorg_geometries = [
        {
            "label": "R_neutral — optimized neutral",
            "atoms": ["H", "H"],
            "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            "charge": 0,
            "multiplicity": 1,
            "note": "",
        }
    ]
    app._on_export_reorg_geometries(None)
    files = [f.name for f in tmp_path.glob("*.xyz")]
    assert files == ["H2_R_neutral_B3LYP_6-31G.xyz"], files
    assert "B3LYP/6-31G" in (tmp_path / files[0]).read_text()


def test_export_level_falls_back_to_the_setup_before_any_result(app):
    from quantui.app_exports import displayed_level

    app._displayed_level = None
    assert displayed_level(app) == ("RHF", "STO-3G")


def test_script_export_follows_the_run_not_a_history_view(
    app, optimized_h2, tmp_path, monkeypatch
):
    # Same rule as the structure exports (audit UI-5): the run's own folder
    # while its molecule is loaded, else the working directory — never the
    # last-viewed History folder.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    viewed = tmp_path / "history_view"
    viewed.mkdir()
    app._last_result_dir = viewed
    app._on_export(None)
    assert len(list(optimized_h2.glob("H2_*.py"))) == 1, app.export_status.value
    assert not list(viewed.glob("*.py"))
    app._set_molecule(_water(), "library")
    app._on_export(None)
    assert [f.name for f in cwd.glob("*.py")] == ["H2O_RHF_STO-3G.py"]
    assert not list(viewed.glob("*.py"))
    assert not list(optimized_h2.glob("H2O*.py"))
