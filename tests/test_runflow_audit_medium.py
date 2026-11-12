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
