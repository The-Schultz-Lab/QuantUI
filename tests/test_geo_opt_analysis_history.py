"""Integration tests for the geometry-optimisation analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result writes result.json with calc_type
                           "geometry_opt"; save_trajectory writes trajectory.json;
                           save_orbitals writes orbitals.npz.
  (2) History context    — _build_history_context reconstructs the context.
  (3) Panel activation   — _apply_analysis_context activates Trajectory, Energies,
                           and Isosurface panels from a history context.
  (4) _do_run end-to-end — Full path: real RHF/STO-3G geometry opt → disk write →
                           _history_load_analysis → all three panels activate.
                           (PySCF-gated; skipped on Windows.)
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule
from quantui.results_storage import (
    list_results,
    load_result,
    save_orbitals,
    save_result,
    save_trajectory,
)

try:
    from quantui.app import _PYSCF_AVAILABLE
except ImportError:
    _PYSCF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _water():
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


def _make_geo_opt_result(with_coeff: bool = True):
    """Minimal namespace mirroring a real RHF/STO-3G geometry opt on water.

    Includes two trajectory steps so _pop_geo_trajectory passes the len >= 2 check.
    """
    water_initial = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    water_final = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.96, 0.0, 0.0]],
    )
    mo_coeff = np.eye(7) if with_coeff else None
    return SimpleNamespace(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.1,
        energy_ev=-2043.0,
        homo_lumo_gap_ev=10.2,
        converged=True,
        n_steps=2,
        n_iterations=10,
        trajectory=[water_initial, water_final],
        energies_hartree=[-75.0, -75.1],
        molecule=water_final,
        mo_energy_hartree=np.array([-20.5, -1.3, -0.7, -0.5, -0.3, 0.5, 0.7]),
        mo_occ=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
        mo_coeff=mo_coeff,
        pyscf_mol_atom=[
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.96, 0.0, 0.0]),
            ("H", [-0.96, 0.0, 0.0]),
        ],
        pyscf_mol_basis="sto-3g",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    a = QuantUIApp()
    a._set_molecule(_water())
    return a


@pytest.fixture
def geo_opt_result():
    return _make_geo_opt_result()


# ---------------------------------------------------------------------------
# Part 1: save_result + save_trajectory + save_orbitals write the correct files
# ---------------------------------------------------------------------------


class TestGeoOptSpectraStructure:
    def _save_all(self, tmp_path, result):
        saved = save_result(
            result, results_dir=tmp_path, calc_type="geometry_opt", spectra={}
        )
        save_trajectory(saved, result.trajectory, result.energies_hartree)
        save_orbitals(saved, result)
        return saved

    def test_result_json_has_geo_opt_calc_type(self, tmp_path, geo_opt_result):
        saved = save_result(
            geo_opt_result,
            results_dir=tmp_path,
            calc_type="geometry_opt",
            spectra={},
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "geometry_opt"

    def test_trajectory_json_written(self, tmp_path, geo_opt_result):
        saved = self._save_all(tmp_path, geo_opt_result)
        assert (saved / "trajectory.json").exists()

    def test_orbitals_npz_written(self, tmp_path, geo_opt_result):
        saved = self._save_all(tmp_path, geo_opt_result)
        assert (saved / "orbitals.npz").exists()

    def test_trajectory_has_multiple_steps(self, tmp_path, geo_opt_result):
        saved = self._save_all(tmp_path, geo_opt_result)
        raw = json.loads((saved / "trajectory.json").read_text())
        assert len(raw["steps"]) >= 2, "trajectory must have >= 2 steps"


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestGeoOptHistoryContext:
    def _save(self, tmp_path, geo_opt_result):
        return save_result(
            geo_opt_result,
            results_dir=tmp_path,
            calc_type="geometry_opt",
            spectra={},
        )

    def test_context_has_correct_calc_type(self, tmp_path, app, geo_opt_result):
        saved = self._save(tmp_path, geo_opt_result)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "geometry_opt"

    def test_context_result_dir_set(self, tmp_path, app, geo_opt_result):
        saved = self._save(tmp_path, geo_opt_result)
        ctx = app._build_history_context(saved)
        assert ctx.result_dir == saved

    def test_context_live_result_is_none(self, tmp_path, app, geo_opt_result):
        saved = self._save(tmp_path, geo_opt_result)
        ctx = app._build_history_context(saved)
        assert ctx.live_result is None


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestGeoOptPanelActivation:
    def _save_all(self, tmp_path, result):
        saved = save_result(
            result, results_dir=tmp_path, calc_type="geometry_opt", spectra={}
        )
        save_trajectory(saved, result.trajectory, result.energies_hartree)
        save_orbitals(saved, result)
        return saved

    def test_trajectory_panel_activates(self, tmp_path, app, geo_opt_result):
        saved = self._save_all(tmp_path, geo_opt_result)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Trajectory" in app._ana_available

    def test_energies_panel_activates(self, tmp_path, app, geo_opt_result):
        saved = self._save_all(tmp_path, geo_opt_result)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Energies" in app._ana_available

    def test_isosurface_activates(self, tmp_path, app):
        result_with_coeff = _make_geo_opt_result(with_coeff=True)
        saved = self._save_all(tmp_path, result_with_coeff)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Isosurface" in app._ana_available

    def test_trajectory_absent_when_trajectory_json_missing(
        self, tmp_path, app, geo_opt_result
    ):
        # save_result + save_orbitals, but NOT save_trajectory
        saved = save_result(
            geo_opt_result,
            results_dir=tmp_path,
            calc_type="geometry_opt",
            spectra={},
        )
        save_orbitals(saved, geo_opt_result)
        assert not (saved / "trajectory.json").exists()
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Trajectory" not in app._ana_available

    def test_missing_geo_opt_trajectory_message_is_context_aware(
        self, tmp_path, app, geo_opt_result
    ):
        saved = save_result(
            geo_opt_result,
            results_dir=tmp_path,
            calc_type="geometry_opt",
            spectra={},
        )
        save_orbitals(saved, geo_opt_result)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)

        msg = app._ana_unavail_msgs["Trajectory"].value
        assert "Not available for this Geometry Opt history result" in msg
        assert "trajectory.json is missing" in msg
        assert "run a Geometry Opt / PES Scan / Frequency pre-opt" not in msg

    def test_no_panels_when_calc_type_wrong(self, tmp_path, app, geo_opt_result):
        saved = save_result(
            geo_opt_result, results_dir=tmp_path, calc_type="", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert len(app._ana_available) == 0
        assert app._to_analysis_btn.layout.display == "none"

    def test_isosurface_stays_silent_when_orbitals_missing(
        self, tmp_path, app, geo_opt_result, monkeypatch
    ):
        """BUG.8 end-to-end regression for the Geo-Opt path.

        A saved geometry_opt result with trajectory.json but no orbitals.npz
        must NOT raise an AttributeError when the Isosurface populator
        runs. The Trajectory and Energies panels behave per their own data
        availability; what we're asserting here is the absence of
        ana_panel_error events from the Isosurface populator.
        """
        saved = save_result(
            geo_opt_result, results_dir=tmp_path, calc_type="geometry_opt", spectra={}
        )
        save_trajectory(
            saved, geo_opt_result.trajectory, geo_opt_result.energies_hartree
        )
        assert not (saved / "orbitals.npz").exists()

        logged: list[tuple[str, str]] = []

        def _capture(event_type, message, **_extra):
            logged.append((event_type, message))

        monkeypatch.setattr("quantui.calc_log.log_event", _capture)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)

        errors = [m for e, m in logged if e == "ana_panel_error"]
        assert errors == [], (
            f"Unexpected ana_panel_error events on Geo-Opt-without-orbitals "
            f"history replay: {errors}"
        )
        # Trajectory must still activate; Isosurface must not (no orbitals).
        assert "Trajectory" in app._ana_available
        assert "Isosurface" not in app._ana_available


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PYSCF_AVAILABLE or sys.platform == "win32",
    reason="PySCF not available or not supported on native Windows",
)
class TestGeoOptDoRunEndToEnd:
    """Full pipeline: real RHF/STO-3G geometry opt → disk → history → panels."""

    def _run_geo_opt(self, app, tmp_dir, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "Geometry Opt"
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_water())
        return a

    def test_do_run_saves_calc_type_geometry_opt(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_geo_opt(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "geometry_opt"

    def test_do_run_saves_trajectory_json(self, tmp_path, running_app, monkeypatch):
        saved = self._run_geo_opt(running_app, tmp_path, monkeypatch)
        assert saved
        result_dir = saved[0]
        assert (
            result_dir / "trajectory.json"
        ).exists(), "trajectory.json must be written by _do_run for history replay"
        raw = json.loads((result_dir / "trajectory.json").read_text())
        assert raw["atoms"] == ["O", "H", "H"]
        assert len(raw["steps"]) >= 1
        assert any(
            s["energy"] is not None for s in raw["steps"]
        ), "energies must be non-empty in trajectory"

    def test_history_load_activates_trajectory_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_geo_opt(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Trajectory" in running_app._ana_available

    def test_history_load_activates_energies_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_geo_opt(running_app, tmp_path, monkeypatch)
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Energies" in running_app._ana_available

    def test_history_load_activates_isosurface_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_geo_opt(running_app, tmp_path, monkeypatch)
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Isosurface" in running_app._ana_available
