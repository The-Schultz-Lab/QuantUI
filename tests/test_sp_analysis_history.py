"""Integration tests for the single-point analysis history roundtrip.

Covers the complete path from "calculation finishes" to "history panels activate":

  (1) Spectra structure  — save_result stores result.json with calc_type
                           "single_point"; save_orbitals writes orbitals.npz.
  (2) History context    — _build_history_context loads calc_type correctly
                           from disk.
  (3) Panel activation   — _apply_analysis_context activates Energies and
                           Isosurface panels from a history context.
  (4) _do_run end-to-end — Full path: real RHF/STO-3G run → disk write →
                           _history_load_analysis → panels activate.
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


def _make_sp_result(with_coeff: bool = True):
    """Minimal namespace mirroring a real RHF/STO-3G result on water.

    7 MOs (matching STO-3G water: 5 occupied + 2 virtual) so that
    orbital_info_from_arrays does not raise on the n_occ >= n_total check.
    """
    mo_coeff = np.eye(7) if with_coeff else None
    return SimpleNamespace(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        energy_ev=-2040.5,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        mo_energy_hartree=np.array([-20.5, -1.3, -0.7, -0.5, -0.3, 0.5, 0.7]),
        mo_occ=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
        mo_coeff=mo_coeff,
        pyscf_mol_atom=[
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.757, 0.586, 0.0]),
            ("H", [-0.757, 0.586, 0.0]),
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
def sp_result():
    return _make_sp_result()


# ---------------------------------------------------------------------------
# Part 1: save_result + save_orbitals write the correct files
# ---------------------------------------------------------------------------


class TestSPSpectraStructure:
    def test_result_json_has_single_point_calc_type(self, tmp_path, sp_result):
        saved = save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        data = json.loads((saved / "result.json").read_text())
        assert data["calc_type"] == "single_point"

    def test_save_orbitals_writes_npz(self, tmp_path, sp_result):
        saved = save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        save_orbitals(saved, sp_result)
        assert (saved / "orbitals.npz").exists()

    def test_no_spectra_keys_for_single_point(self, tmp_path, sp_result):
        saved = save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        data = json.loads((saved / "result.json").read_text())
        assert (
            data.get("spectra", {}) == {}
        ), "single_point produces no spectra data — only orbitals.npz"


# ---------------------------------------------------------------------------
# Part 2: _build_history_context reconstructs the context correctly
# ---------------------------------------------------------------------------


class TestSPHistoryContext:
    def _save(self, tmp_path, sp_result):
        return save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )

    def test_context_has_correct_calc_type(self, tmp_path, app, sp_result):
        saved = self._save(tmp_path, sp_result)
        ctx = app._build_history_context(saved)
        assert ctx is not None
        assert ctx.calc_type == "single_point"

    def test_context_result_dir_set(self, tmp_path, app, sp_result):
        saved = self._save(tmp_path, sp_result)
        ctx = app._build_history_context(saved)
        assert ctx.result_dir == saved

    def test_context_live_result_is_none(self, tmp_path, app, sp_result):
        saved = self._save(tmp_path, sp_result)
        ctx = app._build_history_context(saved)
        assert ctx.live_result is None


# ---------------------------------------------------------------------------
# Part 3: _apply_analysis_context activates the correct panels
# ---------------------------------------------------------------------------


class TestSPPanelActivation:
    def _save_with_orbitals(self, tmp_path, result):
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        save_orbitals(saved, result)
        return saved

    def test_energies_panel_activates_with_orbitals(self, tmp_path, app, sp_result):
        saved = self._save_with_orbitals(tmp_path, sp_result)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Energies" in app._ana_available

    def test_energies_absent_when_orbitals_missing(self, tmp_path, app, sp_result):
        # save_result only — no save_orbitals call, so orbitals.npz is absent
        saved = save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        assert not (saved / "orbitals.npz").exists()
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Energies" not in app._ana_available

    def test_isosurface_activates_when_mo_coeff_present(self, tmp_path, app):
        result_with_coeff = _make_sp_result(with_coeff=True)
        saved = self._save_with_orbitals(tmp_path, result_with_coeff)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Isosurface" in app._ana_available

    def test_no_panels_when_calc_type_wrong(self, tmp_path, app, sp_result):
        saved = save_result(sp_result, results_dir=tmp_path, calc_type="", spectra={})
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert len(app._ana_available) == 0
        assert app._to_analysis_btn.layout.display == "none"

    def test_isosurface_stays_silent_when_orbitals_missing(
        self, tmp_path, app, sp_result, monkeypatch
    ):
        """BUG.8 end-to-end regression: a saved SP result without
        ``orbitals.npz`` on disk must NOT raise an AttributeError when the
        Isosurface populator runs. The unit test (TestPopIsosurfaceBug8)
        covers the bare populator; this exercises the full save → load →
        apply path on disk so we catch any future regression in the
        Energies-or-Isosurface ordering, the populator-loop exception
        handling, or the __init__ + apply_analysis_context reset pair.
        """
        # Save WITHOUT save_orbitals → no orbitals.npz on disk.
        saved = save_result(
            sp_result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        assert not (saved / "orbitals.npz").exists()

        # Capture any ana_panel_error events that get logged by the
        # registry loop. Direct-patch log_event so the assertion is precise.
        logged: list[tuple[str, str]] = []

        def _capture(event_type, message, **_extra):
            logged.append((event_type, message))

        monkeypatch.setattr("quantui.calc_log.log_event", _capture)
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)

        # No ana_panel_error from pop_isosurface — the defensive getattr +
        # __init__ initialization combo prevents AttributeError.
        errors = [m for e, m in logged if e == "ana_panel_error"]
        assert errors == [], (
            f"Unexpected ana_panel_error events on SP-without-orbitals "
            f"history replay: {errors}"
        )
        # Both Energies and Isosurface stay unavailable (no orbitals).
        assert "Energies" not in app._ana_available
        assert "Isosurface" not in app._ana_available


# ---------------------------------------------------------------------------
# Part 4: _do_run end-to-end (PySCF-gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PYSCF_AVAILABLE or sys.platform == "win32",
    reason="PySCF not available or not supported on native Windows",
)
class TestSPDoRunEndToEnd:
    """Full pipeline: real RHF/STO-3G single point → disk → history → panels."""

    def _run_sp(self, app, tmp_dir, monkeypatch):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_dir))
        app.calc_type_dd.value = "Single Point"
        app._do_run()
        return list_results(tmp_dir)

    @pytest.fixture
    def running_app(self):
        a = QuantUIApp()
        a._set_molecule(_water())
        return a

    def test_do_run_saves_calc_type_single_point(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_sp(running_app, tmp_path, monkeypatch)
        assert saved, "No result saved to disk"
        data = load_result(saved[0])
        assert data["calc_type"] == "single_point"

    def test_do_run_saves_orbitals_npz(self, tmp_path, running_app, monkeypatch):
        saved = self._run_sp(running_app, tmp_path, monkeypatch)
        assert saved
        assert (
            saved[0] / "orbitals.npz"
        ).exists(), "orbitals.npz must be written by _do_run for history replay"

    def test_history_load_activates_energies_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_sp(running_app, tmp_path, monkeypatch)
        assert saved
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Energies" in running_app._ana_available

    def test_history_load_activates_isosurface_panel(
        self, tmp_path, running_app, monkeypatch
    ):
        saved = self._run_sp(running_app, tmp_path, monkeypatch)
        running_app._deactivate_all_ana_panels()
        running_app._history_load_analysis(saved[0])
        assert "Isosurface" in running_app._ana_available
