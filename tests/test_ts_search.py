"""Transition State calc type (M-TS TS.2, DEC-024: Sella)."""

from __future__ import annotations

import io
import sys

import pytest

from quantui.molecule import Molecule
from quantui.ts_search import TSResult, classify_stationary_point

HCN_GUESS = Molecule(["C", "N", "H"], [[0, 0, 0], [0, 0, 1.18], [1.05, 0, 0.55]])


class TestVerdict:
    def test_one_imaginary_is_a_ts(self):
        n, imag, verdict = classify_stationary_point([-1246.0, 2100.0, 2500.0])
        assert n == 1 and imag == [-1246.0]
        assert verdict.startswith("Transition state") and "1246i" in verdict

    def test_none_is_a_minimum(self):
        n, _, verdict = classify_stationary_point([700.0, 2100.0, 3300.0])
        assert n == 0 and "minimum" in verdict

    def test_two_is_higher_order(self):
        n, imag, verdict = classify_stationary_point([-900.0, -300.0, 2000.0])
        assert n == 2 and imag == [-900.0, -300.0]
        assert "higher-order saddle" in verdict

    def test_small_values_are_noise(self):
        n, _, verdict = classify_stationary_point([-1246.0, -12.0, 2000.0])
        assert n == 1 and "numerical noise" in verdict


class TestResultObject:
    def test_spectra_block_and_frequency_delegation(self):
        from types import SimpleNamespace

        freq = SimpleNamespace(frequencies_cm1=[-1.0e3, 2.0e3], mulliken_charges=[0.1])
        r = TSResult(
            formula="CHN",
            method="RHF",
            basis="STO-3G",
            molecule=HCN_GUESS,
            energy_hartree=-91.0,
            converged=True,
            search_converged=True,
            n_steps=4,
            n_imaginary=1,
            imaginary_cm1=[-1.0e3],
            verdict="Transition state",
            freq=freq,
        )
        assert r.is_transition_state
        assert r.frequencies_cm1 == [-1.0e3, 2.0e3]
        assert r.mulliken_charges == [0.1]
        assert r.to_spectra() == {
            "search_converged": True,
            "n_steps": 4,
            "n_imaginary": 1,
            "imaginary_cm1": [-1000.0],
            "verdict": "Transition state",
        }
        with pytest.raises(AttributeError):
            _ = r.not_a_field


class TestSearch:
    def test_hcn_hnc_transition_state(self):
        pytest.importorskip("sella")
        pytest.importorskip("pyscf")
        from quantui.ts_search import run_ts_search

        log = io.StringIO()
        r = run_ts_search(HCN_GUESS, "RHF", "STO-3G", progress_stream=log)
        assert r.search_converged and r.n_imaginary == 1
        assert r.is_transition_state
        # The imaginary mode is large (H migrating between C and N).
        assert r.imaginary_cm1[0] < -800
        # The search path is recorded, starting at the guess.
        assert len(r.trajectory) == len(r.energies_hartree) == r.n_steps + 1
        assert r.trajectory[0].coordinates[2] == pytest.approx([1.05, 0, 0.55])
        assert "Transition state" in log.getvalue()
        assert r.mulliken_charges is not None and r.mo_coeff is not None

    def test_post_hf_is_refused(self):
        from quantui.ts_search import run_ts_search

        with pytest.raises(ValueError, match="analytic gradients"):
            run_ts_search(HCN_GUESS, "MP2", "STO-3G")

    def test_missing_sella_says_how_to_install(self, monkeypatch):
        from quantui import ts_search

        monkeypatch.setitem(sys.modules, "sella", None)
        assert not ts_search.sella_available()
        with pytest.raises(ImportError, match=r"quantui\[ts\]"):
            ts_search.run_ts_search(HCN_GUESS, "RHF", "STO-3G")


class TestApp:
    def test_calc_type_panel(self):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        assert "Transition State" in app.calc_type_dd.options
        app.calc_type_dd.value = "Transition State"
        assert app._freq_preopt_cb.layout.display == "none"
        assert app._freq_preopt_cb.value is False
        kids = app.calc_extra_opts.children
        assert app._ts_note in kids
        assert app._ts_fmax_fi in kids[0].children
        assert app._ts_fmax_fi.value == pytest.approx(0.01)

    def test_slurm_refuses_it_for_now(self):
        from quantui.app_slurm import _SUPPORTED_SLURM_CALC_TYPES

        assert "transition_state" not in _SUPPORTED_SLURM_CALC_TYPES

    def test_run_save_and_history(self, tmp_path, monkeypatch):
        pytest.importorskip("sella")
        pytest.importorskip("pyscf")
        from quantui import load_result
        from quantui.app import QuantUIApp
        from quantui.app_formatters import format_past_result

        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        app._set_molecule(HCN_GUESS, "guess")
        app.calc_type_dd.value = "Transition State"
        app._do_run()
        assert app._last_calc_type == "transition_state"
        assert {"Vibrational", "IR Spectrum", "Trajectory", "Populations"} <= set(
            app._ana_available
        )
        saved = app._last_result_dir
        for name in ("trajectory.json", "orbitals.npz", "result.molden"):
            assert (saved / name).exists(), name
        data = load_result(saved)
        assert data["calc_type"] == "transition_state"
        block = data["spectra"]["transition_state"]
        assert block["n_imaginary"] == 1
        # The saved geometry is the stationary point, not the guess.
        assert data["geometry"]["coordinates"][2] != pytest.approx([1.05, 0, 0.55])
        card = format_past_result(data, saved)
        assert "Transition State" in card and "✓ Transition state" in card

        fresh = QuantUIApp()
        fresh._apply_analysis_context(fresh._build_history_context(saved))
        assert {"Vibrational", "Trajectory", "Isosurface"} <= set(fresh._ana_available)

    def test_export_script_says_scf_only(self, tmp_path):
        from quantui.calculator import PySCFCalculation

        src = PySCFCalculation(
            HCN_GUESS, method="RHF", basis="STO-3G"
        ).generate_calculation_script(tmp_path / "ts.py", calc_type="transition_state")
        assert "QuantUI ran a Transition State calculation" in src
