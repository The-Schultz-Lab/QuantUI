"""D1 — Frequency runs report the force at the Hessian geometry.

Harmonic frequencies and thermochemistry assume a stationary point (zero
gradient). A Frequency run on an unoptimized geometry used to show its
frequencies and thermochemistry as if they were valid.
"""

from __future__ import annotations

import pytest

from quantui.app_formatters import (
    format_freq_result,
    format_past_result,
    stationary_point_rows,
)
from quantui.freq_calc import STATIONARY_FMAX_EV_PER_ANG, FreqResult, is_stationary

try:
    import pyscf  # noqa: F401

    _PYSCF = True
except ImportError:
    _PYSCF = False

pyscf_only = pytest.mark.skipif(not _PYSCF, reason="PySCF not installed")


def _freq(**kw) -> FreqResult:
    base = dict(
        energy_hartree=-76.4,
        homo_lumo_gap_ev=9.5,
        converged=True,
        n_iterations=10,
        method="B3LYP",
        basis="6-31G*",
        formula="H2O",
        frequencies_cm1=[1680.6, 3862.7, 3991.8],
    )
    base.update(kw)
    return FreqResult(**base)


class TestVerdict:
    def test_threshold_is_the_geometry_opt_default(self):
        from quantui.config import DEFAULT_FMAX

        assert STATIONARY_FMAX_EV_PER_ANG == DEFAULT_FMAX

    def test_none_means_not_measured(self):
        assert is_stationary(None) is None
        assert stationary_point_rows(None) == ""

    def test_below_and_above_threshold(self):
        assert is_stationary(0.01) is True
        assert is_stationary(0.67) is False


class TestGradientUnavailable:
    def test_check_is_skipped_not_fatal(self):
        import io

        from quantui.freq_calc import _max_residual_force

        class _NoGrad:
            def nuc_grad_method(self):
                raise NotImplementedError("no gradients for this method")

        log = io.StringIO()
        assert _max_residual_force(_NoGrad(), log, lambda _m: None) is None
        assert "Force check skipped" in log.getvalue()


class TestResultCards:
    def test_live_card_warns_on_unoptimized_geometry(self):
        html = format_freq_result(_freq(max_force_ev_per_ang=0.671))
        assert "0.6710 eV/Å" in html
        assert "not a stationary point" in html
        assert "Geometry optimization before calculation" in html

    def test_live_card_is_quiet_at_a_minimum(self):
        html = format_freq_result(_freq(max_force_ev_per_ang=0.004))
        assert "stationary point" in html
        assert "not a stationary point" not in html
        assert "&#9888;" not in html

    def test_older_result_without_the_check_shows_nothing(self):
        assert "Max force" not in format_freq_result(_freq())

    def test_history_card_reads_the_saved_value(self):
        html = format_past_result(
            {
                "calc_type": "frequency",
                "converged": True,
                "energy_hartree": -76.4,
                "energy_ev": -2078.9,
                "homo_lumo_gap_ev": None,
                "n_iterations": 10,
                "formula": "H2O",
                "method": "B3LYP",
                "basis": "6-31G*",
                "spectra": {"ir": {"max_force_ev_per_ang": 0.671}},
            }
        )
        assert "not a stationary point" in html


class TestThermoPanel:
    def test_analysis_thermo_box_carries_the_warning(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "s.json"))
        from quantui.app import QuantUIApp
        from quantui.app_visualization import show_thermo_box
        from quantui.molecule import Molecule

        app = QuantUIApp()
        water = Molecule(
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        )
        freqs = [1680.6, 3862.7, 3991.8]
        show_thermo_box(app, water, -76.4, freqs, max_force_ev_per_ang=0.671)
        assert "Not a minimum" in app._thermo_html.value
        show_thermo_box(app, water, -76.4, freqs, max_force_ev_per_ang=0.004)
        assert "Not a minimum" not in app._thermo_html.value


@pyscf_only
@pytest.mark.slow
class TestRunFreqCalcForceCheck:
    """Real PySCF. HF/STO-3G H2 has its minimum at R = 0.712 Å."""

    @staticmethod
    def _h2(r: float):
        from quantui.molecule import Molecule

        return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, r]])

    def test_minimum_is_stationary(self):
        import io

        from quantui.freq_calc import run_freq_calc

        r = run_freq_calc(self._h2(0.712), progress_stream=io.StringIO())
        assert r.max_force_ev_per_ang is not None
        assert is_stationary(r.max_force_ev_per_ang) is True

    def test_stretched_bond_is_flagged_and_logged(self):
        import io

        from quantui.freq_calc import run_freq_calc

        log = io.StringIO()
        r = run_freq_calc(self._h2(0.80), progress_stream=log)
        assert r.max_force_ev_per_ang > 1.0
        assert is_stationary(r.max_force_ev_per_ang) is False
        assert "NOT a stationary point" in log.getvalue()
