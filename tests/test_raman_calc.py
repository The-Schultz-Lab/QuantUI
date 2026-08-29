"""Tests for quantui.raman_calc — Raman activity invariants and integration."""

from __future__ import annotations

import numpy as np
import pytest

from quantui.raman_calc import _raman_invariants, raman_enabled


class TestRamanInvariants:
    def test_isotropic_mode_gives_45_alpha_bar_squared(self):
        # ∂α/∂Q proportional to identity → γ² = 0
        da = np.eye(3) * 3.0
        assert _raman_invariants(da) == pytest.approx(45.0 * 3.0**2)

    def test_traceless_anisotropic_contributes_via_gamma(self):
        da = np.diag([1.0, -1.0, 0.0])
        alpha_bar = 0.0
        gamma2 = 0.5 * ((1 - (-1)) ** 2 + ((-1) - 0) ** 2 + (0 - 1) ** 2)
        expected = 45 * alpha_bar**2 + 7 * gamma2
        assert _raman_invariants(da) == pytest.approx(expected)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            da = rng.normal(size=(3, 3))
            da = 0.5 * (da + da.T)
            assert _raman_invariants(da) >= 0.0


class TestRamanEnabled:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("QUANTUI_RAMAN", raising=False)
        assert raman_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off"])
    def test_opt_out_values(self, monkeypatch, val):
        monkeypatch.setenv("QUANTUI_RAMAN", val)
        assert raman_enabled() is False


# ---------------------------------------------------------------------------
# Integration — PySCF required
# ---------------------------------------------------------------------------

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except ImportError:
    _HAS_PYSCF = False

pyscf_only = pytest.mark.skipif(not _HAS_PYSCF, reason="PySCF not installed")


def _water():
    from quantui.molecule import Molecule

    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.96, 0.0]],
        charge=0,
        multiplicity=1,
    )


class TestRamanActivitiesIntegration:
    """H₂O / RHF / STO-3G — Raman should ship alongside IR on Frequency runs."""

    @pyscf_only
    @pytest.mark.slow
    def test_raman_activities_non_empty(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.raman_activities, "raman_activities should be non-empty for H₂O/RHF"

    @pyscf_only
    @pytest.mark.slow
    def test_raman_length_matches_frequencies(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert len(result.raman_activities) == len(result.frequencies_cm1)

    @pyscf_only
    @pytest.mark.slow
    def test_raman_opt_out(self, monkeypatch):
        monkeypatch.setenv("QUANTUI_RAMAN", "0")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        assert result.raman_activities == []

    @pyscf_only
    @pytest.mark.slow
    def test_h2o_bend_weaker_ir_than_raman(self, monkeypatch):
        """Teaching check: lowest H₂O mode is weak IR, strong Raman at STO-3G."""
        monkeypatch.setenv("QUANTUI_RAMAN", "1")
        from quantui.freq_calc import run_freq_calc

        result = run_freq_calc(_water(), method="RHF", basis="STO-3G")
        pairs = [
            (f, ir, ram)
            for f, ir, ram in zip(
                result.frequencies_cm1,
                result.ir_intensities,
                result.raman_activities,
            )
            if f > 0
        ]
        assert pairs
        lowest = min(pairs, key=lambda t: t[0])
        _freq, ir_low, ram_low = lowest
        assert ir_low < ram_low, (
            f"Expected bend-like mode IR ({ir_low:.2f}) < Raman ({ram_low:.2f})"
        )
