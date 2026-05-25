"""
Tests for quantui.ir_plot — M-IR.1 acceptance criteria.
All tests run unconditionally (no PySCF required).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from quantui.ir_plot import plot_ir_spectrum

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_FREQS = [1000.0, 2000.0, 3000.0]
_SIMPLE_INTS = [10.0, 50.0, 5.0]


# ---------------------------------------------------------------------------
# Stick mode
# ---------------------------------------------------------------------------


class TestStickMode:
    def test_returns_figure(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        assert isinstance(fig, go.Figure)

    def test_has_lines_and_markers_traces(self):
        # Stick mode renders two traces: vertical lines + marker dots at
        # each stick tip (the dots provide a hover target, mirroring the
        # UV-Vis spectrum affordance).
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        assert len(fig.data) == 2
        assert fig.data[0].mode == "lines"
        assert fig.data[1].mode == "markers"

    def test_trace_is_scatter(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        assert isinstance(fig.data[0], go.Scatter)
        assert isinstance(fig.data[1], go.Scatter)

    def test_xaxis_low_to_high(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        x_range = list(fig.layout.xaxis.range)
        assert x_range[0] < x_range[1], "x-axis must run low → high (400 → 4000)"

    def test_xaxis_title_contains_wavenumber(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        assert "cm" in fig.layout.xaxis.title.text.lower()

    def test_yaxis_title_contains_intensity(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS)
        assert "intensity" in fig.layout.yaxis.title.text.lower()

    def test_negative_frequencies_skipped(self):
        freqs = [-200.0, 1000.0, 2000.0]
        ints = [99.0, 10.0, 20.0]
        fig = plot_ir_spectrum(freqs, ints)
        x_data = [x for x in fig.data[0].x if x is not None]
        assert all(x > 0 for x in x_data)

    def test_zero_frequency_skipped(self):
        freqs = [0.0, 1500.0]
        ints = [100.0, 30.0]
        fig = plot_ir_spectrum(freqs, ints)
        x_data = [x for x in fig.data[0].x if x is not None]
        assert 0.0 not in x_data


# ---------------------------------------------------------------------------
# Broadened mode
# ---------------------------------------------------------------------------


class TestBroadenedMode:
    def test_returns_figure(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS, mode="broadened")
        assert isinstance(fig, go.Figure)

    def test_has_one_trace(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS, mode="broadened")
        assert len(fig.data) == 1

    def test_trace_covers_full_grid(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS, mode="broadened")
        x = np.array(fig.data[0].x)
        assert x.min() >= 400
        assert x.max() <= 4000

    def test_peak_at_correct_frequency(self):
        freqs = [2000.0]
        ints = [1.0]
        fig = plot_ir_spectrum(freqs, ints, mode="broadened", fwhm=10.0)
        x = np.array(fig.data[0].x)
        y = np.array(fig.data[0].y)
        peak_x = float(x[np.argmax(y)])
        assert abs(peak_x - 2000.0) < 2.0, f"Peak at {peak_x:.1f}, expected ~2000"

    def test_fwhm_affects_peak_width(self):
        freqs = [1500.0]
        ints = [1.0]
        fig_narrow = plot_ir_spectrum(freqs, ints, mode="broadened", fwhm=5.0)
        fig_wide = plot_ir_spectrum(freqs, ints, mode="broadened", fwhm=80.0)
        y_narrow = np.array(fig_narrow.data[0].y)
        y_wide = np.array(fig_wide.data[0].y)
        # Lorentzian peak height is always I at ν₀; wider FWHM means larger integrated area
        assert (
            y_wide.sum() > y_narrow.sum()
        ), "Wider FWHM should produce larger integrated area"

    def test_xaxis_low_to_high_broadened(self):
        fig = plot_ir_spectrum(_SIMPLE_FREQS, _SIMPLE_INTS, mode="broadened")
        x_range = list(fig.layout.xaxis.range)
        assert x_range[0] < x_range[1]


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_frequencies_no_exception(self):
        fig = plot_ir_spectrum([], [])
        assert isinstance(fig, go.Figure)

    def test_empty_frequencies_no_traces(self):
        fig = plot_ir_spectrum([], [])
        assert len(fig.data) == 0

    def test_all_negative_frequencies_no_traces(self):
        fig = plot_ir_spectrum([-100.0, -200.0], [5.0, 10.0])
        assert len(fig.data) == 0

    def test_empty_broadened_no_exception(self):
        fig = plot_ir_spectrum([], [], mode="broadened")
        assert isinstance(fig, go.Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
