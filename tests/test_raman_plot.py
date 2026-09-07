"""Tests for quantui.raman_plot."""

from __future__ import annotations

from quantui.raman_plot import plot_raman_spectrum


def test_empty_frequencies_returns_blank_figure():
    fig = plot_raman_spectrum([], [])
    assert fig.data == ()


def test_stick_mode_has_marker_trace():
    fig = plot_raman_spectrum([1600.0, 3600.0], [10.0, 50.0], mode="stick")
    assert len(fig.data) == 2
    assert fig.data[1].mode == "markers"


def test_broadened_mode_single_trace():
    fig = plot_raman_spectrum([1600.0], [10.0], mode="broadened", fwhm=20.0)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"


def test_skips_imaginary_frequencies():
    fig = plot_raman_spectrum([-100.0, 1500.0], [5.0, 20.0], mode="stick")
    assert len(fig.data) == 2
    assert fig.data[1].x == (1500.0,)


def test_range_widens_for_a_high_frequency_mode():
    """AUDIT additional-concerns — mirrors test_ir_plot.py: a real O-H
    stretch above 4000 cm⁻¹ must not be clipped off the fixed default
    range."""
    freqs = [1785.6, 4486.7, 4788.3]
    acts = [2.0, 30.0, 15.0]
    fig = plot_raman_spectrum(freqs, acts, mode="stick")
    x_range = list(fig.layout.xaxis.range)
    assert x_range[1] > 4788.3
    assert 4788.3 in fig.data[1].x


def test_broadened_grid_widens_for_a_high_frequency_mode():
    import numpy as np

    freqs = [1785.6, 4486.7, 4788.3]
    acts = [2.0, 30.0, 15.0]
    fig = plot_raman_spectrum(freqs, acts, mode="broadened", fwhm=20.0)
    x = np.array(fig.data[0].x)
    assert x.max() > 4788.3
