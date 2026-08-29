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
