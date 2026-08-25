"""Tests for quantui.nmr_plot."""

from __future__ import annotations

from quantui.nmr_plot import plot_nmr_spectrum


class TestPlotNMRSpectrum:
    def test_h_shifts_produce_stick_trace(self):
        fig = plot_nmr_spectrum(
            [(1, 4.5), (2, 4.6)],
            ["O", "H", "H"],
            nucleus_label="¹H",
        )
        assert len(fig.data) >= 2
        assert fig.layout.xaxis.title.text == "Chemical shift δ (ppm)"

    def test_empty_shifts_returns_axes_only(self):
        fig = plot_nmr_spectrum([], ["C", "H"], nucleus_label="¹³C")
        assert len(fig.data) == 0

    def test_c_shifts_use_wider_default_window(self):
        fig = plot_nmr_spectrum(
            [(0, 180.0)],
            ["C"],
            nucleus_label="¹³C",
        )
        xr = list(fig.layout.xaxis.range)
        assert xr[1] >= 180.0
