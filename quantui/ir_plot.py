"""
IR spectrum visualization: stick chart and Lorentzian broadened lineshape.

Accepts vibrational frequencies (cm⁻¹) and IR intensities (km/mol)
from a frequency calculation and returns a Plotly Figure.

Typical usage::

    from quantui.ir_plot import plot_ir_spectrum
    fig = plot_ir_spectrum(result.frequencies_cm1, result.ir_intensities)
    fig = plot_ir_spectrum(freqs, intensities, mode="broadened", fwhm=30.0)

AUDIT additional-concerns — broadened-mode normalization
----------------------------------------------------------
The "broadened" Lorentzian kernel below is HEIGHT-normalized: each peak's
value at its own center equals the supplied intensity, exactly like the
stick plot, so switching FWHM only changes peak width, never peak height
— the same convention used by Gaussian/ORCA-style broadened spectra and
by :mod:`quantui.raman_plot` and the UV-Vis broadening in
app_visualization.py. This is deliberate, NOT a bug: it is not an
area-normalized spectral density, so the AREA under a broadened peak
scales with FWHM (∫ γ²/((x-x0)²+γ²) dx = πγ) even though the peak height
does not. A true physical spectral-density plot (area proportional to
the supplied intensity, independent of the chosen FWHM) would need an
area-normalized Lorentzian (dividing by πγ) and different y-axis units —
intentionally out of scope here, since the height-preserving convention
is what students and most external QM software display by default.
"""

from __future__ import annotations

from typing import List, Optional, cast

import numpy as np
import plotly.graph_objects as go

# x-axis range is low → high wavenumber (user-facing convention in QuantUI).
# AUDIT additional-concerns — this used to be the FIXED plot range/grid
# regardless of the actual data: a real water/STO-3G calculation has O-H
# stretches at 4486.7/4788.3 cm⁻¹, which used to be clipped off the right
# edge in stick mode (Plotly's xaxis.range) and never even entered the
# broadened kernel (evaluated only on this fixed grid) — computed modes
# silently invisible. ``_default_xrange``/``_grid_for_range`` below widen
# this default window to always cover every real (positive) frequency
# actually present, while leaving the familiar 400–4000 cm⁻¹ look
# untouched for the common case where every mode already falls inside it.
_DEFAULT_XRANGE = [400, 4000]


def _default_xrange(freqs_real: tuple) -> List[float]:
    """[xmin, xmax] covering 400–4000 cm⁻¹ AND every real frequency present.

    A fixed margin keeps a peak sitting exactly at the edge from being
    clipped by the axis border or cut off mid-lineshape in broadened mode.
    """
    if not freqs_real:
        return [float(_DEFAULT_XRANGE[0]), float(_DEFAULT_XRANGE[1])]
    margin = 100.0
    lo = min(_DEFAULT_XRANGE[0], min(freqs_real) - margin)
    hi = max(_DEFAULT_XRANGE[1], max(freqs_real) + margin)
    return [float(lo), float(hi)]


def _grid_for_range(xrange: List[float]) -> np.ndarray:
    # numpy's stubs (as pinned: mypy~=1.10.0) resolve this call to `Any`
    # rather than `ndarray[Any, dtype[Any]]` for non-literal float bounds
    # — pyscf has no type stubs (ignore_missing_imports) for the same
    # underlying reason elsewhere in this codebase, and the fix there is
    # the same: an explicit cast documents the real, known return type
    # instead of silencing the check.
    return cast(np.ndarray, np.arange(xrange[0], xrange[1] + 1.0, 1.0))


def plot_ir_spectrum(
    frequencies: List[float],
    intensities: List[float],
    *,
    fwhm: float = 20.0,
    mode: str = "stick",
    yaxis_title: str = "IR Intensity (km/mol)",
) -> go.Figure:
    """Return a Plotly figure for the IR absorption spectrum.

    Args:
        frequencies: Vibrational frequencies in cm⁻¹.
            Values ≤ 0 (imaginary / translation / rotation) are silently skipped.
        intensities: IR intensities in km/mol, same length as *frequencies*.
        fwhm: Full width at half maximum for the Lorentzian lineshape in cm⁻¹.
            Only used when ``mode="broadened"``. Default: 20.
        mode: Display mode.
            ``"stick"`` — vertical bars at each active frequency.
            ``"broadened"`` — Lorentzian convolution of all peaks.

    Returns:
        :class:`plotly.graph_objects.Figure` ready for display or wrapping
        in a :class:`~plotly.graph_objects.FigureWidget`.
    """
    real_pairs = [(f, i) for f, i in zip(frequencies, intensities) if f > 0]
    freqs_real_for_range = tuple(f for f, _ in real_pairs)
    xrange = _default_xrange(freqs_real_for_range)

    _base_layout = dict(
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            range=xrange,
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(
            title=yaxis_title,
            rangemode="tozero",
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=20, b=55),
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig = go.Figure(layout=_base_layout)

    if not real_pairs:
        return fig

    freqs_real, ints_real = zip(*real_pairs)

    if mode == "broadened":
        _xgrid = _grid_for_range(xrange)
        half_gamma = fwhm / 2.0
        y_broad = np.zeros_like(_xgrid)
        for nu0, inten in zip(freqs_real, ints_real):
            y_broad += inten * half_gamma**2 / ((_xgrid - nu0) ** 2 + half_gamma**2)

        fig.add_trace(
            go.Scatter(
                x=_xgrid,
                y=y_broad,
                mode="lines",
                line=dict(color="#2563eb", width=1.5),
                name="IR (broadened)",
                hovertemplate="%{x:.0f} cm⁻¹ | %{y:.2f} km/mol<extra></extra>",
            )
        )
    else:  # stick
        x_stick: List[Optional[float]] = []
        y_stick: List[Optional[float]] = []
        for nu, inten in zip(freqs_real, ints_real):
            x_stick.extend([nu, nu, None])
            y_stick.extend([0.0, inten, None])

        fig.add_trace(
            go.Scatter(
                x=x_stick,
                y=y_stick,
                mode="lines",
                line=dict(color="#2563eb", width=2),
                name="IR (stick)",
                hoverinfo="skip",
            )
        )
        # Marker dots at each stick tip — matches the UV-Vis spectrum
        # affordance and gives users a hover-target that surfaces the
        # exact frequency / intensity for each mode.
        fig.add_trace(
            go.Scatter(
                x=list(freqs_real),
                y=list(ints_real),
                mode="markers",
                marker=dict(color="#1d4ed8", size=6),
                name="IR (peaks)",
                showlegend=False,
                hovertemplate=(
                    "Wavenumber: %{x:.1f} cm⁻¹"
                    "<br>Intensity: %{y:.2f} km/mol<extra></extra>"
                ),
            )
        )

    return fig
