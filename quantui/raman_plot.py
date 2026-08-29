"""
Raman spectrum visualization: stick chart and Lorentzian broadened lineshape.

Accepts vibrational frequencies (cm⁻¹) and Raman activities (Å⁴/amu)
from a frequency calculation and returns a Plotly Figure.

Typical usage::

    from quantui.raman_plot import plot_raman_spectrum
    fig = plot_raman_spectrum(result.frequencies_cm1, result.raman_activities)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from quantui.ir_plot import _XGRID, _XRANGE


def plot_raman_spectrum(
    frequencies: List[float],
    activities: List[float],
    *,
    fwhm: float = 20.0,
    mode: str = "stick",
    yaxis_title: str = "Raman Activity (Å⁴/amu)",
) -> go.Figure:
    """Return a Plotly figure for the Raman scattering spectrum."""
    real_pairs = [(f, a) for f, a in zip(frequencies, activities) if f > 0]

    _base_layout = dict(
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            range=_XRANGE,
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
        plot_bgcolor="#fafafa",
    )

    fig = go.Figure(layout=_base_layout)

    if not real_pairs:
        return fig

    freqs_real, acts_real = zip(*real_pairs)

    if mode == "broadened":
        half_gamma = fwhm / 2.0
        y_broad = np.zeros_like(_XGRID)
        for nu0, act in zip(freqs_real, acts_real):
            y_broad += act * half_gamma**2 / ((_XGRID - nu0) ** 2 + half_gamma**2)

        fig.add_trace(
            go.Scatter(
                x=_XGRID,
                y=y_broad,
                mode="lines",
                line=dict(color="#059669", width=1.5),
                name="Raman (broadened)",
                hovertemplate="%{x:.0f} cm⁻¹ | %{y:.2f} Å⁴/amu<extra></extra>",
            )
        )
    else:
        x_stick: List[Optional[float]] = []
        y_stick: List[Optional[float]] = []
        for nu, act in zip(freqs_real, acts_real):
            x_stick.extend([nu, nu, None])
            y_stick.extend([0.0, act, None])

        fig.add_trace(
            go.Scatter(
                x=x_stick,
                y=y_stick,
                mode="lines",
                line=dict(color="#059669", width=2),
                name="Raman (stick)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(freqs_real),
                y=list(acts_real),
                mode="markers",
                marker=dict(color="#047857", size=6),
                name="Raman (peaks)",
                showlegend=False,
                hovertemplate=(
                    "Wavenumber: %{x:.1f} cm⁻¹"
                    "<br>Activity: %{y:.2f} Å⁴/amu<extra></extra>"
                ),
            )
        )

    return fig
