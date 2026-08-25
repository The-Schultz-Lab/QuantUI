"""NMR chemical-shift stick spectrum (Analysis-tab NMR panel).

Mirrors :mod:`quantui.ir_plot` — pure Plotly figure construction with no widget
dependency so populate paths and unit tests share one builder.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import plotly.graph_objects as go

# Typical display windows (ppm) when peaks are sparse.
_DEFAULT_XRANGE = {
    "¹H": (0.0, 12.0),
    "¹³C": (0.0, 220.0),
}


def plot_nmr_spectrum(
    shifts: Sequence[Tuple[int, float]],
    atom_symbols: Sequence[str],
    *,
    nucleus_label: str = "¹H",
    x_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Return a stick Plotly figure for isotropic chemical shifts (δ, ppm).

    Args:
        shifts: ``(atom_index, δ ppm)`` pairs for one nucleus type.
        atom_symbols: Full molecule atom list (for peak labels).
        nucleus_label: Display label (``"¹H"`` or ``"¹³C"``).
        x_range: Optional fixed ``(xmin, xmax)`` in ppm.
    """
    pairs = [(int(i), float(d)) for i, d in shifts if i < len(atom_symbols)]
    pairs.sort(key=lambda p: p[1])

    if x_range is not None:
        xmin, xmax = float(x_range[0]), float(x_range[1])
    elif pairs:
        pad = 1.0 if nucleus_label == "¹H" else 10.0
        xmin = min(d for _, d in pairs) - pad
        xmax = max(d for _, d in pairs) + pad
        default_lo, default_hi = _DEFAULT_XRANGE.get(nucleus_label, (0.0, 12.0))
        xmin = min(xmin, default_lo)
        xmax = max(xmax, default_hi)
    else:
        xmin, xmax = _DEFAULT_XRANGE.get(nucleus_label, (0.0, 12.0))

    if xmin >= xmax:
        xmin, xmax = xmax - 1.0, xmax

    fig = go.Figure(
        layout=dict(
            xaxis=dict(
                title="Chemical shift δ (ppm)",
                range=[xmin, xmax],
                showgrid=True,
                gridcolor="#e5e7eb",
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0.0, 1.2],
            ),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=60, r=20, t=20, b=55),
            height=300,
            plot_bgcolor="#fafafa",
        )
    )

    if not pairs:
        return fig

    stick_x: List[Optional[float]] = []
    stick_y: List[Optional[float]] = []
    markers_x: List[float] = []
    markers_y: List[float] = []
    hover_labels: List[str] = []

    for idx, delta in pairs:
        sym = atom_symbols[idx]
        label = f"{sym}{idx + 1}"
        stick_x.extend([delta, delta, None])
        stick_y.extend([0.0, 1.0, None])
        markers_x.append(delta)
        markers_y.append(1.0)
        hover_labels.append(label)

    fig.add_trace(
        go.Scatter(
            x=stick_x,
            y=stick_y,
            mode="lines",
            line=dict(color="#0d9488", width=2),
            name=f"{nucleus_label} (stick)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=markers_x,
            y=markers_y,
            mode="markers+text",
            text=hover_labels,
            textposition="top center",
            marker=dict(color="#0f766e", size=7),
            showlegend=False,
            hovertemplate=(
                "%{text}<br>δ = %{x:.2f} ppm (vs. reference)<extra></extra>"
            ),
        )
    )
    return fig
