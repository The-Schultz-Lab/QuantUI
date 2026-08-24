"""Mulliken population bar chart (Analysis-tab Populations panel).

Accepts per-atom Mulliken charges and returns a Plotly figure. Mirrors the
role of :mod:`quantui.ir_plot` for IR — pure figure construction, no widget
dependency — so the populate path and unit tests can share one builder.
"""

from __future__ import annotations

from typing import List, Sequence

import plotly.graph_objects as go


def plot_mulliken_charges(
    atom_symbols: Sequence[str],
    charges: Sequence[float],
    *,
    height: int = 320,
) -> go.Figure:
    """Return a Plotly bar chart of Mulliken partial charges.

    Bars are coloured by sign (negative = more electron density / red-ish,
    positive = electron deficient / blue-ish) so students can read polarity
    at a glance. Atom labels are 1-based (``O1``, ``H2``, …), matching the
    Measurement panel and NMR table conventions.
    """
    labels: List[str] = [f"{sym}{i + 1}" for i, sym in enumerate(atom_symbols)]
    values = [float(c) for c in charges]
    colors = ["#2563eb" if c >= 0 else "#dc2626" for c in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                hovertemplate="%{x}: %{y:+.4f} e<extra></extra>",
            )
        ],
        layout=dict(
            xaxis=dict(title="Atom", showgrid=False),
            yaxis=dict(
                title="Mulliken charge (e)",
                zeroline=True,
                zerolinewidth=1.5,
                zerolinecolor="#9ca3af",
                showgrid=True,
                gridcolor="#e5e7eb",
            ),
            template="plotly_white",
            showlegend=False,
            margin=dict(l=60, r=20, t=20, b=55),
            height=height,
            plot_bgcolor="#fafafa",
        ),
    )
    return fig
