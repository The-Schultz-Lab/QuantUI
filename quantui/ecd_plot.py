"""
ECD (electronic circular dichroism) spectrum: Δε broadening and Plotly chart.

Takes TD-DFT excitation energies (eV) and rotatory strengths R_n (in the
standard unit 10⁻⁴⁰ esu²·cm², computed by
:func:`quantui.tddft_calc.rotatory_strengths`) and builds either a stick
spectrum of R_n or a Gaussian-broadened Δε spectrum.

Broadening convention (Stephens & Harada, *Chirality* 2010, 22, 229; the
same convention SpecDis and most ECD papers use)
------------------------------------------------------------------------
Each transition contributes a Gaussian **in energy**::

    Δε(E) = 1 / (2.296×10⁻³⁹ · √π · σ) · Σ_n ΔE_n · R_n · exp(−((E − ΔE_n)/σ)²)

with E, ΔE_n and σ in eV, R_n in esu²·cm² and Δε in L mol⁻¹ cm⁻¹. σ is the
half-width at 1/e of the peak height (FWHM = 2√(ln 2)·σ ≈ 1.665 σ).

The constant comes from the definition of the rotatory strength as the
integrated band intensity,

    R_n = (3 h c 10³ ln 10 / (32 π³ N_A)) ∫ Δε(ν̃)/ν̃ dν̃
        = 2.2965×10⁻³⁹ ∫ Δε/E dE,

so the curve is *area-normalized*: ∫ Δε/E dE over one (narrow) band gives
R_n / 2.296×10⁻³⁹ whatever σ is. Unlike the height-normalized UV-Vis/IR
broadening elsewhere in QuantUI, the Δε axis therefore carries real
molar-absorptivity units, and wider bands are lower. Positive Δε means the
molecule absorbs left-circularly-polarized light more strongly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# h·c in eV·nm — λ(nm) = 1239.84 / E(eV)
EV_NM: float = 1239.84193

# R = ECD_INTEGRAL_CONST · ∫ Δε/E dE   (R in esu²·cm², Δε in L mol⁻¹ cm⁻¹).
# 3·h·c·1000·ln(10) / (32·π³·N_A) with CODATA 2018 constants in cgs.
ECD_INTEGRAL_CONST: float = 2.2965e-39

# Default Gaussian half-width at 1/e (eV). 0.2–0.4 eV is the usual range for
# comparing computed ECD with solution spectra.
DEFAULT_SIGMA_EV: float = 0.3

# |R| below this (in 10⁻⁴⁰ cgs) for every state ⇒ the molecule is achiral and
# its computed R are numerical noise (an achiral molecule gives ~1e-10 here;
# real chiral transitions are typically 0.1–100).
ACHIRAL_R_THRESHOLD: float = 1e-3


def ecd_delta_epsilon(
    energies_ev: Sequence[float],
    rotatory_strengths_1e40: Sequence[float],
    e_grid_ev: Union[Sequence[float], np.ndarray],
    sigma_ev: float = DEFAULT_SIGMA_EV,
) -> np.ndarray:
    """Gaussian-broadened ECD spectrum Δε(E) in L mol⁻¹ cm⁻¹.

    Args:
        energies_ev: Excitation energies ΔE_n (eV).
        rotatory_strengths_1e40: R_n in units of 10⁻⁴⁰ esu²·cm².
        e_grid_ev: Photon energies (eV) at which to evaluate Δε.
        sigma_ev: Half-width at 1/e of each Gaussian (eV), > 0.

    Returns:
        Δε on *e_grid_ev* (same shape).
    """
    if sigma_ev <= 0:
        raise ValueError("sigma_ev must be positive")
    e = np.asarray(e_grid_ev, dtype=float)
    out = np.zeros_like(e)
    # R in 10⁻⁴⁰ cgs → prefactor 1e-40 / 2.2965e-39 = 1 / 22.965
    pref = 1e-40 / (ECD_INTEGRAL_CONST * math.sqrt(math.pi) * sigma_ev)
    for de, r in zip(energies_ev, rotatory_strengths_1e40):
        de, r = float(de), float(r)
        if de <= 0:
            continue
        out += pref * de * r * np.exp(-(((e - de) / sigma_ev) ** 2))
    return out


def is_effectively_achiral(
    rotatory_strengths_1e40: Sequence[float],
    threshold: float = ACHIRAL_R_THRESHOLD,
) -> bool:
    """True when every |R_n| is below *threshold* (10⁻⁴⁰ cgs)."""
    vals = [abs(float(r)) for r in rotatory_strengths_1e40]
    return bool(vals) and max(vals) < threshold


def plot_ecd_spectrum(
    energies_ev: Sequence[float],
    rotatory_strengths_1e40: Sequence[float],
    *,
    mode: str = "broadened",
    sigma_ev: float = DEFAULT_SIGMA_EV,
    x_range_nm: Optional[Tuple[float, float]] = None,
    n_points: int = 800,
):
    """Plotly figure of an ECD spectrum against wavelength (nm).

    ``mode="stick"`` draws one stick per state at λ_n with height R_n
    (10⁻⁴⁰ cgs), red above / blue below zero. ``mode="broadened"`` draws the
    Gaussian Δε curve (:func:`ecd_delta_epsilon`), evaluated on a uniform
    wavelength grid. One y axis per mode (never both at once), symmetric
    about zero so positive and negative Cotton effects read at a glance.
    """
    import plotly.graph_objects as go

    pairs = [
        (float(e), float(r))
        for e, r in zip(energies_ev, rotatory_strengths_1e40)
        if float(e) > 0
    ]
    wl = [EV_NM / e for e, _ in pairs]
    rot = [r for _, r in pairs]
    if x_range_nm is None:
        if wl:
            pad = max(20.0, 0.15 * (max(wl) - min(wl)))
            x_range_nm = (max(1.0, min(wl) - pad), max(wl) + pad)
        else:
            x_range_nm = (100.0, 400.0)
    x_min, x_max = float(x_range_nm[0]), float(x_range_nm[1])

    fig = go.Figure()
    if str(mode).strip().lower() == "broadened":
        lam = np.linspace(max(x_min, 1.0), x_max, max(int(n_points), 50))
        deps = ecd_delta_epsilon(
            [e for e, _ in pairs], rot, EV_NM / lam, sigma_ev=sigma_ev
        )
        fig.add_trace(
            go.Scatter(
                x=lam.tolist(),
                y=deps.tolist(),
                mode="lines",
                line=dict(color="#2563eb", width=2),
                name="Δε (broadened)",
                hovertemplate=(
                    "λ: %{x:.1f} nm<br>Δε: %{y:.3g} L mol⁻¹ cm⁻¹<extra></extra>"
                ),
            )
        )
        y_title = "Δε (L mol⁻¹ cm⁻¹)"
        y_abs = float(np.max(np.abs(deps))) if deps.size else 0.0
    else:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        for x0, r in zip(wl, rot):
            xs.extend([x0, x0, None])
            ys.extend([0.0, r, None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#6b7280", width=2),
                name="Rotatory strength",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=wl,
                y=rot,
                mode="markers",
                marker=dict(
                    color=["#dc2626" if r >= 0 else "#2563eb" for r in rot],
                    size=8,
                ),
                name="R (velocity gauge)",
                showlegend=False,
                hovertemplate=(
                    "λ: %{x:.1f} nm<br>R: %{y:.3f} ×10⁻⁴⁰ cgs<extra></extra>"
                ),
            )
        )
        y_title = "R (10⁻⁴⁰ esu²·cm²)"
        y_abs = max((abs(r) for r in rot), default=0.0)

    # Symmetric about zero; a flat (achiral) spectrum still gets a visible
    # axis instead of Plotly's degenerate [-1e-12, 1e-12] autorange.
    y_lim = 1.15 * y_abs if y_abs > 1e-6 else 1.0
    fig.add_hline(y=0.0, line=dict(color="#9ca3af", width=1))
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title=y_title,
        height=320,
        margin=dict(l=70, r=20, t=30, b=50),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, zeroline=False, range=[x_min, x_max])
    fig.update_yaxes(showgrid=True, zeroline=False, range=[-y_lim, y_lim])
    return fig
