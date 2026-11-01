"""
Side-by-side comparison of multiple quantum chemistry calculations.

Produces:

* An HTML summary table comparing method, basis, energy, gap, and
  convergence across multiple jobs.
* A grouped bar chart (matplotlib) for energy and HOMO–LUMO gap
  comparisons.

All inputs are plain data — either :class:`~quantui.storage.JobMetadata`
objects enriched with parsed results, or lightweight :class:`CalcSummary`
dicts built from ``results.npz`` files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HARTREE_TO_EV: float = 27.211386245988


# ============================================================================
# Lightweight comparison record
# ============================================================================


@dataclass
class CalcSummary:
    """
    One row in a comparison table.

    Can be built from a :class:`JobMetadata` + ``results.npz``, from a
    :class:`SessionResult`, or by hand.
    """

    label: str
    formula: str
    method: str
    basis: str
    energy_hartree: Optional[float] = None
    homo_lumo_gap_ev: Optional[float] = None
    converged: Optional[bool] = None
    n_iterations: Optional[int] = None
    status: str = ""

    @property
    def energy_ev(self) -> Optional[float]:
        if self.energy_hartree is not None:
            return self.energy_hartree * HARTREE_TO_EV
        return None


# ============================================================================
# Builders — turn different data sources into CalcSummary
# ============================================================================


def summary_from_job_metadata(metadata, *, label: str = "") -> CalcSummary:
    """
    Create a :class:`CalcSummary` from a :class:`JobMetadata`.

    If the job's ``results.npz`` file exists, the energy and convergence
    data are loaded.  Otherwise only metadata is included.
    """
    formula = (
        metadata.job_name.split("_")[0]
        if "_" in metadata.job_name
        else metadata.job_name
    )
    calc = CalcSummary(
        label=label or metadata.job_name,
        formula=formula,
        method=metadata.method,
        basis=metadata.basis,
        status=metadata.status,
    )

    results_path = Path(metadata.paths.get("results", ""))
    if results_path.exists():
        try:
            data = np.load(results_path, allow_pickle=False)
            calc.energy_hartree = float(data["energy"])
            calc.converged = bool(data["converged"])
            if "mo_energy" in data:
                mo_e = data["mo_energy"]
                if mo_e.ndim == 2:
                    mo_e = mo_e[0]
                mo_occ = data.get("mo_occ")
                if mo_occ is not None:
                    if mo_occ.ndim == 2:
                        mo_occ = mo_occ[0]
                    n_occ = int((mo_occ > 0).sum())
                else:
                    n_occ = int((mo_e < 0).sum())
                if 0 < n_occ < len(mo_e):
                    calc.homo_lumo_gap_ev = float(
                        (mo_e[n_occ] - mo_e[n_occ - 1]) * HARTREE_TO_EV
                    )
        except Exception as exc:
            logger.warning("Could not load results for %s: %s", calc.label, exc)

    return calc


def summary_from_session_result(result, *, label: str = "") -> CalcSummary:
    """
    Create a :class:`CalcSummary` from a :class:`SessionResult`.
    """
    return CalcSummary(
        label=label or f"{result.formula} {result.method}/{result.basis}",
        formula=result.formula,
        method=result.method,
        basis=result.basis,
        energy_hartree=result.energy_hartree,
        homo_lumo_gap_ev=result.homo_lumo_gap_ev,
        converged=result.converged,
        n_iterations=result.n_iterations,
        status="COMPLETED",
    )


def summary_from_saved_result(data: dict, *, label: str = "") -> CalcSummary:
    """
    Create a :class:`CalcSummary` from a saved ``result.json`` dict.

    Parameters
    ----------
    data:
        Dict loaded from ``result.json`` via
        :func:`~quantui.results_storage.load_result`.
    label:
        Optional display label.  Defaults to
        ``"<formula> <method>/<basis> (<timestamp>)"``.
    """
    formula = data.get("formula", "?")
    method = data.get("method", "?")
    basis = data.get("basis", "?")
    ts = data.get("timestamp", "")
    auto_label = f"{formula} {method}/{basis}" + (f" ({ts})" if ts else "")
    return CalcSummary(
        label=label or auto_label,
        formula=formula,
        method=method,
        basis=basis,
        energy_hartree=data.get("energy_hartree"),
        homo_lumo_gap_ev=data.get("homo_lumo_gap_ev"),
        converged=data.get("converged"),
        n_iterations=data.get("n_iterations"),
        status="COMPLETED",
    )


# ============================================================================
# HTML comparison table
# ============================================================================


def comparison_table_html(summaries: Sequence[CalcSummary]) -> str:
    """
    Return an HTML ``<table>`` comparing calculation summaries.

    Suitable for ``IPython.display.HTML``.
    """
    if not summaries:
        return "<p><em>No calculations to compare.</em></p>"

    header = (
        "<tr>"
        "<th style='text-align:left; padding:6px 12px;'>Label</th>"
        "<th style='text-align:left; padding:6px 12px;'>Formula</th>"
        "<th style='text-align:left; padding:6px 12px;'>Method / Basis</th>"
        "<th style='text-align:right; padding:6px 12px;'>Energy (Ha)</th>"
        "<th style='text-align:right; padding:6px 12px;'>HOMO–LUMO gap (eV)</th>"
        "<th style='text-align:center; padding:6px 12px;'>Converged</th>"
        "<th style='text-align:center; padding:6px 12px;'>Iterations</th>"
        "</tr>"
    )

    rows: List[str] = []
    for s in summaries:
        e_str = f"{s.energy_hartree:.8f}" if s.energy_hartree is not None else "—"
        g_str = f"{s.homo_lumo_gap_ev:.4f}" if s.homo_lumo_gap_ev is not None else "—"
        c_str = "✅" if s.converged else ("❌" if s.converged is not None else "—")
        i_str = str(s.n_iterations) if s.n_iterations is not None else "—"
        rows.append(
            # Explicit, theme-aware row colours: JupyterLab's rendered-HTML
            # CSS stripes odd rows white, which left light text unreadable
            # on every other row in Dark theme.
            f"<tr style='background:var(--q-bg-panel, transparent);"
            f"color:var(--q-text-body, inherit)'>"
            f"<td style='padding:4px 12px;'>{_esc(s.label)}</td>"
            f"<td style='padding:4px 12px;'>{_esc(s.formula)}</td>"
            f"<td style='padding:4px 12px;'>{_esc(s.method)} / {_esc(s.basis)}</td>"
            f"<td style='padding:4px 12px; text-align:right; font-family:monospace;'>{e_str}</td>"
            f"<td style='padding:4px 12px; text-align:right; font-family:monospace;'>{g_str}</td>"
            f"<td style='padding:4px 12px; text-align:center;'>{c_str}</td>"
            f"<td style='padding:4px 12px; text-align:center;'>{i_str}</td>"
            f"</tr>"
        )

    return (
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse; width:100%; font-size:14px;">'
        "<thead style='background:var(--q-surface-muted-bg, #f0f4f8);"
        f"color:var(--q-text-strong, inherit)'>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ============================================================================
# Matplotlib comparison chart
# ============================================================================


def plot_comparison(
    summaries: Sequence[CalcSummary],
    *,
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
):
    """
    Return a matplotlib Figure with grouped bars for energy and HOMO–LUMO gap.

    Parameters
    ----------
    summaries : sequence of CalcSummary
        Two or more calculations to compare.
    figsize : tuple
        Figure size.
    title : str, optional
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    labels = [s.label for s in summaries]
    energies = [s.energy_hartree for s in summaries]
    gaps = [s.homo_lumo_gap_ev for s in summaries]

    has_energies = any(e is not None for e in energies)
    has_gaps = any(g is not None for g in gaps)

    n_panels = int(has_energies) + int(has_gaps)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No energy data available for comparison.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    x = np.arange(len(labels))

    if has_energies:
        ax = axes[idx]
        vals = [e if e is not None else 0.0 for e in energies]
        colours = ["#2171b5" if e is not None else "#bdbdbd" for e in energies]
        ax.bar(x, vals, color=colours, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Total Energy (Ha)")
        ax.set_title("Energy Comparison")
        for xi, v, e in zip(x, vals, energies):
            if e is not None:
                ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        idx += 1

    if has_gaps:
        ax = axes[idx]
        vals = [g if g is not None else 0.0 for g in gaps]
        colours = ["#e6550d" if g is not None else "#bdbdbd" for g in gaps]
        ax.bar(x, vals, color=colours, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("HOMO–LUMO Gap (eV)")
        ax.set_title("HOMO–LUMO Gap Comparison")
        for xi, v, g in zip(x, vals, gaps):
            if g is not None:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title or "Calculation Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
