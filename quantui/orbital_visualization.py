"""
Orbital energy-level diagram and cube-file isosurface visualization.

Two capabilities, each with progressively heavier dependencies:

1. **Orbital energy diagram** (matplotlib only) — works everywhere.
   Draws a horizontal‐line energy‐level diagram with HOMO/LUMO labels,
   colour-coded by occupation.  Input is a NumPy array of MO energies
   (from ``results.npz`` or a live ``SessionResult``).

2. **Cube-file isosurface** (plotly + PySCF ``cubegen``) — Linux only.
   Generates a volumetric cube file for a selected MO, then renders an
   isosurface in 3-D using ``plotly.graph_objects.Isosurface``.  This
   requires PySCF at *generation* time; the viewer works on any platform
   once the cube data is saved.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from quantui import theme as _theme

logger = logging.getLogger(__name__)

# Conversion factor — PySCF stores MO energies in Hartree
HARTREE_TO_EV: float = 27.211386245988
BOHR_PER_ANGSTROM: float = 1.8897261254578281

# Light-weight chemistry tables for drawing atom/bond overlays on cube plots.
_COVALENT_RADII_ANGSTROM = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}
_CPK_COLORS = {
    1: "#f8fafc",  # H
    5: "#f59e0b",  # B
    6: "#374151",  # C
    7: "#2563eb",  # N
    8: "#dc2626",  # O
    9: "#22c55e",  # F
    14: "#f59e0b",  # Si
    15: "#f97316",  # P
    16: "#facc15",  # S
    17: "#16a34a",  # Cl
    35: "#b45309",  # Br
    53: "#7c3aed",  # I
}
_ATOMIC_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


# ============================================================================
# Data container
# ============================================================================


@dataclass
class OrbitalInfo:
    """Lightweight container extracted from a PySCF results file."""

    mo_energies_ev: np.ndarray  # shape (n_mo,)
    n_occupied: int
    homo_energy_ev: float
    lumo_energy_ev: float
    homo_lumo_gap_ev: float
    formula: str  # for chart title

    @property
    def n_virtual(self) -> int:
        return len(self.mo_energies_ev) - self.n_occupied


def load_orbital_info(
    results_path: Path,
    *,
    formula: str = "",
    mo_occ: Optional[np.ndarray] = None,
) -> OrbitalInfo:
    """
    Load orbital energies from a ``results.npz`` file.

    Parameters
    ----------
    results_path : Path
        Path to the ``.npz`` file saved by the PySCF calculation script.
        Must contain at least ``mo_energy``; optionally ``mo_occ``.
    formula : str
        Molecule formula (used in chart title).  If empty, uses the
        stem of *results_path*.
    mo_occ : ndarray, optional
        Occupation numbers.  If *None*, they are read from the file or
        inferred by assuming all orbitals with energy below the midpoint
        between the two lowest-energy unoccupied orbitals are filled.

    Returns
    -------
    OrbitalInfo
    """
    data = np.load(results_path, allow_pickle=False)
    mo_energy_ha: np.ndarray = data["mo_energy"]

    # Handle UHF (2, n_mo) — use alpha spin
    if mo_energy_ha.ndim == 2:
        mo_energy_ha = mo_energy_ha[0]

    mo_energy_ev = mo_energy_ha * HARTREE_TO_EV

    # Determine occupation
    if mo_occ is not None:
        occ = np.asarray(mo_occ)
    elif "mo_occ" in data:
        occ = data["mo_occ"]
        if occ.ndim == 2:
            occ = occ[0]
    else:
        # Fallback: assume first n orbitals with energy < 0 are occupied
        occ = (mo_energy_ha < 0).astype(float)

    n_occ = int((occ > 0).sum())
    if n_occ == 0 or n_occ >= len(mo_energy_ev):
        raise ValueError(
            f"Cannot determine HOMO/LUMO: n_occupied={n_occ}, n_total={len(mo_energy_ev)}"
        )

    homo_ev = float(mo_energy_ev[n_occ - 1])
    lumo_ev = float(mo_energy_ev[n_occ])
    gap_ev = lumo_ev - homo_ev

    return OrbitalInfo(
        mo_energies_ev=mo_energy_ev,
        n_occupied=n_occ,
        homo_energy_ev=homo_ev,
        lumo_energy_ev=lumo_ev,
        homo_lumo_gap_ev=gap_ev,
        formula=formula or results_path.stem,
    )


def orbital_info_from_arrays(
    mo_energy: np.ndarray,
    mo_occ: np.ndarray,
    formula: str = "",
) -> OrbitalInfo:
    """
    Build an :class:`OrbitalInfo` directly from NumPy arrays.

    Useful when working with a live ``SessionResult`` where the data is
    already in memory (no ``.npz`` on disk).
    """
    mo_energy = np.asarray(mo_energy)
    mo_occ = np.asarray(mo_occ)

    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]
    if mo_occ.ndim == 2:
        mo_occ = mo_occ[0]

    mo_ev = mo_energy * HARTREE_TO_EV
    n_occ = int((mo_occ > 0).sum())

    if n_occ == 0 or n_occ >= len(mo_ev):
        raise ValueError(
            f"Cannot determine HOMO/LUMO: n_occupied={n_occ}, n_total={len(mo_ev)}"
        )

    return OrbitalInfo(
        mo_energies_ev=mo_ev,
        n_occupied=n_occ,
        homo_energy_ev=float(mo_ev[n_occ - 1]),
        lumo_energy_ev=float(mo_ev[n_occ]),
        homo_lumo_gap_ev=float(mo_ev[n_occ] - mo_ev[n_occ - 1]),
        formula=formula,
    )


# ============================================================================
# Matplotlib energy-level diagram
# ============================================================================


def plot_orbital_diagram(
    info: OrbitalInfo,
    *,
    max_orbitals: int = 20,
    figsize: Tuple[float, float] = (6, 8),
    title: Optional[str] = None,
):
    """
    Draw a horizontal-line orbital energy-level diagram using matplotlib.

    Occupied orbitals are drawn in blue, virtual in grey.  HOMO and LUMO
    are highlighted and labelled.  An arrow annotates the gap.

    Parameters
    ----------
    info : OrbitalInfo
        Orbital data to plot.
    max_orbitals : int
        Show at most this many orbitals centred on the HOMO–LUMO region.
        Keeps the diagram readable for large basis sets.
    figsize : tuple
        Matplotlib figure size ``(width, height)`` in inches.
    title : str, optional
        Custom title; defaults to ``"Orbital Energy Levels — {formula}"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure

    energies = info.mo_energies_ev
    n_occ = info.n_occupied
    n_total = len(energies)

    # Window around HOMO/LUMO
    half = max_orbitals // 2
    start = max(0, n_occ - half)
    end = min(n_total, n_occ + half)
    subset = energies[start:end]
    subset_occ = np.arange(start, end) < n_occ

    # Use Figure directly (not plt.subplots) to avoid triggering the IPython
    # GUI event loop in interactive / test environments.
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Draw energy levels
    line_half_width = 0.3
    for i, (e, occ) in enumerate(zip(subset, subset_occ)):
        color = "#2171b5" if occ else "#bdbdbd"
        lw = 2.5 if (start + i == n_occ - 1 or start + i == n_occ) else 1.5
        ax.plot(
            [-line_half_width, line_half_width],
            [e, e],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
        )

    # HOMO / LUMO labels
    homo_idx_in_subset = n_occ - 1 - start
    lumo_idx_in_subset = n_occ - start

    if 0 <= homo_idx_in_subset < len(subset):
        ax.annotate(
            "HOMO",
            xy=(line_half_width + 0.05, subset[homo_idx_in_subset]),
            fontsize=10,
            fontweight="bold",
            color="#2171b5",
            va="center",
        )

    if 0 <= lumo_idx_in_subset < len(subset):
        ax.annotate(
            "LUMO",
            xy=(line_half_width + 0.05, subset[lumo_idx_in_subset]),
            fontsize=10,
            fontweight="bold",
            color="#e6550d",
            va="center",
        )

    # Gap arrow
    if 0 <= homo_idx_in_subset < len(subset) and 0 <= lumo_idx_in_subset < len(subset):
        mid_x = -line_half_width - 0.15
        ax.annotate(
            "",
            xy=(mid_x, info.lumo_energy_ev),
            xytext=(mid_x, info.homo_energy_ev),
            arrowprops=dict(arrowstyle="<->", color="#e6550d", lw=1.5),
        )
        gap_mid = (info.homo_energy_ev + info.lumo_energy_ev) / 2.0
        ax.text(
            mid_x - 0.05,
            gap_mid,
            f"{info.homo_lumo_gap_ev:.2f} eV",
            fontsize=9,
            color="#e6550d",
            ha="right",
            va="center",
            fontweight="bold",
        )

    # Axis labels and styling
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_xlim(-0.9, 1.0)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(
        title or f"Orbital Energy Levels — {info.formula}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Legend
    occ_patch = mpatches.Patch(color="#2171b5", label="Occupied")
    virt_patch = mpatches.Patch(color="#bdbdbd", label="Virtual")
    ax.legend(handles=[occ_patch, virt_patch], loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


# ============================================================================
# Plotly interactive energy-level diagram
# ============================================================================


def plot_orbital_diagram_plotly(
    info: OrbitalInfo,
    *,
    max_orbitals: int = 20,
    yrange: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    width: int = 380,
    height: int = 460,
):
    """Interactive Plotly orbital energy-level diagram.

    Returns a ``plotly.graph_objects.Figure`` suitable for embedding in a
    ``go.FigureWidget``.  Each MO is drawn as a short horizontal line;
    hover shows the MO index and energy in eV.  HOMO/LUMO are highlighted
    with labels and a gap annotation.

    Parameters
    ----------
    info:
        Orbital data.
    max_orbitals:
        Maximum number of MOs to display, centred on the HOMO–LUMO gap.
    yrange:
        Explicit ``(y_min, y_max)`` in eV; auto-computed when ``None``.
    title:
        Custom plot title; defaults to ``"Orbital Energy Levels — {formula}"``.
    width, height:
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    energies = info.mo_energies_ev
    n_occ = info.n_occupied
    n_total = len(energies)

    half = max_orbitals // 2
    start = max(0, n_occ - half)
    end = min(n_total, n_occ + half)

    LHW = 0.3  # half-width of each horizontal line in x

    traces = []
    for idx in range(start, end):
        e = float(energies[idx])
        is_homo = idx == n_occ - 1
        is_lumo = idx == n_occ
        is_occ = idx < n_occ

        if is_homo:
            color, lw = "#2171b5", 3.0
            hover = f"MO #{idx + 1} — HOMO<br>{e:+.4f} eV"
        elif is_lumo:
            color, lw = "#e6550d", 3.0
            hover = f"MO #{idx + 1} — LUMO<br>{e:+.4f} eV"
        elif is_occ:
            color, lw = "#2171b5", 1.5
            hover = f"MO #{idx + 1} (occupied)<br>{e:+.4f} eV"
        else:
            color, lw = "#9e9e9e", 1.5
            hover = f"MO #{idx + 1} (virtual)<br>{e:+.4f} eV"

        traces.append(
            go.Scatter(
                x=[-LHW, LHW],
                y=[e, e],
                mode="lines",
                line=dict(color=color, width=lw),
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
                name="",
            )
        )

    homo_e = info.homo_energy_ev
    lumo_e = info.lumo_energy_ev
    gap = info.homo_lumo_gap_ev
    bracket_x = -LHW - 0.15

    annotations = [
        dict(
            x=LHW + 0.04,
            y=homo_e,
            xref="x",
            yref="y",
            text="<b>HOMO</b>",
            showarrow=False,
            font=dict(size=11, color="#2171b5"),
            xanchor="left",
            yanchor="middle",
        ),
        dict(
            x=LHW + 0.04,
            y=lumo_e,
            xref="x",
            yref="y",
            text="<b>LUMO</b>",
            showarrow=False,
            font=dict(size=11, color="#e6550d"),
            xanchor="left",
            yanchor="middle",
        ),
        dict(
            x=bracket_x,
            y=homo_e,
            ax=bracket_x,
            ay=lumo_e,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=f"<b>{gap:.2f} eV</b>",
            font=dict(size=10, color="#e6550d"),
            arrowhead=2,
            arrowwidth=1.5,
            arrowcolor="#e6550d",
            xanchor="right",
        ),
    ]

    subset = energies[start:end]
    span = float(subset.max()) - float(subset.min())
    margin = max(0.5, span * 0.08 + 0.5)
    if yrange is None:
        y_min = float(subset.min()) - margin
        y_max = float(subset.max()) + margin
    else:
        y_min, y_max = yrange

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=60, r=110, t=50, b=30),
        title=dict(
            text=title or f"Orbital Energy Levels — {info.formula}",
            font=dict(size=13, family="Arial"),
        ),
        xaxis=dict(
            range=[-0.9, 0.9],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            title="Energy (eV)",
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#e5e7eb",
            tickformat=".1f",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=annotations,
        hovermode="closest",
    )
    return fig


# ============================================================================
# Summary HTML (for notebooks)
# ============================================================================


def orbital_summary_html(info: OrbitalInfo) -> str:
    """
    Return an HTML card summarising orbital energies.

    Designed for ``IPython.display.HTML`` inside a Jupyter cell.
    """
    return (
        '<div style="background:#f8f9fa; padding:12px; border-radius:6px; '
        'border-left:4px solid #2171b5; margin:8px 0; font-family:monospace;">'
        f"<b>Orbital Summary — {info.formula}</b><br>"
        f"Occupied MOs: {info.n_occupied} &nbsp;|&nbsp; "
        f"Virtual MOs: {info.n_virtual} &nbsp;|&nbsp; "
        f"Total: {len(info.mo_energies_ev)}<br>"
        f"HOMO energy: {info.homo_energy_ev:+.4f} eV &nbsp;|&nbsp; "
        f"LUMO energy: {info.lumo_energy_ev:+.4f} eV<br>"
        f"<b>HOMO–LUMO gap: {info.homo_lumo_gap_ev:.4f} eV</b>"
        "</div>"
    )


# ============================================================================
# Cube-file generation (PySCF — Linux only)
# ============================================================================


def infer_charge_and_spin(
    mol_atom: Optional[list], mo_occ: Optional[np.ndarray | list]
) -> Tuple[int, int]:
    """Infer ``(charge, spin)`` for a ``gto.Mole`` from atoms + MO occupations.

    Cube/isosurface generation from saved MO data does not have direct access
    to the original ``Molecule.charge`` / ``Molecule.multiplicity`` — only the
    atom list and the MO coefficients/occupations that came out of the SCF.
    PySCF's ``Mole.build()`` requires ``charge``/``spin`` consistent with the
    actual electron count, so passing the default ``charge=0, spin=0`` fails
    to build for any charged or open-shell (odd-electron) molecule.

    This reconstructs both from data that's always available:

    - ``spin`` (PySCF's ``2S = n_alpha - n_beta``) is 0 when ``mo_occ`` is
      1-D (closed-shell RHF/RKS — including the MP2/CCSD/CCSD(T) paths, which
      always run on an RHF reference), or ``n_alpha - n_beta`` when ``mo_occ``
      is 2-D (UHF/UKS, shape ``(2, n_mo)``).
    - ``charge`` is the nuclear charge (sum of atomic numbers in ``mol_atom``)
      minus the total electron count (``sum(mo_occ)`` over all spin channels).

    Returns ``(0, 0)`` if ``mol_atom`` or ``mo_occ`` is falsy/``None`` so
    callers can pass through directly without a separate None-check.
    """
    if not mol_atom or mo_occ is None:
        return 0, 0
    from .molecule import ATOMIC_NUMBERS

    occ = np.asarray(mo_occ, dtype=float)
    if occ.ndim == 2:
        n_alpha = float(occ[0].sum())
        n_beta = float(occ[1].sum())
        spin = int(round(n_alpha - n_beta))
        n_electrons = n_alpha + n_beta
    else:
        spin = 0
        n_electrons = float(occ.sum())

    nuclear_charge = sum(ATOMIC_NUMBERS.get(sym, 0) for sym, _ in mol_atom)
    charge = int(round(nuclear_charge - n_electrons))
    return charge, spin


def _resolution_label(nx: int, ny: int, nz: int) -> str:
    """Named preset (M-ORBEXPORT ORBX.2) matching *nx/ny/nz*, or ``"custom"``.

    Only a cubic grid matching one of :data:`ISO_RESOLUTION_PRESETS` gets its
    friendly name back; anything else (a non-cubic grid, or a value nobody
    picked from the dropdown) is reported honestly as custom rather than
    guessed at.
    """
    if nx == ny == nz:
        for label, grid in ISO_RESOLUTION_PRESETS.items():
            if grid == nx:
                return label
    return "custom"


def _write_cube_provenance(
    output_path: Path,
    *,
    basis: str,
    nx: int,
    ny: int,
    nz: int,
    charge: int,
    spin: int,
    method: str = "",
) -> None:
    """Overwrite a freshly written cube file's two free-text comment lines
    with QuantUI provenance (M-EXPORT2 EXP2.4 / M-ORBEXPORT ORBX.4).

    The Gaussian cube format reserves exactly its first two lines for
    human-readable comments — everything from line 3 on (atom count, grid
    header, atoms, volumetric data) is untouched, so this is safe to do
    *after* :func:`pyscf.tools.cubegen.orbital` writes the file rather than
    reimplementing its grid-computation logic just to pass a custom comment
    through. Without this, every cube QuantUI writes carries cubegen's own
    fixed comment ("Orbital value in real space (1/Bohr^3)") and nothing
    about which basis, grid, or charge/spin state produced it — unrecoverable
    once the file has been handed to Avogadro / VMD / Multiwfn or emailed on.
    """
    line1 = (
        f"QuantUI orbital cube — {method}/{basis}"
        if method
        else f"QuantUI orbital cube — basis {basis}"
    )
    label = _resolution_label(nx, ny, nz)
    line2 = f"grid {nx}x{ny}x{nz} ({label}); charge={charge} spin={spin}"
    text = output_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if len(lines) < 2:
        return  # not a well-formed cube file; leave it alone rather than corrupt it
    lines[0] = line1
    lines[1] = line2
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_cube_file(
    results_path: Path,
    orbital_index: int,
    output_path: Path,
    *,
    nx: int = 60,
    ny: int = 60,
    nz: int = 60,
    margin: float = 5.0,
    method: str = "",
) -> Path:
    """
    Generate a Gaussian cube file for a molecular orbital.

    Requires PySCF and the original ``mol`` object data.  This function
    is Linux/WSL only.

    Parameters
    ----------
    results_path : Path
        Path to ``results.npz`` (must also contain ``mol_atom`` and
        ``mol_basis`` keys, added by an extended script template).
    orbital_index : int
        0-based MO index to visualise.
    output_path : Path
        Where to write the ``.cube`` file.
    nx, ny, nz : int
        Grid resolution along each axis.
    margin : float
        Extra space (Bohr) beyond atomic extents.
    method : str
        Method/functional label for the provenance comment (M-EXPORT2
        EXP2.4). ``results.npz`` doesn't carry it, so this has to come from
        the caller; omitted from the comment when blank.

    Returns
    -------
    Path
        The written cube file path.

    Raises
    ------
    ImportError
        If PySCF is not available.
    """
    try:
        from pyscf import gto
        from pyscf.tools import cubegen
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for cube file generation (Linux/WSL only).\n"
            "  conda install -c conda-forge pyscf"
        ) from exc

    data = np.load(results_path, allow_pickle=True)
    mo_coeff = data["mo_coeff"]
    mo_occ = data["mo_occ"] if "mo_occ" in data else None
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]

    atom_str = str(data["mol_atom"]) if "mol_atom" in data else None
    basis_str = str(data["mol_basis"]) if "mol_basis" in data else None

    if atom_str is None or basis_str is None:
        raise ValueError(
            "results.npz does not contain 'mol_atom'/'mol_basis' keys. "
            "Re-run the calculation with the updated script template."
        )

    # Charge/spin aren't stored in results.npz — infer them from the MO
    # occupations (when present) so charged/open-shell molecules don't fail
    # to build. mol_atom here is a PySCF-format string, not the (symbol,
    # coords) tuple list infer_charge_and_spin expects, so parse it first.
    charge, spin = 0, 0
    if mo_occ is not None:
        parsed_atoms = [
            (tok.split()[0], [0.0, 0.0, 0.0])
            for tok in atom_str.replace(";", "\n").splitlines()
            if tok.strip()
        ]
        charge, spin = infer_charge_and_spin(parsed_atoms, mo_occ)

    mol = gto.M(
        atom=atom_str, basis=basis_str, unit="Angstrom", charge=charge, spin=spin
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cubegen.orbital(
        mol,
        str(output_path),
        mo_coeff[:, orbital_index],
        nx=nx,
        ny=ny,
        nz=nz,
        margin=margin,
    )
    _write_cube_provenance(
        output_path,
        basis=basis_str,
        nx=nx,
        ny=ny,
        nz=nz,
        charge=charge,
        spin=spin,
        method=method,
    )
    logger.info("Wrote cube file: %s", output_path)
    return output_path


def generate_cube_from_arrays(
    mol_atom: list,
    mol_basis: str,
    mo_coeff: np.ndarray,
    orbital_index: int,
    output_path: Path,
    *,
    nx: int = 60,
    ny: int = 60,
    nz: int = 60,
    margin: float = 5.0,
    charge: int = 0,
    spin: int = 0,
    method: str = "",
) -> Path:
    """
    Generate a cube file from in-session MO data (no ``.npz`` file required).

    Unlike :func:`generate_cube_file`, this function takes the atom list
    and MO coefficient array directly, as stored in :class:`SessionResult`
    or :class:`OptimizationResult`.

    Parameters
    ----------
    mol_atom : list
        Atom list in PySCF format — list of ``(symbol, [x, y, z])`` tuples
        with coordinates in Angstrom.
    mol_basis : str
        Basis set string (e.g. ``'6-31G*'``).
    mo_coeff : ndarray
        MO coefficient matrix, shape ``(n_ao, n_mo)`` for RHF or
        ``(2, n_ao, n_mo)`` for UHF.  Alpha-spin coefficients are used for UHF.
    orbital_index : int
        0-based MO index to visualise.
    output_path : Path
        Where to write the ``.cube`` file.
    nx, ny, nz : int
        Grid resolution along each axis.
    margin : float
        Extra space (Bohr) beyond atomic extents.
    charge : int
        Total molecular charge. Required for charged species (e.g. H3O+,
        NH4+, OH-) — without it PySCF's electron-count check fails at
        ``mol.build()``. Default 0 (neutral).
    spin : int
        PySCF's ``2S = n_alpha - n_beta``. Required for open-shell
        (odd-electron) molecules — default 0 assumes closed-shell.
    method : str
        Method/functional label for the provenance comment (M-EXPORT2
        EXP2.4) — e.g. ``'B3LYP'``. Best-effort: the caller's current method
        selection, not necessarily re-verified against what actually produced
        *mo_coeff* (this function has no way to check that). Omitted from the
        comment entirely when blank, rather than guessed at.

    Returns
    -------
    Path
        The written cube file path.

    Raises
    ------
    ImportError
        If PySCF is not available.
    """
    try:
        from pyscf import gto
        from pyscf.tools import cubegen
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for cube file generation (Linux/WSL only).\n"
            "  conda install -c conda-forge pyscf"
        ) from exc

    mol = gto.M(
        atom=mol_atom, basis=mol_basis, unit="Angstrom", charge=charge, spin=spin
    )

    coeff = np.asarray(mo_coeff)
    if coeff.ndim == 3:
        coeff = coeff[0]  # UHF: use alpha spin

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cubegen.orbital(
        mol,
        str(output_path),
        coeff[:, orbital_index],
        nx=nx,
        ny=ny,
        nz=nz,
        margin=margin,
    )
    _write_cube_provenance(
        output_path,
        basis=mol_basis,
        nx=nx,
        ny=ny,
        nz=nz,
        charge=charge,
        spin=spin,
        method=method,
    )
    logger.info("Wrote cube file: %s", output_path)
    return output_path


# ============================================================================
# Cube-file isosurface viewer (plotly — works anywhere)
# ============================================================================

# Max grid points handed to one go.Isosurface trace. The cube grid is a fixed
# resolution regardless of molecule size, so the volume is strided down to this
# cap at render time to keep the figure payload bounded; the saved .cube keeps
# full resolution.
# Named grid presets (M-ORBEXPORT ORBX.2). cubegen cost scales as nx*ny*nz,
# so these are roughly 0.3x / 1x / 2.4x / 4.6x the work of the 60^3 default.
ISO_RESOLUTION_PRESETS: dict[str, int] = {
    "coarse": 40,
    "medium": 60,
    "fine": 80,
    "very fine": 100,
}
# Labels carry the grid and a plain adjective. The precise cost multipliers
# they used to quote were noise at the point of choosing — the ordering is what
# the user is actually deciding between, and the real wait depends on the
# molecule and basis anyway.
ISO_RESOLUTION_OPTIONS: list[tuple[str, str]] = [
    ("Coarse (40³) — fastest", "coarse"),
    ("Medium (60³) — default", "medium"),
    ("Fine (80³) — slower", "fine"),
    ("Very fine (100³) — slowest", "very fine"),
]
DEFAULT_ISO_RESOLUTION: str = "medium"

_MAX_ISOSURFACE_POINTS = 48_000

# ⚠️ The two-knob problem, and which knob matters where.
#
# There are two independent resolution controls: the cubegen grid (real
# fidelity, and what lands in the saved .cube) and the render stride below
# (display only). Raising the grid without raising the cap would make the user
# wait longer and then throw the extra detail away.
#
# BUT that only applies to the PLOTLY path. py3Dmol — the primary orbital
# renderer — isosurfaces the cube in the browser at full resolution and never
# strides, so there a finer grid is visible immediately. The cap exists solely
# to bound the Plotly figure payload.
#
# So the cap scales with the chosen grid, preserving the same fraction of
# detail the 60^3 default keeps, with a ceiling: past a few hundred thousand
# points a Plotly Isosurface trace becomes painful to interact with, and a
# fallback renderer that locks up the tab is worse than one that is smooth and
# slightly coarse.
_MAX_ISOSURFACE_POINTS_CEILING = 250_000


def max_render_points(grid_points: int) -> int:
    """Render-stride cap appropriate to a *grid_points*³ cubegen grid.

    Keeps the fraction of retained detail constant relative to the 60³ default
    rather than the absolute count, so "fine" actually looks finer.
    """
    scaled = _MAX_ISOSURFACE_POINTS * (max(1, grid_points) / 60.0) ** 3
    return int(min(_MAX_ISOSURFACE_POINTS_CEILING, max(_MAX_ISOSURFACE_POINTS, scaled)))


def parse_cube_file(cube_path: Path) -> dict:
    """
    Parse a Gaussian cube file into a dict of NumPy arrays.

    Returns
    -------
    dict with keys:
        atoms : list of (Z, x, y, z)
        origin : ndarray (3,)
        axes : ndarray (3, 3) — row i is the step vector for axis i
        nx, ny, nz : int
        data : ndarray (nx, ny, nz) — volumetric data
    """
    with open(cube_path) as fh:
        # First two lines are comments
        fh.readline()
        fh.readline()

        parts = fh.readline().split()
        n_atoms = abs(int(parts[0]))
        origin = np.array([float(x) for x in parts[1:4]])

        axes = np.zeros((3, 3))
        dims = []
        for i in range(3):
            parts = fh.readline().split()
            dims.append(int(parts[0]))
            axes[i] = [float(x) for x in parts[1:4]]

        nx, ny, nz = dims

        atoms = []
        for _ in range(n_atoms):
            parts = fh.readline().split()
            z = int(parts[0])
            x, y, zz = float(parts[2]), float(parts[3]), float(parts[4])
            atoms.append((z, x, y, zz))

        # Volumetric data
        vals: List[float] = []
        for line in fh:
            vals.extend(float(v) for v in line.split())

        data = np.array(vals).reshape((nx, ny, nz))

    return {
        "atoms": atoms,
        "origin": origin,
        "axes": axes,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "data": data,
    }


def enclosed_density_fraction(cube_path: Path, isovalue: float) -> Optional[float]:
    """Fraction of the orbital's probability density inside |psi| >= *isovalue*.

    An isovalue is an amplitude threshold on the wavefunction, which is not an
    intuitive quantity — the question people actually have is "how much of the
    orbital am I looking at?". That is the integral of |psi|^2 over the region
    the surface encloses, divided by the total, and it is a straight sum over
    the cube grid that already exists on disk.

    Returned as a fraction in [0, 1], or None if the cube cannot be read. The
    caller shows it next to the slider; it must never be the reason a render
    fails, hence the broad catch.
    """
    try:
        cube = parse_cube_file(Path(cube_path))
        data = np.asarray(cube["data"], dtype=float)
        # Voxel volume = |a . (b x c)| for the three step vectors. Constant, so
        # it cancels in the ratio — computed anyway to keep the intent readable
        # and in case a caller ever wants an absolute integral.
        density = data * data
        total = float(density.sum())
        if total <= 0.0:
            return None
        inside = float(density[np.abs(data) >= float(isovalue)].sum())
        return max(0.0, min(1.0, inside / total))
    except Exception as exc:  # noqa: BLE001 — a readout must never break a render
        logger.debug("enclosed_density_fraction failed: %s", exc)
        return None


def _build_molecule_overlay_data(atoms: list[tuple[int, float, float, float]]) -> dict:
    """Build marker and bond segments from cube atom records."""
    atom_x: List[float] = []
    atom_y: List[float] = []
    atom_z: List[float] = []
    atom_colors: List[str] = []
    atom_sizes: List[float] = []
    atom_labels: List[str] = []

    for z_num, x, y, z in atoms:
        atom_x.append(x)
        atom_y.append(y)
        atom_z.append(z)
        atom_colors.append(_CPK_COLORS.get(z_num, "#9ca3af"))
        atom_sizes.append(max(6.0, 15.0 * _COVALENT_RADII_ANGSTROM.get(z_num, 0.75)))
        atom_labels.append(_ATOMIC_SYMBOLS.get(z_num, str(z_num)))

    # None entries deliberately break a Plotly line trace between bond
    # segments (x=[x1,x2,None,x3,x4,None,...]) so consecutive bonds don't
    # visually connect.
    bond_x: List[Optional[float]] = []
    bond_y: List[Optional[float]] = []
    bond_z: List[Optional[float]] = []
    for i, (zi, xi, yi, zi_pos) in enumerate(atoms):
        for zj, xj, yj, zj_pos in atoms[i + 1 :]:
            ri = _COVALENT_RADII_ANGSTROM.get(zi, 0.75)
            rj = _COVALENT_RADII_ANGSTROM.get(zj, 0.75)
            cutoff = (ri + rj) * 1.25 * BOHR_PER_ANGSTROM
            dist = float(
                np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi_pos - zj_pos) ** 2)
            )
            if dist <= cutoff:
                bond_x.extend([xi, xj, None])
                bond_y.extend([yi, yj, None])
                bond_z.extend([zi_pos, zj_pos, None])

    return {
        "atom_x": atom_x,
        "atom_y": atom_y,
        "atom_z": atom_z,
        "atom_colors": atom_colors,
        "atom_sizes": atom_sizes,
        "atom_labels": atom_labels,
        "bond_x": bond_x,
        "bond_y": bond_y,
        "bond_z": bond_z,
    }


# Named phase-colour pairs (M-ORBEXPORT ORBX.6). These are conventions people
# recognise from other software, not arbitrary picks, which is why they are
# presets rather than only a colour picker.
ORBITAL_COLOR_SCHEMES: dict[str, tuple[str, str]] = {
    "blue-red": ("#2166ac", "#b2182b"),
    "orange-blue": ("#e08214", "#2166ac"),
    "yellow-blue": ("#e6c619", "#2b4b9b"),
    "green-purple": ("#1b7837", "#762a83"),
    "teal-magenta": ("#01807e", "#c51b7d"),
}
ORBITAL_COLOR_OPTIONS: list[tuple[str, str]] = [
    ("Blue / Red — Avogadro, Jmol", "blue-red"),
    ("Orange / Blue — GaussView", "orange-blue"),
    ("Yellow / Blue — journal figures", "yellow-blue"),
    ("Green / Purple", "green-purple"),
    ("Teal / Magenta", "teal-magenta"),
]
DEFAULT_ORBITAL_COLORS: str = "blue-red"


def orbital_colors(scheme: str) -> tuple[str, str]:
    """(positive, negative) hex pair for *scheme*, falling back to the default."""
    return ORBITAL_COLOR_SCHEMES.get(
        scheme, ORBITAL_COLOR_SCHEMES[DEFAULT_ORBITAL_COLORS]
    )


# Builds the viewer and registers the live-update bridge.
#
# ⚠️ The cube is embedded ONCE, as a JS string, and every 3Dmol call is made
# from it client-side. The obvious implementation — view.addModel(cube) plus two
# view.addVolumetricData(cube) from Python — embeds the payload THREE times,
# which at the 100^3 grid means ~39 MB of HTML per render (measured
# 2026-08-04). One copy is not a micro-optimisation here.
#
# Doing the work in JS is also what makes the controls live. Isovalue, opacity,
# colours and background can then be changed on the existing viewer, so the
# camera the user rotated into survives — a Python re-render cannot preserve it,
# because it replaces the whole viewer (see GOTCHAS: "Camera state does NOT
# persist across atomic HTML swaps").
_ISO_VIEWER_JS = """
(function(){
  var UID="__UID__", DATA=__DATA__, FMT=__FMT__;
  var WITH_SURFACES=__WITH_SURFACES__, SCENE=__SCENE__;
  var state={iso:__ISO__, op:__OP__, pos:__POS__, neg:__NEG__, bg:__BG__, wf:__WF__};
  function v(){ return window["viewer_"+UID]; }

  // ⚠️ Isosurfaces are SHAPES, not surfaces. viewer.addVolumetricData() routes
  // to addIsosurface(), which does this.shapes.push(...) — so
  // removeAllSurfaces() iterates this.surfaces, finds nothing, and removes
  // nothing. Using it meant every update stacked another layer on the old one:
  // a lower isovalue engulfed the previous surface (looked like it worked), a
  // higher one hid inside it (looked dead), and stacked translucent layers read
  // as steadily more opaque. Reported 2026-08-04.
  //
  // So the shapes are tracked explicitly and removed by reference. removeShape
  // is exact; removeAllShapes() would also take out anything else ever added.
  var shapes=[];
  function surfaces(){
    var vw=v(); if(!vw){ return; }
    for(var i=0;i<shapes.length;i++){
      try{ vw.removeShape(shapes[i]); }catch(e){}
    }
    shapes=[];
    // smoothness = Laplacian smoothing passes 3Dmol.js runs on the raw
    // marching-cubes mesh. Default (1) leaves visible triangle facets on the
    // lobes; the roughness is the mesh, not the grid, so more cubegen points
    // don't fix it but a few smoothing passes do (GaussView-like surfaces).
    shapes.push(vw.addVolumetricData(DATA,"cube",
      {isoval: state.iso, color: state.pos, opacity: state.op, smoothness: 5,
       wireframe: state.wf}));
    shapes.push(vw.addVolumetricData(DATA,"cube",
      {isoval: -state.iso, color: state.neg, opacity: state.op, smoothness: 5,
       wireframe: state.wf}));
  }

  function build(){
    var vw=v();
    if(!vw){ setTimeout(build,50); return; }
    vw.addModel(DATA,FMT);
    vw.setStyle({}, {__STYLE__:{}});
    if(WITH_SURFACES){ surfaces(); }
    vw.setBackgroundColor(state.bg);
    vw.zoomTo();
    // Restore the orientation the previous viewer had, but only for the same
    // scene: carrying a camera onto a different molecule would frame it
    // arbitrarily. SCENE is the atom block, so switching ORBITALS keeps the
    // view (the common case) while switching MOLECULES re-frames.
    // A freshly built viewer is never busy. Bumping the sequence invalidates
    // any in-flight busy(true) from the previous viewer, so a stale retry can
    // never land on this one and leave it dimmed with nothing to clear it.
    window.__quantuiIsoBusySeq=(window.__quantuiIsoBusySeq||0)+1;
    var saved=window.__quantuiIsoLastView;
    if(saved && saved.key===SCENE){
      try{ vw.setView(saved.view); }catch(e){}
    }
    vw.render();
  }

  // Called when a generate starts, i.e. the last moment this viewer exists.
  function stash(){
    var vw=v(); if(!vw){ return; }
    try{ window.__quantuiIsoLastView={key:SCENE, view:vw.getView()}; }catch(e){}
  }

  // Live update. Camera is untouched unless the caller asks for a rebuild, and
  // even then getView/setView carries it across — changing an isovalue must
  // not throw away an orientation the user chose.
  window["__quantuiIsoUpdate_"+UID] = function(opts){
    var vw=v(); if(!vw){ return false; }
    var geom=false;
    if(opts.iso!==undefined && opts.iso!==state.iso){ state.iso=opts.iso; geom=true; }
    if(opts.op!==undefined && opts.op!==state.op){ state.op=opts.op; geom=true; }
    if(opts.pos!==undefined && opts.pos!==state.pos){ state.pos=opts.pos; geom=true; }
    if(opts.neg!==undefined && opts.neg!==state.neg){ state.neg=opts.neg; geom=true; }
    // Wireframe is baked into the shape at creation (addVolumetricData), same
    // as colour/opacity above — there is no "restyle" call for an existing
    // isosurface shape, so a toggle rebuilds it like every other appearance
    // change here.
    if(opts.wf!==undefined && opts.wf!==state.wf){ state.wf=opts.wf; geom=true; }
    if(opts.bg!==undefined){ state.bg=opts.bg; vw.setBackgroundColor(state.bg); }
    if(geom){
      var cam=null;
      try{ cam=vw.getView(); }catch(e){}
      surfaces();
      if(cam){ try{ vw.setView(cam); }catch(e){} }
    }
    vw.render();
    return true;
  };
  // Last viewer to load owns the unqualified name; the bridge uses it so the
  // kernel does not have to track uids.
  window.__quantuiIsoUpdate = window["__quantuiIsoUpdate_"+UID];
  // (7) Busy state WITHOUT an output swap. Replacing the viewer with a
  // "Generating..." message collapsed the panel from ~620px to nothing, so the
  // accordion jumped and the page scrolled. Dimming the existing surface keeps
  // the layout identical and still reads as busy.
  window["__quantuiIsoBusy_"+UID] = function(on){
    if(on){ stash(); }   // (1) capture the camera before the viewer is replaced
    var host=document.getElementById("3dmolviewer_"+UID);
    if(!host){ return false; }
    host.style.transition="opacity 0.25s ease";
    host.style.opacity = on ? "0.25" : "1";
    var tag=document.getElementById("orb_busy_"+UID);
    if(tag){ tag.style.display = on ? "block" : "none"; }
    return true;
  };
  window.__quantuiIsoBusy = window["__quantuiIsoBusy_"+UID];
  window.__quantuiIsoCapture = function(transparent){
    var vw=v(); if(!vw || !vw.pngURI){ return null; }
    if(!transparent){ return vw.pngURI(); }
    // Transparent EXPORT without a transparent VIEW: drop the background,
    // capture, put it back. The user asked for the preview to stay opaque.
    var uri=null;
    try{
      vw.setBackgroundColor(state.bg, 0.0); vw.render();
      uri=vw.pngURI();
    } finally {
      vw.setBackgroundColor(state.bg, 1.0); vw.render();
    }
    return uri;
  };
  build();
})();
"""


def render_orbital_isosurface_py3dmol(
    cube_path: Path,
    *,
    isovalue: float = 0.02,
    opacity: float = 0.85,
    wireframe: bool = False,
    width: int = 760,
    height: int = 620,
    color_scheme: str = DEFAULT_ORBITAL_COLORS,
    bgcolor: str = "white",
    style: str = "stick",
    capture_class: str = "",
) -> str:
    """Interactive orbital isosurface, with live client-side controls.

    Emits an empty py3Dmol viewer plus one JS block that embeds the cube a
    single time and does all the 3Dmol work in the browser. See ``_ISO_VIEWER_JS``
    for why that matters (payload size, and camera-preserving live updates).

    Parameters
    ----------
    isovalue, opacity
        Initial surface threshold and transparency. Both are changeable live via
        ``window.__quantuiIsoUpdate`` without rebuilding the viewer.
    wireframe
        Surface finish (M-ORBEXPORT ORBX.7). Re-scoped from the original
        "metallic" request after reading the vendored 3Dmol.js: Lambert
        shading has no specular term, so glossy/metallic isn't reachable on
        this backend — wireframe is what the renderer can actually do.
        Changeable live, same as isovalue/opacity, though 3Dmol.js rebuilds
        the shape to apply it (no in-place restyle for volumetric data).
    color_scheme
        Key into :data:`ORBITAL_COLOR_SCHEMES`.
    bgcolor
        Scene background. Note the EXPORT background is chosen at capture time,
        not here — see ``__quantuiIsoCapture``.
    capture_class
        CSS class of the hidden Textarea receiving PNG data URIs. Empty omits
        the Save-PNG button entirely, so it can never render with nowhere to
        deliver to.
    """
    cube_text = Path(cube_path).read_text()
    return _build_iso_viewer(
        cube_text,
        data_format="cube",
        scene_key=_scene_key(cube_text),
        with_surfaces=True,
        isovalue=isovalue,
        opacity=opacity,
        wireframe=wireframe,
        width=width,
        height=height,
        color_scheme=color_scheme,
        bgcolor=bgcolor,
        style=style,
        capture_class=capture_class,
    )


def _scene_key(cube_text: str) -> str:
    """Identity of the molecule in a cube, used to decide whether a saved
    camera still applies. The atom block only — a different orbital of the same
    molecule must keep the user's orientation, a different molecule must not."""
    import hashlib

    lines = cube_text.splitlines()
    try:
        n_atoms = abs(int(lines[2].split()[0]))
        block = "".join(lines[6 : 6 + n_atoms])
    except Exception:  # noqa: BLE001 — a bad header only costs camera reuse
        block = "".join(lines[:8])
    return hashlib.sha1(block.encode()).hexdigest()[:16]


def render_molecule_placeholder_py3dmol(
    molecule: Any,
    *,
    width: int = 760,
    height: int = 620,
    bgcolor: str = "white",
    style: str = "stick",
) -> str:
    """The same viewer shell, showing only the molecule — no surfaces.

    Shown before the first Generate so the panel is never empty: the first
    isosurface then fades in over an existing viewer rather than appearing in a
    collapsed panel, and the camera the user set while inspecting the structure
    carries into the isosurface (same scene key).
    """
    xyz = (
        f"{len(molecule.atoms)}\n{molecule.get_formula()}\n"
        f"{molecule.to_xyz_string()}"
    )
    return _build_iso_viewer(
        xyz,
        data_format="xyz",
        # Keyed on the geometry, so a cube of this same molecule reuses the
        # camera the user set here.
        scene_key=_scene_key_from_xyz(xyz),
        with_surfaces=False,
        width=width,
        height=height,
        bgcolor=bgcolor,
        style=style,
    )


def _scene_key_from_xyz(xyz: str) -> str:
    import hashlib

    lines = xyz.splitlines()[2:]
    # Cube atom records carry Z and Bohr coordinates while XYZ carries symbols
    # and Angstrom, so the two cannot hash to the same value. Element count and
    # ordering are enough to tell "same molecule" for camera reuse.
    block = "".join(ln.split()[0] for ln in lines if ln.strip())
    return hashlib.sha1(block.encode()).hexdigest()[:16]


def _build_iso_viewer(
    data_text: str,
    *,
    data_format: str,
    scene_key: str,
    with_surfaces: bool,
    isovalue: float = 0.02,
    opacity: float = 0.85,
    wireframe: bool = False,
    width: int = 760,
    height: int = 620,
    color_scheme: str = DEFAULT_ORBITAL_COLORS,
    bgcolor: str = "white",
    style: str = "stick",
    capture_class: str = "",
) -> str:
    import json

    from quantui.viz_assets import make_view

    pos_color, neg_color = orbital_colors(color_scheme)

    view = make_view(width=width, height=height)
    view_html = view._make_html()

    m = re.search(r"3dmolviewer_(\w+)", view_html)
    if m is None:
        logger.error("could not find py3Dmol viewer id; isosurface unavailable")
        return '<p style="color:#b91c1c;padding:8px">Viewer could not be built.</p>'
    uid = m.group(1)

    js = (
        _ISO_VIEWER_JS.replace("__UID__", uid)
        .replace("__DATA__", json.dumps(data_text))
        .replace("__FMT__", json.dumps(data_format))
        .replace("__WITH_SURFACES__", "true" if with_surfaces else "false")
        .replace("__SCENE__", json.dumps(scene_key))
        .replace("__ISO__", repr(float(isovalue)))
        .replace("__OP__", repr(float(opacity)))
        .replace("__WF__", "true" if wireframe else "false")
        .replace("__POS__", json.dumps(pos_color))
        .replace("__NEG__", json.dumps(neg_color))
        .replace("__BG__", json.dumps(bgcolor))
        .replace("__STYLE__", style)
    )
    busy = (
        f'<div id="orb_busy_{uid}" style="display:none;position:absolute;'
        "top:50%;left:0;right:0;transform:translateY(-50%);text-align:center;"
        'font-size:13px;color:#334155;font-style:italic;pointer-events:none">'
        "⏳ Generating…</div>"
    )
    # position:relative on the wrapper so the overlay is placed against the
    # viewer rather than the page.
    body = (
        f'<div style="position:relative">{view_html}{busy}</div>'
        f"<script>{js}</script>"
    )
    if capture_class:
        body += _png_capture_controls(uid, capture_class)
    # Framed like every other 3-D viewer (theme.frame_viewer_html), so the
    # isosurface panel matches the molecule and trajectory viewers.
    return _theme.frame_viewer_html(body, width=width)


# Matches app_visualization._STEPPER_BTN_STYLE so in-viewer controls look like
# one family. Duplicated rather than imported: app_visualization imports from
# this module, and importing back would close the cycle.
_ORB_BTN_STYLE = (
    f"padding:2px 9px;border:1px solid {_theme.BORDER};border-radius:4px;"
    "background:#f8fafc;color:#334155;cursor:pointer;font-size:13px;line-height:1.4;"
)

# JS->kernel bridge for "save what I am looking at" (M-ORBEXPORT ORBX.1).
#
# Direction matters: M-LOGSCROLL route C pushes kernel -> JS, and there is no
# equivalent in this codebase for coming back the other way. The trick used
# here is the standard one for ipywidgets: write into a hidden Textarea's DOM
# node and dispatch an 'input' event, which the widget's own view is already
# listening for, so it syncs `value` back to the kernel like a user typing.
#
# Why a DOM button rather than the Python Button next to "Export cube": a
# Python button would need kernel -> JS -> kernel, and the kernel cannot ask
# the browser for anything synchronously. A DOM button captures and posts in
# one direction, which is also why the frame steppers are built this way.
#
# ⚠️ The capture MUST come from the live viewer, not a re-render. Per GOTCHAS
# ("Camera state does NOT persist across atomic HTML swaps"), anything that
# re-renders first exports the default camera rather than the one the user
# rotated into — which is the entire value of client-side capture.
_PNG_CAPTURE_JS = """
(function(){
  var UID="__UID__", CLS="__CLS__";
  var btn=document.getElementById("orb_png_"+UID);
  if(!btn){ return; }
  btn.addEventListener("click", function(){
    var cap=window["__CAPFN__"];
    if(!cap){ btn.textContent="\\u26a0 viewer not ready"; return; }
    // Transparency is decided HERE, at capture, not by the live viewer — the
    // preview stays opaque while the exported file has no background.
    var wantAlpha=false;
    var cb=document.querySelector("."+CLS+"-transparent input");
    if(cb){ wantAlpha=!!cb.checked; }
    var uri=null;
    try{ uri=cap(wantAlpha); }catch(e){ btn.textContent="\\u26a0 capture failed"; return; }
    if(!uri){ btn.textContent="\\u26a0 capture failed"; return; }
    var box=document.querySelector("."+CLS+" textarea");
    if(!box){ btn.textContent="\\u26a0 no inbox"; return; }
    box.value=uri;
    box.dispatchEvent(new Event("input", {bubbles:true}));
    var old=btn.textContent;
    btn.textContent="\\u2713 saved";
    setTimeout(function(){ btn.textContent=old; }, 2000);
  });
})();
"""


def _png_capture_controls(
    uid: str, capture_class: str, capture_fn: str = "__quantuiIsoCapture"
) -> str:
    """A 'Save PNG' button wired to the viewer identified by *uid*.

    *capture_fn* is the name of the global JS function (already defined
    elsewhere, e.g. ``window[capture_fn] = function(transparent){...}``) that
    does the actual ``pngURI()`` capture. Defaults to the isosurface viewer's
    bare, unscoped global for backward compatibility (ORBX.1). Callers with
    multiple live viewers of the same kind on a page (e.g. a fresh uid per
    render) must pass a uid-scoped name to avoid one viewer's button
    capturing another viewer's frame.
    """
    js = (
        _PNG_CAPTURE_JS.replace("__UID__", uid)
        .replace("__CLS__", capture_class)
        .replace("__CAPFN__", capture_fn)
    )
    return (
        f'<div style="margin:4px 0 2px;padding:0 8px 6px;font-size:13px;">'
        f'<button id="orb_png_{uid}" type="button" '
        f'title="Save this view as a PNG, exactly as you have rotated it" '
        f'style="{_ORB_BTN_STYLE}">\u2b07 Save PNG</button>'
        f"</div><script>{js}</script>"
    )


# Generic, viewer-agnostic UID-scoped capture function (M-EXPORT2 EXP2.2).
# Relocated here from app_visualization.py (2026-08-26) once a fourth viewer
# (the molecule viewer) needed it from visualization_py3dmol.py, a module
# below app_visualization.py in the import graph — this module has no
# app-level dependencies, so it is the safe common home. Unlike
# _PNG_CAPTURE_JS above (which reads live isosurface render options off the
# viewer), this one only calls pngURI() on window["viewer_"+UID] — a global
# 3Dmol.js sets for every viewer regardless of what is displayed in it — so
# it works unmodified for the reorg-geometry, trajectory, vibrational, and
# molecule viewers alike. Each caller gets its own uid-scoped
# window[capture_fn] so one render's button can never capture a different
# render's (possibly already-detached) viewer.
_GENERIC_CAPTURE_JS = """
(function(){
  var UID="__UID__";
  window["__CAPFN__"] = function(transparent){
    var vw = window["viewer_"+UID];
    if(!vw || !vw.pngURI){ return null; }
    if(!transparent){ return vw.pngURI(); }
    var uri=null;
    try{
      vw.setBackgroundColor(__BG__, 0.0); vw.render();
      uri=vw.pngURI();
    } finally {
      vw.setBackgroundColor(__BG__, 1.0); vw.render();
    }
    return uri;
  };
})();
"""


def plot_cube_isosurface(
    cube_path: Path,
    *,
    isovalue: float = 0.02,
    opacity: float = 0.4,
    width: int = 760,
    height: int = 620,
    title: Optional[str] = None,
    show_molecule: bool = False,
    show_grid: bool = True,
    scene_bgcolor: str = "white",
    axis_color: str = "#111827",
    max_points: Optional[int] = None,
    title_color: Optional[str] = None,
    bond_color: str = "#6b7280",
):
    """
    Render an orbital isosurface from a cube file using Plotly.

    Draws both positive and negative lobes (blue / red) of the MO at
    the given *isovalue*.

    Parameters
    ----------
    cube_path : Path
        Path to a Gaussian ``.cube`` file.
    isovalue : float
        Isosurface threshold (e.g. 0.02 for orbitals).
    opacity : float
        Surface opacity (0–1).
    width, height : int
        Figure size in pixels.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    cube = parse_cube_file(cube_path)
    nx, ny, nz = cube["nx"], cube["ny"], cube["nz"]
    data = cube["data"]
    origin = cube["origin"]
    axes = cube["axes"]

    # Downsample so the browser payload + plotly.js isosurfacing stay bounded.
    # Stride each axis so the total point count stays under _MAX_ISOSURFACE_POINTS.
    total = nx * ny * nz
    stride = 1
    cap = _MAX_ISOSURFACE_POINTS if max_points is None else int(max_points)
    if total > cap:
        stride = int(np.ceil((total / cap) ** (1.0 / 3.0)))
    data = data[::stride, ::stride, ::stride]

    # Build coordinate grids (Bohr), strided to match the downsampled volume.
    x = origin[0] + np.arange(nx)[::stride] * axes[0, 0]
    y = origin[1] + np.arange(ny)[::stride] * axes[1, 1]
    z = origin[2] + np.arange(nz)[::stride] * axes[2, 2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    fig = go.Figure()

    # Both lobes in a single trace (half the payload of two): surfaces at
    # -isovalue (red) and +isovalue (blue), via a step colorscale split at the
    # midpoint of [-isovalue, +isovalue].
    fig.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=-isovalue,
            isomax=isovalue,
            surface_count=2,
            opacity=opacity,
            colorscale=[
                [0.0, "rgb(222,45,38)"],
                [0.5, "rgb(222,45,38)"],
                [0.5, "rgb(49,130,189)"],
                [1.0, "rgb(49,130,189)"],
            ],
            cmin=-isovalue,
            cmax=isovalue,
            showscale=False,
            name=f"±{isovalue}",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    if show_molecule and cube["atoms"]:
        overlay = _build_molecule_overlay_data(cube["atoms"])
        if overlay["bond_x"]:
            fig.add_trace(
                go.Scatter3d(
                    x=overlay["bond_x"],
                    y=overlay["bond_y"],
                    z=overlay["bond_z"],
                    mode="lines",
                    line=dict(color=bond_color, width=6),
                    name="Bonds",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=overlay["atom_x"],
                y=overlay["atom_y"],
                z=overlay["atom_z"],
                mode="markers",
                marker=dict(
                    size=overlay["atom_sizes"],
                    color=overlay["atom_colors"],
                    opacity=1.0,
                    line=dict(color=bond_color, width=1),
                ),
                text=overlay["atom_labels"],
                hovertemplate="%{text}<extra></extra>",
                name="Atoms",
                showlegend=False,
            )
        )

    fig.update_layout(
        width=width,
        height=height,
        title=dict(
            text=title or "Molecular Orbital Isosurface",
            font=dict(color=title_color or axis_color),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=48, b=0),
        font=dict(color=axis_color),
        scene=dict(
            xaxis=dict(
                title="X (Bohr)",
                showgrid=show_grid,
                showbackground=show_grid,
                zeroline=False,
                color=axis_color,
            ),
            yaxis=dict(
                title="Y (Bohr)",
                showgrid=show_grid,
                showbackground=show_grid,
                zeroline=False,
                color=axis_color,
            ),
            zaxis=dict(
                title="Z (Bohr)",
                showgrid=show_grid,
                showbackground=show_grid,
                zeroline=False,
                color=axis_color,
            ),
            bgcolor=scene_bgcolor,
            aspectmode="data",
        ),
    )

    return fig
