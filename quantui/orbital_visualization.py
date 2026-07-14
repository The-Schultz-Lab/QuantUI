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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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


def infer_charge_and_spin(mol_atom: list, mo_occ: "np.ndarray | list") -> Tuple[int, int]:
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


def generate_cube_file(
    results_path: Path,
    orbital_index: int,
    output_path: Path,
    *,
    nx: int = 60,
    ny: int = 60,
    nz: int = 60,
    margin: float = 5.0,
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
    logger.info("Wrote cube file: %s", output_path)
    return output_path


# ============================================================================
# Cube-file isosurface viewer (plotly — works anywhere)
# ============================================================================

# Max grid points handed to one go.Isosurface trace. The cube grid is a fixed
# resolution regardless of molecule size, so the volume is strided down to this
# cap at render time to keep the figure payload bounded; the saved .cube keeps
# full resolution.
_MAX_ISOSURFACE_POINTS = 48_000


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

    bond_x: List[float] = []
    bond_y: List[float] = []
    bond_z: List[float] = []
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


def render_orbital_isosurface_py3dmol(
    cube_path: Path,
    *,
    isovalue: float = 0.02,
    opacity: float = 0.85,
    width: int = 760,
    height: int = 620,
    pos_color: str = "blue",
    neg_color: str = "red",
    bgcolor: str = "white",
    style: str = "stick",
) -> str:
    """Render an orbital isosurface from a cube file via py3Dmol.

    Unlike :func:`plot_cube_isosurface` (Plotly), py3Dmol isosurfaces the cube
    *in the browser* at full resolution, so there is no Python-side volume
    downsample and the payload is just the cube text. Both lobes are drawn:
    ``+isovalue`` (``pos_color``) and ``-isovalue`` (``neg_color``).

    Returns HTML via py3Dmol's ``_make_html``. The viewer is built through
    :func:`quantui.viz_assets.make_view`, so it loads 3Dmol.js from the
    vendored bundle (the page bootstrap) rather than the CDN — see
    ``viz_assets`` for why this matters offline.

    Parameters
    ----------
    cube_path : Path
        Path to a Gaussian ``.cube`` file (read at full resolution).
    isovalue : float
        Isosurface threshold; both ``+`` and ``-`` lobes are drawn.
    opacity : float
        Surface opacity (0-1).
    width, height : int
        Viewer size in pixels.
    pos_color, neg_color : str
        Lobe colors for the positive and negative isosurfaces.
    bgcolor : str
        Viewer background color.
    style : str
        py3Dmol style for the embedded atoms (e.g. ``"stick"``).

    Returns
    -------
    str
        Self-contained HTML for the interactive viewer.
    """
    from quantui.viz_assets import make_view

    cube_text = Path(cube_path).read_text()
    view = make_view(width=width, height=height)
    view.addModel(cube_text, "cube")
    view.setStyle({style: {}})
    view.addVolumetricData(
        cube_text,
        "cube",
        {"isoval": isovalue, "color": pos_color, "opacity": opacity},
    )
    view.addVolumetricData(
        cube_text,
        "cube",
        {"isoval": -isovalue, "color": neg_color, "opacity": opacity},
    )
    view.setBackgroundColor(bgcolor)
    view.zoomTo()
    return view._make_html()


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
    if total > _MAX_ISOSURFACE_POINTS:
        stride = int(np.ceil((total / _MAX_ISOSURFACE_POINTS) ** (1.0 / 3.0)))
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
