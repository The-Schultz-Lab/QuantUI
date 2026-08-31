"""
Molecular visualization using py3Dmol (and optional PlotlyMol).

This module provides 3D molecular visualization using py3Dmol as the primary
backend (stable, widely used, already installed). PlotlyMol is supported as
an optional alternative for users who prefer Plotly-based figures.

Author: Jonathan Schultz, NCCU
Created: 2026-02-17
"""

import logging
import os
import tempfile
from typing import Literal, cast

from quantui import theme as _theme

logger = logging.getLogger(__name__)

Py3DmolStyle = Literal["ball+stick", "stick", "sphere", "line", "cartoon"]
BackendName = Literal["auto", "py3dmol", "plotlymol"]

# Check available visualization backends
try:
    import py3Dmol  # noqa: F401 — availability probe; views build via viz_assets.make_view

    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    logger.warning("py3Dmol not available - primary visualization disabled")

try:
    from plotlymol3d import draw_3D_rep
    from plotlymol3d import format_lighting as _plotlymol_format_lighting

    PLOTLYMOL_AVAILABLE = True
except ImportError:
    PLOTLYMOL_AVAILABLE = False
    _plotlymol_format_lighting = None  # type: ignore[assignment]
    logger.info("PlotlyMol not available (optional)")

# ── Visualization style and lighting constants ────────────────────────────────

# Display-style options presented in the UI.  The value is the canonical key
# used internally; each backend maps it to its own representation.
VIZ_STYLE_OPTIONS: list[tuple[str, str]] = [
    ("Ball & Stick", "ball+stick"),
    ("Stick", "stick"),
    ("Sphere (VDW)", "sphere"),
    ("Line", "line"),
]

# Named lighting presets — identical to those in the plotlyMol dash app.
# Only applied when the PlotlyMol backend is active.
LIGHTING_PRESETS: dict[str, dict] = {
    "soft": {"ambient": 0.4, "diffuse": 0.8, "specular": 0.1, "roughness": 0.8},
    "default": {"ambient": 0.0, "diffuse": 1.0, "specular": 0.0, "roughness": 1.0},
    "bright": {"ambient": 0.5, "diffuse": 0.8, "specular": 0.3, "roughness": 0.5},
    "metallic": {"ambient": 0.2, "diffuse": 0.7, "specular": 1.0, "roughness": 0.1},
    "dramatic": {"ambient": 0.0, "diffuse": 1.0, "specular": 0.6, "roughness": 0.2},
}
LIGHTING_OPTIONS: list[tuple[str, str]] = [
    ("Soft", "soft"),
    ("Default", "default"),
    ("Bright", "bright"),
    ("Metallic", "metallic"),
    ("Dramatic", "dramatic"),
]

DEFAULT_STYLE: str = "ball+stick"
DEFAULT_LIGHTING: str = "soft"


def is_visualization_available() -> bool:
    """
    Check if molecular visualization is available.

    Returns:
        True if py3Dmol OR PlotlyMol is available, False otherwise.
    """
    return PY3DMOL_AVAILABLE or PLOTLYMOL_AVAILABLE


def get_available_backends() -> list[str]:
    """
    Get list of available visualization backends.

    Returns:
        List of available backend names (e.g., ['py3dmol', 'plotlymol'])
    """
    backends = []
    if PY3DMOL_AVAILABLE:
        backends.append("py3dmol")
    if PLOTLYMOL_AVAILABLE:
        backends.append("plotlymol")
    return backends


def molecule_to_xyz_string(molecule) -> str:
    """
    Convert QuantUI Molecule to XYZ string format.

    Args:
        molecule: QuantUI Molecule object

    Returns:
        XYZ format string suitable for py3Dmol or PlotlyMol
    """
    from quantui.molecule import Molecule

    if not isinstance(molecule, Molecule):
        raise TypeError("Expected QuantUI Molecule object")

    return molecule.to_xyz_string()


def _add_atom_index_labels(view, molecule) -> None:
    """Overlay 1-based atom indices on a py3Dmol viewer."""
    for i, coord in enumerate(molecule.coordinates):
        x, y, z = (float(c) for c in coord)
        view.addLabel(
            str(i + 1),
            {
                "position": {"x": x, "y": y, "z": z},
                "backgroundColor": "white",
                "backgroundOpacity": 0.75,
                "fontColor": "black",
                "fontSize": 14,
                "borderThickness": 0.5,
                "borderColor": "gray",
            },
        )


def visualize_molecule_py3dmol(
    molecule,
    style: Py3DmolStyle = "ball+stick",
    width: int = 600,
    height: int = 500,
    bgcolor: str = "white",
    lighting: str = "soft",  # accepted for API symmetry; py3Dmol has no preset lighting
    show_atom_indices: bool = False,
):
    """
    Create interactive 3D visualization using py3Dmol.

    Args:
        molecule: QuantUI Molecule object
        style: Visualization style:
            - "stick": Stick representation (default, good for small molecules)
            - "sphere": Van der Waals spheres
            - "line": Line representation
            - "cartoon": Cartoon (for proteins)
        width: Viewer width in pixels (default: 600)
        height: Viewer height in pixels (default: 500)
        bgcolor: Background color (default: "white")

    Returns:
        py3Dmol.view object (call .show() in Jupyter to display)

    Raises:
        ImportError: If py3Dmol is not installed

    Example:
        >>> mol = Molecule(['O', 'H', 'H'], [[0,0,0], [0.757,0.587,0], [-0.757,0.587,0]])
        >>> view = visualize_molecule_py3dmol(mol, style="stick")
        >>> view.show()  # In Jupyter
    """
    if not PY3DMOL_AVAILABLE:
        raise ImportError(
            "py3Dmol is not installed. To enable 3D visualization:\n"
            "  pip install py3dmol"
        )

    # Build a well-formed XYZ block: count line + title line + coordinates.
    # py3Dmol is lenient about the header in most environments, but browsers
    # running the exported HTML require the standard two-line header to parse
    # the format correctly.
    bare_xyz = molecule.to_xyz_string()
    xyz_string = f"{len(molecule.atoms)}\n{molecule.get_formula()}\n{bare_xyz}"

    logger.info(
        f"Creating py3Dmol visualization for {molecule.get_formula()} "
        f"(style={style})"
    )

    # Create viewer — via the offline-safe factory so 3Dmol.js loads from the
    # vendored bundle (the page bootstrap), never the CDN (offline classroom).
    from quantui.viz_assets import make_view

    view = make_view(width=width, height=height)

    # Add molecule
    view.addModel(xyz_string, "xyz")

    # Set style — "ball+stick" requires a compound spec in py3Dmol
    if style == "ball+stick":
        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    else:
        view.setStyle({style: {}})

    # MET.6: 3Dmol.js's own bond perception leaves a coordination metal as a lone
    # dot — it draws no bonds to the centre. Draw the metal↔donor bonds ourselves,
    # dashed (GaussView convention), from the same distance-based connectivity the
    # salt-warning uses. No-op for purely organic molecules.
    _add_coordination_bonds(view, molecule)

    # Set background
    view.setBackgroundColor(bgcolor)

    if show_atom_indices:
        _add_atom_index_labels(view, molecule)

    # Zoom to fit — includes the (now bonded) metal, so it is never off-screen.
    view.zoomTo()

    return view


# Dashed coordination-bond styling (py3Dmol addCylinder): a thin gray dashed
# cylinder from the metal centre to each donor atom.
_COORD_BOND_RADIUS = 0.06
_COORD_BOND_COLOR = "#777777"


def _add_coordination_bonds(view, molecule) -> None:
    """Draw dashed metal↔donor cylinders so a metal centre isn't a lone dot.

    Uses the distance-based, metal-aware connectivity finder. Best-effort: any
    failure (or a molecule with no metal) simply leaves the view unchanged.
    """
    try:
        from quantui.connectivity import metal_coordination_bonds

        coords = molecule.coordinates
        bonds = metal_coordination_bonds(molecule.atoms, coords)
        for i, j in bonds:
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]
            view.addCylinder(
                {
                    "start": {"x": float(xi), "y": float(yi), "z": float(zi)},
                    "end": {"x": float(xj), "y": float(yj), "z": float(zj)},
                    "radius": _COORD_BOND_RADIUS,
                    "color": _COORD_BOND_COLOR,
                    "dashed": True,
                    "fromCap": 1,
                    "toCap": 1,
                }
            )
    except Exception:  # noqa: BLE001 — bond decoration must never break the viewer
        logger.debug("coordination-bond overlay skipped", exc_info=True)


_PY3DMOL_STYLES: tuple[Py3DmolStyle, ...] = (
    "ball+stick",
    "stick",
    "sphere",
    "line",
    "cartoon",
)


def _validate_py3dmol_style(style: str) -> Py3DmolStyle:
    if style not in _PY3DMOL_STYLES:
        raise ValueError(f"style must be one of {list(_PY3DMOL_STYLES)}, got '{style}'")
    return cast(Py3DmolStyle, style)


def visualize_molecule_plotlymol(
    molecule,
    mode: str = "ball+stick",
    resolution: int = 32,
    width: int = 600,
    height: int = 500,
    bgcolor: str = "#ffffff",
    lighting: str = "soft",
    show_atom_indices: bool = False,
):
    """
    Create interactive 3D visualization using PlotlyMol (optional backend).

    Args:
        molecule: QuantUI Molecule object
        mode: Visualization mode - one of:
            - "ball+stick": Full-size atoms with bonds (default)
            - "stick": Small atoms with bonds
            - "vdw": Van der Waals spheres only (no bonds)
        resolution: Sphere tessellation resolution (16-64, default: 32)
        width: Figure width in pixels (default: 600)
        height: Figure height in pixels (default: 500)
        bgcolor: Background color as hex string or name (default: "#ffffff")

    Returns:
        plotly.graph_objects.Figure object

    Raises:
        ImportError: If PlotlyMol is not installed
    """
    if not PLOTLYMOL_AVAILABLE:
        raise ImportError(
            "PlotlyMol is not installed. To enable PlotlyMol visualization:\n"
            "  pip install plotlymol"
        )

    # Validate mode
    valid_modes = ["ball+stick", "stick", "vdw"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    # Convert to XYZ string
    xyz_string = molecule_to_xyz_string(molecule)

    # Get charge for RDKit processing
    charge = molecule.charge

    logger.info(
        f"Creating PlotlyMol visualization for {molecule.get_formula()} "
        f"(mode={mode}, resolution={resolution})"
    )

    # draw_3D_rep takes a file path, not an in-memory string
    full_xyz = f"{len(molecule.atoms)}\n\n{xyz_string}\n"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xyz", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(full_xyz)
        tmp.close()
        try:
            fig = draw_3D_rep(
                xyzfile=tmp.name,
                charge=charge,
                mode=mode,
                resolution=resolution,
            )
        except Exception as exc:
            # RDKit's bond-order perception (rdDetermineBonds), called inside
            # plotlymol3d's draw_3D_rep, raises inconsistently across builds
            # for the same "can't perceive this molecule's bonds" condition —
            # a clean ValueError on most, a raw C++-level IndexError
            # ("unordered_map::at") on at least one Python-3.9 RDKit wheel
            # (a metal with no covalent-radius table entry). Normalized to one
            # type here so callers — including the MET.3 fallback below, which
            # must still catch it — don't depend on a third-party
            # implementation detail that varies by platform/Python version.
            raise ValueError(
                f"Could not determine bonds for {molecule.get_formula()}: {exc}"
            ) from exc
        if _plotlymol_format_lighting is not None:
            preset = LIGHTING_PRESETS.get(lighting, LIGHTING_PRESETS["soft"])
            fig = _plotlymol_format_lighting(fig, **preset)
    finally:
        os.unlink(tmp.name)

    fig.update_layout(
        width=width,
        height=height,
        title=f"{molecule.get_formula()} - {mode.replace('+', ' & ').title()}",
        paper_bgcolor=bgcolor,
        scene=dict(bgcolor=bgcolor),
    )
    if show_atom_indices:
        import plotly.graph_objects as go

        xs = [float(c[0]) for c in molecule.coordinates]
        ys = [float(c[1]) for c in molecule.coordinates]
        zs = [float(c[2]) for c in molecule.coordinates]
        labels = [str(i + 1) for i in range(len(molecule.atoms))]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="text",
                text=labels,
                textfont=dict(size=14, color="#111827"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return fig


def visualize_molecule(
    molecule,
    backend: BackendName = "auto",
    style: str = "ball+stick",
    width: int = 600,
    height: int = 500,
    bgcolor: str = "white",
    lighting: str = "soft",
    **kwargs,
):
    """
    Create interactive 3D visualization (backend-agnostic).

    This is the main visualization function. It automatically selects the
    best available backend or uses the one specified.

    Args:
        molecule: QuantUI Molecule object
        backend: Visualization backend:
            - "auto": Use py3Dmol if available, else PlotlyMol (default)
            - "py3dmol": Use py3Dmol (recommended, stable)
            - "plotlymol": Use PlotlyMol (optional, Plotly-based)
        style: Visualization style (backend-dependent):
            - py3Dmol: "stick", "sphere", "line", "cartoon"
            - PlotlyMol: "ball+stick", "stick", "vdw"
        width: Viewer/figure width in pixels (default: 600)
        height: Viewer/figure height in pixels (default: 500)
        bgcolor: Background color (default: "white")
        **kwargs: Additional backend-specific arguments

    Returns:
        py3Dmol.view or plotly Figure depending on backend

    Raises:
        ImportError: If no visualization backend is available
        ValueError: If specified backend is not available

    Example:
        >>> mol = Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]])
        >>> # Use default backend (py3Dmol)
        >>> view = visualize_molecule(mol)
        >>> view.show()  # In Jupyter
    """
    # Determine backend
    if backend == "auto":
        if PLOTLYMOL_AVAILABLE:
            backend = "plotlymol"
        elif PY3DMOL_AVAILABLE:
            backend = "py3dmol"
        else:
            raise ImportError(
                "No visualization backend available. Install one of:\n"
                "  pip install py3dmol  (recommended)\n"
                "  pip install plotlymol"
            )

    # Use selected backend
    if backend == "py3dmol":
        py3dmol_style = _validate_py3dmol_style(style)
        return visualize_molecule_py3dmol(
            molecule,
            style=py3dmol_style,
            width=width,
            height=height,
            bgcolor=bgcolor,
            lighting=lighting,
            show_atom_indices=bool(kwargs.get("show_atom_indices", False)),
        )
    elif backend == "plotlymol":
        # Map UI style keys to PlotlyMol mode names
        mode_map = {
            "ball+stick": "ball+stick",
            "stick": "stick",
            "sphere": "vdw",
            "line": "stick",  # plotlyMol has no line mode; use stick
        }
        mode = mode_map.get(style, "ball+stick")
        try:
            plotly_kwargs = dict(kwargs)
            show_indices = bool(plotly_kwargs.pop("show_atom_indices", False))
            return visualize_molecule_plotlymol(
                molecule,
                mode=mode,
                width=width,
                height=height,
                bgcolor=bgcolor,
                lighting=lighting,
                show_atom_indices=show_indices,
                **plotly_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — a viewer must never hard-error
            # MET.3: PlotlyMol runs RDKit valence perception, which raises on
            # transition metals ("Atom N has no valences defined"). py3Dmol
            # renders straight from coordinates with no valence model, so fall
            # back to it rather than crash on a valid molecule.
            if not PY3DMOL_AVAILABLE:
                raise
            logger.warning(
                "PlotlyMol could not render %s (%s); falling back to py3Dmol.",
                molecule.get_formula(),
                exc,
            )
            fallback_style = style if style in _PY3DMOL_STYLES else "ball+stick"
            # **kwargs is intentionally not forwarded: it carries PlotlyMol-only
            # options (e.g. resolution) that visualize_molecule_py3dmol doesn't
            # accept — the same reason the backend=="py3dmol" path above omits it.
            return visualize_molecule_py3dmol(
                molecule,
                style=_validate_py3dmol_style(fallback_style),
                width=width,
                height=height,
                bgcolor=bgcolor,
                lighting=lighting,
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _info_box_html(molecule, backend: str) -> str:
    """Build the info-box HTML fragment shown above the 3D viewer."""
    backends = get_available_backends()
    backend_str = ", ".join(backends)
    selected = backend if backend != "auto" else (backends[0] if backends else "")
    return (
        '<div class="quantui-info-box">'
        "<strong>📊 Molecule Information</strong><br>"
        f"<strong>Formula:</strong> {molecule.get_formula()} | "
        f"<strong>Atoms:</strong> {len(molecule.atoms)} | "
        f"<strong>Electrons:</strong> {molecule.get_electron_count()} | "
        f"<strong>Charge:</strong> {molecule.charge} | "
        f"<strong>Multiplicity:</strong> {molecule.multiplicity}<br>"
        f"<small>Using: {selected} "
        f"(available: {backend_str})</small>"
        "</div>"
    )


def _unavailable_html(molecule) -> str:
    """HTML fallback when no 3D visualization backend is installed."""
    return (
        '<div style="padding:10px;font-family:sans-serif;color:#444;">'
        "<p>⚠️ 3D visualization not available.</p>"
        "<p>To enable visualization, install one of:</p>"
        "<ul><li><code>pip install py3dmol</code> (recommended)</li>"
        "<li><code>pip install plotlymol</code></li></ul>"
        "<p><strong>Molecule Information</strong><br>"
        f"Formula: {molecule.get_formula()}<br>"
        f"Atoms: {len(molecule.atoms)}<br>"
        f"Electrons: {molecule.get_electron_count()}<br>"
        f"Charge: {molecule.charge}<br>"
        f"Multiplicity: {molecule.multiplicity}</p>"
        f"<pre>{molecule.to_xyz_string()}</pre>"
        "</div>"
    )


def render_molecule_html(
    molecule,
    backend: Literal["auto", "py3dmol", "plotlymol"] = "auto",
    style: str = "ball+stick",
    show_info: bool = True,
    width: int = 600,
    height: int = 500,
    bgcolor: str = "#ffffff",
    lighting: str = "soft",
    capture_class: str = "",
    show_atom_indices: bool = False,
) -> str:
    """Return self-contained HTML for the molecule viewer (no display side-effects).

    Mirrors :func:`display_molecule` but emits a single HTML string so callers
    can route through an atomic ``Output.outputs`` swap (Rule 6 in
    ``reflections/01-voila-rendering-and-display.md``) rather than
    ``with output: display(viz)`` — the latter is a known root-cause
    family for trajectory and Analysis-tab rendering regressions. Errors are
    caught and returned as inline HTML so the caller sees a
    visible failure message in the viewer slot instead of a blank 🙁 panel.

    ``capture_class`` wires a "Save PNG" button (M-EXPORT2 EXP2.2), the same
    viewer-agnostic bridge as the reorg/trajectory/vibrational viewers
    (``orbital_visualization._GENERIC_CAPTURE_JS`` reads
    ``window["viewer_"+UID]`` directly). **Only wired for the py3Dmol
    backend** — the plotly path has no equivalent client-side capture global,
    and Plotly's own modebar already offers a native "download plot as png"
    button, so this is not a real gap for that backend. Callers that render
    this molecule into more than one output at once (this function is used
    for the Calculate-tab preview, the Results-tab viewer, and the
    Analysis-tab viewer — up to two of which can be showing the same
    molecule simultaneously) must pass a distinct ``capture_class`` per
    output slot, or one slot's button can post into a different slot's
    inbox (``document.querySelector`` matches the first element with that
    class in the whole page, not scoped to which button was clicked).
    """
    if not is_visualization_available():
        return _unavailable_html(molecule)

    parts: list[str] = []
    if show_info:
        parts.append(_info_box_html(molecule, backend))

    try:
        viz = visualize_molecule(
            molecule,
            backend=backend,
            style=style,
            width=width,
            height=height,
            bgcolor=bgcolor,
            lighting=lighting,
            show_atom_indices=show_atom_indices,
        )
        make_html = getattr(viz, "_make_html", None)
        if callable(make_html):
            view_html = viz._make_html()
            if capture_class:
                import json
                import re

                from quantui.orbital_visualization import (
                    _GENERIC_CAPTURE_JS,
                    _png_capture_controls,
                )

                m = re.search(r"3dmolviewer_(\w+)", view_html)
                if m is not None:
                    uid = m.group(1)
                    capture_fn = f"__quantuiMolCapture_{uid}"
                    capture_js = (
                        _GENERIC_CAPTURE_JS.replace("__UID__", uid)
                        .replace("__CAPFN__", capture_fn)
                        .replace("__BG__", json.dumps(bgcolor))
                    )
                    view_html = (
                        view_html
                        + f"<script>{capture_js}</script>"
                        + _png_capture_controls(
                            uid, capture_class, capture_fn=capture_fn
                        )
                    )
            parts.append(view_html)
        else:
            import plotly.io as _pio

            parts.append(
                _pio.to_html(
                    viz,
                    full_html=False,
                    include_plotlyjs="require",
                    config={"responsive": True},
                )
            )
        logger.info(f"Rendered HTML for {molecule.get_formula()}")
    except Exception as e:
        logger.error(f"Render failed for {molecule.get_formula()}: {e}")
        parts.append(
            '<div style="color:#b91c1c;padding:8px;">'
            f"❌ Visualization failed: {e}</div>"
        )
    # Frame the fragment at the viewer's own width — see theme.frame_viewer_html
    # for why the border cannot live on the hosting Output widget's CSS class.
    # The info box is INSIDE the frame so it aligns with the canvas.
    return _theme.frame_viewer_html("\n".join(parts), width=width)


def display_molecule(
    molecule,
    backend: Literal["auto", "py3dmol", "plotlymol"] = "auto",
    style: str = "ball+stick",
    show_info: bool = True,
    width: int = 600,
    height: int = 500,
    bgcolor: str = "#ffffff",
    lighting: str = "soft",
):
    """
    Display molecule in Jupyter notebook with optional info box.

    This is the main function for notebook integration. It handles all
    available backends and provides a consistent interface.

    Args:
        molecule: QuantUI Molecule object
        backend: Visualization backend ("auto", "py3dmol", "plotlymol")
        style: Visualization style (backend-dependent)
        show_info: Whether to show molecular info box
        width: Viewer/figure width in pixels
        height: Viewer/figure height in pixels

    Example:
        >>> # In Jupyter notebook
        >>> mol = Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]])
        >>> display_molecule(mol)  # Uses py3Dmol by default
    """
    from IPython.display import HTML, display

    if not is_visualization_available():
        # Fallback: show text representation
        print("⚠️  3D visualization not available")
        print("\nTo enable visualization, install one of:")
        print("  pip install py3dmol  (recommended)")
        print("  pip install plotlymol")
        print("\nMolecule Information:")
        print(f"  Formula: {molecule.get_formula()}")
        print(f"  Atoms: {len(molecule.atoms)}")
        print(f"  Electrons: {molecule.get_electron_count()}")
        print(f"  Charge: {molecule.charge}")
        print(f"  Multiplicity: {molecule.multiplicity}")
        print("\nXYZ Coordinates:")
        print(molecule.to_xyz_string())
        return

    # Show info box if requested
    if show_info:
        backends = get_available_backends()
        backend_str = ", ".join(backends)
        selected = backend if backend != "auto" else backends[0]

        info_html = (
            '<div class="quantui-info-box">'
            "<strong>📊 Molecule Information</strong><br>"
            f"<strong>Formula:</strong> {molecule.get_formula()} | "
            f"<strong>Atoms:</strong> {len(molecule.atoms)} | "
            f"<strong>Electrons:</strong> {molecule.get_electron_count()} | "
            f"<strong>Charge:</strong> {molecule.charge} | "
            f"<strong>Multiplicity:</strong> {molecule.multiplicity}<br>"
            f"<small>Using: {selected} (available: {backend_str})</small>"
            "</div>"
        )
        display(HTML(info_html))

    # Create and display visualization
    try:
        viz = visualize_molecule(
            molecule,
            backend=backend,
            style=style,
            width=width,
            height=height,
            bgcolor=bgcolor,
            lighting=lighting,
        )

        # display(viz) triggers py3Dmol's _repr_html_() method, which embeds
        # the viewer as self-contained HTML.  This works in both JupyterLab
        # and classic Notebook.  viz.show() uses IPython.display.Javascript
        # which is blocked by JupyterLab's content-security-policy and
        # returns None (causing "None" to appear in cell output).
        display(viz)

        logger.info(f"Successfully displayed {molecule.get_formula()}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        logger.error(f"Display failed for {molecule.get_formula()}: {e}")


def get_installation_message() -> str:
    """
    Get installation instructions for visualization backends.

    Returns:
        Formatted string with installation instructions
    """
    return """
To enable 3D molecular visualization:

Option 1 (Recommended): py3Dmol
  pip install py3dmol

Option 2 (Optional): PlotlyMol
  conda install -c conda-forge rdkit plotly kaleido
  pip install plotlymol

For most users, py3Dmol is sufficient and more stable.
"""


# Module-level check and logging
available = get_available_backends()
if available:
    logger.info(f"Visualization backends available: {', '.join(available)}")
else:
    logger.warning("No visualization backends available")
