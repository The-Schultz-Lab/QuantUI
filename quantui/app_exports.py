"""Export helpers used by QuantUIApp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .results_storage import _safe_name


def on_export(app: Any, btn: Any) -> None:
    """Export a standalone Python calculation script."""
    if app._molecule is None:
        app.export_status.value = "Load a molecule first."
        return
    try:
        from quantui import PySCFCalculation

        calc = PySCFCalculation(
            app._molecule,
            method=app.method_dd.value,
            basis=app.basis_dd.value,
        )
        # M11 audit fix (2026-07-14): the basis set is embedded verbatim in
        # the filename (e.g. "6-31G*.py"), and "*" is invalid in a Windows
        # filename — this export silently failed there. _safe_name (already
        # used by results_storage for the same purpose) replaces anything
        # that isn't alphanumeric/underscore/hyphen with "x".
        fname = (
            f"{_safe_name(app._molecule.get_formula())}"
            f"_{_safe_name(app.method_dd.value)}_{_safe_name(app.basis_dd.value)}.py"
        )
        calc.generate_calculation_script(Path(fname))
        app.export_status.value = f"Saved: {fname}"
    except Exception as exc:
        app.export_status.value = f"Error: {exc}"


def on_export_xyz(app: Any, btn: Any) -> None:
    """Export molecule geometry to an XYZ file."""
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.xyz"
        xyz_body = mol.to_xyz_string()
        full_xyz = (
            f"{len(mol.atoms)}\n{mol.get_formula()} {method}/{basis}\n{xyz_body}\n"
        )
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(full_xyz, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def on_export_mol(app: Any, btn: Any) -> None:
    """Export molecule geometry to a MOL file via RDKit."""
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        from rdkit import Chem

        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.mol"
        rdmol = molecule_to_rdkit(mol)
        if rdmol is None:
            app.struct_export_status.value = "RDKit could not parse the structure."
            return
        mol_block = Chem.MolToMolBlock(rdmol)
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(mol_block, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def on_export_pdb(app: Any, btn: Any) -> None:
    """Export molecule geometry to a PDB file via RDKit."""
    if app._molecule is None:
        app.struct_export_status.value = "Load a molecule first."
        return
    try:
        from rdkit import Chem

        mol, method, basis = export_molecule_and_label(app)
        fname = f"{_safe_name(mol.get_formula())}_{_safe_name(method)}_{_safe_name(basis)}.pdb"
        rdmol = molecule_to_rdkit(mol)
        if rdmol is None:
            app.struct_export_status.value = "RDKit could not parse the structure."
            return
        pdb_block = Chem.MolToPDBBlock(rdmol)
        dest = (app._last_result_dir / fname) if app._last_result_dir else Path(fname)
        dest.write_text(pdb_block, encoding="utf-8")
        app.struct_export_status.value = f"Saved: {dest}"
    except Exception as exc:
        app.struct_export_status.value = f"Error: {exc}"


def export_molecule_and_label(app: Any) -> tuple[Any, str, str]:
    """Return (molecule, method, basis) for structure export.

    For geometry optimization results, returns the final optimized geometry.
    Falls back to the currently loaded molecule for all other calculation types.
    """
    from quantui.optimizer import OptimizationResult

    result = app._last_result
    if isinstance(result, OptimizationResult):
        mol = result.molecule
    else:
        assert app._molecule is not None
        mol = app._molecule
    method = (
        getattr(result, "method", app.method_dd.value)
        if result is not None
        else app.method_dd.value
    )
    basis = (
        getattr(result, "basis", app.basis_dd.value)
        if result is not None
        else app.basis_dd.value
    )
    return mol, method, basis


# Data URIs from 3Dmol's pngURI(). Anchored and length-bounded: this value
# arrives from the browser, and while it is the user's own page rather than an
# untrusted party, decoding whatever lands in a widget traitlet without
# checking its shape is not a habit worth having.
logger = logging.getLogger(__name__)

_PNG_URI_PREFIX = "data:image/png;base64,"
_MAX_PNG_BYTES = 64 * 1024 * 1024


def on_orb_png_captured(app: Any, change: dict) -> None:
    """Write a PNG captured from the live isosurface viewer (ORBX.1).

    Fires when the viewer's Save-PNG button writes a data URI into the hidden
    inbox Textarea. The image is whatever the user is actually looking at,
    camera included — which is the entire reason capture happens client-side
    rather than by re-rendering server-side.

    The inbox is cleared afterwards so saving the same view twice re-triggers
    the traitlet (an unchanged value would not fire ``observe``).
    """
    import base64
    import binascii

    uri = (change or {}).get("new") or ""
    if not uri:
        return

    def _fail(msg: str) -> None:
        app._iso_export_status.value = f'<span style="color:#b22">{msg}</span>'
        # Clear even on failure, or a retry of the identical capture is silent.
        _clear_inbox(app)

    def _clear_inbox(a: Any) -> None:
        box = getattr(a, "_orb_png_inbox", None)
        if box is not None and box.value:
            box.value = ""

    if not uri.startswith(_PNG_URI_PREFIX):
        logger.warning("PNG capture: unexpected data URI prefix")
        _fail("Capture failed (unexpected image format).")
        return
    if len(uri) > _MAX_PNG_BYTES:
        _fail("Capture failed (image too large).")
        return

    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is None or not isinstance(result_dir, Path):
        _fail("No result folder available.")
        return

    try:
        raw = base64.b64decode(uri[len(_PNG_URI_PREFIX) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("PNG capture: could not decode payload: %s", exc)
        _fail("Capture failed (corrupt image data).")
        return

    label = getattr(app, "_last_cube_orbital", None) or "orbital"
    safe = "".join(c if c.isalnum() or c in "-_+" else "_" for c in str(label))
    dest = Path(result_dir) / f"{safe}.png"
    try:
        dest.write_bytes(raw)
    except OSError as exc:
        logger.warning("PNG capture: could not write %s: %s", dest, exc)
        _fail("Could not write the image (see log).")
        return

    logger.info("Saved orbital PNG: %s (%d bytes)", dest, len(raw))
    app._iso_export_status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'
    _clear_inbox(app)


def on_iso_export_cube(app: Any, btn: Any) -> None:
    """Copy the last-generated cube file to the result folder.

    Reads ``app._last_cube_path`` (set by the isosurface render path in
    ``app_visualization.py``) and copies it to
    ``<result_dir>/<orbital_label>.cube`` so the user can hand a
    friendly-named cube to Avogadro / VMD / Multiwfn without scrolling
    through ``isosurfaces/<formula>_<orb>_<timestamp>.cube``.
    """
    from quantui.results_storage import export_cube

    src = getattr(app, "_last_cube_path", None)
    label = getattr(app, "_last_cube_orbital", None) or "orbital"
    result_dir = getattr(app, "_last_result_dir", None)
    if src is None or not isinstance(src, Path) or not src.exists():
        app._iso_export_status.value = (
            '<span style="color:#b22">Generate an isosurface first.</span>'
        )
        return
    if result_dir is None or not isinstance(result_dir, Path):
        app._iso_export_status.value = (
            '<span style="color:#b22">No result folder available.</span>'
        )
        return
    dest = export_cube(src, result_dir, orbital_label=label)
    if dest is None:
        app._iso_export_status.value = (
            '<span style="color:#b22">Cube export failed (see log).</span>'
        )
        return
    app._iso_export_status.value = f'<span style="color:#2a7">Saved: {dest.name}</span>'


def on_export_bundle(app: Any, btn: Any) -> None:
    """Zip the entire result folder for sharing."""
    from quantui.results_storage import export_result_bundle

    result_dir = getattr(app, "_last_result_dir", None)
    if result_dir is None or not isinstance(result_dir, Path):
        app._export_bundle_status.value = "Run or load a calculation first."
        return
    out_path = export_result_bundle(result_dir)
    if out_path is None:
        app._export_bundle_status.value = "Bundle export failed (see log)."
        return
    app._export_bundle_status.value = f"Saved: {out_path}"


def molecule_to_rdkit(mol: Any) -> Any:
    """Convert a Molecule to an RDKit Mol with inferred bonds (best-effort)."""
    try:
        from rdkit import Chem

        xyz_block = f"{len(mol.atoms)}\n{mol.get_formula()}\n{mol.to_xyz_string()}\n"
        rdmol = Chem.MolFromXYZBlock(xyz_block)
        if rdmol is None:
            return None
        try:
            from rdkit.Chem import rdDetermineBonds

            rdDetermineBonds.DetermineBonds(rdmol, charge=mol.charge)
        except Exception:
            pass
        return rdmol
    except Exception:
        return None
