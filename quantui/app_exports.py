"""Export helpers used by QuantUIApp."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        fname = (
            f"{app._molecule.get_formula()}"
            f"_{app.method_dd.value}_{app.basis_dd.value}.py"
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
        fname = f"{mol.get_formula()}_{method}_{basis}.xyz"
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
        fname = f"{mol.get_formula()}_{method}_{basis}.mol"
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
        fname = f"{mol.get_formula()}_{method}_{basis}.pdb"
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
