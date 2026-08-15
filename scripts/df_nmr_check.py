#!/usr/bin/env python3
"""Does density fitting move NMR *chemical shifts*? — M-DF, DF.5.

DF shifts *absolute* GIAO shieldings, but QuantUI reports *chemical shifts*
(δ = σ_TMS − σ_atom), a difference in which a systematic DF error should largely
cancel. ``config.NMR_REFERENCE_SHIELDINGS`` is a hardcoded σ_TMS table, so the
question that decides whether DF is safe for NMR — and whether that table needs
regenerating under DF — is: **how much does δ change when DF is on?**

This standalone tool answers it. For each molecule it computes isotropic ¹H and
¹³C shieldings with and without DF (mirroring ``quantui.nmr_calc``:
``pyscf.prop.nmr`` GIAO on an RKS mean field), computes δ against a TMS
reference calculated the same way, and reports **Δδ = |δ_df − δ_plain|** per
element. Small Δδ (<< 0.1 ppm) confirms the cancellation and means the shipped
reference table survives DF; a large Δδ means DF must not be defaulted on for
NMR without regenerating the references.

Run at the reference level of theory on an NCShare / local node:

    python scripts/df_nmr_check.py --method b3lyp --basis 6-31G* \
        --molecules methanol,benzene --csv df_nmr.csv

Also prints whether the GIAO code even *accepts* a density-fitted mean field in
this PySCF build — itself a DF.5 finding.
"""

from __future__ import annotations

import argparse
import csv as _csv
from dataclasses import dataclass, field
from typing import Optional

_DEFAULT_SMILES: dict[str, str] = {
    "methane": "C",
    "methanol": "CO",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "acetone": "CC(=O)C",
}
_TMS_SMILES = "C[Si](C)(C)C"  # tetramethylsilane, the NMR reference
_ELEMENTS = ("H", "C")


def _geometry_from_smiles(smiles: str):
    """Return (symbols, atom_block) with an RDKit ETKDG + MMFF geometry."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit could not embed {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols, lines = [], []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        symbols.append(atom.GetSymbol())
        lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return symbols, "\n".join(lines)


def _shieldings(atom_block: str, symbols, method: str, basis: str, density_fit: bool):
    """Return {element: mean isotropic shielding (ppm)} or raise.

    Mirrors quantui.nmr_calc: RKS mean field (optionally density-fitted) then
    pyscf.prop.nmr GIAO; isotropic = trace/3 of each atom's tensor.
    """
    import numpy as np
    from pyscf import dft, gto
    from pyscf.prop import nmr as _nmr

    # Apply QuantUI's GIAO compatibility patch — pyscf.prop.nmr's RKS GIAO path
    # asserts on a grid blocksize that isn't a BLKSIZE multiple; nmr_calc patches
    # get_vxc_giao to fix it. Without this, both plain AND DF runs crash, so the
    # measurement must mirror the real path.
    from quantui.nmr_calc import _ensure_nmr_compat_patches_applied

    _ensure_nmr_compat_patches_applied()

    mol = gto.M(atom=atom_block, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = method
    mf.verbose = 0
    if density_fit:
        mf = mf.density_fit()
    mf.kernel()

    tensors = _nmr.RKS(mf).kernel()  # (natm, 3, 3)
    per_element: dict[str, list[float]] = {e: [] for e in _ELEMENTS}
    for sym, tensor in zip(symbols, tensors):
        if sym in per_element:
            per_element[sym].append(float(np.trace(tensor) / 3.0))
    return {e: (sum(v) / len(v)) for e, v in per_element.items() if v}


@dataclass
class Row:
    molecule: str
    element: str
    sigma_plain: Optional[float] = None
    sigma_df: Optional[float] = None
    delta_plain: Optional[float] = None
    delta_df: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def ddelta(self) -> Optional[float]:
        if self.delta_plain is not None and self.delta_df is not None:
            return abs(self.delta_df - self.delta_plain)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Check whether density fitting moves NMR chemical shifts "
        "(M-DF DF.5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--molecules",
        default="methanol,benzene",
        help=f"Comma list of known names and/or name:SMILES (known: "
        f"{', '.join(_DEFAULT_SMILES)}).",
    )
    ap.add_argument("--method", default="b3lyp", help="Functional (default b3lyp).")
    ap.add_argument(
        "--basis",
        default="6-31G*",
        help="Basis (default 6-31G*, matching NMR_DEFAULT_REFERENCE).",
    )
    ap.add_argument("--csv", help="Write per-row results to this CSV path.")
    args = ap.parse_args(argv)

    # Parse molecule set.
    mols: dict[str, str] = {}
    for tok in args.molecules.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, smi = tok.split(":", 1)
            mols[name.strip()] = smi.strip()
        elif tok in _DEFAULT_SMILES:
            mols[tok] = _DEFAULT_SMILES[tok]
        else:
            raise SystemExit(f"Unknown molecule {tok!r}.")

    print(f"# df_nmr_check — {args.method}/{args.basis}")
    print("# delta = sigma_TMS - sigma_atom (ppm); Ddelta = |delta_df - delta_plain|")
    print("# Small Ddelta confirms the DF error cancels in the shift (DF.5).\n")

    # TMS reference, both ways. If DF fails here, the whole DF-NMR path is out.
    print("# computing TMS reference …", flush=True)
    tms_symbols, tms_geom = _geometry_from_smiles(_TMS_SMILES)
    tms_ref: dict[bool, dict[str, float]] = {}
    df_supported = True
    for df_on in (False, True):
        try:
            tms_ref[df_on] = _shieldings(
                tms_geom, tms_symbols, args.method, args.basis, df_on
            )
        except Exception as exc:  # noqa: BLE001
            if df_on:
                df_supported = False
                print(f"#   !! DF NMR unsupported in this PySCF build: {exc}")
            else:
                raise SystemExit(f"TMS reference (plain) failed: {exc}")

    rows: list[Row] = []
    for name, smi in mols.items():
        print(f"# computing {name} …", flush=True)
        try:
            symbols, geom = _geometry_from_smiles(smi)
        except Exception as exc:  # noqa: BLE001
            rows.append(Row(molecule=name, element="-", notes=[f"geom: {exc}"]))
            continue
        sig: dict[bool, dict[str, float]] = {}
        for df_on in (False, True) if df_supported else (False,):
            try:
                sig[df_on] = _shieldings(geom, symbols, args.method, args.basis, df_on)
            except Exception as exc:  # noqa: BLE001
                print(f"#   !! {name} {'df' if df_on else 'plain'} failed: {exc}")
        for element in _ELEMENTS:
            if element not in sig.get(False, {}):
                continue
            row = Row(molecule=name, element=element)
            row.sigma_plain = sig[False][element]
            row.delta_plain = tms_ref[False][element] - row.sigma_plain
            if df_supported and element in sig.get(True, {}):
                row.sigma_df = sig[True][element]
                row.delta_df = tms_ref[True][element] - row.sigma_df
            rows.append(row)

    # Table.
    hdr = ["molecule", "elem", "sig_plain", "sig_df", "d_plain", "d_df", "Ddelta"]
    widths = [12, 5, 10, 10, 9, 9, 9]

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    print("\n" + "  ".join(f"{h:<{w}}" for h, w in zip(hdr, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        vals = [
            r.molecule,
            r.element,
            fmt(r.sigma_plain) if r.sigma_plain is not None else "—",
            fmt(r.sigma_df) if r.sigma_df is not None else "—",
            fmt(r.delta_plain) if r.delta_plain is not None else "—",
            fmt(r.delta_df) if r.delta_df is not None else "—",
            fmt(r.ddelta) if r.ddelta is not None else "—",
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(vals, widths)))
        for note in r.notes:
            print(f"    ! {r.molecule}: {note}")

    dds = [r.ddelta for r in rows if r.ddelta is not None]
    print()
    if not df_supported:
        print("# VERDICT: this PySCF build's GIAO code rejects a density-fitted")
        print(
            "#   mean field, so DF cannot be enabled for NMR as written — "
            "record that\n#   and skip DF for NMR (DF.5)."
        )
    elif dds:
        worst = max(dds)
        verdict = (
            "cancels — reference table survives DF"
            if worst < 0.1
            else (
                "does NOT cancel — regenerate NMR_REFERENCE_SHIELDINGS under DF "
                "before enabling"
            )
        )
        print(f"# VERDICT: max Ddelta = {worst:.4f} ppm → {verdict} (DF.5).")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(hdr + ["notes"])
            for r in rows:
                w.writerow(
                    [
                        r.molecule,
                        r.element,
                        r.sigma_plain,
                        r.sigma_df,
                        r.delta_plain,
                        r.delta_df,
                        r.ddelta,
                        "; ".join(r.notes),
                    ]
                )
        print(f"# wrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
