"""Generate the curated library manifest (M-STRUCT STRUCT.7).

Reads the human-editable SMILES seed (scripts/curated_seed.json), generates 3D
coordinates with RDKit (ETKDGv3, seed=42 → MMFF94, UFF fallback), and writes
the full manifest quantui/data/manifests/curated.json (same schema as
presets.json, with coordinates). Then rebuild the SQLite store:

    python scripts/build_curated_library.py
    python scripts/build_library.py

Deterministic (fixed RDKit seed). Molecules that fail to parse/embed or exceed
the curated heavy-atom ceiling are skipped + reported (no silent drops).
"""

import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from quantui import config

RDLogger.DisableLog("rdApp.*")  # quiet InChI/sanitize chatter

SEED = Path("scripts/curated_seed.json")
OUT = Path("quantui/data/manifests/curated.json")
CEILING = config.LIBRARY_HEAVY_ATOM_CEILING_CURATED


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _embed(mol):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("3D embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass  # ship unoptimized rather than fail


def _generate(entry: dict) -> dict:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        raise ValueError("unparseable SMILES")
    mol = Chem.AddHs(mol)
    _embed(mol)

    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    n_heavy = sum(1 for a in atoms if a != "H")
    if n_heavy > CEILING:
        raise ValueError(f"{n_heavy} heavy atoms exceeds ceiling {CEILING}")

    conf = mol.GetConformer()
    coords = [
        [
            round(conf.GetAtomPosition(i).x, 6),
            round(conf.GetAtomPosition(i).y, 6),
            round(conf.GetAtomPosition(i).z, 6),
        ]
        for i in range(mol.GetNumAtoms())
    ]
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    try:
        inchikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol)) or None
    except Exception:
        inchikey = None

    return {
        "id": _slug(entry["name"]),
        "name": entry["name"],
        "formula": formula,
        "category": entry["category"],
        "source": "curated-ff",
        "charge": int(entry.get("charge", Chem.GetFormalCharge(mol))),
        "multiplicity": int(entry.get("multiplicity", 1)),
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "inchikey": inchikey,
        "synonyms": entry.get("synonyms", ""),
        "description": f"{entry['name']} ({formula})",
        "atoms": atoms,
        "coordinates": coords,
    }


def main() -> None:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    out, failures, seen = [], [], set()
    for entry in seed:
        try:
            rec = _generate(entry)
        except Exception as exc:
            failures.append((entry["name"], str(exc)))
            continue
        if rec["id"] in seen:
            failures.append((entry["name"], f"duplicate id '{rec['id']}'"))
            continue
        seen.add(rec["id"])
        out.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)}/{len(seed)} curated entries to {OUT}")
    by_cat: dict = {}
    for r in out:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("By category:", dict(sorted(by_cat.items())))
    if failures:
        print(f"\nSkipped {len(failures)}:")
        for name, why in failures:
            print(f"  - {name}: {why}")


if __name__ == "__main__":
    main()
