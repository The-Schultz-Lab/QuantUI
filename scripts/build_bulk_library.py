"""Generate the bulk QM9 library manifest (M-STRUCT STRUCT.8).

Downloads the QM9 dataset (Ramakrishnan et al., Sci. Data 2014; GDB-9 subset;
**CC0**), parses the 133,885 B3LYP/6-31G(2df,p)-optimized structures, filters +
dedups against the curated set, stratified-samples for chemical diversity, and
writes quantui/data/manifests/bulk_qm9.json (category="bulk-qm9",
source="qm9-dft"). Then rebuild the store:

    python scripts/build_bulk_library.py [--target N]
    python scripts/build_library.py

Bulk entries are identified by formula + QM9 index and surface via search only
(excluded from the browsable preset dict). Deterministic: stratified sampling
draws in a fixed (formula-sorted, index-ordered) order — no RNG.

The QM9 tarball (~92 MB) is cached under the system temp dir, so re-runs don't
re-download. Maintainer build step; the generated manifest + store are what ship.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import gettempdir

from rdkit import Chem, RDLogger

from quantui import config

RDLogger.DisableLog("rdApp.*")

# figshare ndownloader file ids for dataset 978904.
QM9_TARBALL_URL = (
    "https://ndownloader.figshare.com/files/3195389"  # dsgdb9nsd.xyz.tar.bz2
)
QM9_UNCHAR_URL = "https://ndownloader.figshare.com/files/3195404"  # uncharacterized.txt
_CACHE = Path(gettempdir()) / "quantui_qm9_cache"

OUT = Path("quantui/data/manifests/bulk_qm9.json")
PROVENANCE = Path("quantui/data/library/QM9-PROVENANCE.md")
CURATED = Path("quantui/data/manifests/curated.json")

DEFAULT_TARGET = 2500  # ~0.6 MB store (demo cap; raise to fill toward 10 MB)
PER_FORMULA_CAP = 4  # keep at most N isomers per formula (diversity + bounded memory)
CEILING = config.LIBRARY_HEAVY_ATOM_CEILING_BULK


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "quantui-build"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  {dest.stat().st_size / 1024 / 1024:.1f} MB")
    return dest


def _hill_formula(atoms: list) -> str:
    c = Counter(atoms)
    parts = []
    for sym in ("C", "H"):
        if c.get(sym):
            parts.append(sym + (str(c[sym]) if c[sym] > 1 else ""))
            del c[sym]
    for sym in sorted(c):
        parts.append(sym + (str(c[sym]) if c[sym] > 1 else ""))
    return "".join(parts)


def _parse_qm9_xyz(text: str):
    """Parse one QM9 .xyz → (index, atoms, coords, smiles, inchi).

    Layout: natoms / props / n atom lines / frequencies / SMILES line /
    InChI line. The SMILES and InChI lines each carry two values (GDB17 and
    B3LYP-relaxed); we take the relaxed (last) of each.
    """
    lines = text.splitlines()
    n = int(lines[0])
    index = int(lines[1].split()[1])  # 'gdb <index> <props...>'
    atoms, coords = [], []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        atoms.append(parts[0])
        # QM9 uses Fortran exponent notation like "1.23*^-6".
        xyz = [float(p.replace("*^", "e")) for p in parts[1:4]]
        coords.append([round(v, 6) for v in xyz])
    smiles_line = lines[2 + n + 1].split()  # after the frequencies line
    smiles = smiles_line[-1] if smiles_line else None
    inchi_line = lines[2 + n + 2].split()
    inchi = inchi_line[-1] if inchi_line else ""
    return index, atoms, coords, smiles, inchi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    args = ap.parse_args()

    tarball = _download(QM9_TARBALL_URL, _CACHE / "dsgdb9nsd.xyz.tar.bz2")
    sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    print(f"QM9 tarball sha256: {sha}")

    unchar_path = _download(QM9_UNCHAR_URL, _CACHE / "uncharacterized.txt")
    uncharacterized = set()
    for ln in unchar_path.read_text(encoding="utf-8", errors="replace").splitlines():
        toks = ln.split()
        if toks and toks[0].isdigit():
            uncharacterized.add(int(toks[0]))
    print(f"Excluding {len(uncharacterized)} uncharacterized molecules")

    curated_keys = {
        e["inchikey"]
        for e in json.loads(CURATED.read_text(encoding="utf-8"))
        if e.get("inchikey")
    }

    # Pass: bucket candidates by Hill formula (bounded per formula).
    buckets: dict = defaultdict(list)
    n_seen = n_kept = 0
    with tarfile.open(tarball, mode="r:bz2") as tar:
        for member in tar:
            if not member.name.endswith(".xyz"):
                continue
            n_seen += 1
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            try:
                index, atoms, coords, smiles, inchi = _parse_qm9_xyz(
                    fobj.read().decode("utf-8", "replace")
                )
            except Exception:
                continue
            if index in uncharacterized:
                continue
            if sum(1 for a in atoms if a != "H") > CEILING:
                continue
            formula = _hill_formula(atoms)
            if len(buckets[formula]) < PER_FORMULA_CAP:
                buckets[formula].append((index, atoms, coords, smiles, inchi))
                n_kept += 1
    print(f"Scanned {n_seen} molecules; {len(buckets)} formulas, {n_kept} bucketed")

    # Stratified round-robin draw: one isomer per formula per round, formula-sorted.
    order = []
    for r in range(PER_FORMULA_CAP):
        for formula in sorted(buckets):
            if r < len(buckets[formula]):
                order.append((formula, buckets[formula][r]))

    out, seen_keys, skipped_dup = [], set(), 0
    for formula, (index, atoms, coords, smiles, inchi) in order:
        if len(out) >= args.target:
            break
        try:
            key = Chem.InchiToInchiKey(inchi) if inchi.startswith("InChI=") else None
        except Exception:
            key = None
        if key and (key in curated_keys or key in seen_keys):
            skipped_dup += 1
            continue
        if key:
            seen_keys.add(key)
        out.append(
            {
                "id": f"qm9-{index}",
                "name": f"{formula} (QM9 #{index})",
                "formula": formula,
                "category": "bulk-qm9",
                "source": "qm9-dft",
                "charge": 0,
                "multiplicity": 1,
                "smiles": smiles,
                "inchikey": key,
                "synonyms": "",
                "description": f"{formula} — QM9 #{index} (B3LYP/6-31G(2df,p))",
                "atoms": atoms,
                "coordinates": coords,
            }
        )

    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} bulk entries to {OUT} (deduped {skipped_dup} vs curated)")

    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(
        "# QM9 bulk-library provenance\n\n"
        f"- Entries shipped: {len(out)} (cap {args.target}; "
        f"`category=bulk-qm9`, `source=qm9-dft`)\n"
        f"- Source: QM9 / GDB-9 (figshare dataset 978904)\n"
        f"- Tarball sha256: `{sha}`\n"
        "- Geometries: B3LYP/6-31G(2df,p) optimized\n"
        "- License: CC0 (public domain)\n"
        "- Citation: Ramakrishnan, Dral, Rupp & von Lilienfeld, "
        "*Sci. Data* **1**, 140022 (2014).\n"
        "- Selection: stratified by Hill formula (≤"
        f"{CEILING} heavy atoms, uncharacterized excluded, "
        "InChIKey-deduped vs the curated set), deterministic round-robin.\n"
        "- Regenerate: `python scripts/build_bulk_library.py --target N`\n",
        encoding="utf-8",
    )
    print(f"Wrote provenance to {PROVENANCE}")


if __name__ == "__main__":
    main()
