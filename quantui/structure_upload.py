"""Read an uploaded structure file into a Molecule (DEC-023 Tier-1 #1, INT.4).

Formats and readers:

- ``.xyz`` (last frame of a multi-frame file), ``.pdb``, ``.cif``,
  Gaussian input ``.gjf``/``.com`` and Gaussian output ``.log``/``.out``
  (final geometry): ASE.
- ``.mol``, ``.sdf`` (first record), ``.mol2``: RDKit, which keeps formal
  charges and radical electrons, so charge and multiplicity come from the
  file. 2-D MOL/SDF drawings are embedded in 3-D (ETKDG + MMFF).

Charge and multiplicity come from the file when it records them (Gaussian
charge/multiplicity line, MOL/SDF formal charges and radicals, a QuantUI XYZ
comment ``charge=… multiplicity=…``); otherwise the caller's values are
used and the result says so.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .molecule import Molecule

SUPPORTED_SUFFIXES = (
    ".xyz",
    ".mol",
    ".sdf",
    ".mol2",
    ".pdb",
    ".cif",
    ".gjf",
    ".com",
    ".log",
    ".out",
)
#: Uploads larger than this are refused (structure files are small).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_CHARGE_MULT_COMMENT = re.compile(
    r"charge\s*=\s*(-?\d+).*?mult(?:iplicity)?\s*=\s*(\d+)", re.IGNORECASE
)
_GAUSSIAN_LOG_CHARGE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")


@dataclass
class UploadedStructure:
    molecule: Molecule
    charge_mult_from_file: bool
    notes: List[str]


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).name).strip("._")
    return stem or "upload"


def _gaussian_input_charge_mult(text: str) -> Optional[Tuple[int, int]]:
    """Charge/multiplicity line of a Gaussian input (after route + title)."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("%"):
            continue  # Link 0 lines sit above the route section
        current.append(line)
    if current:
        blocks.append(current)
    # blocks: [route (# ...), title, molecule spec, ...]
    for i, block in enumerate(blocks):
        if block and block[0].startswith("#") and i + 2 < len(blocks):
            fields = blocks[i + 2][0].split()
            try:
                return int(fields[0]), int(fields[1])
            except (IndexError, ValueError):
                return None
    return None


def _from_rdkit(path: Path, suffix: str, notes: List[str]) -> Tuple[Molecule, bool]:
    from rdkit import Chem

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".mol2":
        rdmol = Chem.MolFromMol2Block(text, removeHs=False)
    elif suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        rdmol = next((m for m in supplier if m is not None), None)
        if len(supplier) > 1:
            notes.append(f"SDF has {len(supplier)} records; loaded the first.")
    else:
        rdmol = Chem.MolFromMolBlock(text, removeHs=False)
    if rdmol is None:
        raise ValueError("RDKit could not read this file.")
    conf = rdmol.GetConformer() if rdmol.GetNumConformers() else None
    flat = conf is None or all(
        abs(conf.GetAtomPosition(i).z) < 1e-4 for i in range(rdmol.GetNumAtoms())
    )
    if flat and rdmol.GetNumAtoms() > 3:
        from rdkit.Chem import AllChem

        rdmol = Chem.AddHs(rdmol, addCoords=True)
        if AllChem.EmbedMolecule(rdmol, randomSeed=0xC0FFEE) != 0:
            raise ValueError("2-D structure could not be embedded in 3-D.")
        try:
            AllChem.MMFFOptimizeMolecule(rdmol)
        except Exception:  # noqa: BLE001 — embedding alone is still usable
            pass
        conf = rdmol.GetConformer()
        notes.append(
            "The file had 2-D coordinates; built a 3-D structure (RDKit "
            "ETKDG + MMFF). Optimize it before trusting energies."
        )
    assert conf is not None
    atoms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    coords = [list(conf.GetAtomPosition(i)) for i in range(rdmol.GetNumAtoms())]
    charge = sum(a.GetFormalCharge() for a in rdmol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in rdmol.GetAtoms())
    mol = Molecule(
        atoms,
        coords,
        charge=int(charge),
        multiplicity=int(radicals) + 1,
        validate_spin=False,
    )
    return mol, True


def _from_ase(
    path: Path, suffix: str, charge: int, multiplicity: int, notes: List[str]
) -> Tuple[Molecule, bool]:
    import ase.io

    fmt = {
        ".gjf": "gaussian-in",
        ".com": "gaussian-in",
        ".log": "gaussian-out",
        ".out": "gaussian-out",
    }.get(suffix)
    text = path.read_text(encoding="utf-8", errors="replace")
    from_file = False
    if suffix in (".gjf", ".com"):
        cm = _gaussian_input_charge_mult(text)
        if cm:
            charge, multiplicity = cm
            from_file = True
    elif suffix in (".log", ".out"):
        found = _GAUSSIAN_LOG_CHARGE.findall(text)
        if found:
            charge, multiplicity = int(found[0][0]), int(found[0][1])
            from_file = True
        notes.append("Loaded the last geometry in the Gaussian output.")
    elif suffix == ".xyz":
        lines = text.splitlines()
        if len(lines) > 1:
            m = _CHARGE_MULT_COMMENT.search(lines[1])
            if m:
                charge, multiplicity = int(m.group(1)), int(m.group(2))
                from_file = True
    frames = ase.io.read(str(path), index=":", format=fmt)
    if not frames:
        raise ValueError("No structure found in the file.")
    if len(frames) > 1:
        notes.append(f"File has {len(frames)} structures; loaded the last one.")
    atoms_obj = frames[-1]
    if suffix == ".cif" and any(atoms_obj.pbc):
        notes.append(
            "CIF unit cell contents loaded as a molecule; periodic systems are "
            "not supported, so check it is one whole molecule."
        )
    mol = Molecule(
        list(atoms_obj.get_chemical_symbols()),
        atoms_obj.get_positions().tolist(),
        charge=int(charge),
        multiplicity=int(multiplicity),
        validate_spin=False,
    )
    return mol, from_file


def read_uploaded_structure(
    filename: str,
    content: bytes,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> UploadedStructure:
    """Parse an uploaded file's bytes into a :class:`UploadedStructure`.

    *charge*/*multiplicity* are used when the file does not record them.
    Raises ``ValueError`` with a student-readable message on any failure.
    """
    safe = _safe_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix or safe}'. Supported: "
            + ", ".join(SUPPORTED_SUFFIXES)
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large for a structure upload (limit 20 MB).")
    notes: List[str] = []
    with tempfile.TemporaryDirectory(prefix="quantui-upload-") as tmp:
        path = Path(tmp) / safe
        path.write_bytes(content)
        try:
            if suffix in (".mol", ".sdf", ".mol2"):
                mol, from_file = _from_rdkit(path, suffix, notes)
            else:
                mol, from_file = _from_ase(path, suffix, charge, multiplicity, notes)
        except ImportError as exc:
            raise ValueError(
                f"Reading {suffix} files needs {exc.name or 'an optional package'}."
            ) from exc
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 — reader errors vary widely
            raise ValueError(f"Could not read {safe}: {exc}") from exc
    if not mol.atoms:
        raise ValueError(f"{safe} contains no atoms.")
    if not from_file:
        notes.append(
            "The file does not record charge/multiplicity; using the values "
            "from Calculation Setup."
        )
    return UploadedStructure(molecule=mol, charge_mult_from_file=from_file, notes=notes)
