"""XYZ coordinate formatting, cleanup, and geometry-only molecule loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .inorganic_guards import check_charge_multiplicity
from .molecule import ATOMIC_NUMBERS, Molecule, parse_xyz_input, suggest_multiplicity


def normalize_element_symbol(symbol: str) -> str:
    """Return a canonical element symbol (e.g. ``cl`` → ``Cl``)."""
    sym = symbol.strip()
    if not sym:
        return sym
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def format_xyz_body(
    atoms: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    *,
    precision: int = 6,
) -> str:
    """Format atom rows as a simple XYZ body (no count/title header)."""
    lines: List[str] = []
    for symbol, coord in zip(atoms, coordinates):
        sym = normalize_element_symbol(symbol)
        x, y, z = (float(coord[0]), float(coord[1]), float(coord[2]))
        lines.append(
            f"{sym:2s}  {x:12.{precision}f}  {y:12.{precision}f}  {z:12.{precision}f}"
        )
    return "\n".join(lines)


def electron_count(atoms: Sequence[str], charge: int) -> int:
    """Total electron count for ``atoms`` at ``charge``."""
    nuclear = sum(ATOMIC_NUMBERS.get(atom, 0) for atom in atoms)
    return nuclear - charge


def spin_compatibility_note(
    atoms: Sequence[str], charge: int, multiplicity: int
) -> str | None:
    """Plain-language note when charge/mult disagree with electron count."""
    return check_charge_multiplicity(electron_count(atoms, charge), multiplicity)


@dataclass(frozen=True)
class ChargeMultSuggestion:
    """Conservative charge/multiplicity guess from atomic composition only."""

    charge: int
    multiplicity: int
    n_electrons: int
    formula: str
    explanation: str
    caveats: Tuple[str, ...]


def suggest_charge_multiplicity_from_atoms(
    atoms: Sequence[str],
    *,
    assume_charge: int = 0,
) -> ChargeMultSuggestion:
    """Suggest charge and multiplicity from element symbols (not coordinates).

    Coordinates do not encode charge or spin. This assumes ``assume_charge``
    (default neutral) and picks the lowest multiplicity compatible with the
    resulting electron count (singlet or doublet by parity).
    """
    from .connectivity import hill_formula

    atom_list = list(atoms)
    if not atom_list:
        raise ValueError("No atoms to suggest charge and multiplicity for.")

    n_electrons = electron_count(atom_list, assume_charge)
    multiplicity = suggest_multiplicity(atom_list, assume_charge)
    formula = hill_formula(atom_list)

    parity_word = "even" if n_electrons % 2 == 0 else "odd"
    spin_word = {1: "singlet", 2: "doublet"}.get(
        multiplicity, f"multiplicity {multiplicity}"
    )
    explanation = (
        f"{formula}: {n_electrons} electrons at charge {assume_charge} "
        f"({parity_word} count) → {spin_word} (multiplicity {multiplicity})."
    )

    caveats: List[str] = [
        "Charge cannot be inferred from coordinates — only atomic symbols are used.",
        "Neutral charge (0) is assumed.",
        "Multiplicity is the lowest parity-compatible value, not necessarily the "
        "ground-state spin (e.g. O₂ is a triplet; many radicals need higher multiplicities).",
    ]
    try:
        from .spin_presets import supported_metals

        if any(sym in set(supported_metals()) for sym in atom_list):
            caveats.append(
                "Transition-metal complexes: use the Spin-state helper below for "
                "oxidation-state-based multiplicity suggestions."
            )
    except Exception:
        pass

    return ChargeMultSuggestion(
        charge=assume_charge,
        multiplicity=multiplicity,
        n_electrons=n_electrons,
        formula=formula,
        explanation=explanation,
        caveats=tuple(caveats),
    )


def load_molecule_from_xyz_text(
    xyz_text: str,
    *,
    charge: int,
    multiplicity: int,
) -> Tuple[Molecule, str | None]:
    """Parse XYZ text into a molecule using the supplied charge and multiplicity.

    Spin parity is not enforced at load time — the pre-run guard catches
    incompatible values when the user clicks Run.
    """
    atoms, coords = parse_xyz_input(xyz_text)
    mol = Molecule(
        atoms,
        coords,
        charge=charge,
        multiplicity=multiplicity,
        validate_spin=False,
    )
    return mol, spin_compatibility_note(atoms, charge, multiplicity)


def _normalize_xyz_line(line: str) -> str:
    """Normalize element capitalization on a single coordinate line."""
    comment = ""
    for ch in ("#", "!"):
        if ch in line:
            idx = line.index(ch)
            comment = line[idx:]
            line = line[:idx]
            break
    parts = line.split()
    if len(parts) >= 4:
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
            parts[0] = normalize_element_symbol(parts[0])
            line = " ".join(parts)
        except ValueError:
            pass
    if comment:
        return f"{line} {comment}".strip()
    return line


def _normalize_xyz_symbols_in_text(xyz_text: str) -> str:
    """Rewrite atom-line element tokens to canonical capitalization."""
    return "\n".join(_normalize_xyz_line(ln) for ln in xyz_text.splitlines())


def propose_xyz_cleanup(xyz_text: str) -> Tuple[str, List[str]]:
    """Parse and reformat XYZ input; return cleaned text and change notes."""
    preprocessed = _normalize_xyz_symbols_in_text(xyz_text)
    atoms, coords = parse_xyz_input(preprocessed)
    normalized_atoms = [normalize_element_symbol(a) for a in atoms]
    cleaned_body = format_xyz_body(normalized_atoms, coords)

    notes: List[str] = []
    raw = xyz_text.strip()
    if raw != cleaned_body:
        try:
            strict_atoms, _ = parse_xyz_input(xyz_text)
        except ValueError:
            strict_atoms = None
        if strict_atoms is None or any(
            a != b for a, b in zip(strict_atoms, normalized_atoms)
        ):
            notes.append("Normalized element symbols (e.g. capitalization).")
        if any("#" in line or "!" in line for line in raw.splitlines()):
            notes.append("Removed comment lines and inline comments.")
        if any(not line.strip() for line in raw.splitlines()):
            notes.append("Removed blank lines.")
        first = next(
            (
                ln.strip()
                for ln in raw.splitlines()
                if ln.strip() and not ln.strip().startswith(("#", "!"))
            ),
            "",
        )
        try:
            int(first)
            notes.append("Stripped XYZ file header (atom count + title).")
        except ValueError:
            pass
        notes.append("Aligned coordinates to a consistent column format.")
    else:
        notes.append("Input was already in standard format.")

    return cleaned_body, notes
