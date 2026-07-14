"""
QuantUI Molecule Module

Handles molecule input, validation, and coordinate processing.
Provides classes and functions for representing molecular systems.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from . import config, utils

logger = logging.getLogger(__name__)

# Single source of truth lives in config.ATOMIC_NUMBERS (covers the full
# periodic table, Z=1..118, and derives config.VALID_ATOMS so the two can't
# drift apart). Re-exported here so existing `from .molecule import
# ATOMIC_NUMBERS` call sites (e.g. orbital_visualization's charge/spin
# inference) keep working unchanged.
ATOMIC_NUMBERS: Dict[str, int] = config.ATOMIC_NUMBERS


class Molecule:
    """
    Represents a molecular system with atoms, coordinates, charge, and multiplicity.

    Provides validation and formatting methods for quantum chemistry calculations.
    """

    def __init__(
        self,
        atoms: List[str],
        coordinates: List[List[float]],
        charge: int = 0,
        multiplicity: int = 1,
    ):
        """
        Initialize a molecule.

        Args:
            atoms: List of atomic symbols (e.g., ['H', 'H', 'O'])
            coordinates: List of [x, y, z] coordinates in Angstroms
            charge: Total molecular charge
            multiplicity: Spin multiplicity (2S+1)

        Raises:
            ValueError: If molecule data is invalid
        """
        self.atoms = atoms
        self.coordinates = coordinates
        self.charge = charge
        self.multiplicity = multiplicity

        # Validate molecule
        self._validate()

        logger.info(
            f"Created molecule: {self.get_formula()} (charge={charge}, mult={multiplicity})"
        )

    def _validate(self):
        """
        Validate molecular data.

        Raises:
            ValueError: If any validation fails
        """
        # Check atoms and coordinates have same length
        if len(self.atoms) != len(self.coordinates):
            raise ValueError(
                f"Number of atoms ({len(self.atoms)}) does not match "
                f"number of coordinates ({len(self.coordinates)})"
            )

        # Check at least one atom
        if len(self.atoms) == 0:
            raise ValueError("Molecule must have at least one atom")

        # Validate each atom symbol
        for i, atom in enumerate(self.atoms):
            if not utils.validate_atom_symbol(atom):
                raise ValueError(
                    f"Invalid atom symbol '{atom}' at position {i+1}. "
                    f"Must be a valid element symbol."
                )

        # Validate each coordinate
        for i, coord in enumerate(self.coordinates):
            if not utils.validate_coordinates(coord):
                raise ValueError(
                    f"Invalid coordinates at position {i+1}. "
                    f"Must be a list of 3 numbers: [x, y, z]"
                )

        # Validate charge
        if not utils.validate_charge(self.charge):
            raise ValueError(
                f"Invalid charge {self.charge}. "
                f"Must be an integer between -10 and 10."
            )

        # Validate multiplicity
        if not utils.validate_multiplicity(self.multiplicity):
            raise ValueError(
                f"Invalid multiplicity {self.multiplicity}. "
                f"Must be a positive integer."
            )

        # Check multiplicity compatibility with electron count
        num_electrons = self.get_electron_count()
        if (num_electrons + self.multiplicity) % 2 != 1:
            # Generate list of valid multiplicities (up to 5 options)
            valid_mults = []
            if num_electrons % 2 == 0:
                # Even electrons -> odd multiplicities (1, 3, 5, 7, 9)
                valid_mults = [1, 3, 5, 7, 9]
                explanation = "even number of electrons requires odd multiplicity"
            else:
                # Odd electrons -> even multiplicities (2, 4, 6, 8, 10)
                valid_mults = [2, 4, 6, 8, 10]
                explanation = "odd number of electrons requires even multiplicity"

            # Format valid options nicely
            valid_str = ", ".join(str(m) for m in valid_mults)

            raise ValueError(
                f"❌ Multiplicity Error: Multiplicity {self.multiplicity} is incompatible with "
                f"{num_electrons} electrons.\n\n"
                f"💡 Explanation: This molecule has {num_electrons} electrons (an "
                f"{'even' if num_electrons % 2 == 0 else 'odd'} number), so the {explanation}.\n\n"
                f"✅ Valid multiplicities for this molecule: {valid_str}\n\n"
                f"Common values:\n"
                f"  • Multiplicity 1 (singlet) = all electrons paired\n"
                f"  • Multiplicity 2 (doublet) = 1 unpaired electron (radical)\n"
                f"  • Multiplicity 3 (triplet) = 2 unpaired electrons\n"
            )

    def get_electron_count(self) -> int:
        """
        Calculate total number of electrons.

        Returns:
            int: Number of electrons (nuclear charges - charge)
        """
        nuclear_charge = sum(ATOMIC_NUMBERS.get(atom, 0) for atom in self.atoms)
        return nuclear_charge - self.charge

    def get_formula(self) -> str:
        """
        Get molecular formula (e.g., 'H2O', 'CH4').

        Returns:
            str: Molecular formula
        """
        # Count atoms
        atom_counts: dict[str, int] = {}
        for atom in self.atoms:
            atom_counts[atom] = atom_counts.get(atom, 0) + 1

        # Build formula (C, H, then alphabetical)
        formula_parts = []

        # Carbon first (if present)
        if "C" in atom_counts:
            count = atom_counts["C"]
            formula_parts.append(f"C{count if count > 1 else ''}")
            del atom_counts["C"]

        # Hydrogen second (if present)
        if "H" in atom_counts:
            count = atom_counts["H"]
            formula_parts.append(f"H{count if count > 1 else ''}")
            del atom_counts["H"]

        # Rest alphabetically
        for atom in sorted(atom_counts.keys()):
            count = atom_counts[atom]
            formula_parts.append(f"{atom}{count if count > 1 else ''}")

        return "".join(formula_parts)

    def to_pyscf_format(self) -> str:
        """
        Format molecule for PySCF input.

        Returns:
            str: Molecule string in PySCF format (atom symbol, x, y, z)
        """
        lines = []
        for atom, coord in zip(self.atoms, self.coordinates):
            x, y, z = coord
            lines.append(f"{atom:2s}  {x:12.8f}  {y:12.8f}  {z:12.8f}")

        return "\n".join(lines)

    def to_xyz_string(self) -> str:
        """
        Format molecule as XYZ string (without atom count header).

        This is the simple format expected by PlotlyMol and other
        visualization tools.

        Returns:
            str: XYZ format string (atom symbol, x, y, z per line)

        Example:
            >>> mol = Molecule(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]])
            >>> print(mol.to_xyz_string())
            H 0.0 0.0 0.0
            H 0.0 0.0 0.74
        """
        lines = []
        for atom, coord in zip(self.atoms, self.coordinates):
            x, y, z = coord
            lines.append(f"{atom} {x:.10f} {y:.10f} {z:.10f}")

        return "\n".join(lines)

    def count_electrons(self) -> int:
        """
        Calculate total number of electrons (alias for get_electron_count).

        Returns:
            int: Number of electrons (nuclear charges - charge)

        Note:
            This is an alias for get_electron_count() to maintain
            compatibility with visualization module.
        """
        return self.get_electron_count()

    def get_spin(self) -> int:
        """
        Get spin quantum number S from multiplicity (2S+1).

        Returns:
            int: Number of unpaired electrons / 2
        """
        return (self.multiplicity - 1) // 2

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert molecule to dictionary for storage.

        Returns:
            dict: Molecule data
        """
        return {
            "atoms": self.atoms,
            "coordinates": self.coordinates,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Molecule":
        """
        Create Molecule from dictionary.

        Args:
            data: Dictionary with molecule data

        Returns:
            Molecule: Reconstructed molecule object
        """
        return cls(
            atoms=data["atoms"],
            coordinates=data["coordinates"],
            charge=data.get("charge", 0),
            multiplicity=data.get("multiplicity", 1),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Molecule({self.get_formula()}, "
            f"{len(self.atoms)} atoms, "
            f"charge={self.charge}, "
            f"mult={self.multiplicity})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return self.__str__()


def parse_xyz_input(xyz_text: str) -> Tuple[List[str], List[List[float]]]:
    """
    Parse XYZ coordinate input from text.

    Supports multiple formats:

    1. Simple format (one atom per line):
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74

    2. XYZ file format (with header):
        2
        Hydrogen molecule
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74

    3. With comments (lines starting with # or !):
        # This is a water molecule
        O  0.0  0.0  0.0
        H  0.757  0.587  0.0   # First hydrogen
        H  -0.757  0.587  0.0  ! Second hydrogen

    Args:
        xyz_text: Multi-line text with coordinates

    Returns:
        tuple: (atoms, coordinates) where atoms is list of symbols
               and coordinates is list of [x, y, z] lists

    Raises:
        ValueError: If input format is invalid
    """
    if not xyz_text or not xyz_text.strip():
        raise ValueError(
            "❌ Empty Input: Please provide XYZ coordinates.\n\n"
            "Expected format:\n"
            "  ATOM  X  Y  Z\n\n"
            "Example:\n"
            "  H  0.0  0.0  0.0\n"
            "  H  0.0  0.0  0.74"
        )

    lines = xyz_text.strip().split("\n")

    # ── Step 1: XYZ-file header detection, on RAW lines ──────────────────
    # The header (count line + title line) is POSITIONAL: whichever raw
    # line is the first non-blank, non-full-line-comment line, if it
    # parses as a bare integer, is the atom count — and the very next raw
    # line is the title, regardless of what that title line itself
    # contains (blank, "#"/"!"-prefixed, or free text).
    #
    # This must run BEFORE the blank/comment filtering below: filtering
    # first would remove a blank or comment title line from the list,
    # shifting the next real atom line into the "title" slot, where it
    # was then silently discarded — dropping the first atom of every
    # standard XYZ file whose title line happened to be blank or a
    # comment (both very common in practice).
    expected_atoms = None
    body_start = 0
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            continue
        try:
            expected_atoms = int(stripped)
            body_start = idx + 2  # count line + title line, whatever it is
            logger.debug(f"Detected XYZ file format expecting {expected_atoms} atoms")
        except ValueError:
            pass  # first content line isn't a bare count -> no header
        break  # only the first non-blank/non-comment line is eligible

    body_lines = lines[body_start:]

    # ── Step 2: filter blank lines + comments from the body ─────────────
    processed_lines = []
    original_line_numbers = []  # Track original line numbers for error reporting

    for offset, line in enumerate(body_lines):
        line_num = body_start + offset + 1  # 1-based original line number
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip comment lines (starting with # or !)
        if line.startswith("#") or line.startswith("!"):
            logger.debug(f"Skipping comment line {line_num}: {line}")
            continue

        # Remove inline comments (everything after # or !)
        for comment_char in ["#", "!"]:
            if comment_char in line:
                line = line.split(comment_char)[0].strip()

        if line:  # Only add non-empty lines after comment removal
            processed_lines.append(line)
            original_line_numbers.append(line_num)

    if not processed_lines:
        raise ValueError(
            "❌ No Data: All lines are empty or comments.\n\n"
            "Please provide at least 1 atom with coordinates."
        )

    atoms = []
    coordinates = []

    # Parse coordinate lines (the header, if any, was already consumed
    # positionally above, so every processed line here is an atom row).
    for i, line in enumerate(processed_lines):
        orig_line_num = original_line_numbers[i]
        parts = line.split()

        # Check minimum parts (atom symbol + 3 coordinates)
        if len(parts) < 4:
            raise ValueError(
                f"❌ Line {orig_line_num}: Invalid format - not enough values.\n\n"
                f"Got: {line}\n"
                f"Expected format: ATOM  X  Y  Z\n\n"
                f"Example: H  0.0  0.0  0.74\n\n"
                f"💡 Make sure each line has:\n"
                f"  1. Atom symbol (H, C, N, O, etc.)\n"
                f"  2. Three coordinate values (X, Y, Z)"
            )

        atom_symbol = parts[0]

        # Validate atom symbol with helpful suggestions
        if not utils.validate_atom_symbol(atom_symbol):
            # Try to suggest corrections for common mistakes
            suggestions = []

            # Case sensitivity: check if lowercase/uppercase version exists
            if atom_symbol.capitalize() in config.VALID_ATOMS:
                suggestions.append(
                    f"Did you mean '{atom_symbol.capitalize()}'? (check capitalization)"
                )
            elif atom_symbol.upper() in config.VALID_ATOMS:
                suggestions.append(f"Did you mean '{atom_symbol.upper()}'?")
            elif atom_symbol.lower().capitalize() in config.VALID_ATOMS:
                suggestions.append(
                    f"Did you mean '{atom_symbol.lower().capitalize()}'?"
                )

            # Common typos
            common_typos = {
                "he": "He",
                "li": "Li",
                "be": "Be",
                "ne": "Ne",
                "na": "Na",
                "mg": "Mg",
                "al": "Al",
                "si": "Si",
                "cl": "Cl",
                "ar": "Ar",
                "ca": "Ca",
                "fe": "Fe",
                "cu": "Cu",
                "zn": "Zn",
                "br": "Br",
                "kr": "Kr",
            }

            if atom_symbol.lower() in common_typos:
                correct = common_typos[atom_symbol.lower()]
                if correct not in suggestions:
                    suggestions.append(f"Did you mean '{correct}'?")

            # Build error message
            error_msg = (
                f"❌ Line {orig_line_num}: Invalid atom symbol '{atom_symbol}'.\n\n"
                f"Got: {line}\n"
            )

            if suggestions:
                error_msg += "\n💡 Suggestions:\n"
                for suggestion in suggestions:
                    error_msg += f"  • {suggestion}\n"
            else:
                error_msg += (
                    "\n💡 Valid atom symbols include:\n"
                    "  • H, C, N, O, F, P, S, Cl, Br, I\n"
                    "  • Li, Be, B, Na, Mg, Al, Si\n"
                    "  • K, Ca, Fe, Cu, Zn, etc.\n"
                    "\nNote: Symbols are case-sensitive (e.g., 'C' not 'c')"
                )

            raise ValueError(error_msg)

        # Parse coordinates
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise ValueError(
                f"❌ Line {orig_line_num}: Could not parse coordinates as numbers.\n\n"
                f"Got: {line}\n"
                f"Values: X={parts[1]}, Y={parts[2]}, Z={parts[3]}\n\n"
                f"💡 Coordinates must be numbers (integers or decimals).\n"
                f"Examples: 0.0, 1.5, -2.3, 0.757\n\n"
                f"Error details: {e}"
            ) from e

        atoms.append(atom_symbol)
        coordinates.append([x, y, z])

    # Validate we got atoms
    if not atoms:
        raise ValueError(
            "❌ No atoms found in input after parsing.\n\n"
            "Please check your coordinate format."
        )

    # Verify expected count if XYZ file format
    if expected_atoms is not None and len(atoms) != expected_atoms:
        logger.warning(
            f"XYZ file header specified {expected_atoms} atoms, "
            f"but found {len(atoms)} atoms"
        )

    logger.info(f"Successfully parsed {len(atoms)} atoms from XYZ input")
    return atoms, coordinates


def suggest_multiplicity(atoms: List[str], charge: int) -> int:
    """
    Suggest default spin multiplicity based on molecule composition.

    For simple molecules, suggests singlet (1) or doublet (2) based on
    whether total electron count is even or odd.

    Args:
        atoms: List of atomic symbols
        charge: Molecular charge

    Returns:
        int: Suggested multiplicity
    """
    # Calculate electron count directly without creating Molecule
    # (to avoid validation errors with incompatible multiplicity)
    try:
        nuclear_charge = sum(ATOMIC_NUMBERS.get(atom, 0) for atom in atoms)
        num_electrons = nuclear_charge - charge

        # Even electrons -> singlet, odd electrons -> doublet
        suggested = 1 if num_electrons % 2 == 0 else 2
        logger.debug(
            f"Suggested multiplicity {suggested} for {num_electrons} electrons"
        )
        return suggested

    except Exception as e:
        logger.warning(f"Could not suggest multiplicity: {e}")
        return 1  # Default to singlet


def get_preset_molecule(name: str) -> Optional[Molecule]:
    """
    Get a preset molecule from the library.

    Args:
        name: Molecule name (e.g., 'H2', 'H2O')

    Returns:
        Molecule: Preset molecule, or None if not found
    """
    preset = config.MOLECULE_LIBRARY.get(name)

    if preset is None:
        logger.warning(f"Preset molecule '{name}' not found")
        return None

    return Molecule(
        atoms=preset["atoms"],
        coordinates=preset["coordinates"],
        charge=preset["charge"],
        multiplicity=preset["multiplicity"],
    )


def list_preset_molecules() -> List[str]:
    """
    Get list of available preset molecule names.

    Returns:
        list: Molecule names
    """
    return list(config.MOLECULE_LIBRARY.keys())
