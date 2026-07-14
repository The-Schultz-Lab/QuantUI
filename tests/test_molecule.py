"""
Tests for QuantUI Molecule Module

Tests molecule creation, validation, parsing, and formatting functions.
"""

import pytest

from quantui.molecule import (
    Molecule,
    get_preset_molecule,
    list_preset_molecules,
    parse_xyz_input,
    suggest_multiplicity,
)

# ============================================================================
# Molecule Class Tests
# ============================================================================


class TestMoleculeCreation:
    """Test basic molecule creation and initialization."""

    def test_create_simple_molecule(self):
        """Test creating a simple H2 molecule."""
        atoms = ["H", "H"]
        coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mol = Molecule(atoms, coords, charge=0, multiplicity=1)

        assert mol.atoms == atoms
        assert mol.coordinates == coords
        assert mol.charge == 0
        assert mol.multiplicity == 1

    def test_create_water_molecule(self):
        """Test creating water molecule."""
        atoms = ["O", "H", "H"]
        coords = [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]

        mol = Molecule(atoms, coords)

        assert len(mol.atoms) == 3
        assert mol.get_formula() == "H2O"
        assert mol.charge == 0
        assert mol.multiplicity == 1

    def test_create_charged_molecule(self):
        """Test creating charged molecule."""
        atoms = ["N", "H", "H", "H", "H"]
        coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]

        mol = Molecule(atoms, coords, charge=1, multiplicity=1)

        assert mol.charge == 1
        assert mol.get_electron_count() == 10  # N(7) + 4H(4) - 1 = 10

    def test_create_radical_molecule(self):
        """Test creating radical (doublet) molecule."""
        atoms = ["O", "H"]
        coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]

        mol = Molecule(atoms, coords, charge=0, multiplicity=2)

        assert mol.multiplicity == 2
        assert mol.get_electron_count() == 9  # O(8) + H(1) = 9


class TestMoleculeValidation:
    """Test molecule validation logic."""

    def test_mismatched_atom_coord_count(self):
        """Test error when atom count doesn't match coordinate count."""
        atoms = ["H", "H"]
        coords = [[0.0, 0.0, 0.0]]  # Only one coordinate

        with pytest.raises(ValueError, match="does not match"):
            Molecule(atoms, coords)

    def test_empty_molecule(self):
        """Test error for empty molecule."""
        with pytest.raises(ValueError, match="at least one atom"):
            Molecule([], [])

    def test_invalid_atom_symbol(self):
        """Test error for invalid atomic symbol."""
        atoms = ["X", "Y"]  # Invalid symbols
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(ValueError, match="Invalid atom symbol"):
            Molecule(atoms, coords)

    def test_invalid_coordinates(self):
        """Test error for invalid coordinates."""
        atoms = ["H", "H"]
        coords = [[0.0, 0.0], [1.0, 0.0, 0.0]]  # First coord has only 2 values

        with pytest.raises(ValueError, match="Invalid coordinates"):
            Molecule(atoms, coords)

    def test_invalid_charge(self):
        """Test error for invalid charge."""
        atoms = ["H", "H"]
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(ValueError, match="Invalid charge"):
            Molecule(atoms, coords, charge=100)  # Too large

    def test_invalid_multiplicity(self):
        """Test error for invalid multiplicity."""
        atoms = ["H", "H"]
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(ValueError, match="Invalid multiplicity"):
            Molecule(atoms, coords, multiplicity=0)  # Must be >= 1

    def test_incompatible_multiplicity(self):
        """Test error when multiplicity is incompatible with electron count."""
        atoms = ["H", "H"]  # 2 electrons (even) -> multiplicity must be odd
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(ValueError, match="incompatible"):
            Molecule(atoms, coords, multiplicity=2)  # 2 is even, should be odd

    def test_multiplicity_error_message_contains_suggestions(self):
        """Test that multiplicity error message includes helpful suggestions."""
        atoms = ["H", "H"]  # 2 electrons (even) -> multiplicity must be odd
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        with pytest.raises(ValueError) as exc_info:
            Molecule(atoms, coords, multiplicity=2)

        error_msg = str(exc_info.value)
        # Check for key components of enhanced error message
        assert "Valid multiplicities" in error_msg
        assert "1, 3, 5" in error_msg  # Should suggest odd values
        assert "singlet" in error_msg.lower()
        assert "doublet" in error_msg.lower()
        assert "triplet" in error_msg.lower()

    def test_multiplicity_odd_electrons_error(self):
        """Test multiplicity error for odd electron count suggests even multiplicities."""
        atoms = ["H"]  # 1 electron (odd) -> multiplicity must be even
        coords = [[0.0, 0.0, 0.0]]

        with pytest.raises(ValueError) as exc_info:
            Molecule(atoms, coords, multiplicity=1)  # 1 is odd, should be even

        error_msg = str(exc_info.value)
        assert "2, 4, 6" in error_msg  # Should suggest even values


class TestMoleculeProperties:
    """Test molecule property calculations."""

    def test_get_formula_water(self):
        """Test molecular formula for water."""
        mol = Molecule(
            atoms=["O", "H", "H"], coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        )

        assert mol.get_formula() == "H2O"

    def test_get_formula_methane(self):
        """Test molecular formula for methane."""
        mol = Molecule(
            atoms=["C", "H", "H", "H", "H"],
            coordinates=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        )

        assert mol.get_formula() == "CH4"

    def test_get_formula_complex(self):
        """Test molecular formula for more complex molecule."""
        mol = Molecule(
            atoms=["C", "C", "O", "H", "H", "H", "H", "H", "H"],
            coordinates=[[0, 0, 0] for _ in range(9)],
        )

        assert mol.get_formula() == "C2H6O"

    def test_electron_count_neutral(self):
        """Test electron count for neutral molecule."""
        mol = Molecule(
            atoms=["O", "H", "H"], coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        )

        assert mol.get_electron_count() == 10  # O(8) + 2H(2) = 10

    def test_electron_count_charged(self):
        """Test electron count for charged molecule."""
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            charge=1,
            multiplicity=2,  # 9 electrons (odd) requires even multiplicity
        )

        assert mol.get_electron_count() == 9  # 10 - 1 = 9

    def test_get_spin(self):
        """Test spin quantum number calculation."""
        # Singlet (S=0)
        mol1 = Molecule(["H", "H"], [[0, 0, 0], [1, 0, 0]], multiplicity=1)
        assert mol1.get_spin() == 0

        # Doublet (S=1/2)
        mol2 = Molecule(["H"], [[0, 0, 0]], multiplicity=2)
        assert mol2.get_spin() == 0  # (2-1)//2 = 0 (integer division)

        # Triplet (S=1)
        mol3 = Molecule(["O", "O"], [[0, 0, 0], [2, 0, 0]], multiplicity=3)
        assert mol3.get_spin() == 1

    def test_count_electrons_alias(self):
        """Test that count_electrons is an alias for get_electron_count."""
        mol = Molecule(
            atoms=["C", "H", "H", "H", "H"],
            coordinates=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        )

        assert mol.count_electrons() == mol.get_electron_count()


class TestMoleculeFormatting:
    """Test molecule formatting methods."""

    def test_to_pyscf_format(self):
        """Test conversion to PySCF format."""
        mol = Molecule(
            atoms=["H", "H"], coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        )

        pyscf_str = mol.to_pyscf_format()
        lines = pyscf_str.split("\n")

        assert len(lines) == 2
        assert "H" in lines[0]
        assert "0.00000000" in lines[0]
        assert "0.74000000" in lines[1]

    def test_to_xyz_string(self):
        """Test conversion to XYZ string format."""
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        )

        xyz_str = mol.to_xyz_string()
        lines = xyz_str.split("\n")

        assert len(lines) == 3
        assert lines[0].startswith("O")
        assert lines[1].startswith("H")
        assert "0.757" in lines[1]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        mol = Molecule(
            atoms=["C", "H"],
            coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            charge=1,
            multiplicity=1,  # 6 electrons (even) requires odd multiplicity
        )

        data = mol.to_dict()

        assert data["atoms"] == ["C", "H"]
        assert data["coordinates"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert data["charge"] == 1
        assert data["multiplicity"] == 1

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            "atoms": ["N", "H", "H", "H"],
            "coordinates": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "charge": 0,
            "multiplicity": 1,
        }

        mol = Molecule.from_dict(data)

        assert mol.atoms == data["atoms"]
        assert mol.coordinates == data["coordinates"]
        assert mol.charge == 0
        assert mol.multiplicity == 1

    def test_str_representation(self):
        """Test string representation."""
        mol = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [1, 0, 0]])

        str_repr = str(mol)

        assert "Molecule" in str_repr
        assert "H2" in str_repr
        assert "2 atoms" in str_repr


# ============================================================================
# XYZ Parsing Tests
# ============================================================================


class TestParseXYZInput:
    """Test XYZ coordinate parsing."""

    def test_parse_simple_format(self):
        """Test parsing simple XYZ format (no header)."""
        xyz_text = """H  0.0  0.0  0.0
H  0.0  0.0  0.74"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["H", "H"]
        assert len(coords) == 2
        assert coords[0] == [0.0, 0.0, 0.0]
        assert coords[1] == [0.0, 0.0, 0.74]

    def test_parse_xyz_file_format(self):
        """Test parsing XYZ file format (with header)."""
        xyz_text = """3
Water molecule
O  0.0  0.0  0.0
H  0.757  0.587  0.0
H  -0.757  0.587  0.0"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3
        assert coords[0] == [0.0, 0.0, 0.0]

    def test_parse_with_extra_whitespace(self):
        """Test parsing with irregular whitespace."""
        xyz_text = """  H   0.0   0.0   0.0
  H   0.0   0.0   0.74  """

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["H", "H"]
        assert len(coords) == 2

    def test_parse_with_empty_lines(self):
        """Test parsing with empty lines."""
        xyz_text = """H  0.0  0.0  0.0

H  0.0  0.0  0.74
"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["H", "H"]
        assert len(coords) == 2

    def test_parse_empty_input(self):
        """Test error for empty input."""
        with pytest.raises(ValueError, match="Empty Input"):
            parse_xyz_input("")

    def test_parse_invalid_line(self):
        """Test error for invalid line format."""
        xyz_text = """H  0.0  0.0
H  0.0  0.0  0.74"""  # First line missing z-coordinate

        with pytest.raises(ValueError, match="Invalid format"):
            parse_xyz_input(xyz_text)

    def test_parse_non_numeric_coordinates(self):
        """Test error for non-numeric coordinates."""
        xyz_text = """H  0.0  0.0  abc
H  0.0  0.0  0.74"""

        with pytest.raises(ValueError, match="Could not parse coordinates"):
            parse_xyz_input(xyz_text)

    def test_parse_negative_coordinates(self):
        """Test parsing negative coordinates."""
        xyz_text = """O  0.0  0.0  0.0
H  0.757  0.587  -0.5
H  -0.757  -0.587  0.5"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert coords[1][2] == -0.5
        assert coords[2][0] == -0.757


class TestXyzHeaderTitleLine:
    """XYZ-file-format header (count + title) edge cases (H5 audit fix, 2026-07-14).

    Regression: blank/comment title lines were stripped by the parser's
    blank/comment filtering BEFORE header detection ran, which shifted the
    first real atom line into the "title" slot — silently dropping the
    first atom for any standard XYZ file whose title line was blank or a
    "#"/"!" comment (both very common: many tools write an empty title).
    """

    def test_blank_title_line_keeps_all_atoms(self):
        xyz_text = "3\n\nO  0.0  0.0  0.0\nH  0.757  0.587  0.0\nH  -0.757  0.587  0.0"
        atoms, coords = parse_xyz_input(xyz_text)
        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3
        assert coords[0] == [0.0, 0.0, 0.0]

    def test_comment_title_line_keeps_all_atoms(self):
        xyz_text = (
            "3\n# Water molecule\nO  0.0  0.0  0.0\n"
            "H  0.757  0.587  0.0\nH  -0.757  0.587  0.0"
        )
        atoms, coords = parse_xyz_input(xyz_text)
        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3

    def test_bang_comment_title_line_keeps_all_atoms(self):
        xyz_text = (
            "3\n! Water molecule\nO  0.0  0.0  0.0\n"
            "H  0.757  0.587  0.0\nH  -0.757  0.587  0.0"
        )
        atoms, coords = parse_xyz_input(xyz_text)
        assert atoms == ["O", "H", "H"]

    def test_free_text_title_line_still_works(self):
        xyz_text = (
            "3\nWater molecule\nO  0.0  0.0  0.0\n"
            "H  0.757  0.587  0.0\nH  -0.757  0.587  0.0"
        )
        atoms, coords = parse_xyz_input(xyz_text)
        assert atoms == ["O", "H", "H"]

    def test_leading_blank_line_before_count_still_works(self):
        xyz_text = (
            "\n3\ntitle\nO  0.0  0.0  0.0\n"
            "H  0.757  0.587  0.0\nH  -0.757  0.587  0.0"
        )
        atoms, coords = parse_xyz_input(xyz_text)
        assert atoms == ["O", "H", "H"]


class TestEnhancedXYZParser:
    """Test enhanced XYZ parser features."""

    def test_parse_with_comment_lines(self):
        """Test parsing with comment lines (# prefix)."""
        xyz_text = """# This is a water molecule
O  0.0  0.0  0.0
# First hydrogen
H  0.757  0.587  0.0
# Second hydrogen
H  -0.757  0.587  0.0"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3

    def test_parse_with_exclamation_comments(self):
        """Test parsing with comment lines (! prefix)."""
        xyz_text = """! Water molecule from optimization
O  0.0  0.0  0.0
! Hydrogen atoms
H  0.757  0.587  0.0
H  -0.757  0.587  0.0"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3

    def test_parse_with_inline_comments(self):
        """Test parsing with inline comments."""
        xyz_text = """O  0.0  0.0  0.0  # Oxygen atom
H  0.757  0.587  0.0  ! First H
H  -0.757  0.587  0.0  # Second H"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3
        assert coords[0] == [0.0, 0.0, 0.0]

    def test_parse_minimum_atoms_error(self):
        """Test error for single atom (need at least 2)."""
        xyz_text = """H  0.0  0.0  0.0"""

        with pytest.raises(ValueError, match="Too Few Atoms"):
            parse_xyz_input(xyz_text)

    def test_parse_invalid_atom_symbol_with_suggestion(self):
        """Test that invalid atom symbol provides helpful suggestion."""
        xyz_text = """h  0.0  0.0  0.0
h  0.0  0.0  0.74"""  # Lowercase 'h' instead of 'H'

        with pytest.raises(ValueError) as exc_info:
            parse_xyz_input(xyz_text)

        error_msg = str(exc_info.value)
        assert "Invalid atom symbol" in error_msg
        assert "'H'" in error_msg  # Should suggest capital H
        assert "capitalization" in error_msg.lower()

    def test_parse_common_typo_suggestion(self):
        """Test that common typos provide suggestions."""
        xyz_text = """cl  0.0  0.0  0.0
cl  0.0  0.0  2.0"""  # Lowercase 'cl' instead of 'Cl'

        with pytest.raises(ValueError) as exc_info:
            parse_xyz_input(xyz_text)

        error_msg = str(exc_info.value)
        assert "'Cl'" in error_msg  # Should suggest 'Cl'

    def test_parse_enhanced_error_for_missing_coordinates(self):
        """Test enhanced error message for missing coordinates."""
        xyz_text = """H  0.0  0.0
H  0.0  0.0  0.74"""

        with pytest.raises(ValueError) as exc_info:
            parse_xyz_input(xyz_text)

        error_msg = str(exc_info.value)
        assert "Invalid format" in error_msg
        assert "Expected format: ATOM  X  Y  Z" in error_msg

    def test_parse_enhanced_error_for_non_numeric(self):
        """Test enhanced error message for non-numeric coordinates."""
        xyz_text = """H  0.0  0.0  0.0
H  abc  def  ghi"""

        with pytest.raises(ValueError) as exc_info:
            parse_xyz_input(xyz_text)

        error_msg = str(exc_info.value)
        assert "Could not parse coordinates as numbers" in error_msg
        assert "must be numbers" in error_msg

    def test_parse_all_comments_error(self):
        """Test error when all lines are comments."""
        xyz_text = """# Comment 1
# Comment 2
! Comment 3"""

        with pytest.raises(ValueError, match="No Data"):
            parse_xyz_input(xyz_text)

    def test_parse_mixed_comments_and_blank_lines(self):
        """Test parsing with mix of comments, blank lines, and data."""
        xyz_text = """# Water molecule

O  0.0  0.0  0.0

# Hydrogens
H  0.757  0.587  0.0
H  -0.757  0.587  0.0

"""

        atoms, coords = parse_xyz_input(xyz_text)

        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestSuggestMultiplicity:
    """Test multiplicity suggestion logic."""

    def test_suggest_singlet_even_electrons(self):
        """Test singlet suggestion for even electron count."""
        atoms = ["H", "H"]  # 2 electrons -> singlet
        charge = 0

        mult = suggest_multiplicity(atoms, charge)

        assert mult == 1

    def test_suggest_doublet_odd_electrons(self):
        """Test doublet suggestion for odd electron count."""
        atoms = ["H"]  # 1 electron -> doublet
        charge = 0

        mult = suggest_multiplicity(atoms, charge)

        assert mult == 2

    def test_suggest_with_charge(self):
        """Test multiplicity suggestion with charged molecule."""
        atoms = ["O", "H", "H"]  # 10 electrons neutral
        charge = 1  # 9 electrons -> doublet

        mult = suggest_multiplicity(atoms, charge)

        assert mult == 2

    def test_suggest_water(self):
        """Test multiplicity for water."""
        atoms = ["O", "H", "H"]  # 10 electrons -> singlet
        charge = 0

        mult = suggest_multiplicity(atoms, charge)

        assert mult == 1


class TestPresetMolecules:
    """Test preset molecule library functions."""

    def test_list_preset_molecules(self):
        """Test listing available preset molecules."""
        names = list_preset_molecules()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "H2" in names
        assert "H2O" in names

    def test_get_preset_h2(self):
        """Test getting H2 preset molecule."""
        mol = get_preset_molecule("H2")

        assert mol is not None
        assert mol.get_formula() == "H2"
        assert len(mol.atoms) == 2
        assert mol.charge == 0
        assert mol.multiplicity == 1

    def test_get_preset_water(self):
        """Test getting water preset molecule."""
        mol = get_preset_molecule("H2O")

        assert mol is not None
        assert mol.get_formula() == "H2O"
        assert len(mol.atoms) == 3

    def test_get_preset_invalid(self):
        """Test getting non-existent preset molecule."""
        mol = get_preset_molecule("NotARealMolecule")

        assert mol is None

    def test_preset_molecules_valid(self):
        """Test that all preset molecules are valid."""
        names = list_preset_molecules()

        for name in names:
            mol = get_preset_molecule(name)
            assert mol is not None
            assert len(mol.atoms) > 0
            # Should not raise validation errors
            assert mol.get_electron_count() > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMoleculeIntegration:
    """Test complete molecule workflows."""

    def test_parse_and_create_molecule(self):
        """Test complete workflow: parse XYZ -> create Molecule."""
        xyz_text = """O  0.0  0.0  0.0
H  0.757  0.587  0.0
H  -0.757  0.587  0.0"""

        atoms, coords = parse_xyz_input(xyz_text)
        mol = Molecule(atoms, coords, charge=0, multiplicity=1)

        assert mol.get_formula() == "H2O"
        assert mol.get_electron_count() == 10

    def test_roundtrip_serialization(self):
        """Test molecule -> dict -> molecule roundtrip."""
        original = Molecule(
            atoms=["C", "O", "H", "H"],
            coordinates=[[0, 0, 0], [1.2, 0, 0], [0, 1, 0], [0, 0, 1]],
            charge=-1,
            multiplicity=2,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Molecule.from_dict(data)

        assert restored.atoms == original.atoms
        assert restored.coordinates == original.coordinates
        assert restored.charge == original.charge
        assert restored.multiplicity == original.multiplicity
        assert restored.get_formula() == original.get_formula()

    def test_preset_to_xyz_string(self):
        """Test converting preset molecule to XYZ string."""
        mol = get_preset_molecule("CH4")

        xyz_str = mol.to_xyz_string()

        # Parse it back
        atoms, coords = parse_xyz_input(xyz_str)

        assert atoms == mol.atoms
        assert len(coords) == len(mol.coordinates)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
