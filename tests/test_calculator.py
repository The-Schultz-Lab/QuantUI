"""
Tests for QuantUI Calculator Module

Tests PySCFCalculation class for calculation setup and script generation.
Resource estimation (estimate_resources) was a SLURM-cluster feature and
has been removed from the local version — calculations run directly via
run_in_session(), not via batch submission.
"""

import pytest

from quantui.calculator import PySCFCalculation, create_calculation
from quantui.molecule import Molecule

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def water_molecule():
    """Create a water molecule for testing."""
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture
def h2_molecule():
    """Create an H2 molecule for testing."""
    return Molecule(
        atoms=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture
def radical_molecule():
    """Create a radical (OH) for testing."""
    return Molecule(
        atoms=["O", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        charge=0,
        multiplicity=2,
    )


# ============================================================================
# PySCFCalculation Initialization Tests
# ============================================================================


class TestPySCFCalculationInit:
    """Test PySCFCalculation initialization."""

    def test_create_rhf_calculation(self, water_molecule):
        """Test creating RHF calculation."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")

        assert calc.molecule == water_molecule
        assert calc.method == "RHF"
        assert calc.basis == "6-31G"

    def test_create_uhf_calculation(self, radical_molecule):
        """Test creating UHF calculation."""
        calc = PySCFCalculation(radical_molecule, method="UHF", basis="STO-3G")

        assert calc.molecule == radical_molecule
        assert calc.method == "UHF"
        assert calc.basis == "STO-3G"

    def test_lowercase_method(self, water_molecule):
        """Test that method is converted to uppercase."""
        calc = PySCFCalculation(water_molecule, method="rhf", basis="6-31G")

        assert calc.method == "RHF"

    def test_unsupported_method(self, water_molecule):
        """Test error for unsupported method."""
        # Use a fictional method name to exercise the validation path —
        # the previous stand-in "CCSD" became a real supported method in
        # M8.1 (session 54), so the validator no longer rejects it.
        with pytest.raises(ValueError, match="not supported"):
            PySCFCalculation(water_molecule, method="NONEXISTENT", basis="6-31G")

    def test_mixed_case_method_preserved_canonically(self, water_molecule):
        """Mixed-case SUPPORTED_METHODS entries must not be upper-cased.

        Regression: method.upper() turned "wB97X-D" into "WB97X-D", which
        matches neither SUPPORTED_METHODS nor the xc-alias tables the
        generated script depends on — silently breaking Export Script and
        the educational-notes lookup for every mixed-case method name
        (wB97X-D, CAM-B3LYP, M06-L, M06-2X, HSE06, PBE-D3).
        """
        calc = PySCFCalculation(water_molecule, method="wB97X-D", basis="6-31G")
        assert calc.method == "wB97X-D"

    def test_mixed_case_method_case_insensitive_input(self, water_molecule):
        """A differently-cased spelling still resolves to the canonical form."""
        calc = PySCFCalculation(water_molecule, method="wb97x-d", basis="6-31G")
        assert calc.method == "wB97X-D"

    def test_nonstandard_basis_warning(self, water_molecule, caplog):
        """Test warning for non-standard basis set."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="custom-basis")

        # Should create calculation but log warning
        assert calc.basis == "custom-basis"
        # Check that warning was logged
        assert "not in standard list" in caplog.text


# ============================================================================
# Script Generation Tests
# ============================================================================


class TestScriptGeneration:
    """Test calculation script generation."""

    def test_generate_script_rhf(self, water_molecule, tmp_path):
        """Test generating RHF calculation script."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")
        script_path = tmp_path / "calc.py"

        script_content = calc.generate_calculation_script(script_path)

        assert script_path.exists()
        assert isinstance(script_content, str)
        assert "RHF" in script_content
        assert "6-31G" in script_content
        assert "H2O" in script_content

    def test_generate_script_uhf(self, radical_molecule, tmp_path):
        """Test generating UHF calculation script."""
        calc = PySCFCalculation(radical_molecule, method="UHF", basis="STO-3G")
        script_path = tmp_path / "calc.py"

        script_content = calc.generate_calculation_script(script_path)

        assert script_path.exists()
        assert "UHF" in script_content
        assert "STO-3G" in script_content

    def test_script_contains_geometry(self, water_molecule, tmp_path):
        """Test that script contains molecular geometry."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")
        script_path = tmp_path / "calc.py"

        script_content = calc.generate_calculation_script(script_path)

        # Should contain atom symbols and coordinates
        assert "O" in script_content
        assert "H" in script_content
        assert "0.0" in script_content or "0.00" in script_content

    def test_script_contains_charge_and_multiplicity(self, water_molecule, tmp_path):
        """Test that script contains charge and spin."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")
        script_path = tmp_path / "calc.py"

        script_content = calc.generate_calculation_script(script_path)

        # For singlet H2O: charge=0, spin=0 (mult-1)
        assert "charge = 0" in script_content
        assert "spin = 0" in script_content

    def test_script_charged_molecule(self, tmp_path):
        """Test script generation for charged molecule."""
        mol = Molecule(
            atoms=["O", "H", "H", "H"],
            coordinates=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            charge=1,
            multiplicity=1,
        )
        calc = PySCFCalculation(mol, method="RHF", basis="6-31G")
        script_path = tmp_path / "calc.py"

        script_content = calc.generate_calculation_script(script_path)

        assert "charge = 1" in script_content

    def test_script_creates_parent_directories(self, tmp_path):
        """Test that script creation makes parent directories."""
        script_path = tmp_path / "nested" / "dir" / "calc.py"
        assert not script_path.parent.exists()

        calc = PySCFCalculation(
            Molecule(["H", "H"], [[0, 0, 0], [1, 0, 0]]), method="RHF", basis="6-31G"
        )
        calc.generate_calculation_script(script_path)

        assert script_path.exists()
        assert script_path.parent.exists()


# ============================================================================
# Description and Educational Notes Tests
# ============================================================================


class TestDescriptionAndNotes:
    """Test description and educational note generation."""

    def test_get_description(self, water_molecule):
        """Test getting calculation description."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")

        desc = calc.get_description()

        assert isinstance(desc, str)
        assert "Restricted Hartree-Fock" in desc
        assert "H2O" in desc
        assert "6-31G" in desc

    def test_get_description_uhf(self, radical_molecule):
        """Test UHF description."""
        calc = PySCFCalculation(radical_molecule, method="UHF", basis="STO-3G")

        desc = calc.get_description()

        assert "Unrestricted Hartree-Fock" in desc
        assert "STO-3G" in desc

    def test_get_educational_notes_rhf(self, water_molecule):
        """Test educational notes for RHF."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")

        notes = calc.get_educational_notes()

        assert isinstance(notes, str)
        assert "RHF" in notes or "Restricted" in notes
        assert "closed-shell" in notes

    def test_get_educational_notes_uhf(self, radical_molecule):
        """Test educational notes for UHF."""
        calc = PySCFCalculation(radical_molecule, method="UHF", basis="STO-3G")

        notes = calc.get_educational_notes()

        assert "UHF" in notes or "Unrestricted" in notes
        assert "open-shell" in notes

    def test_educational_notes_minimal_basis(self, h2_molecule):
        """Test educational notes mention basis set characteristics."""
        calc = PySCFCalculation(h2_molecule, method="RHF", basis="STO-3G")

        notes = calc.get_educational_notes()

        assert "STO-3G" in notes
        assert "minimal" in notes.lower()

    def test_educational_notes_split_valence(self, water_molecule):
        """Test educational notes for split-valence basis."""
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G*")

        notes = calc.get_educational_notes()

        assert "6-31G" in notes
        assert "split-valence" in notes or "polarization" in notes

    def test_educational_notes_correlation_consistent(self, h2_molecule):
        """Test educational notes for correlation-consistent basis."""
        calc = PySCFCalculation(h2_molecule, method="RHF", basis="cc-pVDZ")

        notes = calc.get_educational_notes()

        assert "cc-pV" in notes or "correlation-consistent" in notes.lower()

    def test_educational_notes_open_shell(self, radical_molecule):
        """Test that open-shell systems get multiplicity notes."""
        calc = PySCFCalculation(radical_molecule, method="UHF", basis="6-31G")

        notes = calc.get_educational_notes()

        assert "multiplicity" in notes.lower()
        assert "unpaired" in notes.lower()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateCalculation:
    """Test create_calculation factory function."""

    def test_create_calculation_simple(self, water_molecule):
        """Test factory function creates calculation correctly."""
        calc = create_calculation(molecule=water_molecule, method="RHF", basis="6-31G")

        assert isinstance(calc, PySCFCalculation)
        assert calc.molecule == water_molecule
        assert calc.method == "RHF"
        assert calc.basis == "6-31G"

    def test_create_calculation_uhf(self, radical_molecule):
        """Test factory function with UHF."""
        calc = create_calculation(
            molecule=radical_molecule, method="UHF", basis="STO-3G"
        )

        assert isinstance(calc, PySCFCalculation)
        assert calc.method == "UHF"


# ============================================================================
# Integration Tests
# ============================================================================


class TestCalculationIntegration:
    """Test complete calculation workflows."""

    def test_complete_calculation_setup(self, water_molecule, tmp_path):
        """Test complete workflow from molecule to script."""
        # Create calculation
        calc = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")

        # Get description
        desc = calc.get_description()
        assert "Hartree-Fock" in desc

        # Generate script
        script_path = tmp_path / "water_calc.py"
        script_content = calc.generate_calculation_script(script_path)

        # Verify script was created and is valid Python
        assert script_path.exists()
        assert "import" in script_content
        assert "def main" in script_content

    def test_multiple_calculations_same_molecule(self, water_molecule, tmp_path):
        """Test creating multiple calculations for same molecule."""
        calc1 = PySCFCalculation(water_molecule, method="RHF", basis="STO-3G")
        calc2 = PySCFCalculation(water_molecule, method="RHF", basis="6-31G")

        script1 = calc1.generate_calculation_script(tmp_path / "calc1.py")
        script2 = calc2.generate_calculation_script(tmp_path / "calc2.py")

        # Scripts should be different (different basis sets)
        assert "STO-3G" in script1
        assert "6-31G" in script2

    def test_calculation_with_factory(self, water_molecule, tmp_path):
        """Test using factory function in workflow."""
        calc = create_calculation(
            molecule=water_molecule, method="RHF", basis="cc-pVDZ"
        )

        # Should work just like direct instantiation
        script_path = tmp_path / "factory_calc.py"
        calc.generate_calculation_script(script_path)

        assert script_path.exists()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_atom_molecule(self):
        """Test calculation with single atom."""
        mol = Molecule(["H"], [[0, 0, 0]], charge=0, multiplicity=2)
        calc = PySCFCalculation(mol, method="UHF", basis="6-31G")
        assert calc is not None

    def test_highly_charged_molecule(self):
        """Test calculation with high charge."""
        mol = Molecule(
            ["C"] * 5, [[i, 0, 0] for i in range(5)], charge=4, multiplicity=1
        )
        calc = PySCFCalculation(mol, method="RHF", basis="6-31G")
        assert calc is not None

    def test_high_multiplicity(self):
        """Test calculation with high multiplicity."""
        mol = Molecule(
            ["O"] * 4,
            [[i, 0, 0] for i in range(4)],
            charge=0,
            multiplicity=5,  # Quintuplet
        )
        calc = PySCFCalculation(mol, method="UHF", basis="6-31G")

        notes = calc.get_educational_notes()
        assert "5" in notes or "quintuplet" in notes.lower()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
