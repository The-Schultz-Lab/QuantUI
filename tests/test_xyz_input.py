"""Tests for XYZ formatting, cleanup, and geometry-only loading."""

import pytest

from quantui.molecule import Molecule, parse_xyz_input
from quantui.xyz_input import (
    ChargeMultSuggestion,
    format_xyz_body,
    load_molecule_from_xyz_text,
    propose_xyz_cleanup,
    spin_compatibility_note,
    suggest_charge_multiplicity_from_atoms,
)


class TestFormatXyzBody:
    def test_formats_columns(self):
        text = format_xyz_body(["O", "H"], [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0]])
        lines = text.splitlines()
        assert lines[0].startswith("O ")
        assert "0.757000" in lines[1]

    def test_normalizes_symbols(self):
        text = format_xyz_body(["cl", "h"], [[0, 0, 0], [1, 0, 0]])
        assert text.splitlines()[0].startswith("Cl")
        assert text.splitlines()[1].startswith("H ")


class TestProposeXyzCleanup:
    def test_cleanup_messy_input(self):
        messy = """  3
water
o 0 0 0   ! oxygen
h 0.757 0.587 0
h -0.757 0.587 0"""
        cleaned, notes = propose_xyz_cleanup(messy)
        atoms, coords = parse_xyz_input(cleaned)
        assert atoms == ["O", "H", "H"]
        assert len(coords) == 3
        assert any("capitalization" in n.lower() for n in notes)
        assert any("header" in n.lower() for n in notes)

    def test_already_clean_reports_so(self):
        body = format_xyz_body(["N"], [[0.0, 0.0, 0.0]])
        cleaned, notes = propose_xyz_cleanup(body)
        assert cleaned == body
        assert any("already" in n.lower() for n in notes)


class TestLoadMoleculeFromXyzText:
    def test_single_nitrogen_does_not_raise_with_default_spin(self):
        mol, note = load_molecule_from_xyz_text("N 0 0 0", charge=0, multiplicity=1)
        assert mol.get_formula() == "N"
        assert len(mol.atoms) == 1
        assert note is not None
        assert "multiplicity" in note.lower()

    def test_single_nitrogen_doublet_is_consistent(self):
        mol, note = load_molecule_from_xyz_text("N 0 0 0", charge=0, multiplicity=2)
        assert mol.multiplicity == 2
        assert note is None

    def test_validate_spin_false_allows_incompatible_combo(self):
        mol, _ = load_molecule_from_xyz_text("N 0 0 0", charge=0, multiplicity=1)
        assert mol._validate_spin is False

    def test_strict_molecule_still_rejects_bad_spin(self):
        with pytest.raises(ValueError, match="incompatible"):
            Molecule(["N"], [[0.0, 0.0, 0.0]], charge=0, multiplicity=1)


class TestSpinCompatibilityNote:
    def test_water_singlet_ok(self):
        assert spin_compatibility_note(["O", "H", "H"], 0, 1) is None

    def test_nitrogen_doublet_needed(self):
        note = spin_compatibility_note(["N"], 0, 1)
        assert note is not None


class TestSuggestChargeMultiplicityFromAtoms:
    def test_nitrogen_neutral_doublet(self):
        s = suggest_charge_multiplicity_from_atoms(["N"])
        assert s.charge == 0
        assert s.multiplicity == 2
        assert s.formula == "N"
        assert "cannot be inferred" in s.caveats[0].lower()

    def test_water_neutral_singlet(self):
        s = suggest_charge_multiplicity_from_atoms(["O", "H", "H"])
        assert s.charge == 0
        assert s.multiplicity == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No atoms"):
            suggest_charge_multiplicity_from_atoms([])

    def test_returns_frozen_dataclass(self):
        s = suggest_charge_multiplicity_from_atoms(["H", "H"])
        assert isinstance(s, ChargeMultSuggestion)
