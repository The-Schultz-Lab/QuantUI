"""Tests for quantui.pes_scan_ui — PES Scan UI helpers."""

from __future__ import annotations

import pytest

from quantui.measurement import atom_label
from quantui.molecule import Molecule
from quantui.pes_scan_ui import (
    atom_dropdown_options,
    atom_list_html,
    default_atom_selection,
    format_scan_atom_summary,
    suggest_scan_range,
    validate_pes_scan_inputs,
)


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


class TestAtomDropdownOptions:
    def test_empty_when_no_molecule(self):
        opts = atom_dropdown_options(None)
        assert len(opts) == 1
        assert opts[0][1] == 1

    def test_labels_include_index_and_element(self):
        mol = _water()
        opts = atom_dropdown_options(mol)
        assert opts[0] == (f"1 {atom_label(mol, 0)}", 1)
        assert opts[1] == (f"2 {atom_label(mol, 1)}", 2)
        assert opts[2] == (f"3 {atom_label(mol, 2)}", 3)


class TestDefaultAtomSelection:
    def test_bond_defaults(self):
        assert default_atom_selection(3, "bond") == [1, 2]

    def test_angle_defaults(self):
        assert default_atom_selection(3, "angle") == [2, 1, 3]

    def test_dihedral_defaults(self):
        assert default_atom_selection(4, "dihedral") == [1, 2, 3, 4]


class TestSuggestScanRange:
    def test_angle_allows_negative_start(self):
        mol = _water()
        # H–O–H angle: atoms 2–1–3 (vertex at O)
        start, stop = suggest_scan_range(mol, "angle", [2, 1, 3])
        assert start < stop
        assert start < 110.0
        assert stop > 100.0

    def test_dihedral_can_span_negative(self):
        start, stop = suggest_scan_range(None, "dihedral", [1, 2, 3, 4])
        assert start < 0
        assert stop > 0


class TestValidatePesScanInputs:
    def test_rejects_missing_molecule(self):
        problems = validate_pes_scan_inputs(None, "bond", [1, 2], 0.8, 2.0, 5)
        assert any("Load a molecule" in p for p in problems)

    def test_rejects_out_of_range_atom(self):
        problems = validate_pes_scan_inputs(_water(), "bond", [1, 9], 0.8, 2.0, 5)
        assert any("out of range" in p for p in problems)

    def test_allows_negative_angle_start(self):
        problems = validate_pes_scan_inputs(
            _water(), "angle", [1, 2, 3], -30.0, 150.0, 5
        )
        assert problems == []

    def test_rejects_non_positive_bond_length(self):
        problems = validate_pes_scan_inputs(_water(), "bond", [1, 2], 0.0, 2.0, 5)
        assert any("positive" in p for p in problems)

    def test_rejects_duplicate_atoms(self):
        problems = validate_pes_scan_inputs(_water(), "bond", [1, 1], 0.8, 2.0, 5)
        assert any("unique" in p for p in problems)


class TestAtomListHtml:
    def test_renders_numbered_atoms(self):
        html = atom_list_html(_water())
        assert "1" in html and "O1" in html
        assert "2" in html and "H2" in html


class TestFormatScanAtomSummary:
    def test_angle_notes_vertex(self):
        summary = format_scan_atom_summary(_water(), "angle", [1, 2, 3])
        assert "vertex" in summary.lower()
        assert "O1" in summary or "1 (" in summary
