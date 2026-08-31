"""Tests for quantui.pes_scan_ui — PES Scan UI helpers."""

from __future__ import annotations

import pytest

from quantui.measurement import atom_label
from quantui.molecule import Molecule
from quantui.pes_scan_ui import (
    adapt_atoms_for_scan_type_change,
    around_margin_defaults,
    atom_dropdown_options,
    atom_list_html,
    build_scan_grid,
    current_coordinate_value,
    default_atom_selection,
    format_scan_atom_summary,
    scan_range_around_current,
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

    def test_uses_theme_tokens_not_hardcoded_light_chip(self):
        html = atom_list_html(
            _water(), chip_bg="var(--q-bg-panel)", chip_fg="var(--q-text-body)"
        )
        assert "var(--q-bg-panel)" in html
        assert "var(--q-text-body)" in html
        assert "#f1f5f9" not in html


class TestFormatScanAtomSummary:
    def test_angle_notes_vertex(self):
        summary = format_scan_atom_summary(_water(), "angle", [1, 2, 3])
        assert "vertex" in summary.lower()
        assert "O1" in summary or "1 (" in summary


class TestAdaptAtomsForScanTypeChange:
    def test_bond_to_angle_preserves_atoms_when_possible(self):
        adapted = adapt_atoms_for_scan_type_change("bond", "angle", [1, 2], 3)
        assert adapted[:2] == [1, 2]
        assert adapted[2] == 3

    def test_angle_to_bond_keeps_first_two(self):
        adapted = adapt_atoms_for_scan_type_change("angle", "bond", [2, 1, 3], 3)
        assert adapted == [2, 1]


class TestCurrentCoordinateValue:
    def test_bond_distance_for_water(self):
        cur = current_coordinate_value(_water(), "bond", [1, 2])
        assert cur is not None
        val, unit = cur
        assert unit == "Å"
        assert val > 0.5


class TestBuildScanGrid:
    def test_linear_bond_grid(self):
        vals = build_scan_grid(1.0, 2.0, 5, scan_type="bond", grid="linear")
        assert len(vals) == 5
        assert vals[0] == pytest.approx(1.0)
        assert vals[-1] == pytest.approx(2.0)

    def test_log_bond_grid(self):
        vals = build_scan_grid(1.0, 2.0, 5, scan_type="bond", grid="log")
        assert len(vals) == 5
        assert vals[0] == pytest.approx(1.0)
        assert vals[-1] == pytest.approx(2.0)
        assert vals[1] > vals[0]


class TestScanRangeAroundCurrent:
    def test_angle_window_around_current(self):
        mol = _water()
        start, stop = scan_range_around_current(mol, "angle", [2, 1, 3])
        cur = current_coordinate_value(mol, "angle", [2, 1, 3])
        assert cur is not None
        val, _ = cur
        assert start < val < stop

    def test_angle_uses_margin_degrees(self):
        mol = _water()
        start, stop = scan_range_around_current(mol, "angle", [2, 1, 3], margin=30.0)
        cur = current_coordinate_value(mol, "angle", [2, 1, 3])
        assert cur is not None
        val, _ = cur
        assert start == pytest.approx(val - 30.0)
        assert stop == pytest.approx(val + 30.0)


class TestAroundMarginDefaults:
    def test_bond_is_percent(self):
        val, unit = around_margin_defaults("bond")
        assert val == 25.0
        assert "%" in unit

    def test_dihedral_is_degrees(self):
        val, unit = around_margin_defaults("dihedral")
        assert val == 60.0
        assert unit == "°"
