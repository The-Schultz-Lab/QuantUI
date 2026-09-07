"""Tests for quantui.freq_displacement_ids (M-CHECKPOINT CHK.4.2).

Pure functions, no PySCF, no checkpoint I/O — just the id scheme that
CHK.4.3-.5's serial/parallel wiring and resume logic build on.
"""

from __future__ import annotations

import pytest

from quantui.freq_displacement_ids import (
    displacement_id,
    parse_displacement_id,
    required_displacement_ids,
)


class TestDisplacementId:
    def test_format(self):
        assert displacement_id(0, 0, 1) == "d000_x_+"
        assert displacement_id(0, 0, -1) == "d000_x_-"
        assert displacement_id(12, 2, 1) == "d012_z_+"

    def test_rejects_bad_axis(self):
        with pytest.raises(ValueError):
            displacement_id(0, 3, 1)

    def test_rejects_bad_sign(self):
        with pytest.raises(ValueError):
            displacement_id(0, 0, 0)

    def test_ids_are_unique_across_atoms_axes_signs(self):
        ids = {
            displacement_id(a, ax, s)
            for a in range(5)
            for ax in range(3)
            for s in (1, -1)
        }
        assert len(ids) == 5 * 3 * 2


class TestParseDisplacementId:
    def test_round_trips(self):
        for atom_index, axis, sign in [(0, 0, 1), (12, 2, -1), (3, 1, 1)]:
            item_id = displacement_id(atom_index, axis, sign)
            assert parse_displacement_id(item_id) == (atom_index, axis, sign)

    @pytest.mark.parametrize(
        "bad_id", ["", "not_a_displacement", "d000_w_+", "d000_x_0", "d0a0_x_+"]
    )
    def test_rejects_malformed_ids(self, bad_id):
        with pytest.raises(ValueError):
            parse_displacement_id(bad_id)


class TestRequiredDisplacementIds:
    def test_count_is_6n(self):
        assert len(required_displacement_ids(3)) == 18
        assert len(required_displacement_ids(19)) == 114

    def test_zero_atoms_is_empty(self):
        assert required_displacement_ids(0) == []

    def test_every_id_parses_back_and_covers_the_grid(self):
        n_atoms = 4
        ids = required_displacement_ids(n_atoms)
        parsed = {parse_displacement_id(i) for i in ids}
        expected = {
            (a, ax, s) for a in range(n_atoms) for ax in range(3) for s in (1, -1)
        }
        assert parsed == expected

    def test_deterministic_order(self):
        assert required_displacement_ids(2) == required_displacement_ids(2)
