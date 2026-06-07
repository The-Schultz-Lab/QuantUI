"""Tests for the bundled molecule-library store + loader (M-STRUCT STRUCT.6).

Platform-independent (stdlib sqlite3 + struct + json); no RDKit, no PySCF.
"""

import math

import pytest

from quantui import config
from quantui import molecule_library as ml

_BUDGET_BYTES = 10 * 1024 * 1024  # DEC-015


# ============================================================================
# Coordinate codec
# ============================================================================


class TestCoordCodec:
    def test_round_trip_exact_for_3_decimal_coords(self):
        atoms = ["O", "H", "H", "Cl"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.587],
            [0.0, -0.757, 0.587],
            [1.27, 0.0, -2.48],
        ]
        blob = ml.encode_coords(atoms, coords)
        out_atoms, out_coords = ml.decode_coords(blob)
        assert out_atoms == atoms
        for a, b in zip(coords, out_coords):
            for ca, cb in zip(a, b):
                assert math.isclose(ca, cb, abs_tol=1e-9)

    def test_record_size_is_8_bytes_per_atom(self):
        blob = ml.encode_coords(["C", "H"], [[0, 0, 0], [1, 1, 1]])
        assert len(blob) == 2 * 8

    def test_two_char_symbol_round_trips(self):
        atoms, coords = ml.decode_coords(
            ml.encode_coords(["Cl", "Na"], [[0, 0, 0], [2, 0, 0]])
        )
        assert atoms == ["Cl", "Na"]

    def test_rejects_long_symbol(self):
        with pytest.raises(ValueError):
            ml.encode_coords(["Uuo"], [[0, 0, 0]])


# ============================================================================
# Store build + query API
# ============================================================================


class TestStoreBuildAndQuery:
    def test_build_from_manifest_round_trips(self, tmp_path):
        target = tmp_path / "lib.sqlite"
        ml.build_from_manifests(target)
        assert target.exists()

    def test_store_contains_the_original_presets(self):
        # The store grows as STRUCT.7/.8 add content; assert the original 20
        # presets are all still present rather than an exact total.
        preset_ids = {
            "H2",
            "O2",
            "N2",
            "CO",
            "HF",
            "HCl",
            "H2O",
            "CO2",
            "O3",
            "H2O2",
            "CH4",
            "NH3",
            "C2H6",
            "C2H4",
            "C2H2",
            "CH3OH",
            "CH2O",
            "CH3CHO",
            "CH3COOH",
            "C6H6",
        }
        assert ml.count() >= 20
        assert all(ml.get(pid) is not None for pid in preset_ids)

    def test_categories_present(self):
        cats = ml.categories()
        assert {"diatomic", "triatomic", "small-organic", "aromatic"} <= set(cats)

    def test_get_returns_full_entry_with_coords(self):
        entry = ml.get("H2O")
        assert entry is not None
        assert entry["atoms"] == ["O", "H", "H"]
        assert entry["n_heavy"] == 1
        assert entry["coordinates"][1] == [0.0, 0.757, 0.587]

    def test_get_unknown_returns_none(self):
        assert ml.get("NOTAREALID") is None

    def test_search_by_formula(self):
        hits = ml.search("C6H6")
        assert any(h["id"] == "C6H6" for h in hits)

    def test_search_by_category_filter(self):
        hits = ml.search("", category="aromatic")
        assert [h["id"] for h in hits] == ["C6H6"]

    def test_search_returns_lightweight_rows(self):
        hits = ml.search("H2O")
        assert hits and "coordinates" not in hits[0]


# ============================================================================
# Legacy back-compat shim
# ============================================================================


class TestPresetDictBackCompat:
    def test_get_preset_dict_shape(self):
        d = ml.get_preset_dict()
        assert len(d) >= 20
        h2o = d["H2O"]
        assert set(h2o) == {
            "atoms",
            "coordinates",
            "charge",
            "multiplicity",
            "description",
        }

    def test_config_shim_resolves(self):
        lib = config.MOLECULE_LIBRARY
        assert len(lib) >= 20
        assert lib["O2"]["multiplicity"] == 3  # triplet preserved
        assert lib["H2O"]["coordinates"][2] == [0.0, -0.757, 0.587]

    def test_config_shim_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            _ = config.THIS_DOES_NOT_EXIST

    def test_preset_dict_excludes_bulk_categories(self):
        # No bulk entries yet, but the contract must hold for STRUCT.8.
        for entry_id in ml.get_preset_dict():
            full = ml.get(entry_id)
            assert full["category"] not in ml._BULK_CATEGORIES


# ============================================================================
# Size governance (STRUCT.10 precursor)
# ============================================================================


class TestSizeBudget:
    def test_committed_store_within_10mb(self):
        assert ml.db_path().stat().st_size <= _BUDGET_BYTES
