"""Tests for the curated named library (M-STRUCT STRUCT.7).

Platform-independent: reads the committed store + manifest (RDKit is only
needed at build time, not here). No PySCF.
"""

import json

from quantui import config
from quantui import molecule_library as ml

_CURATED = ml._data_dir() / "manifests" / "curated.json"


def _curated_entries():
    return json.loads(_CURATED.read_text(encoding="utf-8"))


# ============================================================================
# Manifest integrity
# ============================================================================


class TestCuratedManifest:
    def test_manifest_exists_and_nonempty(self):
        entries = _curated_entries()
        assert len(entries) >= 150

    def test_every_entry_well_formed(self):
        for e in _curated_entries():
            assert e["atoms"], e["name"]
            assert len(e["coordinates"]) == len(e["atoms"]), e["name"]
            assert e["source"] == "curated-ff"
            assert e["category"]
            assert e["id"]

    def test_ids_unique(self):
        ids = [e["id"] for e in _curated_entries()]
        assert len(ids) == len(set(ids))

    def test_respects_heavy_atom_ceiling(self):
        ceiling = config.LIBRARY_HEAVY_ATOM_CEILING_CURATED
        for e in _curated_entries():
            n_heavy = sum(1 for a in e["atoms"] if a != "H")
            assert n_heavy <= ceiling, f"{e['name']} has {n_heavy} heavy atoms"


# ============================================================================
# Store contents after curated merge
# ============================================================================


class TestStoreWithCurated:
    def test_total_count_is_presets_plus_curated(self):
        assert ml.count() == 20 + len(_curated_entries())

    def test_expected_categories_present(self):
        cats = set(ml.categories())
        assert {
            "amino-acid",
            "nucleobase",
            "biomolecule",
            "solvent",
            "functional-group",
            "hydrocarbon",
            "drug",
            "inorganic",
            "ion",
        } <= cats

    def test_all_20_amino_acids(self):
        rows = ml.search("", category="amino-acid", limit=100)
        assert len(rows) == 20

    def test_aspirin_present_and_correct(self):
        asp = ml.get("aspirin")
        assert asp is not None
        assert asp["formula"] == "C9H8O4"
        assert asp["category"] == "drug"
        assert asp["charge"] == 0

    def test_ion_charges(self):
        assert ml.get("nitrate")["charge"] == -1
        assert ml.get("sulfate")["charge"] == -2
        assert ml.get("phosphate")["charge"] == -3
        assert ml.get("ammonium")["charge"] == 1

    def test_radical_multiplicities(self):
        assert ml.get("nitric-oxide")["multiplicity"] == 2
        assert ml.get("nitrogen-dioxide")["multiplicity"] == 2

    def test_search_by_synonym(self):
        assert any(h["id"] == "glucose" for h in ml.search("dextrose"))
        assert any(h["id"] == "paracetamol" for h in ml.search("acetaminophen"))


# ============================================================================
# Back-compat dict + size budget
# ============================================================================


class TestPresetDictAndBudget:
    def test_preset_dict_includes_curated(self):
        d = config.MOLECULE_LIBRARY
        assert "aspirin" in d
        assert "glycine" in d
        # Presets still lead the (interim) flat dropdown.
        assert list(d)[:3] == ["H2", "O2", "N2"]

    def test_curated_entry_has_legacy_shape(self):
        gly = config.MOLECULE_LIBRARY["glycine"]
        assert set(gly) == {
            "atoms",
            "coordinates",
            "charge",
            "multiplicity",
            "description",
        }

    def test_store_within_budget(self):
        assert ml.db_path().stat().st_size <= config.LIBRARY_SIZE_BUDGET_BYTES
