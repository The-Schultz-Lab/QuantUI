"""Tests for the bulk QM9 library subset (M-STRUCT STRUCT.8).

Platform-independent: reads the committed store + manifest (the QM9 download +
RDKit are build-time only). No PySCF.
"""

import json

import pytest

from quantui import config
from quantui import molecule_library as ml

_BULK = ml._data_dir() / "manifests" / "bulk_qm9.json"

pytestmark = pytest.mark.skipif(
    not _BULK.exists(),
    reason="bulk_qm9.json not built (run scripts/build_bulk_library.py)",
)


def _bulk_entries():
    return json.loads(_BULK.read_text(encoding="utf-8"))


# ============================================================================
# Manifest integrity
# ============================================================================


class TestBulkManifest:
    def test_manifest_nonempty(self):
        assert len(_bulk_entries()) >= 100

    def test_every_entry_well_formed(self):
        ceiling = config.LIBRARY_HEAVY_ATOM_CEILING_BULK
        for e in _bulk_entries():
            assert e["category"] == "bulk-qm9"
            assert e["source"] == "qm9-dft"
            assert e["charge"] == 0 and e["multiplicity"] == 1
            assert len(e["coordinates"]) == len(e["atoms"])
            assert sum(1 for a in e["atoms"] if a != "H") <= ceiling
            assert e["id"].startswith("qm9-")

    def test_ids_unique(self):
        ids = [e["id"] for e in _bulk_entries()]
        assert len(ids) == len(set(ids))

    def test_provenance_file_exists(self):
        prov = ml._data_dir() / "library" / "QM9-PROVENANCE.md"
        assert prov.exists()
        assert "CC0" in prov.read_text(encoding="utf-8")


# ============================================================================
# Store integration + key contracts
# ============================================================================


class TestBulkInStore:
    def test_total_count_includes_bulk(self):
        assert ml.count() >= 190 + len(_bulk_entries())

    def test_bulk_excluded_from_preset_dict(self):
        # The browse dropdown must NOT balloon with thousands of bulk entries.
        d = config.MOLECULE_LIBRARY
        assert all(not k.startswith("qm9-") for k in d)
        assert len(d) == 190  # presets + curated only (+14 inorganic examples)

    def test_bulk_category_present(self):
        assert "bulk-qm9" in ml.categories()

    def test_bulk_reachable_via_search(self):
        rows = ml.search("", category="bulk-qm9", limit=10)
        assert len(rows) == 10
        assert all(r["category"] == "bulk-qm9" for r in rows)

    def test_bulk_entry_materializes_with_coords(self):
        sample_id = _bulk_entries()[0]["id"]
        entry = ml.get(sample_id)
        assert entry is not None
        assert len(entry["coordinates"]) == len(entry["atoms"])
        assert entry["coordinates"]  # non-empty

    def test_store_within_budget(self):
        assert ml.db_path().stat().st_size <= config.LIBRARY_SIZE_BUDGET_BYTES
