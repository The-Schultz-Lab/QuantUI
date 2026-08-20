"""Whole-library governance + round-trip checks (M-STRUCT STRUCT.10).

Validates *every* shipped entry (across all tiers) — well-formed, coordinate
round-trip stable, ids globally unique — plus the ≤10 MB budget and the
preset-dict/bulk contract. Platform-independent; no PySCF, no network.
"""

import math

import pytest

from quantui import config
from quantui import molecule_library as ml

# Reasonable element-symbol whitelist for the bundled tiers (CHONF + curated
# heteroatoms + common ions + the coordination-complex metals from the bundled
# inorganic examples, MET.9). Guards against codec corruption.
_KNOWN = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Zn",
    "Br",
    "I",
    "Pt",
}


class TestEveryEntryWellFormed:
    def test_all_entries_consistent(self):
        seen_ids = set()
        for e in ml.iter_entries():
            eid = e["id"]
            assert eid not in seen_ids, f"duplicate id {eid}"
            seen_ids.add(eid)
            atoms, coords = e["atoms"], e["coordinates"]
            assert atoms, eid
            assert len(atoms) == len(coords) == e["n_atoms"], eid
            assert e["n_heavy"] == sum(1 for a in atoms if a != "H"), eid
            assert isinstance(e["charge"], int) and isinstance(e["multiplicity"], int)
            assert e["formula"], eid
            assert e["category"] and e["source"], eid
            for sym in atoms:
                assert sym in _KNOWN, f"{eid}: unexpected symbol {sym!r}"
            for xyz in coords:
                assert len(xyz) == 3, eid

    def test_coordinate_round_trip_is_stable(self):
        # Re-encoding the stored (already-decoded) coords must be idempotent at
        # the codec's 0.001 Å resolution.
        for e in ml.iter_entries():
            atoms, coords = e["atoms"], e["coordinates"]
            blob = ml.encode_coords(atoms, coords)
            out_atoms, out_coords = ml.decode_coords(blob)
            assert out_atoms == atoms, e["id"]
            for a, b in zip(coords, out_coords):
                for ca, cb in zip(a, b):
                    assert math.isclose(ca, cb, abs_tol=1e-6), e["id"]


class TestBudgetAndContracts:
    def test_store_within_10mb_budget(self):
        size = ml.db_path().stat().st_size
        assert size <= config.LIBRARY_SIZE_BUDGET_BYTES, f"{size} bytes > budget"

    def test_iter_count_matches_count(self):
        assert sum(1 for _ in ml.iter_entries()) == ml.count()

    def test_preset_dict_is_non_bulk_only(self):
        d = ml.get_preset_dict()
        non_bulk = sum(
            1 for e in ml.iter_entries() if e["category"] not in ml._BULK_CATEGORIES
        )
        assert len(d) == non_bulk
        assert all(not k.startswith("qm9-") for k in d)

    def test_reproducible_build_tooling_present(self):
        # No-silent-cap governance: the deterministic builders ship in-repo and
        # report skips/dedup counts to stdout (not hidden truncation).
        scripts = ml._data_dir().parent.parent / "scripts"
        if not scripts.is_dir():
            pytest.skip("scripts/ not present (non-dev install)")
        for name in (
            "build_library.py",
            "build_curated_library.py",
            "build_bulk_library.py",
        ):
            assert (scripts / name).exists(), name
