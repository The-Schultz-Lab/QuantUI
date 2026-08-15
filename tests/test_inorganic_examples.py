"""Bundled inorganic / coordination-complex examples — M-METAL MET.9.

The metal examples ship as explicit-coordinate library entries (not SMILES, which
would scatter the metal). These tests guard that they stay in the vendored store,
load with the right charge/multiplicity, and keep a connected metal centre.
"""

from __future__ import annotations

import math

from quantui import molecule_library as ml

_EXPECTED = {
    "inorganic-cisplatin": {
        "formula": "H6Cl2N2Pt",
        "charge": 0,
        "mult": 1,
        "metal": "Pt",
    },
    "inorganic-hexaamminecobaltiii": {
        "formula": "H18CoN6",
        "charge": 3,
        "mult": 1,
        "metal": "Co",
    },
    "inorganic-ferrocene": {
        "formula": "C10H10Fe",
        "charge": 0,
        "mult": 1,
        "metal": "Fe",
    },
}

# Generous covalent-radius sums (Å) for the connectivity check.
_BOND_CUTOFF = {"Pt": 2.9, "Co": 2.7, "Fe": 2.7}


def test_all_examples_present():
    ids = {e["id"] for e in ml.iter_entries()}
    for eid in _EXPECTED:
        assert eid in ids, f"missing bundled inorganic example: {eid}"


def test_examples_have_correct_charge_multiplicity_and_formula():
    for eid, exp in _EXPECTED.items():
        e = ml.get(eid)
        assert e is not None
        assert e["formula"] == exp["formula"]
        assert e["charge"] == exp["charge"]
        assert e["multiplicity"] == exp["mult"]
        assert e["category"] == "inorganic-complex"


def test_metal_centre_is_connected():
    """The whole point of M-METAL: the metal must not be a detached dot."""
    for eid, exp in _EXPECTED.items():
        e = ml.get(eid)
        atoms = e["atoms"]
        coords = e["coordinates"]
        mi = atoms.index(exp["metal"])
        mx, my, mz = coords[mi]
        cutoff = _BOND_CUTOFF[exp["metal"]]
        neighbours = 0
        for j, (x, y, z) in enumerate(coords):
            if j == mi:
                continue
            d = math.sqrt((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2)
            if d <= cutoff:
                neighbours += 1
        assert neighbours >= 2, f"{eid}: metal has only {neighbours} neighbours"


def test_electron_count_parity_matches_multiplicity():
    """A bundled example must not itself trip the charge/multiplicity guard."""
    from quantui.inorganic_guards import check_charge_multiplicity
    from quantui.molecule import ATOMIC_NUMBERS

    for eid in _EXPECTED:
        e = ml.get(eid)
        n_elec = sum(ATOMIC_NUMBERS.get(a, 0) for a in e["atoms"]) - e["charge"]
        assert check_charge_multiplicity(n_elec, e["multiplicity"]) is None
