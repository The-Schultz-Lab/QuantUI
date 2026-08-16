"""Bundled inorganic / coordination-complex examples — M-METAL MET.9.

The metal examples ship as explicit-coordinate library entries (not SMILES, which
would scatter the metal). These tests guard that they stay in the vendored store,
load with the right charge/multiplicity, and keep a connected metal centre.
"""

from __future__ import annotations

from quantui import molecule_library as ml

# Explicit spot-checks (formula / charge / multiplicity) for a representative
# spread of geometries, spin states, and charges. The structural tests below
# cover *every* inorganic entry, so new ones don't need adding here.
_EXPECTED = {
    "inorganic-cisplatin": {"formula": "H6Cl2N2Pt", "charge": 0, "mult": 1},
    "inorganic-hexaamminecobaltiii": {"formula": "H18CoN6", "charge": 3, "mult": 1},
    "inorganic-ferrocene": {"formula": "C10H10Fe", "charge": 0, "mult": 1},
    "inorganic-hexaaquaironii": {"formula": "H12FeO6", "charge": 2, "mult": 5},
    "inorganic-hexacyanoferrateiii": {"formula": "C6FeN6", "charge": -3, "mult": 2},
    "inorganic-tetracarbonylnickel0": {"formula": "C4NiO4", "charge": 0, "mult": 1},
    "inorganic-permanganate": {"formula": "MnO4", "charge": -1, "mult": 1},
    "inorganic-tetrachloroplatinateii": {"formula": "Cl4Pt", "charge": -2, "mult": 1},
}

_MIN_INORGANIC_EXAMPLES = 14


def _inorganic_entries():
    return [e for e in ml.iter_entries() if e["category"] == "inorganic-complex"]


def test_expected_examples_present_with_right_metadata():
    for eid, exp in _EXPECTED.items():
        e = ml.get(eid)
        assert e is not None, f"missing bundled inorganic example: {eid}"
        assert e["formula"] == exp["formula"]
        assert e["charge"] == exp["charge"]
        assert e["multiplicity"] == exp["mult"]
        assert e["category"] == "inorganic-complex"


def test_library_ships_a_full_inorganic_set():
    assert len(_inorganic_entries()) >= _MIN_INORGANIC_EXAMPLES


def test_every_metal_centre_is_connected():
    """The whole point of M-METAL: no metal is a detached dot. Checked across
    ALL inorganic entries via the shipped, metal-aware connectivity finder."""
    from quantui.connectivity import (
        covalent_components,
        is_metal,
        metal_coordination_bonds,
    )

    for e in _inorganic_entries():
        atoms, coords = e["atoms"], e["coordinates"]
        # One connected component — never a scattered salt.
        assert len(covalent_components(atoms, coords)) == 1, e["id"]
        bonded = {i for bond in metal_coordination_bonds(atoms, coords) for i in bond}
        for i, sym in enumerate(atoms):
            if is_metal(sym):
                degree = sum(
                    1 for a, b in metal_coordination_bonds(atoms, coords) if i in (a, b)
                )
                assert (
                    i in bonded and degree >= 2
                ), f"{e['id']}: {sym} under-coordinated"


def test_every_example_passes_the_charge_multiplicity_guard():
    """A bundled example must not itself trip the charge/multiplicity guard."""
    from quantui.inorganic_guards import check_charge_multiplicity
    from quantui.molecule import ATOMIC_NUMBERS

    for e in _inorganic_entries():
        n_elec = sum(ATOMIC_NUMBERS.get(a, 0) for a in e["atoms"]) - e["charge"]
        assert check_charge_multiplicity(n_elec, e["multiplicity"]) is None, e["id"]
