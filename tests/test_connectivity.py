"""Distance-based connectivity + disconnected-salt warning (M-METAL MET.2).

Pure logic (no RDKit / PySCF / network) plus the load-path wire-in that surfaces
the warning when a fetched name resolves to an ionic salt rather than the
coordinated complex.
"""

from __future__ import annotations

import pytest

from quantui.connectivity import (
    covalent_components,
    describe_disconnection,
    is_disconnected,
)
from quantui.molecule import Molecule


def _water_coords():
    return ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]


def _two_far_waters():
    atoms = ["O", "H", "H", "O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
        [10.0, 0.0, 0.0],
        [10.96, 0.0, 0.0],
        [9.76, 0.93, 0.0],
    ]
    return atoms, coords


def _cisplatin():
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == "inorganic-cisplatin")
    return e["atoms"], e["coordinates"]


class TestCovalentComponents:
    def test_single_molecule_is_one_component(self):
        atoms, coords = _water_coords()
        comps = covalent_components(atoms, coords)
        assert len(comps) == 1
        assert sorted(comps[0]) == [0, 1, 2]

    def test_separated_fragments_split(self):
        atoms, coords = _two_far_waters()
        comps = covalent_components(atoms, coords)
        assert len(comps) == 2
        # Largest-first, deterministic ordering; each water intact.
        assert all(len(c) == 3 for c in comps)

    def test_empty_input(self):
        assert covalent_components([], []) == []

    def test_bundled_metal_complex_stays_connected(self):
        # A correctly coordinated complex must NOT be flagged as a salt.
        atoms, coords = _cisplatin()
        assert len(covalent_components(atoms, coords)) == 1
        assert is_disconnected(atoms, coords) is False


class TestDescribeDisconnection:
    def test_connected_returns_none(self):
        atoms, coords = _water_coords()
        assert describe_disconnection(atoms, coords) is None

    def test_disconnected_names_fragments(self):
        atoms, coords = _two_far_waters()
        msg = describe_disconnection(atoms, coords)
        assert msg is not None
        assert "disconnected" in msg.lower()
        assert "2×H2O" in msg  # identical fragments grouped with a multiplier
        assert "XYZ Input" in msg  # actionable next step

    def test_scattered_metal_salt_is_flagged(self):
        # Simulate the cisplatin salt form: pull the Pt far from its ligands.
        atoms, coords = _cisplatin()
        coords = [list(c) for c in coords]
        pt = atoms.index("Pt")
        coords[pt] = [c + 8.0 for c in coords[pt]]
        msg = describe_disconnection(atoms, coords)
        assert msg is not None
        assert "Pt" in msg


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


class TestLoadPathWarning:
    def test_disconnected_result_warns(self, app):
        atoms, coords = _two_far_waters()
        mol = Molecule(atoms=atoms, coordinates=coords)
        app._apply_pubchem_search_result("salt-like", mol=mol, source="pubchem")
        assert "disconnected" in app.pubchem_msg.value.lower()

    def test_connected_result_no_warning(self, app):
        atoms, coords = _water_coords()
        mol = Molecule(atoms=atoms, coordinates=coords)
        app._apply_pubchem_search_result("water", mol=mol, source="pubchem")
        assert "disconnected" not in app.pubchem_msg.value.lower()
        assert "Loaded" in app.pubchem_msg.value
