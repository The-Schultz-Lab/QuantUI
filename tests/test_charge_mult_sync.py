"""Charge/multiplicity fields sync onto the active molecule.

The calculation reads ``mol.charge`` / ``mol.multiplicity`` (session_calc sets
``mol.charge`` and ``mol.spin = multiplicity - 1``); the pre-run guard reads the
widgets, and the spin-state helper's Apply writes ``mult_si``. Without a
widget→molecule sync, an edited multiplicity (or an applied spin state) never
reached the run and the guard could validate a different value than the calc
used. These tests guard the sync.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


def _load(app, charge=0, mult=1):
    app._set_molecule(
        Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            charge=charge,
            multiplicity=mult,
        ),
        "m",
    )


class TestChargeMultSync:
    def test_multiplicity_edit_reaches_molecule(self, app):
        _load(app)
        app.mult_si.value = 3
        assert app._molecule.multiplicity == 3

    def test_charge_edit_reaches_molecule(self, app):
        _load(app)
        app.charge_si.value = -2
        assert app._molecule.charge == -2

    def test_spin_apply_reaches_the_molecule_the_run_uses(self, app):
        # The whole point: Apply must update mol.multiplicity, not just the field.
        app._set_molecule(
            Molecule(
                atoms=["Co"], coordinates=[[0.0, 0.0, 0.0]], charge=3, multiplicity=1
            ),
            "co",
        )
        app.spin_metal_dd.value = "Co"
        app.spin_ox_si.value = 3
        app.spin_geom_dd.value = "octahedral"
        app._on_spin_suggest()
        app._on_spin_apply(0)  # high-spin, multiplicity 5
        assert app.mult_si.value == 5
        assert app._molecule.multiplicity == 5

    def test_sync_is_safe_with_no_molecule(self, app):
        app._molecule = None
        app.mult_si.value = 2  # must not raise
        app.charge_si.value = -1

    def test_load_sets_fields_and_keeps_them_consistent(self, app):
        _load(app, charge=1, mult=2)
        assert app.charge_si.value == 1 and app.mult_si.value == 2
        assert app._molecule.charge == 1 and app._molecule.multiplicity == 2
