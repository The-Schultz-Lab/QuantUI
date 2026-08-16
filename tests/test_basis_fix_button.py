"""One-click "Switch to def2-SVP" basis fix (M-METAL MET.5).

The pre-run guard blocks a metal on an incompatible basis; this offers a single
click to fix it — but only when def2-SVP actually resolves the coverage, never
for a charge/multiplicity problem it can't fix.
"""

from __future__ import annotations

import pytest

from quantui.molecule import Molecule


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


def _cisplatin() -> Molecule:
    from quantui import molecule_library as ml

    e = next(x for x in ml.iter_entries() if x["id"] == "inorganic-cisplatin")
    return Molecule(
        atoms=e["atoms"],
        coordinates=e["coordinates"],
        charge=e["charge"],
        multiplicity=e["multiplicity"],
    )


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


class TestBasisFixButton:
    def test_hidden_initially(self, app):
        assert app.basis_fix_btn.layout.display == "none"

    def test_metal_on_pople_reveals_fix(self, app):
        app._molecule = _cisplatin()
        app.basis_dd.value = "6-31G"  # no Pt coverage
        app.mult_si.value = 1  # consistent, so the basis is the only problem
        app._on_run_clicked(None)
        assert app.basis_fix_btn.layout.display == ""  # shown

    def test_click_sets_def2_and_hides(self, app):
        app._molecule = _cisplatin()
        app.basis_dd.value = "6-31G"
        app.mult_si.value = 1
        app._on_run_clicked(None)

        app._on_basis_fix(None)
        assert app.basis_dd.value == "def2-SVP"
        assert app.basis_fix_btn.layout.display == "none"
        assert "def2-SVP" in app.run_status.value

    def test_multiplicity_only_problem_does_not_reveal_fix(self, app):
        # Water on a valid basis but an impossible multiplicity: the guard blocks,
        # but def2-SVP can't fix a spin problem, so the button stays hidden.
        app._molecule = _water()
        app.basis_dd.value = "6-31G"  # covers C/H/O/N fine
        app.mult_si.value = 2  # 10 electrons can't be a doublet
        app._on_run_clicked(None)
        assert app.basis_fix_btn.layout.display == "none"

    def test_helper_hides_when_def2_would_not_help(self, app):
        from quantui.app_runflow import _update_basis_fix_button

        app._molecule = _water()
        app.basis_dd.value = "6-31G"  # already fine for water
        _update_basis_fix_button(app, app._molecule)
        assert app.basis_fix_btn.layout.display == "none"
