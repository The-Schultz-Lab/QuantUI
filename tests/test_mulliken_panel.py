"""Tests for Mulliken Populations Analysis panel + plot builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantui.app import QuantUIApp
from quantui.app_analysis import pop_mulliken, show_mulliken_populations
from quantui.molecule import Molecule
from quantui.mulliken_plot import plot_mulliken_charges
from quantui.results_storage import save_result


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


def _sp_result_with_charges(**overrides):
    base = dict(
        formula="H2O",
        method="RHF",
        basis="STO-3G",
        energy_hartree=-75.0,
        energy_ev=-2040.5,
        homo_lumo_gap_ev=10.5,
        converged=True,
        n_iterations=8,
        atom_symbols=["O", "H", "H"],
        mulliken_charges=[-0.66, 0.33, 0.33],
        dipole_moment_debye=1.85,
        dipole_vector_debye=[0.0, 0.0, 1.85],
        mo_energy_hartree=None,
        mo_occ=None,
        mo_coeff=None,
        pyscf_mol_atom=None,
        pyscf_mol_basis=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    a = QuantUIApp()
    a._set_molecule(_water())
    return a


class TestMullikenPlot:
    def test_bar_count_matches_atoms(self):
        fig = plot_mulliken_charges(["O", "H", "H"], [-0.5, 0.25, 0.25])
        assert len(fig.data) == 1
        assert list(fig.data[0].x) == ["O1", "H2", "H3"]
        assert list(fig.data[0].y) == pytest.approx([-0.5, 0.25, 0.25])

    def test_colors_split_by_sign(self):
        fig = plot_mulliken_charges(["O", "H"], [-0.4, 0.4])
        colors = list(fig.data[0].marker.color)
        assert colors[0] != colors[1]


class TestPanelRegistration:
    def test_populations_in_panel_meta(self):
        names = [n for n, _, _ in QuantUIApp._PANEL_META]
        assert "Populations" in names

    def test_single_point_registry_includes_populations(self):
        entries = QuantUIApp._PANEL_REGISTRY["single_point"]
        assert ("Populations", "_pop_mulliken", False) in entries

    def test_geometry_opt_registry_includes_populations(self):
        entries = QuantUIApp._PANEL_REGISTRY["geometry_opt"]
        assert ("Populations", "_pop_mulliken", False) in entries

    def test_accordion_is_a_child_of_analysis_tab(self, app):
        # Missed once for Geometries: registry alone is not enough.
        assert app._mulliken_accordion in app.analysis_tab_panel.children


class TestPopMullikenLive:
    def test_activates_with_charges(self, app):
        ctx = SimpleNamespace(
            calc_type="single_point",
            live_result=_sp_result_with_charges(),
            result_dir=None,
            spectra_data={},
            source="live",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            label="H2O RHF/STO-3G",
            timestamp="",
        )
        assert pop_mulliken(app, ctx) is True
        assert "O1" in app._mulliken_table.value
        assert (
            "-0.6600" in app._mulliken_table.value
            or "-0.66" in app._mulliken_table.value
        )
        assert "1.8500 D" in app._mulliken_summary.value
        assert app._last_mulliken_charges == pytest.approx([-0.66, 0.33, 0.33])
        assert app._last_mulliken_dipole_vector == pytest.approx([0.0, 0.0, 1.85])

    def test_returns_false_without_charges(self, app):
        ctx = SimpleNamespace(
            calc_type="single_point",
            live_result=_sp_result_with_charges(mulliken_charges=None),
            result_dir=None,
            spectra_data={},
            source="live",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            label="H2O",
            timestamp="",
        )
        assert pop_mulliken(app, ctx) is False

    def test_returns_false_on_length_mismatch(self, app):
        ctx = SimpleNamespace(
            calc_type="single_point",
            live_result=_sp_result_with_charges(mulliken_charges=[-0.5]),
            result_dir=None,
            spectra_data={},
            source="live",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            label="H2O",
            timestamp="",
        )
        assert pop_mulliken(app, ctx) is False


class TestPopMullikenHistory:
    def test_history_roundtrip_activates_populations(self, tmp_path, app):
        result = _sp_result_with_charges()
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Populations" in app._ana_available
        assert app._last_mulliken_charges == pytest.approx([-0.66, 0.33, 0.33])
        assert "O1" in app._mulliken_table.value

    def test_history_without_charges_leaves_panel_inactive(self, tmp_path, app):
        result = _sp_result_with_charges(mulliken_charges=None, atom_symbols=None)
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Populations" not in app._ana_available
        unavail = app._ana_unavail_msgs["Populations"].value
        assert "Mulliken charges were not saved" in unavail
        assert "Re-run" in unavail

    def test_live_without_charges_sets_specific_message(self, app):
        ctx = SimpleNamespace(
            calc_type="single_point",
            live_result=_sp_result_with_charges(mulliken_charges=None),
            result_dir=None,
            spectra_data={},
            source="live",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            label="H2O",
            timestamp="",
        )
        assert pop_mulliken(app, ctx) is False
        unavail = app._ana_unavail_msgs["Populations"].value
        assert "Mulliken charges are not available" in unavail


class TestShowMullikenPopulations:
    def test_writes_table_and_caches_state(self, app):
        ok = show_mulliken_populations(
            app, ["O", "H", "H"], [-0.5, 0.25, 0.25], dipole_debye=1.2
        )
        assert ok is True
        assert "H2" in app._mulliken_table.value
        assert app._last_mulliken_dipole == pytest.approx(1.2)


class TestHelpTopic:
    def test_mulliken_topic_exists(self):
        from quantui.help_content import HELP_TOPICS, VALID_TOPICS

        assert "mulliken" in HELP_TOPICS
        assert "mulliken" in VALID_TOPICS
        assert "basis-set dependent" in HELP_TOPICS["mulliken"]["body"]
