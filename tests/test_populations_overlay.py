"""Tests for Mulliken color + dipole-arrow overlays (populations_overlay)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantui.app import QuantUIApp
from quantui.app_measurement import finalize_analysis_html
from quantui.molecule import Molecule
from quantui.populations_overlay import (
    build_overlay_payload,
    center_of_mass,
    charge_colors,
    dipole_arrow_endpoints,
    inject_populations_js,
    push_populations_overlay,
)
from quantui.results_storage import save_result
from quantui.viz_backend_router import VizBackend


def _water() -> Molecule:
    return Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    a = QuantUIApp()
    a._set_molecule(_water())
    a._analysis_displayed_molecule = _water()
    return a


class TestChargeColors:
    def test_opposite_signs_differ(self):
        colors = charge_colors([-0.5, 0.5])
        assert colors[0] != colors[1]
        assert colors[0].startswith("#")
        assert len(colors[0]) == 7

    def test_neutral_near_grey(self):
        colors = charge_colors([0.0, 0.0])
        # Both map to the mid grey endpoint.
        assert colors[0] == colors[1]


class TestDipoleArrow:
    def test_nonzero_vector_returns_endpoints(self):
        arrow = dipole_arrow_endpoints([0.0, 0.0, 0.0], [0.0, 0.0, 1.85])
        assert arrow is not None
        assert arrow["start"]["z"] < 0
        assert arrow["end"]["z"] > 0

    def test_near_zero_returns_none(self):
        assert dipole_arrow_endpoints([0, 0, 0], [0.0, 0.0, 0.0]) is None


class TestCenterOfMass:
    def test_water_com_near_oxygen(self):
        mol = _water()
        com = center_of_mass(mol.atoms, mol.coordinates)
        # Oxygen dominates the mass — COM should sit near the O atom.
        assert abs(com[0]) < 0.2
        assert abs(com[2]) < 0.05


class TestBuildPayload:
    def test_color_and_arrow_when_enabled(self):
        mol = _water()
        payload = build_overlay_payload(
            charges=[-0.6, 0.3, 0.3],
            color_enabled=True,
            dipole_vector=[0.0, 0.0, 1.8],
            dipole_enabled=True,
            symbols=mol.atoms,
            coordinates=mol.coordinates,
        )
        assert payload["colors"] is not None
        assert len(payload["colors"]) == 3
        assert payload["arrow"] is not None

    def test_disabled_toggles_clear_payload(self):
        mol = _water()
        payload = build_overlay_payload(
            charges=[-0.6, 0.3, 0.3],
            color_enabled=False,
            dipole_vector=[0.0, 0.0, 1.8],
            dipole_enabled=False,
            symbols=mol.atoms,
            coordinates=mol.coordinates,
        )
        assert payload["colors"] is None
        assert payload["arrow"] is None


class TestInjectPopulationsJS:
    def test_bridge_function_defined(self):
        from quantui.visualization_py3dmol import (
            PY3DMOL_AVAILABLE,
            render_molecule_html,
        )

        if not PY3DMOL_AVAILABLE:
            pytest.skip("py3dmol not installed")
        html = render_molecule_html(_water(), backend="py3dmol")
        out = inject_populations_js(html)
        assert "__quantuiPopulationsUpdate" in out
        assert "addArrow" in out
        assert "setStyle" in out

    def test_finalize_injects_both_measure_and_populations(self, app):
        from quantui.visualization_py3dmol import (
            PY3DMOL_AVAILABLE,
            render_molecule_html,
        )

        if not PY3DMOL_AVAILABLE:
            pytest.skip("py3dmol not installed")
        html = render_molecule_html(_water(), backend="py3dmol")
        out = finalize_analysis_html(app, html, VizBackend.PY3DMOL)
        assert "setClickable" in out
        assert "__quantuiPopulationsUpdate" in out


class TestPushOverlaySafe:
    def test_no_bridge_is_noop(self, app):
        app._populations_js_bridge = None
        app._last_mulliken_charges = [-0.5, 0.25, 0.25]
        push_populations_overlay(app)  # must not raise


class TestVectorPersistence:
    def test_save_result_writes_dipole_vector(self, tmp_path):
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-2040.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            atom_symbols=["O", "H", "H"],
            mulliken_charges=[-0.66, 0.33, 0.33],
            dipole_moment_debye=1.85,
            dipole_vector_debye=[0.1, -0.2, 1.8],
        )
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        import json

        data = json.loads((saved / "result.json").read_text())
        assert data["dipole_vector_debye"] == pytest.approx([0.1, -0.2, 1.8])

    def test_history_pop_loads_vector(self, tmp_path, app):
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-2040.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            atom_symbols=["O", "H", "H"],
            mulliken_charges=[-0.66, 0.33, 0.33],
            dipole_moment_debye=1.85,
            dipole_vector_debye=[0.0, 0.0, 1.85],
            mo_energy_hartree=None,
            mo_occ=None,
            mo_coeff=None,
        )
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Populations" in app._ana_available
        assert app._last_mulliken_dipole_vector == pytest.approx([0.0, 0.0, 1.85])
        assert (
            "+0.000" in app._mulliken_summary.value
            or "1.850" in app._mulliken_summary.value
        )
        assert app._mulliken_dipole_cb.disabled is False

    def test_old_history_without_vector_disables_arrow(self, tmp_path, app):
        result = SimpleNamespace(
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            energy_ev=-2040.0,
            homo_lumo_gap_ev=10.0,
            converged=True,
            n_iterations=5,
            atom_symbols=["O", "H", "H"],
            mulliken_charges=[-0.66, 0.33, 0.33],
            dipole_moment_debye=1.85,
            # no dipole_vector_debye
        )
        saved = save_result(
            result, results_dir=tmp_path, calc_type="single_point", spectra={}
        )
        ctx = app._build_history_context(saved)
        app._apply_analysis_context(ctx)
        assert "Populations" in app._ana_available
        assert app._last_mulliken_dipole_vector is None
        assert app._mulliken_dipole_cb.disabled is True
