"""Regression tests for the Medium VIZ audit findings (2026-11-10 audit).

- VIZ-3: a theme toggle must keep the persistent single vib viewer (and the
  vib HTML cache must not serve the other theme's background).
- VIZ-4: UHF/UKS HOMO-LUMO gap and orbital diagram span both spin channels.
- VIZ-5: MO numbers the user sees or types are 1-based everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quantui.app import QuantUIApp
from quantui.molecule import Molecule


@pytest.fixture
def app():
    return QuantUIApp()


@pytest.fixture
def water_mol():
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


@pytest.fixture
def fake_freq_result():
    return SimpleNamespace(
        frequencies_cm1=[1600.0, 3800.0, 3850.0],
        ir_intensities=[40.0, 10.0, 50.0],
        displacements=[
            np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]]),
            np.array([[0.0, 0.10, 0.0], [0.05, -0.10, 0.0], [-0.05, -0.10, 0.0]]),
            np.array([[0.0, 0.0, 0.10], [0.05, 0.0, -0.10], [-0.05, 0.0, -0.10]]),
        ],
    )


def _vib_html(app) -> str:
    return app.vib_output.outputs[0]["data"]["text/html"]


# ── VIZ-3 ────────────────────────────────────────────────────────────────


class TestThemeToggleKeepsSingleVibViewer:
    def test_theme_rerender_rebuilds_the_single_viewer(
        self, app, water_mol, fake_freq_result
    ):
        from quantui.app_visualization import (
            _render_vib_single_viewer,
            rerender_3d_scenes_for_theme,
        )

        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_molecule = water_mol
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_data = None
        app._last_result_dir = None
        app.vib_mode_dd.options = [("Mode 1", 1), ("Mode 2", 2), ("Mode 3", 3)]
        app.vib_mode_dd.value = 2
        assert _render_vib_single_viewer(app, fake_freq_result, water_mol, 2, [1, 2, 3])
        assert app._vib_single_viewer_active is True

        app.theme_btn.value = "Dark"
        rerender_3d_scenes_for_theme(app)

        html = _vib_html(app)
        # The mode/amplitude/fps observers keep calling this bridge while the
        # flag is True — the re-rendered HTML must still define it.
        assert app._vib_single_viewer_active is True
        assert "__quantuiVibSetMode" in html
        assert "#1e293b" in html  # Dark scene background

    def test_vib_cache_is_keyed_on_background(
        self, app, water_mol, fake_freq_result, tmp_path
    ):
        from quantui.app_visualization import _render_vib_mode_py3dmol

        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path
        app._vib_single_viewer_active = False

        app.theme_btn.value = "Light"
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        light = _vib_html(app)
        app.theme_btn.value = "Dark"
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        dark = _vib_html(app)
        app.theme_btn.value = "Light"
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        light_again = _vib_html(app)

        assert "#1e293b" in dark and "#1e293b" not in light
        assert "#1e293b" not in light_again


# ── VIZ-4 ────────────────────────────────────────────────────────────────


def _oh_uhf():
    """OH radical, UHF/STO-3G (from a real PySCF run, Hartree).

    alpha-only gap 28.56 eV; frontier gap min(LUMO) - max(HOMO) 20.08 eV.
    """
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = pyscf.scf.UHF(mol).run()
    return mf


class TestUnrestrictedFrontierGap:
    def test_helper_uses_both_spin_channels(self):
        from quantui.session_calc import homo_lumo_gap_hartree

        e = np.array([[-1.0, -0.5, 0.4], [-1.0, -0.2, 0.1]])
        occ = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        # HOMO = max(-0.5 a, -1.0 b) = -0.5; LUMO = min(0.4 a, -0.2 b) = -0.2
        assert homo_lumo_gap_hartree(e, occ) == pytest.approx(0.3)
        # Restricted input is unchanged.
        assert homo_lumo_gap_hartree(e[0], occ[0]) == pytest.approx(0.9)

    def test_orbital_info_gap_is_frontier_gap(self):
        from quantui.orbital_visualization import orbital_info_from_arrays

        mf = _oh_uhf()
        info = orbital_info_from_arrays(mf.mo_energy, mf.mo_occ, "OH")
        assert info.unrestricted
        assert info.n_occupied == 5 and info.n_occupied_beta == 4
        assert info.homo_lumo_gap_ev == pytest.approx(20.08, abs=0.02)

    def test_session_result_gap_is_frontier_gap(self):
        pytest.importorskip("pyscf")
        from quantui.session_calc import run_in_session

        mol = Molecule(
            atoms=["O", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
            multiplicity=2,
        )
        res = run_in_session(mol, method="UHF", basis="STO-3G", verbose=0)
        assert res.homo_lumo_gap_ev == pytest.approx(20.08, abs=0.02)

    def test_diagram_draws_alpha_and_beta_columns(self):
        from quantui.orbital_visualization import (
            orbital_info_from_arrays,
            orbital_summary_html,
            plot_orbital_diagram_plotly,
        )

        mf = _oh_uhf()
        info = orbital_info_from_arrays(mf.mo_energy, mf.mo_occ, "OH")
        fig = plot_orbital_diagram_plotly(info)
        # 6 alpha + 6 beta MOs, in two columns.
        assert len(fig.data) == 12
        xs = sorted({round(float(np.mean(t.x)), 3) for t in fig.data})
        assert len(xs) == 2
        hovers = " ".join(t.hovertemplate for t in fig.data)
        assert " α" in hovers and " β" in hovers
        texts = [a.text for a in fig.layout.annotations]
        assert "<b>20.08 eV</b>" in texts
        assert any("HOMO" in t and "β" in t for t in texts)
        assert "(α)" in orbital_summary_html(info)

    def test_restricted_diagram_unchanged(self):
        from quantui.orbital_visualization import (
            orbital_info_from_arrays,
            plot_orbital_diagram_plotly,
        )

        e = np.array([-1.0, -0.5, 0.4, 0.9])
        occ = np.array([2.0, 2.0, 0.0, 0.0])
        info = orbital_info_from_arrays(e, occ, "X")
        assert not info.unrestricted
        fig = plot_orbital_diagram_plotly(info)
        assert len(fig.data) == 4
        assert {tuple(t.x) for t in fig.data} == {(-0.3, 0.3)}
        assert [a.text for a in fig.layout.annotations][:2] == [
            "<b>HOMO</b>",
            "<b>LUMO</b>",
        ]


# ── VIZ-5 ────────────────────────────────────────────────────────────────


def _water_orbital_state(app):
    """Water-like restricted orbital state: 7 MOs, 5 occupied (HOMO = MO 5)."""
    from quantui.orbital_visualization import orbital_info_from_arrays

    e = np.array([-20.2, -1.27, -0.62, -0.45, -0.39, 0.58, 0.69])
    occ = np.array([2.0, 2, 2, 2, 2, 0, 0])
    app._last_orb_info = orbital_info_from_arrays(e, occ, "H2O")
    app._last_orb_mo_energy = e
    app._last_orb_mo_occ = occ
    app._last_orb_mo_coeff = np.eye(7)
    app._last_orb_mol_atom = [["O", (0, 0, 0)]]
    app._last_orb_mol_basis = "sto-3g"


class TestOneBasedMoNumbers:
    def test_input_widget_is_one_based(self, app):
        assert app._orb_index_input.min == 1
        assert app._orb_index_input.value == 1

    def test_typed_number_maps_to_the_same_mo_as_the_hover(
        self, app, monkeypatch, tmp_path
    ):
        import quantui.orbital_visualization as ov
        from quantui.app_visualization import render_orbital_isosurface

        _water_orbital_state(app)
        app._last_result_dir = tmp_path
        seen = []

        def _fake_cube(_atom, _basis, _coeff, idx, *_a, **_k):
            seen.append(idx)
            raise RuntimeError("stop after index check")

        monkeypatch.setattr(ov, "generate_cube_from_arrays", _fake_cube)
        # The diagram's hover calls the HOMO "MO #5"; typing 5 must draw it.
        render_orbital_isosurface(app, "MO 5")
        render_orbital_isosurface(app, "HOMO")
        assert seen == [4, 4]

    def test_out_of_range_error_uses_mo_numbers(self, app):
        from quantui.app_visualization import render_orbital_isosurface

        _water_orbital_state(app)
        render_orbital_isosurface(app, "MO 8")
        html = app._orb_iso_output.outputs[-1]["data"]["text/html"]
        assert "1–7" in html and "HOMO = MO 5" in html
        render_orbital_isosurface(app, "MO 0")  # below the 1-based range
        assert "out of range" in app._orb_iso_output.outputs[-1]["data"]["text/html"]

    def test_gallery_caption_is_one_based(self):
        from quantui.orbital_visualization import gallery_caption

        # 0-based index 4 is the water HOMO, MO #5 in the diagram hover.
        assert gallery_caption("HOMO", 4, None) == ("HOMO", "MO 5")
