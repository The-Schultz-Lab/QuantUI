"""Regression tests for the Medium VIZ audit findings (2026-11-10 audit).

- VIZ-3: a theme toggle must keep the persistent single vib viewer (and the
  vib HTML cache must not serve the other theme's background).
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
