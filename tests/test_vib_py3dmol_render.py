"""Unit tests for the py3Dmol vibrational animation renderer (VIZBACK.8).

Covers ``_render_vib_mode_py3dmol`` and the router-driven dispatch in
``render_vib_mode``. The py3Dmol path is plotlymol3d-independent and reads
displacements directly from ``freq_result.displacements`` (1-indexed by
``mode_number``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from quantui.app import QuantUIApp
from quantui.app_visualization import (
    _render_vib_mode_py3dmol,
    render_vib_mode,
)
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
    """Three vibrational modes for a 3-atom system, with realistic shapes."""
    return SimpleNamespace(
        frequencies_cm1=[1600.0, 3800.0, 3850.0],
        ir_intensities=[40.0, 10.0, 50.0],
        displacements=[
            np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]]),
            np.array([[0.0, 0.10, 0.0], [0.05, -0.10, 0.0], [-0.05, -0.10, 0.0]]),
            np.array([[0.0, 0.0, 0.10], [0.05, 0.0, -0.10], [-0.05, 0.0, -0.10]]),
        ],
    )


class TestRenderVibModePy3Dmol:
    def test_renders_html_blob_into_vib_output(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)

        outputs = app.vib_output.outputs
        assert len(outputs) == 1, "atomic outputs swap should yield 1 entry"
        html = outputs[0]["data"]["text/html"]
        assert "3Dmol" in html or "py3Dmol" in html
        assert "animate" in html.lower()

    def test_uses_correct_mode_displacements(self, app, water_mol, fake_freq_result):
        """The HTML should contain coordinates that reflect mode-2's
        displacement pattern (different from mode-1's)."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol

        # Render mode 1, capture html
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        html_1 = app.vib_output.outputs[0]["data"]["text/html"]

        # Render mode 2, capture html
        _render_vib_mode_py3dmol(app, water_mol, mode_number=2)
        html_2 = app.vib_output.outputs[0]["data"]["text/html"]

        # Two different modes should produce DIFFERENT HTML (different XYZ
        # frame coordinates). They share boilerplate so simple length check
        # may not catch it — assert they differ.
        assert html_1 != html_2

    def test_n_frames_default_is_24(self, app, water_mol, fake_freq_result):
        """The XYZ multi-frame string should contain 24 frame headers per
        the VIZBACK.8 spec (one full oscillation)."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        html = app.vib_output.outputs[0]["data"]["text/html"]
        # Each frame has a header line "mode N phase ..." - count them.
        # The frames are concatenated in the XYZ payload embedded in the HTML.
        frame_count = html.count("phase +") + html.count("phase -")
        assert frame_count == 24, f"expected 24 frames, found {frame_count}"

    def test_amplitude_scales_displacement(self, app, water_mol, fake_freq_result):
        """Higher amplitude should produce larger coordinate excursions."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol

        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, amplitude=0.1)
        html_low = app.vib_output.outputs[0]["data"]["text/html"]
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, amplitude=0.8)
        html_high = app.vib_output.outputs[0]["data"]["text/html"]
        # Different amplitudes should produce different HTML.
        assert html_low != html_high

    def test_missing_freq_result_shows_error_not_crash(self, app, water_mol):
        """If _last_vib_freq_result is None, should show a user-facing error
        rather than raising."""
        app._last_vib_freq_result = None
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        # outputs should contain an error message
        assert len(app.vib_output.outputs) >= 1

    def test_out_of_range_mode_shows_error_not_crash(
        self, app, water_mol, fake_freq_result
    ):
        """Asking for mode 99 (out of range) should be a graceful error."""
        app._last_vib_freq_result = fake_freq_result
        _render_vib_mode_py3dmol(app, water_mol, mode_number=99)
        assert len(app.vib_output.outputs) >= 1

    def test_shape_mismatch_shows_error_not_crash(self, app):
        """Displacements with wrong shape vs molecule should error gracefully."""
        mol = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        bad_freq = SimpleNamespace(
            frequencies_cm1=[1000.0],
            displacements=[np.array([[0, 0.1, 0]])],  # only 1 atom!
        )
        app._last_vib_freq_result = bad_freq
        _render_vib_mode_py3dmol(app, mol, mode_number=1)
        assert len(app.vib_output.outputs) >= 1


class TestVibCacheIntegration:
    """VIZBACK.9: `_render_vib_mode_py3dmol` should check and populate the
    on-disk cache at ``<result_dir>/vib_frames/``."""

    def test_cache_miss_then_hit_returns_same_html(
        self, app, water_mol, fake_freq_result, tmp_path
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path

        # First call: cache miss → render → save
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        first_html = app.vib_output.outputs[0]["data"]["text/html"]
        # Cache file should now exist
        from quantui.vib_cache import cache_dir

        assert (cache_dir(tmp_path) / "mode_001.html").exists()
        assert (cache_dir(tmp_path) / "index.json").exists()

        # Reset the output before the second call to confirm cache delivers it
        app.vib_output.outputs = ()

        # Second call: cache hit → identical HTML, no re-render
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        second_html = app.vib_output.outputs[0]["data"]["text/html"]
        assert second_html == first_html

    def test_no_result_dir_skips_cache_no_error(self, app, water_mol, fake_freq_result):
        """Without _last_result_dir set, render still works (cache skipped)."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = None  # explicit
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        assert len(app.vib_output.outputs) == 1

    def test_changed_amplitude_invalidates_cache(
        self, app, water_mol, fake_freq_result, tmp_path
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path

        # Render with default amplitude
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, amplitude=0.4)
        html_default = app.vib_output.outputs[0]["data"]["text/html"]

        # Render with different amplitude — should NOT hit the prior cache,
        # should render fresh and produce different HTML
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, amplitude=0.8)
        html_alt = app.vib_output.outputs[0]["data"]["text/html"]

        assert html_default != html_alt


class TestAppWrapperAcceptsRenderToken:
    """Regression test for BUG-VIB-WRAPPER-SIG (session 50): the
    QuantUIApp wrapper ``_render_vib_mode`` must accept ``render_token``
    via kwargs — show_vib_animation / on_vib_mode_changed pass it via
    `threading.Thread(kwargs={"render_token": ...})`. A mismatched
    signature causes the thread to die silently with TypeError and the
    user sees a permanent 'Rendering...' placeholder."""

    def test_wrapper_accepts_render_token_kwarg(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._vib_render_token = 1
        # This call shape mirrors what show_vib_animation's thread does.
        # Must not raise TypeError.
        app._render_vib_mode(None, water_mol, 1, render_token=1)
        assert len(app.vib_output.outputs) == 1


class TestRenderTokenStaleness:
    """The _vib_render_token machinery guarantees that an older render
    thread cannot stomp a newer render's output. Verifies the bug-fix for
    intermittent missing-render symptom on rapid mode switching."""

    def test_stale_token_skips_output_write(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol

        # First, set a baseline output the test can detect.
        app.vib_output.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/html": "BASELINE"},
                "metadata": {},
            },
        )

        # Bump the app's render token to N+1; pass N (stale) to the render.
        # The render should bail and leave the baseline output untouched.
        app._vib_render_token = 5
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, render_token=2)
        assert (
            app.vib_output.outputs[0]["data"]["text/html"] == "BASELINE"
        ), "stale render should not overwrite output"

    def test_matching_token_writes_output(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol

        app._vib_render_token = 7
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, render_token=7)
        outputs = app.vib_output.outputs
        assert len(outputs) == 1
        assert "3Dmol" in outputs[0]["data"]["text/html"]


class TestFpsParameter:
    """fps parameter wiring — value flows into the py3Dmol animate
    interval and into the cache key."""

    def test_fps_affects_animate_interval(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        # 10 fps → interval=100; 60 fps → interval≈17
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, fps=10)
        html_10 = app.vib_output.outputs[0]["data"]["text/html"]
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, fps=60)
        html_60 = app.vib_output.outputs[0]["data"]["text/html"]
        # interval is embedded in the HTML — different fps → different blob
        assert html_10 != html_60
        assert "interval" in html_10.lower()

    def test_fps_changes_invalidate_cache(
        self, app, water_mol, fake_freq_result, tmp_path
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path
        # Render at fps=10 → cache populated for fps=10
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1, fps=10)
        # Re-render at fps=30 → cache should NOT hit (different fps)
        from quantui.vib_cache import has_cached

        assert has_cached(
            tmp_path,
            1,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )
        assert not has_cached(
            tmp_path,
            1,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=30,
        )

    def test_fps_falls_back_to_user_settings_when_not_passed(
        self, app, water_mol, fake_freq_result, tmp_path
    ):
        """When `fps` argument is None, render reads from
        ``app._user_settings.viz.vib_framerate_fps``. We verify by saving
        to cache and checking the cached index records the settings fps."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path
        app._user_settings.viz.vib_framerate_fps = 45
        # Don't pass fps; should pick up 45 from settings → cache should
        # be keyed by fps=45.
        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)
        from quantui.vib_cache import has_cached, load_index

        assert load_index(tmp_path)["fps"] == 45
        assert has_cached(
            tmp_path,
            1,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=45,
        )
        # Wrong-fps query should miss
        assert not has_cached(
            tmp_path,
            1,
            n_frames=24,
            amplitude=0.4,
            renderer="py3dmol",
            fps=10,
        )


class TestRenderVibModeDispatch:
    """render_vib_mode should route through the viz_backend_router."""

    def test_auto_preference_routes_to_py3dmol(self, app, water_mol, fake_freq_result):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        # Force preference = auto. Router policy for VIB_INTERACTIVE is
        # (PY3DMOL, PLOTLYMOL) so auto → py3dmol.
        app._set_viz_preference("auto", persist=False)
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        render_vib_mode(app, vib_data=None, molecule=water_mol, mode_number=1)
        outputs = app.vib_output.outputs
        assert len(outputs) == 1
        html = outputs[0]["data"]["text/html"]
        # py3Dmol path produces 3Dmol-styled HTML, not Plotly.
        assert "3Dmol" in html or "py3Dmol" in html

    def test_explicit_py3dmol_preference_uses_py3dmol(
        self, app, water_mol, fake_freq_result
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._set_viz_preference("py3dmol", persist=False)
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        render_vib_mode(app, vib_data=None, molecule=water_mol, mode_number=1)
        html = app.vib_output.outputs[0]["data"]["text/html"]
        assert "3Dmol" in html or "py3Dmol" in html

    def test_dispatch_does_not_hard_fail_without_plotlymol_vib_data(
        self, app, water_mol, fake_freq_result
    ):
        """The py3Dmol path should not require vib_data to be non-None
        (i.e., plotlymol3d-independent rendering)."""
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._set_viz_preference("py3dmol", persist=False)
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        # vib_data=None mimics plotlymol3d unavailable
        render_vib_mode(app, vib_data=None, molecule=water_mol, mode_number=1)
        # Should still produce an HTML output (not a hard-fail error).
        outputs = app.vib_output.outputs
        assert len(outputs) == 1
        html = outputs[0]["data"]["text/html"]
        assert (
            "requires plotlymol3d" not in html
        ), f"py3Dmol path should not require plotlymol3d; html: {html[:200]}"
