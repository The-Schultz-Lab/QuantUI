"""Orbital isosurface PNG export and tunable resolution (M-ORBEXPORT ORBX.1/.2).

Two features that fail quietly rather than loudly, which is what these guard:

**Resolution has two knobs.** The cubegen grid controls real fidelity and what
lands in the saved ``.cube``; a separate stride cap bounds what reaches a Plotly
figure. Raise the grid alone and the user waits longer for a finer cube, then
the render throws the extra detail away — the feature *looks broken* to the
person who asked for it. (Only the Plotly path strides; py3Dmol isosurfaces the
cube in-browser at full resolution, so there a finer grid shows up immediately.)

**PNG capture crosses the JS/kernel boundary.** The viewer's Save-PNG button
writes a data URI into a hidden Textarea and dispatches an ``input`` event, and
ipywidgets syncs it back. Three things can drift apart without any error: the
CSS class the JS targets vs. the one the widget carries, the button being
rendered with nowhere to deliver to, and the decode path silently accepting
malformed input.

No browser, no PySCF, no GPU — the JS is asserted as text and the Python half is
exercised directly.
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from quantui.app_exports import _PNG_URI_PREFIX, on_orb_png_captured
from quantui.orbital_visualization import (
    DEFAULT_ISO_RESOLUTION,
    ISO_RESOLUTION_OPTIONS,
    ISO_RESOLUTION_PRESETS,
    max_render_points,
    render_orbital_isosurface_py3dmol,
)

# A genuine 1x1 PNG. Using real bytes means the decode path is actually tested
# rather than a base64 round-trip of arbitrary data.
_REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def cube_file(tmp_path: Path) -> Path:
    """A minimal but structurally valid Gaussian cube."""
    n = 4
    lines = [
        "orbital",
        "cube",
        "    2    0.000000    0.000000    0.000000",
        f"    {n}    0.500000    0.000000    0.000000",
        f"    {n}    0.000000    0.500000    0.000000",
        f"    {n}    0.000000    0.000000    0.500000",
        "    1    1.000000    0.000000    0.000000    0.000000",
        "    1    1.000000    1.400000    0.000000    0.000000",
    ]
    vals = np.random.RandomState(0).normal(size=n**3) * 0.05
    for i in range(0, len(vals), 6):
        lines.append("".join(f"{v:13.5E}" for v in vals[i : i + 6]))
    p = tmp_path / "orb.cube"
    p.write_text("\n".join(lines) + "\n")
    return p


class TestResolutionMovesBothKnobs:
    def test_the_default_preset_is_the_historical_grid(self):
        # 60³ was the hardcoded value before this was tunable. Changing the
        # default silently changes every user's cube fidelity and runtime.
        assert ISO_RESOLUTION_PRESETS[DEFAULT_ISO_RESOLUTION] == 60

    def test_presets_are_ordered_coarse_to_fine(self):
        values = [ISO_RESOLUTION_PRESETS[key] for _, key in ISO_RESOLUTION_OPTIONS]
        assert values == sorted(values), "dropdown order must match grid density"

    def test_every_dropdown_option_maps_to_a_preset(self):
        # A label with no preset would silently fall back to the default at
        # generate time, so picking "Fine" would do nothing.
        for _, key in ISO_RESOLUTION_OPTIONS:
            assert key in ISO_RESOLUTION_PRESETS

    def test_a_finer_grid_raises_the_render_cap(self):
        """The whole point of ORBX.2. Without this, choosing Fine costs time
        and buys nothing on the Plotly path."""
        assert max_render_points(80) > max_render_points(60)
        assert max_render_points(100) > max_render_points(80)

    def test_the_cap_never_drops_below_the_historical_value(self):
        # Coarse must not make the fallback renderer *worse* than it was.
        for grid in ISO_RESOLUTION_PRESETS.values():
            assert max_render_points(grid) >= 48_000

    def test_the_cap_is_bounded(self):
        # A cap that scales without limit hands the browser a Plotly trace it
        # cannot interact with — a fallback that locks the tab is worse than a
        # slightly coarse one.
        assert max_render_points(500) <= 250_000

    def test_the_cap_scales_with_volume_not_edge_length(self):
        # Grid work is nx*ny*nz. A linear scaling would under-provision badly:
        # 100³ is 4.6x the points of 60³, not 1.7x.
        ratio = max_render_points(100) / max_render_points(60)
        assert 4.0 < ratio < 5.0, ratio

    def test_bad_input_does_not_raise(self):
        # This feeds a renderer; a stray zero must not produce a divide or a
        # zero cap that strides everything away.
        assert max_render_points(0) >= 48_000
        assert max_render_points(-5) >= 48_000


class TestResolutionPersists:
    def test_the_setting_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "s.json"))
        from quantui.user_settings import UserSettings

        s = UserSettings.load()
        assert s.viz.iso_resolution == DEFAULT_ISO_RESOLUTION
        s.viz.iso_resolution = "fine"
        s.save()
        assert UserSettings.load().viz.iso_resolution == "fine"

    def test_an_unknown_value_falls_back_rather_than_propagating(
        self, tmp_path, monkeypatch
    ):
        # A settings file written by a future version with more presets must
        # degrade to the default, not hand a nonsense key to cubegen.
        import json

        from quantui.user_settings import _SCHEMA_VERSION, UserSettings

        # _schema_version matters: without it the loader discards the file
        # wholesale and returns defaults, so this would pass without the
        # validator ever seeing "ultra".
        p = tmp_path / "s.json"
        p.write_text(
            json.dumps(
                {"_schema_version": _SCHEMA_VERSION, "viz": {"iso_resolution": "ultra"}}
            )
        )
        monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(p))

        assert UserSettings.load().viz.iso_resolution == DEFAULT_ISO_RESOLUTION

    def test_every_preset_is_accepted_by_the_settings_validator(
        self, tmp_path, monkeypatch
    ):
        # user_settings deliberately does not import orbital_visualization, so
        # its valid-value tuple is a copy. This is the test that keeps the copy
        # honest.
        import json

        from quantui.user_settings import _SCHEMA_VERSION, UserSettings

        for key in ISO_RESOLUTION_PRESETS:
            p = tmp_path / f"{key.replace(' ', '_')}.json"
            p.write_text(
                json.dumps(
                    {"_schema_version": _SCHEMA_VERSION, "viz": {"iso_resolution": key}}
                )
            )
            monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(p))
            assert UserSettings.load().viz.iso_resolution == key


class TestCaptureButtonIsOptIn:
    def test_no_button_without_an_inbox(self, cube_file):
        # A Save-PNG button with nowhere to deliver would look functional and
        # do nothing — the failure mode this feature most needs to avoid.
        html = render_orbital_isosurface_py3dmol(cube_file)
        assert "Save PNG" not in html
        assert "pngURI" not in html

    def test_the_button_appears_when_an_inbox_is_named(self, cube_file):
        html = render_orbital_isosurface_py3dmol(cube_file, capture_class="my-inbox")
        assert "Save PNG" in html
        assert "pngURI()" in html
        assert "my-inbox" in html

    def test_the_js_targets_the_class_the_widget_actually_carries(self, cube_file):
        # The JS and the widget agree only because both read this constant. If
        # someone hardcodes either side, capture breaks with no error anywhere.
        from quantui.app_builders import _ORB_PNG_INBOX_CLASS

        html = render_orbital_isosurface_py3dmol(
            cube_file, capture_class=_ORB_PNG_INBOX_CLASS
        )
        assert '"." + CLS' in html or '"."+CLS' in html
        assert _ORB_PNG_INBOX_CLASS in html

    def test_the_sync_event_is_dispatched(self, cube_file):
        # Setting textarea.value alone is invisible to the widget model — the
        # 'input' event is the entire mechanism by which the kernel finds out.
        html = render_orbital_isosurface_py3dmol(cube_file, capture_class="x")
        assert 'dispatchEvent(new Event("input"' in html
        assert "bubbles:true" in html

    def test_the_button_binds_to_this_viewer(self, cube_file):
        # Two viewers on one page must not capture each other's canvas.
        html = render_orbital_isosurface_py3dmol(cube_file, capture_class="x")
        uid = re.search(r"3dmolviewer_(\w+)", html).group(1)
        assert f'var UID="{uid}"' in html
        assert 'window["viewer_"+UID]' in html

    def test_capture_reads_the_live_viewer_not_a_re_render(self, cube_file):
        # Per GOTCHAS, camera state does not survive an atomic HTML swap, so
        # anything that re-renders before grabbing the canvas exports the
        # DEFAULT camera — destroying the only reason to capture client-side.
        html = render_orbital_isosurface_py3dmol(cube_file, capture_class="x")
        capture = html[html.index("addEventListener") :]
        assert "pngURI()" in capture
        for forbidden in ("addModel", "zoomTo", "render()"):
            assert (
                forbidden not in capture
            ), f"capture path calls {forbidden} — that resets the camera"


class TestCapturedPngIsWritten:
    @staticmethod
    def _app(dest: Path) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app._last_cube_orbital = "HOMO"
        app._orb_png_inbox = Mock(value="pending")
        app._iso_export_status = Mock(value="")
        return app

    def _uri(self, raw: bytes = _REAL_PNG) -> str:
        return _PNG_URI_PREFIX + base64.b64encode(raw).decode()

    def test_a_capture_lands_on_disk_byte_for_byte(self, tmp_path):
        app = self._app(tmp_path)
        on_orb_png_captured(app, {"new": self._uri()})
        assert (tmp_path / "HOMO.png").read_bytes() == _REAL_PNG
        assert "Saved" in app._iso_export_status.value

    def test_the_inbox_is_cleared_so_the_same_view_can_be_saved_twice(self, tmp_path):
        # observe() only fires on *change*; leaving the URI in place would make
        # a second identical capture silently do nothing.
        app = self._app(tmp_path)
        on_orb_png_captured(app, {"new": self._uri()})
        assert app._orb_png_inbox.value == ""

    def test_the_inbox_is_cleared_on_failure_too(self, tmp_path):
        app = self._app(tmp_path)
        on_orb_png_captured(app, {"new": "garbage"})
        assert app._orb_png_inbox.value == ""

    def test_the_filename_follows_the_orbital_label(self, tmp_path):
        app = self._app(tmp_path)
        app._last_cube_orbital = "LUMO+1"
        on_orb_png_captured(app, {"new": self._uri()})
        assert (tmp_path / "LUMO+1.png").exists()

    def test_a_hostile_label_cannot_escape_the_result_directory(self, tmp_path):
        # The label reaches here from app state, but it feeds a filesystem path
        # and sanitising it costs nothing.
        app = self._app(tmp_path)
        app._last_cube_orbital = "../../etc/passwd"
        on_orb_png_captured(app, {"new": self._uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].parent == tmp_path

    @pytest.mark.parametrize(
        "payload,expect",
        [
            ("", "no status change"),
            ("http://example.com/x.png", "unexpected image format"),
            (_PNG_URI_PREFIX + "!!!not base64!!!", "corrupt image data"),
        ],
    )
    def test_malformed_input_is_reported_not_written(self, tmp_path, payload, expect):
        app = self._app(tmp_path)
        on_orb_png_captured(app, {"new": payload})
        assert not list(tmp_path.glob("*.png"))
        if payload:
            assert expect in app._iso_export_status.value

    def test_an_oversized_payload_is_refused_before_decoding(self, tmp_path):
        # Guards against turning a runaway data URI into a 64 MB+ allocation.
        app = self._app(tmp_path)
        on_orb_png_captured(app, {"new": _PNG_URI_PREFIX + "A" * (65 * 1024 * 1024)})
        assert not list(tmp_path.glob("*.png"))
        assert "too large" in app._iso_export_status.value

    def test_a_missing_result_directory_is_reported(self):
        app = self._app(Path(tempfile.mkdtemp()))
        app._last_result_dir = None
        on_orb_png_captured(app, {"new": self._uri()})
        assert "result folder" in app._iso_export_status.value


class TestPlotlymolIsOffForOrbitalsAndVibExport:
    """py3Dmol-only routing, 2026-08-04 (user decision).

    Both were reachable via plotlymol before. The user's reasoning — "plotlymol
    is really only good for the molecule viewing windows" — has two concrete
    consequences these tests pin:

      - An orbital isosurface must never be rendered by the strided Plotly path,
        which is downsampled by construction and cannot carry Save-PNG capture.
      - An exported vibrational animation must be the one the user was watching.

    Deliberately reversible: each is one line in ``_TASK_POLICY``. These tests
    are what makes a revert loud rather than silent.
    """

    @pytest.mark.parametrize("task", ["ORBITAL_ISOSURFACE", "VIB_EXPORT"])
    def test_preference_cannot_select_plotlymol(self, task):
        from quantui.viz_backend_router import (
            BackendAvailability,
            VizBackend,
            VizPreference,
            VizTask,
            select_backend,
        )

        both = BackendAvailability(py3dmol=True, plotlymol=True)
        for pref in VizPreference:
            decision = select_backend(getattr(VizTask, task), pref, both)
            assert (
                decision.chosen == VizBackend.PY3DMOL
            ), f"{task} resolved to {decision.chosen} under preference {pref}"

    def test_molecule_viewing_still_honours_the_preference(self):
        # The narrowing must not leak: plotlymol remains selectable for the
        # molecule viewers, which is the case the user explicitly kept.
        from quantui.viz_backend_router import (
            BackendAvailability,
            VizBackend,
            VizPreference,
            VizTask,
            select_backend,
        )

        both = BackendAvailability(py3dmol=True, plotlymol=True)
        decision = select_backend(
            VizTask.MOLECULE_PREVIEW, VizPreference.PLOTLYMOL, both
        )
        assert decision.chosen == VizBackend.PLOTLYMOL

    def test_the_vib_export_builder_has_no_plotly_branch(self):
        # Unlike the Plotly isosurface path — kept, still tested, one line from
        # being restored — this branch duplicated animation-building logic that
        # would rot silently behind a flag, so it was removed outright.
        import inspect

        from quantui.app_visualization import build_vib_export_html

        code = "\n".join(
            ln
            for ln in inspect.getsource(build_vib_export_html).splitlines()
            if not ln.strip().startswith("#")
        )
        body = code[code.index('"""', code.index('"""') + 3) :]
        assert "create_vibration_animation" not in body
        assert "availability.plotlymol" not in body

    def test_the_plotly_isosurface_renderer_is_kept_not_deleted(self):
        # The revert path must stay cheap. Removing this function would turn a
        # one-line policy change into a rewrite.
        from quantui.orbital_visualization import plot_cube_isosurface

        assert callable(plot_cube_isosurface)


class TestThemeChangeReachesThe3DScenes:
    """Reported 2026-08-04: *"the background of the animations and isosurface
    plots is sticky to the theme... but will change if I calculate a new
    isosurface."*

    py3Dmol paints the background into the WebGL scene at render time rather
    than reading it from CSS, so nothing re-reads it on a theme toggle. The fix
    re-renders from cached inputs; these tests pin both that it happens and that
    it stays cheap.
    """

    def test_the_rerender_uses_the_new_background(self, cube_file):
        from quantui.app_visualization import rerender_3d_scenes_for_theme

        seen = {}
        app = Mock()
        app._last_cube_path = cube_file
        app._last_vib_molecule = None
        app._last_vib_freq_result = None
        app._orb_png_inbox = Mock()
        app._set_html_output = lambda out, html: seen.update(html=html)
        app._plotly_theme_colors = lambda: {"scene_bgcolor": "#1e1e1e"}

        rerender_3d_scenes_for_theme(app)
        assert "#1e1e1e" in seen["html"]

    def test_the_rerender_keeps_the_save_png_button(self, cube_file):
        # A theme toggle must not quietly strip a feature off the viewer.
        from quantui.app_visualization import rerender_3d_scenes_for_theme

        seen = {}
        app = Mock()
        app._last_cube_path = cube_file
        app._last_vib_molecule = None
        app._last_vib_freq_result = None
        app._orb_png_inbox = Mock()
        app._set_html_output = lambda out, html: seen.update(html=html)
        app._plotly_theme_colors = lambda: {"scene_bgcolor": "#fff"}

        rerender_3d_scenes_for_theme(app)
        assert "Save PNG" in seen["html"]

    def test_it_never_regenerates_the_cube(self):
        """The re-render must re-read the cube on disk, never re-run cubegen.

        cubegen is 15-30 s at the default grid and up to ~4.6x that at the
        finest — running it on a theme toggle would make the toggle unusable.
        """
        import inspect

        from quantui.app_visualization import rerender_3d_scenes_for_theme

        src = inspect.getsource(rerender_3d_scenes_for_theme)
        body = src[src.index('"""', src.index('"""') + 3) :]
        assert "generate_cube_from_arrays" not in body
        assert "render_orbital_isosurface(" not in body  # the full generate path

    def test_a_theme_toggle_cannot_raise(self, tmp_path):
        # Nothing generated yet, and a cube that has since been deleted. Either
        # would otherwise turn a theme click into a traceback.
        from quantui.app_visualization import rerender_3d_scenes_for_theme

        for cube in (None, tmp_path / "deleted.cube"):
            app = Mock()
            app._last_cube_path = cube
            app._last_vib_molecule = None
            app._last_vib_freq_result = None
            app._plotly_theme_colors = lambda: {"scene_bgcolor": "#fff"}
            rerender_3d_scenes_for_theme(app)  # must not raise

    def test_the_theme_handler_actually_calls_it(self):
        # The function existing is worthless if nothing invokes it.
        import inspect

        from quantui.app import QuantUIApp

        src = inspect.getsource(QuantUIApp._rerender_plotly_theme)
        assert "rerender_3d_scenes_for_theme" in src


class TestIsosurfaceSwapsAreAtomic:
    """Reported 2026-08-04: *"the screen jumps up when I click to calculate a
    new isosurface."*

    ``clear_output()`` followed by ``display()`` leaves the output empty for a
    moment. The panel collapses from ~660px to zero, the document shrinks, the
    browser clamps scrollTop to the new maximum — and the content returning does
    not scroll back. ``_set_html_output`` swaps ``outputs`` in one assignment so
    the empty state is never observed, which is the same fix already applied to
    the IR toggle, the trajectory stepper and the vib viewer.
    """

    def test_no_clear_then_display_on_the_isosurface_output(self):
        import pathlib
        import re

        import quantui

        src = (
            pathlib.Path(quantui.__file__).parent / "app_visualization.py"
        ).read_text(encoding="utf-8")
        # A bare clear_output() immediately followed by a `with` block is the
        # pattern that collapses the panel.
        offenders = re.findall(
            r"app\._orb_iso_output\.clear_output\(\)\s*\n\s*with app\._orb_iso_output:",
            src,
        )
        assert not offenders, (
            "isosurface output is cleared then repopulated — use "
            "app._set_html_output for an atomic swap"
        )
