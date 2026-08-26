"""M-EXPORT2 EXP2.2: PNG capture wired into the single-persistent-viewer
vibrational mode animation — the fourth viewer to get the shared capture
bridge (isosurface, reorg-geometry, trajectory, now vibrational).

No browser, no PySCF — the JS is asserted as text (same approach as the
other EXP2.2 test files) and the Python handler is exercised directly.
Unlike the trajectory panel, the vib panel's widgets (mode dropdown, export
button, js bridge) are built once in app_builders — same shape as the
isosurface/reorg panels — so the capture handler reads persistent
``app._vib_png_inbox`` / ``app._vib_png_status`` attributes rather than
taking them as explicit arguments.

Deliberately not offered on the legacy per-mode plotlymol3d fallback
(``_render_vib_mode_py3dmol`` / ``_render_vib_mode_plotlymol``) — only the
single-viewer py3Dmol path (``build_vib_viewer_html`` /
``_render_vib_single_viewer``) has an equivalent client-side capture global.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from quantui.app_exports import _PNG_URI_PREFIX, on_vib_png_captured
from quantui.app_visualization import build_vib_viewer_html
from quantui.molecule import Molecule

# Same 1x1 PNG used by the other EXP2.2 capture tests — real bytes exercise
# the actual decode path rather than a base64 round-trip of arbitrary data.
_REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_uri(raw: bytes = _REAL_PNG) -> str:
    return _PNG_URI_PREFIX + base64.b64encode(raw).decode()


def _water() -> Molecule:
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


def _fake_freq_result() -> SimpleNamespace:
    return SimpleNamespace(
        frequencies_cm1=[1600.0, 3800.0, 3850.0],
        displacements=[
            np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]]),
            np.array([[0.0, 0.10, 0.0], [0.05, -0.10, 0.0], [-0.05, -0.10, 0.0]]),
            np.array([[0.0, 0.0, 0.10], [0.05, 0.0, -0.10], [-0.05, 0.0, -0.10]]),
        ],
    )


class TestVibViewerCaptureWiring:
    """Mirrors TestTrajectoryViewerCaptureWiring — the vib viewer must get
    its own uid-scoped capture function, never another viewer's."""

    def test_omits_the_button_when_capture_class_is_empty(self):
        html = build_vib_viewer_html(_water(), _fake_freq_result(), [1, 2, 3], 1)
        assert "orb_png_" not in html

    def test_wires_a_uid_scoped_capture_function(self):
        html = build_vib_viewer_html(
            _water(),
            _fake_freq_result(),
            [1, 2, 3],
            1,
            capture_class="quantui-vib-png-inbox",
        )
        assert "__quantuiVibCapture_" in html
        assert 'id="orb_png_' in html
        assert "quantui-vib-png-inbox" in html
        # Never falls back to another viewer's capture function name.
        assert 'window["__quantuiIsoCapture"]' not in html
        assert "__quantuiReorgCapture_" not in html
        assert "__quantuiTrajCapture_" not in html

    def test_capture_button_appears_alongside_the_mode_switch_js(self):
        html = build_vib_viewer_html(
            _water(),
            _fake_freq_result(),
            [1, 2, 3],
            1,
            capture_class="quantui-vib-png-inbox",
        )
        assert "Save PNG" in html
        # The mode-switch bridge is still present — capture is additive.
        assert "__quantuiVibSetMode" in html


class TestVibPngCaptureHandler:
    @staticmethod
    def _app(dest: Path) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app._last_vib_molecule = _water()
        app._last_vib_freq_result = _fake_freq_result()
        app.vib_mode_dd = Mock(value=2)
        app.method_dd = Mock(value="B3LYP")
        app.basis_dd = Mock(value="6-31G*")
        app._vib_png_status = Mock(value="")
        app._vib_png_inbox = Mock(value="pending")
        return app

    def test_a_capture_lands_on_disk(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "mode2" in written[0].name
        assert "Saved" in app._vib_png_status.value

    def test_the_saved_png_carries_method_basis_and_frequency_metadata(self, tmp_path):
        from PIL import Image

        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))[0]
        with Image.open(written) as im:
            assert im.text["Method"] == "B3LYP"
            assert im.text["Basis"] == "6-31G*"
            # mode 2 (1-indexed) -> frequencies_cm1[1] == 3800.0
            assert im.text["Frequency (cm-1)"] == "3800.0"

    def test_inbox_is_cleared_after_a_capture(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": _png_uri()})
        assert app._vib_png_inbox.value == ""

    def test_inbox_is_cleared_on_failure_too(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": "garbage"})
        assert app._vib_png_inbox.value == ""

    def test_malformed_prefix_is_reported_not_written(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": "http://example.com/x.png"})
        assert not list(tmp_path.glob("*.png"))
        assert "unexpected image format" in app._vib_png_status.value

    def test_oversized_payload_is_refused_before_decoding(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {"new": _PNG_URI_PREFIX + "A" * (65 * 1024 * 1024)})
        assert not list(tmp_path.glob("*.png"))
        assert "too large" in app._vib_png_status.value

    def test_missing_result_dir_is_reported(self):
        app = self._app(Path("unused"))
        app._last_result_dir = None
        on_vib_png_captured(app, {"new": _png_uri()})
        assert "run a calculation" in app._vib_png_status.value

    def test_empty_change_is_a_no_op(self, tmp_path):
        app = self._app(tmp_path)
        on_vib_png_captured(app, {})
        assert not list(tmp_path.glob("*.png"))

    def test_no_mode_selected_falls_back_to_a_generic_label(self, tmp_path):
        app = self._app(tmp_path)
        app.vib_mode_dd = Mock(value=None)
        on_vib_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name.endswith("_mode.png")  # generic label, no number

    def test_no_molecule_loaded_falls_back_to_a_generic_formula(self, tmp_path):
        app = self._app(tmp_path)
        app._last_vib_molecule = None
        on_vib_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "molecule" in written[0].name


class TestVibPngCaptureRealAppWiring:
    """End-to-end through a real QuantUIApp: confirms the widget the
    JS writes into is the same one whose observer calls the handler, and
    that app_builders actually built both persistent attributes with the
    right CSS class — not just that the two halves work in isolation."""

    def test_writing_to_the_inbox_triggers_a_real_save(self, tmp_path):
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        app._last_result_dir = tmp_path
        app._last_vib_molecule = _water()
        app._last_vib_freq_result = _fake_freq_result()
        app.vib_mode_dd.options = [("Mode 1", 1)]
        app.vib_mode_dd.value = 1

        assert "quantui-vib-png-inbox" in app._vib_png_inbox._dom_classes

        app._vib_png_inbox.value = _png_uri()  # simulate the JS write

        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "Saved" in app._vib_png_status.value
        assert app._vib_png_inbox.value == ""
