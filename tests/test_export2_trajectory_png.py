"""M-EXPORT2 EXP2.2: PNG capture wired into the geometry-optimization
trajectory viewer — the third viewer to get the shared capture bridge
(isosurface, then reorg-geometry, now trajectory).

No browser, no PySCF — the JS is asserted as text (same approach as
``test_export2_reorg_and_destination.py``) and the Python handler is
exercised directly. The trajectory panel rebuilds its widgets on every
render (unlike the isosurface/reorg accordions), so its capture handler
takes the inbox/status widgets explicitly rather than reading persistent
``app._traj_*`` attributes — that shape is covered here too.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import Mock

from quantui.app_exports import _PNG_URI_PREFIX, on_traj_png_captured
from quantui.app_visualization import build_trajectory_viewer_html

# Same 1x1 PNG used by the isosurface/reorg capture tests — real bytes
# exercise the actual decode path rather than a base64 round-trip of
# arbitrary data.
_REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_uri(raw: bytes = _REAL_PNG) -> str:
    return _PNG_URI_PREFIX + base64.b64encode(raw).decode()


def _xyzblocks() -> list[str]:
    return [
        "2\nH2\nH 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740000\n",
        "2\nH2\nH 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.760000\n",
    ]


class TestTrajectoryViewerCaptureWiring:
    """Mirrors TestReorgViewerCaptureWiring — the trajectory viewer must get
    its own uid-scoped capture function, never the isosurface's bare global."""

    def test_omits_the_button_when_capture_class_is_empty(self):
        html = build_trajectory_viewer_html(_xyzblocks(), formula="H2")
        assert "orb_png_" not in html

    def test_wires_a_uid_scoped_capture_function(self):
        html = build_trajectory_viewer_html(
            _xyzblocks(), formula="H2", capture_class="quantui-traj-png-inbox"
        )
        assert "__quantuiTrajCapture_" in html
        assert 'id="orb_png_' in html
        assert "quantui-traj-png-inbox" in html
        # Never falls back to the isosurface's bare global or the reorg name.
        assert 'window["__quantuiIsoCapture"]' not in html
        assert "__quantuiReorgCapture_" not in html

    def test_capture_button_still_appears_on_a_single_frame_trajectory(self):
        html = build_trajectory_viewer_html(
            _xyzblocks()[:1], formula="H2", capture_class="quantui-traj-png-inbox"
        )
        assert 'id="orb_png_' in html
        # Single frame: no stepper, but the capture button is independent of it.
        assert "Save PNG" in html

    def test_multi_frame_trajectory_keeps_both_stepper_and_capture_button(self):
        html = build_trajectory_viewer_html(
            _xyzblocks(), formula="H2", capture_class="quantui-traj-png-inbox"
        )
        assert "Save PNG" in html
        assert "Step " in html  # stepper label markup


class TestTrajectoryPngCaptureHandler:
    @staticmethod
    def _app(dest: Path) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app.method_dd = Mock(value="B3LYP")
        app.basis_dd = Mock(value="6-31G*")
        return app

    @staticmethod
    def _change(uri: str) -> dict:
        return {"new": uri, "owner": Mock(value=uri)}

    def test_a_capture_lands_on_disk(self, tmp_path):
        app = self._app(tmp_path)
        status = Mock(value="")
        on_traj_png_captured(app, self._change(_png_uri()), formula="H2", status=status)
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "trajectory" in written[0].name
        assert "Saved" in status.value

    def test_the_saved_png_carries_method_and_basis_metadata(self, tmp_path):
        from PIL import Image

        app = self._app(tmp_path)
        status = Mock(value="")
        on_traj_png_captured(app, self._change(_png_uri()), formula="H2", status=status)
        written = list(tmp_path.glob("*.png"))[0]
        with Image.open(written) as im:
            assert im.text["Method"] == "B3LYP"
            assert im.text["Basis"] == "6-31G*"

    def test_inbox_is_cleared_after_a_capture_via_change_owner(self, tmp_path):
        app = self._app(tmp_path)
        change = self._change(_png_uri())
        on_traj_png_captured(app, change, formula="H2", status=Mock(value=""))
        assert change["owner"].value == ""

    def test_inbox_is_cleared_on_failure_too(self, tmp_path):
        app = self._app(tmp_path)
        change = {"new": "garbage", "owner": Mock(value="garbage")}
        on_traj_png_captured(app, change, formula="H2", status=Mock(value=""))
        assert change["owner"].value == ""

    def test_malformed_prefix_is_reported_not_written(self, tmp_path):
        app = self._app(tmp_path)
        status = Mock(value="")
        on_traj_png_captured(
            app,
            {"new": "http://example.com/x.png", "owner": Mock(value="x")},
            formula="H2",
            status=status,
        )
        assert not list(tmp_path.glob("*.png"))
        assert "unexpected image format" in status.value

    def test_oversized_payload_is_refused_before_decoding(self, tmp_path):
        app = self._app(tmp_path)
        status = Mock(value="")
        big = _PNG_URI_PREFIX + "A" * (65 * 1024 * 1024)
        on_traj_png_captured(
            app, {"new": big, "owner": Mock(value=big)}, formula="H2", status=status
        )
        assert not list(tmp_path.glob("*.png"))
        assert "too large" in status.value

    def test_missing_result_dir_is_reported(self):
        app = self._app(Path("unused"))
        app._last_result_dir = None
        status = Mock(value="")
        on_traj_png_captured(app, self._change(_png_uri()), formula="H2", status=status)
        assert "run a calculation" in status.value

    def test_empty_change_is_a_no_op(self, tmp_path):
        app = self._app(tmp_path)
        on_traj_png_captured(app, {}, formula="H2", status=Mock(value=""))
        assert not list(tmp_path.glob("*.png"))

    def test_missing_formula_falls_back_to_a_generic_name(self, tmp_path):
        app = self._app(tmp_path)
        status = Mock(value="")
        on_traj_png_captured(app, self._change(_png_uri()), formula="", status=status)
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "molecule" in written[0].name
