"""M-EXPORT2: destination helper (EXP2.3), reorg XYZ export (EXP2.1), and the
generalized PNG capture bridge wired into the reorg-geometry viewer (EXP2.2).

No browser, no PySCF — the JS is asserted as text (same approach as
``test_orbital_export_and_resolution.py``) and the Python halves are
exercised directly.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import Mock

import pytest

from quantui.app_exports import (
    _PNG_URI_PREFIX,
    export_destination,
    on_export_reorg_geometries,
    on_reorg_png_captured,
)
from quantui.app_visualization import build_reorg_geometry_viewer_html
from quantui.molecule import Molecule
from quantui.orbital_visualization import _png_capture_controls
from quantui.reorganization_energy import reorg_geometries

# Same 1x1 PNG used by the isosurface capture tests — real bytes exercise the
# actual decode path rather than a base64 round-trip of arbitrary data.
_REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_uri(raw: bytes = _REAL_PNG) -> str:
    return _PNG_URI_PREFIX + base64.b64encode(raw).decode()


class TestExportDestination:
    def test_joins_sanitized_parts_under_the_result_dir(self, tmp_path):
        app = Mock()
        app._last_result_dir = tmp_path
        dest = export_destination(app, "widget", "H2O", "R_neutral", suffix=".xyz")
        assert dest.parent == tmp_path
        assert dest.name == "H2O_R_neutral.xyz"

    def test_hostile_name_parts_cannot_escape_the_result_dir(self, tmp_path):
        app = Mock()
        app._last_result_dir = tmp_path
        dest = export_destination(app, "widget", "../../etc/passwd", suffix=".txt")
        assert dest.parent == tmp_path

    def test_missing_result_dir_raises_a_readable_error(self):
        app = Mock()
        app._last_result_dir = None
        with pytest.raises(ValueError, match="widget"):
            export_destination(app, "widget", "x", suffix=".txt")

    def test_empty_name_parts_are_skipped(self, tmp_path):
        app = Mock()
        app._last_result_dir = tmp_path
        dest = export_destination(app, "widget", "H2O", "", "B3LYP", suffix=".xyz")
        assert dest.name == "H2O_B3LYP.xyz"


class TestReorgGeometryXyzExport:
    @staticmethod
    def _app(dest: Path, geoms: list) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app._reorg_geometries = geoms
        app._reorg_export_status = Mock(value="")
        app.method_dd = Mock(value="B3LYP")
        app.basis_dd = Mock(value="6-31G*")
        return app

    @staticmethod
    def _geoms() -> list:
        neutral = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
            charge=0,
            multiplicity=1,
        )
        hole = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.97, 0, 0], [-0.25, 0.94, 0]],
            charge=1,
            multiplicity=2,
        )
        return reorg_geometries(
            neutral_geometry={
                "atoms": neutral.atoms,
                "coordinates": neutral.coordinates,
                "charge": neutral.charge,
                "multiplicity": neutral.multiplicity,
            },
            channels=[
                {
                    "kind": "hole",
                    "ion_charge": 1,
                    "ion_geometry": {
                        "atoms": hole.atoms,
                        "coordinates": hole.coordinates,
                        "charge": hole.charge,
                        "multiplicity": hole.multiplicity,
                    },
                }
            ],
        )

    def test_writes_one_xyz_per_distinct_geometry(self, tmp_path):
        geoms = self._geoms()
        app = self._app(tmp_path, geoms)
        on_export_reorg_geometries(app, Mock())
        written = sorted(p.name for p in tmp_path.glob("*.xyz"))
        assert len(written) == len(geoms)
        assert "Saved" in app._reorg_export_status.value

    def test_each_file_carries_its_own_charge_and_multiplicity(self, tmp_path):
        geoms = self._geoms()
        app = self._app(tmp_path, geoms)
        on_export_reorg_geometries(app, Mock())
        ion_file = next(tmp_path.glob("*R_hole*.xyz"))
        text = ion_file.read_text()
        assert "charge=1" in text
        assert "multiplicity=2" in text

    def test_no_geometries_yet_is_reported_not_a_crash(self, tmp_path):
        app = self._app(tmp_path, [])
        on_export_reorg_geometries(app, Mock())
        assert not list(tmp_path.glob("*.xyz"))
        assert "No geometries" in app._reorg_export_status.value

    def test_missing_result_dir_is_reported(self):
        app = self._app(Path("/nonexistent"), self._geoms())
        app._last_result_dir = None
        on_export_reorg_geometries(app, Mock())
        assert "run a calculation" in app._reorg_export_status.value


class TestReorgPngCaptureBridge:
    @staticmethod
    def _app(dest: Path) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app._molecule = Molecule(
            atoms=["O", "H", "H"],
            coordinates=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        )
        app.method_dd = Mock(value="B3LYP")
        app.basis_dd = Mock(value="6-31G*")
        app._reorg_export_status = Mock(value="")
        app._reorg_png_inbox = Mock(value="pending")
        return app

    def test_a_capture_lands_on_disk(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "Saved" in app._reorg_export_status.value

    def test_the_saved_png_carries_method_and_basis_metadata(self, tmp_path):
        # M-EXPORT2 EXP2.4: metadata only, no DPI stamp — this exporter
        # deliberately has no DPI control of its own (see _with_dpi).
        from PIL import Image

        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))[0]
        with Image.open(written) as im:
            assert im.text["Method"] == "B3LYP"
            assert im.text["Basis"] == "6-31G*"

    def test_inbox_is_cleared_after_a_capture(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": _png_uri()})
        assert app._reorg_png_inbox.value == ""

    def test_inbox_is_cleared_on_failure_too(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": "garbage"})
        assert app._reorg_png_inbox.value == ""

    def test_malformed_prefix_is_reported_not_written(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": "http://example.com/x.png"})
        assert not list(tmp_path.glob("*.png"))
        assert "unexpected image format" in app._reorg_export_status.value

    def test_oversized_payload_is_refused_before_decoding(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {"new": _PNG_URI_PREFIX + "A" * (65 * 1024 * 1024)})
        assert not list(tmp_path.glob("*.png"))
        assert "too large" in app._reorg_export_status.value

    def test_missing_result_dir_is_reported(self):
        app = self._app(Path("unused"))
        app._last_result_dir = None
        on_reorg_png_captured(app, {"new": _png_uri()})
        assert "run a calculation" in app._reorg_export_status.value

    def test_empty_change_is_a_no_op(self, tmp_path):
        app = self._app(tmp_path)
        on_reorg_png_captured(app, {})
        assert not list(tmp_path.glob("*.png"))

    def test_geometry_slug_from_stepper_is_used_in_filename(self, tmp_path):
        app = self._app(tmp_path)
        uri = _png_uri() + "\nR_neutral"
        on_reorg_png_captured(app, {"new": uri})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert "R_neutral" in written[0].name


class TestReorgViewerCaptureWiring:
    """The generalized capture bridge (EXP2.2) — isosurface behaviour must be
    byte-identical (backward compatible default), and the reorg viewer must
    get its own uid-scoped capture function rather than sharing the
    isosurface's bare global.
    """

    @staticmethod
    def _geoms() -> list:
        return [
            {
                "label": "R_neutral",
                "atoms": ["O", "H"],
                "coordinates": [[0, 0, 0], [0.96, 0, 0]],
                "note": "",
            },
            {
                "label": "R_hole",
                "atoms": ["O", "H"],
                "coordinates": [[0, 0, 0], [0.97, 0, 0]],
                "note": "",
            },
        ]

    def test_isosurface_default_capture_fn_is_unchanged(self):
        html = _png_capture_controls("abc123", "quantui-orb-png-inbox")
        assert 'window["__quantuiIsoCapture"]' in html

    def test_reorg_viewer_omits_the_button_when_capture_class_is_empty(self):
        html = build_reorg_geometry_viewer_html(self._geoms(), capture_class="")
        assert "orb_png_" not in html

    def test_reorg_viewer_wires_a_uid_scoped_capture_function(self):
        html = build_reorg_geometry_viewer_html(
            self._geoms(), capture_class="quantui-reorg-png-inbox"
        )
        assert "__quantuiReorgCapture_" in html
        assert 'id="orb_png_' in html
        assert "quantui-reorg-png-inbox" in html
        # Never falls back to the isosurface's bare global.
        assert 'window["__quantuiIsoCapture"]' not in html

    def test_reorg_capture_button_still_appears_with_a_single_geometry(self):
        html = build_reorg_geometry_viewer_html(
            self._geoms()[:1], capture_class="quantui-reorg-png-inbox"
        )
        assert 'id="orb_png_' in html

    def test_no_geometries_returns_placeholder_not_a_crash(self):
        html = build_reorg_geometry_viewer_html(
            [], capture_class="quantui-reorg-png-inbox"
        )
        assert "No geometries" in html
