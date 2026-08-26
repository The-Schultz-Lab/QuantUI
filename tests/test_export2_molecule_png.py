"""M-EXPORT2 EXP2.2: PNG capture wired into the molecule (top) viewer — the
fourth and final viewer, and the hardest one: unlike the isosurface, reorg,
trajectory, and vibrational viewers (all py3Dmol-only, one output slot
each), ``render_molecule_html`` routes through EITHER py3Dmol or plotlymol
depending on ``app._resolve_backend()``, and renders into up to THREE
independent output slots that can be visible at once (Calculate-tab
preview, Results-tab viewer, Analysis-tab viewer).

No browser, no PySCF — the JS is asserted as text (same approach as the
other EXP2.2 test files) and the Python handlers are exercised directly,
plus real ``QuantUIApp()`` integration tests that exercise all three output
slots together to prove there is no cross-talk between them (the risk this
viewer uniquely has: ``document.querySelector`` matches the FIRST element
with a given class anywhere on the page, so two simultaneously-visible
viewers sharing one capture class would let one's button post into the
other's inbox).
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import Mock

import pytest

from quantui.app_exports import (
    _PNG_URI_PREFIX,
    on_mol_analysis_png_captured,
    on_mol_calc_png_captured,
    on_mol_results_png_captured,
)
from quantui.molecule import Molecule
from quantui.visualization_py3dmol import PY3DMOL_AVAILABLE, render_molecule_html

pytestmark = pytest.mark.skipif(not PY3DMOL_AVAILABLE, reason="py3Dmol not installed")

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
        coordinates=[[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
    )


class TestRenderMoleculeHtmlCaptureWiring:
    def test_omits_the_button_when_capture_class_is_empty(self):
        html = render_molecule_html(_water(), backend="py3dmol")
        assert "orb_png_" not in html

    def test_wires_a_uid_scoped_capture_function_for_py3dmol(self):
        html = render_molecule_html(
            _water(), backend="py3dmol", capture_class="quantui-mol-calc-png-inbox"
        )
        assert "__quantuiMolCapture_" in html
        assert 'id="orb_png_' in html
        assert "quantui-mol-calc-png-inbox" in html
        # Never falls back to another viewer's capture function name.
        assert 'window["__quantuiIsoCapture"]' not in html
        assert "__quantuiReorgCapture_" not in html
        assert "__quantuiTrajCapture_" not in html
        assert "__quantuiVibCapture_" not in html

    def test_omits_the_button_for_plotlymol_even_with_a_capture_class(self):
        # The core reason this viewer needed its own scoping pass: the
        # capture bridge is 3Dmol.js-only, so a plotly-backend render must
        # never get a button that would silently do nothing when clicked.
        try:
            html = render_molecule_html(
                _water(),
                backend="plotlymol",
                capture_class="quantui-mol-calc-png-inbox",
            )
        except Exception:
            pytest.skip("plotlymol3d not installed in this env")
        assert "Save PNG" not in html
        assert "quantui-mol-calc-png-inbox" not in html

    def test_distinct_capture_classes_produce_distinct_buttons(self):
        """Each output slot must pass its OWN class — this is the test that
        would catch someone 'simplifying' the three call sites down to one
        shared class."""
        html_calc = render_molecule_html(
            _water(), backend="py3dmol", capture_class="quantui-mol-calc-png-inbox"
        )
        html_results = render_molecule_html(
            _water(), backend="py3dmol", capture_class="quantui-mol-results-png-inbox"
        )
        assert "quantui-mol-results-png-inbox" not in html_calc
        assert "quantui-mol-calc-png-inbox" not in html_results


class TestMolPngCaptureHandlers:
    """One test class per slot handler — thin wrappers around the shared
    ``_on_mol_png_captured`` implementation, but each reads a different
    molecule source and writes a different filename suffix, so each gets
    its own coverage rather than assuming the shared code makes them
    equivalent."""

    @staticmethod
    def _app(dest: Path, *, molecule: Molecule | None) -> Mock:
        app = Mock()
        app._last_result_dir = dest
        app._molecule = molecule
        app._analysis_displayed_molecule = molecule
        app.method_dd = Mock(value="B3LYP")
        app.basis_dd = Mock(value="6-31G*")
        app._mol_calc_png_status = Mock(value="")
        app._mol_calc_png_inbox = Mock(value="pending")
        app._mol_results_png_status = Mock(value="")
        app._mol_results_png_inbox = Mock(value="pending")
        app._mol_analysis_png_status = Mock(value="")
        app._mol_analysis_png_inbox = Mock(value="pending")
        return app

    def test_calc_capture_lands_on_disk_with_the_calc_suffix(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name == "H2O_calc.png"
        assert "Saved" in app._mol_calc_png_status.value
        assert app._mol_calc_png_inbox.value == ""

    def test_results_capture_lands_on_disk_with_the_results_suffix(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_results_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name == "H2O_results.png"
        assert "Saved" in app._mol_results_png_status.value
        assert app._mol_results_png_inbox.value == ""

    def test_analysis_capture_lands_on_disk_with_the_analysis_suffix(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_analysis_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name == "H2O_analysis.png"
        assert "Saved" in app._mol_analysis_png_status.value
        assert app._mol_analysis_png_inbox.value == ""

    def test_analysis_uses_the_displayed_molecule_not_the_active_one(self, tmp_path):
        """The Analysis tab can show a History replay that differs from
        whatever is loaded on the Calculate tab — the handler must read
        _analysis_displayed_molecule, not _molecule, or a capture would be
        mislabeled with the wrong formula after a replay."""
        app = self._app(tmp_path, molecule=None)
        app._molecule = Molecule(
            atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]]
        )
        app._analysis_displayed_molecule = _water()
        on_mol_analysis_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name == "H2O_analysis.png"  # water, not H2

    def test_three_slots_never_overwrite_each_other(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {"new": _png_uri()})
        on_mol_results_png_captured(app, {"new": _png_uri()})
        on_mol_analysis_png_captured(app, {"new": _png_uri()})
        written = sorted(p.name for p in tmp_path.glob("*.png"))
        assert written == ["H2O_analysis.png", "H2O_calc.png", "H2O_results.png"]

    def test_the_saved_png_carries_method_and_basis_metadata(self, tmp_path):
        from PIL import Image

        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))[0]
        with Image.open(written) as im:
            assert im.text["Method"] == "B3LYP"
            assert im.text["Basis"] == "6-31G*"

    def test_no_molecule_falls_back_to_a_generic_formula(self, tmp_path):
        app = self._app(tmp_path, molecule=None)
        on_mol_calc_png_captured(app, {"new": _png_uri()})
        written = list(tmp_path.glob("*.png"))
        assert len(written) == 1
        assert written[0].name == "molecule_calc.png"

    def test_malformed_prefix_is_reported_not_written(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {"new": "http://example.com/x.png"})
        assert not list(tmp_path.glob("*.png"))
        assert "unexpected image format" in app._mol_calc_png_status.value

    def test_oversized_payload_is_refused_before_decoding(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(
            app, {"new": _PNG_URI_PREFIX + "A" * (65 * 1024 * 1024)}
        )
        assert not list(tmp_path.glob("*.png"))
        assert "too large" in app._mol_calc_png_status.value

    def test_missing_result_dir_is_reported(self):
        app = self._app(Path("unused"), molecule=_water())
        app._last_result_dir = None
        on_mol_calc_png_captured(app, {"new": _png_uri()})
        assert "run a calculation" in app._mol_calc_png_status.value

    def test_empty_change_is_a_no_op(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {})
        assert not list(tmp_path.glob("*.png"))

    def test_inbox_is_cleared_on_failure_too(self, tmp_path):
        app = self._app(tmp_path, molecule=_water())
        on_mol_calc_png_captured(app, {"new": "garbage"})
        assert app._mol_calc_png_inbox.value == ""


class TestMolPngRealAppWiring:
    """End-to-end through a real QuantUIApp across all three render call
    sites (_refresh_calc_mol_viewer, show_result_3d's two branches, and by
    extension the widgets _rerender_3d_views also targets) — confirms the
    widgets app_builders actually built, the classes render_molecule_html
    actually emits, and the observers app.py actually wires all agree with
    each other, and that two simultaneously-visible viewers never cross-talk.
    """

    @staticmethod
    def _app_with_molecule() -> tuple:
        from quantui.app import QuantUIApp

        app = QuantUIApp()
        mol = _water()
        app._molecule = mol
        return app, mol

    def test_calc_tab_preview_gets_its_own_capture_button(self):
        app, mol = self._app_with_molecule()
        app._refresh_calc_mol_viewer()
        html = app.viz_output.outputs[0]["data"]["text/html"]
        assert "quantui-mol-calc-png-inbox" in html
        assert "Save PNG" in html

    def test_results_and_analysis_render_together_without_cross_talk(self):
        app, mol = self._app_with_molecule()
        app._show_result_3d(mol, extra_output=app._analysis_mol_output)
        results_html = app.result_viz_output.outputs[0]["data"]["text/html"]
        analysis_html = app._analysis_mol_output.outputs[0]["data"]["text/html"]
        assert "quantui-mol-results-png-inbox" in results_html
        assert "quantui-mol-analysis-png-inbox" in analysis_html
        assert "quantui-mol-calc-png-inbox" not in results_html
        assert "quantui-mol-calc-png-inbox" not in analysis_html
        assert "quantui-mol-results-png-inbox" not in analysis_html
        assert "quantui-mol-analysis-png-inbox" not in results_html

    def test_analysis_viewer_keeps_click_to_measure_alongside_capture(self):
        """finalize_analysis_html only appends its own <script>, so the
        capture button baked in earlier must survive — regression guard for
        the interaction this viewer uniquely has among the four."""
        app, mol = self._app_with_molecule()
        app._show_result_3d(mol, extra_output=app._analysis_mol_output)
        analysis_html = app._analysis_mol_output.outputs[0]["data"]["text/html"]
        assert "__quantuiMolCapture_" in analysis_html
        assert "setClickable" in analysis_html  # click-to-measure wiring

    def test_writing_to_each_inbox_triggers_an_independent_real_save(self, tmp_path):
        app, mol = self._app_with_molecule()
        app._last_result_dir = tmp_path
        app.method_dd.value = "B3LYP"
        app.basis_dd.value = "6-31G*"

        app._refresh_calc_mol_viewer()
        app._show_result_3d(mol, extra_output=app._analysis_mol_output)
        app._analysis_displayed_molecule = mol

        uri = _png_uri()
        app._mol_calc_png_inbox.value = uri
        app._mol_results_png_inbox.value = uri
        app._mol_analysis_png_inbox.value = uri

        written = sorted(p.name for p in tmp_path.glob("*.png"))
        assert written == ["H2O_analysis.png", "H2O_calc.png", "H2O_results.png"]
        assert app._mol_calc_png_inbox.value == ""
        assert app._mol_results_png_inbox.value == ""
        assert app._mol_analysis_png_inbox.value == ""
