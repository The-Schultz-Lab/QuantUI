"""Regression tests for the 2026-11 audit's UI display / escaping findings.

Covers UI-2 (theme CSS), UI-3/APP-9 (History log panel arguments),
UI-4 (result.json strings escaped in formatters), APP-12/DATA-9
(unescaped strings into ``widgets.HTML``), UI-6 (per-viewer highlight
hooks), UI-7 (dark-palette contrast) and UI-8 (History card labels).
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

from quantui.app import QuantUIApp

_XSS = "<img src=x onerror=alert(1)>"


def _make_sp_result_dir(tmp_path, **overrides):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + "000001"
    d = tmp_path / f"{ts}_H2O_RHF_STO-3G"
    d.mkdir()
    data = {
        "_schema_version": 2,
        "timestamp": ts,
        "calc_type": "single_point",
        "formula": "H2O",
        "method": "RHF",
        "basis": "STO-3G",
        "energy_hartree": -75.0,
        "energy_ev": -2041.0,
        "homo_lumo_gap_ev": 8.0,
        "converged": True,
        "n_iterations": 10,
    }
    data.update(overrides)
    (d / "result.json").write_text(json.dumps(data))
    return d


# ── UI-3 / APP-9 ──────────────────────────────────────────────────────────────


class TestHistoryLogPanelArgs:
    def test_view_analysis_puts_log_text_in_panel_and_name_in_label(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        result_dir = _make_sp_result_dir(tmp_path)
        (result_dir / "pyscf.log").write_text(
            "converged SCF energy = -75.0\n" + _XSS + "\n", encoding="utf-8"
        )
        app = QuantUIApp()
        with patch.object(app, "_activity_pulse"):
            app._history_load_analysis(result_dir)
        panel = app._log_output_html.value
        label = app._log_source_lbl.value
        assert "converged SCF energy" in panel
        assert result_dir.name in label
        assert "converged SCF energy" not in label
        assert _XSS not in panel

    def test_render_log_escapes_source_label(self):
        app = QuantUIApp()
        app._render_log("hello", _XSS)
        assert _XSS not in app._log_source_lbl.value
        assert "&lt;img" in app._log_source_lbl.value


# ── UI-2 ──────────────────────────────────────────────────────────────────────


class TestWidgetChromeCss:
    def test_no_doubled_closing_braces(self):
        from quantui import theme

        css = theme._widget_chrome_css()
        assert "}}" not in css
        assert css.count("{") == css.count("}")

    def test_every_rule_parses(self):
        import pytest

        tinycss2 = pytest.importorskip("tinycss2")
        from quantui import theme

        rules = tinycss2.parse_stylesheet(
            theme._widget_chrome_css(), skip_whitespace=True, skip_comments=True
        )
        errors = [r for r in rules if r.type == "error"]
        assert errors == []
        preludes = [tinycss2.serialize(r.prelude) for r in rules]
        assert all("}" not in p for p in preludes), preludes
        assert any(".lm-TabBar-tab, .p-TabBar-tab" in p for p in preludes)
        assert any(".quantui-info-box small" in p for p in preludes)


# ── UI-4 ──────────────────────────────────────────────────────────────────────


class TestPastResultEscaping:
    def _data(self, **overrides):
        data = {
            "calc_type": "single_point",
            "formula": "H2O",
            "method": "RHF",
            "basis": "STO-3G",
            "timestamp": "2026-11-10_10-00-00-000001",
            "energy_hartree": -75.0,
            "converged": True,
            "n_iterations": 5,
        }
        data.update(overrides)
        return data

    def test_free_text_fields_are_escaped(self):
        from quantui.app_formatters import format_past_result

        for key in ("formula", "method", "basis", "timestamp", "solvent"):
            html_out = format_past_result(self._data(**{key: _XSS}))
            assert _XSS not in html_out, key
            assert "&lt;img" in html_out, key

    def test_gpu_name_scf_variant_and_calc_type_are_escaped(self):
        from quantui.app_formatters import format_past_result

        cases = [
            {"gpu_used": True, "gpu_name": _XSS},
            {"scf_variant": _XSS},
            {"calc_type": _XSS},
            {"mulliken_charges": [0.1], "atom_symbols": [_XSS]},
        ]
        for extra in cases:
            html_out = format_past_result(self._data(**extra))
            assert _XSS not in html_out, extra

    def test_reorg_channels_and_comparison_are_escaped(self):
        from quantui.app_formatters import (
            reorg_channels_html,
            reorg_comparison_html,
        )

        ch = {"kind": _XSS, "lambda_hartree": 0.01, "ion_multiplicity": 2}
        assert _XSS not in reorg_channels_html([ch])
        out = reorg_comparison_html(
            [(_XSS, {"method": _XSS, "basis": "b", "reorg_channels": [ch]})]
        )
        assert _XSS not in out

    def test_nmr_reference_strings_are_escaped(self):
        from types import SimpleNamespace

        from quantui.app_formatters import format_nmr_result

        r = SimpleNamespace(
            converged=True,
            reference_compound=_XSS,
            method="B3LYP",
            basis="6-31G*",
            formula="CH4",
            is_fallback_reference=True,
            reference_key=_XSS,
            h_shifts=lambda: [],
            c_shifts=lambda: [],
        )
        assert _XSS not in format_nmr_result(r)

    def test_plain_values_render_unchanged(self):
        from quantui.app_formatters import format_past_result

        out = format_past_result(self._data(basis="6-31G*", solvent="water"))
        assert "RHF/6-31G*" in out
        assert "water" in out


# ── APP-12 / DATA-9 ───────────────────────────────────────────────────────────


def _h2():
    from quantui.molecule import Molecule

    return Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


class TestWidgetHtmlEscaping:
    def test_molecule_label_is_escaped(self):
        app = QuantUIApp()
        app._set_molecule(_h2(), _XSS)
        assert _XSS not in app.mol_info_html.value
        assert _XSS not in app.mol_summary_compact.value
        assert "&lt;img" in app.mol_info_html.value

    def test_event_log_panel_is_escaped(self):
        app = QuantUIApp()
        events = [{"timestamp": "2026-11-10T10:00:00", "event": _XSS, "message": _XSS}]
        with patch("quantui.calc_log.get_recent_events", return_value=events):
            out = app._build_events_html()
        assert _XSS not in out
        assert "&lt;img" in out

    def test_upload_error_text_is_escaped(self):
        from types import SimpleNamespace

        from quantui.app_xyz_input import on_structure_upload

        bad = f"2\ncomment\nH 0 0 0\nH 0 0 {_XSS}\n".encode()
        app = SimpleNamespace(
            structure_upload=SimpleNamespace(
                value=({"name": "bad.xyz", "content": bad},)
            ),
            upload_msg=SimpleNamespace(value=""),
            charge_si=SimpleNamespace(value=0),
            mult_si=SimpleNamespace(value=1),
        )
        on_structure_upload(app)
        assert "⚠" in app.upload_msg.value
        assert "<img" not in app.upload_msg.value

    def test_files_tab_structure_preview_cannot_close_its_script(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("QUANTUI_RESULTS_DIR", str(tmp_path))
        app = QuantUIApp()
        benign = tmp_path / "ok.xyz"
        benign.write_text("2\nplain\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        evil = tmp_path / "evil.xyz"
        evil.write_text(
            f"2\n</script>{_XSS}<script>\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8"
        )
        shown = []

        def _capture(obj, *a, **k):
            shown.append(getattr(obj, "data", obj))

        monkeypatch.setattr("quantui.app.display", _capture)
        app._preview_file_path(benign)
        app._preview_file_path(evil)
        assert len(shown) == 2
        ok_html, evil_html = shown
        assert evil_html.lower().count("</script") == ok_html.lower().count("</script")
        assert _XSS not in evil_html
        assert "3D structure preview" in app._files_status_html.value


# ── UI-8 ──────────────────────────────────────────────────────────────────────


class TestHistoryCardLabels:
    def _card(self, **data):
        from quantui.app_formatters import format_past_result

        return format_past_result(data)

    def test_unconverged_geometry_opt_is_not_an_scf_failure(self):
        out = self._card(
            calc_type="geometry_opt",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-74.9,
            converged=False,
        )
        assert "SCF converged" not in out
        assert "Opt converged" in out
        assert "max steps reached" in out
        assert "Final energy" in out

    def test_pes_card_labels_minimum_and_handles_all_failed(self):
        out = self._card(
            calc_type="pes_scan",
            formula="H2",
            method="RHF",
            basis="STO-3G",
            energy_hartree=float("nan"),
            converged=False,
        )
        assert "Total energy" not in out
        assert "Min energy" in out
        assert "All converged" in out
        assert "nan" not in out.lower()

    def test_old_schema_without_energy_or_formula_renders(self):
        out = self._card(calc_type="single_point", method="RHF", basis="STO-3G")
        assert "Total energy" in out
        assert "—" in out

    def test_single_point_label_unchanged(self):
        out = self._card(
            calc_type="single_point",
            formula="H2O",
            method="RHF",
            basis="STO-3G",
            energy_hartree=-75.0,
            converged=True,
        )
        assert "SCF converged" in out
        assert "-75.00000000 Ha" in out


# ── UI-6 ──────────────────────────────────────────────────────────────────────


def _injected_script(html: str) -> str:
    start = html.rindex("<script>") + len("<script>")
    return html[start : html.rindex("</script>")]


class TestHighlightHooksArePerSlot:
    def _scripts(self):
        from quantui.app_measurement import MEASURE_INBOX_CLASS, inject_click_js
        from quantui.app_structure_edit import EDIT_PICK_INBOX_CLASS

        measure = inject_click_js(
            '<div id="3dmolviewer_AAA"></div>', inbox_class=MEASURE_INBOX_CLASS
        )
        edit = inject_click_js(
            '<div id="3dmolviewer_BBB"></div>', inbox_class=EDIT_PICK_INBOX_CLASS
        )
        return _injected_script(measure), _injected_script(edit)

    def _pushed_js(self, slot):
        from unittest.mock import MagicMock

        from quantui import app_measurement

        app = MagicMock()
        sent = []
        with patch("IPython.display.display", side_effect=sent.append):
            app_measurement.push_highlight(app, [0, 1], slot=slot)
        assert len(sent) == 1
        return sent[0].data

    def test_slots_register_distinct_hooks_and_no_shared_global(self):
        from quantui.app_measurement import MEASURE_INBOX_CLASS, highlight_hook_name
        from quantui.app_pes_pick import PES_PICK_INBOX_CLASS
        from quantui.app_structure_edit import EDIT_PICK_INBOX_CLASS

        names = {
            highlight_hook_name(s)
            for s in (MEASURE_INBOX_CLASS, EDIT_PICK_INBOX_CLASS, PES_PICK_INBOX_CLASS)
        }
        assert len(names) == 3
        measure_js, edit_js = self._scripts()
        assert highlight_hook_name(MEASURE_INBOX_CLASS) in measure_js
        assert highlight_hook_name(EDIT_PICK_INBOX_CLASS) in edit_js
        for js in (measure_js, edit_js):
            assert "window.__quantuiMeasureHighlight =" not in js
            assert "window.__quantuiMeasureHighlight=" not in js

    def test_push_targets_its_own_slot(self):
        from quantui.app_measurement import MEASURE_INBOX_CLASS, highlight_hook_name
        from quantui.app_structure_edit import EDIT_PICK_INBOX_CLASS

        js = self._pushed_js(EDIT_PICK_INBOX_CLASS)
        assert highlight_hook_name(EDIT_PICK_INBOX_CLASS) in js
        assert highlight_hook_name(MEASURE_INBOX_CLASS) not in js

    def test_structure_edit_and_pes_pushes_use_their_slots(self):
        from unittest.mock import MagicMock

        from quantui import app_pes_pick, app_structure_edit

        for mod, fn, slot in (
            (
                app_structure_edit,
                "_push_edit_highlight",
                app_structure_edit.EDIT_PICK_INBOX_CLASS,
            ),
            (app_pes_pick, "push_scan_highlight", app_pes_pick.PES_PICK_INBOX_CLASS),
        ):
            with patch.object(mod, "push_highlight") as ph:
                getattr(mod, fn)(MagicMock(), [0])
            assert ph.call_args.kwargs["slot"] == slot

    def test_highlight_reaches_the_slot_viewer_in_js(self, tmp_path):
        """Run both injected scripts, then an Edit push, under node."""
        import shutil
        import subprocess

        import pytest

        node = shutil.which("node")
        if node is None:
            pytest.skip("node not available")
        from quantui.app_structure_edit import EDIT_PICK_INBOX_CLASS

        measure_js, edit_js = self._scripts()
        push_js = self._pushed_js(EDIT_PICK_INBOX_CLASS)
        harness = """
var window = globalThis; var calls = [];
globalThis.document = {querySelector: function(){ return null; }};
function fakeViewer(name){
  return {
    setClickable: function(){},
    getModel: function(){ return {selectedAtoms: function(){
      return [{x:0,y:0,z:0,elem:"H"}]; }}; },
    addSphere: function(){ calls.push(name); return {}; },
    addLine: function(){ return {}; },
    removeShape: function(){}, render: function(){}
  };
}
window.viewer_AAA = fakeViewer("measure");
window.viewer_BBB = fakeViewer("edit");
"""
        script = (
            harness
            + measure_js
            + "\n"
            + edit_js
            + "\n"
            + measure_js  # the Measure viewer re-renders last
            + "\n"
            + push_js
            + "\nsetTimeout(function(){ console.log(JSON.stringify(calls)); }, 10);"
        )
        path = tmp_path / "harness.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run(
            [node, str(path)], capture_output=True, text=True, timeout=30
        )
        assert out.returncode == 0, out.stderr
        calls = json.loads(out.stdout.strip().splitlines()[-1])
        assert calls and set(calls) == {"edit"}
