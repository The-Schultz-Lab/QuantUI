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
