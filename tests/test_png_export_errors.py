"""PNG export must report the real Kaleido/Chrome error (audit INFRA-6).

``QuantUIApp._export_plot_figure`` rewrote any error mentioning "kaleido" to
"PNG export requires kaleido. Install with: pip install kaleido". On a fresh
install kaleido 1.x *is* installed; what is missing is Chrome, and plotly's
own message says so ("Kaleido requires Google Chrome ... plotly_get_chrome").
The rewrite sent users to reinstall a package they already had.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quantui.app import QuantUIApp

# plotly.io._kaleido.PLOTLY_GET_CHROME_ERROR_MSG (plotly 6/7), raised as a
# RuntimeError by fig.write_image when kaleido >= 1 finds no browser.
_CHROME_MISSING = (
    "\n\nKaleido requires Google Chrome to be installed.\n\n"
    "Either download and install Chrome yourself following Google's "
    "instructions for your operating system,\nor install it from your "
    "terminal by running:\n\n    $ plotly_get_chrome\n\n"
)


@pytest.fixture
def app(tmp_path):
    a = QuantUIApp()
    a._last_result_dir = tmp_path
    return a


def _export_png(app, exc):
    fig = MagicMock()
    fig.write_image.side_effect = exc
    app._export_plot_figure(
        fig=fig, stem="ir_spectrum", fmt="png", status_widget=app._ir_export_status
    )
    return app._ir_export_status.value


def test_missing_chrome_error_is_passed_through(app):
    shown = _export_png(app, RuntimeError(_CHROME_MISSING))
    assert "Export failed" in shown
    assert "Kaleido requires Google Chrome to be installed." in shown
    assert "plotly_get_chrome" in shown
    assert "pip install kaleido" not in shown


def test_error_text_is_html_escaped(app):
    shown = _export_png(app, RuntimeError("bad <path> & kaleido"))
    assert "bad &lt;path&gt; &amp; kaleido" in shown
    assert "<path>" not in shown


def test_successful_png_export_still_reports_saved(app, tmp_path):
    fig = MagicMock()
    fig.write_image.side_effect = lambda path, scale: open(path, "wb").close()
    app._export_plot_figure(
        fig=fig, stem="ir_spectrum", fmt="png", status_widget=app._ir_export_status
    )
    assert "Saved:" in app._ir_export_status.value
    assert list(tmp_path.glob("ir_spectrum_*.png"))
