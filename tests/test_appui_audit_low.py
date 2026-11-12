"""App/UI fixes from the 2026-11 audit (Low findings).

- APP-14: the in-session comparison shows its chart and closes the figure.
"""

from __future__ import annotations

import pytest

from quantui.app import QuantUIApp
from quantui.session_calc import SessionResult

pytest.importorskip("matplotlib")


@pytest.fixture
def app():
    return QuantUIApp()


@pytest.fixture
def shown(monkeypatch):
    """Collect what app_runflow passes to ``display`` (no kernel in tests)."""
    items: list = []
    monkeypatch.setattr("quantui.app_runflow.display", lambda *a, **k: items.extend(a))
    return items


def _html(items) -> str:
    return "\n".join(str(getattr(o, "data", "")) for o in items)


def _sr(energy: float, method: str = "RHF") -> SessionResult:
    return SessionResult(
        energy_hartree=energy,
        homo_lumo_gap_ev=10.0,
        converged=True,
        n_iterations=5,
        method=method,
        basis="STO-3G",
        formula="H2",
    )


# ── APP-14 ────────────────────────────────────────────────────────────────


def test_in_session_comparison_shows_chart_and_closes_figure(app, shown):
    import matplotlib.pyplot as plt

    plt.close("all")
    app._results.extend([_sr(-1.11), _sr(-1.12, "B3LYP")])
    app._refresh_comparison()
    html = _html(shown)
    assert "Comparison chart" in html
    assert "data:image/png;base64," in html
    assert plt.get_fignums() == []


def test_saved_result_comparison_still_shows_chart(app, shown, monkeypatch):
    import matplotlib.pyplot as plt

    from quantui.comparison import summary_from_session_result

    plt.close("all")
    rows = {"/a": _sr(-1.11), "/b": _sr(-1.12, "B3LYP")}
    monkeypatch.setattr(
        "quantui.results_storage.load_result", lambda p: {"key": str(p)}
    )
    monkeypatch.setattr(
        "quantui.summary_from_saved_result",
        lambda d: summary_from_session_result(rows[d["key"]]),
    )
    app.compare_select.options = [("a", "/a"), ("b", "/b")]
    app.compare_select.value = ("/a", "/b")
    app._on_compare(None)
    assert "Comparison chart" in _html(shown)
    assert plt.get_fignums() == []
