"""App/UI fixes from the 2026-11 audit (Low findings).

- APP-14: the in-session comparison shows its chart and closes the figure.
- APP-15: calc-type switching leaks no widgets; Cancel during error
  handling; TS/IRC badges and History chips; Files-tab default selection;
  platform-specific dependency hint.
- UI-10: a reloaded PES/IRC profile whose first point failed is plotted.
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


# ── APP-15 ────────────────────────────────────────────────────────────────


def _calc_types(app) -> list:
    return [o if isinstance(o, str) else o[1] for o in app.calc_type_dd.options]


def test_calc_type_switching_does_not_leak_widgets(app):
    import ipywidgets as widgets

    reg = widgets.widgets.widget._instances
    types = _calc_types(app)
    for ct in types:  # warm up lazily built persistent widgets
        app.calc_type_dd.value = ct
    app.calc_type_dd.value = types[0]
    before = len(reg)
    for _ in range(3):
        for ct in types:
            app.calc_type_dd.value = ct
    app.calc_type_dd.value = types[0]
    assert len(reg) - before == 0

    # Everything shown after a switch is still a live widget.
    def _alive(w):
        assert w.comm is not None, w
        assert w.layout.comm is not None, w
        for c in getattr(w, "children", None) or ():
            _alive(c)

    for ct in types:
        app.calc_type_dd.value = ct
        for child in app.calc_extra_opts.children:
            _alive(child)


def _result_html(app) -> str:
    return "\n".join(
        str(o.get("data", {}).get("text/html", "")) for o in app.result_output.outputs
    )


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ImportError("no pyscf")])
def test_cancel_clicked_as_the_calculation_fails_still_reports_it(app, exc):
    from unittest.mock import patch

    from quantui.molecule import Molecule

    pytest.importorskip("pyscf")
    app._set_molecule(Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]))
    app.method_dd.value = "RHF"
    app.basis_dd.value = "STO-3G"
    app.calc_type_dd.value = "Single Point"

    def _fail(*a, **k):
        app._on_cancel(None)  # click lands just before the error surfaces
        raise exc

    with patch("quantui.engines.run_calc", side_effect=_fail):
        app._begin_run_lock()
        app._run_in_thread()
    assert app.run_status.value in ("Calculation failed.", "Dependency unavailable.")
    html = _result_html(app)
    assert "Calculation Failed" in html or "Dependency Not Available" in html
    assert "Calculation cancelled" not in html


def test_ts_and_irc_have_badges_and_history_chips(app):
    from quantui.app_runflow import _calc_type_badge

    assert _calc_type_badge("transition_state") == "TS"
    assert _calc_type_badge("irc") == "IRC"
    assert {"transition_state", "irc"} <= set(app._history_calc_chips)


def test_files_tab_defaults_to_the_first_listed_entry(app, tmp_path, monkeypatch):
    names = [f"{c}.txt" for c in "abcdefghijklmnop"]
    for n in names:
        (tmp_path / n).write_text("x")
    monkeypatch.setattr(app, "_files_allowed_roots", lambda: [tmp_path])
    app._files_current_dir = tmp_path
    app._update_files_entries()
    assert app._files_entries.value == str(tmp_path / "a.txt")


def test_dependency_hint_is_platform_specific():
    from quantui.app import _dependency_hint

    assert "Apptainer" in _dependency_hint("win32")
    assert "WSL" in _dependency_hint("win32")
    linux = _dependency_hint("linux")
    assert "Windows" not in linux and "Apptainer" not in linux
    assert "pip install" in linux


# ── UI-10 ─────────────────────────────────────────────────────────────────


def _capture_pes(app, monkeypatch) -> list:
    shown_results: list = []
    monkeypatch.setattr(
        app,
        "_show_pes_scan_result",
        lambda r: shown_results.append(r) or True,
    )
    return shown_results


@pytest.mark.parametrize("failed", [float("nan"), None])
def test_reloaded_pes_scan_with_failed_first_point_is_plotted(app, monkeypatch, failed):
    import math
    from types import SimpleNamespace

    from quantui.app_analysis import pop_pes_plot

    got = _capture_pes(app, monkeypatch)
    ctx = SimpleNamespace(
        live_result=None,
        result_dir=None,
        spectra_data={
            "pes_scan": {
                "scan_type": "bond",
                "atom_indices": [0, 1],
                "scan_parameter_values": [0.6, 0.7, 0.8],
                "energies_hartree": [failed, -1.10, -1.11],
            }
        },
    )
    assert pop_pes_plot(app, ctx)
    rel = got[0].energies_relative_kcal
    assert math.isnan(rel[0])
    assert rel[2] == pytest.approx(0.0)
    assert rel[1] == pytest.approx(0.01 * 627.5094740631)


def test_reloaded_irc_with_failed_first_point_is_plotted(app, monkeypatch):
    import math
    from types import SimpleNamespace

    from quantui.app_analysis import pop_irc_profile

    got = _capture_pes(app, monkeypatch)
    ctx = SimpleNamespace(
        live_result=None,
        result_dir=None,
        spectra_data={
            "irc": {
                "energies_hartree": [float("nan"), -1.0, -1.02],
                "reaction_coordinate": [-1.0, 0.0, 1.0],
            }
        },
    )
    assert pop_irc_profile(app, ctx)
    rel = got[0].energies_relative_kcal
    assert math.isnan(rel[0])
    assert rel[2] == pytest.approx(0.0)
