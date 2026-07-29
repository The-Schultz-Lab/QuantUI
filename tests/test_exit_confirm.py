"""Tests for the two-stage Exit confirmation.

Background (2026-06-18 user report on NCShare): the Exit button shut down the
parent server on a single click. On a cluster interactive session that parent
process *is* the job, so one stray click tore down the whole allocation. Fix:
the first click only arms a confirmation (re-labels Exit to "Confirm shutdown"
and reveals a warning + Cancel); only a second click runs the shutdown.

Platform-independent (no PySCF): exercises the arm/cancel/confirm state machine
without ever invoking the real ``os._exit``.
"""

from __future__ import annotations

import pytest

from quantui import app_runflow


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Isolate settings (reflections/10 Rule 6) so construction never touches the
    # real ~/.quantui under pytest-xdist.
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from quantui.app import QuantUIApp

    return QuantUIApp()


class TestExitConfirm:
    def test_idle_state_by_default(self, app):
        assert getattr(app, "_exit_armed", False) is False
        assert app._exit_btn.description == "Exit"
        assert app._exit_cancel_btn.layout.display == "none"
        assert app._exit_warn_html.layout.display == "none"

    def test_first_click_arms_without_shutting_down(self, app, monkeypatch):
        called = {"perform": False}
        monkeypatch.setattr(
            app_runflow,
            "_perform_exit",
            lambda _app: called.__setitem__("perform", True),
        )

        app._on_exit_clicked()

        assert called["perform"] is False  # did NOT shut down
        assert app._exit_armed is True
        assert app._exit_btn.description == "Confirm shutdown"
        assert app._exit_cancel_btn.layout.display == ""
        assert app._exit_warn_html.layout.display == ""
        assert "ends your session" in app._exit_warn_html.value

    def test_cancel_disarms(self, app):
        app._on_exit_clicked()  # arm
        assert app._exit_armed is True

        app._on_exit_cancel()

        assert app._exit_armed is False
        assert app._exit_btn.description == "Exit"
        assert app._exit_cancel_btn.layout.display == "none"
        assert app._exit_warn_html.layout.display == "none"

    def test_second_click_performs_shutdown(self, app, monkeypatch):
        called = {"perform": False}
        monkeypatch.setattr(
            app_runflow,
            "_perform_exit",
            lambda _app: called.__setitem__("perform", True),
        )

        app._on_exit_clicked()  # first click → arm
        assert called["perform"] is False
        app._on_exit_clicked()  # second click → confirm

        assert called["perform"] is True

    def test_perform_exit_does_not_invoke_os_exit_synchronously(self, app, monkeypatch):
        # Neuter the background thread so the real os._exit never runs, then
        # confirm _perform_exit flips the UI into its shutting-down state.
        monkeypatch.setattr(
            app_runflow.threading, "Thread", lambda *a, **k: _NoThread()
        )

        app_runflow._perform_exit(app)

        assert app._exit_armed is False
        assert app._exit_btn.description == "Exiting…"
        assert app._exit_btn.disabled is True
        assert app._exit_cancel_btn.layout.display == "none"
        assert "shut down" in app._welcome_html.value


class _NoThread:
    """Stand-in for threading.Thread that never runs its target."""

    def start(self) -> None:  # pragma: no cover - trivial
        pass
