"""Startup-speed deferrals (perceived-startup optimization).

Construction used to block ~15 s on (1) GPU detection (imports gpu4pyscf/cupy +
CUDA query) and (2) loading every saved result for the History and Compare
dropdowns. Both are now deferred off the synchronous construction path so the UI
paints fast: GPU detection runs on a daemon thread and re-renders the Status
badge; History/Compare population is scheduled on the kernel io loop.

Platform-independent (no PySCF / GPU). ``QUANTUI_DISABLE_GPU=1`` keeps the
background GPU thread from importing gpu4pyscf during the test.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUI_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("QUANTUI_DISABLE_GPU", "1")
    from quantui.app import QuantUIApp

    captured: list = []

    class _FakeLoop:
        def add_callback(self, cb, *a, **k):
            captured.append((cb, a, k))

    # A fake kernel loop captures deferred callbacks instead of running them —
    # exactly how Voilà defers them until after the cell returns / first paint.
    monkeypatch.setattr(QuantUIApp, "_get_kernel_io_loop", lambda self: _FakeLoop())
    instance = QuantUIApp()
    instance._captured_callbacks = captured
    return instance


class TestStartupDeferral:
    def test_history_dropdown_shows_loading_placeholder(self, app):
        # The real load (every saved result) is deferred, so construction leaves
        # a placeholder rather than the populated list.
        labels = [lbl for lbl, _val in app.past_dd.options]
        assert any("loading" in lbl.lower() for lbl in labels)

    def test_compare_dropdown_shows_loading_placeholder(self, app):
        labels = [lbl for lbl, _val in app.compare_select.options]
        assert any("loading" in lbl.lower() for lbl in labels)

    def test_history_and_compare_scheduled_on_loop(self, app):
        # At least the History + Compare population were scheduled (the GPU
        # apply may add a third once the daemon thread resolves).
        assert len(app._captured_callbacks) >= 2

    def test_status_badge_renders_each_gpu_state(self, app):
        # The Status panel is rendered via a closure so the background GPU
        # detector can refresh just the GPU row.
        assert "checking" in app._render_status_html(None).lower()
        assert "MyGPU" in app._render_status_html((True, "MyGPU"))
        assert "not installed" in app._render_status_html((False, None)).lower()
