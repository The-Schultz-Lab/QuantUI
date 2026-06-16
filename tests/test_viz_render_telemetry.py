"""Unit tests for VIZBACK.6 lifecycle telemetry events.

Covers the ``_viz_render_event`` context manager and its integration with
the vib render dispatcher (``render_vib_mode``), the sync cache hit path
(``_try_vib_cache_hit_sync``), and the trajectory frame builder
(``_build_fig``). Verifies ``viz_render_start`` / ``viz_render_done`` /
``viz_render_error`` events fire with the required fields
(``task``, ``pref``, ``backend``, ``elapsed_ms``).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from quantui import calc_log
from quantui.app import QuantUIApp
from quantui.app_visualization import (
    _try_vib_cache_hit_sync,
    _viz_render_event,
    render_vib_mode,
)
from quantui.molecule import Molecule


@pytest.fixture
def app():
    return QuantUIApp()


@pytest.fixture
def water_mol():
    return Molecule(
        atoms=["O", "H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


@pytest.fixture
def fake_freq_result():
    return SimpleNamespace(
        frequencies_cm1=[1600.0, 3800.0],
        ir_intensities=[40.0, 50.0],
        displacements=[
            np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0], [0.0, -0.05, 0.0]]),
            np.array([[0.0, 0.10, 0.0], [0.05, -0.10, 0.0], [-0.05, -0.10, 0.0]]),
        ],
    )


@pytest.fixture
def captured_events(monkeypatch):
    """Replaces ``calc_log.log_event`` with a recording stub. Returns the
    list that gets appended to by each call."""
    events: list[tuple[str, str]] = []

    def _record(event_type, message, **extra):
        events.append((event_type, message))

    monkeypatch.setattr(calc_log, "log_event", _record)
    return events


def _events_of(events, kind):
    return [(t, m) for t, m in events if t == kind]


class TestViewRenderEventContextManager:
    def test_emits_start_and_done(self, app, captured_events):
        with _viz_render_event(app, task="t1", backend="b1"):
            pass
        starts = _events_of(captured_events, "viz_render_start")
        dones = _events_of(captured_events, "viz_render_done")
        assert len(starts) == 1
        assert len(dones) == 1
        assert "task=t1" in starts[0][1]
        assert "backend=b1" in starts[0][1]
        assert "elapsed_ms=" in dones[0][1]

    def test_emits_error_on_exception(self, app, captured_events):
        with pytest.raises(RuntimeError):
            with _viz_render_event(app, task="t1", backend="b1"):
                raise RuntimeError("boom")
        errs = _events_of(captured_events, "viz_render_error")
        dones = _events_of(captured_events, "viz_render_done")
        assert len(errs) == 1
        assert len(dones) == 0
        assert "elapsed_ms=" in errs[0][1]
        assert "RuntimeError" in errs[0][1]

    def test_extras_appended_to_message(self, app, captured_events):
        with _viz_render_event(app, task="t1", backend="b1", mode=3, source="x"):
            pass
        starts = _events_of(captured_events, "viz_render_start")
        assert "mode=3" in starts[0][1]
        assert "source=x" in starts[0][1]

    def test_preference_included(self, app, captured_events):
        app._viz_backend_preference = "py3dmol"
        with _viz_render_event(app, task="t1", backend="b1"):
            pass
        starts = _events_of(captured_events, "viz_render_start")
        assert "pref=py3dmol" in starts[0][1]


class TestVibRenderTelemetry:
    def test_render_vib_mode_emits_lifecycle(
        self, app, water_mol, fake_freq_result, captured_events
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol

        render_vib_mode(app, vib_data=None, molecule=water_mol, mode_number=1)

        # render_vib_mode dispatches the render to a daemon thread
        # (reflections/02 — render off the main thread). Wait for the terminal
        # event before asserting; otherwise the test races the worker and fails
        # intermittently under heavy parallel load (pytest -n=auto).
        deadline = time.time() + 15.0
        while time.time() < deadline and not (
            _events_of(captured_events, "viz_render_done")
            or _events_of(captured_events, "viz_render_error")
        ):
            time.sleep(0.02)

        starts = _events_of(captured_events, "viz_render_start")
        dones = _events_of(captured_events, "viz_render_done")
        assert any("task=vib_interactive" in m for _, m in starts)
        assert any("backend=py3dmol" in m for _, m in starts)
        assert any("mode=1" in m for _, m in starts)
        assert any("elapsed_ms=" in m for _, m in dones)


class TestSyncCacheHitTelemetry:
    def test_sync_cache_hit_emits_lifecycle_with_source(
        self, app, water_mol, fake_freq_result, captured_events, tmp_path
    ):
        if not app._viz_availability.py3dmol:
            pytest.skip("py3Dmol not installed in test env")
        app._last_vib_freq_result = fake_freq_result
        app._last_vib_molecule = water_mol
        app._last_result_dir = tmp_path

        # First call populates the cache (no cache hit yet).
        from quantui.app_visualization import _render_vib_mode_py3dmol

        _render_vib_mode_py3dmol(app, water_mol, mode_number=1)

        # Clear events from the populate call.
        captured_events.clear()

        # Second call: synchronous cache hit.
        hit = _try_vib_cache_hit_sync(app, mode_number=1)
        assert hit is True

        starts = _events_of(captured_events, "viz_render_start")
        dones = _events_of(captured_events, "viz_render_done")
        cache_hits = _events_of(captured_events, "vib_cache_hit")
        assert any("source=cache_sync" in m for _, m in starts)
        assert any("source=cache_sync" in m for _, m in dones)
        assert any("path=sync" in m for _, m in cache_hits)
