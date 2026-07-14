"""Tests for quantui.calc_log estimation behavior."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def isolated_log_dir(tmp_path, monkeypatch):
    """Point QUANTUI_LOG_DIR at a fresh temp directory for every test."""
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    import quantui.calc_log as clog

    importlib.reload(clog)
    yield tmp_path


def test_estimate_time_scopes_by_calc_type(isolated_log_dir):
    import quantui.calc_log as clog

    # Fast single-point history
    for elapsed in (12.0, 14.0, 16.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
            calc_type="single_point",
        )

    # Slow frequency history
    for elapsed in (118.0, 122.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
            calc_type="frequency",
        )

    est_freq = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="frequency",
    )
    est_sp = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="single_point",
    )

    assert est_freq is not None
    assert est_sp is not None
    assert est_freq["n_samples"] == 2
    assert est_freq["seconds"] > 80
    assert est_sp["seconds"] < 30


def test_estimate_time_non_single_point_ignores_legacy_untyped_records(
    isolated_log_dir,
):
    """Legacy untyped records must not enter the freq pool as *direct* matches.

    Before M-EST / EST.2 (session 55) this asserted ``est_freq is None`` —
    a strict "no freq records → no freq estimate" rule. EST.2 added a
    structured cost-model fallback that intentionally reuses the SP
    history (where legacy untyped records DO count) to derive a freq
    estimate when no direct freq records exist. So the contract today
    is two-fold:

    1. Legacy records still don't count as frequency-typed (strategies
       1-4 produce no direct prediction).
    2. The cost-model fallback DOES fire — producing a structured
       SCF-anchor + Hessian + 6N IR estimate — and its value is much
       larger than the underlying SP time (otherwise we know the
       cost-model decomposition collapsed to just the SP anchor).
    """
    import quantui.calc_log as clog

    for elapsed in (10.0, 12.0, 15.0):
        clog.log_calculation(
            formula="CH2O",
            n_atoms=4,
            n_electrons=16,
            method="B3LYP",
            basis="6-31G",
            n_iterations=12,
            elapsed_s=elapsed,
            converged=True,
            n_basis=44,
            n_cores=1,
        )

    est_freq = clog.estimate_time(
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_basis=44,
        calc_type="frequency",
    )

    # EST.2 fallback fires: not None, and noticeably larger than the
    # bare SP median (~12 s) thanks to the +Hessian + 6×n_atoms × SP term.
    assert est_freq is not None
    assert est_freq["seconds"] > 100.0, (
        f"Expected freq estimate > 100 s (SP ~12 s × ~21 cost-model multiplier "
        f"for 4 atoms), got {est_freq['seconds']:.1f} s — suggests the cost "
        "model isn't firing on legacy SP records"
    )


# ============================================================================
# Event log: append/prune race + throttled pruning (M8 audit fix, 2026-07-14)
# ============================================================================


def test_log_event_basic_roundtrip(isolated_log_dir):
    import quantui.calc_log as clog

    clog.log_event("test_event", "hello world", extra_field=42)
    events = clog.get_recent_events(10)
    assert len(events) == 1
    assert events[0]["event"] == "test_event"
    assert events[0]["message"] == "hello world"
    assert events[0]["extra_field"] == 42


def test_prune_events_holds_lock_across_read_and_rewrite(isolated_log_dir):
    """Concurrent log_event() calls must never lose an append to a
    concurrent prune_events() rewrite.

    Regression: prune_events() used to read the file (acquiring and
    releasing the module lock) and only later reacquire the lock to
    rewrite it. An append landing in that gap got silently discarded when
    the rewrite replaced the whole file with the (stale) filtered list
    computed before that append happened. Fixed by making the read +
    filter + rewrite one lock-held critical section.
    """
    import threading

    import quantui.calc_log as clog

    n_threads = 8
    n_events_per_thread = 30

    def worker(tid: int) -> None:
        for i in range(n_events_per_thread):
            clog.log_event("stress_test", f"thread {tid} event {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = clog.get_recent_events(10_000)
    expected = n_threads * n_events_per_thread
    assert len(events) == expected, (
        f"expected {expected} events, got {len(events)} — "
        f"{expected - len(events)} were lost to the append/prune race"
    )


def test_prune_not_triggered_on_every_single_event(isolated_log_dir, monkeypatch):
    """log_event() must not call the full prune on every single append.

    Regression: log_event() called prune_events() (a full read + rewrite
    of the whole file) after every single append — O(file size) work per
    event, O(N^2) over a session. It should only prune periodically.
    """
    import quantui.calc_log as clog

    prune_calls = {"n": 0}
    real_prune = clog.prune_events

    def _counting_prune(*args, **kwargs):
        prune_calls["n"] += 1
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(clog, "prune_events", _counting_prune)

    n_events = clog._PRUNE_EVERY_N_EVENTS * 3 - 1
    for i in range(n_events):
        clog.log_event("test_event", f"event {i}")

    # Should prune roughly every _PRUNE_EVERY_N_EVENTS calls, not once per event.
    assert prune_calls["n"] < n_events
    assert prune_calls["n"] == n_events // clog._PRUNE_EVERY_N_EVENTS


def test_prune_events_still_removes_old_entries(isolated_log_dir):
    """The periodic-prune change must not break the actual 7-day TTL."""
    import json
    from datetime import datetime, timedelta, timezone

    import quantui.calc_log as clog

    path = clog._event_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"timestamp": old_ts, "event": "old", "message": "stale"}) + "\n"
        )
        fh.write(
            json.dumps({"timestamp": new_ts, "event": "new", "message": "fresh"}) + "\n"
        )

    clog.prune_events()

    remaining = clog.get_recent_events(10)
    assert len(remaining) == 1
    assert remaining[0]["event"] == "new"
