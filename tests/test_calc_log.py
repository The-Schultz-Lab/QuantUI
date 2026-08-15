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


def test_read_all_tolerates_partial_multibyte_tail(isolated_log_dir):
    """A concurrent/interrupted append can leave a partial multibyte UTF-8
    sequence at EOF. Reading must skip that line, not raise UnicodeDecodeError.

    Regression: the strict-utf-8 read crashed ``QuantUIApp()`` construction
    (via ``get_recent_events``) intermittently under ``pytest -n=auto``.
    """
    import json

    import quantui.calc_log as clog

    path = clog._event_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(json.dumps({"event": "good", "message": "ok"}).encode() + b"\n")
        fh.write(b'{"event": "partial"}\xc3')  # truncated multibyte at EOF

    # Must not raise, and must still return the intact record.
    events = clog.get_recent_events(10)
    assert [e["event"] for e in events] == ["good"]

    # prune_events reads the same file; it must not raise either.
    clog.prune_events()


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


def test_read_all_caches_between_unchanged_calls(isolated_log_dir):
    """L audit fix: _read_all must not re-parse the perf log when it hasn't
    changed since the last call — estimate_time() re-read the entire
    (indefinitely-kept) file on every UI refresh even when nothing new
    had been written.
    """
    import quantui.calc_log as clog

    clog.log_calculation(
        formula="H2",
        n_atoms=2,
        n_electrons=2,
        method="RHF",
        basis="STO-3G",
        n_iterations=5,
        elapsed_s=1.0,
        converged=True,
        n_basis=2,
        n_cores=1,
        calc_type="single_point",
    )

    first = clog._read_all(clog._perf_path())
    assert len(first) == 1

    calls = {"n": 0}
    real_loads = clog.json.loads

    def _counting_loads(s):
        calls["n"] += 1
        return real_loads(s)

    clog.json.loads = _counting_loads
    try:
        second = clog._read_all(clog._perf_path())
    finally:
        clog.json.loads = real_loads

    assert second == first
    assert calls["n"] == 0  # cache hit: no re-parsing of the unchanged file


def test_read_all_cache_invalidates_on_new_write(isolated_log_dir):
    """The caching fix above must not go stale after a genuine new write."""
    import quantui.calc_log as clog

    clog.log_calculation(
        formula="H2",
        n_atoms=2,
        n_electrons=2,
        method="RHF",
        basis="STO-3G",
        n_iterations=5,
        elapsed_s=1.0,
        converged=True,
        n_basis=2,
        n_cores=1,
        calc_type="single_point",
    )
    first = clog._read_all(clog._perf_path())
    assert len(first) == 1

    clog.log_calculation(
        formula="H2",
        n_atoms=2,
        n_electrons=2,
        method="RHF",
        basis="STO-3G",
        n_iterations=6,
        elapsed_s=2.0,
        converged=True,
        n_basis=2,
        n_cores=1,
        calc_type="single_point",
    )
    second = clog._read_all(clog._perf_path())
    assert len(second) == 2


def test_6_31gss_he_basis_count_matches_pyscf(isolated_log_dir):
    """L audit fix: He under 6-31G** must have 5 basis functions, not 2.

    6-31G** adds a p-polarization shell on H/He on top of 6-31G*'s bare
    s-only He (2 bf), the same way 6-31G* already adds p on the heavy
    atoms — so He should follow H's pattern (2 -> 5), not stay at 2.
    Verified against ``pyscf.gto.M(atom="He", basis="6-31g**").nao == 5``.
    """
    import quantui.calc_log as clog

    assert clog.count_basis_functions(["He"], "6-31G**") == 5


def test_basis_function_table_internally_consistent(isolated_log_dir):
    """H and He must carry equal counts in every basis in the lookup table.

    Both are period-1 elements with the same shell structure in every
    basis set this table covers, so a basis that gives H and He different
    counts indicates a transcription error (this caught the 6-31G** nit).
    """
    import quantui.calc_log as clog

    for basis, table in clog._BASIS_FUNCTIONS.items():
        assert (
            table["H"] == table["He"]
        ), f"{basis}: H={table['H']} but He={table['He']}, expected equal"


# ── B3: outer-step telemetry + geom-opt step prior ──────────────────────────


def _log_geom_opt(clog, *, n_steps, method="B3LYP", basis="6-31G", converged=True):
    clog.log_calculation(
        formula="CH2O",
        n_atoms=4,
        n_electrons=16,
        method=method,
        basis=basis,
        n_iterations=10,
        elapsed_s=30.0,
        converged=converged,
        n_basis=44,
        n_cores=1,
        calc_type="geometry_opt",
        n_steps=n_steps,
    )


def test_log_calculation_records_n_steps(isolated_log_dir):
    import quantui.calc_log as clog

    _log_geom_opt(clog, n_steps=7)
    records = clog._read_all(clog._perf_path())
    assert records and records[-1]["n_steps"] == 7


def test_estimate_opt_steps_returns_median(isolated_log_dir):
    import quantui.calc_log as clog

    for n in (5, 7, 9):
        _log_geom_opt(clog, n_steps=n)
    assert clog.estimate_opt_steps("B3LYP", "6-31G") == 7.0


def test_estimate_opt_steps_none_without_history(isolated_log_dir):
    import quantui.calc_log as clog

    assert clog.estimate_opt_steps("B3LYP", "6-31G") is None


def test_estimate_opt_steps_excludes_unconverged_and_other_types(isolated_log_dir):
    import quantui.calc_log as clog

    # Unconverged geom-opt + a single-point with n_steps must be ignored.
    _log_geom_opt(clog, n_steps=99, converged=False)
    clog.log_calculation(
        formula="CH2O",
        n_atoms=4,
        n_electrons=16,
        method="B3LYP",
        basis="6-31G",
        n_iterations=10,
        elapsed_s=12.0,
        converged=True,
        n_basis=44,
        calc_type="single_point",
        n_steps=42,
    )
    assert clog.estimate_opt_steps("B3LYP", "6-31G") is None
    # One valid converged geom-opt → still None (needs the record to exist).
    _log_geom_opt(clog, n_steps=6)
    assert clog.estimate_opt_steps("B3LYP", "6-31G") == 6.0
