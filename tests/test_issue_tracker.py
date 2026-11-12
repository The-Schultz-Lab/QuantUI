"""Tests for quantui.issue_tracker."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_log_dir(tmp_path, monkeypatch):
    """Point QUANTUI_LOG_DIR at a fresh tmp dir for every test."""
    # issue_tracker._log_dir() (and calc_log's) read the env var on every
    # call, so no importlib.reload is needed (audit INFRA-18).
    monkeypatch.setenv("QUANTUI_LOG_DIR", str(tmp_path))
    yield tmp_path


# ---------------------------------------------------------------------------
# Basic persistence
# ---------------------------------------------------------------------------


def test_log_issue_creates_db(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.log_issue("First test issue")
    assert (isolated_log_dir / "issues.db").exists()


def test_log_issue_returns_integer_id(isolated_log_dir):
    import quantui.issue_tracker as tracker

    issue_id = tracker.log_issue("Test issue")
    assert isinstance(issue_id, int)
    assert issue_id >= 1


def test_log_issue_ids_are_monotonically_increasing(isolated_log_dir):
    import quantui.issue_tracker as tracker

    id1 = tracker.log_issue("First")
    id2 = tracker.log_issue("Second")
    id3 = tracker.log_issue("Third")
    assert id1 < id2 < id3


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_get_issues_empty_before_any_reports(isolated_log_dir):
    import quantui.issue_tracker as tracker

    assert tracker.get_issues() == []


def test_get_issues_returns_most_recent_first(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.log_issue("Older")
    tracker.log_issue("Newer")
    issues = tracker.get_issues()
    assert issues[0]["description"] == "Newer"
    assert issues[1]["description"] == "Older"


def test_get_issues_respects_n_limit(isolated_log_dir):
    import quantui.issue_tracker as tracker

    for i in range(5):
        tracker.log_issue(f"Issue {i}")
    assert len(tracker.get_issues(n=3)) == 3


def test_get_issues_returns_all_fields(isolated_log_dir):
    import quantui.issue_tracker as tracker

    ctx = {"molecule": "H2O", "method": "RHF"}
    issue_id = tracker.log_issue("Field test", context=ctx, session_id="abc123")
    issues = tracker.get_issues(n=1)
    assert issues[0]["id"] == issue_id
    assert issues[0]["description"] == "Field test"
    assert issues[0]["session_id"] == "abc123"
    assert issues[0]["context"]["molecule"] == "H2O"
    assert "timestamp" in issues[0]


def test_log_issue_stores_context_as_dict(isolated_log_dir):
    import quantui.issue_tracker as tracker

    ctx = {"key": "value", "nested": {"a": 1}}
    tracker.log_issue("Context test", context=ctx)
    issues = tracker.get_issues(n=1)
    assert issues[0]["context"]["nested"]["a"] == 1


def test_log_issue_handles_empty_context(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.log_issue("No context")
    issues = tracker.get_issues(n=1)
    assert issues[0]["context"] == {}


# ---------------------------------------------------------------------------
# clear_issues
# ---------------------------------------------------------------------------


def test_clear_issues_removes_all_records(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.log_issue("A")
    tracker.log_issue("B")
    tracker.clear_issues()
    assert tracker.get_issues() == []


def test_clear_issues_on_empty_db_is_safe(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.clear_issues()  # should not raise


def test_can_log_after_clear(isolated_log_dir):
    import quantui.issue_tracker as tracker

    tracker.log_issue("Before")
    tracker.clear_issues()
    new_id = tracker.log_issue("After clear")
    issues = tracker.get_issues()
    assert len(issues) == 1
    assert issues[0]["description"] == "After clear"
    assert isinstance(new_id, int)


# ---------------------------------------------------------------------------
# Event log mirroring
# ---------------------------------------------------------------------------


def test_log_issue_mirrors_to_event_log(isolated_log_dir):
    import quantui.calc_log as clog
    import quantui.issue_tracker as tracker

    tracker.log_issue("Mirror test", session_id="sess42")
    events = clog.get_recent_events(50)
    issue_events = [e for e in events if e.get("event") == "issue_filed"]
    assert len(issue_events) == 1
    assert issue_events[0]["session_id"] == "sess42"
    assert "Mirror test" in issue_events[0]["message"]


# ---------------------------------------------------------------------------
# calc_log.clear_event_log
# ---------------------------------------------------------------------------


def test_clear_event_log_removes_file(isolated_log_dir):
    import quantui.calc_log as clog

    clog.log_event("test", "hello")
    assert (isolated_log_dir / "event_log.jsonl").exists()
    clog.clear_event_log()
    assert not (isolated_log_dir / "event_log.jsonl").exists()


def test_clear_event_log_when_no_file_is_safe(isolated_log_dir):
    import quantui.calc_log as clog

    clog.clear_event_log()  # should not raise


def test_clear_event_log_does_not_touch_perf_log(isolated_log_dir):
    import quantui.calc_log as clog

    clog.log_calculation(
        formula="H2",
        n_atoms=2,
        n_electrons=2,
        method="RHF",
        basis="STO-3G",
        n_iterations=8,
        elapsed_s=1.2,
        converged=True,
    )
    clog.log_event("test", "event to clear")
    clog.clear_event_log()
    assert not (isolated_log_dir / "event_log.jsonl").exists()
    assert (isolated_log_dir / "perf_log.jsonl").exists()


def test_clear_event_log_does_not_touch_issues_db(isolated_log_dir):
    import quantui.calc_log as clog
    import quantui.issue_tracker as tracker

    tracker.log_issue("Do not delete me")
    clog.clear_event_log()
    assert len(tracker.get_issues()) == 1
