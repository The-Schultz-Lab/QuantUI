"""
Issue tracking for QuantUI.

User-reported issues are stored in a local SQLite database alongside the
session event log.  Each issue captures a description and a snapshot of the
app state at the time of the report, making it possible to reconstruct the
sequence of events leading up to a problem.

Database location
-----------------
``<log_dir>/issues.db``  where ``<log_dir>`` is the same directory used by
``calc_log`` (``~/.quantui/logs`` by default, or ``$QUANTUI_LOG_DIR``).

Public API
----------
``log_issue(description, context, session_id)``
    Save an issue and mirror it to the event log.

``get_issues(n)``
    Return the *n* most recent issues as a list of dicts.

``clear_issues()``
    Delete all issue records (drops and recreates the table).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Path helpers (mirror calc_log so both use the same QUANTUI_LOG_DIR env var)
# ---------------------------------------------------------------------------


def _log_dir() -> Path:
    env = os.environ.get("QUANTUI_LOG_DIR")
    return Path(env) if env else Path.home() / ".quantui" / "logs"


def _db_path() -> Path:
    return _log_dir() / "issues.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS issues (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    description  TEXT    NOT NULL,
    session_id   TEXT,
    context_json TEXT
);
"""


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open *path*, commit on success, and always CLOSE the connection.

    ``sqlite3.connect`` used as a context manager only commits/rolls back;
    it never closes, so every call leaked a connection (audit DATA-17).
    """
    conn = sqlite3.connect(str(path))
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _init_db() -> None:
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db) as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_issue(
    description: str,
    context: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> int:
    """Save an issue report to SQLite and the event log.

    Args:
        description: Free-text description of the observed issue.
        context:     Optional snapshot dict (molecule, settings, last result,
                     recent events).  Stored as JSON in the DB.
        session_id:  Caller-supplied session identifier for cross-referencing
                     with the event log.

    Returns:
        The auto-incremented issue id.
    """
    _init_db()
    ts = datetime.now(timezone.utc).isoformat()
    ctx_json = json.dumps(context or {}, ensure_ascii=False)
    with _LOCK:
        with _connect(_db_path()) as conn:
            cursor = conn.execute(
                "INSERT INTO issues (timestamp, description, session_id, context_json)"
                " VALUES (?, ?, ?, ?)",
                (ts, description, session_id, ctx_json),
            )
            conn.commit()
            issue_id: int = cursor.lastrowid  # type: ignore[assignment]

    # Mirror to the sequential event log so issues appear in context
    try:
        from quantui import calc_log as _clog

        _clog.log_event(
            "issue_filed",
            description[:200],
            issue_id=issue_id,
            session_id=session_id,
        )
    except Exception:
        pass

    return issue_id


def get_issues(n: int = 50) -> list[dict]:
    """Return the *n* most recent issues, newest first.

    Args:
        n: Maximum number of issues to return.

    Returns:
        List of dicts with keys: ``id``, ``timestamp``, ``description``,
        ``session_id``, ``context``.
    """
    db = _db_path()
    if not db.exists():
        return []
    with _connect(db) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, description, session_id, context_json"
            " FROM issues ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "description": row[2],
            "session_id": row[3],
            "context": json.loads(row[4] or "{}"),
        }
        for row in rows
    ]


def clear_issues() -> None:
    """Delete all issue records from the database.

    Drops and recreates the ``issues`` table.  The database file itself is
    kept so the path remains stable.
    """
    db = _db_path()
    if not db.exists():
        return
    with _LOCK:
        with _connect(db) as conn:
            conn.execute("DROP TABLE IF EXISTS issues")
            conn.execute(_CREATE_TABLE)
            conn.commit()
