"""Session event log, backed by SQLite so history survives server restarts.

One table, one row per event:
    search  — the visitor typed a query
    view    — the visitor clicked a movie card

interest.py reads this table to build a live interest vector per session.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_FILE = Path(__file__).parent / "events.db"

EVENT_TYPES = {"search", "view"}


def connect() -> sqlite3.Connection:
    """A fresh connection per request.

    Uvicorn runs sync endpoints on a thread pool, and SQLite connections are
    not safe to share across threads, so this never returns a cached handle.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the table and indexes if they do not exist yet."""
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                type       TEXT    NOT NULL,
                query      TEXT,
                imdb_id    TEXT,
                created_at TEXT    NOT NULL
            )
        """)
        # Phase 3 reads events by session, newest first — index for that shape.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session "
            "ON events (session_id, id DESC)"
        )


def new_session_id() -> str:
    return uuid.uuid4().hex


def log_event(
    session_id: str,
    event_type: str,
    query: str | None = None,
    imdb_id: str | None = None,
) -> int:
    """Append one event and return its row id."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type: {event_type!r}")

    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO events (session_id, type, query, imdb_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                event_type,
                query,
                imdb_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return cur.lastrowid


def events_for(session_id: str, limit: int = 200) -> list[dict]:
    """Most recent events for one session, newest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, session_id, type, query, imdb_id, created_at "
            "FROM events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def forget_session(session_id: str) -> int:
    """Delete every event belonging to one session; returns how many rows went.

    Scoped to a single session_id on purpose — a visitor resetting their own
    history must not be able to touch anybody else's.
    """
    with connect() as conn:
        cur = conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
        return cur.rowcount


def summarise(session_id: str) -> dict:
    """Counts per event type, for the debug endpoint."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM events WHERE session_id = ? "
            "GROUP BY type",
            (session_id,),
        ).fetchall()
        total_sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS n FROM events"
        ).fetchone()["n"]
    return {
        "by_type": {r["type"]: r["n"] for r in rows},
        "total_sessions_seen": total_sessions,
    }
