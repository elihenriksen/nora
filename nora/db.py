"""SQLite connection management.

A single file holds the entire world state. Connections are short-lived;
foreign keys are enforced; rows come back as dict-like objects.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DB_PATH

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    # autocommit; explicit BEGIN is used by `transaction()` when atomicity is needed.
    # WAL: lets readers and the recording connection (usage table inserts) make
    #   progress while another connection holds a write transaction. Sticky on
    #   the DB file across connections, so setting per-connect is harmless.
    # timeout=30s: if two writers do contend, wait up to 30s instead of the
    #   5s default — gives long LLM calls room without surprise OperationalError.
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")  # safe with WAL, faster commits
    return conn


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = _connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection wrapped in an immediate transaction.

    On exception the transaction is rolled back, otherwise committed on exit.
    """
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    """Create the database file and apply the schema if not present."""
    schema = SCHEMA_PATH.read_text()
    with connect(path) as conn:
        conn.executescript(schema)
    _migrate(path)


def _migrate(path: Path = DB_PATH) -> None:
    """Idempotent post-schema migrations.

    - Backfill messages_fts for any messages inserted before FTS5 existed.
    - Add conversations.processed_through_message_id for resumable processing.
    """
    import sqlite3 as _sqlite3
    with connect(path) as conn:
        # Detect missing FTS5 rows. Only insert what isn't already there.
        conn.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, content FROM messages "
            "WHERE id NOT IN (SELECT rowid FROM messages_fts)"
        )
        # Add processed-watermark column if not present.
        try:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN processed_through_message_id INTEGER"
            )
        except _sqlite3.OperationalError:
            pass  # column already exists


def is_initialized(path: Path = DB_PATH) -> bool:
    if not path.exists():
        return False
    try:
        with connect(path) as conn:
            row = conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            return row is not None
    except sqlite3.DatabaseError:
        return False
