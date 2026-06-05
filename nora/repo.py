"""Data access layer.

Pure functions over a sqlite3.Connection. No globals, no caching, no ORM.
Every function takes a connection — callers control transactions via
`nora.db.transaction()`.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from .models import (
    USER_ID,
    Character,
    Conversation,
    ConversationType,
    ContextExchange,
    CharacterTrait,
    EntityType,
    Message,
    Participant,
    Postcard,
    Relationship,
    RelationshipContextEntry,
    SenderType,
    WorldEvent,
    canonicalize_pair,
)


def _new_id() -> str:
    return uuid.uuid4().hex


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    # SQLite CURRENT_TIMESTAMP returns naive UTC text "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


# ----- Characters ----------------------------------------------------------

def create_character(
    conn: sqlite3.Connection,
    *,
    name: str,
    interest: str,
    seed_traits: Optional[list[str]] = None,
    backstory: Optional[str] = None,
    voice_notes: Optional[str] = None,
) -> Character:
    cid = _new_id()
    conn.execute(
        """INSERT INTO characters (id, name, interest, seed_traits, backstory, voice_notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            cid,
            name.strip(),
            interest.strip(),
            json.dumps(seed_traits or []),
            backstory,
            voice_notes,
        ),
    )
    return get_character_by_id(conn, cid)


def _row_to_character(row: sqlite3.Row) -> Character:
    return Character(
        id=row["id"],
        name=row["name"],
        interest=row["interest"],
        seed_traits=json.loads(row["seed_traits"] or "[]"),
        backstory=row["backstory"],
        voice_notes=row["voice_notes"],
        created_at=_parse_dt(row["created_at"]),
    )


def get_character_by_id(conn: sqlite3.Connection, cid: str) -> Character:
    row = conn.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
    if not row:
        raise LookupError(f"character id={cid} not found")
    return _row_to_character(row)


def get_character_by_name(conn: sqlite3.Connection, name: str) -> Optional[Character]:
    row = conn.execute(
        "SELECT * FROM characters WHERE name = ? COLLATE NOCASE", (name.strip(),)
    ).fetchone()
    return _row_to_character(row) if row else None


def list_characters(conn: sqlite3.Connection) -> list[Character]:
    rows = conn.execute("SELECT * FROM characters ORDER BY created_at").fetchall()
    return [_row_to_character(r) for r in rows]


def delete_character(conn: sqlite3.Connection, cid: str) -> None:
    conn.execute("DELETE FROM characters WHERE id = ?", (cid,))


# ----- Conversations + messages -------------------------------------------

def create_conversation(
    conn: sqlite3.Connection,
    *,
    type: ConversationType,
    participants: list[Participant],
    topic: Optional[str] = None,
) -> Conversation:
    conv_id = _new_id()
    conn.execute(
        "INSERT INTO conversations (id, type, topic) VALUES (?, ?, ?)",
        (conv_id, type, topic),
    )
    for p in participants:
        conn.execute(
            """INSERT INTO conversation_participants
               (conversation_id, participant_type, participant_id)
               VALUES (?, ?, ?)""",
            (conv_id, p.type, p.id),
        )
    return get_conversation(conn, conv_id)


def get_conversation(conn: sqlite3.Connection, conv_id: str) -> Conversation:
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not row:
        raise LookupError(f"conversation id={conv_id} not found")
    return Conversation(
        id=row["id"],
        type=row["type"],
        topic=row["topic"],
        started_at=_parse_dt(row["started_at"]),
        ended_at=_parse_dt(row["ended_at"]),
        processed_at=_parse_dt(row["processed_at"]),
    )


def end_conversation(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute(
        "UPDATE conversations SET ended_at = CURRENT_TIMESTAMP WHERE id = ? AND ended_at IS NULL",
        (conv_id,),
    )


def mark_conversation_processed(conn: sqlite3.Connection, conv_id: str) -> None:
    conn.execute(
        "UPDATE conversations SET processed_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conv_id,),
    )


def list_conversations(
    conn: sqlite3.Connection, limit: int = 50
) -> list[Conversation]:
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        Conversation(
            id=r["id"],
            type=r["type"],
            topic=r["topic"],
            started_at=_parse_dt(r["started_at"]),
            ended_at=_parse_dt(r["ended_at"]),
            processed_at=_parse_dt(r["processed_at"]),
        )
        for r in rows
    ]


def list_participants(conn: sqlite3.Connection, conv_id: str) -> list[Participant]:
    rows = conn.execute(
        """SELECT participant_type, participant_id FROM conversation_participants
           WHERE conversation_id = ?""",
        (conv_id,),
    ).fetchall()
    return [Participant(type=r["participant_type"], id=r["participant_id"]) for r in rows]


def add_message(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    sender_type: SenderType,
    sender_id: str,
    content: str,
) -> Message:
    cur = conn.execute(
        """INSERT INTO messages (conversation_id, sender_type, sender_id, content)
           VALUES (?, ?, ?, ?)""",
        (conversation_id, sender_type, sender_id, content),
    )
    msg_id = cur.lastrowid
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        sender_type=row["sender_type"],
        sender_id=row["sender_id"],
        content=row["content"],
        created_at=_parse_dt(row["created_at"]),
    )


def list_messages(conn: sqlite3.Connection, conv_id: str) -> list[Message]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conv_id,)
    ).fetchall()
    return [
        Message(
            id=r["id"],
            conversation_id=r["conversation_id"],
            sender_type=r["sender_type"],
            sender_id=r["sender_id"],
            content=r["content"],
            created_at=_parse_dt(r["created_at"]),
        )
        for r in rows
    ]


# ----- Relationships -------------------------------------------------------

def get_relationship_if_exists(
    conn: sqlite3.Connection,
    *,
    a_type: EntityType,
    a_id: str,
    b_type: EntityType,
    b_id: str,
) -> Optional[Relationship]:
    """Return a relationship if it already exists, else None.

    Read-only — unlike `get_or_create_relationship`, never inserts. Use this
    on render paths (system prompt building, inspection) so we don't pollute
    the table with empty rows just to look up "do they have history?".
    """
    a_type, a_id, b_type, b_id = canonicalize_pair(a_type, a_id, b_type, b_id)
    row = conn.execute(
        """SELECT * FROM relationships
           WHERE entity_a_type = ? AND entity_a_id = ?
             AND entity_b_type = ? AND entity_b_id = ?""",
        (a_type, a_id, b_type, b_id),
    ).fetchone()
    if row is None:
        return None
    return Relationship(
        id=row["id"], a_type=row["entity_a_type"], a_id=row["entity_a_id"],
        b_type=row["entity_b_type"], b_id=row["entity_b_id"],
    )


def get_or_create_relationship(
    conn: sqlite3.Connection,
    *,
    a_type: EntityType,
    a_id: str,
    b_type: EntityType,
    b_id: str,
) -> Relationship:
    a_type, a_id, b_type, b_id = canonicalize_pair(a_type, a_id, b_type, b_id)
    row = conn.execute(
        """SELECT * FROM relationships
           WHERE entity_a_type = ? AND entity_a_id = ?
             AND entity_b_type = ? AND entity_b_id = ?""",
        (a_type, a_id, b_type, b_id),
    ).fetchone()
    if row:
        return Relationship(
            id=row["id"], a_type=row["entity_a_type"], a_id=row["entity_a_id"],
            b_type=row["entity_b_type"], b_id=row["entity_b_id"],
        )
    rid = _new_id()
    conn.execute(
        """INSERT INTO relationships
           (id, entity_a_type, entity_a_id, entity_b_type, entity_b_id)
           VALUES (?, ?, ?, ?, ?)""",
        (rid, a_type, a_id, b_type, b_id),
    )
    return Relationship(
        id=rid, a_type=a_type, a_id=a_id, b_type=b_type, b_id=b_id
    )


def list_relationships_for(
    conn: sqlite3.Connection, *, entity_type: EntityType, entity_id: str
) -> list[Relationship]:
    rows = conn.execute(
        """SELECT * FROM relationships
           WHERE (entity_a_type = ? AND entity_a_id = ?)
              OR (entity_b_type = ? AND entity_b_id = ?)""",
        (entity_type, entity_id, entity_type, entity_id),
    ).fetchall()
    return [
        Relationship(
            id=r["id"], a_type=r["entity_a_type"], a_id=r["entity_a_id"],
            b_type=r["entity_b_type"], b_id=r["entity_b_id"],
        )
        for r in rows
    ]


def list_context_for_relationship(
    conn: sqlite3.Connection, relationship_id: str
) -> list[RelationshipContextEntry]:
    rows = conn.execute(
        """SELECT * FROM relationship_context
           WHERE relationship_id = ?
           ORDER BY last_updated_at DESC""",
        (relationship_id,),
    ).fetchall()
    return [
        RelationshipContextEntry(
            relationship_id=r["relationship_id"],
            topic=r["topic"],
            a_perspective=r["a_perspective"],
            b_perspective=r["b_perspective"],
            shared_history=r["shared_history"],
            last_updated_at=_parse_dt(r["last_updated_at"]),
            last_conversation_id=r["last_conversation_id"],
        )
        for r in rows
    ]


def upsert_relationship_context(
    conn: sqlite3.Connection,
    *,
    relationship_id: str,
    topic: str,
    a_perspective: Optional[str] = None,
    b_perspective: Optional[str] = None,
    shared_history: Optional[str] = None,
    last_conversation_id: Optional[str] = None,
) -> None:
    """Insert or update a topic entry for a relationship.

    For updates, non-None fields overwrite existing values. shared_history is
    appended (kept as a running narrative) rather than replaced.
    """
    existing = conn.execute(
        "SELECT * FROM relationship_context WHERE relationship_id = ? AND topic = ?",
        (relationship_id, topic),
    ).fetchone()

    if existing is None:
        conn.execute(
            """INSERT INTO relationship_context
               (id, relationship_id, topic, a_perspective, b_perspective,
                shared_history, last_conversation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                _new_id(), relationship_id, topic,
                a_perspective, b_perspective, shared_history, last_conversation_id,
            ),
        )
        return

    new_a = a_perspective if a_perspective is not None else existing["a_perspective"]
    new_b = b_perspective if b_perspective is not None else existing["b_perspective"]
    if shared_history:
        prior = existing["shared_history"] or ""
        new_history = (prior + "\n" if prior else "") + shared_history
    else:
        new_history = existing["shared_history"]

    conn.execute(
        """UPDATE relationship_context
           SET a_perspective = ?, b_perspective = ?, shared_history = ?,
               last_updated_at = CURRENT_TIMESTAMP, last_conversation_id = ?
           WHERE relationship_id = ? AND topic = ?""",
        (new_a, new_b, new_history, last_conversation_id, relationship_id, topic),
    )


# ----- Character traits (emergent identity) -------------------------------

def list_traits(
    conn: sqlite3.Connection, character_id: str
) -> list[CharacterTrait]:
    rows = conn.execute(
        """SELECT * FROM character_traits WHERE character_id = ?
           ORDER BY strength DESC, last_updated_at DESC""",
        (character_id,),
    ).fetchall()
    return [_row_to_trait(r) for r in rows]


def get_trait(
    conn: sqlite3.Connection, character_id: str, topic: str
) -> Optional[CharacterTrait]:
    row = conn.execute(
        "SELECT * FROM character_traits WHERE character_id = ? AND topic = ?",
        (character_id, topic),
    ).fetchone()
    return _row_to_trait(row) if row else None


def _row_to_trait(row: sqlite3.Row) -> CharacterTrait:
    return CharacterTrait(
        character_id=row["character_id"],
        topic=row["topic"],
        perspective=row["perspective"],
        strength=row["strength"],
        origin_summary=row["origin_summary"],
        formed_via=json.loads(row["formed_via"] or "[]"),
        last_updated_at=_parse_dt(row["last_updated_at"]),
    )


def upsert_trait(
    conn: sqlite3.Connection,
    *,
    character_id: str,
    topic: str,
    perspective: str,
    strength: int = 1,
    origin_summary: Optional[str] = None,
    formed_via_add: Optional[list[str]] = None,
) -> None:
    """Insert a trait or update an existing one.

    On update: perspective overwrites; strength is set to the max of old and
    new; formed_via list grows (deduplicated); origin_summary overwrites.
    """
    existing = conn.execute(
        "SELECT * FROM character_traits WHERE character_id = ? AND topic = ?",
        (character_id, topic),
    ).fetchone()

    if existing is None:
        conn.execute(
            """INSERT INTO character_traits
               (id, character_id, topic, perspective, strength, origin_summary, formed_via)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                _new_id(), character_id, topic, perspective,
                max(1, min(5, strength)), origin_summary,
                json.dumps(list(dict.fromkeys(formed_via_add or []))),
            ),
        )
        return

    prior_via = json.loads(existing["formed_via"] or "[]")
    merged_via = list(dict.fromkeys(prior_via + (formed_via_add or [])))
    new_strength = max(existing["strength"], max(1, min(5, strength)))
    conn.execute(
        """UPDATE character_traits
           SET perspective = ?, strength = ?, origin_summary = ?,
               formed_via = ?, last_updated_at = CURRENT_TIMESTAMP
           WHERE character_id = ? AND topic = ?""",
        (
            perspective, new_strength,
            origin_summary if origin_summary is not None else existing["origin_summary"],
            json.dumps(merged_via),
            character_id, topic,
        ),
    )


# ----- Context exchanges --------------------------------------------------

def record_exchange(
    conn: sqlite3.Connection, exchange: ContextExchange
) -> None:
    conn.execute(
        """INSERT INTO context_exchanges
           (id, conversation_id, from_character_id, to_character_id, exchange_type, payload)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            _new_id(),
            exchange.conversation_id,
            exchange.from_character_id,
            exchange.to_character_id,
            exchange.exchange_type,
            json.dumps(exchange.payload),
        ),
    )


def list_exchanges_for_pair(
    conn: sqlite3.Connection, char_a_id: str, char_b_id: str, limit: int = 50
) -> list[ContextExchange]:
    rows = conn.execute(
        """SELECT * FROM context_exchanges
           WHERE (from_character_id = ? AND to_character_id = ?)
              OR (from_character_id = ? AND to_character_id = ?)
           ORDER BY created_at DESC LIMIT ?""",
        (char_a_id, char_b_id, char_b_id, char_a_id, limit),
    ).fetchall()
    return [
        ContextExchange(
            conversation_id=r["conversation_id"],
            from_character_id=r["from_character_id"],
            to_character_id=r["to_character_id"],
            exchange_type=r["exchange_type"],
            payload=json.loads(r["payload"]),
        )
        for r in rows
    ]


# ----- World events + postcards -------------------------------------------

def record_world_event(conn: sqlite3.Connection, event: WorldEvent) -> None:
    conn.execute(
        """INSERT INTO world_events (conversation_id, event_type, summary, payload)
           VALUES (?, ?, ?, ?)""",
        (
            event.conversation_id,
            event.event_type,
            event.summary,
            json.dumps(event.payload) if event.payload is not None else None,
        ),
    )


def list_world_events(conn: sqlite3.Connection, limit: int = 50) -> list[WorldEvent]:
    rows = conn.execute(
        "SELECT * FROM world_events ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        WorldEvent(
            conversation_id=r["conversation_id"],
            event_type=r["event_type"],
            summary=r["summary"],
            payload=json.loads(r["payload"]) if r["payload"] else None,
        )
        for r in rows
    ]


def create_postcard(
    conn: sqlite3.Connection,
    *,
    character_a_id: str,
    character_b_id: str,
    summary: str,
    scene: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Postcard:
    pid = _new_id()
    a, b = sorted([character_a_id, character_b_id])
    conn.execute(
        """INSERT INTO postcards
           (id, conversation_id, character_a_id, character_b_id, summary, scene)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (pid, conversation_id, a, b, summary, scene),
    )
    row = conn.execute("SELECT * FROM postcards WHERE id = ?", (pid,)).fetchone()
    return Postcard(
        id=row["id"],
        conversation_id=row["conversation_id"],
        character_a_id=row["character_a_id"],
        character_b_id=row["character_b_id"],
        summary=row["summary"],
        scene=row["scene"],
        created_at=_parse_dt(row["created_at"]),
    )


# ----- Settings (key-value) ----------------------------------------------

def get_setting(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn: sqlite3.Connection, key: str, value: Optional[str]) -> None:
    if value is None:
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    else:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


# Convenience accessors for the user name (the most-used setting).
def get_user_name(conn: sqlite3.Connection) -> Optional[str]:
    return get_setting(conn, "user_name")


def set_user_name(conn: sqlite3.Connection, name: str) -> None:
    set_setting(conn, "user_name", name.strip() or None)


# ----- Usage / cost telemetry --------------------------------------------

def usage_totals(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS n, "
        "COALESCE(SUM(prompt_tokens),0) AS pt, "
        "COALESCE(SUM(completion_tokens),0) AS ct, "
        "COALESCE(SUM(cost_usd),0) AS cost "
        "FROM usage"
    ).fetchone()
    return {
        "calls": row["n"],
        "prompt_tokens": row["pt"],
        "completion_tokens": row["ct"],
        "cost_usd": row["cost"],
    }


def usage_by_model(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT model, COUNT(*) AS n, "
        "COALESCE(SUM(prompt_tokens),0) AS pt, "
        "COALESCE(SUM(completion_tokens),0) AS ct, "
        "COALESCE(SUM(cost_usd),0) AS cost "
        "FROM usage GROUP BY model ORDER BY cost DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def usage_by_purpose(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT purpose, COUNT(*) AS n, "
        "COALESCE(SUM(prompt_tokens),0) AS pt, "
        "COALESCE(SUM(completion_tokens),0) AS ct, "
        "COALESCE(SUM(cost_usd),0) AS cost "
        "FROM usage GROUP BY purpose ORDER BY cost DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def usage_recent_sessions(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT u.conversation_id, COUNT(*) AS n, "
        "COALESCE(SUM(u.cost_usd),0) AS cost, "
        "COALESCE(SUM(u.prompt_tokens),0) AS pt, "
        "COALESCE(SUM(u.completion_tokens),0) AS ct, "
        "MAX(u.created_at) AS last_at, "
        "c.type, c.topic "
        "FROM usage u "
        "LEFT JOIN conversations c ON c.id = u.conversation_id "
        "WHERE u.conversation_id IS NOT NULL "
        "GROUP BY u.conversation_id "
        "ORDER BY last_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ----- Search ------------------------------------------------------------

def search_messages(
    conn: sqlite3.Connection, query: str, limit: int = 30
) -> list[dict]:
    """Full-text search over message content. Returns rows joined with sender info."""
    rows = conn.execute(
        "SELECT m.id, m.conversation_id, m.sender_type, m.sender_id, "
        "m.content, m.created_at, c.type AS conv_type, c.topic AS conv_topic "
        "FROM messages_fts f "
        "JOIN messages m ON m.id = f.rowid "
        "JOIN conversations c ON c.id = m.conversation_id "
        "WHERE messages_fts MATCH ? "
        "ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def list_postcards(conn: sqlite3.Connection, limit: int = 50) -> list[Postcard]:
    rows = conn.execute(
        "SELECT * FROM postcards ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        Postcard(
            id=r["id"],
            conversation_id=r["conversation_id"],
            character_a_id=r["character_a_id"],
            character_b_id=r["character_b_id"],
            summary=r["summary"],
            scene=r["scene"],
            created_at=_parse_dt(r["created_at"]),
        )
        for r in rows
    ]
