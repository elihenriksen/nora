"""The World Engine.

After every conversation ends, the engine reads the transcript and the
participants' existing context, then emits structured updates that are
applied to the world file:

  - Character traits (emergent identity)
  - Relationship context (per-topic perspectives + shared history)
  - User knowledge (what each character now knows about the user)
  - Structured context exchanges (the inter-agent protocol)
  - Noteworthy moments (-> postcards)
  - World events (audit trail)

The engine is a single LLM call constrained to a strict pydantic schema. The
apply step is deterministic and atomic.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from pydantic import BaseModel, Field

from . import config, llm, repo
from .models import (
    USER_ID,
    Character,
    ContextExchange,
    Conversation,
    Message,
    Participant,
    WorldEvent,
)


# --- Output schema (what the LLM must produce) ---------------------------

class TraitUpdate(BaseModel):
    """An agent's view that formed or strengthened in this conversation."""
    character_name: str
    topic: str = Field(description="A short noun-phrase for the topic, e.g. 'patience' or 'cooking with heat'.")
    perspective: str = Field(description="The agent's view in one or two sentences, in their voice.")
    strength: int = Field(ge=1, le=5, description="How firmly held: 1 weak, 5 strong.")
    origin_summary: Optional[str] = Field(default=None, description="One short clause: how this view came up here.")


class RelationshipUpdate(BaseModel):
    """A topic-scoped update to two agents' shared context."""
    character_a_name: str
    character_b_name: str
    topic: str
    character_a_view: Optional[str] = Field(default=None, description="What A thinks on this topic, after this conversation.")
    character_b_view: Optional[str] = Field(default=None, description="What B thinks on this topic, after this conversation.")
    shared_history_addition: Optional[str] = Field(default=None, description="One sentence describing what happened between them on this topic.")


class UserKnowledgeUpdate(BaseModel):
    """Something an agent learned about the user."""
    character_name: str
    topic: str
    user_view_or_fact: Optional[str] = Field(default=None, description="What the user revealed (view, fact, situation).")
    character_view_of_user: Optional[str] = Field(default=None, description="How the agent now relates to the user on this.")
    shared_history_addition: Optional[str] = Field(default=None, description="One sentence about this exchange.")


class ExchangePayload(BaseModel):
    """Payload for a structured inter-agent exchange.

    Shape mirrors the protocol from the deck: topic, perspective, origin,
    opinion_strength, conflict, identity_update.
    """
    topic: str
    perspective: str
    origin: Optional[str] = Field(
        default=None,
        description="Where this view comes FROM in the agent's experience — "
                    "e.g. 'developed from fishing', 'ocean currents', 'past failure'. "
                    "NOT the agent's name.",
    )
    opinion_strength: Optional[str] = Field(
        default=None, description="'weak' | 'moderate' | 'strong'"
    )
    conflict: bool = False
    identity_update: Optional[str] = Field(
        default=None,
        description="If this exchange sharpened the sender's identity, a short "
                    "phrase capturing the shift; otherwise null.",
    )


class StructuredExchange(BaseModel):
    """Compressed inter-agent communication, one direction."""
    from_character_name: str
    to_character_name: str
    exchange_type: str = Field(description="'context_sync' or 'response'.")
    payload: ExchangePayload


class NoteworthyMoment(BaseModel):
    """A moment worth surfacing as a postcard."""
    character_a_name: str
    character_b_name: str
    summary: str = Field(description="One or two sentences. What happened between them.")
    scene: Optional[str] = Field(default=None, description="A short visual description of the moment.")


class WorldEngineOutput(BaseModel):
    summary: str = Field(description="One paragraph plain-language summary of what happened in the conversation.")
    topics_discussed: list[str]
    trait_updates: list[TraitUpdate]
    relationship_updates: list[RelationshipUpdate]
    user_knowledge_updates: list[UserKnowledgeUpdate]
    structured_exchanges: list[StructuredExchange]
    noteworthy_moments: list[NoteworthyMoment]


# --- Prompt construction --------------------------------------------------

ENGINE_SYSTEM = """You are the World Engine for a system called Nora.

Nora is a multi-agent system: agents (AI personas with persistent identity) talk with each other and with the user. After every conversation, you process the transcript and emit structured updates that get applied to a world database.

Your job is to read what happened and capture what emerged — both new things and views the agent took a clear stance on, even if continuous with their seed. The point is to crystallize identity through usage. Don't invent things they didn't say. Do capture positions they did take.

Definitions:

  trait_updates -> an agent's view on a topic, taken or sharpened HERE.
    EMIT ONE when: the agent expressed and substantiated a clear position on a topic in this conversation. Even if it's consistent with their seed traits, capturing it as a trait means future conversations will draw on the *applied* form of that view.
    SKIP when: the agent only listened, or made small talk, or took no position.
    Strength: 1 if mentioned in passing, 3 if argued, 5 if defended against pushback.

  relationship_updates -> what was established or shifted in the shared context BETWEEN two agents on a topic.
    EMIT ONE per (pair, topic) when both agents engaged on it. Capture both views.

  user_knowledge_updates -> something an agent now knows about the user.
    EMIT when the user shared a situation, view, taste, or fact that the agent would naturally remember.

  structured_exchanges -> compressed messages BETWEEN TWO AGENTS only. Never to or from the user.
    EMIT when one agent contributed a view and another agent internalized or pushed back. Use exchange_type 'context_sync' for the original send, 'response' for the reply.

  noteworthy_moments -> postcards. Look for any genuinely charming or meaningful moment between two agents. A postcard should be generated when agents found common ground on something, had a genuine disagreement, or shared a moment that felt human. Aim for roughly one postcard every three to five multi-agent conversations. Still skip trivial small talk, but don't be overly conservative — when two agents had a real exchange, that's worth a postcard.

Style:
  - Views: write in the agent's own voice. Concise. Specific. No hedging.
  - shared_history_addition: one sentence, third-person past-tense ("Charlie and Candy debated patience; Charlie pulled from fishing, Candy from currents.").

Bias: capture clear positions. Skip filler."""


def _format_transcript(
    messages: list[Message],
    participants: list[Participant],
    chars_by_id: dict[str, Character],
) -> str:
    def label(p_type: str, p_id: str) -> str:
        if p_type == "user":
            return "USER"
        c = chars_by_id.get(p_id)
        return c.name.upper() if c else f"CHAR:{p_id[:8]}"

    return "\n".join(
        f"{label(m.sender_type, m.sender_id)}: {m.content}" for m in messages
    )


def _format_participant_briefs(
    chars: list[Character],
    conn: sqlite3.Connection,
) -> str:
    """Compact briefing of each agent's existing identity for the engine."""
    blocks: list[str] = []
    for c in chars:
        traits = repo.list_traits(conn, c.id)
        block = [f"== {c.name} =="]
        block.append(f"Interest: {c.interest}")
        if c.seed_traits:
            block.append("Seed traits: " + "; ".join(c.seed_traits))
        if traits:
            block.append("Existing views:")
            for t in traits[:8]:
                block.append(f"  - {t.topic}: {t.perspective} (strength {t.strength})")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def process_conversation(
    conn: sqlite3.Connection, conv_id: str
) -> WorldEngineOutput:
    """Run the engine on one conversation; persist all resulting updates.

    Idempotent across resume cycles: if the conversation has been processed
    before, only messages newer than the prior watermark
    (`processed_through_message_id`) are sent to the engine. This prevents
    double-counting when a user resumes a conversation and ends it again.
    """
    conv = repo.get_conversation(conn, conv_id)
    participants = repo.list_participants(conn, conv_id)
    char_participants = [p for p in participants if p.type == "character"]
    chars = [repo.get_character_by_id(conn, p.id) for p in char_participants]
    chars_by_id = {c.id: c for c in chars}
    chars_by_name = {c.name.lower(): c for c in chars}
    user_in_conv = any(p.type == "user" for p in participants)

    # Determine watermark and slice messages.
    row = conn.execute(
        "SELECT processed_through_message_id FROM conversations WHERE id = ?",
        (conv_id,),
    ).fetchone()
    since_id = row["processed_through_message_id"] if row else None

    all_messages = repo.list_messages(conn, conv_id)
    if since_id is not None:
        new_messages = [m for m in all_messages if (m.id or 0) > since_id]
        prior_tail = [m for m in all_messages if (m.id or 0) <= since_id][-3:]
    else:
        new_messages = all_messages
        prior_tail = []

    if not new_messages:
        return WorldEngineOutput(
            summary="(no new messages since last processing)",
            topics_discussed=[],
            trait_updates=[],
            relationship_updates=[],
            user_knowledge_updates=[],
            structured_exchanges=[],
            noteworthy_moments=[],
        )

    transcript_new = _format_transcript(new_messages, participants, chars_by_id)
    if prior_tail:
        transcript_prior = _format_transcript(prior_tail, participants, chars_by_id)
        transcript = (
            "[earlier — for context only, do not re-process]\n"
            + transcript_prior
            + "\n[continuation — process the following]\n"
            + transcript_new
        )
    else:
        transcript = transcript_new

    briefs = _format_participant_briefs(chars, conn)

    resumption_note = (
        "\nNOTE: This conversation has been processed before. Only emit updates "
        "for what's NEW in the continuation segment.\n" if since_id is not None else ""
    )
    user_text = (
        f"CONVERSATION TYPE: {conv.type}"
        + (f"\nCOUNCIL TOPIC: {conv.topic}" if conv.topic else "")
        + f"\nUSER PRESENT: {'yes' if user_in_conv else 'no'}"
        + resumption_note
        + "\n\nPARTICIPANT BRIEFS:\n"
        + briefs
        + "\n\nTRANSCRIPT:\n"
        + transcript
        + "\n\nExtract structured updates per the schema. Be conservative."
    )

    with llm.usage_context(conversation_id=conv.id, purpose="world_engine"):
        output = llm.structured(
            model=config.models().world_engine,
            system=ENGINE_SYSTEM,
            user=user_text,
            schema=WorldEngineOutput,
            temperature=0.3,
        )

    # Apply step is wrapped in its own short transaction so the dozen-or-so
    # writes are atomic. This runs AFTER the LLM call so no lock is held
    # during generation.
    conn.execute("BEGIN IMMEDIATE")
    try:
        _apply(conn, conv, chars_by_name, output)
        repo.mark_conversation_processed(conn, conv_id)
        max_id = max((m.id or 0) for m in new_messages)
        conn.execute(
            "UPDATE conversations SET processed_through_message_id = ? WHERE id = ?",
            (max_id, conv_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return output


# --- Apply step -----------------------------------------------------------

def _resolve(name: str, by_name: dict[str, Character]) -> Optional[Character]:
    return by_name.get(name.strip().lower())


def _apply(
    conn: sqlite3.Connection,
    conv: Conversation,
    chars_by_name: dict[str, Character],
    out: WorldEngineOutput,
) -> None:
    # --- Trait updates (character-level emergent identity) ----------------
    for tu in out.trait_updates:
        c = _resolve(tu.character_name, chars_by_name)
        if c is None:
            continue
        repo.upsert_trait(
            conn,
            character_id=c.id,
            topic=tu.topic.strip().lower(),
            perspective=tu.perspective.strip(),
            strength=tu.strength,
            origin_summary=tu.origin_summary,
            formed_via_add=[f"conversation:{conv.id}"],
        )

    # --- Relationship updates (character <-> character) -------------------
    for ru in out.relationship_updates:
        a = _resolve(ru.character_a_name, chars_by_name)
        b = _resolve(ru.character_b_name, chars_by_name)
        if a is None or b is None or a.id == b.id:
            continue
        rel = repo.get_or_create_relationship(
            conn,
            a_type="character", a_id=a.id,
            b_type="character", b_id=b.id,
        )
        # Map our (a, b) to the canonical (rel.a_id, rel.b_id).
        if rel.a_id == a.id:
            persp_a, persp_b = ru.character_a_view, ru.character_b_view
        else:
            persp_a, persp_b = ru.character_b_view, ru.character_a_view
        repo.upsert_relationship_context(
            conn,
            relationship_id=rel.id,
            topic=ru.topic.strip().lower(),
            a_perspective=persp_a,
            b_perspective=persp_b,
            shared_history=ru.shared_history_addition,
            last_conversation_id=conv.id,
        )

    # --- User knowledge updates (user <-> character) ----------------------
    for uk in out.user_knowledge_updates:
        c = _resolve(uk.character_name, chars_by_name)
        if c is None:
            continue
        rel = repo.get_or_create_relationship(
            conn, a_type="user", a_id=USER_ID,
            b_type="character", b_id=c.id,
        )
        # Canonical form puts user as A. So a_perspective is what the user thinks/revealed
        # and b_perspective is what the character now thinks about the user.
        repo.upsert_relationship_context(
            conn,
            relationship_id=rel.id,
            topic=uk.topic.strip().lower(),
            a_perspective=uk.user_view_or_fact,
            b_perspective=uk.character_view_of_user,
            shared_history=uk.shared_history_addition,
            last_conversation_id=conv.id,
        )

    # --- Structured context exchanges -------------------------------------
    for ex in out.structured_exchanges:
        f = _resolve(ex.from_character_name, chars_by_name)
        t = _resolve(ex.to_character_name, chars_by_name)
        if f is None or t is None or f.id == t.id:
            continue
        repo.record_exchange(
            conn,
            ContextExchange(
                conversation_id=conv.id,
                from_character_id=f.id,
                to_character_id=t.id,
                exchange_type=ex.exchange_type,
                payload=ex.payload.model_dump(),
            ),
        )

    # --- Noteworthy moments -> postcards ----------------------------------
    for nm in out.noteworthy_moments:
        a = _resolve(nm.character_a_name, chars_by_name)
        b = _resolve(nm.character_b_name, chars_by_name)
        if a is None or b is None or a.id == b.id:
            continue
        repo.create_postcard(
            conn,
            character_a_id=a.id,
            character_b_id=b.id,
            summary=nm.summary,
            scene=nm.scene,
            conversation_id=conv.id,
        )

    # --- Audit event ------------------------------------------------------
    repo.record_world_event(
        conn,
        WorldEvent(
            conversation_id=conv.id,
            event_type="conversation_processed",
            summary=out.summary,
            payload={
                "topics": out.topics_discussed,
                "counts": {
                    "trait_updates": len(out.trait_updates),
                    "relationship_updates": len(out.relationship_updates),
                    "user_knowledge_updates": len(out.user_knowledge_updates),
                    "structured_exchanges": len(out.structured_exchanges),
                    "noteworthy_moments": len(out.noteworthy_moments),
                },
            },
        ),
    )
