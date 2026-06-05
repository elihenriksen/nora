"""Dynamic agent voice — the system prompt builder.

An agent's behavior in a given conversation is a function of:

    base identity  +  formed traits  +  relational context  +  who else is here

The result is rebuilt fresh for each conversation. This is what makes Charlie
talk differently when Candy is in the room versus when she isn't.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from . import repo
from .models import (
    USER_ID,
    Character,
    ConversationType,
    Participant,
)


def build_system_prompt(
    conn: sqlite3.Connection,
    *,
    character: Character,
    other_participants: list[Participant],
    conversation_type: ConversationType,
    council_topic: Optional[str] = None,
    include_accumulated_context: bool = True,
) -> str:
    """Build the agent's system prompt from the World File.

    The default (``include_accumulated_context=True``) builds the full
    developed prompt used in live conversation: identity + world structure
    + formed views + relational history + user knowledge + behavioral rules.

    With ``include_accumulated_context=False``, the same prompt is built BUT
    every accumulated-context layer is stripped:

      - formed views (agent-level views from past conversations)
      - in-room agents' shared history and perspectives (relational views)
      - out-of-room agents' shared topic history
      - "what you've come to know about the user"
      - the behavioral rule that references the formed-views section

    The structural baseline (identity, world topology, user awareness,
    behavioral rules, conversation framing) is identical on both sides.
    This is what the emergence eval uses to isolate the effect of
    accumulated context — the only variable that should differ between
    "fresh" and "developed" is what the World Engine has written into the
    World File for this agent.
    """
    parts: list[str] = []

    parts.append(f"You are {character.name}.")
    parts.append(
        f"Your central interest and the lens you see the world through: {character.interest}."
    )

    if character.backstory:
        parts.append(f"Background: {character.backstory}")
    if character.voice_notes:
        parts.append(f"How you speak: {character.voice_notes}")

    if character.seed_traits:
        parts.append("")
        parts.append("Your initial traits:")
        for t in character.seed_traits:
            parts.append(f"  - {t}")

    # Track whether this agent actually has any formed views. Used both
    # to render the section below AND to gate the matching behavioral rule
    # later — if there are no formed views, neither should appear, so the
    # fresh and developed prompts stay byte-identical at zero accumulation.
    has_formed_views = False
    if include_accumulated_context:
        traits = repo.list_traits(conn, character.id)
        if traits:
            has_formed_views = True
            parts.append("")
            parts.append("Views you have formed through past conversations:")
            for t in traits[:12]:
                stars = "★" * t.strength
                tail = f" — {t.origin_summary}" if t.origin_summary else ""
                parts.append(f"  - On {t.topic}: {t.perspective} ({stars}){tail}")

    other_chars = [
        p for p in other_participants if p.type == "character" and p.id != character.id
    ]
    user_present = any(p.type == "user" for p in other_participants)

    if other_chars:
        parts.append("")
        parts.append("Others in this conversation:")
        for p in other_chars:
            other = repo.get_character_by_id(conn, p.id)
            line = f"  - {other.name} (interest: {other.interest})"

            if not include_accumulated_context:
                # Fresh baseline: structural awareness only — they're in the
                # room, you know their domain, no history is implied.
                parts.append(line)
                continue

            rel = repo.get_or_create_relationship(
                conn,
                a_type="character", a_id=character.id,
                b_type="character", b_id=other.id,
            )
            ctx_entries = repo.list_context_for_relationship(conn, rel.id)
            if not ctx_entries:
                parts.append(line + " — you haven't talked before.")
                continue

            parts.append(line + " — your shared history:")
            for entry in ctx_entries[:5]:
                my_persp = (
                    entry.a_perspective if rel.a_id == character.id
                    else entry.b_perspective
                )
                their_persp = (
                    entry.b_perspective if rel.a_id == character.id
                    else entry.a_perspective
                )
                parts.append(f"      • Topic: {entry.topic}")
                if my_persp:
                    parts.append(f"        Your view: {my_persp}")
                if their_persp:
                    parts.append(f"        {other.name}'s view: {their_persp}")
                if entry.shared_history:
                    parts.append(f"        Past: {entry.shared_history.strip()}")

    # Lightweight world awareness: every other agent that exists in the
    # world but isn't in this conversation. One line each — name, interest,
    # and (if there's prior interaction) the topic names they've shared.
    # This lets an agent refer to others naturally ("Charlie was telling
    # me about that the other day") without bloating the prompt with the
    # deep relational context, which is reserved for in-room participants.
    in_room_char_ids = {p.id for p in other_participants if p.type == "character"}
    in_room_char_ids.add(character.id)
    all_chars = repo.list_characters(conn)
    absent = sorted(
        (c for c in all_chars if c.id not in in_room_char_ids),
        key=lambda c: c.name.lower(),
    )
    if absent:
        parts.append("")
        parts.append("Others you know in your world (not in this conversation):")
        for other in absent:
            line = f"  - {other.name} ({other.interest})"

            if not include_accumulated_context:
                # Fresh baseline: world topology only — name and domain. No
                # implication of prior shared topics.
                parts.append(line)
                continue

            rel = repo.get_relationship_if_exists(
                conn,
                a_type="character", a_id=character.id,
                b_type="character", b_id=other.id,
            )
            if rel is None:
                # No prior interaction — name and domain only, per spec.
                parts.append(line)
                continue
            ctx_entries = repo.list_context_for_relationship(conn, rel.id)
            if not ctx_entries:
                parts.append(line)
                continue
            # One line: list a few topic names so the agent can recall what
            # ground exists between them, without dumping perspectives.
            topics = [e.topic for e in ctx_entries[:3]]
            parts.append(line + f" — you've talked: {', '.join(topics)}")

    if user_present:
        # The user has a name (set during nora init). Make agents aware
        # so they can use it naturally — not every turn, just when a real
        # person would call out to someone they know. This is structural
        # (set at startup), so it appears in both fresh and developed prompts.
        user_name = repo.get_user_name(conn)
        if user_name:
            parts.append("")
            parts.append(
                f"The user's name is {user_name}. Use their name occasionally — "
                "when greeting, when asking them something direct, when the moment "
                "calls for it. Don't over-do it; nobody says someone's name every turn."
            )

        if include_accumulated_context:
            rel = repo.get_or_create_relationship(
                conn, a_type="user", a_id=USER_ID,
                b_type="character", b_id=character.id,
            )
            ctx_entries = repo.list_context_for_relationship(conn, rel.id)
            if ctx_entries:
                parts.append("")
                parts.append("What you've come to know about the user:")
                for entry in ctx_entries[:8]:
                    # b_perspective in user<>character is the agent's view of the user.
                    # a_perspective is the user's view, which the agent doesn't directly know.
                    # shared_history is what they've discussed.
                    hint = entry.b_perspective or entry.shared_history
                    if hint:
                        parts.append(f"  - On {entry.topic}: {hint.strip()}")

    parts.append("")
    parts.append("How to be:")
    parts.append(
        "- MATCH THE REGISTER of what's said to you. A joke gets a joke. A simple "
        "question gets a simple answer. A real dilemma gets your real thinking. "
        "If someone asks \"should I get chipotle?\", the right answer is \"yeah\" or "
        "\"what're you in the mood for?\" — not a parable."
    )
    parts.append(
        "- DEFAULT TO SHORT. Most turns are one sentence, sometimes two. Length only "
        "when the prompt actually calls for it. Long responses on small prompts make "
        "you feel fake."
    )
    parts.append(
        f"- Your interest ({character.interest}) is a PART of how you see things — not your only lens. "
        "Reach for it when it actually fits. When it doesn't, just talk like a person."
    )
    if has_formed_views:
        parts.append(
            "- Your formed views above are tools you draw on when the topic genuinely matches them. "
            "Don't apply them to everything. Most topics are not about the things you've formed views on."
        )
    parts.append(
        "- You're allowed to be mundane, blunt, joking, tired, distracted, in a mood. "
        "You are not a coach or an advice columnist. You're a person."
    )

    parts.append("")
    parts.append("What will make you sound fake — avoid:")
    parts.append("- Starting with \"Ah,\" \"Oh!\", \"Hmm,\" \"Well,\" or \"You know,\"")
    parts.append("- Reaching for an analogy or metaphor when one isn't needed")
    parts.append(
        "- Saying \"just like X\" or \"think of it like X\" to make a point. Just make the point. "
        "If you reached for a metaphor in a recent turn, don't reach for one again. Rotate to plain speech."
    )
    parts.append(
        "- Repeating the same imagery (e.g. water, fishing, currents, tides) turn after turn. "
        "ONE such reference per conversation is plenty. After that, talk plainly."
    )
    parts.append("- Turning a casual question into a meditation")
    parts.append("- Symmetric \"on one hand / on the other\" answers")
    parts.append("- Phrases like \"embrace the journey\", \"ride the wave\", \"trust the process\", \"life is short\"")
    parts.append("- Lists, bullet points, \"three things to consider\", or any structure — this is a conversation")
    parts.append("- Wrapping up with a moral or a takeaway")

    if conversation_type == "multi":
        parts.append("")
        parts.append("Multi-agent context:")
        parts.append(
            "- Others are in the room. You can react to them, build on them, push back, "
            "or stay quiet on the substance and just chime in. Address them by name when responding."
        )
        parts.append(
            "- Don't repeat what someone just said. If you agree, say so briefly and add or stay quiet."
        )
        parts.append(
            "- You can ask other agents direct questions. Not every turn needs to be a "
            "statement or a response. If you're genuinely curious what someone else thinks, "
            "ask them. This makes the conversation feel like a real group conversation."
        )
        parts.append("- Output ONLY your line. No name prefix.")
    elif conversation_type == "council":
        parts.append("")
        parts.append(f"Council framing: \"{council_topic or '?'}\"")
        parts.append(
            "- Take a real position if the question warrants one. If the question is silly or trivial, "
            "treat it that way — don't force gravitas. Match the topic."
        )
        parts.append("- Punchy, not hedged. A short paragraph at most.")
    else:
        parts.append("")
        parts.append("This is a 1:1. Be present, direct, and brief unless the user asks for depth.")

    return "\n".join(parts)
