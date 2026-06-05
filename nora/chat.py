"""Conversation runtime: 1:1 and multi-agent chat.

A `ChatSession` owns a conversation row and the live message log. It exposes
two operations: take input from the user, and have one or more agents
take their turn. Council mode is built on top of this in `nora.council`.

Multi-agent turn-taking uses a small moderator LLM call to pick which
agents speak next and in what order. The user can also direct turns
explicitly with @name.
"""
from __future__ import annotations

import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field

from . import config, db, debug, llm, repo, voice
from .models import (
    USER_ID,
    Character,
    Conversation,
    ConversationType,
    Message,
    Participant,
)


# --- Session ---------------------------------------------------------------

@dataclass
class ChatSession:
    conn: sqlite3.Connection
    conv: Conversation
    participants: list[Participant]
    characters: list[Character]
    council_topic: Optional[str] = None
    # Per-session state for the "agent wants to join" feature.
    requested_to_join_ids: set[str] = field(default_factory=set)
    pending_join: Optional[Character] = None

    @property
    def chars_by_id(self) -> dict[str, Character]:
        return {c.id: c for c in self.characters}

    @property
    def chars_by_name(self) -> dict[str, Character]:
        return {c.name.lower(): c for c in self.characters}

    def messages(self) -> list[Message]:
        return repo.list_messages(self.conn, self.conv.id)

    def user_says(self, content: str) -> Message:
        return repo.add_message(
            self.conn,
            conversation_id=self.conv.id,
            sender_type="user",
            sender_id=USER_ID,
            content=content.strip(),
        )

    def character_responds(
        self,
        character: Character,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Message:
        """Generate one character turn.

        If `on_delta` is provided, the underlying LLM call streams and each
        text chunk is passed to the callback. The full response is still
        written to the DB at the end.
        """
        from . import timing as _timing
        t = _timing.timer(f"character_responds({character.name})")

        sys_prompt = voice.build_system_prompt(
            self.conn,
            character=character,
            other_participants=[p for p in self.participants if not (
                p.type == "character" and p.id == character.id
            )],
            conversation_type=self.conv.type,
            council_topic=self.council_topic,
        )
        t.step("system prompt built")

        msgs = self.messages()
        t.step(f"messages loaded ({len(msgs)})")

        if self.conv.type == "1on1":
            llm_messages = self._format_1on1(msgs, character)
            user_addendum = None
        else:
            llm_messages, user_addendum = self._format_multi(msgs, character)
        t.step("messages formatted")

        if user_addendum is not None:
            llm_messages = [*llm_messages, {"role": "user", "content": user_addendum}]

        with llm.usage_context(conversation_id=self.conv.id, purpose="character"):
            t.step("llm.chat starting")
            text = llm.chat(
                model=config.models().character,
                system=sys_prompt,
                messages=llm_messages,
                temperature=0.7,
                max_tokens=200,
                on_delta=on_delta,
            )
            t.step("llm.chat returned")
        text = self._strip_self_label(text, character)
        t.step("text stripped")
        msg = repo.add_message(
            self.conn,
            conversation_id=self.conv.id,
            sender_type="character",
            sender_id=character.id,
            content=text,
        )
        t.step("message saved to db")
        t.done()
        return msg

    def end(self) -> None:
        repo.end_conversation(self.conn, self.conv.id)

    # --- internal --------

    def _format_1on1(
        self, msgs: list[Message], character: Character
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in msgs:
            if m.sender_type == "user":
                out.append({"role": "user", "content": m.content})
            elif m.sender_type == "character" and m.sender_id == character.id:
                out.append({"role": "assistant", "content": m.content})
            else:
                # System or other-character noise; rare in 1:1.
                out.append({"role": "user", "content": f"[{m.sender_type}] {m.content}"})
        return out

    def _format_multi(
        self, msgs: list[Message], character: Character
    ) -> tuple[list[dict[str, str]], str]:
        transcript_lines: list[str] = []
        for m in msgs:
            label = self._label(m)
            transcript_lines.append(f"{label}: {m.content}")
        transcript = "\n".join(transcript_lines) if transcript_lines else "(empty)"

        addendum = (
            "Conversation so far:\n"
            f"{transcript}\n\n"
            f"Now it is your turn. Reply as {character.name}. "
            "Output ONLY your line — no name prefix, no quotes, no narration."
        )
        return [], addendum

    def _label(self, m: Message) -> str:
        if m.sender_type == "user":
            return "USER"
        if m.sender_type == "character":
            c = self.chars_by_id.get(m.sender_id)
            return c.name.upper() if c else "CHARACTER"
        return "SYSTEM"

    def _strip_self_label(self, text: str, character: Character) -> str:
        # Models sometimes prefix their own name despite the instruction.
        text = text.strip()
        prefixes = [f"{character.name}:", f"{character.name.upper()}:", f"[{character.name}]"]
        for p in prefixes:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].lstrip()
        return text


# --- Session factories ----------------------------------------------------

def start_one_on_one(conn: sqlite3.Connection, character: Character) -> ChatSession:
    participants = [
        Participant(type="user", id=USER_ID),
        Participant(type="character", id=character.id),
    ]
    conv = repo.create_conversation(conn, type="1on1", participants=participants)
    return ChatSession(conn=conn, conv=conv, participants=participants, characters=[character])


def start_multi(
    conn: sqlite3.Connection, characters: list[Character]
) -> ChatSession:
    if len(characters) < 2:
        raise ValueError("multi-agent session needs at least 2 agents")
    participants = [
        Participant(type="user", id=USER_ID),
        *[Participant(type="character", id=c.id) for c in characters],
    ]
    conv = repo.create_conversation(conn, type="multi", participants=participants)
    return ChatSession(conn=conn, conv=conv, participants=participants, characters=characters)


def start_council(
    conn: sqlite3.Connection, characters: list[Character], topic: str
) -> ChatSession:
    if len(characters) < 2:
        raise ValueError("council needs at least 2 agents")
    participants = [
        Participant(type="user", id=USER_ID),
        *[Participant(type="character", id=c.id) for c in characters],
    ]
    conv = repo.create_conversation(
        conn, type="council", participants=participants, topic=topic
    )
    return ChatSession(
        conn=conn, conv=conv, participants=participants,
        characters=characters, council_topic=topic,
    )


def resume(conn: sqlite3.Connection, conversation_id: str) -> ChatSession:
    """Reopen an existing conversation as a live session.

    Re-derives participants and characters from the DB. Clears `ended_at`
    so the user can keep typing. The previous `processed_through_message_id`
    watermark is preserved so the next /done only re-processes new messages.
    """
    conv = repo.get_conversation(conn, conversation_id)
    parts = repo.list_participants(conn, conversation_id)
    chars: list[Character] = []
    for p in parts:
        if p.type == "character":
            chars.append(repo.get_character_by_id(conn, p.id))
    council_topic = conv.topic if conv.type == "council" else None

    if conv.ended_at is not None:
        conn.execute(
            "UPDATE conversations SET ended_at = NULL WHERE id = ?",
            (conv.id,),
        )

    return ChatSession(
        conn=conn, conv=conv, participants=parts,
        characters=chars, council_topic=council_topic,
    )


# --- Moderator: picks who speaks next -------------------------------------

class TurnPlan(BaseModel):
    speakers: list[str] = Field(
        description="Agent names in the order they should speak. May be empty if no one needs to respond."
    )


_MODERATOR_SYSTEM = """You are a turn-taking moderator for a multi-agent conversation.

Pick which agents speak next, in what order. ALWAYS pick at least one — every user turn gets a response.

KEY PRINCIPLE: include EVERY agent who would naturally have something to say. Don't cap at 2 — if all the agents in the room would chime in, pick all of them. Multi-agent means multi-agent.

How to decide:
- For a substantive prompt (a question, dilemma, choice, opinion request): include every agent whose interest, voice, or perspective would lead them to respond. With 3 agents, that's often all 3. With 5, it might be 4 or 5.
- For a prompt that touches multiple agents' interests, include ALL of them.
- Filter OUT an agent only if they truly wouldn't have anything to add — their interest is unrelated and they'd just be repeating someone else.
- If the user @-mentioned an agent, that agent speaks first; others can still follow.
- For pure small talk ("hi", "lol", "ok"), one agent is fine.

Order: most relevant or most-addressed first. Output agent names exactly as given.
"""


_MENTION_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_-]{0,40})")


def pick_next_speakers(session: ChatSession) -> list[Character]:
    """Decide which characters speak next, in order.

    Falls back gracefully: if the moderator returns names we don't recognize,
    they're filtered. If empty, defaults to letting the most-recently-not-spoken
    character take a turn.
    """
    msgs = session.messages()
    if not msgs:
        return []

    # First turn after the user opens (only one message in the log) — skip the
    # moderator call entirely. The most-relevant-by-interest pick happens via
    # round-robin (which on an empty history just returns characters in order).
    user_msgs = [m for m in msgs if m.sender_type == "user"]
    char_msgs = [m for m in msgs if m.sender_type == "character"]
    if len(user_msgs) == 1 and len(char_msgs) == 0:
        return _round_robin_fallback(session)

    last_user = next(
        (m for m in reversed(msgs) if m.sender_type == "user"), None
    )

    # Honor @mentions in the most recent user message.
    if last_user:
        mentioned: list[Character] = []
        for raw in _MENTION_RE.findall(last_user.content):
            c = session.chars_by_name.get(raw.lower())
            if c and c not in mentioned:
                mentioned.append(c)
        if mentioned:
            return mentioned

    # Build moderator input.
    transcript = "\n".join(
        f"{session._label(m)}: {m.content}" for m in msgs[-6:]
    )
    char_lines = "\n".join(
        f"  - {c.name} (interest: {c.interest})" for c in session.characters
    )

    user_text = (
        f"Agents in the room:\n{char_lines}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        "Decide who speaks next."
    )

    try:
        with llm.usage_context(conversation_id=session.conv.id, purpose="moderator"):
            plan = llm.structured(
                model=config.models().moderator,
                system=_MODERATOR_SYSTEM,
                user=user_text,
                schema=TurnPlan,
                temperature=0.5,
            )
    except Exception:
        # Moderator is best-effort. Fall back to round-robin.
        return _round_robin_fallback(session)

    speakers: list[Character] = []
    for name in plan.speakers:
        c = session.chars_by_name.get(name.strip().lower())
        if c and c not in speakers:
            speakers.append(c)
    # Guarantee at least one speaker per user turn.
    if not speakers:
        speakers = _round_robin_fallback(session)
    return speakers


def _round_robin_fallback(session: ChatSession) -> list[Character]:
    """Pick the agent who has spoken least recently."""
    msgs = session.messages()
    last_spoke: dict[str, int] = {c.id: -1 for c in session.characters}
    for i, m in enumerate(msgs):
        if m.sender_type == "character":
            last_spoke[m.sender_id] = i
    ordered = sorted(session.characters, key=lambda c: last_spoke[c.id])
    return ordered[:1]


# --- Character "wants to join" check (cheap moderator-model call) --------

class JoinInterest(BaseModel):
    interested: bool = Field(
        description="True only if the absent agent has a strong, on-topic reason to join the conversation."
    )


JOIN_CHECK_INTERVAL = 4         # check every N user messages
JOIN_CHECK_TAIL = 8             # recent messages (user + agent mixed) the moderator sees
JOIN_MENTION_LOOKBACK = 4       # name-mention fast-path scans the last N user messages


_JOIN_SYSTEM = """You decide whether {name} would have a strong, on-topic reason to join a conversation in progress.

You will see {name}'s interest and the recent transcript (the most recent ~8 messages of the conversation, both user and agents). Read the WHOLE trajectory — where the conversation has been going matters more than just the last line.

Say YES if the conversation has been moving toward {name}'s domain, or if the participants are wrestling with something where {name}'s perspective would substantively help.

Say NO if the topic is outside {name}'s wheelhouse, or if their contribution would just be polite small talk, or if they'd be chiming in for the sake of chiming in.

Default toward NO when uncertain. Joining a conversation should feel earned."""


# Word-boundary, case-insensitive name match.
def _user_mentioned_in_recent(name: str, recent_user_messages: list[Message]) -> bool:
    """Did the user mention `name` in any of the recent user messages?

    Word-boundary, case-insensitive. So "Charlie" matches in "Charlie's view"
    but not inside "uncharitable". Multi-word names work too — `\\b` anchors
    on each end of the whole pattern.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    return any(pattern.search(m.content or "") for m in recent_user_messages)


def _check_wants_to_join(
    absent: Character, in_room: list[Character], transcript: str
) -> bool:
    others = ", ".join(f"{c.name} ({c.interest})" for c in in_room) or "(no one)"
    user_msg = (
        f"You are {absent.name}. Your interest: {absent.interest}.\n\n"
        f"Agents currently in the room: {others}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        "Would you have a strong, on-topic reason to join this conversation right now?"
    )
    with llm.usage_context(purpose="wants_to_join"):
        out = llm.structured(
            model=config.models().moderator,
            system=_JOIN_SYSTEM.format(name=absent.name),
            user=user_msg,
            schema=JoinInterest,
            temperature=0.4,
        )
    return out.interested


@dataclass
class JoinCheckResult:
    """What `maybe_request_join` decided, with enough detail to display.

    `yes_votes` is the union of all interested characters. `mentioned` is the
    subset that bypassed the moderator because the user named them in a
    recent message — the strongest possible signal of interest.
    """
    ran: bool
    skip_reason: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    mentioned: list[str] = field(default_factory=list)
    yes_votes: list[str] = field(default_factory=list)
    no_votes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requester: Optional[Character] = None


def maybe_request_join(session: ChatSession) -> JoinCheckResult:
    """If conditions are met, check absent characters and pick one to request a join.

    Returns a JoinCheckResult with audit info (whether we ran, who was checked,
    who said yes/no, who got picked). The caller is free to display this for
    diagnostic purposes.

    Conditions to actually run:
      - Not in council mode (council membership is intentional, not open)
      - User-message count is a positive multiple of JOIN_CHECK_INTERVAL
      - At least one absent character hasn't already requested in this session
      - The cheap moderator-model check returns yes for at least one

    Set NORA_DEBUG=1 (or use launcher --debug) for stderr traces.
    """
    debug.log("join", f"called for conv={session.conv.id[:8]} type={session.conv.type}")

    if session.conv.type == "council":
        debug.log("join", "skip: council mode (membership is intentional)")
        return JoinCheckResult(ran=False, skip_reason="council mode")

    msgs = session.messages()
    user_count = sum(1 for m in msgs if m.sender_type == "user")
    debug.log(
        "join",
        f"user_msg_count={user_count} (trigger every {JOIN_CHECK_INTERVAL})"
    )
    if user_count == 0:
        debug.log("join", "skip: no user messages yet")
        return JoinCheckResult(ran=False, skip_reason="no user messages yet")
    if user_count % JOIN_CHECK_INTERVAL != 0:
        next_at = ((user_count // JOIN_CHECK_INTERVAL) + 1) * JOIN_CHECK_INTERVAL
        debug.log("join", f"skip: not at interval (next check at user msg #{next_at})")
        return JoinCheckResult(ran=False, skip_reason=f"next check at user msg #{next_at}")

    in_room_ids = {c.id for c in session.characters}
    debug.log(
        "join",
        f"in_room: {[c.name for c in session.characters]}  "
        f"already_requested: {sorted(session.requested_to_join_ids)}"
    )

    with db.connect() as conn:
        all_chars = repo.list_characters(conn)
    absent = [
        c for c in all_chars
        if c.id not in in_room_ids and c.id not in session.requested_to_join_ids
    ]
    debug.log("join", f"absent candidates: {[c.name for c in absent]}")

    if not absent:
        debug.log("join", "skip: no absent candidates left to consider")
        return JoinCheckResult(ran=False, skip_reason="no absent candidates left")

    # Split absent into two groups:
    #   - mentioned_chars: user named them in the last N user messages → auto-yes,
    #     skip the moderator call (explicit user intent beats inferred relevance)
    #   - to_check: everyone else → moderator decides based on transcript trajectory
    recent_user = [m for m in msgs if m.sender_type == "user"][-JOIN_MENTION_LOOKBACK:]
    mentioned_chars: list[Character] = []
    to_check: list[Character] = []
    for c in absent:
        if _user_mentioned_in_recent(c.name, recent_user):
            mentioned_chars.append(c)
            debug.log("join", f"  {c.name}: YES (name-mention fast-path)")
        else:
            to_check.append(c)

    transcript = "\n".join(
        f"{session._label(m)}: {m.content}" for m in msgs[-JOIN_CHECK_TAIL:]
    )

    yes_votes: list[str] = [c.name for c in mentioned_chars]
    no_votes: list[str] = []
    errors: list[str] = []
    moderator_yes: list[Character] = []

    if to_check:
        debug.log("join", f"running parallel moderator check ({len(to_check)} candidates)")
        with ThreadPoolExecutor(max_workers=len(to_check)) as pool:
            futures = {
                pool.submit(_check_wants_to_join, c, session.characters, transcript): c
                for c in to_check
            }
            for fut in futures:
                c = futures[fut]
                try:
                    yes = fut.result()
                    debug.log("join", f"  {c.name}: {'YES' if yes else 'no'} (moderator)")
                    if yes:
                        yes_votes.append(c.name)
                        moderator_yes.append(c)
                    else:
                        no_votes.append(c.name)
                except Exception as e:
                    debug.log("join", f"  {c.name}: ERROR {type(e).__name__}: {e}")
                    errors.append(f"{c.name}: {type(e).__name__}: {e}")
    else:
        debug.log("join", "all candidates fast-pathed via name mention; no moderator call needed")

    result = JoinCheckResult(
        ran=True,
        candidates=[c.name for c in absent],
        mentioned=[c.name for c in mentioned_chars],
        yes_votes=yes_votes,
        no_votes=no_votes,
        errors=errors,
    )

    # Mentioned characters take priority — explicit user request is the strongest
    # signal possible. Within each tier we keep the candidate-list order.
    interested = mentioned_chars + moderator_yes

    if not interested:
        debug.log("join", "no character interested — no request this round")
        return result

    requester = interested[0]
    via = "mention" if requester in mentioned_chars else "moderator"
    debug.log(
        "join",
        f"REQUEST: {requester.name} via {via} (interest: {requester.interest})"
        + (f"  · also interested: {[c.name for c in interested[1:]]}" if len(interested) > 1 else "")
    )
    session.requested_to_join_ids.add(requester.id)
    session.pending_join = requester
    result.requester = requester
    return result


# --- Join / leave session helpers ----------------------------------------

def add_character_to_session(
    session: ChatSession, character: Character
) -> None:
    """Add a character to a live session. Persists the participant row.

    If the session was 1on1, promotes it to multi so the formatter switches.
    Idempotent — does nothing if the character is already in the room.
    """
    if any(c.id == character.id for c in session.characters):
        return
    session.characters.append(character)
    session.participants.append(Participant(type="character", id=character.id))
    session.conn.execute(
        "INSERT OR IGNORE INTO conversation_participants "
        "(conversation_id, participant_type, participant_id) VALUES (?, ?, ?)",
        (session.conv.id, "character", character.id),
    )
    if session.conv.type == "1on1":
        session.conv = session.conv.model_copy(update={"type": "multi"})
        session.conn.execute(
            "UPDATE conversations SET type = 'multi' WHERE id = ?",
            (session.conv.id,),
        )


def remove_character_from_session(
    session: ChatSession, name: str
) -> Optional[Character]:
    """Remove a character by name. Returns the removed Character, or None.

    Their messages stay in the DB so the world engine still processes them.
    The participant row is removed so future renders don't list them.
    """
    target = next(
        (c for c in session.characters if c.name.lower() == name.strip().lower()),
        None,
    )
    if target is None:
        return None
    session.characters = [c for c in session.characters if c.id != target.id]
    session.participants = [
        p for p in session.participants
        if not (p.type == "character" and p.id == target.id)
    ]
    session.conn.execute(
        "DELETE FROM conversation_participants "
        "WHERE conversation_id = ? AND participant_type = 'character' AND participant_id = ?",
        (session.conv.id, target.id),
    )
    return target
