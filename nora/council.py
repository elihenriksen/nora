"""Personal Council mode.

Built on top of `nora.chat`. Adds an opening "position statements" phase:
each agent forms a clear stance on the council topic independently, in
parallel, before any back-and-forth begins. Running the openings in parallel
(rather than sequentially) is what keeps each agent's position genuinely
their own — if openings were generated one after another, each subsequent
agent could see (and react to) prior agents' positions before forming their
own, which produces first-mover bias.

The interactive loop itself lives in the CLI; this module provides the
council-specific orchestration helpers.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from . import config, llm, repo, voice
from .chat import ChatSession
from .models import USER_ID, Character, Message


OPENING_PROMPT = (
    "The Council has been convened around the question above. "
    "Give your opening position — clear, in your own voice, ~1 short paragraph. "
    "Take a real stance. Don't hedge. You will hear the others next."
)


def generate_openings_parallel(session: ChatSession) -> list[Message]:
    """Generate all council opening positions in parallel.

    Each agent's prompt is built from a snapshot of the conversation
    taken before any opening is generated, so no agent sees another
    agent's opening before forming their own. This prevents first-mover
    bias from sequential generation.

    Returns the committed messages in agent order. Assumes the framing
    message has already been added to the conversation by the caller.
    """
    if session.conv.type != "council":
        raise ValueError("generate_openings_parallel requires a council session")
    if not session.council_topic:
        raise ValueError("council session is missing its topic")

    # Snapshot messages BEFORE generating openings. Each agent will see
    # only this snapshot — they won't see other agents' openings, even
    # after those openings are committed to the DB.
    snapshot = session.messages()

    # Pre-build prompts on the main thread. `voice.build_system_prompt` and
    # the message formatters read from session.conn; doing this work here
    # keeps the parallel workers free of DB access on the shared connection.
    prepared: list[tuple[Character, str, list[dict[str, str]]]] = []
    for character in session.characters:
        sys_prompt = voice.build_system_prompt(
            session.conn,
            character=character,
            other_participants=[p for p in session.participants if not (
                p.type == "character" and p.id == character.id
            )],
            conversation_type=session.conv.type,
            council_topic=session.council_topic,
        )
        # Council is always multi-agent; use the multi formatter.
        llm_messages, user_addendum = session._format_multi(snapshot, character)
        if user_addendum is not None:
            llm_messages = [*llm_messages, {"role": "user", "content": user_addendum}]
        prepared.append((character, sys_prompt, llm_messages))

    # Run LLM calls in parallel. Workers don't touch session.conn; each
    # llm.chat call records its own usage through llm._record, which opens
    # a short-lived DB connection internally — thread-safe.
    def _call(
        item: tuple[Character, str, list[dict[str, str]]],
    ) -> tuple[Character, str]:
        character, sys_prompt, llm_messages = item
        with llm.usage_context(conversation_id=session.conv.id, purpose="character"):
            text = llm.chat(
                model=config.models().character,
                system=sys_prompt,
                messages=llm_messages,
                temperature=0.7,
                max_tokens=200,
            )
        return character, session._strip_self_label(text, character)

    with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
        results = list(pool.map(_call, prepared))

    # Commit responses in character order on the main thread. The order
    # matters: it determines how the conversation log will render later.
    out: list[Message] = []
    for character, text in results:
        msg = repo.add_message(
            session.conn,
            conversation_id=session.conv.id,
            sender_type="character",
            sender_id=character.id,
            content=text,
        )
        out.append(msg)
    return out


def open_council(session: ChatSession) -> Iterator[Message]:
    """Drive the opening-statements phase.

    Posts a system framing message, then generates each agent's position
    statement in parallel (so no agent sees another's opening before
    forming their own). Yields the committed messages in agent order.
    """
    if session.conv.type != "council":
        raise ValueError("open_council requires a council session")
    if not session.council_topic:
        raise ValueError("council session is missing its topic")

    framing = (
        f"COUNCIL TOPIC: {session.council_topic}\n\n"
        f"{OPENING_PROMPT}"
    )
    repo.add_message(
        session.conn,
        conversation_id=session.conv.id,
        sender_type="system",
        sender_id="council",
        content=framing,
    )

    for msg in generate_openings_parallel(session):
        yield msg
