"""Autonomous agent-to-agent conversation mode.

Agents talk among themselves while the user watches. After each turn,
a cheap "do you have something to say?" check runs against every other
agent — in parallel via a thread pool. Those who say yes speak. The
conversation ends when nobody has anything to add or a turn cap is hit.

The user can inject a message at any time by typing and pressing Enter.
The input loop is non-blocking — `nora.input_queue.StdinReader` runs a
daemon thread that pushes lines into a queue; the main loop polls between
turns. This pattern works identically on macOS, Linux, and Windows — no
`select`, no `fcntl`, no `termios`.

Typing `/stop` (or Ctrl+C) ends the session.
"""
from __future__ import annotations

import re
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from pydantic import BaseModel, Field

from . import chat as chat_mod
from . import config, input_queue, llm, repo
from .models import USER_ID, Character, Message


# Module-level stop flag set by SIGINT handler. Checked in the main loop.
# (Used in addition to try/except KeyboardInterrupt around LLM calls because
# some HTTP libraries swallow the exception during retries.)
_stop_requested = False
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# --- Output schemas for the cheap checks ---------------------------------

class WantsToSpeak(BaseModel):
    yes: bool = Field(description="True only if the agent would naturally jump in right now.")


class OpenerPick(BaseModel):
    speaker_name: str = Field(description="Name of the agent who should open. Must match exactly.")


# --- Prompts --------------------------------------------------------------

_WANTS_TEMPLATE = """You are {name}. You're in a group conversation, listening to the other agents. Decide if you would naturally jump in right now.

Say YES if any of these are true:
- You have a reaction, an opinion, or a take of your own to add — even a small one
- You want to ask one of the others a direct question
- You disagree (even gently) with something just said
- The topic touches your interests or experience and you'd naturally chime in
- It's the start of a conversation and you'd contribute your own take
- You'd build on what someone else said with a related thought

Say NO if:
- You'd literally just be repeating what someone else said
- The thread has truly run its course and silence is right
- You spoke very recently and stepping back is more natural
- Someone else clearly has more to say and you'd be cutting them off

Don't default. Be honest about whether you'd actually speak. In a fresh conversation on a topic that touches you, YES is usually right. In a wind-down, NO is right. Use your judgment.
"""


_OPENER_SYSTEM = """You pick which character opens a group conversation.

Pick the character whose interest most relates to the topic, or whose voice would naturally start it.
If there's no specific topic, pick whichever character seems most likely to break the silence first.
Output the chosen character's name exactly as given.
"""


# --- Public API -----------------------------------------------------------

OnUserMessage = Callable[[Message], None]
OnStatus = Callable[[str], None]
# print_speak is a factory: given a character about to speak, returns a
# context manager whose __enter__ yields the on_delta callback for streaming.
PrintSpeak = Callable[[Character], "object"]  # ContextManager[Callable[[str], None]]


def run(
    session: chat_mod.ChatSession,
    *,
    seed_topic: Optional[str] = None,
    max_turns: int = 20,
    print_speak: PrintSpeak,
    on_user_message: OnUserMessage = lambda m: None,
    on_status: OnStatus = lambda s: None,
) -> int:
    """Drive the autonomous loop. Returns total turn count produced.

    The session must already be created. Caller is responsible for closing
    the session and running the world engine afterward.

    Ctrl+C is caught via a SIGINT handler that sets a stop flag, so even if
    a long LLM call is in flight the loop exits at the next checkpoint.
    """
    global _stop_requested
    _stop_requested = False
    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    _drain_stdin()  # discard any leftover bytes from menu/prompt prior

    try:
        return _run_inner(
            session, seed_topic, max_turns,
            print_speak, on_user_message, on_status,
        )
    finally:
        signal.signal(signal.SIGINT, old_handler)


def _run_inner(
    session: chat_mod.ChatSession,
    seed_topic: Optional[str],
    max_turns: int,
    print_speak: PrintSpeak,
    on_user_message: OnUserMessage,
    on_status: OnStatus,
) -> int:
    if seed_topic:
        repo.add_message(
            session.conn,
            conversation_id=session.conv.id,
            sender_type="system",
            sender_id="autonomous",
            content=(
                f"You're in a group chat together. Loose topic: {seed_topic}. "
                "Anyone can open. Talk to each other, not to a moderator."
            ),
        )
    else:
        repo.add_message(
            session.conn,
            conversation_id=session.conv.id,
            sender_type="system",
            sender_id="autonomous",
            content=(
                "You're in a group chat together. No agenda. "
                "Anyone can open. Talk like real people who happen to be in the same room."
            ),
        )

    opener = _pick_opener(session, seed_topic)
    on_status(f"{opener.name} opens.")
    try:
        with print_speak(opener) as on_delta:
            msg = session.character_responds(opener, on_delta=on_delta)
    except KeyboardInterrupt:
        return 0
    last_speaker_id = opener.id
    turns = 1

    while turns < max_turns and not _stop_requested:
        # 1) Did the user type anything?
        user_line = _poll_input(timeout=0.05)
        if user_line is not None:
            if _is_stop(user_line):
                on_status("ended by user")
                break
            u_msg = repo.add_message(
                session.conn,
                conversation_id=session.conv.id,
                sender_type="user", sender_id=USER_ID, content=user_line,
            )
            on_user_message(u_msg)
            speakers = chat_mod.pick_next_speakers(session)
            for c in speakers:
                if turns >= max_turns or _stop_requested:
                    break
                try:
                    with print_speak(c) as on_delta:
                        session.character_responds(c, on_delta=on_delta)
                except KeyboardInterrupt:
                    on_status("ended by user")
                    return turns
                last_speaker_id = c.id
                turns += 1
            continue

        # 2) Otherwise, check who organically wants to speak.
        # Run all candidate checks in parallel — these are independent cheap
        # gpt-4o-mini calls and the sequential cost was the main dead-air gap.
        candidates = [c for c in session.characters if c.id != last_speaker_id]
        transcript = _build_transcript_tail(session, n=10)
        wants: list[Character] = []
        if candidates:
            try:
                wants = _check_wants_parallel(candidates, session.characters, transcript)
            except KeyboardInterrupt:
                on_status("ended by user")
                return turns

        if _stop_requested:
            break
        if not wants:
            on_status("no one wants to speak — ending")
            break

        for c in wants:
            if turns >= max_turns or _stop_requested:
                break
            try:
                with print_speak(c) as on_delta:
                    session.character_responds(c, on_delta=on_delta)
            except KeyboardInterrupt:
                on_status("ended by user")
                return turns
            last_speaker_id = c.id
            turns += 1

            # Inter-speaker poll for user injection.
            user_line = _poll_input(timeout=0.0)
            if user_line is not None:
                if _is_stop(user_line):
                    on_status("ended by user")
                    return turns
                u_msg = repo.add_message(
                    session.conn,
                    conversation_id=session.conv.id,
                    sender_type="user", sender_id=USER_ID, content=user_line,
                )
                on_user_message(u_msg)
                break  # outer loop will recompute wants with the new context

    if _stop_requested:
        on_status("ended by user")
    return turns


# --- Internals ------------------------------------------------------------

def _on_sigint(signum, frame):
    """SIGINT handler — sets the stop flag instead of raising.

    This is more reliable than try/except KeyboardInterrupt because some HTTP
    libraries swallow the exception during retries. The loop checks the flag
    at every checkpoint.
    """
    global _stop_requested
    _stop_requested = True


def _is_stop(text: str) -> bool:
    return text.strip().lower() in ("/stop", "/done", "/exit", "/quit")


def _poll_input(timeout: float = 0.0) -> Optional[str]:
    """Non-blocking input — returns clean text from the StdinReader queue or None.

    `timeout` is accepted for API symmetry with the old select-based version
    but ignored here: the queue read is instantaneous, and the daemon thread
    is the one that actually blocks on stdin.

    Skips empty lines, whitespace-only lines, and lines that are entirely
    terminal escape sequences (rare, but can leak in from menu rendering).
    """
    line = input_queue.get_reader().get_line_nowait()
    if line is None:
        return None
    text = _ANSI_RE.sub("", line).strip()
    if not text:
        return None
    return text


def _drain_stdin() -> None:
    """Discard all queued input — used at the start of autonomous mode so
    leftover keystrokes from the menu/picker don't get re-injected.
    """
    input_queue.get_reader().drain()


def _build_transcript_tail(session: chat_mod.ChatSession, n: int = 10) -> str:
    msgs = session.messages()[-n:]
    lines: list[str] = []
    for m in msgs:
        if m.sender_type == "user":
            label = "USER"
        elif m.sender_type == "character":
            c = session.chars_by_id.get(m.sender_id)
            label = c.name.upper() if c else "?"
        else:
            label = "SYSTEM"
        lines.append(f"{label}: {m.content}")
    return "\n".join(lines) if lines else "(empty)"


def _check_wants_parallel(
    candidates: list[Character],
    all_characters: list[Character],
    transcript: str,
) -> list[Character]:
    """Run wants-to-speak checks for all candidates in parallel.

    Preserves candidate order in the returned list so the speaking sequence
    is deterministic. Failures are silently treated as 'no' to keep the loop
    moving.
    """
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {
            pool.submit(_check_wants_to_speak, c, all_characters, transcript): c
            for c in candidates
        }
        for fut in futures:
            c = futures[fut]
            try:
                results[c.id] = fut.result()
            except Exception:
                results[c.id] = False
    return [c for c in candidates if results.get(c.id)]


def _check_wants_to_speak(
    character: Character,
    all_characters: list[Character],
    transcript: str,
) -> bool:
    others = ", ".join(
        f"{c.name} ({c.interest})" for c in all_characters if c.id != character.id
    )
    user_msg = (
        f"Other agents in the room: {others}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        f"Do you, as {character.name}, have something specific to say right now? "
        "Default to no unless you genuinely would speak."
    )
    with llm.usage_context(purpose="wants_to_speak"):
        out = llm.structured(
            model=config.models().moderator,
            system=_WANTS_TEMPLATE.format(name=character.name),
            user=user_msg,
            schema=WantsToSpeak,
            temperature=0.5,
        )
    return out.yes


def _pick_opener(
    session: chat_mod.ChatSession, seed_topic: Optional[str]
) -> Character:
    if not seed_topic:
        return session.characters[0]
    char_lines = "\n".join(
        f"  - {c.name}: {c.interest}" for c in session.characters
    )
    user = (
        f"Topic: {seed_topic}\n\n"
        f"Agents:\n{char_lines}\n\n"
        "Pick which agent opens."
    )
    try:
        with llm.usage_context(purpose="moderator"):
            pick = llm.structured(
                model=config.models().moderator,
                system=_OPENER_SYSTEM,
                user=user,
                schema=OpenerPick,
                temperature=0.3,
            )
        match = next(
            (c for c in session.characters if c.name.lower() == pick.speaker_name.strip().lower()),
            None,
        )
        if match:
            return match
    except Exception:
        pass
    return session.characters[0]
