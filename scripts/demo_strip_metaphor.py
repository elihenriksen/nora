"""Diagnostic: prove that the water metaphors come from the seed, not the engine.

Builds a 'plain Charlie' clone with the same voice but NO mention of fishing
analogies in voice_notes and seed traits stripped of water imagery.
Compares his answer to the same prompt against current Charlie.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import config, llm, voice
from nora import db, repo
from nora.models import USER_ID, Character, Participant


console = Console()

PROMPT = "what if I'm not ready"


def main() -> None:
    with db.connect() as conn:
        original_charlie = repo.get_character_by_name(conn, "Charlie")
        original_sys = voice.build_system_prompt(
            conn, character=original_charlie,
            other_participants=[Participant(type="user", id=USER_ID)],
            conversation_type="1on1",
        )

    # Build a stripped version in-memory only.
    plain_charlie = Character(
        id="ephemeral-plain",
        name="Charlie",
        interest="thinking before acting",
        seed_traits=[
            "Speaks in short sentences.",
            "Skeptical of urgency.",
            "Patient by temperament.",
        ],
        backstory="Grew up in a small town in Wisconsin.",
        voice_notes="Plain and unhurried. Says less rather than more.",
    )
    # build prompt manually since plain_charlie isn't in DB
    parts = [
        f"You are {plain_charlie.name}.",
        f"Your central interest: {plain_charlie.interest}.",
        f"Background: {plain_charlie.backstory}",
        f"How you speak: {plain_charlie.voice_notes}",
        "Your initial traits:",
        *[f"  - {t}" for t in plain_charlie.seed_traits],
        "",
        "Match the register of what's said to you. Default to short — one or two sentences.",
        "Don't reach for analogies. Don't moralize. Speak plainly.",
        "Avoid: starting with Ah/Oh/Hmm, fishing or water imagery, 'just like X', wrapping with a moral.",
    ]
    plain_sys = "\n".join(parts)

    console.rule("[bold]Original Charlie (with seed metaphors)[/bold]")
    console.print(llm.chat(
        model=config.models().character,
        system=original_sys,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=200,
    ))

    console.rule("[bold]Plain Charlie (metaphor-free seed)[/bold]")
    console.print(llm.chat(
        model=config.models().character,
        system=plain_sys,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=200,
    ))


if __name__ == "__main__":
    main()
