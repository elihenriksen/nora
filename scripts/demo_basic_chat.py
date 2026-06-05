"""End-to-end smoke test.

Creates two characters, runs a short multi-character conversation via the
real LLM backend, runs the World Engine, then prints what was learned.

This is a paid test (it makes OpenAI calls). Run only when wiring changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `python tests/smoke.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import chat as chat_mod
from nora import db, repo, world_engine, inspect_view


console = Console()


def reset_db() -> None:
    db_path = Path(__file__).resolve().parent.parent / "nora.db"
    if db_path.exists():
        db_path.unlink()
    db.init_db()


def build_world() -> tuple[str, str]:
    with db.transaction() as conn:
        charlie = repo.create_character(
            conn,
            name="Charlie",
            interest="fishing in still freshwater",
            seed_traits=[
                "Speaks in short sentences.",
                "Believes the best moves come from waiting.",
                "Reads the surface of the water before anything else.",
            ],
            backstory="Grew up on small lakes in northern Wisconsin.",
            voice_notes="Plain and unhurried. Uses fishing analogies.",
        )
        candy = repo.create_character(
            conn,
            name="Candy",
            interest="ocean currents and tides",
            seed_traits=[
                "Energetic, fast-talking.",
                "Trusts momentum over stillness.",
                "Sees everything as a system in motion.",
            ],
            backstory="Marine science background; spent years on research vessels.",
            voice_notes="Animated. Uses kinetic verbs. Sometimes finishes thoughts mid-air.",
        )
        return charlie.id, candy.id


def run_multi_chat(charlie_id: str, candy_id: str) -> str:
    with db.transaction() as conn:
        charlie = repo.get_character_by_id(conn, charlie_id)
        candy = repo.get_character_by_id(conn, candy_id)
        session = chat_mod.start_multi(conn, [charlie, candy])

    user_lines = [
        "I keep getting stuck on a decision. I have a chance to leave my job for something exciting but I'm scared.",
        "What is more important: waiting until the right moment, or moving fast on something that calls to me?",
        "What would each of you do?",
    ]

    console.print(f"[dim]Session id: {session.conv.id}[/dim]\n")

    for line in user_lines:
        console.print(f"[bold green]you[/bold green]: {line}")
        with db.transaction() as conn:
            session.conn = conn
            session.user_says(line)
            speakers = chat_mod.pick_next_speakers(session)
            if not speakers:
                speakers = chat_mod._round_robin_fallback(session)
            for c in speakers:
                msg = session.character_responds(c)
                console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}")
        console.print()

    with db.transaction() as conn:
        session.conn = conn
        session.end()

    return session.conv.id


def process(conv_id: str) -> None:
    with db.transaction() as conn:
        out = world_engine.process_conversation(conn, conv_id)
    console.rule("World Engine output")
    console.print(f"[bold]Summary:[/bold] {out.summary}")
    console.print(f"[dim]Topics:[/dim] {', '.join(out.topics_discussed)}")
    console.print(f"[dim]Trait updates:[/dim] {len(out.trait_updates)}")
    for t in out.trait_updates:
        console.print(f"  - {t.character_name} on '{t.topic}': {t.perspective} (★{t.strength})")
    console.print(f"[dim]Relationship updates:[/dim] {len(out.relationship_updates)}")
    for r in out.relationship_updates:
        console.print(f"  - {r.character_a_name}↔{r.character_b_name} on '{r.topic}'")
    console.print(f"[dim]User knowledge updates:[/dim] {len(out.user_knowledge_updates)}")
    for uk in out.user_knowledge_updates:
        console.print(f"  - {uk.character_name} on '{uk.topic}': {uk.character_view_of_user or uk.user_view_or_fact}")
    console.print(f"[dim]Structured exchanges:[/dim] {len(out.structured_exchanges)}")
    for ex in out.structured_exchanges:
        console.print(f"  - {ex.from_character_name}→{ex.to_character_name} ({ex.exchange_type}): {ex.payload.topic} :: {ex.payload.perspective}")
    console.print(f"[dim]Postcards:[/dim] {len(out.noteworthy_moments)}")
    for nm in out.noteworthy_moments:
        console.print(f"  - {nm.character_a_name} & {nm.character_b_name}: {nm.summary}")


def show_state() -> None:
    with db.connect() as conn:
        console.rule("Charlie")
        c = repo.get_character_by_name(conn, "Charlie")
        inspect_view.show_character(conn, c, console)
        console.rule("Candy")
        c = repo.get_character_by_name(conn, "Candy")
        inspect_view.show_character(conn, c, console)
        console.rule("Pair")
        a = repo.get_character_by_name(conn, "Charlie")
        b = repo.get_character_by_name(conn, "Candy")
        inspect_view.show_pair(conn, a, b, console)


def main() -> None:
    console.print("[bold]Resetting DB[/bold]")
    reset_db()
    console.print("[bold]Building world[/bold]")
    charlie_id, candy_id = build_world()
    console.print("[bold]Running multi-character chat[/bold]\n")
    conv_id = run_multi_chat(charlie_id, candy_id)
    console.print("\n[bold]Running World Engine[/bold]\n")
    process(conv_id)
    console.print("\n[bold]Final state[/bold]\n")
    show_state()


if __name__ == "__main__":
    main()
