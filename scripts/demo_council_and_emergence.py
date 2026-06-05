"""Extra end-to-end checks: council mode + emergence eval.

Assumes `tests/smoke.py` has already run and Charlie + Candy exist with
formed views. Runs a short council and an emergence comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import chat as chat_mod
from nora import council, db, emergence, repo, world_engine


console = Console()


def test_council() -> None:
    console.rule("[bold]Council demo[/bold]")
    with db.transaction() as conn:
        charlie = repo.get_character_by_name(conn, "Charlie")
        candy = repo.get_character_by_name(conn, "Candy")
        if not charlie or not candy:
            raise SystemExit("Run tests/smoke.py first to seed Charlie + Candy.")
        session = chat_mod.start_council(
            conn,
            [charlie, candy],
            topic="Is it braver to commit fully to one path, or to keep doors open?",
        )

    # Opening statements
    with db.transaction() as conn:
        session.conn = conn
        for msg in council.open_council(session):
            c = session.chars_by_id[msg.sender_id]
            console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}")
            console.print()

    # One user turn pushing back
    with db.transaction() as conn:
        session.conn = conn
        user_msg = "But what if I genuinely don't know which path I want?"
        console.print(f"[bold green]you[/bold green]: {user_msg}")
        session.user_says(user_msg)
        speakers = chat_mod.pick_next_speakers(session)
        if not speakers:
            speakers = chat_mod._round_robin_fallback(session)
        for c in speakers:
            msg = session.character_responds(c)
            console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}")
        session.end()

    with db.transaction() as conn:
        out = world_engine.process_conversation(conn, session.conv.id)
    console.print(f"\n[dim]Engine summary:[/dim] {out.summary}")
    console.print(
        f"[dim]traits {len(out.trait_updates)} | "
        f"rels {len(out.relationship_updates)} | "
        f"exchanges {len(out.structured_exchanges)} | "
        f"postcards {len(out.noteworthy_moments)}[/dim]"
    )


def test_emergence() -> None:
    console.rule("[bold]Emergence eval[/bold]")
    with db.connect() as conn:
        charlie = repo.get_character_by_name(conn, "Charlie")
        result = emergence.compare(
            conn,
            character=charlie,
            topic="taking a creative risk",
            prompt="I have an idea for something risky. How should I think about it?",
        )

    console.print("[bold dim]FRESH (seed only):[/bold dim]")
    console.print(result.fresh_response)
    console.print()
    console.print("[bold cyan]DEVELOPED (with formed views):[/bold cyan]")
    console.print(result.developed_response)
    console.print()
    j = result.judgment
    console.print(
        f"[bold magenta]Judgment[/bold magenta]: distinct={j.distinct}  "
        f"score={j.score}/10\n{j.notes}"
    )


if __name__ == "__main__":
    test_council()
    test_emergence()
