"""Verify the moderator picks all three characters on a substantive prompt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import chat as chat_mod, db, repo


console = Console()


def main() -> None:
    with db.transaction() as conn:
        chars = repo.list_characters(conn)
        session = chat_mod.start_multi(conn, chars)

    prompts = [
        "I had a really rough night and feel completely drained. What should I do today?",
        "I want to start a new hobby but I have no idea what to pick.",
    ]

    with db.connect() as conn:
        session.conn = conn
        for p in prompts:
            console.print(f"[bold green]you[/bold green]: {p}")
            session.user_says(p)
            speakers = chat_mod.pick_next_speakers(session)
            console.print(f"[dim](moderator picked: {[c.name for c in speakers]})[/dim]")
            for c in speakers:
                msg = session.character_responds(c)
                console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}")
            console.print()
        session.end()


if __name__ == "__main__":
    main()
