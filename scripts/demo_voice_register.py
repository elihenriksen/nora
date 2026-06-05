"""Verify that characters match register on simple prompts.

The previous voice prompt made Charlie reach for a fishing parable on
'do i get chipotle?'. This test confirms the new voice produces a short,
register-matched response and a sensible multi-char interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import chat as chat_mod
from nora import db, repo


console = Console()


def main() -> None:
    with db.transaction() as conn:
        charlie = repo.get_character_by_name(conn, "Charlie")
        candy = repo.get_character_by_name(conn, "Candy")
        session = chat_mod.start_multi(conn, [charlie, candy])

    user_msg = "do I get chipotle"
    console.print(f"[bold green]you[/bold green]: {user_msg}\n")

    with db.transaction() as conn:
        session.conn = conn
        session.user_says(user_msg)
        speakers = chat_mod.pick_next_speakers(session)
        console.print(f"[dim](moderator picked: {[c.name for c in speakers]})[/dim]\n")
        for c in speakers:
            msg = session.character_responds(c)
            console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}\n")
        session.end()


if __name__ == "__main__":
    main()
