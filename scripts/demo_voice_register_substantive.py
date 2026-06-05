"""Second voice-register check: a substantive prompt should engage multiple characters,
and they should disagree without sounding like fortune cookies.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from nora import chat as chat_mod, db, repo


console = Console()


def main() -> None:
    with db.transaction() as conn:
        charlie = repo.get_character_by_name(conn, "Charlie")
        candy = repo.get_character_by_name(conn, "Candy")
        session = chat_mod.start_multi(conn, [charlie, candy])

    for user_msg in [
        "I'm thinking about quitting my job to start a company. it's terrifying.",
        "what if I'm not ready",
    ]:
        console.print(f"[bold green]you[/bold green]: {user_msg}\n")
        with db.transaction() as conn:
            session.conn = conn
            session.user_says(user_msg)
            speakers = chat_mod.pick_next_speakers(session)
            console.print(f"[dim](moderator picked: {[c.name for c in speakers]})[/dim]")
            for c in speakers:
                msg = session.character_responds(c)
                console.print(f"[bold cyan]{c.name}[/bold cyan]: {msg.content}")
        console.print()

    with db.transaction() as conn:
        session.conn = conn
        session.end()


if __name__ == "__main__":
    main()
