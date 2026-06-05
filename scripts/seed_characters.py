"""Reset the database and create the four starter characters.

Usage from the project root:

    .venv/bin/python scripts/seed_characters.py        # create if missing
    .venv/bin/python scripts/seed_characters.py --reset  # delete db first

The character data lives in `nora/starter.py` so `nora init` and this script
stay in sync.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

from nora import config, db, starter


console = Console()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="Delete the existing nora.db before seeding.")
    args = ap.parse_args()

    if args.reset and config.DB_PATH.exists():
        config.DB_PATH.unlink()
        console.print(f"[dim]Deleted {config.DB_PATH}[/dim]")

    db.init_db()
    console.print(f"[dim]DB ready at {config.DB_PATH}[/dim]")

    with db.transaction() as conn:
        created = starter.seed_starters(conn)

    if created:
        console.print()
        console.print("[bold green]Created:[/bold green]")
        for name in created:
            console.print(f"  · {name}")
    else:
        console.print()
        console.print("[dim]All starter characters already exist.[/dim]")

    console.print()
    console.print("[dim]Try:[/dim] [bold]nora chat Charlie[/bold]   or   [bold]./Nora.command[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
