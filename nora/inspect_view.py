"""Inspection commands — render world state to the terminal.

Pure rendering. No mutations. Agent names always go through
`style.name_markup()` so they appear in their permanent color.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import repo, style
from .models import USER_ID, Character


def _section_rule(console: Console, label: str) -> None:
    console.print()
    console.print(f"[dim]{label}[/dim]")


def show_character(conn: sqlite3.Connection, character: Character, console: Console) -> None:
    color = style.character_color(character.name)

    head = (
        f"[bold {color}]{character.name}[/bold {color}]"
        f"   [dim]·[/dim]   [dim]{character.interest}[/dim]"
    )
    console.print()
    console.print(head)
    console.print(f"[dim]{'─' * 60}[/dim]")
    console.print()

    if character.backstory:
        console.print(f"[dim]background[/dim]")
        console.print(f"  {character.backstory}")
        console.print()
    if character.voice_notes:
        console.print(f"[dim]voice[/dim]")
        console.print(f"  {character.voice_notes}")
        console.print()
    if character.seed_traits:
        console.print(f"[dim]seed traits[/dim]")
        for s in character.seed_traits:
            console.print(f"  · {s}")
        console.print()

    traits = repo.list_traits(conn, character.id)
    if traits:
        console.print(f"[dim]formed views (emergent identity)[/dim]")
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("topic", style=color)
        t.add_column("view")
        t.add_column("strength", justify="right")
        t.add_column("origin", style="dim")
        for i, tr in enumerate(traits):
            if i > 0:
                t.add_row("", "", "", "")
            t.add_row(tr.topic, tr.perspective, "★" * tr.strength, tr.origin_summary or "")
        console.print(t)
    else:
        console.print("[dim]no formed views yet — this agent hasn't been talked to enough[/dim]")

    rels = repo.list_relationships_for(conn, entity_type="character", entity_id=character.id)
    if rels:
        _section_rule(console, "relationships")
        for rel in rels:
            other_type, other_id = rel.other("character", character.id)
            if other_type == "user":
                other_label_markup = style.user_markup("user")
                other_lower = "user"
            else:
                try:
                    other = repo.get_character_by_id(conn, other_id)
                except LookupError:
                    continue
                other_label_markup = style.name_markup(other.name)
                other_lower = other.name.lower()
            ctx = repo.list_context_for_relationship(conn, rel.id)
            console.print(
                f"  {style.name_markup(character.name)}  ↔  {other_label_markup}  "
                f"[dim]({len(ctx)} topic{'s' if len(ctx) != 1 else ''})[/dim]"
            )
            for c in ctx[:5]:
                my_view = c.a_perspective if rel.a_id == character.id else c.b_perspective
                their_view = c.b_perspective if rel.a_id == character.id else c.a_perspective
                console.print(f"      · [bold]{c.topic}[/bold]")
                if my_view:
                    console.print(f"        [dim]you:[/dim] {my_view}")
                if their_view:
                    console.print(f"        [dim]{other_lower}:[/dim] {their_view}")
                if c.shared_history:
                    console.print(f"        [dim italic]history:[/dim italic] {c.shared_history}")
    console.print()


def show_characters_list(conn: sqlite3.Connection, console: Console) -> None:
    chars = repo.list_characters(conn)
    if not chars:
        console.print("[dim]No agents yet. Run `nora agent new` to make one.[/dim]")
        return
    console.print()
    console.print("[dim]agents[/dim]")
    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("name")
    t.add_column("interest")
    t.add_column("formed views", justify="right")
    for c in chars:
        traits = repo.list_traits(conn, c.id)
        t.add_row(
            style.name_markup(c.name),
            c.interest,
            str(len(traits)),
        )
    console.print(t)
    console.print()


def show_world(conn: sqlite3.Connection, console: Console, limit: int = 20) -> None:
    chars = repo.list_characters(conn)
    convs = repo.list_conversations(conn, limit=limit)
    events = repo.list_world_events(conn, limit=limit)
    cards = repo.list_postcards(conn, limit=limit)

    console.print()
    console.print(f"[bold cyan]world[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    console.print(style.world_stats_line(
        characters=len(chars), conversations=len(convs), postcards=len(cards),
    ))

    if chars:
        console.print()
        names_line = "  " + "  ·  ".join(style.name_markup(c.name) for c in chars)
        console.print(names_line)

    if convs:
        _section_rule(console, "recent conversations")
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("when", style="dim")
        t.add_column("type")
        t.add_column("participants")
        t.add_column("topic", style="dim")
        for c in convs[:10]:
            parts = repo.list_participants(conn, c.id)
            tokens = []
            for p in parts:
                if p.type == "user":
                    tokens.append(style.user_markup("user"))
                else:
                    try:
                        tokens.append(style.name_markup(repo.get_character_by_id(conn, p.id).name))
                    except LookupError:
                        tokens.append("?")
            when = c.started_at.strftime("%m-%d %H:%M") if c.started_at else "?"
            t.add_row(when, c.type, "  ·  ".join(tokens), c.topic or "")
        console.print(t)

    if events:
        _section_rule(console, "recent world events")
        for e in events[:10]:
            console.print(f"  · [dim]{e.event_type}[/dim]  {e.summary}")

    if cards:
        _section_rule(console, "postcards")
        for p in cards[:10]:
            try:
                a = repo.get_character_by_id(conn, p.character_a_id).name
                b = repo.get_character_by_id(conn, p.character_b_id).name
            except LookupError:
                continue
            console.print(
                f"  · {style.name_markup(a)}  &  {style.name_markup(b)}  "
                f"[dim]—[/dim] {p.summary}"
            )
            if p.scene:
                console.print(f"      [dim italic]{p.scene}[/dim italic]")
    console.print()


def show_pair(
    conn: sqlite3.Connection,
    a: Character,
    b: Character,
    console: Console,
) -> None:
    rel = repo.get_or_create_relationship(
        conn, a_type="character", a_id=a.id, b_type="character", b_id=b.id,
    )
    ctx = repo.list_context_for_relationship(conn, rel.id)
    exchanges = repo.list_exchanges_for_pair(conn, a.id, b.id)

    console.print()
    console.print(f"  {style.name_markup(a.name)}   ↔   {style.name_markup(b.name)}")
    console.print(f"[dim]{'─' * 60}[/dim]")

    if not ctx:
        console.print()
        console.print("[dim]no shared topics yet[/dim]")
    else:
        for c in ctx:
            a_view = c.a_perspective if rel.a_id == a.id else c.b_perspective
            b_view = c.b_perspective if rel.a_id == a.id else c.a_perspective
            console.print()
            console.print(f"[bold]{c.topic}[/bold]")
            if a_view:
                console.print(f"  {style.name_markup(a.name)}: {a_view}")
            if b_view:
                console.print(f"  {style.name_markup(b.name)}: {b_view}")
            if c.shared_history:
                console.print(f"  [dim italic]history:[/dim italic] {c.shared_history}")

    if exchanges:
        _section_rule(console, f"structured context exchanges ({len(exchanges)})")
        for ex in exchanges[:10]:
            from_name = a.name if ex.from_character_id == a.id else b.name
            to_name = b.name if ex.to_character_id == b.id else a.name
            console.print(
                f"  {style.name_markup(from_name)}  →  {style.name_markup(to_name)}  "
                f"[dim]{ex.exchange_type}[/dim]"
            )
            for k, v in ex.payload.items():
                console.print(f"      [dim]{k}:[/dim] {v}")
    console.print()
