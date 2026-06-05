"""Interactive menu launcher for Nora.

Designed for double-click from Finder via the `Nora.command` shim. Walks the
user through the common operations by invoking the `nora` CLI as a subprocess
so the interactive REPL inherits the terminal cleanly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from . import _platform  # noqa: F401  (import-time UTF-8 setup on Windows)
from . import config, db, repo, style, validate


PROJECT_ROOT = Path(__file__).resolve().parent.parent


console = Console()


# --- repo helpers ---------------------------------------------------------

def _list_characters() -> list[str]:
    with db.connect() as conn:
        return [c.name for c in repo.list_characters(conn)]


def _stats_line() -> str:
    with db.connect() as conn:
        chars = repo.list_characters(conn)
        convs = repo.list_conversations(conn, limit=999)
        cards = repo.list_postcards(conn, limit=999)
    return style.world_stats_line(
        characters=len(chars),
        conversations=len(convs),
        postcards=len(cards),
    )


def _greeting_line() -> Optional[str]:
    """If a user name is set, return a subtle greeting line for the header."""
    try:
        with db.connect() as conn:
            name = repo.get_user_name(conn)
    except Exception:
        name = None
    if not name:
        return None
    return f"[dim]hi, [/dim][bold]{name}[/bold]"


def _character_palette_line() -> str:
    """Show the available agents in their colors, on one line."""
    names = _list_characters()
    if not names:
        return "[dim]no agents yet[/dim]"
    return "  " + "  ·  ".join(style.name_markup(n) for n in names)


# --- pickers --------------------------------------------------------------

def _resolve_token(token: str, names: list[str]) -> str | None:
    """Map a user-typed token to an agent name.

    Accepts either the name (case-insensitive) or the 1-based index from the
    rendered picker list.
    """
    t = token.strip()
    if not t:
        return None
    if t.isdigit():
        idx = int(t)
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None
    lower = t.lower()
    for n in names:
        if n.lower() == lower:
            return n
    return None


def _pick_character(prompt_text: str = "Agent name") -> str | None:
    names = _list_characters()
    if not names:
        console.print("[yellow]No agents yet — make one first.[/yellow]")
        return None
    console.print()
    console.print("[dim]available[/dim]")
    for i, n in enumerate(names, 1):
        console.print(f"  [dim]{i}[/dim]  {style.name_markup(n)}")
    console.print()
    raw = Prompt.ask(f"[cyan]{prompt_text}[/cyan]  [dim](number or name)[/dim]").strip()
    resolved = _resolve_token(raw, names)
    if resolved is None and raw:
        console.print(f"[yellow]No match for '{raw}'.[/yellow]")
    return resolved


def _pick_multiple(prompt_text: str, min_count: int = 2) -> list[str] | None:
    names = _list_characters()
    if not names:
        console.print("[yellow]No agents yet — make some first.[/yellow]")
        return None
    if len(names) < min_count:
        console.print(
            f"[yellow]Need at least {min_count} agents; you have {len(names)}.[/yellow]"
        )
        return None

    console.print()
    console.print("[dim]available[/dim]")
    for i, n in enumerate(names, 1):
        console.print(f"  [dim]{i}[/dim]  {style.name_markup(n)}")
    console.print()
    hint = (
        f"[cyan]{prompt_text}[/cyan]  "
        f"[dim](number or name; comma-separated for multiple, or 'all' for all {len(names)})[/dim]"
    )
    raw = Prompt.ask(hint).strip()

    if raw.lower() in ("all", "*"):
        return names

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    chosen: list[str] = []
    for t in tokens:
        resolved = _resolve_token(t, names)
        if resolved is None:
            console.print(f"[yellow]No match for '{t}'.[/yellow]")
            return None
        if resolved not in chosen:
            chosen.append(resolved)
    if len(chosen) < min_count:
        console.print(f"[yellow]Need at least {min_count}; got {len(chosen)}.[/yellow]")
        return None
    return chosen


# --- subprocess runner ----------------------------------------------------

def _run(*args: str) -> None:
    """Invoke the nora CLI with stdio inherited.

    Cross-platform: uses `sys.executable -m nora.cli ...` instead of a
    hardcoded path to a venv binary. Windows venvs put scripts in
    `.venv\\Scripts\\` (with `.exe` suffix); macOS/Linux use `.venv/bin/`.
    `sys.executable` always points at whichever interpreter is running this
    process, so the subprocess inherits the same Python.
    """
    cmd = [sys.executable, "-m", "nora.cli", *args]
    subprocess.run(cmd, check=False)


# --- Actions --------------------------------------------------------------

def action_chat() -> None:
    """Unified chat: pick one agent for 1:1, or several for multi.

    No new/resume prompt — resuming lives under the History menu now.
    """
    names = _pick_multiple("Who would you like to chat with?", min_count=1)
    if names:
        _run("chat", *names)


def action_history() -> None:
    """History sub-menu: resume a conversation or search across all messages."""
    console.print()
    console.print("[dim]history[/dim]")
    console.print("  [bold cyan]1[/bold cyan]   Resume a conversation")
    console.print("  [bold cyan]2[/bold cyan]   Search conversations")
    console.print()
    choice = Prompt.ask(
        "[cyan]Pick[/cyan]  [dim](1, 2, or blank to go back)[/dim]",
        default="",
    ).strip()
    if choice == "1":
        conv_id = _pick_recent_conversation(types=["1on1", "multi", "council"])
        if conv_id:
            _run("chat", "--resume", conv_id)
    elif choice == "2":
        action_search()
    # blank or anything else → just return to the main menu


def _pick_recent_conversation(types: list[str]) -> str | None:
    with db.connect() as conn:
        all_recent = repo.list_conversations(conn, limit=30)
        filtered = [c for c in all_recent if c.type in types][:5]
        if not filtered:
            console.print("[yellow]No conversations to resume.[/yellow]")
            return None
        rows = []
        for c in filtered:
            parts = repo.list_participants(conn, c.id)
            names = []
            for p in parts:
                if p.type == "user":
                    continue
                try:
                    names.append(repo.get_character_by_id(conn, p.id).name)
                except LookupError:
                    names.append("?")
            rows.append((c, names))

    console.print()
    console.print("[dim]recent conversations[/dim]")
    for i, (c, names) in enumerate(rows, 1):
        when = c.started_at.strftime("%m-%d %H:%M") if c.started_at else "?"
        topic = (c.topic[:60] + "…") if c.topic and len(c.topic) > 60 else (c.topic or "")
        console.print(
            f"  [dim]{i:>2}[/dim]  [dim]{when}[/dim]  "
            f"[dim]{c.type}[/dim]  "
            + "  ·  ".join(style.name_markup(n) for n in names)
            + (f"  [dim italic]{topic}[/dim italic]" if topic else "")
        )
    console.print()
    raw = Prompt.ask("[cyan]Resume which?[/cyan]  [dim](number)[/dim]").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(rows)):
        console.print("[yellow]No selection.[/yellow]")
        return None
    return rows[int(raw) - 1][0].id


def action_council() -> None:
    names = _pick_multiple("Which council members?", min_count=2)
    if not names:
        return
    # /file <path> works here so long memos / transcripts can be the question
    # without hitting the terminal's ~1KB line-input limit.
    question = style.prompt_with_file(
        console,
        "\n[cyan]Council question[/cyan]  [dim](or /file <path>)[/dim]",
    ).strip()
    if not question:
        console.print("[yellow]No question — cancelled.[/yellow]")
        return
    _run("council", question, "--with", ",".join(names))


def action_autonomous() -> None:
    names = _pick_multiple("Which agents?", min_count=2)
    if not names:
        return
    topic = style.prompt_with_file(
        console,
        "\n[cyan]Topic[/cyan]  [dim](optional — leave blank, or /file <path>)[/dim]",
        default="",
    ).strip()
    max_turns_raw = Prompt.ask(
        "[cyan]Max turns[/cyan]", default="20"
    ).strip()
    try:
        max_turns = int(max_turns_raw)
    except ValueError:
        max_turns = 20
    args = ["autonomous", "--with", ",".join(names), "--max-turns", str(max_turns)]
    if topic:
        args += ["--topic", topic]
    _run(*args)


def action_new_character() -> None:
    _run("agent", "new")


def action_show_world() -> None:
    _run("world")


def action_show_pair() -> None:
    names = _pick_multiple("Which pair? (exactly two)", min_count=2)
    if not names:
        return
    if len(names) != 2:
        console.print("[yellow]Pair view needs exactly two names.[/yellow]")
        return
    _run("pair", names[0], names[1])


def action_show_character() -> None:
    name = _pick_character("Which agent?")
    if name:
        _run("agent", "show", name)


def action_postcards() -> None:
    _run("postcards")


def action_emergence() -> None:
    name = _pick_character("Which agent to test?")
    if not name:
        return
    topic = style.prompt_with_file(
        console,
        "\n[cyan]Topic[/cyan]  [dim](e.g. 'creative risk', or /file <path>)[/dim]",
    ).strip()
    if not topic:
        return
    prompt = style.prompt_with_file(
        console,
        "[cyan]Prompt[/cyan]  [dim](the question to ask both versions, or /file <path>)[/dim]",
    ).strip()
    if not prompt:
        return
    _run("eval", "emergence", name, "--topic", topic, "--prompt", prompt)


def _resolve_multi_or_all(raw: str, names: list[str]) -> Optional[list[str]]:
    """Resolve user input into a list of agent names.

    Accepts: a single token (number or name), a comma-separated list of
    tokens, or 'all' / '*'. Returns None on any unresolved token (caller
    should bail) and an empty list if the user typed nothing meaningful.
    """
    if raw.lower() in ("all", "*"):
        return names
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return []
    chosen: list[str] = []
    for t in tokens:
        resolved = _resolve_token(t, names)
        if resolved is None:
            console.print(f"[yellow]No match for '{t}'.[/yellow]")
            return None
        if resolved not in chosen:
            chosen.append(resolved)
    return chosen


def action_eval_suite() -> None:
    names = _list_characters()
    if not names:
        console.print("[yellow]No agents yet — make one first.[/yellow]")
        return
    console.print()
    console.print("[dim]available[/dim]")
    for i, n in enumerate(names, 1):
        console.print(f"  [dim]{i}[/dim]  {style.name_markup(n)}")
    console.print()
    raw = Prompt.ask(
        "[cyan]Agent[/cyan]  "
        "[dim](number, name, comma-separated for multiple, or blank for all)[/dim]",
        default="",
    ).strip()

    # Resolve agent selection (blank = all agents).
    chosen: list[str] = []
    if raw:
        resolved = _resolve_multi_or_all(raw, names)
        if resolved is None:
            return  # unresolved token, message already printed
        chosen = resolved or []

    # Offer to regenerate fixtures. Useful when the agent's accumulated
    # state has grown since the cached probes were generated and the user
    # wants the suite to test against current views.
    regenerate = Confirm.ask(
        "[cyan]Regenerate evaluation probes?[/cyan]  "
        "[dim](rebuilds fixtures from current accumulated state — slower, more accurate)[/dim]",
        default=False,
    )

    args = ["eval", "suite"]
    if chosen:
        args.extend(["--agent", ",".join(chosen)])
    if regenerate:
        args.append("--regenerate")
    _run(*args)


def action_consolidate() -> None:
    names = _list_characters()
    if not names:
        console.print("[yellow]No agents yet — make one first.[/yellow]")
        return
    console.print()
    console.print("[dim]available[/dim]")
    for i, n in enumerate(names, 1):
        console.print(f"  [dim]{i}[/dim]  {style.name_markup(n)}")
    console.print()
    raw = Prompt.ask(
        "[cyan]Agent[/cyan]  "
        "[dim](number, name, comma-separated for multiple, or blank for all)[/dim]",
        default="",
    ).strip()
    if not raw:
        _run("consolidate")
        return
    chosen = _resolve_multi_or_all(raw, names)
    if chosen is None:
        return
    if not chosen:
        _run("consolidate")
        return
    _run("consolidate", "--agent", ",".join(chosen))


def action_search() -> None:
    query = Prompt.ask("\n[cyan]Search messages for[/cyan]").strip()
    if not query:
        return
    _run("search", query)


def action_usage() -> None:
    _run("usage")


def action_doctor() -> None:
    _run("doctor")


def action_reset() -> None:
    _run("reset")


def action_change_name() -> None:
    with db.connect() as conn:
        current = repo.get_user_name(conn)
    if current:
        console.print(f"\n[dim]agents currently know you as[/dim] [bold cyan]{current}[/bold cyan]")
    raw = Prompt.ask(
        "[cyan]What name should agents use?[/cyan]  [dim](blank to cancel)[/dim]",
        default="",
    ).strip()
    if not raw:
        console.print("[dim]cancelled.[/dim]")
        return
    _run("me", raw)


# --- Menu -----------------------------------------------------------------

MENU_GROUPS = [
    ("conversations", [
        ("1",  "Chat",                             action_chat),
        ("2",  "Convene a Personal Council",       action_council),
        ("3",  "Watch agents talk autonomously",   action_autonomous),
    ]),
    ("history", [
        ("4",  "History  (resume or search)",      action_history),
    ]),
    ("create", [
        ("5",  "Create a new agent",               action_new_character),
    ]),
    ("inspect", [
        ("6",  "Show the world",                   action_show_world),
        ("7",  "Inspect a pair of agents",         action_show_pair),
        ("8",  "Show one agent in full",           action_show_character),
        ("9",  "List postcards",                   action_postcards),
    ]),
    ("evaluate", [
        ("10", "Run an emergence eval",            action_emergence),
        ("11", "Run the full eval suite",          action_eval_suite),
    ]),
    ("maintain", [
        ("12", "Consolidate agent memory",         action_consolidate),
        ("13", "Show LLM usage and cost",          action_usage),
        ("14", "Run a health check (doctor)",      action_doctor),
        ("15", "Change your name",                 action_change_name),
        ("16", "Reset world (delete everything)",  action_reset),
    ]),
]


def _flat_menu() -> list[tuple[str, str, callable]]:
    return [item for _, group in MENU_GROUPS for item in group]


def _render_menu() -> None:
    for group_label, items in MENU_GROUPS:
        console.print(f"  [dim]{group_label}[/dim]")
        for key, label, _ in items:
            console.print(f"    [bold cyan]{key:>2}[/bold cyan]   {label}")
        console.print()
    console.print(f"    [bold cyan]{'q':>2}[/bold cyan]   Quit")


# --- Main loop ------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    # Argument handling — only one flag for now: --debug enables NORA_DEBUG
    # for this process AND for every subprocess we spawn (subprocess inherits
    # the parent env). Useful when double-clicking from Finder where no shell
    # env is available.
    import os
    args = list(argv) if argv is not None else sys.argv[1:]
    if "--debug" in args or "-d" in args:
        os.environ["NORA_DEBUG"] = "1"
        # Re-import the debug module's ENABLED flag so any module that
        # imported it before this point picks up the new value.
        from . import debug as _debug
        _debug.ENABLED = True
        console.print("[dim italic][debug mode on — NORA_DEBUG=1 propagated to subprocesses][/dim italic]")

    # Validate environment + required keys (presence only — not a live API call).
    report = validate.check_keys_present()
    if not report.all_passed:
        lines = ["[red]Nora isn't ready to launch.[/red]\n"]
        for c in report.checks:
            mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
            lines.append(f"  {mark} {c.name}  [dim]{c.detail}[/dim]")
        lines.append("")
        if not config.ENV_PATH.exists():
            lines.append(
                f"[dim]Create your env file:[/dim]\n"
                f"  [bold]cp .env.example .env[/bold]\n"
                f"  [bold]$EDITOR {config.ENV_PATH}[/bold]"
            )
        else:
            missing = [c.name.replace(" set", "") for c in report.failed if c.name.endswith(" set")]
            if missing:
                lines.append(
                    f"[dim]Add the missing key(s) to[/dim] [cyan]{config.ENV_PATH}[/cyan]:\n  "
                    + "\n  ".join(f"[bold]{k}[/bold]" for k in missing)
                )
        lines.append("")
        lines.append("[dim]Then re-launch. Run [bold]nora doctor[/bold] for a deeper check.[/dim]")
        console.print(Panel("\n".join(lines), expand=False, border_style="red"))
        return 1

    # First-run + back-fill flow:
    # - If the DB doesn't exist, run `nora init` to create it AND prompt for name + starters.
    # - If the DB exists but no user name is set (existing user, pre-feature),
    #   run `nora init` so they get the name prompt now. `init` is idempotent
    #   and skips parts that are already done.
    needs_setup = not db.is_initialized()
    if not needs_setup:
        try:
            with db.connect() as conn:
                if repo.get_user_name(conn) is None:
                    needs_setup = True
        except Exception:
            needs_setup = True
    if needs_setup:
        _run("init")

    flat = _flat_menu()

    while True:
        console.clear()
        style.app_header(console, stats=_stats_line())
        greeting = _greeting_line()
        if greeting:
            console.print(greeting)
        console.print(_character_palette_line())
        console.print()
        _render_menu()

        choice = Prompt.ask("\n[bold cyan]>[/bold cyan]").strip().lower()
        if choice in ("q", "quit", "exit"):
            console.print("[dim]bye[/dim]")
            return 0
        action = next((a for k, _, a in flat if k == choice), None)
        if action is None:
            console.print("[yellow]Unknown choice.[/yellow]")
            Prompt.ask("[dim]press enter to continue[/dim]", default="")
            continue

        console.clear()
        try:
            action()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            Prompt.ask("[dim]press enter to return to menu[/dim]", default="")


def _has_api_key() -> bool:
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    sys.exit(main())
