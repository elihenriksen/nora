"""Nora CLI — entry point.

Provides the `nora` command. Subcommands:

  init                          create the world database
  agent new|list|show|delete    manage agents
  chat NAME [NAME...]           1:1 or multi-agent conversation
  council QUESTION --with ...   convene a Personal Council
  world [pair A B]              inspect world state
  postcards                     list postcards
  eval emergence NAME ...       compare fresh vs developed agent

Inside a chat or council, type your message and press Enter. Special commands
inside a session:

  /done    end the conversation and run the World Engine
  /exit    end immediately (skips world engine)
  /info    show who's in the room
  /skip    let agents take another turn without you saying anything
  @name    address a specific agent
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import _platform  # noqa: F401  (import-time UTF-8 setup on Windows)
from . import autonomous as autonomous_mod
from . import chat as chat_mod
from . import config, consolidate as consolidate_mod, council, db, emergence, inspect_view, llm, repo, starter, style, timing, world_engine
from .models import USER_ID, Character


def _version_callback(value: bool) -> None:
    if value:
        # Read from installed package metadata — most reliable across Python
        # versions and across editable / wheel installs.
        try:
            from importlib.metadata import version, PackageNotFoundError
            try:
                v = version("nora")
            except PackageNotFoundError:
                v = "unknown (not installed)"
        except ImportError:
            v = "unknown"
        console.print(f"nora {v}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Nora — conversational AI built on a social ontology.",
)


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Root callback — exposes --version on the top-level app."""
    pass

character_app = typer.Typer(no_args_is_help=True, help="Manage agents.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluations.")
app.add_typer(character_app, name="agent")
app.add_typer(eval_app, name="eval")

console = Console()


# --- Helpers --------------------------------------------------------------

_db_ready = False


def _ensure_db() -> None:
    """Initialize / migrate the DB on first access.

    Idempotent — safe to re-run. Users never need to remember `nora init`.
    Also pre-warms HTTP clients on the first call so the first LLM call
    doesn't pay the TLS handshake.
    """
    global _db_ready
    if _db_ready:
        return
    try:
        db.init_db()
    except Exception as e:
        console.print(f"[yellow]Schema init warning: {e}[/yellow]")
    _db_ready = True
    llm.prewarm_clients()


def _resolve_character(name: str) -> Character:
    with db.connect() as conn:
        c = repo.get_character_by_name(conn, name)
    if c is None:
        console.print(f"[red]No agent named '{name}'.[/red]")
        raise typer.Exit(code=1)
    return c


def _print_msg(name: str, content: str) -> None:
    """Render an agent message in their permanent color."""
    style.print_character_speech(console, name, content)


# --- init -----------------------------------------------------------------

def _prompt_user_name_if_needed() -> bool:
    """Ask for the user's name if not yet set. Returns True if a name is now set."""
    with db.connect() as conn:
        existing = repo.get_user_name(conn)
    if existing:
        return True
    console.print()
    console.print(
        "[dim]One thing first — agents will address you by name in conversations.[/dim]"
    )
    raw = Prompt.ask("[cyan]What's your name?[/cyan]").strip()
    if not raw:
        console.print("[dim]skipped — agents will just say 'you'.[/dim]")
        return False
    with db.transaction() as conn:
        repo.set_user_name(conn, raw)
    console.print(
        f"[dim]got it — agents will know you as[/dim] [bold cyan]{raw}[/bold cyan]"
    )
    return True


def _offer_starters_if_needed(seed: Optional[bool]) -> None:
    """Offer to create the starter agents if the world is empty."""
    with db.connect() as conn:
        existing = repo.list_characters(conn)
    if existing:
        return

    if seed is None:
        console.print()
        console.print(
            "[dim]No agents in the world yet. Four richly-seeded starter "
            "agents (Charlie, Candy, Gabriel, Finn) will give you something to "
            "talk to immediately.[/dim]"
        )
        seed = Confirm.ask(
            "[cyan]Create the starter agents?[/cyan]", default=True
        )

    if seed:
        with db.transaction() as conn:
            created = starter.seed_starters(conn)
        if created:
            names_inline = "  ·  ".join(style.name_markup(n) for n in created)
            console.print()
            console.print(f"[green]Created:[/green] {names_inline}")
            console.print()
            console.print(
                "[dim]Try:[/dim] [bold]nora chat Charlie[/bold]   "
                "[dim]·[/dim]   [bold]nora chat Charlie Candy Gabriel Finn[/bold]"
            )
    else:
        console.print()
        console.print("[dim]Run [bold]nora agent new[/bold] when you're ready.[/dim]")


def _final_state_summary() -> None:
    """Print the post-init/reset state so the user can read it before any
    parent process (the launcher) clears the screen."""
    with db.connect() as conn:
        name = repo.get_user_name(conn)
        chars = repo.list_characters(conn)
    console.print()
    console.print(f"[dim]world is ready[/dim]")
    console.print(
        f"  [dim]your name:[/dim]   "
        + (f"[bold cyan]{name}[/bold cyan]" if name else "[dim]not set[/dim]")
    )
    if chars:
        names_inline = "  ·  ".join(style.name_markup(c.name) for c in chars)
        console.print(f"  [dim]agents:[/dim]  {names_inline}")
    else:
        console.print("  [dim]agents:[/dim]  [dim]none yet[/dim]")
    console.print()
    # Pause so the launcher (or any parent) doesn't immediately clear this off-screen.
    Prompt.ask("[dim]press enter to continue[/dim]", default="")


@app.command()
def init(
    seed: Optional[bool] = typer.Option(
        None, "--seed/--no-seed",
        help="Create the four starter agents. If omitted, you'll be asked.",
    ),
) -> None:
    """Create or upgrade the world database. Idempotent — safe to re-run."""
    fresh = not db.is_initialized()
    db.init_db()

    if fresh:
        console.print(
            Panel(
                f"[bold green]Nora initialized.[/bold green]\n"
                f"World file: [cyan]{config.DB_PATH}[/cyan]",
                expand=False,
            )
        )
    else:
        console.print("[dim]Schema applied (idempotent). New tables and indexes are now present.[/dim]")

    # First-run flow: name first, then agents.
    # Note whether we'll prompt anything interactive — if we do, show the
    # final state summary at the end so the launcher's screen-clear doesn't
    # immediately hide the result.
    with db.connect() as conn:
        will_prompt_name = repo.get_user_name(conn) is None
        will_prompt_starters = not repo.list_characters(conn)
    _prompt_user_name_if_needed()
    _offer_starters_if_needed(seed)
    if will_prompt_name or will_prompt_starters:
        _final_state_summary()


@app.command(name="me")
def me(
    name: Optional[str] = typer.Argument(
        None, help="New name to use. Omit to show the current name. Use --clear to remove.",
    ),
    clear: bool = typer.Option(False, "--clear", help="Remove the saved name."),
) -> None:
    """Show or change the name agents use for you."""
    _ensure_db()
    with db.connect() as conn:
        current = repo.get_user_name(conn)

    if clear:
        with db.transaction() as conn:
            repo.set_user_name(conn, "")
        console.print(
            f"[dim]cleared — agents will say 'you' from now on.[/dim]"
            + (f"  [dim](was [bold]{current}[/bold])[/dim]" if current else "")
        )
        return

    if name is None:
        # Show current.
        if current:
            console.print(
                f"[dim]agents know you as[/dim] [bold cyan]{current}[/bold cyan]"
            )
            console.print(
                f"[dim]change it with[/dim] [bold]nora me <new-name>[/bold]"
                f"  [dim]·  remove with[/dim] [bold]nora me --clear[/bold]"
            )
        else:
            console.print(
                "[dim]no name set — agents will say 'you'.[/dim]\n"
                "[dim]set one with[/dim] [bold]nora me <your-name>[/bold]"
            )
        return

    new_name = name.strip()
    if not new_name:
        console.print("[yellow]No name provided.[/yellow]")
        raise typer.Exit(code=1)

    with db.transaction() as conn:
        repo.set_user_name(conn, new_name)

    if current and current != new_name:
        console.print(
            f"[dim]changed:[/dim] [bold]{current}[/bold] → [bold cyan]{new_name}[/bold cyan]"
        )
    else:
        console.print(
            f"[dim]agents will know you as[/dim] [bold cyan]{new_name}[/bold cyan]"
        )


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Wipe the world database and start fresh. PERMANENT."""
    # Show what's about to disappear, even if --yes (so it's still in the log).
    if config.DB_PATH.exists():
        try:
            with db.connect() as conn:
                n_chars = len(repo.list_characters(conn))
                n_convs_row = conn.execute(
                    "SELECT COUNT(*) AS n FROM conversations"
                ).fetchone()
                n_convs = n_convs_row["n"] if n_convs_row else 0
                n_cards = len(repo.list_postcards(conn))
                user_name = repo.get_user_name(conn)
        except Exception:
            n_chars = n_convs = n_cards = 0
            user_name = None
    else:
        console.print("[dim]Nothing to reset — no world database exists.[/dim]")
        return

    console.print()
    console.print(Panel(
        f"[bold red]Reset world[/bold red]\n\n"
        f"This will [bold]permanently delete[/bold] the world file at\n"
        f"  [cyan]{config.DB_PATH}[/cyan]\n\n"
        f"You will lose:\n"
        + (f"  · {n_chars} agent{'s' if n_chars != 1 else ''}\n" if n_chars else "")
        + (f"  · {n_convs} conversation{'s' if n_convs != 1 else ''}\n" if n_convs else "")
        + (f"  · {n_cards} postcard{'s' if n_cards != 1 else ''}\n" if n_cards else "")
        + (f"  · saved name: [bold]{user_name}[/bold]\n" if user_name else "")
        + "\n[dim]This cannot be undone.[/dim]",
        border_style="red", expand=False,
    ))

    if not yes:
        if not Confirm.ask("[bold red]Really delete everything?[/bold red]", default=False):
            console.print("[dim]cancelled — nothing was deleted.[/dim]")
            return

    # Wipe DB + WAL + SHM + journal sidecars.
    deleted = 0
    base = str(config.DB_PATH)
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = Path(base + suffix)
        if path.exists():
            path.unlink()
            deleted += 1
    console.print(f"[dim]deleted {deleted} file{'s' if deleted != 1 else ''}.[/dim]")

    # Reset cached state so a subsequent _ensure_db re-initializes cleanly.
    global _db_ready
    _db_ready = False

    # Re-init: schema + first-run flow (name + starter agents).
    db.init_db()
    console.print()
    console.print("[bold green]fresh world.[/bold green]")
    _prompt_user_name_if_needed()
    _offer_starters_if_needed(seed=None)
    # Pause so the launcher's screen-clear doesn't immediately wipe the
    # post-reset summary (this was the source of confusing stale-screen reports).
    _final_state_summary()


# --- agent ----------------------------------------------------------------

@character_app.command("new")
def character_new(
    name: Optional[str] = typer.Option(None, help="Agent name (interactive if omitted)."),
    interest: Optional[str] = typer.Option(None, help="Their central interest/domain."),
) -> None:
    """Create a new agent. Interactive."""
    _ensure_db()
    if not name:
        name = Prompt.ask("[cyan]Agent name[/cyan]").strip()
    if not interest:
        interest = Prompt.ask("[cyan]Central interest or domain[/cyan]").strip()

    backstory = style.prompt_with_file(
        console,
        "[cyan]Background (optional)[/cyan]  [dim](or /file <path>)[/dim]",
        default="",
    ).strip() or None
    voice_notes = style.prompt_with_file(
        console,
        "[cyan]How they speak (optional)[/cyan]  [dim](or /file <path>)[/dim]",
        default="",
    ).strip() or None

    console.print(
        "[dim]Seed traits — short sentences (one per line). Empty line to finish.[/dim]"
    )
    traits: list[str] = []
    while True:
        line = Prompt.ask(f"  trait #{len(traits) + 1}", default="").strip()
        if not line:
            break
        traits.append(line)

    with db.transaction() as conn:
        existing = repo.get_character_by_name(conn, name)
        if existing:
            console.print(f"[red]An agent named '{name}' already exists.[/red]")
            raise typer.Exit(code=1)
        c = repo.create_character(
            conn, name=name, interest=interest,
            seed_traits=traits, backstory=backstory, voice_notes=voice_notes,
        )

    console.print(
        Panel(
            f"[bold green]{c.name}[/bold green] is in the world. "
            f"[dim]({c.interest})[/dim]",
            expand=False,
        )
    )


@character_app.command("list")
def character_list() -> None:
    """List all agents."""
    _ensure_db()
    with db.connect() as conn:
        inspect_view.show_characters_list(conn, console)


@character_app.command("show")
def character_show(name: str) -> None:
    """Show full details for an agent."""
    _ensure_db()
    c = _resolve_character(name)
    with db.connect() as conn:
        inspect_view.show_character(conn, c, console)


@character_app.command("delete")
def character_delete(name: str) -> None:
    """Delete an agent (and all their relationships and traits)."""
    _ensure_db()
    c = _resolve_character(name)
    if not Confirm.ask(f"Really delete [bold red]{c.name}[/bold red] and all their relationships?"):
        raise typer.Exit()
    with db.transaction() as conn:
        repo.delete_character(conn, c.id)
    console.print(f"[dim]{c.name} removed.[/dim]")


# --- chat (1:1 and multi) -------------------------------------------------

@app.command()
def chat(
    names: list[str] = typer.Argument(
        None, help="One or more agent names. Omit when using --resume.",
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume",
        help="Conversation id to resume. Use `nora resume` (no id) to pick from recent.",
    ),
) -> None:
    """Open a 1:1, multi, or resumed chat."""
    _ensure_db()
    llm.reset_session_totals()

    if resume:
        session = _open_resumed_chat(resume)
        if session is None:
            return
        chars = session.characters
    else:
        if not names:
            console.print("[red]Need at least one agent name (or --resume).[/red]")
            raise typer.Exit(code=1)
        chars = [_resolve_character(n) for n in names]
        if len(chars) == 1:
            with db.transaction() as conn:
                session = chat_mod.start_one_on_one(conn, chars[0])
        else:
            with db.transaction() as conn:
                session = chat_mod.start_multi(conn, chars)

    console.clear()
    style.chat_header(console, characters=chars, conv_type=session.conv.type, council_topic=session.council_topic)
    if resume:
        with db.connect() as conn:
            session.conn = conn
            msgs = session.messages()
        if msgs:
            _print_msg_history(msgs, session)
        console.print(style.hint_line(f"resumed — {len(msgs)} prior messages"))
    _run_repl(session, opener=None)


def _open_resumed_chat(resume_arg: str) -> Optional[chat_mod.ChatSession]:
    """Resolve a --resume argument to a session. Returns None if cancelled."""
    with db.connect() as conn:
        if resume_arg.lower() == "pick":
            recent = repo.list_conversations(conn, limit=15)
            if not recent:
                console.print("[yellow]No conversations to resume.[/yellow]")
                return None
            console.print()
            console.print("[dim]recent conversations[/dim]")
            for i, c in enumerate(recent, 1):
                parts = repo.list_participants(conn, c.id)
                names = []
                for p in parts:
                    if p.type == "user":
                        continue
                    try:
                        names.append(repo.get_character_by_id(conn, p.id).name)
                    except LookupError:
                        names.append("?")
                when = c.started_at.strftime("%m-%d %H:%M") if c.started_at else "?"
                topic = (c.topic[:50] + "…") if c.topic and len(c.topic) > 50 else (c.topic or "")
                console.print(
                    f"  [dim]{i:>2}[/dim]  [dim]{when}[/dim]  "
                    f"[bold]{c.type}[/bold]  "
                    + "  ·  ".join(style.name_markup(n) for n in names)
                    + (f"  [dim italic]{topic}[/dim italic]" if topic else "")
                )
            console.print()
            raw = Prompt.ask("[cyan]Resume which?[/cyan]  [dim](number)[/dim]").strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(recent)):
                console.print("[yellow]No selection.[/yellow]")
                return None
            conv_id = recent[int(raw) - 1].id
        else:
            conv_id = resume_arg

    with db.transaction() as conn:
        try:
            return chat_mod.resume(conn, conv_id)
        except LookupError:
            console.print(f"[red]No conversation with id {conv_id}.[/red]")
            return None


def _print_msg_history(msgs, session) -> None:
    for m in msgs:
        if m.sender_type == "user":
            console.print()
            console.print(f"{style.user_markup()}: [white]{m.content}[/white]")
        elif m.sender_type == "character":
            char = session.chars_by_id.get(m.sender_id)
            if char:
                style.print_character_speech(console, char.name, m.content)


# --- council --------------------------------------------------------------

@app.command(name="council")
def council_cmd(
    question: str = typer.Argument(..., help="The question for the council."),
    with_: str = typer.Option(
        ..., "--with", help="Comma-separated agent names.",
    ),
) -> None:
    """Convene a Personal Council around a question."""
    _ensure_db()
    llm.reset_session_totals()
    names = [n.strip() for n in with_.split(",") if n.strip()]
    if len(names) < 2:
        console.print("[red]Council needs at least 2 agents.[/red]")
        raise typer.Exit(code=1)
    chars = [_resolve_character(n) for n in names]

    with db.transaction() as conn:
        session = chat_mod.start_council(conn, chars, topic=question)

    console.clear()
    style.chat_header(
        console, characters=chars, conv_type="council", council_topic=question,
    )
    console.print(style.hint_line(
        "Each agent gives an opening position. Then you join. "
        "Type /done to close · /exit to abandon · @name to address."
    ))

    # Opening positions are generated in parallel so no agent sees
    # another's opening before forming their own. After generation, the
    # openings are revealed in agent order.
    # Use db.connect() (autocommit) so the long LLM calls don't block the
    # usage-table inserts from llm._record().
    with db.connect() as conn:
        session.conn = conn
        repo.add_message(
            conn,
            conversation_id=session.conv.id,
            sender_type="system", sender_id="council",
            content=f"COUNCIL TOPIC: {question}\n\n{council.OPENING_PROMPT}",
        )
        with console.status(
            "[dim]Council members are forming their positions...[/dim]",
            spinner="dots",
        ):
            opening_msgs = council.generate_openings_parallel(session)
        for msg in opening_msgs:
            character = session.chars_by_id[msg.sender_id]
            style.print_character_speech(console, character.name, msg.content)

    _run_repl(session, opener=None)


# --- inspection -----------------------------------------------------------

@app.command()
def world() -> None:
    """Show world overview: agents, recent conversations, events, postcards."""
    _ensure_db()
    with db.connect() as conn:
        inspect_view.show_world(conn, console)


@app.command()
def pair(
    a: str = typer.Argument(..., help="First agent name."),
    b: str = typer.Argument(..., help="Second agent name."),
) -> None:
    """Inspect the shared context between two agents."""
    _ensure_db()
    ca = _resolve_character(a)
    cb = _resolve_character(b)
    with db.connect() as conn:
        inspect_view.show_pair(conn, ca, cb, console)


@app.command()
def resume() -> None:
    """Pick a recent conversation to resume."""
    _ensure_db()
    llm.reset_session_totals()
    chat(names=None, resume="pick")


@app.command()
def autonomous(
    with_: str = typer.Option(
        ..., "--with",
        help="Comma-separated agent names, or 'all' for every agent.",
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", help="Loose topic. Optional — leave blank for an open chat."
    ),
    max_turns: int = typer.Option(
        20, "--max-turns", help="Hard cap on agent turns. Default 20."
    ),
) -> None:
    """Watch agents talk among themselves. Type to inject a message; /stop to end."""
    _ensure_db()
    llm.reset_session_totals()

    if with_.strip().lower() in ("all", "*"):
        with db.connect() as conn:
            chars = repo.list_characters(conn)
    else:
        names = [n.strip() for n in with_.split(",") if n.strip()]
        chars = [_resolve_character(n) for n in names]

    if len(chars) < 2:
        console.print("[red]Need at least 2 agents.[/red]")
        raise typer.Exit(code=1)

    with db.transaction() as conn:
        session = chat_mod.start_multi(conn, chars)

    console.clear()
    style.chat_header(console, characters=chars, conv_type="multi")
    if topic:
        console.print(f"[dim]topic[/dim]  [italic]{topic}[/italic]")
    console.print(style.hint_line(
        f"Max turns: {max_turns}. Type to inject. /stop to end."
    ))

    def print_speak(character):
        # Returns a context manager whose __enter__ yields the streaming
        # callback. style.streaming_speech prints the name immediately, then
        # streams chunks via stdout, and prints a newline on exit.
        return style.streaming_speech(console, character.name)

    def on_user_message(msg):
        style.print_user_speech(console, msg.content)

    def on_status(s):
        console.print(f"\n[dim]· {s}[/dim]")

    # The autonomous loop manages its own writes via session.conn.
    # Use a non-transactional connection so the long loop doesn't hold a
    # write lock during LLM calls.
    with db.connect() as conn:
        session.conn = conn
        try:
            turns = autonomous_mod.run(
                session,
                seed_topic=topic,
                max_turns=max_turns,
                print_speak=print_speak,
                on_user_message=on_user_message,
                on_status=on_status,
            )
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
            turns = 0

    console.print(f"\n[dim]· {turns} agent turn{'s' if turns != 1 else ''}[/dim]")
    _close_with_engine(session)


@app.command()
def consolidate(
    character: Optional[str] = typer.Option(
        None, "--agent",
        help="Single agent or comma-separated list (e.g. 'Charlie,Candy'). Omit for all.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the cost-confirmation prompt.",
    ),
) -> None:
    """Run a memory consolidation pass on agent traits and shared history."""
    _ensure_db()
    llm.reset_session_totals()

    with db.connect() as conn:
        if character:
            tokens = [t.strip() for t in character.split(",") if t.strip()]
            chars = [_resolve_character(t) for t in tokens]
        else:
            chars = repo.list_characters(conn)

    if not chars:
        console.print("[dim]No agents to consolidate.[/dim]")
        return

    # Cost estimate: ~1 LLM call per agent (gpt-4o), ~$0.01 each.
    # Long shared-history rows add 1 call per row but are usually rare.
    if not yes:
        est_calls = len(chars)
        est_low = est_calls * 0.005
        est_high = est_calls * 0.02
        console.print()
        console.print(
            f"[dim]This will run a [bold]gpt-4o[/bold] consolidation pass for "
            f"{len(chars)} agent{'s' if len(chars) != 1 else ''}. "
            f"Estimated cost: [bold]~${est_low:.2f}–${est_high:.2f}[/bold].[/dim]"
        )
        if not Confirm.ask("[cyan]Proceed?[/cyan]", default=True):
            console.print("[dim]cancelled[/dim]")
            return

    console.print()
    console.print(f"[bold cyan]consolidating[/bold cyan]  [dim]{', '.join(c.name for c in chars)}[/dim]")
    console.print(f"[dim]{'─' * 60}[/dim]")

    results = []
    for c in chars:
        with console.status(f"[dim]consolidating {style.name_markup(c.name)}[/dim]", spinner="dots"):
            with db.connect() as conn:
                result = consolidate_mod.consolidate_character(conn, c)
        results.append(result)

        if result.error:
            console.print(f"\n{style.name_markup(result.character_name)}  [red]{result.error}[/red]")
            continue

        delta = result.before_traits - result.after_traits
        delta_str = (f"[red]−{delta}[/red]" if delta > 0
                     else f"[green]+{-delta}[/green]" if delta < 0
                     else "no change")
        console.print()
        console.print(
            f"{style.name_markup(result.character_name)}  "
            f"[dim]{result.before_traits} → {result.after_traits} traits[/dim]  ({delta_str})"
            + (f"  [dim]·  {result.history_summarized} history entries summarized[/dim]"
               if result.history_summarized else "")
        )
        console.print(f"  [dim italic]{result.notes}[/dim italic]")

    calls, pt, ct, cost = llm.session_totals()
    if calls > 0:
        console.print()
        console.print(style.session_cost_line(calls, pt, ct, cost))
    console.print()


@app.command()
def search(
    query: str = typer.Argument(..., help="Full-text search query (FTS5 syntax)."),
    limit: int = typer.Option(20, "--limit", help="Max results to show."),
) -> None:
    """Search messages across all conversations. Pick a result to resume that thread."""
    _ensure_db()

    with db.connect() as conn:
        try:
            results = repo.search_messages(conn, query, limit=limit)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            return

    if not results:
        console.print(f"[dim]No matches for '{query}'.[/dim]")
        return

    console.print()
    console.print(f"[bold cyan]search[/bold cyan]  [dim]'{query}'  ·  {len(results)} match{'es' if len(results) != 1 else ''}[/dim]")
    console.print(f"[dim]{'─' * 60}[/dim]")

    # Build display rows; group by conversation for readability.
    with db.connect() as conn:
        char_cache: dict[str, str] = {}
        rows: list[dict] = []
        for r in results:
            sender_name: str
            if r["sender_type"] == "user":
                sender_name = "user"
            else:
                if r["sender_id"] in char_cache:
                    sender_name = char_cache[r["sender_id"]]
                else:
                    try:
                        sender_name = repo.get_character_by_id(conn, r["sender_id"]).name
                    except LookupError:
                        sender_name = "?"
                    char_cache[r["sender_id"]] = sender_name
            rows.append({**r, "sender_name": sender_name})

    # Render with conversation headers.
    convs_seen: dict[str, int] = {}  # conversation_id -> index for resume picker
    pickable: list[str] = []
    for row in rows:
        conv_id = row["conversation_id"]
        if conv_id not in convs_seen:
            convs_seen[conv_id] = len(pickable) + 1
            pickable.append(conv_id)
            when = (row["created_at"] or "")[:16]
            ctype = row["conv_type"]
            ctopic = row["conv_topic"] or ""
            ctopic = (ctopic[:50] + "…") if len(ctopic) > 50 else ctopic
            console.print()
            header = f"[dim]{convs_seen[conv_id]:>2}[/dim]  [dim]{when}[/dim]  [bold]{ctype}[/bold]"
            if ctopic:
                header += f"  [dim italic]{ctopic}[/dim italic]"
            console.print(header)
        # Indented match line
        if row["sender_type"] == "user":
            label = style.user_markup("user", bold=False)
        else:
            label = style.name_markup(row["sender_name"], bold=False)
        snippet = row["content"]
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        console.print(f"      {label}: {snippet}")

    console.print()
    raw = Prompt.ask(
        "[cyan]Resume which conversation?[/cyan]  [dim](number, or blank to skip)[/dim]",
        default="",
    ).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(pickable):
        conv_id = pickable[int(raw) - 1]
        # Re-invoke chat --resume in the same process
        from typer import Exit
        try:
            chat(names=None, resume=conv_id)
        except Exit:
            pass


@app.command()
def doctor(
    skip_live: bool = typer.Option(
        False, "--skip-live", help="Skip the live API-call checks (offline mode).",
    ),
) -> None:
    """Run health checks: env file, API keys (live), database, schema, fixtures."""
    from . import validate

    console.print()
    console.print(f"[bold cyan]nora doctor[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")

    def _print_report(title: str, report: validate.ValidationReport) -> None:
        console.print()
        console.print(f"[dim]{title}[/dim]")
        for c in report.checks:
            mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
            detail = f"  [dim]{c.detail}[/dim]" if c.detail else ""
            console.print(f"  {mark} {c.name}{detail}")

    presence = validate.check_keys_present()
    _print_report("environment", presence)

    if not presence.all_passed:
        console.print()
        console.print("[yellow]Fix the above before running live checks.[/yellow]")
        raise typer.Exit(code=1)

    if not skip_live:
        with console.status("[dim]checking API keys (one tiny call per provider)[/dim]", spinner="dots"):
            live = validate.check_keys_live()
        _print_report("api keys (live)", live)

    db_health = validate.check_db_health()
    _print_report("database", db_health)

    fixtures = validate.check_fixtures()
    _print_report("eval fixtures", fixtures)

    console.print()
    all_reports = [presence, db_health, fixtures] + ([live] if not skip_live else [])
    if all(r.all_passed for r in all_reports):
        console.print("[bold green]all checks passed.[/bold green]")
    else:
        console.print("[bold red]some checks failed — see above.[/bold red]")
        raise typer.Exit(code=1)
    console.print()


@app.command()
def usage() -> None:
    """Show LLM token usage and estimated cost across all sessions."""
    _ensure_db()
    with db.connect() as conn:
        totals = repo.usage_totals(conn)
        by_model = repo.usage_by_model(conn)
        by_purpose = repo.usage_by_purpose(conn)
        recent = repo.usage_recent_sessions(conn, limit=10)

    console.print()
    console.print(f"[bold cyan]usage[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    console.print(
        f"[dim]total spend[/dim]   [bold]${totals['cost_usd']:.4f}[/bold]   "
        f"[dim]·  {totals['calls']} calls  ·  "
        f"{totals['prompt_tokens'] + totals['completion_tokens']:,} tokens[/dim]"
    )

    if by_model:
        console.print()
        console.print("[dim]by model[/dim]")
        from rich.table import Table
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("model")
        t.add_column("calls", justify="right")
        t.add_column("input tokens", justify="right")
        t.add_column("output tokens", justify="right")
        t.add_column("cost", justify="right")
        for r in by_model:
            t.add_row(
                r["model"], str(r["n"]),
                f"{r['pt']:,}", f"{r['ct']:,}",
                f"${r['cost']:.4f}",
            )
        console.print(t)

    if by_purpose:
        console.print()
        console.print("[dim]by purpose[/dim]")
        from rich.table import Table
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("purpose")
        t.add_column("calls", justify="right")
        t.add_column("cost", justify="right")
        for r in by_purpose:
            t.add_row(r["purpose"], str(r["n"]), f"${r['cost']:.4f}")
        console.print(t)

    if recent:
        console.print()
        console.print("[dim]recent sessions[/dim]")
        from rich.table import Table
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("when", style="dim")
        t.add_column("type")
        t.add_column("topic", style="dim")
        t.add_column("calls", justify="right")
        t.add_column("cost", justify="right")
        for r in recent:
            when = (r["last_at"] or "")[:16]
            t.add_row(
                when, r["type"] or "?", (r["topic"] or "")[:40],
                str(r["n"]), f"${r['cost']:.4f}",
            )
        console.print(t)
    console.print()


@app.command()
def postcards() -> None:
    """List postcards — meaningful moments between agents."""
    _ensure_db()
    with db.connect() as conn:
        cards = repo.list_postcards(conn)
    if not cards:
        console.print("[dim]No postcards yet.[/dim]")
        return
    for p in cards:
        with db.connect() as conn:
            try:
                a = repo.get_character_by_id(conn, p.character_a_id).name
                b = repo.get_character_by_id(conn, p.character_b_id).name
            except LookupError:
                continue
        when = p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "?"
        title_line = (
            f"{style.name_markup(a)}  &  {style.name_markup(b)}"
            f"   [dim]{when}[/dim]"
        )
        body = title_line + "\n\n" + p.summary
        if p.scene:
            body += f"\n\n[dim italic]{p.scene}[/dim italic]"
        console.print(Panel(body, border_style="magenta", expand=False))
        console.print()


# --- eval -----------------------------------------------------------------

@eval_app.command("emergence")
def eval_emergence(
    name: str = typer.Argument(..., help="Agent to test."),
    topic: str = typer.Option(..., "--topic", help="Topic the prompt is about."),
    prompt: str = typer.Option(..., "--prompt", help="The question to ask both versions."),
) -> None:
    """Compare an agent's fresh vs developed response on a topic."""
    _ensure_db()
    llm.reset_session_totals()
    c = _resolve_character(name)
    with db.connect() as conn:
        result = emergence.compare(conn, character=c, topic=topic, prompt=prompt)

    console.print(Panel(
        f"[bold]Topic:[/bold] {topic}\n[bold]Prompt:[/bold] {prompt}",
        title=f"Emergence eval — {c.name}", expand=False,
    ))
    console.print(Panel(result.fresh_response, title="Fresh (seed only)", border_style="dim"))
    console.print(Panel(
        result.developed_response,
        title="Developed (full social context)",
        border_style="cyan",
    ))
    j = result.judgment
    console.print(Panel(
        f"[bold]Distinct?[/bold] {'yes' if j.distinct else 'no'}    "
        f"[bold]Score:[/bold] {j.score}/10\n\n{j.notes}",
        title="Judgment", border_style="magenta",
    ))


@eval_app.command("suite")
def eval_suite(
    character: Optional[str] = typer.Option(
        None, "--agent",
        help="Single agent or comma-separated list (e.g. 'Charlie,Candy'). Omit to run every agent in the world.",
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip the post-suite validation pass."
    ),
    regenerate: bool = typer.Option(
        False, "--regenerate",
        help="Delete and rebuild auto-generated fixtures before running. Use when accumulated state has grown since the last generation and the cached probes no longer reflect what the agent has actually developed views on.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the cost-confirmation prompt.",
    ),
) -> None:
    """Run the full probe suite. Optionally validate the most recent world engine output."""
    _ensure_db()
    llm.reset_session_totals()

    # Parse --agent. Accept a single name OR a comma-separated list.
    # Each token gets resolved against the world (case-insensitive, numeric
    # selectors not supported at the CLI layer — those live in the launcher).
    chosen_names: Optional[list[str]] = None
    if character:
        tokens = [t.strip() for t in character.split(",") if t.strip()]
        chosen_names = [_resolve_character(t).name for t in tokens]

    # Determine which agents will be evaluated. Used for both the cost
    # estimate and the up-front fixture-generation notice.
    with db.connect() as conn:
        if chosen_names:
            eval_targets = chosen_names
        else:
            eval_targets = [c.name for c in repo.list_characters(conn)]

    if not eval_targets:
        console.print()
        console.print(
            "[yellow]No agents in the world to evaluate.[/yellow]  "
            "[dim]Create one with [bold]nora agent new[/bold] or "
            "seed the starters with [bold]nora init --seed[/bold].[/dim]"
        )
        return

    # --regenerate: wipe existing fixtures for eval_targets so they get
    # rebuilt from current accumulated state on this run.
    regenerated: list[str] = []
    if regenerate:
        for name in eval_targets:
            fixture_path = emergence.FIXTURES_DIR / f"{name.lower()}.json"
            if fixture_path.exists():
                fixture_path.unlink()
                regenerated.append(name)

    # Cost estimate: per agent we run 5 probes × 3 LLM calls each
    # (fresh / developed / judge). Validation adds ~12 calls if it runs.
    # Auto-generating a missing fixture adds 1 structured call per agent
    # (cheap relative to the probe run).
    if not yes:
        n_chars = len(eval_targets)
        n_probes = n_chars * 5
        missing = [n for n in eval_targets if not emergence.fixture_exists_for(n)]
        est_calls = n_probes * 3 + (0 if skip_validation else 12) + len(missing)
        est_low = est_calls * 0.001
        est_high = est_calls * 0.006
        console.print()
        console.print(
            f"[dim]This will run roughly [bold]{est_calls}[/bold] LLM calls "
            f"({n_probes} probes × 3 + validation). Estimated cost: "
            f"[bold]~${est_low:.2f}–${est_high:.2f}[/bold].[/dim]"
        )
        if regenerated:
            console.print(
                f"[dim]Regenerating fixtures for: [bold]{', '.join(regenerated)}[/bold]"
                f"  (cached probes wiped — new ones built from current accumulated state).[/dim]"
            )
        if missing:
            missing_inline = ", ".join(missing)
            console.print(
                f"[dim]Auto-generating eval fixtures for: [bold]{missing_inline}[/bold]  "
                f"(saved to [cyan]nora/eval_fixtures/[/cyan] — edit to refine probes).[/dim]"
            )
        if not Confirm.ask("[cyan]Proceed?[/cyan]", default=True):
            console.print("[dim]cancelled[/dim]")
            return

    console.print()
    console.print(f"[bold cyan]eval suite[/bold cyan]")
    console.print(f"[dim]{'─' * 60}[/dim]")

    from rich.table import Table
    with console.status("[dim]running probes[/dim]", spinner="dots"):
        with db.connect() as conn:
            results = emergence.run_suite(conn, character_names=chosen_names)

    if not results:
        console.print("[yellow]Suite produced no results.[/yellow]")
        return

    for cr in results:
        console.print()
        # Average is in-domain only — out-of-domain probes are controls and
        # would deflate the score by design if averaged in.
        n_ood = sum(1 for p in cr.probes if p.out_of_domain)
        avg_label = (
            "avg score (in-domain)" if n_ood else "avg score"
        )
        console.print(
            f"{style.name_markup(cr.character_name)}  "
            f"[dim]·  {avg_label}[/dim] [bold]{cr.avg_score:.1f}/10[/bold]"
        )
        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("topic")
        t.add_column("score", justify="right")
        t.add_column("distinct", justify="center")
        t.add_column("notes", style="dim", overflow="fold")
        # Sort: in-domain probes first by descending score, then out-of-domain
        # probes at the bottom so the control sits visually apart.
        sorted_probes = sorted(
            cr.probes,
            key=lambda x: (x.out_of_domain, -x.score),
        )
        for p in sorted_probes:
            # Out-of-domain probes get a quiet "(control)" tag so the reader
            # knows that low score is expected by design, not a failure.
            topic_display = (
                f"{p.topic}  [dim](control · out-of-domain)[/dim]"
                if p.out_of_domain
                else p.topic
            )
            if p.error:
                t.add_row(topic_display, "—", "—", f"[red]error: {p.error}[/red]")
                continue
            color = "green" if p.score >= 6 else ("yellow" if p.score >= 3 else "red")
            t.add_row(
                topic_display,
                f"[{color}]{p.score}/10[/{color}]",
                "✓" if p.distinct else "·",
                p.notes,
            )
        console.print(t)

    if not skip_validation:
        console.print()
        console.print(f"[bold cyan]validation pass[/bold cyan]  [dim](most recent processed conversation)[/dim]")
        console.print(f"[dim]{'─' * 60}[/dim]")
        with console.status("[dim]validating[/dim]", spinner="dots"):
            with db.connect() as conn:
                vr = emergence.run_validation(conn)
        if vr is None:
            console.print("[dim]no processed conversation to validate against.[/dim]")
        else:
            tp, tt = vr.trait_pass_rate
            rp, rt = vr.relationship_pass_rate
            ep, et = vr.exchange_pass_rate
            console.print(
                f"  [dim]traits[/dim]         "
                f"[bold]{tp}/{tt}[/bold] verified in agent behavior"
            )
            console.print(
                f"  [dim]relationships[/dim]  "
                f"[bold]{rp}/{rt}[/bold] reflected in two-turn exchange"
            )
            console.print(
                f"  [dim]exchanges[/dim]      "
                f"[bold]{ep}/{et}[/bold] supported by transcript"
            )
            failed = [c for c in vr.checks if not c.passed and c.error is None]
            if failed:
                console.print()
                console.print("[dim]failures (potential confabulation)[/dim]")
                for f in failed:
                    console.print(
                        f"  [red]·[/red] [{f.kind}] {f.summary}  [dim italic]{f.notes}[/dim italic]"
                    )

    calls, pt, ct, cost = llm.session_totals()
    if calls > 0:
        console.print()
        console.print(style.session_cost_line(calls, pt, ct, cost))
    console.print()


# --- Interactive REPL -----------------------------------------------------

def _run_repl(session: chat_mod.ChatSession, opener: Optional[str]) -> None:
    console.print(style.hint_line(
        "/done to end and run the World Engine  ·  /exit to abandon  ·  "
        "/skip to let them speak  ·  /file path  ·  /kick name  ·  @name to address"
    ))

    if opener is not None:
        with db.connect() as conn:
            session.conn = conn
            session.user_says(opener)
            _drive_responses(session)
            _maybe_announce_join(session)

    try:
        while True:
            try:
                console.print()
                line = Prompt.ask(style.user_markup()).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                console.print(style.hint_line("ending — running World Engine"))
                _close_with_engine(session)
                return

            if not line:
                continue

            lower = line.lower()

            if lower == "/done":
                _close_with_engine(session)
                return
            if lower in ("/exit", "/quit"):
                with db.transaction() as conn:
                    session.conn = conn
                    session.end()
                console.print(style.hint_line("exited; conversation NOT processed"))
                return
            if lower == "/info":
                names_in_room = "  ·  ".join(
                    style.name_markup(c.name) for c in session.characters
                )
                console.print(f"[dim]in the room[/dim]  {names_in_room}")
                continue
            if lower == "/skip":
                with db.connect() as conn:
                    session.conn = conn
                    _drive_responses(session, force_speakers=True)
                    _maybe_announce_join(session)
                continue

            # /add — bring in a pending join requester
            if lower == "/add":
                if session.pending_join is None:
                    console.print("[dim]nobody's waiting to join.[/dim]")
                    continue
                requester = session.pending_join
                session.pending_join = None
                with db.connect() as conn:
                    session.conn = conn
                    chat_mod.add_character_to_session(session, requester)
                    color = style.character_color(requester.name)
                    console.print(
                        f"\n[dim italic]·[/dim italic] [bold {color}]{requester.name}[/]"
                        f" [dim italic]joined the conversation.[/dim italic]"
                    )
                    try:
                        with style.streaming_speech(console, requester.name) as on_delta:
                            session.character_responds(requester, on_delta=on_delta)
                    except Exception as e:
                        console.print(f"[red]({requester.name} couldn't respond: {e})[/red]")
                continue

            # /ignore — dismiss a pending join requester
            if lower == "/ignore":
                if session.pending_join is None:
                    console.print("[dim]nothing to ignore.[/dim]")
                else:
                    session.pending_join = None
                    console.print("[dim]· dismissed[/dim]")
                continue

            # /file PATH — read a text file and treat its contents as the user's
            # next turn. Bypasses the terminal's ~1KB canonical-mode line-input
            # limit (so long pastes / voice-memo transcripts work). Same
            # downstream as a normal user message.
            if lower.startswith("/file"):
                content = style.load_file_as_text(console, line[len("/file"):].strip())
                if content is None:
                    continue
                # Treat exactly like a normal user message from here on:
                # dismiss any pending join request, persist the message,
                # drive responses, then re-check join interest.
                if session.pending_join is not None:
                    session.pending_join = None
                with db.connect() as conn:
                    session.conn = conn
                    session.user_says(content)
                    _drive_responses(session)
                    _maybe_announce_join(session)
                continue

            # /kick NAME — remove an agent from a multi or council session
            if lower.startswith("/kick"):
                rest = line[len("/kick"):].strip()
                if not rest:
                    console.print("[yellow]Usage:[/yellow] [bold]/kick <name>[/bold]")
                    continue
                if session.conv.type == "1on1":
                    console.print(
                        "[yellow]Can't kick in a 1:1 chat. Use [bold]/exit[/bold] to end.[/yellow]"
                    )
                    continue
                if len(session.characters) <= 1:
                    console.print("[yellow]Can't kick the last agent in the room.[/yellow]")
                    continue
                with db.connect() as conn:
                    session.conn = conn
                    removed = chat_mod.remove_character_from_session(session, rest)
                if removed:
                    style.print_character_left(console, removed.name)
                else:
                    console.print(f"[yellow]No agent named '{rest}' is in this conversation.[/yellow]")
                continue

            # Normal user message. Any normal input dismisses an unanswered
            # pending join request (treated as implicit ignore).
            if session.pending_join is not None:
                session.pending_join = None

            t_repl = timing.timer("REPL turn (post-input)")
            # Use db.connect() (autocommit) — NOT db.transaction(). Holding a
            # write transaction across the LLM call would block the usage-table
            # INSERT from `_record()` for ~5s every turn (sqlite busy_timeout).
            # Each insert here is independently meaningful; no batch atomicity needed.
            with db.connect() as conn:
                t_repl.step("db.connect opened (autocommit)")
                session.conn = conn
                session.user_says(line)
                t_repl.step("user message persisted")
                _drive_responses(session)
                t_repl.step("_drive_responses returned")
                _maybe_announce_join(session)
            t_repl.done()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


def _maybe_announce_join(session: chat_mod.ChatSession) -> None:
    """Run the join-interest check (cheap moderator call) and display a request
    between turns if any absent agent has a strong reason to join.

    The check itself is rate-limited inside `chat_mod.maybe_request_join` —
    it only runs every N user messages — so calling this after every turn
    is cheap. Use NORA_DEBUG=1 (or `./Nora.command --debug`) to trace what
    the check is doing on stderr.
    """
    from . import debug as _debug
    try:
        result = chat_mod.maybe_request_join(session)
    except Exception as e:
        _debug.log("join", f"check raised exception: {type(e).__name__}: {e}")
        return
    if result.requester is not None:
        style.print_join_request(console, result.requester.name)


def _drive_responses(
    session: chat_mod.ChatSession, force_speakers: bool = False
) -> None:
    """After user input (or /skip), pick speakers and have them respond in order."""
    t = timing.timer("_drive_responses")
    if session.conv.type == "1on1":
        speakers = [session.characters[0]]
        t.step("1on1: speakers picked (no moderator)")
    else:
        speakers = chat_mod.pick_next_speakers(session)
        t.step("multi: pick_next_speakers returned")
        if not speakers and force_speakers:
            speakers = chat_mod._round_robin_fallback(session)
            t.step("multi: round-robin fallback")

    for c in speakers:
        try:
            with style.streaming_speech(console, c.name) as on_delta:
                t.step(f"streaming context entered ({c.name})")
                session.character_responds(c, on_delta=on_delta)
                t.step(f"character_responds returned ({c.name})")
            t.step(f"streaming context exited ({c.name})")
        except Exception as e:
            console.print(f"\n[red]({c.name} couldn't respond: {e})[/red]")
            continue
    t.done()


def _close_with_engine(session: chat_mod.ChatSession) -> None:
    with db.transaction() as conn:
        session.conn = conn
        session.end()

    console.print()
    with console.status("[dim]running World Engine[/dim]", spinner="dots"):
        try:
            # Autocommit conn — the world engine LLM call must not hold a
            # write lock (would block usage-table inserts). The engine's
            # internal apply step manages its own atomicity if needed.
            with db.connect() as conn:
                out = world_engine.process_conversation(conn, session.conv.id)
        except Exception as e:
            console.print(f"[red]World Engine failed: {e}[/red]")
            return

    console.print(Panel(
        out.summary,
        title="[dim]world engine[/dim]",
        border_style="dim",
        expand=False,
        padding=(1, 2),
    ))
    counts = style.world_engine_counts(out)
    if counts:
        console.print(counts)

    # Session cost summary
    calls, pt, ct, cost = llm.session_totals()
    if calls > 0:
        console.print()
        console.print(style.session_cost_line(calls, pt, ct, cost))
    console.print()


if __name__ == "__main__":
    app()
