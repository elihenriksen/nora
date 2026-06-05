"""Visual style for Nora.

Centralizes everything that touches presentation: character colors, the
NORA wordmark, common formatters for chat and inspection. Everywhere a
character name is shown — chat, tables, postcards, world events — it goes
through `name_markup()` so the color is consistent across the product.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# --- Colors ---------------------------------------------------------------

USER_COLOR = "cyan"

# Hand-assigned colors for the seed characters. Permanent, used everywhere.
NAMED_COLORS = {
    "charlie": "green",
    "candy": "magenta",
    "gabriel": "yellow",
}

# Fallback palette for any other character. Deterministic by name hash, so a
# given name always maps to the same color across runs.
PALETTE = [
    "blue", "red", "bright_blue", "bright_red",
    "bright_green", "bright_magenta", "bright_cyan",
    "deep_sky_blue1", "orange3", "pale_violet_red1",
    "spring_green3", "dark_orange3",
]


def character_color(name: str) -> str:
    """Stable color for an agent. Hand-assigned where defined, else hashed."""
    key = name.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def name_markup(name: str, *, bold: bool = True) -> str:
    """Rich markup for an agent's name in their permanent color."""
    color = character_color(name)
    style = f"bold {color}" if bold else color
    return f"[{style}]{name}[/{style}]"


def names_inline(names: Iterable[str], sep: str = "  ·  ") -> str:
    """A list of agent names joined inline, each in their own color."""
    return sep.join(name_markup(n) for n in names)


def user_markup(label: str = "you", *, bold: bool = True) -> str:
    style = f"bold {USER_COLOR}" if bold else USER_COLOR
    return f"[{style}]{label}[/{style}]"


# --- Branding -------------------------------------------------------------

NORA_ASCII = (
    "███╗   ██╗ ██████╗ ██████╗  █████╗ \n"
    "████╗  ██║██╔═══██╗██╔══██╗██╔══██╗\n"
    "██╔██╗ ██║██║   ██║██████╔╝███████║\n"
    "██║╚██╗██║██║   ██║██╔══██╗██╔══██║\n"
    "██║ ╚████║╚██████╔╝██║  ██║██║  ██║\n"
    "╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"
)


def app_header(console: Console, *, stats: Optional[str] = None) -> None:
    """Print the full app header: wordmark, tagline, divider, optional stats."""
    console.print(NORA_ASCII, style="bold cyan")
    console.print("[dim]Social Agents[/dim]")
    console.print(_divider())
    if stats:
        console.print(stats)


def _divider(width: int = 60) -> str:
    return "[dim]" + "─" * width + "[/dim]"


def world_stats_line(
    *, characters: int, conversations: int, postcards: int
) -> str:
    """Compact stats line — no box."""
    parts = [
        f"[dim]agents[/dim] [bold]{characters}[/bold]",
        f"[dim]conversations[/dim] [bold]{conversations}[/bold]",
        f"[dim]postcards[/dim] [bold]{postcards}[/bold]",
    ]
    return "  ·  ".join(parts)


# --- Chat / session formatting -------------------------------------------

def chat_header(
    console: Console,
    *,
    characters: list,
    conv_type: str,
    council_topic: Optional[str] = None,
) -> None:
    """Pretty header shown when entering a chat or council session."""
    names = names_inline(c.name for c in characters)

    if conv_type == "council":
        body = (
            f"[dim]COUNCIL[/dim]\n\n"
            f"[bold]{council_topic}[/bold]\n\n"
            f"{names}"
        )
        panel = Panel(body, border_style="magenta", padding=(0, 2), expand=False)
    elif conv_type == "1on1":
        body = (
            f"[dim]CONVERSATION[/dim]\n\n"
            f"with {names}"
        )
        panel = Panel(body, border_style="cyan", padding=(0, 2), expand=False)
    else:
        body = (
            f"[dim]IN THE ROOM[/dim]\n\n"
            f"{names}"
        )
        panel = Panel(body, border_style="cyan", padding=(0, 2), expand=False)

    console.print(panel)
    console.print()


def hint_line(text: str) -> str:
    return f"[dim]{text}[/dim]"


def print_character_speech(console: Console, name: str, content: str) -> None:
    """Render one agent turn with the right color and breathing room."""
    console.print()
    console.print(f"{name_markup(name)}: [white]{content}[/white]")


from contextlib import contextmanager
import sys as _sys


@contextmanager
def streaming_speech(console: Console, name: str):
    """Context manager that prints an agent speech header, then yields a
    callback the LLM stream can write deltas to. On exit, prints a newline.

    Usage:
        with style.streaming_speech(console, character.name) as on_delta:
            session.character_responds(character, on_delta=on_delta)
    """
    color = character_color(name)
    console.print()
    # Print name + colon without a newline so streamed content continues the line.
    console.print(f"[bold {color}]{name}[/bold {color}]: ", end="")
    _sys.stdout.flush()

    def on_delta(chunk: str) -> None:
        _sys.stdout.write(chunk)
        _sys.stdout.flush()

    try:
        yield on_delta
    finally:
        console.print()


def print_user_speech(console: Console, content: str) -> None:
    """Render an injected user message in the autonomous flow."""
    console.print()
    console.print(f"{user_markup()}: [white]{content}[/white]")


def load_file_as_text(console: Console, path_str: str) -> Optional[str]:
    """Read a UTF-8 text file by path string. Returns the file contents,
    or None on any failure (no such file, not UTF-8, other read error).

    Prints a friendly red error to console on failure, a dim "loaded"
    confirmation on success, and a yellow size warning if the file is
    large (>50K chars). The caller decides what to do with None.

    Tilde expansion works. Paths with spaces work without quoting since
    callers pass the entire tail-after-`/file ` as path_str.
    """
    from pathlib import Path

    if not path_str:
        console.print("[yellow]Usage:[/yellow] [bold]/file <path>[/bold]")
        return None

    path = Path(path_str).expanduser()
    try:
        path = path.resolve(strict=True)
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        console.print(f"[red]No such file: {path}[/red]")
        return None
    except UnicodeDecodeError:
        console.print(f"[red]File isn't UTF-8 text: {path}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Couldn't read {path}: {e}[/red]")
        return None

    n = len(content)
    console.print(f"[dim]· loaded {path.name}  ({n:,} chars)[/dim]")
    if n > 50_000:
        console.print(
            f"[yellow]warning: very large file ({n:,} chars) — "
            f"may exceed model context window or run up cost[/yellow]"
        )
    return content


def prompt_with_file(console: Console, prompt_text: str, **kwargs) -> str:
    """Like Rich's Prompt.ask, but also accepts `/file <path>`.

    If the user types `/file <path>`, the file is read (UTF-8) and its
    contents are returned in place of the typed input. This lets long
    pastes (voice memos, transcripts) bypass the terminal's canonical-mode
    line-input limit (~1KB on macOS).

    On any /file error returns an empty string (the caller's normal
    "blank input means cancel" path applies). Otherwise returns the
    user-typed text or the file contents.
    """
    from rich.prompt import Prompt

    raw = Prompt.ask(prompt_text, **kwargs).strip()
    if not raw.lower().startswith("/file"):
        return raw
    content = load_file_as_text(console, raw[len("/file"):].strip())
    return content if content is not None else ""


def print_join_request(console: Console, name: str) -> None:
    """Render a between-turn join request from an absent agent."""
    color = character_color(name)
    console.print()
    console.print(
        f"[dim italic]·[/dim italic] [bold {color}]{name}[/bold {color}]"
        f" [dim italic]wants to join.[/dim italic]  "
        f"[dim]/add to bring them in  ·  /ignore to dismiss[/dim]"
    )


def print_character_left(console: Console, name: str) -> None:
    """Render a brief, no-drama departure note when an agent is kicked."""
    color = character_color(name)
    console.print()
    console.print(
        f"[dim italic]·[/dim italic] [bold {color}]{name}[/bold {color}]"
        f" [dim italic]left the conversation.[/dim italic]"
    )


# --- World engine summary -------------------------------------------------

ENGINE_COUNT_FORMATS = [
    # (attr name on output object, label, optional special color)
    ("trait_updates", "trait updates", None),
    ("relationship_updates", "relationship updates", None),
    ("user_knowledge_updates", "things learned about you", None),
    ("structured_exchanges", "context exchanges", None),
    ("noteworthy_moments", "postcards", "magenta"),
]


def session_cost_line(calls: int, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> str:
    total_tokens = prompt_tokens + completion_tokens
    return (
        f"[dim]session cost[/dim]  "
        f"[bold]${cost_usd:.4f}[/bold]  "
        f"[dim]·  {calls} call{'s' if calls != 1 else ''}, "
        f"{total_tokens:,} tokens[/dim]"
    )


def world_engine_counts(out) -> str:
    """Compact bulleted counts. Skips zeros. Postcards highlighted."""
    lines: list[str] = []
    for attr, label, special in ENGINE_COUNT_FORMATS:
        n = len(getattr(out, attr))
        if n == 0:
            continue
        if special:
            lines.append(f"  • [bold {special}]{label}[/]: {n}")
        else:
            lines.append(f"  • [dim]{label}[/dim]: {n}")
    if not lines:
        return "[dim]  (no updates)[/dim]"
    return "\n".join(lines)
