"""LLM client wrapper.

A thin layer over the OpenAI and Anthropic SDKs that exposes two operations:

    chat(...)       -> str           # plain text completion
    structured(...) -> BaseModel     # validated JSON via pydantic schema

`chat()` routes by model id:
  - model starts with "claude" -> Anthropic SDK
  - otherwise                  -> OpenAI SDK

`structured()` is OpenAI-only. The world engine, moderator, and emergence
judge use it for schema-constrained generation.

Every call is recorded to the `usage` table with token counts and an estimated
USD cost. Callers wrap their code in `usage_context(...)` to associate calls
with a conversation and a purpose tag.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterator, Optional, Type, TypeVar

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from . import db
from .config import require_anthropic_key, require_openai_key

T = TypeVar("T", bound=BaseModel)


# --- Clients --------------------------------------------------------------

@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    return OpenAI(api_key=require_openai_key())


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=require_anthropic_key())


def prewarm_clients() -> None:
    """Instantiate both clients up-front so the first real call doesn't pay
    the TLS handshake / pool setup latency. Safe to call repeatedly (lru_cache).
    """
    import os
    import threading
    def _warm():
        try:
            _openai_client()
        except Exception:
            pass
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                _anthropic_client()
            except Exception:
                pass
    threading.Thread(target=_warm, daemon=True).start()


def _is_claude(model: str) -> bool:
    return model.startswith("claude")


# --- Pricing (USD per token) ---------------------------------------------

# Approximate public list prices. Used only for cost estimation in the usage
# table — not authoritative. Update as Anthropic/OpenAI change pricing.
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":             (2.50e-6,  10.00e-6),
    "gpt-4o-mini":        (0.15e-6,   0.60e-6),
    "gpt-4-turbo":        (10.00e-6, 30.00e-6),
    # Anthropic
    "claude-opus-4-7":               (15.00e-6, 75.00e-6),
    "claude-sonnet-4-6":             (3.00e-6,  15.00e-6),
    "claude-sonnet-4-20250514":      (3.00e-6,  15.00e-6),
    "claude-haiku-4-5-20251001":     (1.00e-6,   5.00e-6),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING.get(model)
    if rates is None:
        # Unknown model — return 0 so we still record tokens but don't lie about cost.
        return 0.0
    in_rate, out_rate = rates
    return prompt_tokens * in_rate + completion_tokens * out_rate


# --- Usage context (per-call attribution) --------------------------------

@dataclass
class _UsageContext:
    conversation_id: Optional[str] = None
    purpose: str = "unknown"
    # In-memory accumulator for the active session, so the CLI can print a
    # "session cost: $X (N calls)" line at the end without an extra query.
    session_calls: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cost: float = 0.0


_ctx = _UsageContext()


@contextmanager
def usage_context(*, conversation_id: Optional[str] = None, purpose: str = "unknown") -> Iterator[None]:
    """Set the conversation/purpose attributed to LLM calls inside the block."""
    prev = (_ctx.conversation_id, _ctx.purpose)
    _ctx.conversation_id = conversation_id
    _ctx.purpose = purpose
    try:
        yield
    finally:
        _ctx.conversation_id, _ctx.purpose = prev


def reset_session_totals() -> None:
    _ctx.session_calls = 0
    _ctx.session_prompt_tokens = 0
    _ctx.session_completion_tokens = 0
    _ctx.session_cost = 0.0


def session_totals() -> tuple[int, int, int, float]:
    """Returns (calls, prompt_tokens, completion_tokens, cost_usd) since reset."""
    return (
        _ctx.session_calls,
        _ctx.session_prompt_tokens,
        _ctx.session_completion_tokens,
        _ctx.session_cost,
    )


def _record(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    _ctx.session_calls += 1
    _ctx.session_prompt_tokens += prompt_tokens
    _ctx.session_completion_tokens += completion_tokens
    _ctx.session_cost += cost
    try:
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO usage "
                "(conversation_id, purpose, model, prompt_tokens, completion_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_ctx.conversation_id, _ctx.purpose, model, prompt_tokens, completion_tokens, cost),
            )
    except sqlite3.OperationalError:
        # The DB might not exist yet (e.g. running before nora init). Don't crash.
        pass


# --- chat -----------------------------------------------------------------

def chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    temperature: float = 0.85,
    max_tokens: int = 600,
    on_delta: Optional[Callable[[str], None]] = None,
) -> str:
    """Single completion. `messages` are role-tagged dicts (user|assistant).

    If `on_delta` is provided, the response is streamed and each text fragment
    is passed to the callback as it arrives. The full response is still
    returned at the end (and recorded in the usage table).
    """
    if _is_claude(model):
        # Wrap the system prompt as a content block with ephemeral caching.
        # On follow-up turns with the same system prompt, Anthropic returns a
        # cache hit — much faster TTFT and ~90% cheaper on cached tokens.
        system_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]

        if on_delta is not None:
            return _anthropic_stream(
                model=model, system=system_blocks, messages=messages,
                temperature=temperature, max_tokens=max_tokens, on_delta=on_delta,
            )

        resp = _anthropic_client().messages.create(
            model=model,
            system=system_blocks,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        if usage:
            _record(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        return text.strip()

    # OpenAI path
    full = [{"role": "system", "content": system}, *messages]

    if on_delta is not None:
        return _openai_stream(
            model=model, messages=full,
            temperature=temperature, max_tokens=max_tokens, on_delta=on_delta,
        )

    resp = _openai_client().chat.completions.create(
        model=model,
        messages=full,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        _record(
            model,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
    return (resp.choices[0].message.content or "").strip()


def _anthropic_stream(
    *, model: str, system: list, messages: list[dict[str, str]],
    temperature: float, max_tokens: int, on_delta: Callable[[str], None],
) -> str:
    """Stream from Anthropic, calling on_delta with each text chunk."""
    from . import timing as _timing
    t = _timing.timer(f"_anthropic_stream({model})")

    pieces: list[str] = []
    with _anthropic_client().messages.stream(
        model=model,
        system=system,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        t.step("stream context entered")
        first = True
        for delta in stream.text_stream:
            if first:
                t.step("first text chunk received (TTFT)")
                first = False
            pieces.append(delta)
            on_delta(delta)
        t.step(f"text_stream exhausted ({len(pieces)} chunks)")
        final = stream.get_final_message()
        t.step("get_final_message returned")
    t.step("stream context exited")
    usage = getattr(final, "usage", None)
    if usage:
        _record(model, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
    t.step("usage recorded")
    t.done()
    return "".join(pieces).strip()


def _openai_stream(
    *, model: str, messages: list[dict[str, str]],
    temperature: float, max_tokens: int, on_delta: Callable[[str], None],
) -> str:
    """Stream from OpenAI, calling on_delta with each text chunk."""
    pieces: list[str] = []
    stream = _openai_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
            if delta:
                pieces.append(delta)
                on_delta(delta)
        usage = getattr(chunk, "usage", None)
        if usage:
            _record(
                model,
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
            )
    return "".join(pieces).strip()


def structured(
    *,
    model: str,
    system: str,
    user: str,
    schema: Type[T],
    temperature: float = 0.4,
) -> T:
    """Constrained generation against a pydantic schema (OpenAI structured outputs)."""
    resp = _openai_client().chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
        temperature=temperature,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        _record(
            model,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        refusal = resp.choices[0].message.refusal
        raise RuntimeError(f"structured generation returned no parsed output: {refusal}")
    return parsed
