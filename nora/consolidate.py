"""Memory consolidation pass.

Without consolidation, agents accumulate traits and relationship history
indefinitely. After many conversations an agent will have dozens of traits
on overlapping topics, contradictions left in place, and shared history
narratives that grow into unreadable walls of text. The dynamic system prompt
bloats and the agent drifts incoherent.

This module runs an LLM pass over an agent's accumulated state and
rewrites it: merging overlapping traits, resolving contradictions, demoting
weak traits that haven't been reinforced, capping the list, and shortening
old shared history.

The LLM call uses the world-engine model (gpt-4o by default).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from . import config, llm, repo, voice
from .models import USER_ID, Character


# --- Tunables -------------------------------------------------------------

MAX_TRAITS_PER_CHARACTER = 12
SHARED_HISTORY_SUMMARIZE_THRESHOLD_CHARS = 800  # if a relationship's shared history grows past this, summarize


# --- Output schemas ------------------------------------------------------

class ConsolidatedTrait(BaseModel):
    topic: str = Field(description="Short noun-phrase, e.g. 'patience'.")
    perspective: str = Field(description="The agent's view in their voice. 1–2 sentences.")
    strength: int = Field(ge=1, le=5)
    origin_summary: Optional[str] = Field(default=None, description="One short clause: where this view came from. Optional.")


class TraitConsolidationOutput(BaseModel):
    consolidated_traits: list[ConsolidatedTrait]
    notes: str = Field(description="One paragraph plain-language summary of what changed in this consolidation.")


class HistorySummary(BaseModel):
    summarized_history: str = Field(description="A shorter narrative of the shared history between the two agents on this topic.")


# --- Public surface ------------------------------------------------------

@dataclass
class ConsolidationResult:
    character_name: str
    before_traits: int
    after_traits: int
    notes: str
    history_summarized: int = 0  # number of relationship_context rows whose shared_history was summarized
    error: Optional[str] = None


def consolidate_character(
    conn: sqlite3.Connection,
    character: Character,
) -> ConsolidationResult:
    """Run a consolidation pass on one character.

    Steps:
      1. Read current traits.
      2. Ask the LLM to rewrite them into a tighter, max-12 set.
      3. Replace traits in the DB with the consolidated set.
      4. Summarize any relationship-context shared_history that's grown long.
    """
    traits = repo.list_traits(conn, character.id)
    before = len(traits)

    if before == 0:
        return ConsolidationResult(
            character_name=character.name, before_traits=0, after_traits=0,
            notes="(no traits to consolidate)",
        )

    try:
        new_traits, notes = _consolidate_traits(character, traits)
    except Exception as e:
        return ConsolidationResult(
            character_name=character.name, before_traits=before, after_traits=before,
            notes="", error=str(e),
        )

    # Replace traits atomically.
    conn.execute("DELETE FROM character_traits WHERE character_id = ?", (character.id,))
    for nt in new_traits:
        repo.upsert_trait(
            conn,
            character_id=character.id,
            topic=nt.topic.strip().lower(),
            perspective=nt.perspective.strip(),
            strength=nt.strength,
            origin_summary=nt.origin_summary,
            formed_via_add=[f"consolidated:{character.id}"],
        )

    # Shared-history summarization for relationships involving this character.
    summarized = _summarize_long_histories(conn, character)

    return ConsolidationResult(
        character_name=character.name,
        before_traits=before,
        after_traits=len(new_traits),
        notes=notes,
        history_summarized=summarized,
    )


def consolidate_all(conn: sqlite3.Connection) -> list[ConsolidationResult]:
    results: list[ConsolidationResult] = []
    for c in repo.list_characters(conn):
        results.append(consolidate_character(conn, c))
    return results


# --- Trait consolidation -------------------------------------------------

_TRAITS_SYSTEM = """You are consolidating the formed views of a single agent in a multi-agent system called Nora.

You will see the agent's identity (interest, seed traits, voice) and their current list of formed views. Your job is to produce a tighter, cleaner list of formed views that better represents who they actually are.

Rules:
1. MERGE traits that are about the same or overlapping topics. Combine their perspectives into one stronger statement (in the agent's voice).
2. RESOLVE CONTRADICTIONS. If two traits contradict each other, keep the more recent or stronger view. Where useful, note the evolution in origin_summary (e.g. "Earlier was less sure; came around through Candy's pushback.").
3. DEMOTE weak traits. If a trait is strength 1 and isn't reinforced by anything else, drop it unless it's genuinely distinctive.
4. CAP THE LIST. Output at most {max_traits} traits. Pick the {max_traits} that best capture this agent's actual identity — most distinct, most reinforced, most active in conversation.
5. KEEP THE VOICE. Each consolidated perspective should sound like the agent speaking, not like a bullet from a self-help book. No "embrace the journey" language.
6. STRENGTH should reflect how firmly held the view actually is given how it's appeared. 1 = passing mention, 5 = repeatedly defended.
7. Order by strength descending.

Output the new list and a short notes paragraph describing what you did (merges, drops, evolutions)."""


def _consolidate_traits(character: Character, traits: list) -> tuple[list[ConsolidatedTrait], str]:
    char_block = (
        f"NAME: {character.name}\n"
        f"INTEREST: {character.interest}\n"
        + (f"BACKSTORY: {character.backstory}\n" if character.backstory else "")
        + (f"VOICE: {character.voice_notes}\n" if character.voice_notes else "")
        + ("SEED TRAITS:\n" + "\n".join(f"  - {t}" for t in character.seed_traits) + "\n"
           if character.seed_traits else "")
    )
    traits_block = "\n".join(
        f"  - topic: {t.topic}\n    perspective: {t.perspective}\n    strength: {t.strength}"
        + (f"\n    origin: {t.origin_summary}" if t.origin_summary else "")
        for t in traits
    )

    user_text = (
        f"AGENT:\n{char_block}\n"
        f"CURRENT FORMED VIEWS ({len(traits)}):\n{traits_block}\n\n"
        "Consolidate per the rules. Output the new list and a notes paragraph."
    )

    with llm.usage_context(purpose="consolidate"):
        out = llm.structured(
            model=config.models().world_engine,
            system=_TRAITS_SYSTEM.format(max_traits=MAX_TRAITS_PER_CHARACTER),
            user=user_text,
            schema=TraitConsolidationOutput,
            temperature=0.3,
        )
    # Cap defensively in case the model exceeded the limit.
    return out.consolidated_traits[:MAX_TRAITS_PER_CHARACTER], out.notes


# --- Shared-history summarization ----------------------------------------

_HISTORY_SYSTEM = """You are summarizing a long shared-history narrative between two agents on one topic.

The narrative has accumulated over many conversations and is getting unwieldy. Produce a shorter narrative (3–5 sentences max) that preserves the key beats: what they've discussed, where they agreed, where they pushed back. Keep specific moments only if they shaped the relationship. Drop filler.

Output a single 'summarized_history' string."""


def _summarize_long_histories(conn: sqlite3.Connection, character: Character) -> int:
    rels = repo.list_relationships_for(conn, entity_type="character", entity_id=character.id)
    summarized = 0
    for rel in rels:
        ctx = repo.list_context_for_relationship(conn, rel.id)
        for entry in ctx:
            if not entry.shared_history:
                continue
            if len(entry.shared_history) < SHARED_HISTORY_SUMMARIZE_THRESHOLD_CHARS:
                continue
            # Identify the two participants for context.
            other_label = _other_label(conn, rel, character.id)
            user_text = (
                f"BETWEEN: {character.name}  and  {other_label}\n"
                f"TOPIC: {entry.topic}\n\n"
                f"CURRENT NARRATIVE ({len(entry.shared_history)} chars):\n{entry.shared_history}\n\n"
                "Summarize."
            )
            try:
                with llm.usage_context(purpose="consolidate"):
                    out = llm.structured(
                        model=config.models().world_engine,
                        system=_HISTORY_SYSTEM,
                        user=user_text,
                        schema=HistorySummary,
                        temperature=0.2,
                    )
            except Exception:
                continue
            # Replace shared_history (preserve perspectives).
            conn.execute(
                "UPDATE relationship_context SET shared_history = ?, last_updated_at = CURRENT_TIMESTAMP "
                "WHERE relationship_id = ? AND topic = ?",
                (out.summarized_history.strip(), rel.id, entry.topic),
            )
            summarized += 1
    return summarized


def _other_label(conn: sqlite3.Connection, rel, this_character_id: str) -> str:
    other_type, other_id = rel.other("character", this_character_id)
    if other_type == "user":
        return "the user"
    try:
        return repo.get_character_by_id(conn, other_id).name
    except LookupError:
        return "(unknown)"
