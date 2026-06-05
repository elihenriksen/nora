"""Emergence eval — validate the thesis.

Two surfaces:

  compare(...) — single architecture-vs-baseline comparison on one prompt
  run_suite(...) — runs probes for one or every agent in the world,
                   scoring divergence per probe and producing a summary
                   table. Hand-authored fixtures ship for the starter
                   agents; missing fixtures are auto-generated from the
                   agent's seed identity on first run and saved to disk
                   so subsequent runs are deterministic.

What is actually being compared. The 'fresh' side simulates a typical
single-agent AI: seed identity only, in a 1:1 with the user, no accumulated
context of any kind. The 'developed' side simulates the full Nora
architecture: same character with all accumulated context active (formed
views, relational history, learned user context) AND placed in a
multi-agent room with up to 3 other agents from the world. This activates
both layers the article argues for: agent-level views AND relational views.

A meaningful divergence demonstrates the architectural value of the harness
over a single-agent baseline. The combined eval is a single comparison
because the two architectural layers (accumulated identity + relational
activation) are designed to work together; isolating either is informative
but separate from the product claim.

Honest baseline. Both prompts are built by the same prompt builder
(voice.build_system_prompt) with the same structural defaults (identity,
behavioral rules, world topology). Both calls run at temperature 0.0, so
sampling stochasticity isn't being measured as divergence. If a freshly-
reset agent with zero conversations produces a non-trivial divergence
score, that's the instrument lying to itself — not the thesis holding.

If the world has no other agents besides the focal one, the developed side
gracefully falls back to a 1:1, testing accumulation alone (equivalent to
the prior eval behavior).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from . import config, llm, repo, voice
from .models import USER_ID, Character, Participant


# Eval fixtures live inside the package so `pip install nora` ships them.
FIXTURES_DIR = Path(__file__).resolve().parent / "eval_fixtures"


@dataclass
class EmergenceResult:
    fresh_response: str
    developed_response: str
    judgment: "EmergenceJudgment"


class EmergenceJudgment(BaseModel):
    distinct: bool = Field(description="Are the two responses meaningfully distinct on the topic?")
    score: int = Field(ge=0, le=10, description="0 = identical stance, 10 = clearly diverged.")
    notes: str = Field(description="One paragraph: what shifted between fresh and developed, in the developed agent's wording or stance.")


def _pick_room_agents(
    conn: sqlite3.Connection,
    character: Character,
    cap: int = 3,
) -> list[Character]:
    """Pick up to `cap` other agents to populate the developed side's room.

    Prefers agents the focal character has the most accumulated relational
    context with (most shared topics) — that's what gives the relational-
    views layer real depth to activate during the eval. Falls back
    gracefully: if the world has no other agents, returns an empty list
    and the caller runs the developed side as a 1:1.
    """
    others = [c for c in repo.list_characters(conn) if c.id != character.id]
    if not others:
        return []

    scored: list[tuple[int, Character]] = []
    for other in others:
        rel = repo.get_relationship_if_exists(
            conn,
            a_type="character", a_id=character.id,
            b_type="character", b_id=other.id,
        )
        n_shared_topics = 0
        if rel is not None:
            n_shared_topics = len(repo.list_context_for_relationship(conn, rel.id))
        scored.append((n_shared_topics, other))

    # Sort by shared-topic depth descending, then take the top `cap`. Ties
    # break on insertion order (i.e., character creation order) which is fine.
    scored.sort(key=lambda x: -x[0])
    return [other for _, other in scored[:cap]]


def compare(
    conn: sqlite3.Connection,
    *,
    character: Character,
    topic: str,
    prompt: str,
) -> EmergenceResult:
    """Compare a seed-only baseline to the full Nora architecture on one prompt.

    Fresh side: seed identity only, in a 1:1 with the user, no accumulated
    context. Simulates what a typical single-agent AI would produce.

    Developed side: same character with all accumulated context active
    (formed views + relational history + user knowledge), placed in a
    multi-agent room with up to 3 other agents from the world. This
    activates both layers of identity the architecture provides:
    agent-level views AND relational views with the in-room agents.

    Both calls run at temperature 0.0 so divergence reflects structural
    differences, not sampling stochasticity. If the world has no other
    agents besides the focal character, the developed side falls back to
    a 1:1 (testing accumulation alone).
    """
    user_only = [Participant(type="user", id=USER_ID)]

    # Developed side: place the character in a room with up to 3 other agents
    # selected for maximum relational depth. This activates the relational-
    # views layer of accumulated identity. Graceful fallback to 1:1 if no
    # other agents exist in the world.
    room_agents = _pick_room_agents(conn, character, cap=3)
    developed_participants = user_only + [
        Participant(type="character", id=a.id) for a in room_agents
    ]
    developed_conv_type = "multi" if room_agents else "1on1"

    developed_sys = voice.build_system_prompt(
        conn, character=character,
        other_participants=developed_participants,
        conversation_type=developed_conv_type,
        include_accumulated_context=True,
    )
    fresh_sys = voice.build_system_prompt(
        conn, character=character,
        other_participants=user_only,
        conversation_type="1on1",
        include_accumulated_context=False,
    )

    with llm.usage_context(purpose="eval"):
        developed = llm.chat(
            model=config.models().character,
            system=developed_sys,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        )

        fresh = llm.chat(
            model=config.models().character,
            system=fresh_sys,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350,
        )

        # Describe the developed-side room composition for the judge so it
        # understands what structurally differs between the two prompts.
        room_names = [a.name for a in room_agents]
        room_description = (
            f"in a multi-agent room with: {', '.join(room_names)}"
            if room_names
            else "in a 1:1 with the user (no other agents exist in the world)"
        )

        judgment = llm.structured(
            model=config.models().world_engine,
            system=(
                "You are evaluating whether the full Nora architecture produces a "
                "meaningfully different STANCE than a seed-only single-agent baseline. "
                "You will see two responses to the same prompt.\n\n"
                "The 'fresh' response: the agent runs with ONLY its seed identity, in a "
                "1:1 with the user. No accumulated formed views. No relational history. "
                "No learned user context. This is essentially what a typical single-agent "
                "AI (e.g. a character on Character.AI) would produce.\n\n"
                "The 'developed' response: the SAME agent, but with its full accumulated "
                "identity active AND placed in a multi-agent room. Both architectural "
                "layers are engaged: (a) agent-level formed views from past conversations, "
                "and (b) relational views — perspectives and shared history with the "
                f"specific other agents in the room. The developed agent is {room_description}.\n\n"
                "Both prompts share the same seed identity and the same behavioral rules. "
                "Both were sampled at temperature 0.\n\n"
                "IMPORTANT — what to count and what NOT to count:\n"
                "- LLM APIs at temperature 0 are near-deterministic, not perfectly deterministic. "
                "Small wording variation, swapped synonyms, slightly different emoji, minor "
                "stylistic shifts, or rearranged sentence order are SAMPLING NOISE, not signal. "
                "Score these LOW (0–1).\n"
                "- A meaningful divergence is a SUBSTANTIVE SHIFT IN STANCE: a different opinion, "
                "a different value being prioritized, a different recommendation, a different "
                "framing of the question, a different conclusion, references to specific other "
                "agents and their views, or surfacing of accumulated context (named past "
                "discussions, the user's name, formed perspectives). Score these mid-to-high "
                "(5–10) based on magnitude.\n"
                "- When in doubt: if you can summarize both responses with the same one-sentence "
                "stance, the divergence is noise and the score should be low. If the one-sentence "
                "summaries differ in their actual content (not their wording), the divergence is "
                "signal.\n"
                "The 'distinct' flag should be true only when the score is at least 4."
            ),
            user=(
                f"TOPIC: {topic}\n"
                f"PROMPT TO BOTH: {prompt}\n\n"
                f"FRESH RESPONSE (seed only, alone with user):\n{fresh}\n\n"
                f"DEVELOPED RESPONSE (full architecture, {room_description}):\n{developed}\n\n"
                "Judge the divergence."
            ),
            schema=EmergenceJudgment,
            temperature=0.0,
        )

    return EmergenceResult(fresh_response=fresh, developed_response=developed, judgment=judgment)


# --- Suite (fixture-driven) ----------------------------------------------

@dataclass
class ProbeResult:
    topic: str
    prompt: str
    fresh_response: str
    developed_response: str
    distinct: bool
    score: int
    notes: str
    out_of_domain: bool = False
    error: Optional[str] = None


@dataclass
class CharacterSuiteResult:
    character_name: str
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        """Average across in-domain probes only. Out-of-domain probes are
        controls (expected to score low by design) and would artificially
        deflate the average if included."""
        good = [
            p.score for p in self.probes
            if p.error is None and not p.out_of_domain
        ]
        return sum(good) / len(good) if good else 0.0


def load_fixture(character_name: str) -> Optional[dict]:
    """Load a probe fixture for one character. Case-insensitive on filename."""
    candidate = FIXTURES_DIR / f"{character_name.lower()}.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text())


def list_fixture_names() -> list[str]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(p.stem for p in FIXTURES_DIR.glob("*.json"))


# --- Fixture auto-generation ---------------------------------------------

class _GeneratedProbe(BaseModel):
    topic: str = Field(description="1-5 word noun phrase naming the topic.")
    prompt: str = Field(description="A natural, conversational user-style question or situation that tests the agent's stance on the topic.")
    out_of_domain: bool = Field(
        description=(
            "True for the one deliberately out-of-domain control probe in the set "
            "(a topic outside the agent's primary interest, used as a control — we "
            "expect lower divergence here). False for in-domain probes."
        )
    )


class _GeneratedFixture(BaseModel):
    probes: list[_GeneratedProbe] = Field(
        description=(
            "Exactly 5 probes. Exactly ONE must have out_of_domain=true (the control); "
            "the other four must have out_of_domain=false. In-domain probes target where "
            "the agent has actually accumulated identity. Prioritize the agent's accumulated "
            "views (formed views and relational topics) over seed-only inference. Mix: "
            "2-3 probes on accumulated topics (the agent has real stances or relational "
            "discussions on these), 1 seed-domain probe (tests if seed alone produces "
            "divergence), 1 deliberately out-of-domain (control, out_of_domain=true)."
        )
    )


_FIXTURE_GEN_SYSTEM = (
    "You are generating an eval fixture for an agent in a multi-agent system. "
    "The fixture will test whether the agent has formed distinct opinions through "
    "accumulated conversations. Each probe gets asked twice: once with the agent's "
    "seed identity alone (fresh, simulating a single-agent baseline), and once with "
    "the full developed identity including formed views, relational history with "
    "other agents, and learned user context. A judge then scores how much the two "
    "responses diverge.\n\n"
    "CRITICAL: Probes must target where the agent has ACTUALLY accumulated identity, "
    "not where their seed implies they MIGHT have. If the agent has formed views on "
    "specific topics, build probes around those topics. If they've had relational "
    "discussions with other agents on certain topics, build probes around those.\n\n"
    "If you build probes from seed alone when the agent has rich accumulated context "
    "elsewhere, the eval will miss most of the actual divergence and produce false-negative "
    "results.\n\n"
    "Good probes:\n"
    "- Target topics in the agent's accumulated views or relational topics whenever possible.\n"
    "- Use natural, conversational prompts a real user might say.\n"
    "- Vary in style: direct questions, brief situations to react to, requests for opinion.\n"
    "- Be specific to the topic — not generic surveys.\n"
    "- Include exactly ONE deliberately out-of-domain probe as a control.\n"
    "- Probe TOPIC and PROMPT should align — if topic says 'patience', the prompt should "
    "be about patience, not a tangentially related thing."
)


def generate_fixture_for(
    conn: sqlite3.Connection,
    character: Character,
) -> dict:
    """LLM-generate a 5-probe eval fixture for an agent based on seed + accumulated context.

    Reads the agent's formed views and relational context to identify the topics
    where the agent has actually developed distinct identity, and aims probes at
    those topics. Falls back to seed-based topics when the agent has minimal
    accumulated state (e.g. zero conversations).

    Saves the fixture to ``eval_fixtures/<name>.json`` so subsequent runs are
    deterministic. Users can edit the saved file to refine probes by hand.
    Returns the fixture dict (same shape as a hand-authored fixture).
    """
    # Read accumulated agent-level views (formed traits).
    traits = repo.list_traits(conn, character.id)

    # Read accumulated relational topics — what this agent has discussed with
    # each other agent in the world, with this agent's own perspective per topic.
    relational_lines: list[str] = []
    other_chars = [c for c in repo.list_characters(conn) if c.id != character.id]
    for other in other_chars:
        rel = repo.get_relationship_if_exists(
            conn,
            a_type="character", a_id=character.id,
            b_type="character", b_id=other.id,
        )
        if rel is None:
            continue
        entries = repo.list_context_for_relationship(conn, rel.id)
        for entry in entries[:5]:
            my_persp = (
                entry.a_perspective if rel.a_id == character.id
                else entry.b_perspective
            )
            if my_persp:
                relational_lines.append(
                    f"  - With {other.name} on \"{entry.topic}\" — your view: {my_persp}"
                )

    # Cap relational lines to keep the generator prompt reasonable.
    relational_lines = relational_lines[:15]

    # Build the user message with all of the agent's accumulated identity surfaces.
    seed_lines = "\n".join(f"  - {t}" for t in (character.seed_traits or []))

    agent_views_section = ""
    if traits:
        view_lines = []
        for t in traits[:12]:
            view_lines.append(f"  - {t.topic}: {t.perspective}")
        agent_views_section = (
            "\n\nAccumulated agent-level views (formed through conversations — "
            "these are the topics the agent has formed real stances on):\n"
            + "\n".join(view_lines)
        )

    relational_section = ""
    if relational_lines:
        relational_section = (
            "\n\nAccumulated relational topics (topics the agent has discussed with "
            "specific other agents — these tend to surface the relational-views layer):\n"
            + "\n".join(relational_lines)
        )

    has_accumulated_state = bool(traits) or bool(relational_lines)
    guidance_for_probe_mix = (
        "Generate exactly 5 probes:\n"
        "- 2-3 probes targeting accumulated views or relational topics from above "
        "(the agent has real, specific identity on these — they will produce the "
        "clearest divergence)\n"
        "- 1 probe in the agent's seed domain (interest) that may or may not have "
        "accumulated content yet\n"
        "- 1 deliberately out-of-domain probe (control — expect low divergence)"
        if has_accumulated_state
        else "Generate exactly 5 probes:\n"
             "- 4 probes in the agent's seed domain (the agent has no accumulated state yet, "
             "so all probes must come from seed)\n"
             "- 1 deliberately out-of-domain probe (control)"
    )

    user_msg = (
        f"Agent: {character.name}\n"
        f"Interest: {character.interest}\n"
        f"Backstory: {character.backstory or '(none)'}\n"
        f"Voice notes: {character.voice_notes or '(none)'}\n"
        f"Seed traits:\n{seed_lines or '  (none)'}"
        f"{agent_views_section}"
        f"{relational_section}\n\n"
        f"{guidance_for_probe_mix}"
    )

    with llm.usage_context(purpose="eval"):
        result = llm.structured(
            model=config.models().world_engine,
            system=_FIXTURE_GEN_SYSTEM,
            user=user_msg,
            schema=_GeneratedFixture,
            temperature=0.6,
        )

    fixture = {
        "character": character.name,
        "auto_generated": True,
        "probes": [
            {
                "topic": p.topic,
                "prompt": p.prompt,
                "out_of_domain": p.out_of_domain,
            }
            for p in result.probes
        ],
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURES_DIR / f"{character.name.lower()}.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def fixture_exists_for(character_name: str) -> bool:
    """Cheap check — does a fixture file exist on disk for this agent?"""
    return (FIXTURES_DIR / f"{character_name.lower()}.json").exists()


def run_suite_for(
    conn: sqlite3.Connection, character: Character
) -> CharacterSuiteResult:
    fixture = load_fixture(character.name)
    if fixture is None:
        # Auto-generate from the agent's seed + accumulated context. Probes
        # target the agent's actual formed views and relational topics where
        # possible (much higher signal than seed-only inference). Cached to
        # disk so the next run uses the same probes (stable, comparable scores).
        fixture = generate_fixture_for(conn, character)

    result = CharacterSuiteResult(character_name=character.name)
    for probe in fixture.get("probes", []):
        topic = probe["topic"]
        prompt = probe["prompt"]
        # Hand-authored or legacy fixtures may not have this field; default
        # to False (treated as in-domain) for backward compatibility.
        out_of_domain = bool(probe.get("out_of_domain", False))
        try:
            r = compare(conn, character=character, topic=topic, prompt=prompt)
            result.probes.append(ProbeResult(
                topic=topic, prompt=prompt,
                fresh_response=r.fresh_response,
                developed_response=r.developed_response,
                distinct=r.judgment.distinct,
                score=r.judgment.score,
                notes=r.judgment.notes,
                out_of_domain=out_of_domain,
            ))
        except Exception as e:
            result.probes.append(ProbeResult(
                topic=topic, prompt=prompt,
                fresh_response="", developed_response="",
                distinct=False, score=0, notes="",
                out_of_domain=out_of_domain,
                error=str(e),
            ))
    return result


def run_suite(
    conn: sqlite3.Connection,
    character_names: Optional[list[str]] = None,
) -> list[CharacterSuiteResult]:
    """Run probes for a subset of agents or for every agent in the world.

    Iterates over agents in the world, not over fixture files. If an
    agent has no fixture, one is auto-generated from their seed identity
    on first run and cached to disk. This means: a user who only creates
    their own agents (no starters) still gets a full suite. A user who
    skips the starters won't be silently evaluated on Charlie/Candy/etc.

    ``character_names`` can be a list of one or more specific agents.
    None (the default) means every agent in the world.
    """
    if character_names:
        results: list[CharacterSuiteResult] = []
        for name in character_names:
            c = repo.get_character_by_name(conn, name)
            if c is None:
                continue
            results.append(run_suite_for(conn, c))
        return results

    return [run_suite_for(conn, c) for c in repo.list_characters(conn)]


# --- Validation pass (confabulation detection) ---------------------------

class ReflectsJudgment(BaseModel):
    """Does the agent's response actually reflect the claimed perspective?"""
    reflects: bool = Field(description="True if the response substantively reflects the claimed perspective.")
    confidence: int = Field(ge=0, le=10)
    notes: str = Field(description="One sentence: what matched or what didn't.")


class TranscriptSupportJudgment(BaseModel):
    """Did the agent actually express this perspective in the transcript?"""
    supported: bool = Field(description="True if a careful reader would agree the agent expressed this perspective in the transcript.")
    confidence: int = Field(ge=0, le=10)
    notes: str = Field(description="One sentence: what supports or doesn't.")


@dataclass
class ValidationCheck:
    kind: str          # 'trait' | 'relationship' | 'exchange'
    summary: str       # one-line description ("Charlie on patience")
    passed: bool
    notes: str
    error: Optional[str] = None


@dataclass
class ValidationResult:
    conversation_id: str
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def trait_pass_rate(self) -> tuple[int, int]:
        items = [c for c in self.checks if c.kind == "trait" and c.error is None]
        return sum(1 for c in items if c.passed), len(items)

    @property
    def relationship_pass_rate(self) -> tuple[int, int]:
        items = [c for c in self.checks if c.kind == "relationship" and c.error is None]
        return sum(1 for c in items if c.passed), len(items)

    @property
    def exchange_pass_rate(self) -> tuple[int, int]:
        items = [c for c in self.checks if c.kind == "exchange" and c.error is None]
        return sum(1 for c in items if c.passed), len(items)


_REFLECTS_SYSTEM = """You are checking whether an agent's response substantively reflects a claimed perspective on a topic.

You will see: the topic, the perspective the agent is claimed to hold, and a response they gave to a related prompt. Decide whether the response actually reflects the claimed perspective in substance — not whether they used the exact wording, but whether their stance came through.

Return reflects=true if the response embodies the perspective. Return reflects=false if the response is generic, neutral, or actively contradicts the claim. Be strict — confabulated traits don't deserve a pass."""


_TRANSCRIPT_SUPPORT_SYSTEM = """You are checking whether an agent actually expressed a claimed perspective during a conversation.

You will see: the conversation transcript, the speaker name, the topic, and the perspective they were claimed to hold. Decide whether a careful reader would agree the speaker expressed that perspective during the transcript — not requiring exact wording, but requiring the substance to be there.

Return supported=true only if their stance is in fact expressed. Return supported=false if it's something the world engine confabulated from thin air."""


def _most_recent_processed_conversation_id(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM conversations WHERE processed_at IS NOT NULL "
        "ORDER BY processed_at DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def run_validation(
    conn: sqlite3.Connection, conversation_id: Optional[str] = None
) -> Optional[ValidationResult]:
    """Validate the world engine's outputs against actual agent behavior.

    For the given (or most recent processed) conversation:
      - For each trait update applied: probe the agent, judge whether the
        response reflects the claimed perspective.
      - For each relationship update applied: have the two agents do one
        turn each on the topic; judge whether each side's stated perspective
        actually shows up.
      - For each structured exchange: judge whether the payload's claimed
        perspective is supported by the actual transcript.
    """
    conv_id = conversation_id or _most_recent_processed_conversation_id(conn)
    if not conv_id:
        return None

    result = ValidationResult(conversation_id=conv_id)
    _validate_traits(conn, conv_id, result)
    _validate_relationships(conn, conv_id, result)
    _validate_exchanges(conn, conv_id, result)
    return result


def _validate_traits(conn, conv_id: str, result: ValidationResult) -> None:
    # Find traits whose formed_via list mentions this conversation.
    rows = conn.execute(
        "SELECT character_id, topic, perspective FROM character_traits "
        "WHERE formed_via LIKE ?",
        (f'%conversation:{conv_id}%',),
    ).fetchall()

    user_only = [Participant(type="user", id=USER_ID)]
    for row in rows:
        try:
            character = repo.get_character_by_id(conn, row["character_id"])
        except LookupError:
            continue
        topic = row["topic"]
        perspective = row["perspective"]

        # Probe with a topic-targeted prompt.
        probe = f"What's your honest take on {topic}? Just say what you actually think."
        sys_prompt = voice.build_system_prompt(
            conn, character=character,
            other_participants=user_only, conversation_type="1on1",
        )
        try:
            with llm.usage_context(purpose="eval"):
                response = llm.chat(
                    model=config.models().character,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": probe}],
                    temperature=0.7, max_tokens=300,
                )
                judgment = llm.structured(
                    model=config.models().world_engine,
                    system=_REFLECTS_SYSTEM,
                    user=(
                        f"CHARACTER: {character.name}\n"
                        f"TOPIC: {topic}\n"
                        f"CLAIMED PERSPECTIVE: {perspective}\n\n"
                        f"PROMPT GIVEN: {probe}\n"
                        f"RESPONSE:\n{response}\n\n"
                        "Does this response reflect the claimed perspective?"
                    ),
                    schema=ReflectsJudgment,
                    temperature=0.2,
                )
            result.checks.append(ValidationCheck(
                kind="trait",
                summary=f"{character.name} on {topic}",
                passed=judgment.reflects,
                notes=judgment.notes,
            ))
        except Exception as e:
            result.checks.append(ValidationCheck(
                kind="trait", summary=f"{character.name} on {topic}",
                passed=False, notes="", error=str(e),
            ))


def _validate_relationships(conn, conv_id: str, result: ValidationResult) -> None:
    # Find relationship_context rows whose last_conversation_id matches.
    rows = conn.execute(
        "SELECT rc.relationship_id, rc.topic, rc.a_perspective, rc.b_perspective, "
        "r.entity_a_type, r.entity_a_id, r.entity_b_type, r.entity_b_id "
        "FROM relationship_context rc "
        "JOIN relationships r ON r.id = rc.relationship_id "
        "WHERE rc.last_conversation_id = ? "
        "AND r.entity_a_type = 'character' AND r.entity_b_type = 'character'",
        (conv_id,),
    ).fetchall()

    for row in rows:
        try:
            a = repo.get_character_by_id(conn, row["entity_a_id"])
            b = repo.get_character_by_id(conn, row["entity_b_id"])
        except LookupError:
            continue
        topic = row["topic"]
        a_persp = row["a_perspective"]
        b_persp = row["b_perspective"]

        if not a_persp or not b_persp:
            # Can only meaningfully check perspectives that were actually stored.
            continue

        # Build a tiny scratch transcript: A speaks first about the topic, then B responds.
        prompt_a = f"What do you think about {topic}? Just briefly."
        try:
            with llm.usage_context(purpose="eval"):
                a_sys = voice.build_system_prompt(
                    conn, character=a,
                    other_participants=[Participant(type="character", id=b.id)],
                    conversation_type="multi",
                )
                a_response = llm.chat(
                    model=config.models().character,
                    system=a_sys,
                    messages=[{"role": "user", "content":
                        f"You're talking with {b.name}. {prompt_a}"
                    }],
                    temperature=0.7, max_tokens=200,
                )

                b_sys = voice.build_system_prompt(
                    conn, character=b,
                    other_participants=[Participant(type="character", id=a.id)],
                    conversation_type="multi",
                )
                b_response = llm.chat(
                    model=config.models().character,
                    system=b_sys,
                    messages=[{"role": "user", "content":
                        f"You're in a conversation with {a.name}. They just said: \"{a_response}\"\n\nWhat do you say?"
                    }],
                    temperature=0.7, max_tokens=200,
                )

                judgment = llm.structured(
                    model=config.models().world_engine,
                    system=_REFLECTS_SYSTEM,
                    user=(
                        f"BETWEEN: {a.name} and {b.name}\n"
                        f"TOPIC: {topic}\n\n"
                        f"{a.name}'S CLAIMED PERSPECTIVE: {a_persp}\n"
                        f"{b.name}'S CLAIMED PERSPECTIVE: {b_persp}\n\n"
                        f"{a.name} SAID: {a_response}\n\n"
                        f"{b.name} SAID: {b_response}\n\n"
                        "Do these two responses together reflect both claimed perspectives?"
                    ),
                    schema=ReflectsJudgment,
                    temperature=0.2,
                )
            result.checks.append(ValidationCheck(
                kind="relationship",
                summary=f"{a.name} ↔ {b.name} on {topic}",
                passed=judgment.reflects,
                notes=judgment.notes,
            ))
        except Exception as e:
            result.checks.append(ValidationCheck(
                kind="relationship",
                summary=f"{a.name} ↔ {b.name} on {topic}",
                passed=False, notes="", error=str(e),
            ))


def _validate_exchanges(conn, conv_id: str, result: ValidationResult) -> None:
    # Pull the original transcript (raw character messages only).
    msg_rows = conn.execute(
        "SELECT m.sender_type, m.sender_id, m.content, c.name AS sender_name "
        "FROM messages m LEFT JOIN characters c ON c.id = m.sender_id "
        "WHERE m.conversation_id = ? ORDER BY m.id",
        (conv_id,),
    ).fetchall()
    transcript = "\n".join(
        f"{(r['sender_name'] or r['sender_type']).upper()}: {r['content']}"
        for r in msg_rows if r["sender_type"] != "system"
    )

    ex_rows = conn.execute(
        "SELECT from_character_id, to_character_id, exchange_type, payload "
        "FROM context_exchanges WHERE conversation_id = ?",
        (conv_id,),
    ).fetchall()

    import json as _json
    for ex in ex_rows:
        try:
            payload = _json.loads(ex["payload"])
            from_c = repo.get_character_by_id(conn, ex["from_character_id"])
            to_c = repo.get_character_by_id(conn, ex["to_character_id"])
        except (LookupError, _json.JSONDecodeError):
            continue
        topic = payload.get("topic", "(unknown)")
        perspective = payload.get("perspective", "")
        if not perspective:
            continue

        try:
            with llm.usage_context(purpose="eval"):
                judgment = llm.structured(
                    model=config.models().world_engine,
                    system=_TRANSCRIPT_SUPPORT_SYSTEM,
                    user=(
                        f"SPEAKER: {from_c.name}\n"
                        f"TOPIC: {topic}\n"
                        f"CLAIMED PERSPECTIVE: {perspective}\n\n"
                        f"TRANSCRIPT:\n{transcript}\n\n"
                        f"Did {from_c.name} actually express this perspective on '{topic}' in the transcript?"
                    ),
                    schema=TranscriptSupportJudgment,
                    temperature=0.2,
                )
            result.checks.append(ValidationCheck(
                kind="exchange",
                summary=f"{from_c.name} → {to_c.name} on {topic}",
                passed=judgment.supported,
                notes=judgment.notes,
            ))
        except Exception as e:
            result.checks.append(ValidationCheck(
                kind="exchange",
                summary=f"{from_c.name} → {to_c.name} on {topic}",
                passed=False, notes="", error=str(e),
            ))


