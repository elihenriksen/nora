"""Environment validation — used by launcher startup and `nora doctor`.

Two surfaces:

  required_keys()    -> figure out which provider keys this config needs
  check_keys_present()  -> just check existence (fast, used at every launch)
  check_keys_live()  -> actually call each API to confirm the key works
                        (used by `nora doctor` and optionally at launch)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# --- Which keys does this config need? -----------------------------------

def required_keys() -> list[str]:
    """Return the env-var names this configuration needs.

    OpenAI is always required (world engine + moderator + structured outputs).
    Anthropic is required when the agent voice model is a Claude model.
    """
    needed = ["OPENAI_API_KEY"]
    char_model = os.environ.get("NORA_MODEL_CHARACTER", "gpt-4o-mini")
    if char_model.startswith("claude"):
        needed.append("ANTHROPIC_API_KEY")
    return needed


# --- Cheap presence check (no network) -----------------------------------

def check_keys_present() -> ValidationReport:
    """Verify env file + each required key has a non-empty value.

    Cheap (no network). Suitable for every launcher startup.
    """
    report = ValidationReport()

    report.checks.append(CheckResult(
        name=".env file exists",
        passed=config.ENV_PATH.exists(),
        detail=str(config.ENV_PATH),
    ))

    for key in required_keys():
        val = os.environ.get(key, "").strip()
        report.checks.append(CheckResult(
            name=f"{key} set",
            passed=bool(val),
            detail=("…" + val[-4:] if val else "missing"),
        ))
    return report


# --- Live API-key validation (one tiny call per provider) ----------------

def check_keys_live() -> ValidationReport:
    """Make a minimal API call to each required provider to confirm the key works.

    Slower (one network call per provider) and costs a fraction of a cent.
    Used by `nora doctor`, not by every launch.
    """
    report = ValidationReport()
    needed = required_keys()

    if "OPENAI_API_KEY" in needed:
        report.checks.append(_check_openai_live())
    if "ANTHROPIC_API_KEY" in needed:
        report.checks.append(_check_anthropic_live())
    return report


def _check_openai_live() -> CheckResult:
    try:
        from openai import OpenAI
        from openai import AuthenticationError
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=10.0)
        # models.list is the cheapest sanity check — no token cost.
        client.models.list()
        return CheckResult(name="OpenAI API reachable", passed=True, detail="models.list OK")
    except AuthenticationError as e:
        return CheckResult(name="OpenAI API reachable", passed=False, detail=f"auth failed: {e}")
    except Exception as e:
        return CheckResult(name="OpenAI API reachable", passed=False, detail=str(e))


def _check_anthropic_live() -> CheckResult:
    try:
        from anthropic import Anthropic
        from anthropic import AuthenticationError
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=10.0)
        # Smallest possible completion as a key sanity check (~10 input tokens).
        client.messages.create(
            model=os.environ.get("NORA_MODEL_CHARACTER", "claude-haiku-4-5-20251001"),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return CheckResult(name="Anthropic API reachable", passed=True, detail="ping OK")
    except AuthenticationError as e:
        return CheckResult(name="Anthropic API reachable", passed=False, detail=f"auth failed: {e}")
    except Exception as e:
        return CheckResult(name="Anthropic API reachable", passed=False, detail=str(e))


# --- DB / fixtures health ------------------------------------------------

def check_db_health() -> ValidationReport:
    from . import db, repo
    report = ValidationReport()

    report.checks.append(CheckResult(
        name="world database initialized",
        passed=db.is_initialized(),
        detail=str(config.DB_PATH),
    ))

    if db.is_initialized():
        try:
            with db.connect() as conn:
                row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
                version = row["v"] if row else None
                report.checks.append(CheckResult(
                    name="schema version present",
                    passed=version is not None,
                    detail=f"v{version}",
                ))
                # Spot-check that critical tables exist.
                expected = {"characters", "conversations", "messages", "usage", "messages_fts"}
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                ).fetchall()
                actual = {r["name"] for r in rows}
                missing = expected - actual
                report.checks.append(CheckResult(
                    name="all expected tables present",
                    passed=not missing,
                    detail="" if not missing else f"missing: {sorted(missing)}",
                ))
                # Agent count (informational, always passes).
                count = len(repo.list_characters(conn))
                report.checks.append(CheckResult(
                    name="agents in world",
                    passed=True,
                    detail=f"{count}",
                ))
                # User name (informational).
                name = repo.get_user_name(conn)
                report.checks.append(CheckResult(
                    name="user name set",
                    passed=name is not None,
                    detail=name or "not set — agents will say 'you'",
                ))
        except Exception as e:
            report.checks.append(CheckResult(
                name="schema reachable",
                passed=False, detail=str(e),
            ))
    return report


def check_fixtures() -> ValidationReport:
    from . import emergence
    report = ValidationReport()
    fixtures = emergence.list_fixture_names()
    report.checks.append(CheckResult(
        name="eval fixtures present",
        passed=bool(fixtures),
        detail=f"{len(fixtures)} ({', '.join(fixtures)})" if fixtures else "none found",
    ))
    return report
