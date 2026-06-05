"""Runtime configuration for Nora.

Loads the .env file in the project root and exposes typed accessors. The DB
path is fixed to `nora.db` next to the .env so a clone of the repo is always
self-contained.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DB_PATH = PROJECT_ROOT / "nora.db"

# override=True makes the .env authoritative even when an empty/stale value
# already exists in the shell environment.
load_dotenv(ENV_PATH, override=True)


@dataclass(frozen=True)
class Models:
    character: str
    world_engine: str
    moderator: str


def models() -> Models:
    return Models(
        character=os.environ.get("NORA_MODEL_CHARACTER", "gpt-4o-mini"),
        world_engine=os.environ.get("NORA_MODEL_WORLD_ENGINE", "gpt-4o"),
        moderator=os.environ.get("NORA_MODEL_MODERATOR", "gpt-4o-mini"),
    )


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env in the project root."
        )
    return key


def require_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env in the project root "
            "(needed when NORA_MODEL_CHARACTER is a Claude model)."
        )
    return key


# Backwards-compatible alias.
require_api_key = require_openai_key
