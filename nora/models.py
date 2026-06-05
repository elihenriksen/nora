"""Domain models — the in-memory shape of the social ontology.

Pydantic v2 used throughout for validation and JSON round-tripping. SQL rows
are converted into these models by the repo layer; LLM outputs are validated
against the same models or against subsets defined in `nora.world_engine`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


USER_ID = "user"  # The single implicit user. One per installation.

EntityType = Literal["user", "character"]
SenderType = Literal["user", "character", "system"]
ConversationType = Literal["1on1", "multi", "council"]


class Character(BaseModel):
    id: str
    name: str
    interest: str
    seed_traits: list[str] = Field(default_factory=list)
    backstory: Optional[str] = None
    voice_notes: Optional[str] = None
    created_at: Optional[datetime] = None

    @field_validator("name")
    @classmethod
    def _name_clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be empty")
        return v


class Conversation(BaseModel):
    id: str
    type: ConversationType
    topic: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


class Participant(BaseModel):
    type: EntityType
    id: str  # USER_ID for the user, character.id for characters


class Message(BaseModel):
    id: Optional[int] = None
    conversation_id: str
    sender_type: SenderType
    sender_id: str
    content: str
    created_at: Optional[datetime] = None


class Relationship(BaseModel):
    id: str
    a_type: EntityType
    a_id: str
    b_type: EntityType
    b_id: str

    def other(self, my_type: EntityType, my_id: str) -> tuple[EntityType, str]:
        """Return the (type, id) of the other side of this relationship."""
        if self.a_type == my_type and self.a_id == my_id:
            return self.b_type, self.b_id
        if self.b_type == my_type and self.b_id == my_id:
            return self.a_type, self.a_id
        raise ValueError(f"({my_type}, {my_id}) is not part of relationship {self.id}")


class RelationshipContextEntry(BaseModel):
    """A single topic-scoped slice of context for one relationship.

    Two perspectives are stored separately so each entity sees its own view.
    `shared_history` is the narrative both sides agree happened.
    """
    relationship_id: str
    topic: str
    a_perspective: Optional[str] = None
    b_perspective: Optional[str] = None
    shared_history: Optional[str] = None
    last_updated_at: Optional[datetime] = None
    last_conversation_id: Optional[str] = None


class CharacterTrait(BaseModel):
    """A character-level opinion. The emergent identity bucket.

    Aggregated over time as relational context accumulates. `formed_via` is a
    list of relationship ids that contributed.
    """
    character_id: str
    topic: str
    perspective: str
    strength: int = Field(ge=1, le=5, default=1)
    origin_summary: Optional[str] = None
    formed_via: list[str] = Field(default_factory=list)
    last_updated_at: Optional[datetime] = None


class ContextExchange(BaseModel):
    """A structured update one character sends to another.

    Represents the compressed inter-agent communication described in the deck.
    Type and payload are intentionally open: the world engine emits these in
    whatever shape is useful for the moment, validated downstream when read.
    """
    conversation_id: Optional[str] = None
    from_character_id: str
    to_character_id: str
    exchange_type: str
    payload: dict[str, Any]


class WorldEvent(BaseModel):
    conversation_id: Optional[str] = None
    event_type: str
    summary: str
    payload: Optional[dict[str, Any]] = None


class Postcard(BaseModel):
    id: str
    conversation_id: Optional[str] = None
    character_a_id: str
    character_b_id: str
    summary: str
    scene: Optional[str] = None
    created_at: Optional[datetime] = None


def canonicalize_pair(
    a_type: EntityType, a_id: str, b_type: EntityType, b_id: str
) -> tuple[EntityType, str, EntityType, str]:
    """Order an entity pair canonically for relationship storage.

    Rules: user always first. For two characters, lexicographic by id.
    """
    if a_type == "user" and b_type == "character":
        return a_type, a_id, b_type, b_id
    if a_type == "character" and b_type == "user":
        return b_type, b_id, a_type, a_id
    # both characters
    if a_id <= b_id:
        return a_type, a_id, b_type, b_id
    return b_type, b_id, a_type, a_id
