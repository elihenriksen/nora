-- Nora schema. The social ontology made concrete.
--
-- Entities: characters (and the implicit single user).
-- Relationships: pairwise links between entities, canonicalized.
-- Relationship context: topic-scoped opinions and shared history per relationship.
-- Character traits: emergent identity aggregated at the character level.
-- Conversations + messages: dialogue history.
-- Context exchanges: the structured inter-agent communication protocol.
-- World events + postcards: noteworthy moments surfaced to the user.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS characters (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    interest        TEXT NOT NULL,
    seed_traits     TEXT,
    backstory       TEXT,
    voice_notes     TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL CHECK(type IN ('1on1', 'multi', 'council')),
    topic           TEXT,
    started_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at        TEXT,
    processed_at    TEXT
);

CREATE TABLE IF NOT EXISTS conversation_participants (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    participant_type TEXT NOT NULL CHECK(participant_type IN ('user', 'character')),
    participant_id  TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_type, participant_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_type     TEXT NOT NULL CHECK(sender_type IN ('user', 'character', 'system')),
    sender_id       TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS relationships (
    id              TEXT PRIMARY KEY,
    entity_a_type   TEXT NOT NULL CHECK(entity_a_type IN ('user', 'character')),
    entity_a_id     TEXT NOT NULL,
    entity_b_type   TEXT NOT NULL CHECK(entity_b_type IN ('user', 'character')),
    entity_b_id     TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (entity_a_type, entity_a_id, entity_b_type, entity_b_id)
);

CREATE TABLE IF NOT EXISTS relationship_context (
    id              TEXT PRIMARY KEY,
    relationship_id TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
    topic           TEXT NOT NULL,
    a_perspective   TEXT,
    b_perspective   TEXT,
    shared_history  TEXT,
    last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_conversation_id TEXT,
    UNIQUE (relationship_id, topic)
);

CREATE TABLE IF NOT EXISTS character_traits (
    id              TEXT PRIMARY KEY,
    character_id    TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    topic           TEXT NOT NULL,
    perspective     TEXT NOT NULL,
    strength        INTEGER NOT NULL DEFAULT 1 CHECK(strength BETWEEN 1 AND 5),
    origin_summary  TEXT,
    formed_via      TEXT,
    last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (character_id, topic)
);

CREATE TABLE IF NOT EXISTS context_exchanges (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    from_character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    to_character_id   TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    exchange_type   TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS world_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    event_type      TEXT NOT NULL,
    summary         TEXT NOT NULL,
    payload         TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS postcards (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    character_a_id  TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    character_b_id  TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    summary         TEXT NOT NULL,
    scene           TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    purpose         TEXT NOT NULL,            -- 'character' | 'world_engine' | 'moderator' | 'wants_to_speak' | 'consolidate' | 'eval'
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_usage_conv ON usage(conversation_id);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);

-- Full-text search over messages. Standalone FTS5 (duplicates content; small).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Tiny key-value store for app-wide settings (user name, etc.).
CREATE TABLE IF NOT EXISTS settings (
    key             TEXT PRIMARY KEY,
    value           TEXT
);

CREATE TABLE IF NOT EXISTS schema_version (
    version         INTEGER PRIMARY KEY
);
INSERT OR IGNORE INTO schema_version (version) VALUES (2);
