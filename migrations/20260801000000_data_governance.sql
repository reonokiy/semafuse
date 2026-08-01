CREATE TABLE IF NOT EXISTS storage_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO storage_metadata(key, value)
VALUES('storage_id', lower(hex(randomblob(16))));

CREATE TABLE IF NOT EXISTS raw_contexts (
    raw_record_id TEXT PRIMARY KEY REFERENCES raw_records(id) ON DELETE CASCADE,
    session_id TEXT,
    repository TEXT,
    occurred_at_ms INTEGER,
    CHECK(session_id IS NULL OR length(trim(session_id)) > 0),
    CHECK(repository IS NULL OR length(trim(repository)) > 0)
);

CREATE INDEX IF NOT EXISTS raw_contexts_session_idx
ON raw_contexts(session_id, occurred_at_ms);

CREATE INDEX IF NOT EXISTS raw_contexts_repository_idx
ON raw_contexts(repository, occurred_at_ms);

CREATE TABLE IF NOT EXISTS event_attachments (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    object_key TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY(event_id, object_key)
);

CREATE INDEX IF NOT EXISTS event_attachments_object_idx
ON event_attachments(object_key, event_id);

CREATE TABLE IF NOT EXISTS forget_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_hash TEXT,
    repository_hash TEXT,
    agent TEXT CHECK(agent IS NULL OR agent IN ('amp','codex','claude-code','opencode')),
    before_ms INTEGER,
    created_at_ms INTEGER NOT NULL,
    CHECK(
        session_hash IS NOT NULL OR repository_hash IS NOT NULL OR
        agent IS NOT NULL OR before_ms IS NOT NULL
    ),
    CHECK(session_hash IS NULL OR length(session_hash) = 64),
    CHECK(repository_hash IS NULL OR length(repository_hash) = 64)
);

-- SQLite considers NULL values distinct in ordinary UNIQUE constraints. Use
-- explicit sentinels so repeating the same bounded deletion is idempotent.
CREATE UNIQUE INDEX IF NOT EXISTS forget_rules_selectors_unique
ON forget_rules(
    COALESCE(session_hash, ''),
    COALESCE(repository_hash, ''),
    COALESCE(agent, ''),
    COALESCE(CAST(before_ms AS TEXT), '')
);

CREATE TABLE IF NOT EXISTS forgotten_raw_records (
    raw_record_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS forgotten_events (
    event_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS blob_gc_queue (
    backend_id TEXT NOT NULL,
    object_key TEXT NOT NULL,
    reason TEXT NOT NULL CHECK(reason IN ('forgotten','staged')),
    enqueued_at_ms INTEGER NOT NULL,
    not_before_ms INTEGER NOT NULL,
    last_error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(backend_id, object_key)
);

CREATE INDEX IF NOT EXISTS blob_gc_queue_enqueued_idx
ON blob_gc_queue(not_before_ms, enqueued_at_ms, backend_id, object_key);

CREATE TABLE IF NOT EXISTS unmanaged_blob_residue (
    backend_id TEXT NOT NULL,
    object_key TEXT NOT NULL,
    reason TEXT NOT NULL,
    noted_at_ms INTEGER NOT NULL,
    PRIMARY KEY(backend_id, object_key)
);

INSERT INTO events_fts(events_fts, rank) VALUES('secure-delete', 1);
INSERT INTO memories_fts(memories_fts, rank) VALUES('secure-delete', 1);
