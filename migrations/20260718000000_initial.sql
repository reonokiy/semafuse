CREATE TABLE IF NOT EXISTS raw_records (
    id TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    agent TEXT NOT NULL CHECK(agent IN ('amp','codex','claude-code','opencode')),
    source TEXT NOT NULL,
    record_index INTEGER NOT NULL,
    native_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    ingested_at_ms INTEGER NOT NULL,
    UNIQUE(agent, source, record_index, digest)
);

CREATE INDEX IF NOT EXISTS raw_records_source_idx
ON raw_records(agent, source, record_index);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    spec_version TEXT NOT NULL CHECK(spec_version = 'semafuse.event/v1'),
    agent TEXT NOT NULL CHECK(agent IN ('amp','codex','claude-code','opencode')),
    session_id TEXT NOT NULL,
    repository TEXT,
    event_type TEXT NOT NULL CHECK(event_type IN ('session_started','session_ended','message','tool_call','tool_result','compaction','error','unknown')),
    occurred_at_ms INTEGER,
    ordering_timestamp_ms INTEGER NOT NULL,
    search_text TEXT NOT NULL,
    event_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_raw_records (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    raw_record_id TEXT NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    PRIMARY KEY(event_id, raw_record_id)
);

CREATE TABLE IF NOT EXISTS conversion_diagnostics (
    id TEXT PRIMARY KEY,
    raw_record_id TEXT NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    severity TEXT NOT NULL CHECK(severity IN ('warning','error')),
    code TEXT NOT NULL CHECK(length(code) > 0),
    message TEXT NOT NULL CHECK(length(message) > 0),
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS conversion_diagnostics_raw_idx
ON conversion_diagnostics(raw_record_id, created_at_ms);

CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id, ordering_timestamp_ms);
CREATE INDEX IF NOT EXISTS events_repo_idx ON events(repository, ordering_timestamp_ms DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
USING fts5(search_text, content='events', content_rowid='rowid');

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, search_text)
    VALUES('delete', old.rowid, old.search_text);
END;

CREATE TABLE IF NOT EXISTS domain_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_checkpoints (
    strategy TEXT PRIMARY KEY,
    last_seq INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    domain_event_seq INTEGER,
    error TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_event_ids_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY(namespace, id)
);

CREATE INDEX IF NOT EXISTS memories_namespace_updated_idx
ON memories(namespace, updated_at_ms DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(namespace UNINDEXED, id UNINDEXED, content);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, namespace, id, content)
    VALUES(new.rowid, new.namespace, new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, namespace, id, content)
    VALUES('delete', old.rowid, old.namespace, old.id, old.content);
    INSERT INTO memories_fts(rowid, namespace, id, content)
    VALUES(new.rowid, new.namespace, new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, namespace, id, content)
    VALUES('delete', old.rowid, old.namespace, old.id, old.content);
END;
