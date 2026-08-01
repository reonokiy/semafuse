ALTER TABLE events ADD COLUMN ingest_seq INTEGER;

UPDATE events SET ingest_seq = rowid WHERE ingest_seq IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS events_ingest_seq_idx
ON events(ingest_seq);

CREATE INDEX IF NOT EXISTS events_scoped_session_order_idx
ON events(agent, repository, session_id, ingest_seq);

CREATE TABLE IF NOT EXISTS strategy_event_vectors (
    namespace TEXT NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector BLOB NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY(namespace, event_id, model)
);

CREATE INDEX IF NOT EXISTS strategy_event_vectors_lookup_idx
ON strategy_event_vectors(namespace, model, event_id);
