ALTER TABLE blob_gc_queue ADD COLUMN claimed_until_ms INTEGER;
ALTER TABLE blob_gc_queue ADD COLUMN claim_token TEXT;

CREATE INDEX IF NOT EXISTS blob_gc_queue_claim_idx
ON blob_gc_queue(not_before_ms, claimed_until_ms, enqueued_at_ms, backend_id, object_key);
