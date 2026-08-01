//! Persistence contracts for canonical events and strategy-derived memories.
//!
//! Event stores deduplicate by canonical fingerprint. Durable implementations
//! persist explicit raw provenance and append one domain event for each
//! transaction that inserts at least one previously unseen canonical event.

mod memory;
mod sqlite;

pub use memory::MemoryStore;
pub use sqlite::SqliteStore;

use crate::model::{
    AgentKind, CanonicalEvent, ConversionDiagnostic, DomainEventSequence, MemoryQuery,
    MemoryRecord, QueryMode, RawRecord, SearchHit, SearchQuery, VectorHit, VectorQuery,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::Serialize;

/// Outcome of one atomic event-ingestion operation.
#[derive(Debug, Default, Clone)]
pub struct IngestStats {
    /// Number of previously unseen raw records inserted.
    pub raw_inserted: usize,
    /// Number of raw records already present.
    pub raw_deduplicated: usize,
    /// Number of native records rejected by a durable forget rule.
    pub raw_tombstoned: usize,
    /// Number of previously unseen fingerprints inserted.
    pub inserted: usize,
    /// Number of inputs already present by fingerprint.
    pub deduplicated: usize,
    /// Number of canonical events rejected by a durable forget rule.
    pub tombstoned: usize,
    /// Newly inserted events, preserving input order.
    pub inserted_events: Vec<CanonicalEvent>,
    /// Durable batch position, absent when no event was inserted or unsupported.
    pub domain_event_seq: Option<DomainEventSequence>,
    /// Whether replay-triggered legacy cleanup still needs a physical scrub.
    pub sqlite_cleanup_pending: bool,
    /// Last physical-cleanup failure after a replay-triggered deletion committed.
    pub sqlite_cleanup_error: Option<String>,
    /// Legacy managed attachment objects durably queued by replay cleanup.
    pub attachment_objects_queued: usize,
    /// Legacy or foreign attachment objects that cannot be safely deleted.
    pub unmanaged_attachment_objects: usize,
}

/// One atomic raw-to-canonical persistence unit.
pub struct IngestBatch<'a> {
    /// Immutable source records.
    pub raw_records: &'a [RawRecord],
    /// Session, repository, and source-time ownership for every raw record.
    pub raw_contexts: &'a [RawContext],
    /// Valid canonical projections.
    pub events: &'a [CanonicalEvent],
    /// Conversion and validation diagnostics.
    pub diagnostics: &'a [ConversionDiagnostic],
}

/// Deletion-relevant ownership retained separately from the immutable raw payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RawContext {
    /// Raw record to which this context belongs.
    pub raw_record_id: String,
    /// Source session, when known.
    pub session_id: Option<String>,
    /// Normalized repository/worktree identifier, when known.
    pub repository: Option<String>,
    /// Native source time; ingestion time is used when absent.
    pub occurred_at_ms: Option<i64>,
}

/// One canonical event paired with its stable, store-wide ingestion ordinal.
///
/// Native timestamps are retained on the event but are not reliable enough to
/// reconstruct the original message sequence when clocks are duplicated or
/// skewed. Query-time memory processors use this ordinal for exact replay.
#[derive(Debug, Clone)]
pub struct OrderedEvent {
    /// One-based durable position assigned when the event was first inserted.
    pub ordinal: u64,
    /// Immutable canonical event at that position.
    pub event: CanonicalEvent,
}

/// Explicit selectors for destructive event removal.
///
/// At least one field must be present. `before_ms` matches an event's effective
/// ordering timestamp strictly before the supplied bound.
#[derive(Debug, Clone, Default)]
pub struct ForgetFilter {
    /// Canonical source session ID.
    pub session_id: Option<String>,
    /// Normalized repository or worktree identifier.
    pub repository: Option<String>,
    /// Source agent.
    pub agent: Option<AgentKind>,
    /// Exclusive retention cutoff in Unix milliseconds.
    pub before_ms: Option<i64>,
}

impl ForgetFilter {
    /// Rejects an unbounded removal operation.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.session_id.is_some()
                || self.repository.is_some()
                || self.agent.is_some()
                || self.before_ms.is_some(),
            "forget requires at least one of session, repository, agent, or before"
        );
        anyhow::ensure!(
            self.session_id
                .as_deref()
                .is_none_or(|value| !value.trim().is_empty()),
            "forget session selector cannot be empty"
        );
        anyhow::ensure!(
            self.repository
                .as_deref()
                .is_none_or(|value| !value.trim().is_empty()),
            "forget repository selector cannot be empty"
        );
        Ok(())
    }
}

/// Read-only impact report for a forget operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ForgetPreview {
    /// Canonical events that match every supplied selector.
    pub events: usize,
    /// Raw records that become unreferenced and can be deleted.
    pub raw_records: usize,
    /// Attachment objects that become unreferenced and enter durable GC.
    pub attachment_objects: usize,
    /// Legacy or foreign keys that are logically forgotten but cannot be safely deleted.
    pub unmanaged_attachment_objects: usize,
}

/// Committed result of a forget operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ForgetStats {
    /// Canonical events deleted.
    pub events: usize,
    /// Raw records deleted after their last event reference disappeared.
    pub raw_records: usize,
    /// Attachment objects durably queued for deletion.
    pub attachment_objects_queued: usize,
    /// Legacy or foreign keys intentionally left untouched to avoid cross-store deletion.
    pub unmanaged_attachment_objects: usize,
    /// Canonical events remaining after deletion.
    pub remaining_events: usize,
    /// Whether SQLite/WAL physical cleanup still needs a retry.
    pub sqlite_cleanup_pending: bool,
    /// Last physical-cleanup failure, when logical deletion already committed.
    pub sqlite_cleanup_error: Option<String>,
}

/// One pending attachment deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobGcEntry {
    /// Stable fingerprint of the configured OpenDAL backend.
    pub backend_id: String,
    /// Backend object key.
    pub object_key: String,
    /// Failed deletion attempts already recorded.
    pub attempts: i64,
    /// Opaque lease proving this worker owns the deletion attempt.
    pub claim_token: String,
}

/// Storage and retrieval contract for immutable canonical events.
#[async_trait]
pub trait EventStore: Send + Sync {
    /// Ingests already-canonical events when no raw conversion batch is available.
    /// Durable implementations require every declared raw provenance ID to
    /// already exist and reject events without provenance.
    async fn ingest(&self, events: &[CanonicalEvent]) -> Result<IngestStats>;
    /// Atomically persists raw records, canonical projections, provenance, and diagnostics.
    async fn ingest_batch(&self, batch: IngestBatch<'_>) -> Result<IngestStats> {
        self.ingest(batch.events).await
    }
    /// Searches event text, applying repository and agent filters.
    async fn lexical_search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>>;
    /// Returns every non-empty message/tool event in stable ingestion order.
    ///
    /// Unlike ranked search, this method does not apply `query.limit`. It is the
    /// exact corpus view used by retrieval methods such as Okapi BM25.
    async fn scoped_message_events(&self, query: &SearchQuery) -> Result<Vec<CanonicalEvent>>;
    /// Returns filtered events in reverse chronological order.
    async fn recent(&self, query: &SearchQuery) -> Result<Vec<SearchHit>>;
    /// Returns a session timeline in chronological order.
    async fn session_events(&self, session_id: &str) -> Result<Vec<CanonicalEvent>>;
    /// Returns a session timeline with stable store-wide ingestion positions.
    ///
    /// Stores without a durable ordinal may use the deterministic session
    /// order as a fallback. Durable implementations should override this.
    async fn ordered_session_events(&self, session_id: &str) -> Result<Vec<OrderedEvent>> {
        self.session_events(session_id).await.map(|events| {
            events
                .into_iter()
                .enumerate()
                .map(|(index, event)| OrderedEvent {
                    ordinal: index as u64 + 1,
                    event,
                })
                .collect()
        })
    }
    /// Counts unique canonical events.
    async fn count(&self) -> Result<usize>;

    /// Routes a query to the selected retrieval mode.
    async fn query(&self, mode: QueryMode, query: &SearchQuery) -> Result<Vec<SearchHit>>;
}

/// Read-only access to namespace-isolated strategy memories.
#[async_trait]
pub trait MemoryRepository: Send + Sync {
    /// Gets one record by its namespace-local identity.
    async fn get_memory(&self, namespace: &str, id: &str) -> Result<Option<MemoryRecord>>;
    /// Searches only the requested namespace, or returns recent records for empty text.
    async fn query_memories(&self, query: &MemoryQuery) -> Result<Vec<MemoryRecord>>;
}

/// Read-only dense retrieval over strategy-owned event vectors.
#[async_trait]
pub trait VectorRepository: Send + Sync {
    /// Returns cosine-ranked canonical events within the supplied event scope.
    async fn query_vectors(&self, query: &VectorQuery) -> Result<Vec<VectorHit>>;
    /// Returns exact event IDs covered by one model-specific vector index scope.
    async fn indexed_event_ids(
        &self,
        namespace: &str,
        model: &str,
        scope: &SearchQuery,
    ) -> Result<Vec<String>>;
}

/// Read-only consistency information for checkpointed strategy state.
#[async_trait]
pub trait StrategyStateRepository: Send + Sync {
    /// Returns `(strategy checkpoint, domain-event high-water)` from one snapshot.
    async fn strategy_progress(
        &self,
        strategy: &str,
    ) -> Result<(DomainEventSequence, DomainEventSequence)>;
}
