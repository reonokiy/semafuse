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
    CanonicalEvent, ConversionDiagnostic, DomainEventSequence, MemoryQuery, MemoryRecord,
    QueryMode, RawRecord, SearchHit, SearchQuery,
};
use anyhow::Result;
use async_trait::async_trait;

/// Outcome of one atomic event-ingestion operation.
#[derive(Debug, Default, Clone)]
pub struct IngestStats {
    /// Number of previously unseen raw records inserted.
    pub raw_inserted: usize,
    /// Number of raw records already present.
    pub raw_deduplicated: usize,
    /// Number of previously unseen fingerprints inserted.
    pub inserted: usize,
    /// Number of inputs already present by fingerprint.
    pub deduplicated: usize,
    /// Newly inserted events, preserving input order.
    pub inserted_events: Vec<CanonicalEvent>,
    /// Durable batch position, absent when no event was inserted or unsupported.
    pub domain_event_seq: Option<DomainEventSequence>,
}

/// One atomic raw-to-canonical persistence unit.
pub struct IngestBatch<'a> {
    /// Immutable source records.
    pub raw_records: &'a [RawRecord],
    /// Valid canonical projections.
    pub events: &'a [CanonicalEvent],
    /// Conversion and validation diagnostics.
    pub diagnostics: &'a [ConversionDiagnostic],
}

/// Storage and retrieval contract for immutable canonical events.
#[async_trait]
pub trait EventStore: Send + Sync {
    /// Ingests already-canonical events when no raw conversion batch is available.
    async fn ingest(&self, events: &[CanonicalEvent]) -> Result<IngestStats>;
    /// Atomically persists raw records, canonical projections, provenance, and diagnostics.
    async fn ingest_batch(&self, batch: IngestBatch<'_>) -> Result<IngestStats> {
        self.ingest(batch.events).await
    }
    /// Searches event text, applying repository and agent filters.
    async fn lexical_search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>>;
    /// Returns filtered events in reverse chronological order.
    async fn recent(&self, query: &SearchQuery) -> Result<Vec<SearchHit>>;
    /// Returns a session timeline in chronological order.
    async fn session_events(&self, session_id: &str) -> Result<Vec<CanonicalEvent>>;
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
