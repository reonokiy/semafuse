//! Agent-neutral event and memory data model.

mod archive;
mod event;
mod query;
mod source;
mod time;

use serde::Serialize;
use std::fmt;

pub use archive::{
    ArchiveSpecVersion, CanonicalArchive, ConversionDiagnostic, DiagnosticSeverity, RawRecord,
    canonical_archive_schema,
};
pub use event::{
    AgentKind, AttachmentRef, CanonicalEvent, ContentPart, ErrorInfo, EventData, EventEnvelope,
    EventKind, EventSpecVersion, MessageRole, ToolStatus,
};
pub use query::{
    MemoryMutation, MemoryQuery, MemoryRecord, QueryMode, SearchHit, SearchQuery,
    StrategyEventVector, VectorHit, VectorQuery,
};
pub use source::{EventSource, normalize_repository};
pub use time::now_ms;

/// Monotonically increasing position of a committed domain event.
///
/// Sequence zero represents the position before the first domain event. Values
/// persisted in `domain_events.seq` start at one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct DomainEventSequence(i64);

impl DomainEventSequence {
    /// Position before any committed domain event.
    pub const ORIGIN: Self = Self(0);

    /// Returns the durable integer position.
    pub const fn get(self) -> i64 {
        self.0
    }

    pub(crate) const fn from_database(value: i64) -> Self {
        Self(value)
    }
}

impl fmt::Display for DomainEventSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
