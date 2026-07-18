//! Local, agent-neutral memory for coding-agent conversations.
//!
//! Agent-specific [`adapters`] normalize imports and live hook payloads into
//! immutable [`CanonicalEvent`] values and content-addressed attachments. Event
//! [`storage`] deduplicates those values while retaining provenance; callers can
//! search them directly through [`retrieval`] or feed the durable domain-event
//! log to checkpointed [`strategy`] implementations that maintain derived
//! memories. [`importer`], [`hooks`], and [`mcp`] are the principal entry points.
#![warn(missing_docs)]

pub mod adapters;
/// Content-addressed attachment storage and extraction.
pub mod blob;
/// Configuration defaults, loading, and validation.
pub mod config;
/// Real-time hook ingestion and durable spool replay.
pub mod hooks;
/// Agent history discovery and import.
pub mod importer;
/// Model Context Protocol server over standard input and output.
pub mod mcp;
/// Canonical events, queries, memories, and shared value types.
pub mod model;
/// Raw-event retrieval strategies.
pub mod retrieval;
/// Event and derived-memory persistence.
pub mod storage;
/// Checkpointed derived-memory strategies and recall routing.
pub mod strategy;

pub use model::{
    AgentKind, ArchiveSpecVersion, AttachmentRef, CanonicalArchive, CanonicalEvent, ContentPart,
    ConversionDiagnostic, DiagnosticSeverity, DomainEventSequence, ErrorInfo, EventData,
    EventEnvelope, EventKind, EventSource, EventSpecVersion, MemoryMutation, MemoryQuery,
    MemoryRecord, MessageRole, QueryMode, RawRecord, SearchHit, SearchQuery, ToolStatus,
    canonical_archive_schema,
};
