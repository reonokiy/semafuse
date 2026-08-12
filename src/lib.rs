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
/// Isolated JSONL benchmark support that exercises the production strategies.
#[cfg(feature = "benchmarks")]
pub mod benchmark;
/// Deterministic model-free retrieval and oracle baselines.
#[cfg(feature = "benchmarks")]
pub mod benchmark_baselines;
/// Complete compile-time source identity for reproducible benchmark artifacts.
#[cfg(feature = "benchmarks")]
#[doc(hidden)]
pub mod benchmark_identity;
/// Crash-safe, identity-checked JSONL benchmark journaling.
#[cfg(feature = "benchmarks")]
pub mod benchmark_journal;
/// Retrieval metrics with both standard and LongMemEval-compatible nDCG.
#[cfg(feature = "benchmarks")]
pub mod benchmark_metrics;
/// Fail-closed aggregation and paired bootstrap comparisons.
#[cfg(feature = "benchmarks")]
pub mod benchmark_report;
/// Native modular LongMemEval suite orchestration.
#[cfg(feature = "benchmarks")]
pub mod benchmark_suite;
/// Content-addressed attachment storage and extraction.
pub mod blob;
/// Configuration defaults, loading, and validation.
pub mod config;
/// Foreground daemon, local services, and Unix-socket server.
#[cfg(feature = "daemon")]
#[allow(missing_docs)]
pub mod daemon;
#[doc(hidden)]
pub mod daemon_ipc;
/// Portable export and explicit retention operations.
pub mod governance;
/// Real-time hook ingestion and durable spool replay.
pub mod hooks;
/// Agent history discovery and import.
pub mod importer;
mod inference;
/// Label-isolated native loader for the pinned LoCoMo release.
#[cfg(feature = "benchmarks")]
pub mod locomo;
#[cfg(feature = "benchmarks")]
mod locomo_answer_protocol;
/// End-to-end LoCoMo memory construction, common answering, and official F1.
#[cfg(feature = "benchmarks")]
pub mod locomo_answer_suite;
/// Frozen official category-aware LoCoMo token-F1 evaluator.
#[cfg(feature = "benchmarks")]
pub mod locomo_eval;
/// Modular retrieval-only LoCoMo paper-v2 comparison suite.
#[cfg(feature = "benchmarks")]
pub mod locomo_suite;
#[cfg(feature = "benchmarks")]
mod locomo_suite_common;
/// Label-isolated native loader for the official LongMemEval release.
#[cfg(feature = "benchmarks")]
pub mod longmemeval;
/// Frozen LongMemEval answer and official judge prompt helpers.
#[cfg(feature = "benchmarks")]
pub mod longmemeval_eval;
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

pub use strategy::RecallItem;

pub use model::{
    AgentKind, ArchiveSpecVersion, AttachmentRef, CanonicalArchive, CanonicalEvent, ContentPart,
    ConversionDiagnostic, DiagnosticSeverity, DomainEventSequence, ErrorInfo, EventData,
    EventEnvelope, EventKind, EventSource, EventSpecVersion, MemoryMutation, MemoryQuery,
    MemoryRecord, MessageRole, QueryMode, RawRecord, SearchHit, SearchQuery, StrategyEventVector,
    ToolStatus, VectorHit, VectorQuery, canonical_archive_schema,
};
