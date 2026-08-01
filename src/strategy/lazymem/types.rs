//! Stable, model-agnostic values shared by the LazyMem query-time stages.

use crate::model::{AgentKind, EventSource, MessageRole};
use serde::{Deserialize, Serialize};

/// Reciprocal-rank-fusion constant used by the paper's retrieval pipeline.
pub(super) const PAPER_RRF_K: u32 = 60;
/// Number of messages retained after cross-encoder reranking in the benchmark.
pub(super) const PAPER_RETRIEVAL_TOP_N: usize = 50;
/// Number of neighboring messages restored on either side of an anchor.
pub(super) const PAPER_CONTEXT_RADIUS: usize = 2;
/// Maximum messages sent to one memory-processing model call.
pub(super) const PAPER_WINDOW_LENGTH: usize = 8;
/// Step between consecutive full model windows (one-message overlap).
pub(super) const PAPER_WINDOW_STRIDE: usize = 7;

/// A conversation scope whose session ordinals form one coherent timeline.
///
/// Repository is part of the key because native session identifiers are not
/// guaranteed to be globally unique, and neighbors must never cross a
/// repository boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ScopeKey {
    pub agent: AgentKind,
    pub session_id: String,
    pub repository: Option<String>,
}

impl ScopeKey {
    pub(super) fn from_source(source: &EventSource) -> Self {
        Self {
            agent: source.agent,
            session_id: source.session_id.clone(),
            repository: source.repository.clone(),
        }
    }

    /// A deterministic representation used only as a final ordering tie-break.
    pub(super) fn stable_key(&self) -> (&str, &str, Option<&str>) {
        (
            self.agent.as_str(),
            self.session_id.as_str(),
            self.repository.as_deref(),
        )
    }
}

/// One verbatim message stored without write-time summarization.
///
/// `global_ordinal` is stable across the whole event store and determines the
/// final memory order. `session_ordinal` is stable inside [`ScopeKey`] and is
/// the coordinate used for neighborhood expansion. Timestamps are retained as
/// evidence metadata, but never used as the sole ordering key because native
/// timestamps may be missing, duplicated, or skewed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredMessage {
    pub message_id: String,
    pub global_ordinal: u64,
    pub session_ordinal: u64,
    pub role: MessageRole,
    pub timestamp_ms: i64,
    pub source: EventSource,
    pub content: String,
}

impl StoredMessage {
    pub(super) fn scope(&self) -> ScopeKey {
        ScopeKey::from_source(&self.source)
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use super::StoredMessage;
    use crate::model::{AgentKind, EventSource, MessageRole};

    pub(crate) fn message(id: &str, global_ordinal: u64, session_ordinal: u64) -> StoredMessage {
        message_in_scope(
            id,
            global_ordinal,
            session_ordinal,
            "session-a",
            Some("/repo/a"),
        )
    }

    pub(crate) fn message_in_scope(
        id: &str,
        global_ordinal: u64,
        session_ordinal: u64,
        session_id: &str,
        repository: Option<&str>,
    ) -> StoredMessage {
        StoredMessage {
            message_id: id.to_owned(),
            global_ordinal,
            session_ordinal,
            role: MessageRole::User,
            timestamp_ms: 1_700_000_000_000 + global_ordinal as i64,
            source: EventSource::new(
                AgentKind::Codex,
                session_id,
                None,
                Some(id.to_owned()),
                "message",
                None,
                repository.map(str::to_owned),
            ),
            content: format!("content for {id}"),
        }
    }
}
