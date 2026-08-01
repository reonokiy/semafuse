use crate::model::{CanonicalEvent, SearchHit};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One ranked result emitted by a memory strategy.
///
/// Event-backed and derived-memory results share this tagged contract so hooks
/// and MCP clients do not need strategy-specific response handling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecallItem {
    /// An immutable canonical event selected directly from raw history.
    Event {
        /// Retrieved canonical event, including native provenance.
        event: Box<CanonicalEvent>,
        /// Strategy-relative relevance score.
        score: f64,
    },
    /// A strategy-produced memory with explicit canonical-event provenance.
    Memory {
        /// Strategy namespace that produced the memory.
        namespace: String,
        /// Stable identity within the strategy namespace.
        id: String,
        /// Text intended for recall injection.
        content: String,
        /// Strategy-defined structured attributes.
        #[serde(default)]
        metadata: Map<String, Value>,
        /// Canonical events supporting this memory.
        #[serde(default)]
        source_event_ids: Vec<String>,
        /// Strategy-relative relevance score.
        score: f64,
    },
}

impl From<SearchHit> for RecallItem {
    fn from(hit: SearchHit) -> Self {
        Self::Event {
            event: Box::new(hit.event),
            score: hit.score,
        }
    }
}

impl RecallItem {
    /// Returns the strategy-relative score shared by both result variants.
    pub fn score(&self) -> f64 {
        match self {
            Self::Event { score, .. } | Self::Memory { score, .. } => *score,
        }
    }
}
