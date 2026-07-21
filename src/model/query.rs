use super::{AgentKind, CanonicalEvent};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Filters and bounds a search over canonical events.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Text interpreted by the selected retrieval strategy.
    pub text: String,
    /// Exact normalized repository filter, or all repositories when absent.
    pub repository: Option<String>,
    /// Exact source-agent filter, or all agents when absent.
    pub agent: Option<AgentKind>,
    /// Inclusive lower bound for `ordering_timestamp_ms`.
    pub since_ms: Option<i64>,
    /// Inclusive upper bound for `ordering_timestamp_ms`.
    pub until_ms: Option<i64>,
    /// Maximum number of results requested.
    pub limit: usize,
}

/// Algorithm used to retrieve canonical events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    /// Full-text or substring matching, depending on the store.
    Lexical,
    /// Newest events first without considering query text.
    Recent,
    /// Rank fusion of lexical and recent results.
    #[default]
    Hybrid,
}

impl std::str::FromStr for QueryMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lexical" | "fts" => Ok(Self::Lexical),
            "recent" => Ok(Self::Recent),
            "hybrid" => Ok(Self::Hybrid),
            _ => bail!("unknown query mode: {value}"),
        }
    }
}

impl QueryMode {
    /// Returns the stable snake-case mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Recent => "recent",
            Self::Hybrid => "hybrid",
        }
    }
}

/// A strategy-owned, derived memory with canonical-event provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    /// Ownership boundary; strategies may mutate only their own namespace.
    pub namespace: String,
    /// Stable identifier unique within the namespace.
    pub id: String,
    /// Searchable derived-memory text.
    pub content: String,
    /// Strategy-defined structured attributes.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// Canonical event IDs from which this memory was derived.
    #[serde(default)]
    pub source_event_ids: Vec<String>,
    /// Creation time in Unix milliseconds.
    pub created_at_ms: i64,
    /// Last update time in Unix milliseconds.
    pub updated_at_ms: i64,
}

/// Namespace-scoped search over derived memories.
#[derive(Debug, Clone, Default)]
pub struct MemoryQuery {
    /// Required strategy namespace.
    pub namespace: String,
    /// Full-text query; empty text requests the most recently updated records.
    pub text: String,
    /// Maximum number of records requested.
    pub limit: usize,
}

/// Atomic change requested by a memory strategy.
#[derive(Debug, Clone)]
pub enum MemoryMutation {
    /// Inserts or updates a record identified by namespace and ID.
    Upsert(MemoryRecord),
    /// Deletes one strategy-owned record.
    Delete {
        /// Namespace that owns the record.
        namespace: String,
        /// Identifier within the namespace.
        id: String,
    },
}

/// A canonical event paired with a strategy-dependent relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Retrieved canonical event, including its provenance fields.
    pub event: CanonicalEvent,
    /// Relative ranking score; meaningful within one result set.
    pub score: f64,
}
