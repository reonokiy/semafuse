//! Pluggable retrieval over immutable canonical events.

use crate::{
    model::{SearchHit, SearchQuery},
    storage::EventStore,
};
use anyhow::{Result, bail};
use async_trait::async_trait;
use std::collections::HashMap;

/// A named algorithm for querying an [`EventStore`].
#[async_trait]
pub trait RetrievalStrategy: Send + Sync {
    /// Returns the configuration name for this algorithm.
    fn name(&self) -> &'static str;
    /// Executes the query and returns at most its requested result count.
    async fn retrieve(&self, store: &dyn EventStore, query: &SearchQuery)
    -> Result<Vec<SearchHit>>;
}

/// Delegates to the store's lexical/full-text search implementation.
pub struct Lexical;
#[async_trait]
impl RetrievalStrategy for Lexical {
    fn name(&self) -> &'static str {
        "lexical"
    }
    async fn retrieve(
        &self,
        store: &dyn EventStore,
        query: &SearchQuery,
    ) -> Result<Vec<SearchHit>> {
        store.lexical_search(query).await
    }
}

/// Returns store-filtered events newest first, ignoring query text.
pub struct Recent;
#[async_trait]
impl RetrievalStrategy for Recent {
    fn name(&self) -> &'static str {
        "recent"
    }
    async fn retrieve(
        &self,
        store: &dyn EventStore,
        query: &SearchQuery,
    ) -> Result<Vec<SearchHit>> {
        store.recent(query).await
    }
}

/// Combines lexical relevance and recency using rank-based score boosts.
///
/// Candidates are deduplicated by event fingerprint and the candidate pool is
/// expanded to twice the requested limit before final score ordering.
pub struct Hybrid;
#[async_trait]
impl RetrievalStrategy for Hybrid {
    fn name(&self) -> &'static str {
        "hybrid"
    }
    async fn retrieve(
        &self,
        store: &dyn EventStore,
        query: &SearchQuery,
    ) -> Result<Vec<SearchHit>> {
        let mut expanded = query.clone();
        expanded.limit = query.limit.saturating_mul(2).max(query.limit);
        let lexical = store.lexical_search(&expanded).await?;
        let recent = store.recent(&expanded).await?;
        let mut merged: HashMap<String, SearchHit> = HashMap::new();
        for (rank, mut hit) in lexical.into_iter().enumerate() {
            hit.score += 1.0 / (rank as f64 + 1.0);
            merged.insert(hit.event.fingerprint.clone(), hit);
        }
        for (rank, mut hit) in recent.into_iter().enumerate() {
            let recency = 0.25 / (rank as f64 + 1.0);
            merged
                .entry(hit.event.fingerprint.clone())
                .and_modify(|h| h.score += recency)
                .or_insert_with(|| {
                    hit.score = recency;
                    hit
                });
        }
        let mut hits: Vec<_> = merged.into_values().collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(query.limit);
        Ok(hits)
    }
}

/// Constructs a retrieval strategy by name.
///
/// `fts` is accepted as an alias for `lexical`; unknown names return an error.
pub fn strategy(name: &str) -> Result<Box<dyn RetrievalStrategy>> {
    match name {
        "lexical" | "fts" => Ok(Box::new(Lexical)),
        "recent" => Ok(Box::new(Recent)),
        "hybrid" => Ok(Box::new(Hybrid)),
        _ => bail!("unknown retrieval strategy: {name}"),
    }
}
