//! Lightweight in-process event storage, primarily useful for tests and embedding.

use super::{EventStore, IngestStats};
use crate::model::{CanonicalEvent, QueryMode, SearchHit, SearchQuery};
use anyhow::Result;
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

/// Thread-safe, non-durable event store deduplicated by fingerprint.
///
/// Lexical matching is case-insensitive substring matching with constant scores;
/// ties and recent results are ordered by descending event timestamp. It does
/// not maintain provenance, domain-event sequences, or derived memories.
#[derive(Default)]
pub struct MemoryStore {
    events: RwLock<HashMap<String, CanonicalEvent>>,
}

fn matches(event: &CanonicalEvent, query: &SearchQuery) -> bool {
    query.agent.is_none_or(|a| a == event.source.agent)
        && query
            .repository
            .as_ref()
            .is_none_or(|r| event.source.repository.as_ref() == Some(r))
}

#[async_trait]
impl EventStore for MemoryStore {
    async fn ingest(&self, events: &[CanonicalEvent]) -> Result<IngestStats> {
        let mut stats = IngestStats::default();
        let mut stored = self.events.write().expect("memory store poisoned");
        for event in events {
            if stored.contains_key(&event.fingerprint) {
                stats.deduplicated += 1;
            } else {
                stored.insert(event.fingerprint.clone(), event.clone());
                stats.inserted += 1;
                stats.inserted_events.push(event.clone());
            }
        }
        Ok(stats)
    }

    async fn lexical_search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        let needle = query.text.to_lowercase();
        let stored = self.events.read().expect("memory store poisoned");
        let mut hits: Vec<_> = stored
            .values()
            .filter(|e| matches(e, query) && e.search_text().to_lowercase().contains(&needle))
            .cloned()
            .map(|event| SearchHit { event, score: 1.0 })
            .collect();
        hits.sort_by_key(|h| std::cmp::Reverse(h.event.ordering_timestamp_ms()));
        hits.truncate(query.limit);
        Ok(hits)
    }

    async fn recent(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        let stored = self.events.read().expect("memory store poisoned");
        let mut hits: Vec<_> = stored
            .values()
            .filter(|e| matches(e, query))
            .cloned()
            .map(|event| SearchHit { event, score: 0.1 })
            .collect();
        hits.sort_by_key(|h| std::cmp::Reverse(h.event.ordering_timestamp_ms()));
        hits.truncate(query.limit);
        Ok(hits)
    }

    async fn session_events(&self, session_id: &str) -> Result<Vec<CanonicalEvent>> {
        let stored = self.events.read().expect("memory store poisoned");
        let mut events: Vec<_> = stored
            .values()
            .filter(|e| e.source.session_id == session_id)
            .cloned()
            .collect();
        events.sort_by_key(CanonicalEvent::ordering_timestamp_ms);
        Ok(events)
    }
    async fn count(&self) -> Result<usize> {
        Ok(self.events.read().expect("memory store poisoned").len())
    }

    async fn query(&self, mode: QueryMode, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        match mode {
            QueryMode::Lexical => self.lexical_search(query).await,
            QueryMode::Recent => self.recent(query).await,
            QueryMode::Hybrid => {
                crate::retrieval::strategy("hybrid")?
                    .retrieve(self, query)
                    .await
            }
        }
    }
}
