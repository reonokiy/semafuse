//! Checkpointed processing and recall over strategy-derived memory.
//!
//! Each strategy consumes durable event batches in order. Its mutations and
//! checkpoint commit atomically, failures leave the batch pending, and rebuilds
//! construct replacement state off to the side before an atomic swap.

mod history;
mod lazymem;
mod recall;
mod registry;

pub use recall::RecallItem;

use crate::{
    config::{Config, StrategyConfig},
    model::{CanonicalEvent, DomainEventSequence, MemoryMutation, SearchQuery},
    storage::{
        EventStore, MemoryRepository, SqliteStore, StrategyStateRepository, VectorRepository,
    },
};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde_json::{Map, Value};

/// A committed batch of newly ingested canonical events delivered to a memory strategy.
///
/// One value corresponds to one `events_ingested` row in the domain-event log.
/// [`Self::sequence`] is that row's durable position and becomes the strategy
/// checkpoint only after all mutations returned by [`MemoryStrategy::on_event`]
/// are applied successfully. [`Self::events`] contains only canonical events
/// newly inserted by the ingest transaction; deduplicated inputs are excluded.
pub struct StrategyEvent {
    /// Durable position of the ingested batch.
    pub sequence: DomainEventSequence,
    /// Newly inserted canonical events in the batch.
    pub events: Vec<CanonicalEvent>,
}

/// Canonical search request routed to a memory strategy.
pub struct RecallRequest {
    /// Event search text, filters, and result limit.
    pub query: SearchQuery,
    /// Explicit wall-clock reference used by query-conditioned processors.
    ///
    /// Passing this through the request keeps benchmark replays deterministic
    /// and prevents strategies from consulting the current clock internally.
    pub query_time_ms: i64,
}

/// Result returned by a memory strategy's recall path.
pub struct RecallResult {
    /// Ranked event-backed or strategy-derived memories selected as context.
    pub items: Vec<RecallItem>,
    /// Strategy-owned, non-context trace metadata for the completed recall.
    ///
    /// This remains available when no context item is emitted, which lets
    /// callers audit retrieval and timing without treating an empty memory as
    /// a zero-cost operation.
    pub metadata: Map<String, Value>,
}

/// Combined read interface exposed to strategy implementations.
///
/// Mutation and checkpoint methods are intentionally owned by the dispatcher.
pub trait StrategyContext:
    EventStore + MemoryRepository + VectorRepository + StrategyStateRepository
{
}
impl<T: EventStore + MemoryRepository + VectorRepository + StrategyStateRepository> StrategyContext
    for T
{
}

/// Derives namespaced memories from events and serves recall requests.
#[async_trait]
pub trait MemoryStrategy: Send + Sync {
    /// Returns the unique configuration and namespace name.
    fn name(&self) -> &'static str;
    /// Converts one committed event batch into atomic memory mutations.
    async fn on_event(
        &self,
        context: &dyn StrategyContext,
        event: &StrategyEvent,
    ) -> Result<Vec<MemoryMutation>>;
    /// Selects event-backed or derived memories to inject as relevant context.
    async fn recall(
        &self,
        context: &dyn StrategyContext,
        request: &RecallRequest,
    ) -> Result<RecallResult>;
}

fn configured_strategy(name: &str, config: &StrategyConfig) -> Result<Box<dyn MemoryStrategy>> {
    registry::build(name, config.settings.get(name))
}

/// Validates unique enabled names and an enabled recall route.
pub fn validate(config: &Config) -> Result<()> {
    registry::validate_settings(&config.strategies.settings)?;
    let mut names = std::collections::HashSet::new();
    for name in &config.strategies.enabled {
        ensure!(names.insert(name), "duplicate enabled strategy: {name}");
        configured_strategy(name, &config.strategies)?;
    }
    ensure!(
        names.contains(&config.strategies.recall),
        "recall strategy {} is not enabled",
        config.strategies.recall
    );
    Ok(())
}

/// Delivers pending domain events in order to enabled strategies.
///
/// If `only` is supplied it must name an enabled strategy. Each successful
/// batch atomically commits mutations and its checkpoint. A handler failure is
/// recorded, stops that strategy for this dispatch, and leaves its checkpoint
/// unchanged for retry; storage failures are returned immediately.
pub async fn dispatch_pending(
    store: &SqliteStore,
    config: &Config,
    only: Option<&str>,
) -> Result<usize> {
    validate(config)?;
    if let Some(selected) = only {
        ensure!(
            config
                .strategies
                .enabled
                .iter()
                .any(|name| name == selected),
            "strategy {selected} is not enabled"
        );
    }
    let mut processed = 0;
    for name in &config.strategies.enabled {
        if only.is_some_and(|selected| selected != name) {
            continue;
        }
        let handler = configured_strategy(name, &config.strategies)?;
        let (checkpoint, _, _) = store.strategy_checkpoint(name).await?;
        for (sequence, ids) in store.pending_domain_events(checkpoint).await? {
            let event = StrategyEvent {
                sequence,
                events: store.events_by_ids(&ids).await?,
            };
            match handler.on_event(store, &event).await {
                Ok(mutations) => {
                    store
                        .apply_strategy_event(name, sequence, &mutations)
                        .await?;
                    processed += 1;
                }
                Err(error) => {
                    store
                        .record_strategy_failure(name, &format!("{error:#}"))
                        .await?;
                    break;
                }
            }
        }
    }
    Ok(processed)
}

/// Routes a query exclusively through the configured recall strategy.
pub async fn recall(
    store: &SqliteStore,
    config: &Config,
    query: SearchQuery,
) -> Result<Vec<RecallItem>> {
    recall_at(store, config, query, crate::model::now_ms()).await
}

/// Routes a query with an explicit reference time for reproducible replay.
pub async fn recall_at(
    store: &SqliteStore,
    config: &Config,
    query: SearchQuery,
    query_time_ms: i64,
) -> Result<Vec<RecallItem>> {
    Ok(recall_result_at(store, config, query, query_time_ms)
        .await?
        .items)
}

/// Routes a query with an explicit reference time and retains strategy trace
/// metadata even when the strategy selects no context items.
pub async fn recall_result_at(
    store: &SqliteStore,
    config: &Config,
    query: SearchQuery,
    query_time_ms: i64,
) -> Result<RecallResult> {
    validate(config)?;
    configured_strategy(&config.strategies.recall, &config.strategies)?
        .recall(
            store,
            &RecallRequest {
                query,
                query_time_ms,
            },
        )
        .await
}

/// Runs LazyMem retrieval, reranking, and evidence-window expansion without
/// invoking its prompt processor or constructing a compressed memory.
///
/// This path exists for retrieval benchmarks that must isolate ranking and
/// coverage quality from generation quality. It still requires the configured
/// LazyMem embedding and reranker backends and the same fully indexed corpus as
/// production recall.
pub async fn recall_lazymem_retrieval_result_at(
    store: &SqliteStore,
    config: &Config,
    query: SearchQuery,
    query_time_ms: i64,
) -> Result<RecallResult> {
    validate(config)?;
    ensure!(
        config.strategies.recall == "lazymem",
        "retrieval-only recall requires strategies.recall = \"lazymem\""
    );
    lazymem::recall_retrieval_only(
        config.strategies.settings.get("lazymem"),
        store,
        &RecallRequest {
            query,
            query_time_ms,
        },
    )
    .await
}

/// Rebuilds one enabled strategy from all canonical events.
///
/// State is derived in a temporary database first. Only after every event is
/// processed successfully are the target namespace and checkpoint replaced in
/// one transaction, so the existing strategy state remains visible on failure.
pub async fn rebuild(store: &SqliteStore, config: &Config, name: &str) -> Result<usize> {
    ensure!(
        config
            .strategies
            .enabled
            .iter()
            .any(|enabled| enabled == name),
        "strategy {name} is not enabled"
    );
    let handler = configured_strategy(name, &config.strategies)?;
    let (events, latest) = store.events_and_latest_seq().await?;
    let count = events.len();
    let staging = SqliteStore::open_memory().await?;
    for event in events {
        let stats = staging.ingest(std::slice::from_ref(&event)).await?;
        let sequence = stats
            .domain_event_seq
            .context("staging ingest did not emit a domain event")?;
        let input = StrategyEvent {
            sequence,
            events: vec![event],
        };
        let mutations = handler.on_event(&staging, &input).await?;
        staging
            .apply_strategy_event(name, sequence, &mutations)
            .await?;
    }
    let records = staging.all_memories(name).await?;
    let vectors = staging.all_strategy_vectors(name).await?;
    store
        .replace_strategy_state(name, &records, &vectors, latest)
        .await?;
    Ok(count)
}
