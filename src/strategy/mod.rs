//! Checkpointed processing and recall over strategy-derived memory.
//!
//! Each strategy consumes durable event batches in order. Its mutations and
//! checkpoint commit atomically, failures leave the batch pending, and rebuilds
//! construct replacement state off to the side before an atomic swap.

mod history;

use crate::{
    config::Config,
    model::{CanonicalEvent, DomainEventSequence, MemoryMutation, SearchHit, SearchQuery},
    storage::{EventStore, MemoryRepository, SqliteStore},
};
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use history::HistoryStrategy;

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
}

/// Result returned by a memory strategy's recall path.
pub struct RecallResult {
    /// Ranked canonical events selected as relevant context.
    pub hits: Vec<SearchHit>,
}

/// Combined read interface exposed to strategy implementations.
///
/// Mutation and checkpoint methods are intentionally owned by the dispatcher.
pub trait StrategyContext: EventStore + MemoryRepository {}
impl<T: EventStore + MemoryRepository> StrategyContext for T {}

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
    /// Selects canonical events to inject as relevant context.
    async fn recall(
        &self,
        context: &dyn StrategyContext,
        request: &RecallRequest,
    ) -> Result<RecallResult>;
}

/// Constructs a derived-memory strategy by configuration name.
pub fn strategy(name: &str) -> Result<Box<dyn MemoryStrategy>> {
    match name {
        "history" => Ok(Box::new(HistoryStrategy)),
        _ => bail!("unknown memory strategy: {name}"),
    }
}

/// Validates unique enabled names and an enabled recall route.
pub fn validate(config: &Config) -> Result<()> {
    let mut names = std::collections::HashSet::new();
    for name in &config.strategies.enabled {
        ensure!(names.insert(name), "duplicate enabled strategy: {name}");
        strategy(name)?;
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
        let handler = strategy(name)?;
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
) -> Result<Vec<SearchHit>> {
    validate(config)?;
    Ok(strategy(&config.strategies.recall)?
        .recall(store, &RecallRequest { query })
        .await?
        .hits)
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
    strategy(name)?;
    let handler = strategy(name)?;
    let events = store.all_events().await?;
    let count = events.len();
    let staging = SqliteStore::open_memory().await?;
    for event in events {
        staging.ingest(std::slice::from_ref(&event)).await?;
        let input = StrategyEvent {
            sequence: DomainEventSequence::ORIGIN,
            events: vec![event],
        };
        let mutations = handler.on_event(&staging, &input).await?;
        staging
            .apply_strategy_event(name, DomainEventSequence::ORIGIN, &mutations)
            .await?;
    }
    let records = staging.all_memories(name).await?;
    let latest = store.latest_domain_event_seq().await?;
    store.replace_strategy_state(name, &records, latest).await?;
    Ok(count)
}
