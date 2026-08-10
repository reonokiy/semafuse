//! Isolated benchmark execution through Semafuse's production strategy paths.
//!
//! The types in this module deliberately mirror the normalized JSONL records
//! emitted by `benchmarks/lazymem`. Benchmark histories are never mixed with a
//! user's durable store: one-shot calls get a fresh in-memory SQLite store,
//! while [`StrategyBenchmarkSession`] owns one isolated store per explicitly
//! identity-checked shared history. Both paths use ordinary ingestion, strategy
//! dispatch, and timed recall APIs.

use crate::{
    AgentKind, CanonicalEvent, EventSource, MessageRole, RawRecord, SearchQuery,
    config::Config,
    storage::{EventStore, IngestBatch, RawContext, SqliteStore},
    strategy::{self, RecallItem},
};
use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};

const BENCHMARK_TRACE_VERSION: &str = "semafuse-memory-benchmark/v2";
const BENCHMARK_REPOSITORY_ROOT: &str = "/__semafuse_benchmark__";

/// One normalized LazyMem benchmark question accepted as a JSONL record.
///
/// The shape accepts `Question.to_dict()` from the Python harness. The
/// question's answer and gold annotations are intentionally not retained by
/// this runner; only the question text is used for query-time recall.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkQuestion {
    /// Evaluator-side question identifier.
    pub id: String,
    /// Dataset name used for isolation and reporting.
    pub dataset: String,
    /// Dataset-defined category; never passed to the strategy.
    #[serde(default)]
    pub question_type: String,
    /// Query presented to retrieval and memory construction.
    #[serde(alias = "question")]
    pub query: String,
    /// Historical query timestamp retained for deterministic replay.
    #[serde(default)]
    pub query_time: String,
    /// Chronological history visible to the system under test.
    pub messages: Vec<BenchmarkMessage>,
}

/// One chronological message from a normalized benchmark history.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkMessage {
    /// Evaluator-side message identifier, remapped before storage.
    pub id: String,
    /// Optional asserted chronological position.
    #[serde(default)]
    pub index: Option<usize>,
    /// Conversation role.
    pub role: String,
    /// Message text.
    pub content: String,
    /// Native dataset timestamp.
    #[serde(default)]
    pub timestamp: String,
    /// Evaluator-side session identifier, remapped before storage.
    #[serde(default)]
    pub session_id: String,
}

/// One JSONL response consumed by the LazyMem benchmark harness.
#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    /// Question identifier copied to the evaluator-side output.
    pub question_id: String,
    /// Production strategy that produced the result.
    pub strategy: String,
    /// Context produced for a downstream answer model.
    pub context: String,
    /// Final ranked retrieval anchors.
    pub retrieved_message_ids: Vec<String>,
    /// Unordered evidence coverage after neighbor expansion.
    pub covered_message_ids: Vec<String>,
    /// Messages retained by memory construction.
    pub retained_message_ids: Vec<String>,
    /// Ranked outputs of available retrieval pipeline stages.
    pub stage_rankings: BTreeMap<String, Vec<String>>,
    /// Query retrieval latency in seconds.
    pub retrieval_latency_s: f64,
    /// Window construction and processing latency in seconds.
    pub processing_latency_s: f64,
    /// Auditable strategy and runner trace.
    pub metadata: Map<String, Value>,
}

struct PreparedQuestion {
    repository: String,
    history_max_ms: Option<i64>,
    query_time_ms: i64,
    query_time_source: &'static str,
    events: Vec<CanonicalEvent>,
    raw_records: Vec<RawRecord>,
    raw_contexts: Vec<RawContext>,
    message_ids_by_event: HashMap<String, String>,
    skipped_empty_message_ids: Vec<String>,
    synthetic_message_timestamp_count: usize,
    monotonic_timestamp_adjustments: usize,
}

/// One persistent retrieval session's shared construction measurements.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkConstructionMetrics {
    /// Complete isolated-store open, ingest, and strategy-index wall time.
    pub total_s: f64,
    /// Event ingestion wall time.
    pub ingest_s: f64,
    /// Strategy indexing wall time.
    pub index_s: f64,
    /// Raw records newly inserted into the isolated store.
    pub raw_inserted: usize,
    /// Canonical events newly inserted into the isolated store.
    pub events_inserted: usize,
    /// Strategy event batches dispatched during construction.
    pub event_batches_indexed: usize,
}

/// Production strategy execution over one isolated, persistent benchmark
/// history.
///
/// Construction ingests and indexes exactly once. Every query is checked
/// against the constructor's SHA-256 history identity before recall, so a
/// caller cannot accidentally reuse memory across datasets, users, roles,
/// messages, sessions, or timestamps. Question IDs, query text, query time,
/// and evaluator-only category may vary between queries. Session queries see
/// the complete supplied history, which matches benchmarks such as LoCoMo
/// where questions are asked after the full conversation.
pub struct StrategyBenchmarkSession {
    config: Config,
    store: SqliteStore,
    prepared: PreparedQuestion,
    history_identity_sha256: String,
    construction: BenchmarkConstructionMetrics,
    completed_queries: usize,
}

/// Backwards-compatible retrieval-only view over a persistent benchmark
/// session.
///
/// New benchmark code that also needs query-conditioned memory construction
/// should use [`StrategyBenchmarkSession`] directly.
pub struct StrategyRetrievalSession {
    inner: StrategyBenchmarkSession,
}

#[derive(Clone, Copy)]
struct ParsedTimestamp {
    start_ms: i64,
    cutoff_ms: i64,
    date_only: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BenchmarkExecution {
    Full,
    RetrievalOnly,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BenchmarkHistoryVisibility {
    AtQueryTime,
    AllProvidedMessages,
}

impl BenchmarkHistoryVisibility {
    fn as_str(self) -> &'static str {
        match self {
            Self::AtQueryTime => "at_query_time",
            Self::AllProvidedMessages => "all_provided_messages",
        }
    }

    fn strategy_reference_time_ms(self, query_time_ms: i64, prepared: &PreparedQuestion) -> i64 {
        match self {
            Self::AtQueryTime => query_time_ms,
            Self::AllProvidedMessages => prepared
                .events
                .iter()
                .map(CanonicalEvent::ordering_timestamp_ms)
                .max()
                .map_or(query_time_ms, |history_max| {
                    query_time_ms.max(history_max.saturating_add(1))
                }),
        }
    }
}

impl BenchmarkExecution {
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::RetrievalOnly => "retrieval_only",
        }
    }
}

/// Runs one benchmark question through real Semafuse ingestion, LazyMem
/// indexing, and deterministic query-time recall.
///
/// The supplied configuration must select the `lazymem` strategy for recall.
/// Its evidence fields use benchmark message IDs rather than internal
/// canonical event IDs, so the harness can use its established metrics. The
/// nested strategy metadata remains available for implementation auditing.
pub async fn run_lazymem_question(
    config: &Config,
    question: BenchmarkQuestion,
    limit: usize,
) -> Result<BenchmarkResponse> {
    ensure!(
        config.strategies.recall == "lazymem",
        "the LazyMem benchmark runner requires strategies.recall = \"lazymem\""
    );
    ensure!(
        config
            .strategies
            .enabled
            .iter()
            .any(|strategy| strategy == "lazymem"),
        "the LazyMem benchmark runner requires lazymem in strategies.enabled"
    );
    run_strategy_question(config, question, limit).await
}

/// Runs one normalized question through the configured production strategy.
///
/// Every invocation uses a fresh SQLite database. The benchmark labels are not
/// represented by [`BenchmarkQuestion`] and therefore cannot enter storage or
/// any model request. External message and session identifiers are replaced by
/// opaque per-question identifiers before ingestion and are only mapped back in
/// the returned evaluator-side trace.
pub async fn run_strategy_question(
    config: &Config,
    question: BenchmarkQuestion,
    limit: usize,
) -> Result<BenchmarkResponse> {
    run_strategy_question_with_execution(config, question, limit, BenchmarkExecution::Full).await
}

/// Runs one question through LazyMem's production ingestion, indexing, dense
/// and sparse retrieval, RRF, reranker, and evidence-window expansion stages.
///
/// The prompt processor is deliberately skipped so retrieval metrics do not
/// require or measure memory generation. Consequently `retained_message_ids`
/// is empty and `context` contains explicitly marked, uncompressed expanded
/// evidence rather than a constructed memory. All supplied history is visible,
/// matching retrieval benchmarks whose fixture timestamps may extend beyond a
/// separate question-date annotation.
pub async fn run_strategy_retrieval_question(
    config: &Config,
    question: BenchmarkQuestion,
    limit: usize,
) -> Result<BenchmarkResponse> {
    let mut session = StrategyRetrievalSession::new(config, &question).await?;
    session.query(question, limit).await
}

impl StrategyBenchmarkSession {
    /// Constructs one isolated production strategy index for a shared history.
    ///
    /// The store is populated and strategy indexing is dispatched during this
    /// call only. Both [`Self::query_retrieval`] and [`Self::query_full`] reuse
    /// that construction for every identity-compatible question.
    pub async fn new(config: &Config, exemplar: &BenchmarkQuestion) -> Result<Self> {
        validate_strategy_configuration(config, BenchmarkExecution::Full)?;
        validate_benchmark_question(exemplar)?;
        let history_identity_sha256 = benchmark_history_identity(exemplar)?;
        let prepared = prepare_shared_history(exemplar, &history_identity_sha256)?;
        let construction_started = Instant::now();
        let store = SqliteStore::open_memory()
            .await
            .context("open isolated persistent benchmark store")?;
        let construction = construct_strategy_store(
            &store,
            config,
            &prepared,
            &exemplar.id,
            construction_started,
        )
        .await?;
        Ok(Self {
            config: config.clone(),
            store,
            prepared,
            history_identity_sha256,
            construction,
            completed_queries: 0,
        })
    }

    /// Returns the stable identity of the exact history owned by this session.
    pub fn history_identity_sha256(&self) -> &str {
        &self.history_identity_sha256
    }

    /// Returns once-per-history construction measurements.
    pub fn construction_metrics(&self) -> &BenchmarkConstructionMetrics {
        &self.construction
    }

    /// Runs retrieval and evidence-window expansion without invoking the
    /// query-conditioned memory processor.
    ///
    /// The complete history must match the constructor's identity. Validation
    /// happens before recall and a mismatch never mutates the session.
    pub async fn query_retrieval(
        &mut self,
        question: BenchmarkQuestion,
        limit: usize,
    ) -> Result<BenchmarkResponse> {
        validate_strategy_configuration(&self.config, BenchmarkExecution::RetrievalOnly)?;
        self.query_with_execution(question, limit, BenchmarkExecution::RetrievalOnly)
            .await
    }

    /// Runs the configured production strategy including query-conditioned
    /// memory construction over the already-constructed index.
    ///
    /// Full session queries deliberately make every supplied message visible,
    /// even when the dataset's query-time string parses earlier than the last
    /// message. This is the LoCoMo lifecycle: the question is asked after the
    /// complete conversation, and `query_time` remains evaluator metadata
    /// rather than a corpus cutoff. The returned benchmark trace records
    /// `history_visibility = "all_provided_messages"` and both timestamps.
    pub async fn query_full(
        &mut self,
        question: BenchmarkQuestion,
        limit: usize,
    ) -> Result<BenchmarkResponse> {
        validate_strategy_configuration(&self.config, BenchmarkExecution::Full)?;
        self.query_with_execution(question, limit, BenchmarkExecution::Full)
            .await
    }

    async fn query_with_execution(
        &mut self,
        question: BenchmarkQuestion,
        limit: usize,
        execution: BenchmarkExecution,
    ) -> Result<BenchmarkResponse> {
        ensure!(limit > 0, "benchmark retrieval limit must be positive");
        validate_benchmark_question(&question)?;
        ensure_session_history(&self.history_identity_sha256, &question)?;
        let (query_time_ms, query_time_source) =
            resolve_query_time(&question.query_time, self.prepared.history_max_ms);
        let response = recall_prepared_question(
            &self.store,
            &self.config,
            &self.prepared,
            &question,
            limit,
            execution,
            BenchmarkHistoryVisibility::AllProvidedMessages,
            query_time_ms,
            query_time_source,
            &self.construction,
            "persistent_sqlite_memory_per_history",
            Some(&self.history_identity_sha256),
            Some(self.completed_queries),
        )
        .await?;
        self.completed_queries += 1;
        Ok(response)
    }
}

impl StrategyRetrievalSession {
    /// Constructs a retrieval-only session over one shared LazyMem history.
    pub async fn new(config: &Config, exemplar: &BenchmarkQuestion) -> Result<Self> {
        validate_strategy_configuration(config, BenchmarkExecution::RetrievalOnly)?;
        Ok(Self {
            inner: StrategyBenchmarkSession::new(config, exemplar).await?,
        })
    }

    /// Returns the stable identity of the exact history owned by this session.
    pub fn history_identity_sha256(&self) -> &str {
        self.inner.history_identity_sha256()
    }

    /// Returns once-per-history construction measurements.
    pub fn construction_metrics(&self) -> &BenchmarkConstructionMetrics {
        self.inner.construction_metrics()
    }

    /// Queries the already-constructed LazyMem index without invoking its
    /// memory processor.
    pub async fn query(
        &mut self,
        question: BenchmarkQuestion,
        limit: usize,
    ) -> Result<BenchmarkResponse> {
        self.inner.query_retrieval(question, limit).await
    }
}

fn ensure_session_history(
    expected_history_identity_sha256: &str,
    question: &BenchmarkQuestion,
) -> Result<()> {
    let query_history_identity = benchmark_history_identity(question)?;
    ensure!(
        query_history_identity == expected_history_identity_sha256,
        "persistent benchmark retrieval session history mismatch"
    );
    Ok(())
}

async fn run_strategy_question_with_execution(
    config: &Config,
    question: BenchmarkQuestion,
    limit: usize,
    execution: BenchmarkExecution,
) -> Result<BenchmarkResponse> {
    ensure!(limit > 0, "benchmark retrieval limit must be positive");
    validate_strategy_configuration(config, execution)?;
    validate_benchmark_question(&question)?;
    let prepared = prepare_question(&question)?;
    let construction_started = Instant::now();
    let store = SqliteStore::open_memory()
        .await
        .context("open isolated in-memory benchmark store")?;
    let construction = construct_strategy_store(
        &store,
        config,
        &prepared,
        &question.id,
        construction_started,
    )
    .await?;
    recall_prepared_question(
        &store,
        config,
        &prepared,
        &question,
        limit,
        execution,
        BenchmarkHistoryVisibility::AtQueryTime,
        prepared.query_time_ms,
        prepared.query_time_source,
        &construction,
        "fresh_sqlite_memory_per_question",
        None,
        None,
    )
    .await
}

fn validate_strategy_configuration(config: &Config, execution: BenchmarkExecution) -> Result<()> {
    let strategy_name = config.strategies.recall.as_str();
    ensure!(
        config
            .strategies
            .enabled
            .iter()
            .any(|strategy| strategy == strategy_name),
        "benchmark recall strategy {strategy_name} is not enabled"
    );
    ensure!(
        execution != BenchmarkExecution::RetrievalOnly || strategy_name == "lazymem",
        "retrieval-only benchmark execution requires strategies.recall = \"lazymem\""
    );
    Ok(())
}

fn validate_benchmark_question(question: &BenchmarkQuestion) -> Result<()> {
    ensure!(
        !question.id.trim().is_empty(),
        "benchmark question id must be non-empty"
    );
    ensure!(
        !question.dataset.trim().is_empty(),
        "benchmark question {} has an empty dataset",
        question.id
    );
    ensure!(
        !question.query.trim().is_empty(),
        "benchmark question {} has an empty query",
        question.id
    );
    let mut known_message_ids = HashSet::new();
    for (position, message) in question.messages.iter().enumerate() {
        ensure!(
            !message.id.trim().is_empty(),
            "benchmark question {} has a message with an empty id",
            question.id
        );
        ensure!(
            message.index.is_none_or(|index| index == position),
            "benchmark question {} has non-chronological message index {} at position {}",
            question.id,
            message.index.unwrap_or(position),
            position
        );
        ensure!(
            known_message_ids.insert(message.id.as_str()),
            "benchmark question {} has duplicate message id {}",
            question.id,
            message.id
        );
    }
    Ok(())
}

async fn construct_strategy_store(
    store: &SqliteStore,
    config: &Config,
    prepared: &PreparedQuestion,
    question_id: &str,
    construction_started: Instant,
) -> Result<BenchmarkConstructionMetrics> {
    let ingest_started = Instant::now();
    let ingest = store
        .ingest_batch(IngestBatch {
            raw_records: &prepared.raw_records,
            raw_contexts: &prepared.raw_contexts,
            events: &prepared.events,
            diagnostics: &[],
        })
        .await
        .with_context(|| format!("ingest benchmark history for question {question_id}"))?;
    let ingest_elapsed = ingest_started.elapsed();

    let strategy_name = config.strategies.recall.as_str();
    let indexing_started = Instant::now();
    let indexed_event_batches = strategy::dispatch_pending(store, config, Some(strategy_name))
        .await
        .with_context(|| {
            format!("index benchmark history for question {question_id} with {strategy_name}")
        })?;
    let (_, strategy_error, strategy_attempts) = store
        .strategy_checkpoint(strategy_name)
        .await
        .with_context(|| format!("read {strategy_name} benchmark indexing checkpoint"))?;
    if let Some(error) = strategy_error {
        bail!(
            "benchmark indexing failed for {strategy_name} after {strategy_attempts} attempt(s): {error}"
        );
    }
    let indexing_elapsed = indexing_started.elapsed();
    Ok(BenchmarkConstructionMetrics {
        total_s: construction_started.elapsed().as_secs_f64(),
        ingest_s: ingest_elapsed.as_secs_f64(),
        index_s: indexing_elapsed.as_secs_f64(),
        raw_inserted: ingest.raw_inserted,
        events_inserted: ingest.inserted,
        event_batches_indexed: indexed_event_batches,
    })
}

#[allow(clippy::too_many_arguments)]
async fn recall_prepared_question(
    store: &SqliteStore,
    config: &Config,
    prepared: &PreparedQuestion,
    question: &BenchmarkQuestion,
    limit: usize,
    execution: BenchmarkExecution,
    history_visibility: BenchmarkHistoryVisibility,
    query_time_ms: i64,
    query_time_source: &'static str,
    construction: &BenchmarkConstructionMetrics,
    isolation: &'static str,
    history_identity_sha256: Option<&str>,
    session_query_index: Option<usize>,
) -> Result<BenchmarkResponse> {
    let strategy_name = config.strategies.recall.as_str();
    let strategy_reference_time_ms =
        history_visibility.strategy_reference_time_ms(query_time_ms, prepared);
    let recall_started = Instant::now();
    let query = SearchQuery {
        text: question.query.clone(),
        repository: Some(prepared.repository.clone()),
        agent: Some(AgentKind::Codex),
        since_ms: None,
        until_ms: Some(strategy_reference_time_ms),
        limit,
    };
    let recall_result = match execution {
        BenchmarkExecution::Full => {
            strategy::recall_result_at(store, config, query, strategy_reference_time_ms).await
        }
        BenchmarkExecution::RetrievalOnly => {
            strategy::recall_lazymem_retrieval_result_at(
                store,
                config,
                query,
                strategy_reference_time_ms,
            )
            .await
        }
    }
    .with_context(|| {
        format!(
            "recall benchmark question {} with {strategy_name}",
            question.id
        )
    })?;
    let recall_elapsed = recall_started.elapsed();

    let (mut context, item_metadata, source_event_ids) =
        extract_strategy_items(strategy_name, recall_result.items)?;
    let strategy_metadata = if recall_result.metadata.is_empty() {
        item_metadata
    } else {
        recall_result.metadata
    };
    let (retrieved_event_ids, retrieved_source) =
        retrieval_event_ids(&strategy_metadata, &source_event_ids);
    let (covered_event_ids, covered_source) =
        covered_event_ids(&strategy_metadata, &source_event_ids);
    let (retained_event_ids, retained_source) = if execution == BenchmarkExecution::RetrievalOnly {
        (Vec::new(), "retrieval_only_no_memory_construction")
    } else {
        strategy_benchmark_trace_ids(&strategy_metadata, "source_event_ids")
            .map(|ids| (ids, "strategy_benchmark_trace.source_event_ids"))
            .unwrap_or_else(|| (source_event_ids.clone(), "memory_source_event_ids_fallback"))
    };
    let retrieved_message_ids = map_event_ids(&retrieved_event_ids, &prepared.message_ids_by_event)
        .context("map strategy retrieval IDs back to benchmark messages")?;
    let covered_message_ids = map_event_ids(&covered_event_ids, &prepared.message_ids_by_event)
        .context("map strategy coverage IDs back to benchmark messages")?;
    let retained_message_ids = map_event_ids(&retained_event_ids, &prepared.message_ids_by_event)
        .context("map retained source IDs back to benchmark messages")?;
    let context_source = if execution == BenchmarkExecution::RetrievalOnly {
        context = render_retrieval_only_context(&covered_event_ids, &prepared.events)?;
        "expanded_messages_unprocessed"
    } else {
        "strategy_recall_items"
    };
    let stage_rankings = stage_rankings(
        strategy_name,
        &strategy_metadata,
        &retrieved_event_ids,
        &prepared.message_ids_by_event,
    )?;

    let (retrieval_latency_s, processing_latency_s, strategy_total_latency_s, timing_source) =
        strategy_timing(&strategy_metadata, recall_elapsed.as_secs_f64());

    let benchmark_trace = json!({
        "schema_version": BENCHMARK_TRACE_VERSION,
        "strategy": strategy_name,
        "execution_mode": execution.as_str(),
        "context_source": context_source,
        "isolation": isolation,
        "history_identity_sha256": history_identity_sha256,
        "session_query_index": session_query_index,
        "construction_scope": if history_identity_sha256.is_some() { "shared_history" } else { "question" },
        "repository": prepared.repository,
        "source_agent": "codex_benchmark_placeholder",
        "question_type": question.question_type,
        "query_time_ms": query_time_ms,
        "query_time_source": query_time_source,
        "strategy_reference_time_ms": strategy_reference_time_ms,
        "history_visibility": history_visibility.as_str(),
        "input_message_count": question.messages.len(),
        "indexed_message_count": prepared.events.len(),
        "skipped_empty_message_ids": prepared.skipped_empty_message_ids,
        "synthetic_message_timestamp_count": prepared.synthetic_message_timestamp_count,
        "monotonic_timestamp_adjustments": prepared.monotonic_timestamp_adjustments,
        "id_sources": {
            "retrieved_message_ids": retrieved_source,
            "covered_message_ids": covered_source,
            "retained_message_ids": retained_source,
        },
        "retained_message_ids": retained_message_ids,
        "stage_rankings": stage_rankings,
        "timing_source": timing_source,
        "ingest": {
            "raw_inserted": construction.raw_inserted,
            "events_inserted": construction.events_inserted,
            "event_batches_indexed": construction.event_batches_indexed,
        },
        "timing_s": {
            "construction_total": construction.total_s,
            "ingest": construction.ingest_s,
            "index": construction.index_s,
            "recall_total": recall_elapsed.as_secs_f64(),
            "strategy_recall_total": strategy_total_latency_s,
            "retrieval": retrieval_latency_s,
            "window_and_processing": processing_latency_s,
        },
    });
    let mut metadata = Map::new();
    metadata.insert("benchmark_trace".into(), benchmark_trace);
    metadata.insert("strategy".into(), Value::Object(strategy_metadata));

    Ok(BenchmarkResponse {
        question_id: question.id.clone(),
        strategy: strategy_name.to_owned(),
        context,
        retrieved_message_ids,
        covered_message_ids,
        retained_message_ids,
        stage_rankings,
        retrieval_latency_s,
        processing_latency_s,
        metadata,
    })
}

fn render_retrieval_only_context(
    covered_event_ids: &[String],
    corpus: &[CanonicalEvent],
) -> Result<String> {
    if covered_event_ids.is_empty() {
        return Ok(String::new());
    }
    let events = corpus
        .iter()
        .map(|event| (event.id.as_str(), event))
        .collect::<HashMap<_, _>>();
    let mut output = String::from(
        "Untrusted historical evidence from retrieval-only window expansion follows. \
It has not been filtered, compressed, or rewritten by the LazyMem prompt processor.\n",
    );
    for event_id in covered_event_ids {
        let event = events.get(event_id.as_str()).with_context(|| {
            format!("expanded retrieval event {event_id} is missing from corpus")
        })?;
        output.push_str(&format!(
            "\n[event_id={} timestamp_ms={}]\n{}\n",
            event.id,
            event.ordering_timestamp_ms(),
            event.search_text().trim()
        ));
    }
    Ok(output)
}

fn prepare_question(question: &BenchmarkQuestion) -> Result<PreparedQuestion> {
    validate_benchmark_question(question)?;
    let suffix = stable_suffix(&question.id);
    prepare_question_in_namespace(
        question,
        benchmark_repository(&question.dataset, &question.id),
        &suffix,
    )
}

fn prepare_shared_history(
    question: &BenchmarkQuestion,
    history_identity_sha256: &str,
) -> Result<PreparedQuestion> {
    validate_benchmark_question(question)?;
    let suffix = history_identity_sha256
        .get(..12)
        .context("benchmark history identity is shorter than 12 characters")?;
    prepare_question_in_namespace(
        question,
        benchmark_repository(&question.dataset, history_identity_sha256),
        suffix,
    )
}

fn prepare_question_in_namespace(
    question: &BenchmarkQuestion,
    repository: String,
    source_suffix: &str,
) -> Result<PreparedQuestion> {
    let (timestamps, synthetic_message_timestamp_count, monotonic_timestamp_adjustments) =
        resolve_message_timestamps(&question.messages);
    let history_max_ms = timestamps.iter().copied().max();
    let (query_time_ms, query_time_source) =
        resolve_query_time(&question.query_time, history_max_ms);
    let source_name = format!(
        "benchmark:{}:{}",
        path_segment(&question.dataset),
        source_suffix
    );

    let mut events = Vec::with_capacity(question.messages.len());
    let mut raw_records = Vec::with_capacity(question.messages.len());
    let mut raw_contexts = Vec::with_capacity(question.messages.len());
    let mut message_ids_by_event = HashMap::with_capacity(question.messages.len());
    let mut skipped_empty_message_ids = Vec::new();
    let mut opaque_sessions = HashMap::<String, String>::new();

    for (position, (message, timestamp_ms)) in question.messages.iter().zip(timestamps).enumerate()
    {
        if message.content.trim().is_empty() {
            skipped_empty_message_ids.push(message.id.clone());
            continue;
        }
        let external_session_id = if message.session_id.trim().is_empty() {
            "__default__"
        } else {
            message.session_id.as_str()
        };
        let next_session_index = opaque_sessions.len();
        let session_id = opaque_sessions
            .entry(external_session_id.to_owned())
            .or_insert_with(|| format!("session-{next_session_index:06}"))
            .clone();
        let opaque_message_id = format!("message-{position:08}");
        let source = EventSource::new(
            AgentKind::Codex,
            session_id.clone(),
            None,
            Some(opaque_message_id.clone()),
            "benchmark_message",
            None,
            Some(repository.clone()),
        );
        let mut event = CanonicalEvent::message_from_text(
            source,
            message_role(&message.role),
            message.content.clone(),
            Some(timestamp_ms),
        );
        let raw = RawRecord::new(
            AgentKind::Codex,
            source_name.clone(),
            position as u64,
            benchmark_raw_payload(
                message,
                position,
                &opaque_message_id,
                &session_id,
                &repository,
            ),
        );
        event.attach_raw_record(raw.id.clone());
        message_ids_by_event.insert(event.id.clone(), message.id.clone());
        raw_contexts.push(RawContext {
            raw_record_id: raw.id.clone(),
            session_id: Some(session_id),
            repository: Some(repository.clone()),
            occurred_at_ms: Some(timestamp_ms),
        });
        raw_records.push(raw);
        events.push(event);
    }

    Ok(PreparedQuestion {
        repository,
        history_max_ms,
        query_time_ms,
        query_time_source,
        events,
        raw_records,
        raw_contexts,
        message_ids_by_event,
        skipped_empty_message_ids,
        synthetic_message_timestamp_count,
        monotonic_timestamp_adjustments,
    })
}

fn benchmark_raw_payload(
    message: &BenchmarkMessage,
    position: usize,
    opaque_message_id: &str,
    session_id: &str,
    repository: &str,
) -> Value {
    json!({
        "type": "benchmark_message",
        "message": {
            "id": opaque_message_id,
            "index": message.index.unwrap_or(position),
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "session_id": session_id,
        },
        "source": {
            "agent": "codex",
            "native_event": "benchmark_message",
            "repository": repository,
        },
    })
}

fn message_role(role: &str) -> MessageRole {
    match role.trim().to_ascii_lowercase().as_str() {
        "user" | "human" => MessageRole::User,
        "assistant" | "model" | "bot" => MessageRole::Assistant,
        "system" => MessageRole::System,
        "developer" => MessageRole::Developer,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Unknown,
    }
}

fn benchmark_repository(dataset: &str, question_id: &str) -> String {
    format!(
        "{}/{}/case-{}",
        BENCHMARK_REPOSITORY_ROOT,
        path_segment(dataset),
        stable_suffix(question_id)
    )
}

fn benchmark_history_identity(question: &BenchmarkQuestion) -> Result<String> {
    validate_benchmark_question(question)?;
    let mut digest = Sha256::new();
    digest.update(b"semafuse/benchmark-history-identity/v1");
    digest_history_field(&mut digest, &question.dataset);
    digest.update((question.messages.len() as u64).to_be_bytes());
    for (position, message) in question.messages.iter().enumerate() {
        digest.update((message.index.unwrap_or(position) as u64).to_be_bytes());
        for field in [
            message.id.as_str(),
            message.role.as_str(),
            message.content.as_str(),
            message.timestamp.as_str(),
            message.session_id.as_str(),
        ] {
            digest_history_field(&mut digest, field);
        }
    }
    Ok(hex::encode(digest.finalize()))
}

fn digest_history_field(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value.as_bytes());
}

fn path_segment(value: &str) -> String {
    let segment = value
        .chars()
        .map(|character| match character {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => character,
            _ => '_',
        })
        .collect::<String>();
    if segment.trim_matches('_').is_empty() || matches!(segment.as_str(), "." | "..") {
        "unnamed".into()
    } else {
        segment
    }
}

fn stable_suffix(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    hex::encode(digest)[..12].to_owned()
}

fn resolve_message_timestamps(messages: &[BenchmarkMessage]) -> (Vec<i64>, usize, usize) {
    let mut timestamps = Vec::with_capacity(messages.len());
    let mut previous = None;
    let mut synthetic_count = 0;
    let mut monotonic_adjustments = 0;
    for message in messages {
        let parsed = parse_timestamp(&message.timestamp);
        let mut timestamp = match parsed {
            Some(value) => value.start_ms,
            None => {
                synthetic_count += 1;
                previous.map_or(0, |value: i64| value.saturating_add(1))
            }
        };
        if previous.is_some_and(|value| timestamp <= value) {
            timestamp = previous.expect("checked above").saturating_add(1);
            monotonic_adjustments += 1;
        }
        previous = Some(timestamp);
        timestamps.push(timestamp);
    }
    (timestamps, synthetic_count, monotonic_adjustments)
}

fn resolve_query_time(input: &str, history_max_ms: Option<i64>) -> (i64, &'static str) {
    match parse_timestamp(input) {
        Some(parsed) if parsed.date_only => (parsed.cutoff_ms, "input_date_only_end_of_day"),
        Some(parsed) => (parsed.cutoff_ms, "input_timestamp"),
        None => (
            history_max_ms.map_or(0, |value| value.saturating_add(1)),
            "history_max_plus_one",
        ),
    }
}

fn parse_timestamp(value: &str) -> Option<ParsedTimestamp> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(raw) = value.parse::<i64>() {
        let milliseconds = if raw.unsigned_abs() < 10_000_000_000 {
            raw.saturating_mul(1_000)
        } else {
            raw
        };
        return Some(ParsedTimestamp {
            start_ms: milliseconds,
            cutoff_ms: milliseconds,
            date_only: false,
        });
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        let milliseconds = timestamp.timestamp_millis();
        return Some(ParsedTimestamp {
            start_ms: milliseconds,
            cutoff_ms: milliseconds,
            date_only: false,
        });
    }
    for format in [
        "%Y/%m/%d (%a) %H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%d %B %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S",
    ] {
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(value, format) {
            let milliseconds = timestamp.and_utc().timestamp_millis();
            return Some(ParsedTimestamp {
                start_ms: milliseconds,
                cutoff_ms: milliseconds,
                date_only: false,
            });
        }
    }
    for format in ["%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"] {
        if let Ok(date) = NaiveDate::parse_from_str(value, format) {
            return Some(ParsedTimestamp {
                start_ms: date.and_hms_opt(0, 0, 0)?.and_utc().timestamp_millis(),
                cutoff_ms: date
                    .and_hms_milli_opt(23, 59, 59, 999)?
                    .and_utc()
                    .timestamp_millis(),
                date_only: true,
            });
        }
    }
    None
}

fn extract_strategy_items(
    strategy_name: &str,
    items: Vec<RecallItem>,
) -> Result<(String, Map<String, Value>, Vec<String>)> {
    match items.as_slice() {
        [] => Ok((String::new(), Map::new(), Vec::new())),
        [
            RecallItem::Memory {
                namespace,
                content,
                metadata,
                source_event_ids,
                ..
            },
        ] => {
            ensure!(
                namespace == strategy_name,
                "benchmark strategy {strategy_name} returned memory namespace {namespace}"
            );
            Ok((content.clone(), metadata.clone(), source_event_ids.clone()))
        }
        _ if items
            .iter()
            .all(|item| matches!(item, RecallItem::Event { .. })) =>
        {
            ensure!(
                strategy_name != "lazymem",
                "LazyMem benchmark received an event-backed result"
            );
            let mut context = String::new();
            let mut source_event_ids = Vec::with_capacity(items.len());
            for item in items {
                let RecallItem::Event { event, .. } = item else {
                    unreachable!("event-only result was checked above")
                };
                source_event_ids.push(event.id.clone());
                context.push_str(&format!(
                    "[event_id={} timestamp_ms={}]\n{}\n\n",
                    event.id,
                    event.ordering_timestamp_ms(),
                    event.search_text().trim()
                ));
            }
            Ok((context.trim_end().to_owned(), Map::new(), source_event_ids))
        }
        _ => bail!(
            "benchmark strategy {strategy_name} mixed event and memory items or returned multiple memories"
        ),
    }
}

fn stage_rankings(
    strategy_name: &str,
    metadata: &Map<String, Value>,
    retrieved_event_ids: &[String],
    message_ids_by_event: &HashMap<String, String>,
) -> Result<BTreeMap<String, Vec<String>>> {
    let mut rankings = BTreeMap::new();
    if strategy_name == "lazymem" {
        if let Some(candidates) = metadata
            .get("rrf")
            .and_then(|value| value.get("candidates"))
            .and_then(Value::as_array)
        {
            for (stage, rank_field) in [("dense", "dense_rank"), ("bm25", "sparse_rank")] {
                let mut ranked = candidates
                    .iter()
                    .filter_map(|candidate| {
                        Some((
                            candidate.get(rank_field)?.as_u64()? as usize,
                            candidate.get("message_id")?.as_str()?.to_owned(),
                        ))
                    })
                    .collect::<Vec<_>>();
                ranked.sort_by_key(|(rank, _)| *rank);
                let event_ids = ranked.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
                rankings.insert(
                    stage.to_owned(),
                    map_event_ids(&event_ids, message_ids_by_event)
                        .with_context(|| format!("map {stage} stage IDs"))?,
                );
            }
        }
        if let Some(event_ids) = metadata
            .get("rrf")
            .and_then(|value| value.get("trace"))
            .and_then(|value| value.get("fused_order"))
            .and_then(Value::as_array)
            .map(|values| string_values(values))
        {
            rankings.insert(
                "rrf".into(),
                map_event_ids(&event_ids, message_ids_by_event).context("map RRF stage IDs")?,
            );
        }
    }
    rankings.insert(
        "retrieved".into(),
        map_event_ids(retrieved_event_ids, message_ids_by_event)
            .context("map final retrieval stage IDs")?,
    );
    Ok(rankings)
}

fn retrieval_event_ids(
    metadata: &Map<String, Value>,
    fallback: &[String],
) -> (Vec<String>, &'static str) {
    if let Some(ids) = strategy_benchmark_trace_ids(metadata, "retrieved_event_ids") {
        return (ids, "strategy_benchmark_trace.retrieved_event_ids");
    }
    let ids = metadata
        .get("rerank")
        .and_then(|value| value.get("output_order"))
        .and_then(Value::as_array)
        .map(|values| string_values(values))
        .unwrap_or_default();
    if ids.is_empty() {
        (fallback.to_vec(), "memory_source_event_ids_fallback")
    } else {
        (ids, "rerank.output_order")
    }
}

fn covered_event_ids(
    metadata: &Map<String, Value>,
    fallback: &[String],
) -> (Vec<String>, &'static str) {
    if let Some(ids) = strategy_benchmark_trace_ids(metadata, "covered_event_ids") {
        return (ids, "strategy_benchmark_trace.covered_event_ids");
    }
    let Some(windows) = metadata
        .get("windows")
        .and_then(|value| value.get("windows"))
        .and_then(Value::as_array)
    else {
        return (fallback.to_vec(), "memory_source_event_ids_fallback");
    };
    let mut ids = Vec::new();
    let mut seen = HashSet::new();
    for window in windows {
        let Some(message_ids) = window.get("message_ids").and_then(Value::as_array) else {
            continue;
        };
        for id in string_values(message_ids) {
            if seen.insert(id.clone()) {
                ids.push(id);
            }
        }
    }
    if ids.is_empty() {
        (fallback.to_vec(), "memory_source_event_ids_fallback")
    } else {
        (ids, "windows.windows[].message_ids")
    }
}

fn strategy_benchmark_trace_ids(metadata: &Map<String, Value>, field: &str) -> Option<Vec<String>> {
    metadata
        .get("strategy_benchmark_trace")
        .and_then(|trace| trace.get(field))
        .and_then(Value::as_array)
        .map(|values| string_values(values))
}

fn string_values(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn map_event_ids(
    ids: &[String],
    message_ids_by_event: &HashMap<String, String>,
) -> Result<Vec<String>> {
    let mut mapped = Vec::new();
    let mut seen = HashSet::new();
    for id in ids {
        let message_id = message_ids_by_event
            .get(id)
            .with_context(|| format!("strategy trace refers to unknown canonical event {id}"))?;
        if seen.insert(message_id.clone()) {
            mapped.push(message_id.clone());
        }
    }
    Ok(mapped)
}

fn metadata_timing_ms(metadata: &Map<String, Value>, name: &str) -> f64 {
    metadata
        .get("timing")
        .and_then(|value| value.get(name))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}

fn strategy_timing(
    metadata: &Map<String, Value>,
    runner_recall_s: f64,
) -> (f64, f64, f64, &'static str) {
    if metadata.get("timing").is_some() {
        let retrieval = metadata_timing_ms(metadata, "retrieval_ms") / 1_000.0;
        let processing = (metadata_timing_ms(metadata, "window_construction_ms")
            + metadata_timing_ms(metadata, "processing_ms"))
            / 1_000.0;
        let total = metadata_timing_ms(metadata, "total_ms") / 1_000.0;
        (retrieval, processing, total, "strategy_metadata_timing")
    } else {
        (
            runner_recall_s,
            0.0,
            runner_recall_s,
            "runner_recall_total_fallback",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BlobConfig, StrategyConfig};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    fn message(id: &str, timestamp: &str, content: &str, session_id: &str) -> BenchmarkMessage {
        BenchmarkMessage {
            id: id.into(),
            index: None,
            role: "assistant".into(),
            content: content.into(),
            timestamp: timestamp.into(),
            session_id: session_id.into(),
        }
    }

    #[test]
    fn date_only_query_cutoff_includes_same_day_history_in_order() {
        let (timestamps, synthetic, adjustments) = resolve_message_timestamps(&[
            message("m1", "2026-01-03", "one", "s"),
            message("m2", "2026-01-03", "two", "s"),
        ]);
        assert_eq!(synthetic, 0);
        assert_eq!(adjustments, 1);
        assert!(timestamps[0] < timestamps[1]);
        let (query_time, source) = resolve_query_time("2026-01-03", timestamps.last().copied());
        assert_eq!(source, "input_date_only_end_of_day");
        assert!(query_time > timestamps[1]);
    }

    #[test]
    fn parses_longmemeval_weekday_timestamp() {
        let parsed = parse_timestamp("2023/05/30 (Tue) 23:40").unwrap();
        assert!(!parsed.date_only);
        assert_eq!(
            parsed.start_ms,
            NaiveDate::from_ymd_opt(2023, 5, 30)
                .unwrap()
                .and_hms_opt(23, 40, 0)
                .unwrap()
                .and_utc()
                .timestamp_millis()
        );
    }

    #[test]
    fn unparseable_query_time_uses_history_not_the_wall_clock() {
        let (query_time, source) = resolve_query_time("not-a-time", Some(42));
        assert_eq!(query_time, 43);
        assert_eq!(source, "history_max_plus_one");
    }

    #[test]
    fn date_only_query_time_does_not_include_a_future_day() {
        let future = parse_timestamp("2026-01-04").unwrap().start_ms;
        let (query_time, _) = resolve_query_time("2026-01-03", Some(future));
        assert!(future > query_time);
    }

    #[test]
    fn preparation_isolated_and_does_not_retain_gold_or_query_fields() {
        let question: BenchmarkQuestion = serde_json::from_value(json!({
            "id": "conv-49:q0001",
            "dataset": "locomo",
            "question_type": "multi-hop",
            "query": "Where did they meet?",
            "answer": "reference answer that must not enter the store",
            "gold_message_ids": ["m1"],
            "metadata": {"referenced_reasoning_chain": "also a label"},
            "query_time": "1 January 2026",
            "messages": [
                {"id": "m1", "index": 0, "role": "assistant", "content": "Past evidence", "timestamp": "1 January 2026", "session_id": "session_1", "has_answer": true},
                {"id": "m2", "index": 1, "role": "assistant", "content": "   ", "timestamp": "1 January 2026", "session_id": "session_1"},
            ],
        }))
        .unwrap();
        let prepared = prepare_question(&question).unwrap();
        assert!(prepared.repository.starts_with(BENCHMARK_REPOSITORY_ROOT));
        assert_eq!(prepared.events.len(), 1);
        assert_eq!(prepared.skipped_empty_message_ids, vec!["m2"]);
        assert_eq!(prepared.events[0].source.native_event, "benchmark_message");
        let raw = serde_json::to_string(&prepared.raw_records[0].payload).unwrap();
        assert!(!raw.contains("Where did they meet"));
        assert!(!raw.contains("reference answer"));
        assert!(!raw.contains("referenced_reasoning_chain"));
        assert!(!raw.contains("has_answer"));
        assert!(raw.contains("Past evidence"));
    }

    #[test]
    fn internal_ids_are_mapped_back_to_benchmark_message_ids() {
        let mapped = map_event_ids(
            &["evt_b".into(), "evt_a".into(), "evt_b".into()],
            &HashMap::from([
                ("evt_a".into(), "message-a".into()),
                ("evt_b".into(), "message-b".into()),
            ]),
        )
        .unwrap();
        assert_eq!(mapped, vec!["message-b", "message-a"]);
    }

    #[test]
    fn persistent_session_history_identity_is_query_agnostic_and_history_strict() {
        let base: BenchmarkQuestion = serde_json::from_value(json!({
            "id": "q1",
            "dataset": "locomo",
            "question_type": "single-hop",
            "query": "first query",
            "query_time": "2026-01-02",
            "messages": [
                {"id": "m1", "role": "user", "content": "alpha", "timestamp": "2026-01-01", "session_id": "s1"},
                {"id": "m2", "role": "assistant", "content": "beta", "timestamp": "2026-01-01", "session_id": "s1"}
            ]
        }))
        .unwrap();
        let identity = benchmark_history_identity(&base).unwrap();

        let mut another_query = base.clone();
        another_query.id = "q2".into();
        another_query.query = "different query".into();
        another_query.query_time = "2026-02-01".into();
        another_query.question_type = "temporal".into();
        another_query.messages[0].index = Some(0);
        another_query.messages[1].index = Some(1);
        ensure_session_history(&identity, &another_query).unwrap();

        let mut mutations = Vec::new();
        let mut changed = base.clone();
        changed.dataset = "another-dataset".into();
        mutations.push(changed);
        let mut changed = base.clone();
        changed.messages[0].id = "another-message".into();
        mutations.push(changed);
        let mut changed = base.clone();
        changed.messages[0].role = "assistant".into();
        mutations.push(changed);
        let mut changed = base.clone();
        changed.messages[0].content = "changed".into();
        mutations.push(changed);
        let mut changed = base.clone();
        changed.messages[0].timestamp = "2026-01-03".into();
        mutations.push(changed);
        let mut changed = base;
        changed.messages[0].session_id = "another-session".into();
        mutations.push(changed);

        for mutation in mutations {
            let error = ensure_session_history(&identity, &mutation).unwrap_err();
            assert!(
                format!("{error:#}")
                    .contains("persistent benchmark retrieval session history mismatch")
            );
        }
    }

    #[test]
    fn stable_strategy_benchmark_trace_takes_precedence_over_internal_metadata() {
        let metadata = Map::from_iter([
            (
                "strategy_benchmark_trace".into(),
                json!({"retrieved_event_ids": ["evt_trace"]}),
            ),
            ("rerank".into(), json!({"output_order": ["evt_internal"]})),
        ]);
        let (ids, source) = retrieval_event_ids(&metadata, &[]);
        assert_eq!(ids, vec!["evt_trace"]);
        assert_eq!(source, "strategy_benchmark_trace.retrieved_event_ids");
    }

    #[test]
    fn generic_memory_strategy_uses_stable_trace_timing_and_namespace() {
        let metadata = Map::from_iter([
            (
                "strategy_benchmark_trace".into(),
                json!({
                    "retrieved_event_ids": ["retrieved"],
                    "covered_event_ids": ["covered"],
                    "source_event_ids": ["retained"],
                }),
            ),
            (
                "timing".into(),
                json!({
                    "retrieval_ms": 125.0,
                    "window_construction_ms": 25.0,
                    "processing_ms": 50.0,
                    "total_ms": 225.0,
                }),
            ),
        ]);
        let (retrieved, retrieved_source) = retrieval_event_ids(&metadata, &["fallback".into()]);
        let (covered, covered_source) = covered_event_ids(&metadata, &["fallback".into()]);
        assert_eq!(retrieved, ["retrieved"]);
        assert_eq!(
            retrieved_source,
            "strategy_benchmark_trace.retrieved_event_ids"
        );
        assert_eq!(covered, ["covered"]);
        assert_eq!(covered_source, "strategy_benchmark_trace.covered_event_ids");
        assert_eq!(
            strategy_benchmark_trace_ids(&metadata, "source_event_ids").unwrap(),
            ["retained"]
        );
        assert_eq!(
            strategy_timing(&metadata, 9.0),
            (0.125, 0.075, 0.225, "strategy_metadata_timing")
        );

        let memory = RecallItem::Memory {
            namespace: "reconstructive".into(),
            id: "memory".into(),
            content: "verbatim context".into(),
            metadata: metadata.clone(),
            source_event_ids: vec!["retained".into()],
            score: 1.0,
        };
        let (content, extracted_metadata, source_ids) =
            extract_strategy_items("reconstructive", vec![memory]).unwrap();
        assert_eq!(content, "verbatim context");
        assert_eq!(extracted_metadata, metadata);
        assert_eq!(source_ids, ["retained"]);

        let wrong_namespace = RecallItem::Memory {
            namespace: "lazymem".into(),
            id: "memory".into(),
            content: String::new(),
            metadata: Map::new(),
            source_event_ids: Vec::new(),
            score: 0.0,
        };
        let error = extract_strategy_items("reconstructive", vec![wrong_namespace]).unwrap_err();
        assert!(format!("{error:#}").contains("returned memory namespace lazymem"));
    }

    #[tokio::test]
    async fn generic_runner_scores_history_without_a_lazymem_backend() {
        let config = Config {
            strategies: StrategyConfig {
                enabled: vec!["history".into()],
                recall: "history".into(),
                settings: HashMap::from([("history".into(), json!({"retrieval": "lexical"}))]),
            },
            blob: BlobConfig::default(),
        };
        let question: BenchmarkQuestion = serde_json::from_value(json!({
            "id": "answer_case_abs",
            "dataset": "longmemeval",
            "question_type": "single-session-user",
            "query": "quartz",
            "query_time": "2023/06/01 (Thu) 12:00",
            "messages": [
                {"id": "answer_session_01", "index": 0, "role": "user", "content": "quartz evidence", "timestamp": "2023/05/30 (Tue) 23:40", "session_id": "answer_session"},
                {"id": "noise_01", "index": 1, "role": "assistant", "content": "unrelated", "timestamp": "2023/05/31 (Wed) 10:00", "session_id": "noise"}
            ],
            "answer": "must be ignored",
            "gold_message_ids": ["answer_session_01"]
        }))
        .unwrap();

        let response = run_strategy_question(&config, question, 50).await.unwrap();
        assert_eq!(response.strategy, "history");
        assert_eq!(response.retrieved_message_ids, vec!["answer_session_01"]);
        assert_eq!(response.covered_message_ids, response.retrieved_message_ids);
        assert_eq!(
            response.retained_message_ids,
            response.retrieved_message_ids
        );
        assert_eq!(
            response.stage_rankings["retrieved"],
            response.retrieved_message_ids
        );
        let trace = &response.metadata["benchmark_trace"];
        assert_eq!(trace["synthetic_message_timestamp_count"], 0);
        assert!(!response.context.contains("answer_session"));
    }

    #[tokio::test]
    async fn persistent_session_reuses_index_and_processes_every_full_query() {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock inference server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let embedding_batch_sizes = Arc::new(Mutex::new(Vec::new()));
        let processor_calls = Arc::new(AtomicUsize::new(0));
        let server_embedding_batch_sizes = embedding_batch_sizes.clone();
        let server_processor_calls = processor_calls.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let embedding_batch_sizes = server_embedding_batch_sizes.clone();
                let processor_calls = server_processor_calls.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0_u8; 4_096];
                    let header_end = loop {
                        let count = stream.read(&mut buffer).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        request.extend_from_slice(&buffer[..count]);
                        if let Some(position) =
                            request.windows(4).position(|part| part == b"\r\n\r\n")
                        {
                            break position + 4;
                        }
                    };
                    let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    while request.len() < header_end + content_length {
                        let count = stream.read(&mut buffer).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        request.extend_from_slice(&buffer[..count]);
                    }
                    let request_line = headers.lines().next().unwrap();
                    let body: Value =
                        serde_json::from_slice(&request[header_end..header_end + content_length])
                            .unwrap();
                    let response = if request_line.contains("/embeddings") {
                        embedding_batch_sizes
                            .lock()
                            .unwrap()
                            .push(body["input"].as_array().unwrap().len());
                        json!({
                            "data": body["input"].as_array().unwrap().iter().enumerate().map(
                                |(index, _)| json!({"index": index, "embedding": [1.0, 0.0]})
                            ).collect::<Vec<_>>()
                        })
                    } else if request_line.contains("/rerank") {
                        Value::Array(
                            body["texts"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .enumerate()
                                .map(|(index, _)| json!({"index": index, "score": 1.0 - index as f64 / 100.0}))
                                .collect(),
                        )
                    } else if request_line.contains("/chat/completions") {
                        processor_calls.fetch_add(1, Ordering::SeqCst);
                        let prompt = body["messages"][1]["content"].as_str().unwrap();
                        let (_, raw_window) = prompt.split_once("# History Window\n").unwrap();
                        let window: Value = serde_json::from_str(raw_window).unwrap();
                        let decisions = ["previous_bridge", "core_messages", "next_bridge"]
                            .into_iter()
                            .flat_map(|section| window[section].as_array().unwrap())
                            .map(|message| {
                                json!({
                                    "op": "KEEP",
                                    "compressed_content": message["content"],
                                    "reason": "relevant test evidence"
                                })
                            })
                            .collect::<Vec<_>>();
                        json!({
                            "choices": [{"finish_reason": "stop", "message": {"content": serde_json::to_string(&decisions).unwrap()}}]
                        })
                    } else {
                        panic!(
                            "persistent benchmark session made an unexpected request: {request_line}"
                        );
                    };
                    let encoded = serde_json::to_vec(&response).unwrap();
                    let header = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        encoded.len()
                    );
                    stream.write_all(header.as_bytes()).await.unwrap();
                    stream.write_all(&encoded).await.unwrap();
                });
            }
        });

        let endpoint = format!("http://{address}");
        let config = Config {
            strategies: StrategyConfig {
                enabled: vec!["lazymem".into()],
                recall: "lazymem".into(),
                settings: HashMap::from([(
                    "lazymem".into(),
                    json!({
                        "protocol": "custom",
                        "embedding": {"backend": "openai_compatible", "endpoint": {"base_url": format!("{endpoint}/v1"), "model": "embed"}},
                        "reranker": {"backend": "openai_compatible", "endpoint": {"base_url": endpoint, "model": "reranker"}},
                        "processor": {"backend": "openai_compatible", "endpoint": {"base_url": format!("{endpoint}/v1"), "model": "processor"}},
                        "retrieval": {"dense_limit": 3, "sparse_limit": 3, "fused_limit": 3, "rerank_top_n": 2},
                        "window": {"radius": 1, "merge_rule": "repository_contiguous", "max_messages": 3, "stride": 2},
                        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "max_concurrency": 1, "enable_thinking": false}
                    }),
                )]),
            },
            blob: BlobConfig::default(),
        };
        let question: BenchmarkQuestion = serde_json::from_value(json!({
            "id": "retrieval-only-case",
            "dataset": "fixture",
            "query": "migration",
            "query_time": "2025-12-31",
            "messages": [
                {"id": "m1", "role": "user", "content": "migration planning", "timestamp": "2026-01-01 10:00:00", "session_id": "s1"},
                {"id": "m2", "role": "assistant", "content": "migration completed", "timestamp": "2026-01-01 10:01:00", "session_id": "s1"},
                {"id": "m3", "role": "user", "content": "unrelated note", "timestamp": "2026-01-01 10:02:00", "session_id": "s1"}
            ]
        }))
        .unwrap();

        let mut session = StrategyBenchmarkSession::new(&config, &question)
            .await
            .unwrap();
        assert_eq!(session.construction_metrics().events_inserted, 3);
        let response = session.query_retrieval(question.clone(), 2).await.unwrap();
        assert!(response.retained_message_ids.is_empty());
        assert!(!response.retrieved_message_ids.is_empty());
        assert!(!response.covered_message_ids.is_empty());
        assert!(response.context.contains("retrieval-only window expansion"));
        assert!(
            response
                .context
                .contains("has not been filtered, compressed")
        );
        assert_eq!(response.metadata["strategy"]["processor_invoked"], false);
        assert_eq!(processor_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            response.metadata["benchmark_trace"]["history_visibility"],
            "all_provided_messages"
        );
        assert!(
            response.metadata["benchmark_trace"]["strategy_reference_time_ms"]
                .as_i64()
                .unwrap()
                > response.metadata["benchmark_trace"]["query_time_ms"]
                    .as_i64()
                    .unwrap()
        );
        assert_eq!(
            response.metadata["strategy"]["processing"]["status"],
            "skipped_retrieval_only"
        );
        assert_eq!(
            response.metadata["benchmark_trace"]["id_sources"]["retained_message_ids"],
            "retrieval_only_no_memory_construction"
        );
        for stage in ["dense", "bm25", "rrf", "retrieved"] {
            assert!(response.stage_rankings.contains_key(stage));
            assert!(
                response.stage_rankings[stage]
                    .iter()
                    .all(|id| matches!(id.as_str(), "m1" | "m2" | "m3")),
                "persistent session must map canonical IDs back to source messages"
            );
        }
        assert_eq!(
            response.metadata["benchmark_trace"]["isolation"],
            "persistent_sqlite_memory_per_history"
        );
        assert_eq!(
            response.metadata["benchmark_trace"]["session_query_index"],
            0
        );
        let mut first_full_question = question.clone();
        first_full_question.id = "full-case-one".into();
        first_full_question.query = "completed".into();
        let first_full = session.query_full(first_full_question, 2).await.unwrap();
        assert_eq!(
            first_full.metadata["benchmark_trace"]["session_query_index"],
            1
        );
        assert_eq!(
            response.metadata["benchmark_trace"]["history_identity_sha256"],
            first_full.metadata["benchmark_trace"]["history_identity_sha256"]
        );
        assert_eq!(
            first_full.metadata["benchmark_trace"]["execution_mode"],
            "full"
        );
        assert_eq!(
            first_full.metadata["benchmark_trace"]["history_visibility"],
            "all_provided_messages"
        );
        assert!(
            first_full.metadata["benchmark_trace"]["strategy_reference_time_ms"]
                .as_i64()
                .unwrap()
                > first_full.metadata["benchmark_trace"]["query_time_ms"]
                    .as_i64()
                    .unwrap(),
            "full session execution must not cut off supplied history after a parsed query time"
        );
        assert_eq!(first_full.retained_message_ids, vec!["m1", "m2", "m3"]);
        assert!(first_full.context.contains("migration completed"));

        let mut second_full_question = question;
        second_full_question.id = "full-case-two".into();
        second_full_question.query = "planning".into();
        let second_full = session.query_full(second_full_question, 2).await.unwrap();
        assert_eq!(
            second_full.metadata["benchmark_trace"]["session_query_index"],
            2
        );
        assert_eq!(processor_calls.load(Ordering::SeqCst), 2);
        let embedding_batch_sizes = embedding_batch_sizes.lock().unwrap().clone();
        assert_eq!(
            embedding_batch_sizes
                .iter()
                .filter(|batch_size| **batch_size == 3)
                .count(),
            1,
            "history embeddings must be constructed exactly once"
        );
        assert_eq!(
            embedding_batch_sizes
                .iter()
                .filter(|batch_size| **batch_size == 1)
                .count(),
            3,
            "each query should make one query-embedding request"
        );
        server.abort();
    }
}
