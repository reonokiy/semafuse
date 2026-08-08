//! Isolated benchmark execution through Semafuse's production strategy paths.
//!
//! The types in this module deliberately mirror the normalized JSONL records
//! emitted by `benchmarks/lazymem`. A benchmark question is never mixed with a
//! user's durable store: every invocation gets a fresh in-memory SQLite store,
//! then uses the ordinary ingestion, strategy dispatch, and timed recall APIs.

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
    run_strategy_question(config, question, 50).await
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
    run_strategy_question_with_execution(config, question, limit, BenchmarkExecution::RetrievalOnly)
        .await
}

async fn run_strategy_question_with_execution(
    config: &Config,
    question: BenchmarkQuestion,
    limit: usize,
    execution: BenchmarkExecution,
) -> Result<BenchmarkResponse> {
    ensure!(limit > 0, "benchmark retrieval limit must be positive");
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

    let prepared = prepare_question(&question)?;
    let store = SqliteStore::open_memory()
        .await
        .context("open isolated in-memory benchmark store")?;

    let ingest_started = Instant::now();
    let ingest = store
        .ingest_batch(IngestBatch {
            raw_records: &prepared.raw_records,
            raw_contexts: &prepared.raw_contexts,
            events: &prepared.events,
            diagnostics: &[],
        })
        .await
        .with_context(|| format!("ingest benchmark question {}", question.id))?;
    let ingest_elapsed = ingest_started.elapsed();

    let indexing_started = Instant::now();
    let indexed_event_batches = strategy::dispatch_pending(&store, config, Some(strategy_name))
        .await
        .with_context(|| {
            format!(
                "index benchmark question {} with {strategy_name}",
                question.id
            )
        })?;
    let indexing_elapsed = indexing_started.elapsed();

    // LongMemEval retrieval evaluates the complete supplied haystack. Some
    // temporal-reasoning fixtures deliberately contain evidence timestamps
    // later than `question_date`, so retrieval-only execution must not treat
    // that field as a corpus cutoff. Full memory execution keeps the ordinary
    // point-in-time product semantics.
    let strategy_reference_time_ms = match execution {
        BenchmarkExecution::Full => prepared.query_time_ms,
        BenchmarkExecution::RetrievalOnly => prepared
            .events
            .iter()
            .map(CanonicalEvent::ordering_timestamp_ms)
            .max()
            .map_or(prepared.query_time_ms, |history_max| {
                prepared.query_time_ms.max(history_max.saturating_add(1))
            }),
    };
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
            strategy::recall_result_at(&store, config, query, strategy_reference_time_ms).await
        }
        BenchmarkExecution::RetrievalOnly => {
            strategy::recall_lazymem_retrieval_result_at(
                &store,
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
    let (retrieved_event_ids, retrieved_source) = if strategy_name == "lazymem" {
        retrieval_event_ids(&strategy_metadata, &source_event_ids)
    } else {
        (source_event_ids.clone(), "event_items")
    };
    let (covered_event_ids, covered_source) = if strategy_name == "lazymem" {
        covered_event_ids(&strategy_metadata, &source_event_ids)
    } else {
        (source_event_ids.clone(), "event_items")
    };
    let (retained_event_ids, retained_source) = if execution == BenchmarkExecution::RetrievalOnly {
        (Vec::new(), "retrieval_only_no_memory_construction")
    } else if strategy_name == "lazymem" {
        strategy_benchmark_trace_ids(&strategy_metadata, "source_event_ids")
            .map(|ids| (ids, "strategy_benchmark_trace.source_event_ids"))
            .unwrap_or_else(|| (source_event_ids.clone(), "memory_source_event_ids_fallback"))
    } else {
        (source_event_ids.clone(), "event_items")
    };
    let retrieved_message_ids = map_event_ids(&retrieved_event_ids, &prepared.message_ids_by_event)
        .context("map LazyMem retrieval IDs back to benchmark messages")?;
    let covered_message_ids = map_event_ids(&covered_event_ids, &prepared.message_ids_by_event)
        .context("map LazyMem coverage IDs back to benchmark messages")?;
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
        if strategy_name == "lazymem" && strategy_metadata.get("timing").is_some() {
            let retrieval = metadata_timing_ms(&strategy_metadata, "retrieval_ms") / 1_000.0;
            let processing = (metadata_timing_ms(&strategy_metadata, "window_construction_ms")
                + metadata_timing_ms(&strategy_metadata, "processing_ms"))
                / 1_000.0;
            let total = metadata_timing_ms(&strategy_metadata, "total_ms") / 1_000.0;
            (retrieval, processing, total, "lazymem_metadata_timing")
        } else {
            (
                recall_elapsed.as_secs_f64(),
                0.0,
                recall_elapsed.as_secs_f64(),
                "runner_recall_total_fallback",
            )
        };

    let benchmark_trace = json!({
        "schema_version": BENCHMARK_TRACE_VERSION,
        "strategy": strategy_name,
        "execution_mode": execution.as_str(),
        "context_source": context_source,
        "isolation": "fresh_sqlite_memory_per_question",
        "repository": prepared.repository,
        "source_agent": "codex_benchmark_placeholder",
        "question_type": question.question_type,
        "query_time_ms": prepared.query_time_ms,
        "query_time_source": prepared.query_time_source,
        "strategy_reference_time_ms": strategy_reference_time_ms,
        "history_visibility": match execution {
            BenchmarkExecution::Full => "at_query_time",
            BenchmarkExecution::RetrievalOnly => "all_provided_messages",
        },
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
            "raw_inserted": ingest.raw_inserted,
            "events_inserted": ingest.inserted,
            "event_batches_indexed": indexed_event_batches,
        },
        "timing_s": {
            "ingest": ingest_elapsed.as_secs_f64(),
            "index": indexing_elapsed.as_secs_f64(),
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
        question_id: question.id,
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

    let (timestamps, synthetic_message_timestamp_count, monotonic_timestamp_adjustments) =
        resolve_message_timestamps(&question.messages);
    let history_max_ms = timestamps.iter().copied().max();
    let (query_time_ms, query_time_source) =
        resolve_query_time(&question.query_time, history_max_ms);
    let repository = benchmark_repository(&question.dataset, &question.id);
    let source_name = format!(
        "benchmark:{}:{}",
        path_segment(&question.dataset),
        stable_suffix(&question.id)
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
    if strategy_name != "lazymem" {
        let mut context = String::new();
        let mut source_event_ids = Vec::with_capacity(items.len());
        for item in items {
            match item {
                RecallItem::Event { event, .. } => {
                    source_event_ids.push(event.id.clone());
                    context.push_str(&format!(
                        "[event_id={} timestamp_ms={}]\n{}\n\n",
                        event.id,
                        event.ordering_timestamp_ms(),
                        event.search_text().trim()
                    ));
                }
                RecallItem::Memory {
                    namespace,
                    content,
                    metadata,
                    source_event_ids,
                    ..
                } => {
                    ensure!(
                        namespace == strategy_name,
                        "benchmark strategy {strategy_name} returned memory namespace {namespace}"
                    );
                    ensure!(
                        context.is_empty(),
                        "benchmark strategy {strategy_name} mixed event and memory items"
                    );
                    return Ok((content, metadata, source_event_ids));
                }
            }
        }
        return Ok((context.trim_end().to_owned(), Map::new(), source_event_ids));
    }
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
                namespace == "lazymem",
                "LazyMem benchmark received a memory from unexpected namespace {namespace}"
            );
            Ok((content.clone(), metadata.clone(), source_event_ids.clone()))
        }
        [RecallItem::Event { .. }] => bail!("LazyMem benchmark received an event-backed result"),
        _ => bail!(
            "LazyMem benchmark expected at most one constructed memory, received {} items",
            items.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BlobConfig, StrategyConfig};
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
    async fn retrieval_only_runner_skips_an_unreachable_processor() {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock inference server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
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
                    } else {
                        panic!("retrieval-only runner made an unexpected request: {request_line}");
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
                        "processor": {"backend": "openai_compatible", "endpoint": {"base_url": "http://127.0.0.1:1/v1", "model": "unreachable-processor"}},
                        "retrieval": {"dense_limit": 3, "sparse_limit": 3, "fused_limit": 3, "rerank_top_n": 2},
                        "window": {"radius": 1, "merge_rule": "repository_contiguous", "max_messages": 3, "stride": 2}
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

        let response = run_strategy_retrieval_question(&config, question, 2)
            .await
            .unwrap();
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
        }
        server.abort();
    }
}
