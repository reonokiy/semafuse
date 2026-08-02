//! Isolated benchmark execution through Semafuse's production LazyMem path.
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
    collections::{HashMap, HashSet},
    time::Instant,
};

const BENCHMARK_TRACE_VERSION: &str = "semafuse-lazymem-benchmark/v1";
const BENCHMARK_REPOSITORY_ROOT: &str = "/__semafuse_benchmark__";

/// One normalized LazyMem benchmark question accepted as a JSONL record.
///
/// The shape accepts `Question.to_dict()` from the Python harness. The
/// question's answer and gold annotations are intentionally not retained by
/// this runner; only the question text is used for query-time recall.
#[derive(Debug, Deserialize)]
pub struct BenchmarkQuestion {
    id: String,
    dataset: String,
    #[serde(default)]
    question_type: String,
    #[serde(alias = "question")]
    query: String,
    #[serde(default)]
    query_time: String,
    messages: Vec<BenchmarkMessage>,
}

/// One chronological message from a normalized benchmark history.
#[derive(Debug, Deserialize)]
struct BenchmarkMessage {
    id: String,
    #[serde(default)]
    index: Option<usize>,
    role: String,
    content: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    session_id: String,
}

/// One JSONL response consumed by the LazyMem benchmark harness.
#[derive(Debug, Serialize)]
pub struct BenchmarkResponse {
    question_id: String,
    context: String,
    retrieved_message_ids: Vec<String>,
    covered_message_ids: Vec<String>,
    retrieval_latency_s: f64,
    processing_latency_s: f64,
    metadata: Map<String, Value>,
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
    let indexed_event_batches = strategy::dispatch_pending(&store, config, Some("lazymem"))
        .await
        .with_context(|| format!("index benchmark question {} with LazyMem", question.id))?;
    let indexing_elapsed = indexing_started.elapsed();

    let recall_started = Instant::now();
    let recall_result = strategy::recall_result_at(
        &store,
        config,
        SearchQuery {
            text: question.query.clone(),
            repository: Some(prepared.repository.clone()),
            agent: Some(AgentKind::Codex),
            since_ms: None,
            until_ms: Some(prepared.query_time_ms),
            limit: 50,
        },
        prepared.query_time_ms,
    )
    .await
    .with_context(|| format!("recall benchmark question {} with LazyMem", question.id))?;
    let recall_elapsed = recall_started.elapsed();

    let (context, item_metadata, source_event_ids) = extract_lazymem_memory(recall_result.items)?;
    let strategy_metadata = if recall_result.metadata.is_empty() {
        item_metadata
    } else {
        recall_result.metadata
    };
    let (retrieved_event_ids, retrieved_source) =
        retrieval_event_ids(&strategy_metadata, &source_event_ids);
    let (covered_event_ids, covered_source) =
        covered_event_ids(&strategy_metadata, &source_event_ids);
    let (retained_event_ids, retained_source) =
        strategy_benchmark_trace_ids(&strategy_metadata, "source_event_ids")
            .map(|ids| (ids, "strategy_benchmark_trace.source_event_ids"))
            .unwrap_or_else(|| (source_event_ids.clone(), "memory_source_event_ids_fallback"));
    let retrieved_message_ids = map_event_ids(&retrieved_event_ids, &prepared.message_ids_by_event)
        .context("map LazyMem retrieval IDs back to benchmark messages")?;
    let covered_message_ids = map_event_ids(&covered_event_ids, &prepared.message_ids_by_event)
        .context("map LazyMem coverage IDs back to benchmark messages")?;
    let retained_message_ids =
        map_event_ids(&retained_event_ids, &prepared.message_ids_by_event)
            .context("map LazyMem retained source IDs back to benchmark messages")?;

    let (retrieval_latency_s, processing_latency_s, strategy_total_latency_s, timing_source) =
        if strategy_metadata.get("timing").is_some() {
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
        "strategy": "lazymem",
        "isolation": "fresh_sqlite_memory_per_question",
        "repository": prepared.repository,
        "source_agent": "codex_benchmark_placeholder",
        "question_type": question.question_type,
        "query_time_ms": prepared.query_time_ms,
        "query_time_source": prepared.query_time_source,
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
        context,
        retrieved_message_ids,
        covered_message_ids,
        retrieval_latency_s,
        processing_latency_s,
        metadata,
    })
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

    for (position, (message, timestamp_ms)) in question.messages.iter().zip(timestamps).enumerate()
    {
        if message.content.trim().is_empty() {
            skipped_empty_message_ids.push(message.id.clone());
            continue;
        }
        let session_id = if message.session_id.trim().is_empty() {
            "benchmark-session".to_owned()
        } else {
            message.session_id.clone()
        };
        let source = EventSource::new(
            AgentKind::Codex,
            session_id.clone(),
            None,
            Some(message.id.clone()),
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
            benchmark_raw_payload(message, position, &session_id, &repository),
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
    session_id: &str,
    repository: &str,
) -> Value {
    json!({
        "type": "benchmark_message",
        "message": {
            "id": message.id,
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
        "{}/{}/{}-{}",
        BENCHMARK_REPOSITORY_ROOT,
        path_segment(dataset),
        path_segment(question_id),
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

fn extract_lazymem_memory(
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
            .with_context(|| format!("LazyMem trace refers to unknown canonical event {id}"))?;
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
}
