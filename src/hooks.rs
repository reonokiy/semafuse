//! Real-time hook processing and crash-safe local spooling.

use crate::{
    adapters::AgentAdapter,
    blob::{BlobStore, extract_attachments},
    config::Config,
    model::{
        AgentKind, CanonicalEvent, ConversionDiagnostic, DiagnosticSeverity, RawRecord,
        SearchQuery, normalize_repository,
    },
    storage::{EventStore, IngestBatch, SqliteStore},
    strategy,
};
use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

#[derive(Debug, Serialize, Deserialize)]
struct SpoolRecord {
    agent: AgentKind,
    event: String,
    payload: Value,
}

/// Normalizes, optionally recalls against, and stores one real-time hook payload.
///
/// When `configured_event` is `auto`, the native event name is read first from
/// `hook_event_name`, then from `event.type`, and defaults to `other`. The
/// selected adapter performs the native mapping. Attachments are bound to the
/// resulting event and its identity is refreshed before ingestion.
pub async fn process_hook(
    adapter: &dyn AgentAdapter,
    store: &SqliteStore,
    configured_event: &str,
    payload: &Value,
    recall: bool,
    blobs: Option<&dyn BlobStore>,
    config: &Config,
) -> Result<Option<Value>> {
    let event_name = resolve_event_name(configured_event, payload);
    let (raw_record, events, diagnostics) = normalize_hook(
        adapter,
        event_name,
        payload,
        blobs,
        format!("hook:{}:{event_name}", adapter.kind()),
    )
    .await?;
    // Recall before ingesting this hook so the current prompt cannot consume
    // one of its own historical-context slots.
    let response = if recall {
        let prompt = payload
            .get("prompt")
            .and_then(Value::as_str)
            .or_else(|| payload.get("message").and_then(Value::as_str))
            .unwrap_or("decision constraint workaround");
        let cwd = payload
            .get("cwd")
            .and_then(Value::as_str)
            .or_else(|| payload.get("worktree").and_then(Value::as_str))
            .or_else(|| payload.get("directory").and_then(Value::as_str))
            .map(str::to_owned);
        let query = SearchQuery {
            text: prompt.to_owned(),
            repository: normalize_repository(cwd.as_deref(), None),
            agent: None,
            limit: 5,
        };
        let hits = match strategy::recall(store, config, query).await {
            Ok(hits) => hits,
            Err(error) => {
                eprintln!("semafuse recall unavailable: {error:#}");
                Vec::new()
            }
        };
        adapter.recall_response(event_name, &hits)
    } else {
        None
    };
    store
        .ingest_batch(IngestBatch {
            raw_records: std::slice::from_ref(&raw_record),
            events: &events,
            diagnostics: &diagnostics,
        })
        .await?;
    if let Err(error) = strategy::dispatch_pending(store, config, None).await {
        eprintln!("semafuse strategy delivery pending: {error:#}");
    }
    Ok(response)
}

fn resolve_event_name<'a>(configured_event: &'a str, payload: &'a Value) -> &'a str {
    if configured_event != "auto" {
        return configured_event;
    }
    payload
        .get("hook_event_name")
        .and_then(Value::as_str)
        .or_else(|| {
            payload
                .get("event")
                .and_then(|event| event.get("type"))
                .and_then(Value::as_str)
        })
        .unwrap_or("other")
}

async fn normalize_hook(
    adapter: &dyn AgentAdapter,
    event_name: &str,
    payload: &Value,
    blobs: Option<&dyn BlobStore>,
    source: String,
) -> Result<(RawRecord, Vec<CanonicalEvent>, Vec<ConversionDiagnostic>)> {
    let raw_record = RawRecord::new(adapter.kind(), source, 0, payload.clone());
    let mut events = adapter.hook_event(event_name, payload)?;
    for event in &mut events {
        event.attach_raw_record(raw_record.id.clone());
    }
    if let Some(blobs) = blobs {
        let attachments = extract_attachments(payload, blobs).await?;
        if !attachments.is_empty()
            && let Some(event) = events.last_mut()
        {
            event.add_attachments(attachments);
        }
    }
    let mut diagnostics = Vec::new();
    events.retain(|event| match event.validate() {
        Ok(()) => true,
        Err(error) => {
            diagnostics.push(ConversionDiagnostic::new(
                &raw_record.id,
                DiagnosticSeverity::Error,
                "invalid_canonical_event",
                error.to_string(),
            ));
            false
        }
    });
    Ok((raw_record, events, diagnostics))
}

/// Appends one hook payload to the JSONL spool under an exclusive writer lock.
///
/// The record is flushed with `sync_data` before the lock is released. The
/// native event name is resolved before writing, including `auto` detection.
pub fn append_spool(dir: &Path, agent: AgentKind, event: &str, payload: &Value) -> Result<()> {
    fs::create_dir_all(dir)?;
    let mut bytes = serde_json::to_vec(&SpoolRecord {
        agent,
        event: resolve_event_name(event, payload).to_owned(),
        payload: payload.clone(),
    })?;
    bytes.push(b'\n');
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("hooks.lock"))?;
    lock.lock_exclusive()?;
    let file = dir.join("hooks.jsonl");
    let mut handle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .with_context(|| format!("open spool {}", file.display()))?;
    handle.write_all(&bytes)?;
    handle.sync_data()?;
    lock.unlock()?;
    Ok(())
}

/// Replays all claimed spool generations and returns the number of records ingested.
///
/// A process-wide replay lock serializes replayers. Under the append lock the
/// live file is atomically renamed, allowing new hooks to append to a fresh
/// generation while replay continues. A generation is removed only after every
/// record succeeds, so failures leave it available for a later retry. Strategy
/// dispatch after ingestion is best effort.
pub async fn replay_spool(
    dir: &Path,
    store: &SqliteStore,
    blobs: Option<&dyn BlobStore>,
    config: &Config,
) -> Result<usize> {
    fs::create_dir_all(dir)?;
    let replay_lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("replay.lock"))?;
    replay_lock.lock_exclusive()?;
    let append_lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("hooks.lock"))?;
    append_lock.lock_exclusive()?;
    let live = dir.join("hooks.jsonl");
    if live.exists() && live.metadata()?.len() > 0 {
        let stem = format!(
            "hooks.replay.{}.{}",
            std::process::id(),
            crate::model::now_ms()
        );
        let mut suffix = 0_u32;
        let claimed = loop {
            let path = dir.join(format!("{stem}.{suffix}.jsonl"));
            if !path.try_exists()? {
                break path;
            }
            suffix += 1;
        };
        fs::rename(&live, claimed)?;
    }
    append_lock.unlock()?;

    let mut generations = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("hooks.replay.") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    generations.sort();
    let mut count = 0;
    for generation in generations {
        let text = fs::read_to_string(&generation)?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let record: SpoolRecord = serde_json::from_str(line)?;
            let adapter = crate::adapters::adapter(record.agent);
            // Resolve again for spool files written by older versions that stored
            // the literal "auto" value.
            let event_name = resolve_event_name(&record.event, &record.payload);
            let (raw_record, events, diagnostics) = normalize_hook(
                &*adapter,
                event_name,
                &record.payload,
                blobs,
                format!("spool:{}:{event_name}", record.agent),
            )
            .await?;
            store
                .ingest_batch(IngestBatch {
                    raw_records: std::slice::from_ref(&raw_record),
                    events: &events,
                    diagnostics: &diagnostics,
                })
                .await?;
            count += 1;
        }
        fs::remove_file(generation)?;
    }
    if let Err(error) = strategy::dispatch_pending(store, config, None).await {
        eprintln!("semafuse strategy delivery pending: {error:#}");
    }
    replay_lock.unlock()?;
    Ok(count)
}
