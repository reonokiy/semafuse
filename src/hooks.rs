//! Real-time hook processing and crash-safe local spooling.

use crate::{
    adapters::{HookProjector, native_raw_context},
    blob::{BlobStore, drain_blob_gc, extract_attachments},
    config::Config,
    model::{
        AgentKind, CanonicalEvent, ConversionDiagnostic, DiagnosticSeverity, RawRecord,
        SearchQuery, normalize_repository,
    },
    storage::{EventStore, ForgetFilter, IngestBatch, RawContext, SqliteStore},
    strategy,
};
use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{self, Read, Seek, Write},
    path::{Path, PathBuf},
    time::{Duration, UNIX_EPOCH},
};

/// Environment variable set for an isolated LazyMem processor child.
pub const PROCESSOR_DEPTH_ENV: &str = "SEMAFUSE_PROCESSOR_DEPTH";

/// Returns whether the current process is an isolated LazyMem processor child.
///
/// Hook entry points use this marker as a final recursion barrier when managed
/// Codex policy forces hooks back on for a processor launched with hooks
/// disabled.
pub fn is_processor_child() -> bool {
    std::env::var_os(PROCESSOR_DEPTH_ENV).is_some()
}

#[derive(Debug, Serialize, Deserialize)]
struct SpoolRecord {
    agent: AgentKind,
    event: String,
    payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    captured_at_ms: Option<i64>,
}

/// Impact or result of filtering durable hook-spool payloads through a forget selector.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SpoolForgetReport {
    /// Existing spool directories inspected.
    pub directories: usize,
    /// Payload-bearing spool files inspected.
    pub files: usize,
    /// Records matching every supplied selector.
    pub matched: usize,
    /// Records left in the spool.
    pub retained: usize,
}

/// A hook payload durably staged before processing begins.
///
/// Dropping an unfinished record releases its advisory lock and leaves the
/// payload available to [`replay_spool`].
pub struct PendingSpool {
    file: std::fs::File,
    path: PathBuf,
}

impl PendingSpool {
    /// Marks the staged payload complete and removes it from the replay queue.
    pub fn complete(self) -> Result<()> {
        // Truncate and sync while the advisory lock is still held. If another
        // process observes the file between unlock and removal, it can only see
        // an empty, successfully completed generation.
        self.file.set_len(0)?;
        self.file.sync_data()?;
        self.file.unlock()?;
        drop(self.file);
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("remove completed spool {}", self.path.display())),
        }
    }
}

/// Normalizes, optionally recalls against, and stores one real-time hook payload.
///
/// When `configured_event` is `auto`, the native event name is read first from
/// `hook_event_name`, then from `event.type`, and defaults to `other`. The
/// selected adapter performs the native mapping. Attachments are bound to the
/// resulting event and its identity is refreshed before ingestion.
pub async fn process_hook(
    adapter: &dyn HookProjector,
    store: &SqliteStore,
    configured_event: &str,
    payload: &Value,
    recall: bool,
    blobs: Option<&dyn BlobStore>,
    config: &Config,
) -> Result<Option<Value>> {
    let event_name = resolve_event_name(configured_event, payload);
    // Recall before ingesting this hook so the current prompt cannot consume
    // one of its own historical-context slots.
    let recall = recall && adapter.supports_recall(event_name);
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
            since_ms: None,
            until_ms: None,
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
    let lifecycle_guard = store
        .try_lock_lifecycle()?
        .context("data lifecycle operation is busy; hook must be spooled")?;
    let (raw_record, events, diagnostics) = normalize_hook(
        adapter,
        store,
        event_name,
        payload,
        blobs,
        format!("hook:{}:{event_name}", adapter.kind()),
        None,
    )
    .await?;
    let context = raw_context(&raw_record, &events, payload, None);
    let ingest_stats = store
        .ingest_batch(IngestBatch {
            raw_records: std::slice::from_ref(&raw_record),
            raw_contexts: std::slice::from_ref(&context),
            events: &events,
            diagnostics: &diagnostics,
        })
        .await?;
    if let Some(error) = &ingest_stats.sqlite_cleanup_error {
        eprintln!(
            "semafuse removed replayed forgotten data, but SQLite physical cleanup remains pending: {error}"
        );
    }
    drop(lifecycle_guard);
    cleanup_replayed_attachments(
        store,
        blobs,
        ingest_stats.attachment_objects_queued,
        ingest_stats.unmanaged_attachment_objects,
    )
    .await;
    let deadline_sensitive = adapter.kind() == AgentKind::Codex
        && matches!(
            event_name.to_ascii_lowercase().as_str(),
            "sessionend" | "session_end"
        );
    if !deadline_sensitive && let Err(error) = strategy::dispatch_pending(store, config, None).await
    {
        eprintln!("semafuse strategy delivery pending: {error:#}");
    }
    Ok(response)
}

async fn cleanup_replayed_attachments(
    store: &SqliteStore,
    blobs: Option<&dyn BlobStore>,
    queued: usize,
    unmanaged: usize,
) {
    if unmanaged > 0 {
        eprintln!(
            "semafuse removed replayed forgotten data, but {unmanaged} legacy attachment object(s) are unmanaged and require manual cleanup"
        );
    }
    if queued == 0 {
        return;
    }
    let Some(blobs) = blobs else {
        eprintln!(
            "semafuse queued {queued} forgotten attachment object(s); run `semafuse gc` with the configured backend"
        );
        return;
    };
    match drain_blob_gc(store, blobs, queued.clamp(1, 10_000)).await {
        Ok(report) if report.pending == 0 && report.failures.is_empty() => {}
        Ok(report) => eprintln!(
            "semafuse attachment cleanup remains incomplete: {} queued, {} failure(s); run `semafuse gc`",
            report.pending,
            report.failures.len()
        ),
        Err(error) => {
            eprintln!("semafuse attachment cleanup remains queued; run `semafuse gc`: {error:#}")
        }
    }
}

/// Checks a native hook payload against durable raw-data tombstones without
/// invoking the adapter or touching attachment storage.
pub async fn hook_payload_is_tombstoned(
    adapter: &dyn HookProjector,
    store: &SqliteStore,
    configured_event: &str,
    payload: &Value,
) -> Result<bool> {
    let event_name = resolve_event_name(configured_event, payload);
    let raw_record = RawRecord::new(
        adapter.kind(),
        format!("hook:{}:{event_name}", adapter.kind()),
        0,
        payload.clone(),
    );
    let context = raw_context(&raw_record, &[], payload, None);
    if store.raw_is_tombstoned(&raw_record, &context).await? {
        return Ok(true);
    }
    // Failure handling must not spool a payload whose adapter projection is
    // already forgotten. Projection is safe here because attachment discovery
    // and writes are deliberately not invoked.
    let mut events = match adapter.hook_event(event_name, payload) {
        Ok(events) => events,
        Err(_) => return Ok(false),
    };
    for event in &mut events {
        event.attach_raw_record(raw_record.id.clone());
    }
    hook_projections_are_tombstoned(store, &raw_record, &events).await
}

async fn hook_projections_are_tombstoned(
    store: &SqliteStore,
    raw: &RawRecord,
    events: &[CanonicalEvent],
) -> Result<bool> {
    for event in events {
        let context = RawContext {
            raw_record_id: raw.id.clone(),
            session_id: Some(event.source.session_id.clone()),
            repository: event.source.repository.clone(),
            occurred_at_ms: event.occurred_at_ms,
        };
        if store.raw_is_tombstoned(raw, &context).await? || store.event_is_tombstoned(event).await?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn raw_context(
    raw: &RawRecord,
    events: &[CanonicalEvent],
    payload: &Value,
    captured_at_ms: Option<i64>,
) -> RawContext {
    let event = events.first();
    let native = native_raw_context(raw.agent, payload);
    RawContext {
        raw_record_id: raw.id.clone(),
        session_id: event
            .map(|event| event.source.session_id.clone())
            .or(native.session_id),
        repository: event
            .and_then(|event| event.source.repository.clone())
            .or(native.repository),
        occurred_at_ms: event
            .and_then(|event| event.occurred_at_ms)
            .or(native.occurred_at_ms)
            .or(captured_at_ms),
    }
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
    adapter: &dyn HookProjector,
    store: &SqliteStore,
    event_name: &str,
    payload: &Value,
    blobs: Option<&dyn BlobStore>,
    source: String,
    captured_at_ms: Option<i64>,
) -> Result<(RawRecord, Vec<CanonicalEvent>, Vec<ConversionDiagnostic>)> {
    let raw_record = RawRecord::new(adapter.kind(), source, 0, payload.clone());
    let initial_context = raw_context(&raw_record, &[], payload, captured_at_ms);
    if store
        .raw_is_tombstoned(&raw_record, &initial_context)
        .await?
    {
        return Ok((raw_record, Vec::new(), Vec::new()));
    }
    let mut events = adapter.hook_event(event_name, payload)?;
    for event in &mut events {
        event.attach_raw_record(raw_record.id.clone());
    }
    if hook_projections_are_tombstoned(store, &raw_record, &events).await? {
        // Preserve the projection as tombstone evidence for the atomic ingest
        // batch while skipping every attachment read/write below.
        return Ok((raw_record, events, Vec::new()));
    }
    if let Some(blobs) = blobs {
        let mut attachments = Vec::new();
        let mut rejected = false;
        for event in &events {
            if blobs.rejects_event(event).await? {
                rejected = true;
                break;
            }
        }
        if !rejected {
            for source in adapter.hook_attachment_sources(event_name, payload) {
                attachments.extend(extract_attachments(&source, blobs).await?);
            }
        }
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
    secure_spool_dir(dir)?;
    let mut bytes = serde_json::to_vec(&SpoolRecord {
        agent,
        event: resolve_event_name(event, payload).to_owned(),
        payload: payload.clone(),
        captured_at_ms: Some(crate::model::now_ms()),
    })?;
    bytes.push(b'\n');
    let lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("hooks.lock"))?;
    secure_open_file(&lock)?;
    lock_exclusive_wait(&lock)?;
    let file = dir.join("hooks.jsonl");
    let mut handle = private_open_options()
        .create(true)
        .append(true)
        .open(&file)
        .with_context(|| format!("open spool {}", file.display()))?;
    secure_open_file(&handle)?;
    handle.write_all(&bytes)?;
    handle.sync_data()?;
    lock.unlock()?;
    Ok(())
}

/// Durably stages one payload in its own replay generation and keeps it locked.
///
/// The pending file stays locked through processing. This provides crash safety
/// and makes concurrent forget cleanup fail explicitly instead of racing a
/// post-failure append.
pub fn stage_spool(
    dir: &Path,
    agent: AgentKind,
    event: &str,
    payload: &Value,
) -> Result<PendingSpool> {
    secure_spool_dir(dir)?;
    let mut bytes = serde_json::to_vec(&SpoolRecord {
        agent,
        event: resolve_event_name(event, payload).to_owned(),
        payload: payload.clone(),
        captured_at_ms: Some(crate::model::now_ms()),
    })?;
    bytes.push(b'\n');
    let append_lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("hooks.lock"))?;
    secure_open_file(&append_lock)?;
    lock_exclusive_wait(&append_lock)?;
    recover_pending_spool_writes(dir)?;
    remove_incomplete_forget_files(dir)?;
    let stem = format!(
        "hooks.pending.{}.{}",
        std::process::id(),
        crate::model::now_ms()
    );
    let mut suffix = 0_u32;
    let (path, write_path, mut file) = loop {
        let path = dir.join(format!("{stem}.{suffix}.jsonl"));
        let write_path = dir.join(format!(
            ".hooks.pending-write.{}.{}.tmp",
            std::process::id(),
            suffix
        ));
        if path.try_exists()? {
            suffix += 1;
            continue;
        }
        match private_open_options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&write_path)
        {
            Ok(file) => break (path, write_path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => suffix += 1,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("create write-ahead spool {}", write_path.display()));
            }
        }
    };
    if let Err(error) = (|| -> Result<()> {
        secure_open_file(&file)?;
        lock_exclusive_wait(&file)?;
        file.write_all(&bytes)?;
        file.sync_data()?;
        fs::rename(&write_path, &path)?;
        sync_directory(dir)?;
        Ok(())
    })() {
        let _ = file.unlock();
        drop(file);
        let _ = append_lock.unlock();
        return Err(error);
    }
    append_lock.unlock()?;
    Ok(PendingSpool { file, path })
}

/// Counts hook-spool records selected by a bounded forget filter without rewriting files.
///
/// Missing directories are ignored and are not created. Existing directories
/// are locked against append and replay while their live, pending, and claimed
/// replay generations are inspected.
pub fn preview_spool_forget(
    directories: &[PathBuf],
    filter: &ForgetFilter,
) -> Result<SpoolForgetReport> {
    filter_spool_records(directories, filter, false)
}

/// Removes matching records from live, pending, and claimed hook-spool generations.
///
/// Each changed file is replaced atomically with a synced private file while
/// the directory's append and replay locks are held. Missing directories are
/// ignored and are not created.
pub fn forget_spool_records(
    directories: &[PathBuf],
    filter: &ForgetFilter,
) -> Result<SpoolForgetReport> {
    filter_spool_records(directories, filter, true)
}

fn filter_spool_records(
    directories: &[PathBuf],
    filter: &ForgetFilter,
    apply: bool,
) -> Result<SpoolForgetReport> {
    filter.validate()?;
    let mut report = SpoolForgetReport::default();
    let mut seen = HashSet::new();
    for directory in directories {
        if !seen.insert(directory.clone()) || !directory.try_exists()? {
            continue;
        }
        anyhow::ensure!(
            directory.is_dir(),
            "hook spool path {} is not a directory",
            directory.display()
        );
        filter_spool_directory(directory, filter, apply, &mut report)?;
    }
    Ok(report)
}

fn filter_spool_directory(
    directory: &Path,
    filter: &ForgetFilter,
    apply: bool,
    report: &mut SpoolForgetReport,
) -> Result<()> {
    let replay_lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("replay.lock"))?;
    secure_open_file(&replay_lock)?;
    lock_exclusive_wait(&replay_lock)?;
    let append_lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("hooks.lock"))?;
    secure_open_file(&append_lock)?;
    lock_exclusive_wait(&append_lock)?;

    report.directories += 1;
    if apply {
        recover_pending_spool_writes(directory)?;
        remove_incomplete_forget_files(directory)?;
    }
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| is_spool_payload_file(path))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        filter_spool_file(&path, filter, apply, report)?;
    }
    append_lock.unlock()?;
    replay_lock.unlock()?;
    Ok(())
}

fn is_spool_payload_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name == "hooks.jsonl"
                || (name.starts_with("hooks.pending.") && name.ends_with(".jsonl"))
                || (name.starts_with("hooks.replay.") && name.ends_with(".jsonl"))
                || (name.starts_with(".hooks.pending-write.") && name.ends_with(".tmp"))
        })
}

fn filter_spool_file(
    path: &Path,
    filter: &ForgetFilter,
    apply: bool,
    report: &mut SpoolForgetReport,
) -> Result<()> {
    let mut file = match private_open_options().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("open spool {}", path.display())),
    };
    match file.try_lock_exclusive() {
        Ok(()) => {}
        Err(error) if lock_is_contended(&error) => {
            anyhow::bail!(
                "hook spool {} is still owned by an active hook; retry forget",
                path.display()
            )
        }
        Err(error) => {
            return Err(error).with_context(|| format!("lock spool {}", path.display()));
        }
    }
    if !path.try_exists()? {
        file.unlock()?;
        return Ok(());
    }
    let captured_fallback = file
        .metadata()?
        .modified()
        .unwrap_or(UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    file.rewind()?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .with_context(|| format!("read spool {}", path.display()))?;

    let mut retained = Vec::with_capacity(text.len());
    let mut matched = 0;
    let mut retained_records = 0;
    for chunk in text.split_inclusive('\n') {
        let line = chunk.trim();
        if line.is_empty() {
            retained.extend_from_slice(chunk.as_bytes());
            continue;
        }
        let record: SpoolRecord = serde_json::from_str(line)
            .with_context(|| format!("parse spool record in {}", path.display()))?;
        let captured_at_ms = record.captured_at_ms.unwrap_or(captured_fallback);
        if spool_record_matches(&record, filter, captured_at_ms) {
            matched += 1;
        } else {
            retained_records += 1;
            retained.extend_from_slice(chunk.as_bytes());
        }
    }
    report.files += 1;
    report.matched += matched;
    report.retained += retained_records;
    file.unlock()?;
    drop(file);

    if apply && matched > 0 {
        replace_spool_file(path, &retained)?;
    }
    Ok(())
}

fn spool_record_matches(record: &SpoolRecord, filter: &ForgetFilter, captured_at_ms: i64) -> bool {
    if filter.agent.is_some_and(|agent| agent != record.agent) {
        return false;
    }
    if filter.session_id.is_none() && filter.repository.is_none() && filter.before_ms.is_none() {
        return true;
    }

    let adapter = crate::adapters::hook_projector(record.agent);
    let event_name = resolve_event_name(&record.event, &record.payload);
    if let Ok(events) = adapter.hook_event(event_name, &record.payload)
        && events.iter().any(|event| {
            selectors_match(
                filter,
                Some(&event.source.session_id),
                event.source.repository.as_deref(),
                event.occurred_at_ms.unwrap_or(captured_at_ms),
            )
        })
    {
        return true;
    }

    // Projection and native payload metadata are independent coherent
    // ownership views. A successful projection must not mask a native
    // repository/session that the adapter could not carry into the event.
    let native = native_raw_context(record.agent, &record.payload);
    selectors_match(
        filter,
        native.session_id.as_deref(),
        native.repository.as_deref(),
        native.occurred_at_ms.unwrap_or(captured_at_ms),
    )
}

fn selectors_match(
    filter: &ForgetFilter,
    session_id: Option<&str>,
    repository: Option<&str>,
    timestamp_ms: i64,
) -> bool {
    filter
        .session_id
        .as_deref()
        .is_none_or(|expected| session_id == Some(expected))
        && filter
            .repository
            .as_deref()
            .is_none_or(|expected| repository == Some(expected))
        && filter.before_ms.is_none_or(|before| timestamp_ms < before)
}

fn replace_spool_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("spool file has no parent directory")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".hooks.forget-write.")
        .suffix(".tmp")
        .tempfile_in(parent)
        .with_context(|| format!("create filtered spool beside {}", path.display()))?;
    secure_open_file(temporary.as_file())?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("replace filtered spool {}", path.display()))?;
    sync_directory(parent)?;
    if bytes.is_empty() {
        match fs::remove_file(path) {
            Ok(()) => sync_directory(parent)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove empty spool {}", path.display()));
            }
        }
    }
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
    secure_spool_dir(dir).with_context(|| format!("prepare spool directory {}", dir.display()))?;
    let replay_lock_path = dir.join("replay.lock");
    let replay_lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&replay_lock_path)
        .with_context(|| format!("open replay lock {}", replay_lock_path.display()))?;
    secure_open_file(&replay_lock)?;
    lock_exclusive_wait(&replay_lock)
        .with_context(|| format!("acquire replay lock {}", replay_lock_path.display()))?;
    let append_lock_path = dir.join("hooks.lock");
    let append_lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&append_lock_path)
        .with_context(|| format!("open append lock {}", append_lock_path.display()))?;
    secure_open_file(&append_lock)?;
    lock_exclusive_wait(&append_lock)
        .with_context(|| format!("acquire append lock {}", append_lock_path.display()))?;
    recover_pending_spool_writes(dir).context("recover interrupted spool writes")?;
    remove_incomplete_forget_files(dir).context("remove interrupted spool forget files")?;
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
        fs::rename(&live, &claimed).with_context(|| {
            format!(
                "claim live spool generation {} as {}",
                live.display(),
                claimed.display()
            )
        })?;
    }
    let pending = fs::read_dir(dir)
        .with_context(|| format!("list pending spool files in {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("hooks.pending.") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    for path in pending {
        let file = private_open_options()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open pending spool {}", path.display()))?;
        match file.try_lock_exclusive() {
            Ok(()) => {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("non-Unicode pending spool filename")?;
                let claimed = dir.join(name.replacen("hooks.pending.", "hooks.replay.", 1));
                file.unlock()
                    .with_context(|| format!("unlock pending spool {}", path.display()))?;
                drop(file);
                fs::rename(&path, &claimed).with_context(|| {
                    format!(
                        "claim pending spool {} as {}",
                        path.display(),
                        claimed.display()
                    )
                })?;
            }
            Err(error) if lock_is_contended(&error) => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("lock pending spool {}", path.display()));
            }
        }
    }
    append_lock
        .unlock()
        .with_context(|| format!("release append lock {}", append_lock_path.display()))?;
    let lifecycle_guard = store.lock_lifecycle()?;

    let mut generations = fs::read_dir(dir)
        .with_context(|| format!("list replay generations in {}", dir.display()))?
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
    let mut attachment_objects_queued = 0;
    let mut unmanaged_attachment_objects = 0;
    for generation in generations {
        secure_path(&generation)?;
        let generation_captured_at_ms = generation
            .metadata()?
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let generation_lock = private_open_options()
            .read(true)
            .write(true)
            .open(&generation)
            .with_context(|| format!("open replay generation {}", generation.display()))?;
        match generation_lock.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if lock_is_contended(&error) => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("lock replay generation {}", generation.display()));
            }
        }
        let text = fs::read_to_string(&generation)
            .with_context(|| format!("read replay generation {}", generation.display()))?;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let record: SpoolRecord = serde_json::from_str(line)?;
            let adapter = crate::adapters::hook_projector(record.agent);
            // Resolve again for spool files written by older versions that stored
            // the literal "auto" value.
            let event_name = resolve_event_name(&record.event, &record.payload);
            let captured_at_ms = record.captured_at_ms.or(Some(generation_captured_at_ms));
            let (raw_record, events, diagnostics) = normalize_hook(
                &*adapter,
                store,
                event_name,
                &record.payload,
                blobs,
                format!("spool:{}:{event_name}", record.agent),
                captured_at_ms,
            )
            .await?;
            let ingest_stats = store
                .ingest_batch(IngestBatch {
                    raw_records: std::slice::from_ref(&raw_record),
                    raw_contexts: std::slice::from_ref(&raw_context(
                        &raw_record,
                        &events,
                        &record.payload,
                        captured_at_ms,
                    )),
                    events: &events,
                    diagnostics: &diagnostics,
                })
                .await?;
            if let Some(error) = &ingest_stats.sqlite_cleanup_error {
                eprintln!(
                    "semafuse removed replayed forgotten data, but SQLite physical cleanup remains pending: {error}"
                );
            }
            attachment_objects_queued += ingest_stats.attachment_objects_queued;
            unmanaged_attachment_objects += ingest_stats.unmanaged_attachment_objects;
            count += 1;
        }
        generation_lock
            .unlock()
            .with_context(|| format!("unlock replay generation {}", generation.display()))?;
        drop(generation_lock);
        fs::remove_file(&generation)
            .with_context(|| format!("remove replay generation {}", generation.display()))?;
    }
    drop(lifecycle_guard);
    cleanup_replayed_attachments(
        store,
        blobs,
        attachment_objects_queued,
        unmanaged_attachment_objects,
    )
    .await;
    if let Err(error) = strategy::dispatch_pending(store, config, None).await {
        eprintln!("semafuse strategy delivery pending: {error:#}");
    }
    replay_lock
        .unlock()
        .with_context(|| format!("release replay lock {}", replay_lock_path.display()))?;
    Ok(count)
}

fn private_open_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn recover_pending_spool_writes(dir: &Path) -> Result<()> {
    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with(".hooks.pending-write.") && name.ends_with(".tmp")
                })
        })
        .collect::<Vec<_>>();
    paths.sort();

    let mut changed = false;
    for path in paths {
        let mut file = private_open_options()
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open interrupted write-ahead spool {}", path.display()))?;
        match file.try_lock_exclusive() {
            Ok(()) => {}
            Err(error) if lock_is_contended(&error) => {
                anyhow::bail!(
                    "write-ahead spool {} is still owned by an active hook",
                    path.display()
                )
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("lock write-ahead spool {}", path.display()));
            }
        }
        let mut text = String::new();
        file.read_to_string(&mut text)
            .with_context(|| format!("read interrupted write-ahead spool {}", path.display()))?;
        if text.trim().is_empty() {
            file.unlock()?;
            drop(file);
            fs::remove_file(&path)?;
            changed = true;
            continue;
        }
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            serde_json::from_str::<SpoolRecord>(line).with_context(|| {
                format!(
                    "interrupted write-ahead spool {} is incomplete; preserved for recovery",
                    path.display()
                )
            })?;
        }

        let stem = format!("hooks.pending.recovered.{}", crate::model::now_ms());
        let mut suffix = 0_u32;
        let recovered = loop {
            let candidate = dir.join(format!("{stem}.{suffix}.jsonl"));
            if !candidate.try_exists()? {
                break candidate;
            }
            suffix += 1;
        };
        file.unlock()?;
        drop(file);
        fs::rename(&path, recovered)
            .with_context(|| format!("recover write-ahead spool {}", path.display()))?;
        changed = true;
    }
    if changed {
        sync_directory(dir)?;
    }
    Ok(())
}

fn remove_incomplete_forget_files(dir: &Path) -> Result<()> {
    let mut removed = false;
    for path in fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with(".hooks.forget-write.") && name.ends_with(".tmp")
                })
        })
    {
        fs::remove_file(path)?;
        removed = true;
    }
    if removed {
        sync_directory(dir)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    std::fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn lock_is_contended(error: &io::Error) -> bool {
    let expected = fs2::lock_contended_error();
    error.kind() == io::ErrorKind::WouldBlock
        || (error.raw_os_error().is_some() && error.raw_os_error() == expected.raw_os_error())
}

/// Preserves blocking lock semantics on platforms where `fs2` can surface a
/// contention error from its nominally blocking operation.
fn lock_exclusive_wait(file: &std::fs::File) -> io::Result<()> {
    loop {
        match file.lock_exclusive() {
            Ok(()) => return Ok(()),
            Err(error) if lock_is_contended(&error) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(unix)]
fn secure_spool_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    if path.try_exists()? {
        return Ok(());
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    match builder.create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(not(unix))]
fn secure_spool_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(unix)]
fn secure_open_file(file: &std::fs::File) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn secure_open_file(_file: &std::fs::File) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn secure_path(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn secure_path(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ForgetFilter;

    #[test]
    fn spool_forget_ignores_missing_directories_without_creating_them() {
        let temp = tempfile::TempDir::new().unwrap();
        let missing = temp.path().join("missing-spool");
        let report = forget_spool_records(
            std::slice::from_ref(&missing),
            &ForgetFilter {
                session_id: Some("private-session".into()),
                ..ForgetFilter::default()
            },
        )
        .unwrap();
        assert_eq!(report, SpoolForgetReport::default());
        assert!(!missing.exists());
    }

    #[test]
    fn spool_forget_filters_live_pending_and_replay_generations_with_and_semantics() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        let repository = temp.path().join("repository");
        fs::create_dir_all(repository.join(".git")).unwrap();
        let target = serde_json::json!({
            "session_id":"private-session",
            "cwd":repository,
            "timestamp":1000,
            "last_assistant_message":"remove me"
        });
        append_spool(&directory, AgentKind::Codex, "Stop", &target).unwrap();
        fs::rename(
            directory.join("hooks.jsonl"),
            directory.join("hooks.replay.test.jsonl"),
        )
        .unwrap();
        let pending = stage_spool(&directory, AgentKind::Codex, "Stop", &target).unwrap();
        drop(pending);
        append_spool(
            &directory,
            AgentKind::Codex,
            "Stop",
            &serde_json::json!({
                "session_id":"other-session",
                "cwd":repository,
                "timestamp":1000,
                "last_assistant_message":"retain me"
            }),
        )
        .unwrap();
        append_spool(&directory, AgentKind::Codex, "Stop", &target).unwrap();

        let filter = ForgetFilter {
            session_id: Some("private-session".into()),
            repository: Some(repository.canonicalize().unwrap().display().to_string()),
            agent: Some(AgentKind::Codex),
            before_ms: Some(2_000_000),
        };
        let preview = preview_spool_forget(std::slice::from_ref(&directory), &filter).unwrap();
        assert_eq!(preview.directories, 1);
        assert_eq!(preview.files, 3);
        assert_eq!(preview.matched, 3);
        assert_eq!(preview.retained, 1);

        let deleted = forget_spool_records(std::slice::from_ref(&directory), &filter).unwrap();
        assert_eq!(deleted.matched, 3);
        assert_eq!(deleted.retained, 1);
        let remaining = fs::read_dir(&directory)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| is_spool_payload_file(&entry.path()))
            .map(|entry| fs::read_to_string(entry.path()).unwrap())
            .collect::<String>();
        assert!(!remaining.contains("remove me"));
        assert!(remaining.contains("retain me"));
    }

    #[test]
    fn spool_forget_removes_crash_leftovers_from_atomic_rewrites() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        fs::create_dir_all(&directory).unwrap();
        let orphan = directory.join(".hooks.forget-write.crashed.tmp");
        fs::write(
            &orphan,
            br#"{"agent":"codex","event":"Stop","payload":{"session_id":"private-session","secret":"plaintext"}}
"#,
        )
        .unwrap();

        forget_spool_records(
            std::slice::from_ref(&directory),
            &ForgetFilter {
                session_id: Some("private-session".into()),
                ..ForgetFilter::default()
            },
        )
        .unwrap();

        assert!(!orphan.exists());
    }

    #[test]
    fn interrupted_write_ahead_spool_is_recovered_without_selector_drift() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        fs::create_dir_all(&directory).unwrap();
        let orphan = directory.join(".hooks.pending-write.crashed.0.tmp");
        let mut bytes = serde_json::to_vec(&SpoolRecord {
            agent: AgentKind::Codex,
            event: "Stop".into(),
            payload: serde_json::json!({
                "session_id":"retained-session",
                "last_assistant_message":"durable secret"
            }),
            captured_at_ms: Some(1_000),
        })
        .unwrap();
        bytes.push(b'\n');
        let mut file = private_open_options()
            .create_new(true)
            .write(true)
            .open(&orphan)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_data().unwrap();
        drop(file);

        let unrelated = ForgetFilter {
            session_id: Some("different-session".into()),
            ..ForgetFilter::default()
        };
        let preview = preview_spool_forget(std::slice::from_ref(&directory), &unrelated).unwrap();
        assert_eq!(preview.files, 1);
        assert_eq!(preview.matched, 0);
        assert_eq!(preview.retained, 1);
        let applied = forget_spool_records(std::slice::from_ref(&directory), &unrelated).unwrap();
        assert_eq!(applied, preview);
        assert!(!orphan.exists());
        let recovered = fs::read_dir(&directory)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("hooks.pending.recovered.") && name.ends_with(".jsonl")
                    })
            })
            .unwrap();
        assert!(
            fs::read_to_string(&recovered)
                .unwrap()
                .contains("durable secret")
        );

        let removed = forget_spool_records(
            std::slice::from_ref(&directory),
            &ForgetFilter {
                session_id: Some("retained-session".into()),
                ..ForgetFilter::default()
            },
        )
        .unwrap();
        assert_eq!(removed.matched, 1);
        assert_eq!(removed.retained, 0);
        assert!(!recovered.exists());
    }

    #[test]
    fn incomplete_write_ahead_spool_is_preserved_and_blocks_false_cleanup() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        fs::create_dir_all(&directory).unwrap();
        let orphan = directory.join(".hooks.pending-write.crashed.0.tmp");
        fs::write(&orphan, br#"{"agent":"codex","secret":"plaintext""#).unwrap();

        let error = forget_spool_records(
            std::slice::from_ref(&directory),
            &ForgetFilter {
                session_id: Some("private-session".into()),
                ..ForgetFilter::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("preserved for recovery"));
        assert!(orphan.exists());
        assert!(fs::read_to_string(orphan).unwrap().contains("plaintext"));
    }

    #[test]
    fn active_write_ahead_hook_prevents_forget_from_claiming_cleanup_complete() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        let pending = stage_spool(
            &directory,
            AgentKind::OpenCode,
            "session.created",
            &serde_json::json!({
                "event":{"properties":{"info":{"id":"private-session"}}},
                "secret":"plaintext"
            }),
        )
        .unwrap();

        let error = forget_spool_records(
            std::slice::from_ref(&directory),
            &ForgetFilter {
                session_id: Some("private-session".into()),
                ..ForgetFilter::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("still owned by an active hook"));

        drop(pending);
        assert_eq!(
            forget_spool_records(
                std::slice::from_ref(&directory),
                &ForgetFilter {
                    session_id: Some("private-session".into()),
                    ..ForgetFilter::default()
                },
            )
            .unwrap()
            .matched,
            1
        );
    }

    #[test]
    fn spool_forget_checks_native_ownership_even_when_projection_succeeds() {
        let temp = tempfile::TempDir::new().unwrap();
        let directory = temp.path().join("spool");
        let native_repository = temp.path().join("not-a-git-repository");
        append_spool(
            &directory,
            AgentKind::OpenCode,
            "session.created",
            &serde_json::json!({
                "directory":native_repository,
                "event":{"properties":{"info":{"id":"allowed-session"}}},
                "secret":"plaintext"
            }),
        )
        .unwrap();

        let report = forget_spool_records(
            std::slice::from_ref(&directory),
            &ForgetFilter {
                repository: Some(native_repository.display().to_string()),
                ..ForgetFilter::default()
            },
        )
        .unwrap();

        assert_eq!(report.matched, 1);
        assert_eq!(report.retained, 0);
        let retained = fs::read_to_string(directory.join("hooks.jsonl")).unwrap_or_default();
        assert!(!retained.contains("plaintext"));
    }
}
