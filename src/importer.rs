//! Reporting and finalization helpers for adapter-owned history importers.

use crate::{
    adapters::{ImportContext, ProjectedEvent, native_raw_context},
    blob::{BlobStore, extract_attachments},
    model::{
        AgentKind, ArchiveSpecVersion, CanonicalArchive, CanonicalEvent, ConversionDiagnostic,
        DiagnosticSeverity, RawRecord, normalize_repository,
    },
    storage::{RawContext, SqliteStore},
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

const DEFERRED_IMPORT_BACKEND_ID: &str = "semafuse-internal:deferred-import-attachments";

/// Marker blob store that asks importers to defer attachment work until
/// [`ImportReport::finalize_events`] can apply database tombstones.
pub struct DeferredImportBlobStore;

#[async_trait]
impl BlobStore for DeferredImportBlobStore {
    async fn put(
        &self,
        _bytes: Vec<u8>,
        _media_type: &str,
        _name: Option<String>,
    ) -> Result<crate::model::AttachmentRef> {
        anyhow::bail!("deferred import blob store cannot persist attachments")
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        anyhow::bail!("deferred import blob {key} was not persisted")
    }

    async fn delete(&self, key: &str) -> Result<()> {
        anyhow::bail!("deferred import blob {key} was not persisted")
    }

    fn backend_id(&self) -> &str {
        DEFERRED_IMPORT_BACKEND_ID
    }

    fn manages_key(&self, _key: &str) -> bool {
        false
    }
}

/// Aggregate result of importing one path or agent-native history service.
#[derive(Debug, Default)]
pub struct ImportReport {
    /// Number of successfully read files or remote sessions.
    pub files: usize,
    /// Number of native top-level records passed to an adapter.
    pub records: usize,
    /// Successfully normalized events, not yet inserted into an event store.
    pub events: Vec<CanonicalEvent>,
    /// Immutable native records from which canonical events were projected.
    pub raw_records: Vec<RawRecord>,
    /// Deletion-relevant ownership for every native record, including failed projections.
    pub raw_contexts: Vec<RawContext>,
    /// Structured per-record normalization and validation diagnostics.
    pub diagnostics: Vec<ConversionDiagnostic>,
    /// Source-qualified parse, mapping, network, or attachment errors.
    pub errors: Vec<String>,
    pending_attachments: Vec<PendingAttachment>,
    tombstone_inputs: Vec<CanonicalEvent>,
}

#[derive(Debug)]
struct PendingAttachment {
    event_index: usize,
    source: Value,
}

/// One adapter-owned native source ready for raw retention and projection.
pub struct NativeSource {
    /// Agent that owns the native source schema.
    pub agent: AgentKind,
    /// Human-readable source identifier, normally a path or agent protocol ID.
    pub source: String,
    /// Session ID to use when the native source lacks one.
    pub fallback_session_id: String,
    /// Working directory supplied by source-level context, when available.
    pub cwd: Option<String>,
    /// Top-level native JSON values read from this source.
    pub values: Vec<Value>,
}

impl ImportReport {
    /// Returns the formal, language-neutral conversion result without operational counters.
    pub fn canonical_archive(&self) -> CanonicalArchive {
        CanonicalArchive {
            spec_version: ArchiveSpecVersion::V1,
            raw_records: self.raw_records.clone(),
            events: self.events.clone(),
            diagnostics: self.diagnostics.clone(),
        }
    }

    /// Returns accepted projections plus attachment-free tombstone evidence
    /// required by the storage layer's atomic provenance closure.
    pub fn events_for_storage(&self) -> Vec<CanonicalEvent> {
        self.events
            .iter()
            .chain(&self.tombstone_inputs)
            .cloned()
            .collect()
    }

    /// Applies durable raw tombstones before resolving deferred attachments.
    ///
    /// Native importers may project several canonical events from one raw
    /// record. If any deletion context rejects that raw record, every linked
    /// projection is removed before attachment decoding or backend writes.
    pub async fn finalize_events(
        &mut self,
        store: Option<&SqliteStore>,
        blobs: &dyn BlobStore,
    ) -> Result<()> {
        self.tombstone_inputs.clear();
        let mut rejected_raw_ids = HashSet::new();
        let mut rejected_raw_contexts = HashMap::new();
        let mut rejected_event_indices = HashSet::new();
        if let Some(store) = store {
            let mut projected_contexts = HashMap::<String, Vec<RawContext>>::new();
            for event in &self.events {
                for raw_record_id in &event.raw_record_ids {
                    projected_contexts
                        .entry(raw_record_id.clone())
                        .or_default()
                        .push(RawContext {
                            raw_record_id: raw_record_id.clone(),
                            session_id: Some(event.source.session_id.clone()),
                            repository: event.source.repository.clone(),
                            occurred_at_ms: event.occurred_at_ms,
                        });
                }
            }
            for raw in &self.raw_records {
                let native = native_raw_context(raw.agent, &raw.payload);
                let native_context = RawContext {
                    raw_record_id: raw.id.clone(),
                    session_id: native.session_id,
                    repository: native.repository,
                    occurred_at_ms: native.occurred_at_ms,
                };
                let base_context = self
                    .raw_contexts
                    .iter()
                    .find(|context| context.raw_record_id == raw.id)
                    .cloned()
                    .unwrap_or_else(|| native_context.clone());
                if store.raw_is_tombstoned(raw, &base_context).await? {
                    rejected_raw_ids.insert(raw.id.clone());
                    rejected_raw_contexts.insert(raw.id.clone(), base_context);
                    continue;
                }
                // Projection can replace source-level context in the report.
                // Always test native ownership independently before any
                // attachment is decoded or written.
                if store.raw_is_tombstoned(raw, &native_context).await? {
                    rejected_raw_ids.insert(raw.id.clone());
                    rejected_raw_contexts.insert(raw.id.clone(), native_context);
                    continue;
                }
                for context in projected_contexts.get(&raw.id).into_iter().flatten() {
                    if store.raw_is_tombstoned(raw, context).await? {
                        rejected_raw_ids.insert(raw.id.clone());
                        rejected_raw_contexts.insert(raw.id.clone(), context.clone());
                        break;
                    }
                }
            }
            for (event_index, event) in self.events.iter().enumerate() {
                if store.event_is_tombstoned(event).await? {
                    rejected_event_indices.insert(event_index);
                }
            }
            // Forget applies to connected provenance components. Resolve the
            // full incoming graph before attachment I/O so a chained sibling
            // projection cannot leave an untracked staged blob behind.
            loop {
                let raw_count = rejected_raw_ids.len();
                let event_count = rejected_event_indices.len();
                for (event_index, event) in self.events.iter().enumerate() {
                    if rejected_event_indices.contains(&event_index)
                        || event
                            .raw_record_ids
                            .iter()
                            .any(|raw_id| rejected_raw_ids.contains(raw_id))
                    {
                        rejected_event_indices.insert(event_index);
                        rejected_raw_ids.extend(event.raw_record_ids.iter().cloned());
                    }
                }
                if raw_count == rejected_raw_ids.len()
                    && event_count == rejected_event_indices.len()
                {
                    break;
                }
            }
        }
        for (raw_id, rejected_context) in rejected_raw_contexts {
            if let Some(context) = self
                .raw_contexts
                .iter_mut()
                .find(|context| context.raw_record_id == raw_id)
            {
                *context = rejected_context;
            } else {
                self.raw_contexts.push(rejected_context);
            }
        }

        let pending = std::mem::take(&mut self.pending_attachments)
            .into_iter()
            .map(|pending| (pending.event_index, pending.source))
            .collect::<HashMap<_, _>>();
        let mut events = Vec::with_capacity(self.events.len());
        for (event_index, mut event) in std::mem::take(&mut self.events).into_iter().enumerate() {
            if rejected_event_indices.contains(&event_index) {
                // Keep the projection as a tombstone input for the atomic
                // storage batch, but never resolve its attachments.
                self.tombstone_inputs.push(event);
                continue;
            }
            match blobs.rejects_event(&event).await {
                Ok(true) => {
                    self.tombstone_inputs.push(event);
                    continue;
                }
                Ok(false) => {}
                Err(error) => {
                    self.diagnostics.push(ConversionDiagnostic::new(
                        event
                            .raw_record_ids
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "raw_missing".into()),
                        DiagnosticSeverity::Error,
                        "forget_rule_check_failed",
                        error.to_string(),
                    ));
                    continue;
                }
            }
            if let Some(source) = pending.get(&event_index) {
                match extract_attachments(source, blobs).await {
                    Ok(attachments) if !attachments.is_empty() => {
                        event.add_attachments(attachments)
                    }
                    Ok(_) => {}
                    Err(error) => {
                        self.diagnostics.push(ConversionDiagnostic::new(
                            event
                                .raw_record_ids
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "raw_missing".into()),
                            DiagnosticSeverity::Error,
                            "attachment_extraction_failed",
                            error.to_string(),
                        ));
                        continue;
                    }
                }
            }
            if let Err(error) = event.validate() {
                self.diagnostics.push(ConversionDiagnostic::new(
                    event
                        .raw_record_ids
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "raw_missing".into()),
                    DiagnosticSeverity::Error,
                    "invalid_canonical_event",
                    error.to_string(),
                ));
                continue;
            }
            events.push(event);
        }
        self.events = events;
        Ok(())
    }

    /// Adds one successfully read native source to the report.
    pub async fn add_native_source(
        &mut self,
        native: NativeSource,
        blobs: Option<&dyn BlobStore>,
        project: impl FnOnce(&[Value], &ImportContext) -> Result<Vec<ProjectedEvent>>,
    ) {
        let raw_records = native
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                RawRecord::new(
                    native.agent,
                    native.source.clone(),
                    index as u64,
                    value.clone(),
                )
            })
            .collect::<Vec<_>>();
        let fallback_repository = normalize_repository(native.cwd.as_deref(), None);
        let mut codex_session = None;
        let mut codex_repository = None;
        let raw_contexts = raw_records
            .iter()
            .zip(&native.values)
            .map(|(raw, value)| {
                let native_context = native_raw_context(native.agent, value);
                if native.agent == AgentKind::Codex {
                    if let Some(session_id) = &native_context.session_id {
                        codex_session = Some(session_id.clone());
                    }
                    if let Some(repository) = &native_context.repository {
                        codex_repository = Some(repository.clone());
                    }
                }
                RawContext {
                    raw_record_id: raw.id.clone(),
                    session_id: native_context
                        .session_id
                        .or_else(|| codex_session.clone())
                        .or_else(|| Some(native.fallback_session_id.clone())),
                    repository: native_context
                        .repository
                        .or_else(|| codex_repository.clone())
                        .or_else(|| fallback_repository.clone()),
                    occurred_at_ms: native_context.occurred_at_ms,
                }
            })
            .collect::<Vec<_>>();
        let context = ImportContext {
            source: native.source.clone(),
            fallback_session_id: native.fallback_session_id,
            cwd: native.cwd,
            raw_record_ids: raw_records.iter().map(|record| record.id.clone()).collect(),
        };
        self.files += 1;
        self.records += native.values.len();
        self.raw_records.extend(raw_records);
        let context_offset = self.raw_contexts.len();
        self.raw_contexts.extend(raw_contexts);
        match project(&native.values, &context) {
            Ok(projected) => {
                let (events, diagnostics, pending_sources) = match blobs {
                    Some(blobs) if blobs.backend_id() == DEFERRED_IMPORT_BACKEND_ID => {
                        stage_projected_events(projected)
                    }
                    Some(blobs) => {
                        let (events, diagnostics) =
                            resolve_projected_events(projected, Some(blobs)).await;
                        (events, diagnostics, Vec::new())
                    }
                    None => {
                        let (events, diagnostics) = resolve_projected_events(projected, None).await;
                        (events, diagnostics, Vec::new())
                    }
                };
                let source_sessions = events
                    .iter()
                    .map(|event| event.source.session_id.clone())
                    .collect::<HashSet<_>>();
                let source_session = (source_sessions.len() == 1)
                    .then(|| source_sessions.into_iter().next())
                    .flatten();
                let source_repositories = events
                    .iter()
                    .filter_map(|event| event.source.repository.clone())
                    .collect::<HashSet<_>>();
                let source_repository = (source_repositories.len() == 1)
                    .then(|| source_repositories.into_iter().next())
                    .flatten();
                for raw_context in &mut self.raw_contexts[context_offset..] {
                    if let Some(session) = &source_session {
                        raw_context.session_id = Some(session.clone());
                    }
                    if let Some(repository) = &source_repository {
                        raw_context.repository = Some(repository.clone());
                    }
                }
                for event in &events {
                    for raw_record_id in &event.raw_record_ids {
                        if let Some(raw_context) = self.raw_contexts[context_offset..]
                            .iter_mut()
                            .find(|context| context.raw_record_id == *raw_record_id)
                        {
                            raw_context.session_id = Some(event.source.session_id.clone());
                            raw_context.repository = event.source.repository.clone();
                            raw_context.occurred_at_ms =
                                event.occurred_at_ms.or(raw_context.occurred_at_ms);
                        }
                    }
                }
                let event_offset = self.events.len();
                self.pending_attachments
                    .extend(
                        pending_sources
                            .into_iter()
                            .enumerate()
                            .map(|(index, source)| PendingAttachment {
                                event_index: event_offset + index,
                                source,
                            }),
                    );
                self.events.extend(events);
                self.diagnostics.extend(diagnostics);
            }
            Err(error) => {
                self.diagnostics
                    .extend(context.raw_record_ids.iter().map(|raw_id| {
                        ConversionDiagnostic::new(
                            raw_id,
                            DiagnosticSeverity::Error,
                            "adapter_conversion_failed",
                            error.to_string(),
                        )
                    }));
                self.errors.push(format!("{}: {error:#}", context.source));
            }
        }
    }
}

fn stage_projected_events(
    projected: Vec<ProjectedEvent>,
) -> (Vec<CanonicalEvent>, Vec<ConversionDiagnostic>, Vec<Value>) {
    let (events, sources) = projected
        .into_iter()
        .map(|projected| (projected.event, projected.attachment_source))
        .unzip();
    (events, Vec::new(), sources)
}

async fn resolve_projected_events(
    projected: Vec<ProjectedEvent>,
    blobs: Option<&dyn BlobStore>,
) -> (Vec<CanonicalEvent>, Vec<ConversionDiagnostic>) {
    let mut events = Vec::with_capacity(projected.len());
    let mut diagnostics = Vec::new();
    for projected in projected {
        let mut event = projected.event;
        if let Some(blobs) = blobs {
            match blobs.rejects_event(&event).await {
                Ok(true) => {
                    events.push(event);
                    continue;
                }
                Ok(false) => {}
                Err(error) => {
                    diagnostics.push(ConversionDiagnostic::new(
                        event
                            .raw_record_ids
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "raw_missing".into()),
                        DiagnosticSeverity::Error,
                        "forget_rule_check_failed",
                        error.to_string(),
                    ));
                    continue;
                }
            }
            match extract_attachments(&projected.attachment_source, blobs).await {
                Ok(attachments) if !attachments.is_empty() => event.add_attachments(attachments),
                Ok(_) => {}
                Err(error) => {
                    diagnostics.push(ConversionDiagnostic::new(
                        event
                            .raw_record_ids
                            .first()
                            .cloned()
                            .unwrap_or_else(|| "raw_missing".into()),
                        DiagnosticSeverity::Error,
                        "attachment_extraction_failed",
                        error.to_string(),
                    ));
                    continue;
                }
            }
        }
        if let Err(error) = event.validate() {
            diagnostics.push(ConversionDiagnostic::new(
                event
                    .raw_record_ids
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "raw_missing".into()),
                DiagnosticSeverity::Error,
                "invalid_canonical_event",
                error.to_string(),
            ));
        } else {
            events.push(event);
        }
    }
    (events, diagnostics)
}

/// Lists JSON and JSONL source files without following symbolic links.
pub(crate) fn json_source_files(path: &Path) -> Vec<PathBuf> {
    if path.is_dir() {
        WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_type().is_file()
                    && matches!(
                        e.path().extension().and_then(|s| s.to_str()),
                        Some("json" | "jsonl")
                    )
            })
            .map(|e| e.into_path())
            .collect()
    } else {
        vec![path.to_owned()]
    }
}

/// Reads one JSON or JSONL source into top-level JSON values.
pub(crate) fn read_json_values(path: &Path) -> Result<Vec<Value>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
        text.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(n, line)| {
                serde_json::from_str(line)
                    .with_context(|| format!("parse {} line {}", path.display(), n + 1))
            })
            .collect()
    } else {
        let value: Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        Ok(match value {
            Value::Array(values) => values,
            value => vec![value],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ImportReport, NativeSource, PendingAttachment, resolve_projected_events};
    use crate::{
        adapters::{CodexAdapter, HistoryImporter, ProjectedEvent},
        blob::{BlobStore, DryRunBlobStore},
        model::{AgentKind, AttachmentRef, CanonicalEvent, EventSource, MessageRole, RawRecord},
        storage::{EventStore, ForgetFilter, IngestBatch, RawContext, SqliteStore},
    };
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingBlobStore(AtomicUsize);

    #[async_trait]
    impl BlobStore for CountingBlobStore {
        async fn put(
            &self,
            _bytes: Vec<u8>,
            _media_type: &str,
            _name: Option<String>,
        ) -> Result<AttachmentRef> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(AttachmentRef {
                object_key: "unexpected".into(),
                digest: "0".repeat(64),
                media_type: "application/octet-stream".into(),
                size: 0,
                name: None,
            })
        }

        async fn get(&self, _key: &str) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }

        async fn delete(&self, _key: &str) -> Result<()> {
            Ok(())
        }

        fn backend_id(&self) -> &str {
            "counting"
        }

        fn manages_key(&self, _key: &str) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn attachment_failure_rejects_the_event_instead_of_silently_dropping_binary_data() {
        let raw = RawRecord::new(AgentKind::Codex, "test.json", 0, json!({"type":"message"}));
        let source = EventSource::new(
            AgentKind::Codex,
            "attachment-session",
            None,
            Some("message-1".into()),
            "message",
            None,
            None,
        );
        let mut event =
            CanonicalEvent::message_from_text(source, MessageRole::User, "image follows", Some(1));
        event.attach_raw_record(&raw.id);
        let projected = ProjectedEvent {
            event,
            attachment_source: json!({
                "type":"image",
                "source":{"media_type":"image/png","data":"not-valid-base64!"}
            }),
        };

        let (events, diagnostics) =
            resolve_projected_events(vec![projected], Some(&DryRunBlobStore)).await;
        assert!(events.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code, "attachment_extraction_failed");
        assert_eq!(
            diagnostics[0].severity,
            crate::model::DiagnosticSeverity::Error
        );
    }

    #[tokio::test]
    async fn forgotten_native_context_closes_chained_provenance_before_attachment_io() {
        let store = SqliteStore::open_memory().await.unwrap();
        store
            .forget(&ForgetFilter {
                session_id: Some("native-forgotten".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        let raw_a = RawRecord::new(
            AgentKind::Codex,
            "chain.jsonl",
            0,
            json!({"session_id":"native-forgotten","secret":"A"}),
        );
        let raw_b = RawRecord::new(
            AgentKind::Codex,
            "chain.jsonl",
            1,
            json!({"session_id":"allowed","secret":"B"}),
        );
        let raw_c = RawRecord::new(
            AgentKind::Codex,
            "chain.jsonl",
            2,
            json!({"session_id":"allowed","secret":"C"}),
        );
        let mut event_ab = CanonicalEvent::message_from_text(
            EventSource::new(
                AgentKind::Codex,
                "allowed",
                None,
                Some("event-ab".into()),
                "message",
                None,
                None,
            ),
            MessageRole::User,
            "first projection",
            Some(1),
        );
        event_ab.attach_raw_record(&raw_a.id);
        event_ab.attach_raw_record(&raw_b.id);
        let mut event_bc = CanonicalEvent::message_from_text(
            EventSource::new(
                AgentKind::Codex,
                "allowed",
                None,
                Some("event-bc".into()),
                "message",
                None,
                None,
            ),
            MessageRole::Assistant,
            "second projection",
            Some(2),
        );
        event_bc.attach_raw_record(&raw_b.id);
        event_bc.attach_raw_record(&raw_c.id);
        let mut report = ImportReport {
            raw_records: vec![raw_a.clone(), raw_b.clone(), raw_c.clone()],
            raw_contexts: [raw_a.id.clone(), raw_b.id.clone(), raw_c.id.clone()]
                .into_iter()
                .map(|raw_record_id| RawContext {
                    raw_record_id,
                    session_id: Some("allowed".into()),
                    ..RawContext::default()
                })
                .collect(),
            events: vec![event_ab, event_bc],
            pending_attachments: vec![PendingAttachment {
                event_index: 1,
                source: json!({
                    "type":"image",
                    "source":{
                        "media_type":"application/octet-stream",
                        "data":"Rk9SR09UVEVOX0FUVEFDSE1FTlQ="
                    }
                }),
            }],
            ..ImportReport::default()
        };
        let blobs = CountingBlobStore::default();

        report.finalize_events(Some(&store), &blobs).await.unwrap();
        assert_eq!(blobs.0.load(Ordering::SeqCst), 0);
        assert!(report.events.is_empty());
        assert_eq!(report.tombstone_inputs.len(), 2);
        let storage_events = report.events_for_storage();
        let stats = store
            .ingest_batch(IngestBatch {
                raw_records: &report.raw_records,
                raw_contexts: &report.raw_contexts,
                events: &storage_events,
                diagnostics: &report.diagnostics,
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_tombstoned, 3);
        assert_eq!(stats.tombstoned, 2);
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );
    }

    #[tokio::test]
    async fn failed_codex_rollout_projection_keeps_source_context_for_forget_and_replay() {
        let values = vec![
            json!({
                "type":"session_meta",
                "payload":{
                    "id":"failed-rollout-session",
                    "cwd":"/workspace/private",
                    "secret":"first plaintext"
                }
            }),
            json!({"type":123,"payload":{"secret":"second plaintext"}}),
        ];
        let source = NativeSource {
            agent: AgentKind::Codex,
            source: "failed-rollout.jsonl".into(),
            fallback_session_id: "misleading-file-name".into(),
            cwd: None,
            values: values.clone(),
        };
        let mut report = ImportReport::default();
        report
            .add_native_source(source, None, |values, context| {
                CodexAdapter.import_values(values, context)
            })
            .await;
        assert!(report.events.is_empty());
        assert_eq!(report.raw_records.len(), 2);
        assert_eq!(report.raw_contexts.len(), 2);
        assert!(
            report
                .raw_contexts
                .iter()
                .all(|context| context.session_id.as_deref() == Some("failed-rollout-session"))
        );

        let store = SqliteStore::open_memory().await.unwrap();
        let stats = store
            .ingest_batch(IngestBatch {
                raw_records: &report.raw_records,
                raw_contexts: &report.raw_contexts,
                events: &report.events,
                diagnostics: &report.diagnostics,
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_inserted, 2);
        assert_eq!(
            store
                .forget(&ForgetFilter {
                    session_id: Some("failed-rollout-session".into()),
                    ..ForgetFilter::default()
                })
                .await
                .unwrap()
                .raw_records,
            2
        );
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );

        let mut replay = ImportReport::default();
        replay
            .add_native_source(
                NativeSource {
                    agent: AgentKind::Codex,
                    source: "failed-rollout.jsonl".into(),
                    fallback_session_id: "misleading-file-name".into(),
                    cwd: None,
                    values,
                },
                None,
                |values, context| CodexAdapter.import_values(values, context),
            )
            .await;
        replay
            .finalize_events(Some(&store), &DryRunBlobStore)
            .await
            .unwrap();
        let replay_stats = store
            .ingest_batch(IngestBatch {
                raw_records: &replay.raw_records,
                raw_contexts: &replay.raw_contexts,
                events: &replay.events_for_storage(),
                diagnostics: &replay.diagnostics,
            })
            .await
            .unwrap();
        assert_eq!(replay_stats.raw_tombstoned, 2);
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );
    }
}
