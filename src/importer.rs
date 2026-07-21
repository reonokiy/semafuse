//! Reporting and finalization helpers for adapter-owned history importers.

use crate::{
    adapters::{ImportContext, ProjectedEvent},
    blob::{BlobStore, extract_attachments},
    model::{
        AgentKind, ArchiveSpecVersion, CanonicalArchive, CanonicalEvent, ConversionDiagnostic,
        DiagnosticSeverity, RawRecord,
    },
};
use anyhow::{Context, Result};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

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
    /// Structured per-record normalization and validation diagnostics.
    pub diagnostics: Vec<ConversionDiagnostic>,
    /// Source-qualified parse, mapping, network, or attachment errors.
    pub errors: Vec<String>,
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
        let context = ImportContext {
            source: native.source.clone(),
            fallback_session_id: native.fallback_session_id,
            cwd: native.cwd,
            raw_record_ids: raw_records.iter().map(|record| record.id.clone()).collect(),
        };
        self.files += 1;
        self.records += native.values.len();
        self.raw_records.extend(raw_records);
        match project(&native.values, &context) {
            Ok(projected) => {
                let (events, diagnostics) = resolve_projected_events(projected, blobs).await;
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

async fn resolve_projected_events(
    projected: Vec<ProjectedEvent>,
    blobs: Option<&dyn BlobStore>,
) -> (Vec<CanonicalEvent>, Vec<ConversionDiagnostic>) {
    let mut events = Vec::with_capacity(projected.len());
    let mut diagnostics = Vec::new();
    for projected in projected {
        let mut event = projected.event;
        if let Some(blobs) = blobs {
            match extract_attachments(&projected.attachment_source, blobs).await {
                Ok(attachments) if !attachments.is_empty() => event.add_attachments(attachments),
                Ok(_) => {}
                Err(error) => diagnostics.push(ConversionDiagnostic::new(
                    event
                        .raw_record_ids
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "raw_missing".into()),
                    DiagnosticSeverity::Warning,
                    "attachment_extraction_failed",
                    error.to_string(),
                )),
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
