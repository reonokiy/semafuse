//! Discovery and normalization of persisted agent histories.

use crate::{
    adapters::{AgentAdapter, ImportContext, ImportedEvent},
    blob::{BlobStore, extract_attachments},
    model::{
        AgentKind, ArchiveSpecVersion, CanonicalArchive, CanonicalEvent, ConversionDiagnostic,
        DiagnosticSeverity, RawRecord,
    },
};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Stdio},
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::Command as TokioCommand,
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
}

/// Imports from the conventional history source for the adapter's agent.
///
/// Amp, Claude Code, and Codex use their platform data directories (with
/// `CODEX_HOME` honored). Codex first tries its App Server and falls back to
/// rollout JSONL when that service is unavailable. OpenCode is queried through
/// a temporary local server. Source-specific failures that can be isolated are
/// accumulated in [`ImportReport::errors`].
pub async fn import_default(
    adapter: &dyn AgentAdapter,
    blobs: Option<&dyn BlobStore>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<ImportReport> {
    let home = dirs::home_dir().context("home directory not found")?;
    match adapter.kind() {
        AgentKind::Amp => {
            let root = dirs::data_local_dir()
                .context("local data directory not found")?
                .join("amp/threads");
            import_path(adapter, &root, blobs, progress).await
        }
        AgentKind::Codex => {
            let root = std::env::var("CODEX_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| home.join(".codex"));
            match import_codex_app_server(adapter, blobs, progress).await {
                Ok(report) => Ok(report),
                Err(error) => {
                    eprintln!(
                        "warning: Codex App Server import unavailable ({error:#}); falling back to rollout JSONL"
                    );
                    import_path(adapter, &root.join("sessions"), blobs, progress).await
                }
            }
        }
        AgentKind::ClaudeCode => {
            import_path(adapter, &home.join(".claude/projects"), blobs, progress).await
        }
        AgentKind::OpenCode => import_opencode_server(adapter, blobs, progress).await,
    }
}

/// Imports a JSON/JSONL file or recursively imports such files below a directory.
///
/// Symbolic links are not followed. Each source is independent: parse, adapter,
/// and attachment failures are added to the report and processing continues.
/// The progress callback receives one-based progress, total sources, and path.
pub async fn import_path(
    adapter: &dyn AgentAdapter,
    path: &Path,
    blobs: Option<&dyn BlobStore>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<ImportReport> {
    let mut report = ImportReport::default();
    let paths: Vec<_> = if path.is_dir() {
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
    };
    let total = paths.len();
    for (index, file) in paths.into_iter().enumerate() {
        match parse_file(&file) {
            Ok(values) => {
                report.files += 1;
                report.records += values.len();
                let fallback = file
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_owned();
                let source = file.display().to_string();
                let raw_records = values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        RawRecord::new(adapter.kind(), source.clone(), index as u64, value.clone())
                    })
                    .collect::<Vec<_>>();
                let context = ImportContext {
                    source,
                    fallback_session_id: fallback,
                    cwd: None,
                    raw_record_ids: raw_records.iter().map(|record| record.id.clone()).collect(),
                };
                report.raw_records.extend(raw_records);
                match adapter.import_values(&values, &context) {
                    Ok(events) => {
                        let (events, diagnostics) = resolve_imported_events(events, blobs).await;
                        report.events.extend(events);
                        report.diagnostics.extend(diagnostics);
                    }
                    Err(error) => {
                        report
                            .diagnostics
                            .extend(context.raw_record_ids.iter().map(|raw_id| {
                                ConversionDiagnostic::new(
                                    raw_id,
                                    DiagnosticSeverity::Error,
                                    "adapter_conversion_failed",
                                    error.to_string(),
                                )
                            }));
                        report.errors.push(format!("{}: {error:#}", file.display()));
                    }
                }
            }
            Err(error) => report.errors.push(format!("{}: {error:#}", file.display())),
        }
        progress(index + 1, total, &file.display().to_string());
    }
    Ok(report)
}

async fn import_opencode_server(
    adapter: &dyn AgentAdapter,
    blobs: Option<&dyn BlobStore>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<ImportReport> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let child = StdCommand::new("opencode")
        .args([
            "serve",
            "--hostname",
            "127.0.0.1",
            "--port",
            &port.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start opencode server")?;
    let mut child = ChildGuard(child);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let base = format!("http://127.0.0.1:{port}");
    let sessions = async {
        for _ in 0..100 {
            if let Ok(response) = client.get(format!("{base}/session")).send().await
                && response.status().is_success()
            {
                return Ok::<Value, anyhow::Error>(response.json().await?);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        bail!("OpenCode server did not become ready")
    }
    .await;
    let sessions = match sessions {
        Ok(value) => value,
        Err(error) => return Err(error),
    };
    let mut report = ImportReport::default();
    let sessions = sessions.as_array().cloned().unwrap_or_default();
    let total = sessions.len();
    for (index, item) in sessions.iter().enumerate() {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            progress(index + 1, total, "invalid session");
            continue;
        };
        progress(index + 1, total, id);
        let mut request = client.get(format!("{base}/session/{id}/message"));
        if let Some(directory) = item.get("directory").and_then(Value::as_str) {
            request = request.header("x-opencode-directory", directory);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                report
                    .errors
                    .push(format!("opencode messages {id} failed: {error}"));
                continue;
            }
        };
        if !response.status().is_success() {
            report.errors.push(format!(
                "opencode messages {id} failed: {}",
                response.status()
            ));
            continue;
        }
        let messages: Value = response.json().await?;
        let value = json!({"info":item,"messages":messages});
        let source = format!("opencode:{id}");
        let raw_record = RawRecord::new(adapter.kind(), &source, 0, value.clone());
        let context = ImportContext {
            source,
            fallback_session_id: id.to_owned(),
            cwd: None,
            raw_record_ids: vec![raw_record.id.clone()],
        };
        report.raw_records.push(raw_record);
        match adapter.import_values(std::slice::from_ref(&value), &context) {
            Ok(imported) => {
                let (events, diagnostics) = resolve_imported_events(imported, blobs).await;
                report.events.extend(events);
                report.diagnostics.extend(diagnostics);
            }
            Err(error) => {
                report.diagnostics.push(ConversionDiagnostic::new(
                    &context.raw_record_ids[0],
                    DiagnosticSeverity::Error,
                    "adapter_conversion_failed",
                    error.to_string(),
                ));
                report
                    .errors
                    .push(format!("opencode session {id}: {error:#}"));
                continue;
            }
        }
        report.files += 1;
        report.records += 1;
    }
    child.kill();
    Ok(report)
}

struct ChildGuard(std::process::Child);

impl ChildGuard {
    fn kill(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

async fn import_codex_app_server(
    adapter: &dyn AgentAdapter,
    blobs: Option<&dyn BlobStore>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<ImportReport> {
    let mut command = TokioCommand::new("codex");
    command
        .args(["app-server", "--listen", "stdio://"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command.kill_on_drop(true);
    let mut child = command.spawn().context("start Codex App Server")?;
    let Some(stdin) = child.stdin.take() else {
        let _ = child.start_kill();
        let _ = child.wait().await;
        bail!("Codex App Server stdin unavailable");
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.start_kill();
        let _ = child.wait().await;
        bail!("Codex App Server stdout unavailable");
    };
    let result = run_codex_protocol(stdin, stdout, adapter, blobs, progress).await;
    let _ = child.start_kill();
    let cleanup = child.wait().await;
    match result {
        Ok(report) => {
            cleanup.context("wait for Codex App Server shutdown")?;
            Ok(report)
        }
        Err(error) => {
            let _ = cleanup;
            Err(error)
        }
    }
}

async fn run_codex_protocol(
    mut stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    adapter: &dyn AgentAdapter,
    blobs: Option<&dyn BlobStore>,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<ImportReport> {
    let mut reader = BufReader::new(stdout);
    send_json(
        &mut stdin,
        &json!({"method":"initialize","id":0,"params":{"clientInfo":{"name":"semafuse","title":"Semafuse","version":env!("CARGO_PKG_VERSION")}}}),
    ).await?;
    let _ = read_response(&mut reader, 0).await?;
    send_json(&mut stdin, &json!({"method":"initialized","params":{}})).await?;
    let mut report = ImportReport::default();
    let mut request_id = 1_i64;
    let mut thread_ids = Vec::new();
    let mut seen = HashSet::new();
    for archived in [false, true] {
        let mut cursor: Option<String> = None;
        loop {
            send_json(
                &mut stdin,
                &json!({"method":"thread/list","id":request_id,"params":{"cursor":cursor,"limit":100,"archived":archived}}),
            ).await?;
            let response = read_response(&mut reader, request_id).await?;
            request_id += 1;
            if let Some(error) = response.get("error") {
                bail!("Codex thread/list error: {error}");
            }
            let result = response.get("result").cloned().unwrap_or(Value::Null);
            for summary in result
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(id) = summary.get("id").and_then(Value::as_str) else {
                    continue;
                };
                if seen.insert(id.to_owned()) {
                    thread_ids.push(id.to_owned());
                }
            }
            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
    }
    let total = thread_ids.len();
    for (index, id) in thread_ids.into_iter().enumerate() {
        send_json(
            &mut stdin,
            &json!({"method":"thread/read","id":request_id,"params":{"threadId":id,"includeTurns":true}}),
        ).await?;
        let response = read_response(&mut reader, request_id).await?;
        request_id += 1;
        let source = format!("codex-app-server:{id}");
        let raw_record = RawRecord::new(adapter.kind(), &source, 0, response.clone());
        let context = ImportContext {
            source,
            fallback_session_id: id.clone(),
            cwd: None,
            raw_record_ids: vec![raw_record.id.clone()],
        };
        report.raw_records.push(raw_record);
        match adapter.import_values(std::slice::from_ref(&response), &context) {
            Ok(imported) => {
                let (events, diagnostics) = resolve_imported_events(imported, blobs).await;
                report.events.extend(events);
                report.diagnostics.extend(diagnostics);
            }
            Err(error) => {
                report.diagnostics.push(ConversionDiagnostic::new(
                    &context.raw_record_ids[0],
                    DiagnosticSeverity::Error,
                    "adapter_conversion_failed",
                    error.to_string(),
                ));
                report.errors.push(format!("codex thread {id}: {error:#}"));
                continue;
            }
        }
        report.files += 1;
        report.records += 1;
        progress(index + 1, total, &id);
    }
    Ok(report)
}

async fn send_json(writer: &mut (impl AsyncWrite + Unpin), value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    tokio::time::timeout(Duration::from_secs(30), async {
        writer.write_all(&bytes).await?;
        writer.flush().await?;
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("timed out writing agent protocol request")?
}

async fn read_response(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
    id: i64,
) -> Result<Value> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).await? == 0 {
                bail!("agent protocol closed before response id {id}");
            }
            let value: Value = serde_json::from_str(&line)?;
            if value.get("id").and_then(Value::as_i64) == Some(id) {
                return Ok(value);
            }
        }
    })
    .await
    .with_context(|| format!("timed out waiting for agent response id {id}"))?
}

async fn resolve_imported_events(
    imported: Vec<ImportedEvent>,
    blobs: Option<&dyn BlobStore>,
) -> (Vec<CanonicalEvent>, Vec<ConversionDiagnostic>) {
    let mut events = Vec::with_capacity(imported.len());
    let mut diagnostics = Vec::new();
    for imported in imported {
        let mut event = imported.event;
        if let Some(blobs) = blobs {
            match extract_attachments(&imported.attachment_source, blobs).await {
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

fn parse_file(path: &Path) -> Result<Vec<Value>> {
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
