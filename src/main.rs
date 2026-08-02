use anyhow::{Context, Result, bail};
use chrono::DateTime;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use semafuse::{
    AgentKind, DiagnosticSeverity, QueryMode, SearchQuery,
    adapters::{executable::command_available, history_importer, hook_installer, hook_projector},
    blob::{BlobStore, DryRunBlobStore, OpenDalBlobStore, drain_blob_gc},
    canonical_archive_schema, config,
    daemon_ipc::DaemonClient,
    governance::export_directory,
    hooks,
    importer::DeferredImportBlobStore,
    mcp,
    storage::{EventStore, ForgetFilter, IngestBatch, SqliteStore},
    strategy,
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fmt,
    io::Read,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
struct ReportedCommandFailure(String);

impl fmt::Display for ReportedCommandFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ReportedCommandFailure {}

#[cfg(test)]
use semafuse::adapters::executable::resolve_executable_in;
#[cfg(test)]
use std::ffi::OsStr;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Emit a stable machine-readable JSON result
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize or migrate the SQLite database
    Init {
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Import historical agent sessions idempotently
    Import {
        /// Source agent: amp, codex, claude-code, or opencode
        #[arg(long)]
        agent: AgentKind,
        /// JSON/JSONL file or directory; omit to use the agent's default source
        #[arg(long)]
        source: Option<PathBuf>,
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Parse and report without writing the database or blob store
        #[arg(long)]
        dry_run: bool,
    },
    /// Read one hook JSON payload from stdin and record it
    Hook {
        /// Agent that emitted the hook
        #[arg(long)]
        agent: AgentKind,
        /// Native lifecycle event name, or auto when present in the payload
        #[arg(long, default_value = "auto")]
        event: String,
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
        /// Return relevant prior context in the agent's native hook response
        #[arg(long)]
        recall: bool,
        /// Directory for payloads spooled after processing failures
        #[arg(long)]
        spool_dir: Option<PathBuf>,
        /// Emit the agent-native hook response (used by generated integrations)
        #[arg(long, hide = true)]
        protocol: bool,
    },
    /// Search normalized events across agents and sessions
    Search {
        /// Full-text query
        query: String,
        #[arg(long)]
        db: Option<PathBuf>,
        /// Restrict results to a repository/worktree identifier
        #[arg(long)]
        repository: Option<String>,
        /// Restrict results to one source agent
        #[arg(long)]
        agent: Option<AgentKind>,
        /// Raw event query mode: hybrid, lexical, or recent
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Include only events at or after this time (for example `2d`, `48h`, RFC3339, or Unix milliseconds)
        #[arg(long)]
        since: Option<String>,
        /// Include only events at or before this time (for example `2026-07-21T00:00:00Z` or Unix milliseconds)
        #[arg(long)]
        until: Option<String>,
        /// Maximum result count (clamped to 1..=100)
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// List all normalized events in a session
    Session {
        /// Canonical source session ID
        session_id: String,
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Show immutable native records by raw-record or canonical-event ID
    Raw {
        /// A `raw_...` record ID or an `evt_...` canonical event ID
        id: String,
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Export canonical records and verified attachments to a new directory
    Export {
        /// New output directory; existing paths are never overwritten
        #[arg(long)]
        output: PathBuf,
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Preview or confirm bounded deletion of retained conversation data
    Forget {
        /// Match one canonical source session ID
        #[arg(long)]
        session: Option<String>,
        /// Match one normalized repository/worktree identifier
        #[arg(long)]
        repository: Option<String>,
        /// Match one source agent
        #[arg(long)]
        agent: Option<AgentKind>,
        /// Match data strictly before this duration, RFC3339 time, or Unix milliseconds
        #[arg(long)]
        before: Option<String>,
        /// Commit the previewed deletion; omitted means no data is changed
        #[arg(long)]
        confirm: bool,
        /// Additional hook spool directory to clean; repeatable (the configured default is always included)
        #[arg(long)]
        spool_dir: Vec<PathBuf>,
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Retry attachment deletion and SQLite/WAL physical cleanup
    Gc {
        /// Maximum queued objects to process in this pass
        #[arg(long, default_value_t = 1_000)]
        limit: usize,
        /// SQLite database path (overrides SEMAFUSE_DB)
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Print the authoritative canonical archive JSON Schema
    Schema,
    /// Print or install native agent hook and MCP integration files
    Hooks {
        #[command(subcommand)]
        command: HookCommands,
    },
    /// Manage the durable hook failure queue
    Spool {
        #[command(subcommand)]
        command: SpoolCommands,
    },
    /// Inspect, retry, or rebuild memory strategies
    Strategy {
        #[command(subcommand)]
        command: StrategyCommands,
    },
    /// Inspect the separately managed semafused process
    Daemon {
        #[command(subcommand)]
        command: DaemonCommands,
    },
    /// Run the stdio MCP server used by coding-agent clients
    Mcp {
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Check database state and locally available agent commands
    Doctor {
        #[arg(long)]
        db: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum HookCommands {
    /// Print generated integration paths and contents without writing files
    Print {
        #[arg(long)]
        agent: AgentKind,
        #[arg(long)]
        binary: Option<PathBuf>,
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Install integration files into a project, refusing existing paths
    Install {
        #[arg(long)]
        agent: AgentKind,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        binary: Option<PathBuf>,
        #[arg(long)]
        db: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum SpoolCommands {
    /// Reprocess hook payloads saved while the database was unavailable
    Replay {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        spool_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum StrategyCommands {
    /// List configured strategies
    List {
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Show delivery checkpoints and failures
    Status {
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Retry pending events, optionally for one strategy
    Retry {
        name: Option<String>,
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Delete and rebuild one strategy from raw events
    Rebuild {
        name: String,
        #[arg(long)]
        db: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum DaemonCommands {
    /// Probe a running daemon without changing its services
    Status {
        /// Absolute Unix-domain socket path
        #[arg(long)]
        socket: PathBuf,
        #[arg(long, default_value_t = 1_000)]
        connect_timeout_ms: u64,
        #[arg(long, default_value_t = 120_000)]
        request_timeout_ms: u64,
    },
}

impl Commands {
    fn name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Import { .. } => "import",
            Self::Hook { .. } => "hook",
            Self::Search { .. } => "search",
            Self::Session { .. } => "session",
            Self::Raw { .. } => "raw",
            Self::Export { .. } => "export",
            Self::Forget { .. } => "forget",
            Self::Gc { .. } => "gc",
            Self::Schema => "schema",
            Self::Hooks {
                command: HookCommands::Print { .. },
            } => "hooks.print",
            Self::Hooks {
                command: HookCommands::Install { .. },
            } => "hooks.install",
            Self::Spool { .. } => "spool.replay",
            Self::Strategy { .. } => "strategy",
            Self::Daemon { .. } => "daemon.status",
            Self::Mcp { .. } => "mcp",
            Self::Doctor { .. } => "doctor",
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let json_output = cli.json;
    let command = cli.command.name();
    if let Err(error) = run(cli).await {
        if json_output && error.downcast_ref::<ReportedCommandFailure>().is_some() {
            // The command already emitted its complete machine-readable report.
        } else if json_output {
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok":false, "command":command, "error":format!("{error:#}")
                }))
                .expect("error JSON serialization cannot fail")
            );
        } else {
            eprintln!("Error: {error:#}");
        }
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    let json_output = cli.json;
    match cli.command {
        Commands::Init { db } => {
            let path = db_path(db);
            SqliteStore::open(&path).await?;
            if json_output {
                print_json(
                    &json!({"ok":true,"command":"init","initialized":true,"database":path}),
                )?;
            } else {
                println!("Initialized database at {}.", path.display());
            }
        }
        Commands::Import {
            agent,
            source,
            db,
            dry_run,
        } => {
            let importer = history_importer(agent);
            let mut progress = TerminalProgress::new("Discovering import sources");
            let runtime_config = if dry_run { None } else { Some(config::load()?) };
            let store = if dry_run {
                None
            } else {
                Some(SqliteStore::open(&db_path(db)).await?)
            };
            let lifecycle_guard = store
                .as_ref()
                .map(SqliteStore::lock_lifecycle)
                .transpose()?;
            let dry_run_blobs = DryRunBlobStore;
            let deferred_blobs = DeferredImportBlobStore;
            let blob_store = if let (Some(config), Some(store)) = (&runtime_config, &store) {
                Some(OpenDalBlobStore::from_config_for_store(&config.blob, store).await?)
            } else {
                None
            };
            let blobs: &dyn BlobStore = if dry_run {
                &dry_run_blobs
            } else {
                blob_store
                    .as_ref()
                    .map(|store| store as &dyn BlobStore)
                    .expect("blob store opened for import")
            };
            let mut report = if let Some(source) = source {
                progress.spinner(format!(
                    "Importing {agent} history from {}",
                    source.display()
                ));
                importer.import_path(&source, Some(&deferred_blobs)).await?
            } else {
                progress.spinner(format!("Importing {agent} default history"));
                importer.import_default(Some(&deferred_blobs)).await?
            };
            report.finalize_events(store.as_ref(), blobs).await?;
            let event_count = report.events.len();
            let raw_count = report.raw_records.len();
            let diagnostic_count = report.diagnostics.len();
            let failed_records = report
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
                .map(|diagnostic| diagnostic.raw_record_id.as_str())
                .collect::<HashSet<_>>()
                .len();
            let mut complete = report.errors.is_empty() && failed_records == 0;
            if dry_run {
                progress.finish();
                if json_output {
                    print_json(&json!({
                        "ok":complete, "complete":complete,
                        "command":"import", "agent":agent, "dry_run":true,
                        "files":report.files, "records":report.records,
                        "raw_records":raw_count, "events":event_count,
                        "diagnostics":diagnostic_count, "failed_records":failed_records,
                        "source_errors":report.errors.len(), "errors":report.errors,
                    }))?;
                } else {
                    println!("Import preview for {agent}:");
                    println!("  Files:   {}", report.files);
                    println!("  Records: {}", report.records);
                    println!("  Raw:     {raw_count}");
                    println!("  Events:  {event_count}");
                    println!("  Diagnostics: {diagnostic_count}");
                    println!("  Failed records: {failed_records}");
                    println!("  Errors:  {}", report.errors.len());
                }
            } else {
                progress.spinner(format!("Writing {event_count} events"));
                let store = store.as_ref().expect("store opened for import");
                let storage_events = report.events_for_storage();
                let stats = store
                    .ingest_batch(IngestBatch {
                        raw_records: &report.raw_records,
                        raw_contexts: &report.raw_contexts,
                        events: &storage_events,
                        diagnostics: &report.diagnostics,
                    })
                    .await?;
                drop(lifecycle_guard);
                let (attachment_gc, attachment_gc_error) = if stats.attachment_objects_queued > 0 {
                    match drain_blob_gc(
                        store,
                        blob_store.as_ref().expect("blob store opened for import"),
                        stats.attachment_objects_queued.min(10_000),
                    )
                    .await
                    {
                        Ok(report) => (Some(report), None),
                        Err(error) => (
                            None,
                            Some(format!("attachment cleanup remains queued: {error:#}")),
                        ),
                    }
                } else {
                    (None, None)
                };
                complete &= !stats.sqlite_cleanup_pending
                    && stats.unmanaged_attachment_objects == 0
                    && attachment_gc_error.is_none()
                    && attachment_gc
                        .as_ref()
                        .is_none_or(|report| report.pending == 0 && report.failures.is_empty());
                let delivered = strategy::dispatch_pending(
                    store,
                    runtime_config.as_ref().expect("config loaded for import"),
                    None,
                )
                .await?;
                progress.finish();
                if json_output {
                    print_json(&json!({
                        "ok":complete, "complete":complete,
                        "command":"import", "agent":agent, "dry_run":false,
                        "files":report.files, "records":report.records,
                        "raw_records":raw_count, "events":event_count,
                        "diagnostics":diagnostic_count,
                        "raw_inserted":stats.raw_inserted,
                        "raw_deduplicated":stats.raw_deduplicated,
                        "raw_tombstoned":stats.raw_tombstoned,
                        "inserted":stats.inserted,
                        "deduplicated":stats.deduplicated, "tombstoned":stats.tombstoned,
                        "failed_records":failed_records, "source_errors":report.errors.len(),
                        "sqlite_cleanup_pending":stats.sqlite_cleanup_pending,
                        "sqlite_cleanup_error":stats.sqlite_cleanup_error,
                        "attachment_objects_queued":stats.attachment_objects_queued,
                        "unmanaged_attachment_objects":stats.unmanaged_attachment_objects,
                        "attachment_gc":attachment_gc,
                        "attachment_gc_error":attachment_gc_error,
                        "strategy_events_delivered":delivered, "errors":report.errors,
                    }))?;
                } else {
                    println!("Imported {agent} history:");
                    println!("  Files:        {}", report.files);
                    println!("  Records:      {}", report.records);
                    println!("  Raw inserted: {}", stats.raw_inserted);
                    println!("  Raw blocked by forget rules: {}", stats.raw_tombstoned);
                    println!("  Inserted:     {}", stats.inserted);
                    println!("  Deduplicated: {}", stats.deduplicated);
                    println!("  Events blocked by forget rules: {}", stats.tombstoned);
                    println!(
                        "  Legacy attachments queued: {}",
                        stats.attachment_objects_queued
                    );
                    println!(
                        "  Legacy/unmanaged attachments: {}",
                        stats.unmanaged_attachment_objects
                    );
                    println!("  Diagnostics:  {diagnostic_count}");
                    println!("  Failed records: {failed_records}");
                    println!("  Errors:       {}", report.errors.len());
                    if let Some(error) = &stats.sqlite_cleanup_error {
                        eprintln!(
                            "Warning: replay cleanup committed, but SQLite physical cleanup remains pending: {error}"
                        );
                    }
                    if let Some(gc) = &attachment_gc {
                        println!("  Legacy attachments deleted: {}", gc.deleted);
                        if gc.pending > 0 || !gc.failures.is_empty() {
                            eprintln!(
                                "Warning: {} attachment object(s) remain queued with {} failure(s); run `semafuse gc`.",
                                gc.pending,
                                gc.failures.len()
                            );
                        }
                    }
                    if let Some(error) = &attachment_gc_error {
                        eprintln!("Warning: {error}");
                    }
                }
            }
            if !json_output {
                for error in &report.errors {
                    eprintln!("Warning: {error}");
                }
            }
            if !complete {
                return Err(ReportedCommandFailure(format!(
                    "import incomplete: {failed_records} record(s) failed normalization, {} source error(s) occurred, or SQLite/attachment cleanup remains pending",
                    report.errors.len(),
                ))
                .into());
            }
        }
        Commands::Hook {
            agent,
            event,
            db,
            recall,
            spool_dir,
            protocol,
        } => {
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            // Codex-backed LazyMem processing launches an isolated child Codex
            // with hooks disabled. Managed policy can force hooks back on, so
            // the inherited depth marker is the final recursion barrier. Drain
            // stdin first so the parent hook runner never sees a broken pipe.
            if hooks::is_processor_child() {
                return Ok(());
            }
            let payload: Value = if input.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&input).context("parse hook JSON from stdin")?
            };
            let path = db_path(db);
            let adapter = hook_projector(agent);
            let spool_path = spool_dir.unwrap_or_else(config::default_spool_dir);
            let resolved_event = if event == "auto" {
                payload
                    .get("hook_event_name")
                    .and_then(Value::as_str)
                    .unwrap_or("other")
            } else {
                &event
            };
            let deadline_sensitive = agent == AgentKind::Codex
                && matches!(
                    resolved_event.to_ascii_lowercase().as_str(),
                    "sessionend" | "session_end"
                );
            // Write-ahead staging makes both crashes and concurrent forget
            // cleanup explicit: success removes this locked generation;
            // failure leaves it for replay or a tombstone-aware discard.
            let pending_spool = hooks::stage_spool(&spool_path, agent, &event, &payload)?;
            let processing = async {
                let runtime_config = config::load()?;
                let store = SqliteStore::open(&path).await?;
                let blob_store =
                    OpenDalBlobStore::from_config_for_store(&runtime_config.blob, &store).await?;
                hooks::process_hook(
                    &*adapter,
                    &store,
                    &event,
                    &payload,
                    recall,
                    Some(&blob_store),
                    &runtime_config,
                )
                .await
            };
            // Codex caps SessionEnd hooks at three seconds. Reserve enough time
            // to durably append the original payload to the spool if SQLite is
            // busy or another processing step stalls.
            let result: Result<Option<Value>> = if deadline_sensitive {
                match tokio::time::timeout(Duration::from_millis(1_500), processing).await {
                    Ok(result) => result,
                    Err(_) => Err(anyhow::anyhow!(
                        "Codex SessionEnd processing exceeded the 1.5 second persistence budget"
                    )),
                }
            } else {
                processing.await
            };
            match result {
                Ok(response) => {
                    if let Err(error) = pending_spool.complete() {
                        eprintln!(
                            "semafuse recorded the hook but could not remove its deduplicated write-ahead spool: {error:#}"
                        );
                    }
                    if protocol {
                        if let Some(response) = response {
                            println!("{}", serde_json::to_string(&response)?);
                        }
                    } else if json_output {
                        print_json(&json!({
                            "ok":true, "command":"hook", "agent":agent, "event":event,
                            "recorded":true, "spooled":false, "recall":response,
                        }))?;
                    } else {
                        println!("Recorded {agent} hook event {event}.");
                        if response.is_some() {
                            println!("Relevant prior context was found.");
                        }
                    }
                }
                Err(error) => {
                    // Recheck durable tombstones after a failure that happened
                    // before normal processing (for example while opening the
                    // blob backend). The locked write-ahead generation prevents
                    // an in-flight forget from falsely claiming spool cleanup.
                    let mut tombstoned = false;
                    if let Ok(store) = SqliteStore::open(&path).await {
                        // A committed forget rule is safe to read while another
                        // lifecycle operation owns the writer lock. An
                        // uncommitted forget cannot claim cleanup complete while
                        // this pending file is still locked by the hook.
                        tombstoned =
                            hooks::hook_payload_is_tombstoned(&*adapter, &store, &event, &payload)
                                .await
                                .unwrap_or(false);
                    }
                    if tombstoned {
                        pending_spool.complete()?;
                        if !protocol && json_output {
                            print_json(&json!({
                                "ok":true, "command":"hook", "agent":agent, "event":event,
                                "recorded":false, "spooled":false, "forgotten":true,
                            }))?;
                        } else if !protocol {
                            println!(
                                "Discarded {agent} hook event {event} because it matches a forget rule."
                            );
                        }
                    } else if protocol {
                        eprintln!("semafuse hook spooled after processing error: {error:#}");
                    } else if json_output {
                        print_json(&json!({
                            "ok":true, "command":"hook", "agent":agent, "event":event,
                            "recorded":false, "spooled":true, "error":format!("{error:#}"),
                        }))?;
                    } else {
                        println!("Could not record the hook event; it was saved for replay.");
                        eprintln!("Reason: {error:#}");
                    }
                }
            }
        }
        Commands::Search {
            query,
            db,
            repository,
            agent,
            mode,
            since,
            until,
            limit,
        } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            let query = SearchQuery {
                text: query,
                repository: repository.filter(|s| !s.is_empty()),
                agent,
                since_ms: since
                    .as_deref()
                    .map(parse_time_bound)
                    .transpose()
                    .context("parse --since")?,
                until_ms: until
                    .as_deref()
                    .map(parse_time_bound)
                    .transpose()
                    .context("parse --until")?,
                limit: limit.clamp(1, 100),
            };
            if let (Some(since), Some(until)) = (query.since_ms, query.until_ms)
                && since > until
            {
                bail!("--since must be earlier than or equal to --until");
            }
            let mode: QueryMode = mode.parse()?;
            let hits = store.query(mode, &query).await?;
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"search", "query":query.text,
                    "mode":mode.as_str(), "since_ms":query.since_ms,
                    "until_ms":query.until_ms, "count":hits.len(), "results":hits,
                }))?;
            } else {
                if hits.is_empty() {
                    println!("No matching events found.");
                }
                for (index, hit) in hits.into_iter().enumerate() {
                    let content = hit.event.search_text().replace('\n', " ");
                    let content = if content.trim().is_empty() {
                        "(no textual content)"
                    } else {
                        &content
                    };
                    println!(
                        "{}. {} · session {} · score {:.3}\n   {}",
                        index + 1,
                        hit.event.source.agent,
                        hit.event.source.session_id,
                        hit.score,
                        content
                    );
                }
            }
        }
        Commands::Session { session_id, db } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            let events = store.session_events(&session_id).await?;
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"session", "session_id":session_id,
                    "count":events.len(), "events":events,
                }))?;
            } else {
                println!("Session {session_id} ({} events):", events.len());
                for event in events {
                    println!(
                        "  {} · {} · {}",
                        event.ordering_timestamp_ms(),
                        event.event_type(),
                        event.search_text().replace('\n', " ")
                    );
                }
            }
        }
        Commands::Raw { id, db } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            let records = if id.starts_with("evt_") {
                store.raw_records_for_event(&id).await?
            } else {
                store.raw_record(&id).await?.into_iter().collect()
            };
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"raw", "id":id,
                    "count":records.len(), "records":records,
                }))?;
            } else if records.is_empty() {
                println!("No raw records found for {id}.");
            } else {
                print_json(&serde_json::to_value(records)?)?;
            }
        }
        Commands::Export { output, db } => {
            let runtime_config = config::load()?;
            let store = SqliteStore::open(&db_path(db)).await?;
            let blob_store =
                OpenDalBlobStore::from_config_for_store(&runtime_config.blob, &store).await?;
            let report = export_directory(&store, &blob_store, &output).await?;
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"export", "output":report.output,
                    "raw_records":report.raw_records, "events":report.events,
                    "diagnostics":report.diagnostics,
                    "attachment_objects":report.attachment_objects,
                }))?;
            } else {
                println!(
                    "Exported a portable archive to {}:",
                    report.output.display()
                );
                println!("  Raw records:  {}", report.raw_records);
                println!("  Events:       {}", report.events);
                println!("  Diagnostics:  {}", report.diagnostics);
                println!("  Attachments:  {}", report.attachment_objects);
            }
        }
        Commands::Forget {
            session,
            repository,
            agent,
            before,
            confirm,
            spool_dir,
            db,
        } => {
            let filter = ForgetFilter {
                session_id: session,
                repository,
                agent,
                before_ms: before
                    .as_deref()
                    .map(parse_time_bound)
                    .transpose()
                    .context("parse --before")?,
            };
            filter.validate()?;
            let store = SqliteStore::open(&db_path(db)).await?;
            let preview = store.preview_forget(&filter).await?;
            let mut spool_directories = vec![config::default_spool_dir()];
            for directory in spool_dir {
                if !spool_directories.contains(&directory) {
                    spool_directories.push(directory);
                }
            }
            if !confirm {
                let spool_preview = hooks::preview_spool_forget(&spool_directories, &filter)?;
                if json_output {
                    print_json(&json!({
                        "ok":true, "command":"forget", "confirmed":false,
                        "changed":false, "preview":preview,
                        "spool_preview":spool_preview,
                    }))?;
                } else {
                    println!("Forget preview (no data changed):");
                    println!("  Events:       {}", preview.events);
                    println!("  Raw records:  {}", preview.raw_records);
                    println!("  Attachments:  {}", preview.attachment_objects);
                    println!(
                        "  Legacy/unmanaged attachments: {}",
                        preview.unmanaged_attachment_objects
                    );
                    println!("  Spooled hook records: {}", spool_preview.matched);
                    println!("Run the same command with --confirm to commit this deletion.");
                }
                return Ok(());
            }

            let runtime_config = config::load()?;
            strategy::validate(&runtime_config)?;
            let blob_store =
                OpenDalBlobStore::from_config_for_store(&runtime_config.blob, &store).await?;
            let stats = store.forget(&filter).await?;
            let (spool_cleanup, spool_error) =
                match hooks::forget_spool_records(&spool_directories, &filter) {
                    Ok(report) => (Some(report), None),
                    Err(error) => (
                        None,
                        Some(format!("hook spool cleanup remains incomplete: {error:#}")),
                    ),
                };
            let (gc, gc_error) = match drain_blob_gc(&store, &blob_store, 10_000).await {
                Ok(report) => (Some(report), None),
                Err(error) => (
                    None,
                    Some(format!("attachment cleanup remains queued: {error:#}")),
                ),
            };
            let (strategy_events_delivered, strategy_error) =
                match strategy::dispatch_pending(&store, &runtime_config, None).await {
                    Ok(delivered) => (delivered, None),
                    Err(error) => (
                        0,
                        Some(format!("derived-memory replay remains pending: {error:#}")),
                    ),
                };
            let final_cleanup_guard = store.lock_lifecycle()?;
            let sqlite_cleanup_pending = store.sqlite_cleanup_is_pending().await?;
            let pending_attachment_gc = store.pending_blob_gc_count().await?;
            let unmanaged_blob_residue = store.unmanaged_blob_residue_count().await?;
            drop(final_cleanup_guard);
            let incomplete = unmanaged_blob_residue > 0
                || sqlite_cleanup_pending
                || pending_attachment_gc > 0
                || spool_error.is_some()
                || gc_error.is_some()
                || gc.as_ref().is_none_or(|report| !report.failures.is_empty())
                || strategy_error.is_some();
            if json_output {
                print_json(&json!({
                    "ok":!incomplete, "command":"forget", "confirmed":true,
                    "logical_deletion_committed":true, "cleanup_complete":!incomplete,
                    "changed":stats.events > 0 || stats.raw_records > 0
                        || spool_cleanup.as_ref().is_some_and(|report| report.matched > 0),
                    "preview":preview, "deleted":stats, "attachment_gc":gc,
                    "attachment_gc_error":gc_error,
                    "pending_attachment_gc":pending_attachment_gc,
                    "sqlite_cleanup_pending":sqlite_cleanup_pending,
                    "unmanaged_attachment_residue":unmanaged_blob_residue,
                    "spool_cleanup":spool_cleanup, "spool_cleanup_error":spool_error,
                    "strategy_events_delivered":strategy_events_delivered,
                    "strategy_error":strategy_error,
                }))?;
            } else {
                println!("Forgot retained data:");
                println!("  Events:       {}", stats.events);
                println!("  Raw records:  {}", stats.raw_records);
                println!("  Attachments queued: {}", stats.attachment_objects_queued);
                println!(
                    "  Legacy/unmanaged attachments not deleted: {}",
                    unmanaged_blob_residue
                );
                println!("  Events remaining:  {}", stats.remaining_events);
                if let Some(spool_cleanup) = spool_cleanup {
                    println!("  Spooled hook records deleted: {}", spool_cleanup.matched);
                }
                if let Some(error) = spool_error {
                    eprintln!("Warning: {error}");
                }
                if let Some(gc) = gc {
                    println!("  Attachments deleted: {}", gc.deleted);
                    if pending_attachment_gc > 0 {
                        eprintln!(
                            "Warning: {} attachment object(s) remain queued; run `semafuse gc`.",
                            pending_attachment_gc
                        );
                    }
                }
                if let Some(error) = gc_error {
                    eprintln!("Warning: {error}");
                }
                if let Some(error) = strategy_error {
                    eprintln!("Warning: {error}");
                }
                if let Some(error) = &stats.sqlite_cleanup_error {
                    eprintln!(
                        "Warning: SQLite physical cleanup remains pending; rerun this forget command or `semafuse gc`: {error}"
                    );
                }
            }
            if incomplete {
                bail!(
                    "logical deletion committed, but SQLite, spool, attachment, or derived-memory cleanup remains incomplete"
                );
            }
        }
        Commands::Gc { limit, db } => {
            anyhow::ensure!(
                (1..=10_000).contains(&limit),
                "--limit must be between 1 and 10000"
            );
            let runtime_config = config::load()?;
            let store = SqliteStore::open(&db_path(db)).await?;
            let blob_store =
                OpenDalBlobStore::from_config_for_store(&runtime_config.blob, &store).await?;
            let mut report = drain_blob_gc(&store, &blob_store, limit).await?;
            let _guard = store.lock_lifecycle()?;
            store
                .scrub_deleted_pages()
                .await
                .context("SQLite physical cleanup remains incomplete")?;
            let sqlite_cleanup_pending = store.sqlite_cleanup_is_pending().await?;
            report.pending = store.pending_blob_gc_count().await?;
            let unmanaged_blob_residue = store.unmanaged_blob_residue_count().await?;
            let incomplete = report.pending > 0
                || !report.failures.is_empty()
                || sqlite_cleanup_pending
                || unmanaged_blob_residue > 0;
            if json_output {
                print_json(&json!({
                    "ok":!incomplete, "command":"gc",
                    "deleted":report.deleted, "pending":report.pending,
                    "failures":report.failures,
                    "sqlite_cleanup_pending":sqlite_cleanup_pending,
                    "unmanaged_attachment_residue":unmanaged_blob_residue,
                }))?;
            } else {
                println!("Deleted {} orphaned attachment object(s).", report.deleted);
                if report.pending > 0 {
                    eprintln!(
                        "Warning: {} attachment object(s) remain queued.",
                        report.pending
                    );
                }
                if unmanaged_blob_residue > 0 {
                    eprintln!(
                        "Warning: {unmanaged_blob_residue} legacy/unmanaged attachment object(s) require manual cleanup."
                    );
                }
                for failure in report.failures {
                    eprintln!("Warning: {}: {}", failure.object_key, failure.error);
                }
            }
            if incomplete {
                bail!("attachment garbage collection remains incomplete");
            }
        }
        Commands::Schema => print_json(&canonical_archive_schema())?,
        Commands::Hooks { command } => match command {
            HookCommands::Print { agent, binary, db } => {
                let artifacts = artifacts(agent, binary, db)?;
                if json_output {
                    let artifacts: Vec<_> = artifacts
                        .into_iter()
                        .map(|(path, content)| json!({"path":path,"content":content}))
                        .collect();
                    print_json(&json!({
                        "ok":true, "command":"hooks.print", "agent":agent, "artifacts":artifacts
                    }))?;
                } else {
                    println!("Generated integration files for {agent}:");
                    for (path, content) in artifacts {
                        println!("\n--- {path} ---\n{content}");
                    }
                }
            }
            HookCommands::Install {
                agent,
                target,
                binary,
                db,
            } => {
                let artifacts = artifacts(agent, binary, db)?;
                let conflicts: Vec<_> = artifacts
                    .iter()
                    .map(|(relative, _)| target.join(relative))
                    .filter(|path| path.exists())
                    .collect();
                if !conflicts.is_empty() {
                    let paths = conflicts
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    bail!("refusing to overwrite existing integration files: {paths}");
                }
                let mut installed = Vec::new();
                for (relative, content) in artifacts {
                    let path = target.join(relative);
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&path, content)?;
                    installed.push(path);
                }
                if json_output {
                    print_json(&json!({
                        "ok":true, "command":"hooks.install", "agent":agent,
                        "installed":installed,
                    }))?;
                } else {
                    println!(
                        "Installed {} integration files for {agent}:",
                        installed.len()
                    );
                    for path in installed {
                        println!("  - {}", path.display());
                    }
                }
            }
        },
        Commands::Spool {
            command: SpoolCommands::Replay { db, spool_dir },
        } => {
            let runtime_config = config::load()?;
            let store = SqliteStore::open(&db_path(db)).await?;
            let blob_store =
                OpenDalBlobStore::from_config_for_store(&runtime_config.blob, &store).await?;
            let count = hooks::replay_spool(
                &spool_dir.unwrap_or_else(config::default_spool_dir),
                &store,
                Some(&blob_store),
                &runtime_config,
            )
            .await?;
            if json_output {
                print_json(&json!({"ok":true,"command":"spool.replay","replayed":count}))?;
            } else if count == 0 {
                println!("No spooled hook events to replay.");
            } else {
                println!("Replayed {count} spooled hook event(s).");
            }
        }
        Commands::Strategy { command } => {
            let db = match &command {
                StrategyCommands::List { db }
                | StrategyCommands::Status { db }
                | StrategyCommands::Retry { db, .. }
                | StrategyCommands::Rebuild { db, .. } => db.clone(),
            };
            let store = SqliteStore::open(&db_path(db)).await?;
            let strategy_config = config::load()?;
            strategy::validate(&strategy_config)?;
            match command {
                StrategyCommands::List { .. } => {
                    let values: Vec<_> = strategy_config.strategies.enabled.iter().map(|name| json!({"name":name,"recall":name == &strategy_config.strategies.recall})).collect();
                    if json_output {
                        print_json(
                            &json!({"ok":true,"command":"strategy.list","strategies":values}),
                        )?;
                    } else {
                        for value in values {
                            println!(
                                "{}{}",
                                value["name"].as_str().unwrap_or_default(),
                                if value["recall"] == true {
                                    " (recall)"
                                } else {
                                    ""
                                }
                            );
                        }
                    }
                }
                StrategyCommands::Status { .. } => {
                    let mut values = Vec::new();
                    for name in &strategy_config.strategies.enabled {
                        let (last_seq, last_error, attempts) =
                            store.strategy_checkpoint(name).await?;
                        values.push(json!({"name":name,"last_seq":last_seq,"last_error":last_error,"attempts":attempts}));
                    }
                    if json_output {
                        print_json(
                            &json!({"ok":true,"command":"strategy.status","strategies":values}),
                        )?;
                    } else {
                        for value in values {
                            println!(
                                "{} · seq {} · attempts {}{}",
                                value["name"].as_str().unwrap_or_default(),
                                value["last_seq"],
                                value["attempts"],
                                value["last_error"]
                                    .as_str()
                                    .map(|e| format!(" · {e}"))
                                    .unwrap_or_default()
                            );
                        }
                    }
                }
                StrategyCommands::Retry { name, .. } => {
                    let delivered =
                        strategy::dispatch_pending(&store, &strategy_config, name.as_deref())
                            .await?;
                    if json_output {
                        print_json(
                            &json!({"ok":true,"command":"strategy.retry","delivered":delivered}),
                        )?;
                    } else {
                        println!("Delivered {delivered} pending strategy event(s).");
                    }
                }
                StrategyCommands::Rebuild { name, .. } => {
                    let delivered = strategy::rebuild(&store, &strategy_config, &name).await?;
                    if json_output {
                        print_json(
                            &json!({"ok":true,"command":"strategy.rebuild","strategy":name,"delivered":delivered}),
                        )?;
                    } else {
                        println!("Rebuilt {name} from {delivered} domain event(s).");
                    }
                }
            }
        }
        Commands::Daemon {
            command:
                DaemonCommands::Status {
                    socket,
                    connect_timeout_ms,
                    request_timeout_ms,
                },
        } => {
            let status = DaemonClient::status_at(
                socket.clone(),
                Duration::from_millis(connect_timeout_ms),
                Duration::from_millis(request_timeout_ms),
            )
            .await
            .context("semafused status probe failed")?;
            let models = status.models().context("invalid models service status")?;
            if json_output {
                print_json(&json!({
                    "ok": true,
                    "command": "daemon.status",
                    "socket": socket,
                    "status": status,
                }))?;
            } else {
                println!(
                    "semafused {:?} at {} (version {}).",
                    status.state,
                    socket.display(),
                    status.daemon_version
                );
                println!(
                    "  Services: {}",
                    status
                        .services
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if let Some(models) = models {
                    println!(
                        "  Models: {:?} (version {})",
                        models.state, models.service_version
                    );
                    if let Some(identity) = models.embedding {
                        println!(
                            "    Embedding: {} ({}, {} dimensions)",
                            identity.model,
                            identity.fingerprint,
                            identity.dimensions.unwrap_or_default()
                        );
                    }
                    if let Some(identity) = models.reranker {
                        println!(
                            "    Reranker:  {} ({})",
                            identity.model, identity.fingerprint
                        );
                    }
                }
            }
        }
        Commands::Mcp { db } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            mcp::serve(&store).await?;
        }
        Commands::Doctor { db } => {
            let path = db_path(db);
            let store = SqliteStore::open(&path).await?;
            let event_count = store.count().await?;
            let pending_blob_gc = store.pending_blob_gc_count().await?;
            let unmanaged_blob_residue = store.unmanaged_blob_residue_count().await?;
            let mut agents = serde_json::Map::new();
            for agent in [
                AgentKind::Amp,
                AgentKind::Codex,
                AgentKind::ClaudeCode,
                AgentKind::OpenCode,
            ] {
                let command = match agent {
                    AgentKind::ClaudeCode => "claude",
                    _ => agent.as_str(),
                };
                agents.insert(agent.to_string(), json!(command_exists(command)));
            }
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"doctor", "database":path,
                    "events":event_count, "pending_attachment_gc":pending_blob_gc,
                    "unmanaged_attachment_residue":unmanaged_blob_residue,
                    "database_encrypted":false, "agents":agents,
                }))?;
            } else {
                println!("Semafuse status:");
                println!("  Database: {}", path.display());
                println!("  Events:   {event_count}");
                println!("  Pending attachment GC: {pending_blob_gc}");
                println!("  Unmanaged attachment residue: {unmanaged_blob_residue}");
                println!("  Database encryption: not enabled (use full-disk encryption)");
                println!("  Agents:");
                for (agent, available) in agents {
                    let status = if available == Value::Bool(true) {
                        "available"
                    } else {
                        "not found"
                    };
                    println!("    - {agent}: {status}");
                }
            }
        }
    }
    Ok(())
}

fn db_path(path: Option<PathBuf>) -> PathBuf {
    path.unwrap_or_else(config::default_db_path)
}

fn parse_time_bound(value: &str) -> Result<i64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("time bound cannot be empty");
    }
    if let Some(amount) = trimmed.strip_suffix("ms") {
        let amount: i64 = amount.parse().context("parse millisecond duration")?;
        return Ok(now_ms().saturating_sub(amount));
    }
    if let Some(unit) = trimmed
        .chars()
        .last()
        .filter(|c| matches!(c, 's' | 'm' | 'h' | 'd' | 'w'))
    {
        let amount = &trimmed[..trimmed.len() - unit.len_utf8()];
        if !amount.chars().all(|c| c.is_ascii_digit()) {
            bail!("duration amount must be an integer in {trimmed:?}");
        }
        let amount: i64 = amount
            .parse()
            .with_context(|| format!("parse duration amount in {trimmed:?}"))?;
        let multiplier: i64 = match unit {
            's' => 1_000,
            'm' => 60_000,
            'h' => 3_600_000,
            'd' => 86_400_000,
            'w' => 604_800_000,
            _ => unreachable!("duration units are filtered above"),
        };
        return Ok(now_ms().saturating_sub(amount.saturating_mul(multiplier)));
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return trimmed.parse().context("parse Unix millisecond timestamp");
    }
    DateTime::parse_from_rfc3339(trimmed)
        .map(|timestamp| timestamp.timestamp_millis())
        .context("parse RFC3339 timestamp")
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn artifacts(
    agent: AgentKind,
    binary: Option<PathBuf>,
    db: Option<PathBuf>,
) -> Result<Vec<(String, String)>> {
    let binary = binary.unwrap_or(std::env::current_exe()?);
    let database = db_path(db);
    let mut artifacts = hook_installer(agent).hook_artifacts(&binary, &database);
    let mcp_command = binary.display().to_string();
    let args = vec!["mcp", "--db", database.to_str().unwrap_or_default()];
    match agent {
        AgentKind::Amp => artifacts.push((".amp/settings.json".into(), serde_json::to_string_pretty(&json!({"amp.mcpServers":{"semafuse":{"command":mcp_command,"args":args}}}))?)),
        AgentKind::Codex => artifacts.push((".codex/config.toml".into(), format!("[mcp_servers.semafuse]\ncommand = {mcp_command:?}\nargs = {args:?}\n"))),
        AgentKind::ClaudeCode => artifacts.push((".mcp.json".into(), serde_json::to_string_pretty(&json!({"mcpServers":{"semafuse":{"command":mcp_command,"args":args}}}))?)),
        AgentKind::OpenCode => artifacts.push(("opencode.json".into(), serde_json::to_string_pretty(&json!({"$schema":"https://opencode.ai/config.json","mcp":{"semafuse":{"type":"local","command":[mcp_command,"mcp","--db",database]}}}))?)),
    }
    Ok(artifacts)
}

struct TerminalProgress {
    bar: Option<ProgressBar>,
}

impl TerminalProgress {
    fn new(message: &'static str) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(spinner_style());
        bar.set_message(message);
        bar.enable_steady_tick(Duration::from_millis(80));
        Self { bar: Some(bar) }
    }

    fn spinner(&mut self, message: String) {
        if let Some(previous) = self.bar.take() {
            previous.finish_and_clear();
        }
        let bar = ProgressBar::new_spinner();
        bar.set_style(spinner_style());
        bar.set_message(message);
        bar.enable_steady_tick(Duration::from_millis(80));
        self.bar = Some(bar);
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} {msg}")
        .expect("static spinner template is valid")
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
}

impl Drop for TerminalProgress {
    fn drop(&mut self) {
        self.finish();
    }
}

fn command_exists(command: &str) -> bool {
    command_available(command)
}

#[cfg(test)]
fn command_exists_in(
    command: &str,
    paths: Option<&OsStr>,
    path_ext: Option<&OsStr>,
    windows: bool,
) -> bool {
    resolve_executable_in(command, paths, path_ext, windows).is_some()
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
