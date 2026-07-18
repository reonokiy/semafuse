use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use semafuse::{
    AgentKind, QueryMode, SearchQuery,
    adapters::adapter,
    blob::{BlobStore, OpenDalBlobStore},
    canonical_archive_schema, config, hooks, importer, mcp,
    storage::{EventStore, IngestBatch, SqliteStore},
    strategy,
};
use serde_json::{Value, json};
use std::{io::Read, path::PathBuf, time::Duration};

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

impl Commands {
    fn name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Import { .. } => "import",
            Self::Hook { .. } => "hook",
            Self::Search { .. } => "search",
            Self::Session { .. } => "session",
            Self::Raw { .. } => "raw",
            Self::Schema => "schema",
            Self::Hooks {
                command: HookCommands::Print { .. },
            } => "hooks.print",
            Self::Hooks {
                command: HookCommands::Install { .. },
            } => "hooks.install",
            Self::Spool { .. } => "spool.replay",
            Self::Strategy { .. } => "strategy",
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
        if json_output {
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
            let adapter = adapter(agent);
            let mut progress = TerminalProgress::new("Discovering import sources");
            let runtime_config = if dry_run { None } else { Some(config::load()?) };
            let blob_store = runtime_config
                .as_ref()
                .map(|config| OpenDalBlobStore::from_config(&config.blob))
                .transpose()?;
            let report = if let Some(source) = source {
                importer::import_path(
                    &*adapter,
                    &source,
                    blob_store.as_ref().map(|store| store as &dyn BlobStore),
                    &mut |current, total, item| progress.update(current, total, item),
                )
                .await?
            } else {
                importer::import_default(
                    &*adapter,
                    blob_store.as_ref().map(|store| store as &dyn BlobStore),
                    &mut |current, total, item| progress.update(current, total, item),
                )
                .await?
            };
            let event_count = report.events.len();
            let raw_count = report.raw_records.len();
            let diagnostic_count = report.diagnostics.len();
            if dry_run {
                progress.finish();
                if json_output {
                    print_json(&json!({
                        "ok":true, "command":"import", "agent":agent, "dry_run":true,
                        "files":report.files, "records":report.records,
                        "raw_records":raw_count, "events":event_count,
                        "diagnostics":diagnostic_count, "errors":report.errors,
                    }))?;
                } else {
                    println!("Import preview for {agent}:");
                    println!("  Files:   {}", report.files);
                    println!("  Records: {}", report.records);
                    println!("  Raw:     {raw_count}");
                    println!("  Events:  {event_count}");
                    println!("  Diagnostics: {diagnostic_count}");
                    println!("  Errors:  {}", report.errors.len());
                }
            } else {
                progress.spinner(format!("Writing {event_count} events"));
                let store = SqliteStore::open(&db_path(db)).await?;
                let stats = store
                    .ingest_batch(IngestBatch {
                        raw_records: &report.raw_records,
                        events: &report.events,
                        diagnostics: &report.diagnostics,
                    })
                    .await?;
                let delivered = strategy::dispatch_pending(
                    &store,
                    runtime_config.as_ref().expect("config loaded for import"),
                    None,
                )
                .await?;
                progress.finish();
                if json_output {
                    print_json(&json!({
                        "ok":true, "command":"import", "agent":agent, "dry_run":false,
                        "files":report.files, "records":report.records,
                        "raw_records":raw_count, "events":event_count,
                        "diagnostics":diagnostic_count,
                        "raw_inserted":stats.raw_inserted,
                        "raw_deduplicated":stats.raw_deduplicated,
                        "inserted":stats.inserted,
                        "deduplicated":stats.deduplicated, "strategy_events_delivered":delivered, "errors":report.errors,
                    }))?;
                } else {
                    println!("Imported {agent} history:");
                    println!("  Files:        {}", report.files);
                    println!("  Records:      {}", report.records);
                    println!("  Raw inserted: {}", stats.raw_inserted);
                    println!("  Inserted:     {}", stats.inserted);
                    println!("  Deduplicated: {}", stats.deduplicated);
                    println!("  Diagnostics:  {diagnostic_count}");
                    println!("  Errors:       {}", report.errors.len());
                }
            }
            if !json_output {
                for error in report.errors {
                    eprintln!("Warning: {error}");
                }
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
            let payload: Value = if input.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&input).context("parse hook JSON from stdin")?
            };
            let path = db_path(db);
            let adapter = adapter(agent);
            let spool_path = spool_dir.unwrap_or_else(config::default_spool_dir);
            let result: Result<Option<Value>> = async {
                let runtime_config = config::load()?;
                let blob_store = OpenDalBlobStore::from_config(&runtime_config.blob)?;
                let store = SqliteStore::open(&path).await?;
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
            }
            .await;
            match result {
                Ok(response) => {
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
                    hooks::append_spool(&spool_path, agent, &event, &payload)?;
                    if protocol {
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
            limit,
        } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            let query = SearchQuery {
                text: query,
                repository: repository.filter(|s| !s.is_empty()),
                agent,
                limit: limit.clamp(1, 100),
            };
            let mode: QueryMode = mode.parse()?;
            let hits = store.query(mode, &query).await?;
            if json_output {
                print_json(&json!({
                    "ok":true, "command":"search", "query":query.text,
                    "mode":mode.as_str(), "count":hits.len(), "results":hits,
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
            let blob_store = OpenDalBlobStore::from_config(&runtime_config.blob)?;
            let store = SqliteStore::open(&db_path(db)).await?;
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
        Commands::Mcp { db } => {
            let store = SqliteStore::open(&db_path(db)).await?;
            mcp::serve(&store).await?;
        }
        Commands::Doctor { db } => {
            let path = db_path(db);
            let store = SqliteStore::open(&path).await?;
            let event_count = store.count().await?;
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
                    "events":event_count, "agents":agents,
                }))?;
            } else {
                println!("Semafuse status:");
                println!("  Database: {}", path.display());
                println!("  Events:   {event_count}");
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

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn artifacts(
    agent: AgentKind,
    binary: Option<PathBuf>,
    db: Option<PathBuf>,
) -> Result<Vec<(String, String)>> {
    let binary = binary.unwrap_or(std::env::current_exe()?);
    let database = db_path(db);
    let mut artifacts = adapter(agent).hook_artifacts(&binary, &database);
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

    fn update(&mut self, current: usize, total: usize, item: &str) {
        if total == 0 {
            return;
        }
        if self.bar.as_ref().and_then(ProgressBar::length) != Some(total as u64) {
            if let Some(previous) = self.bar.take() {
                previous.finish_and_clear();
            }
            let bar = ProgressBar::new(total as u64);
            bar.set_style(progress_style());
            bar.set_prefix("Reading");
            bar.enable_steady_tick(Duration::from_millis(80));
            self.bar = Some(bar);
        }
        let bar = self.bar.as_ref().expect("progress bar initialized");
        bar.set_message(item.to_owned());
        bar.set_position(current as u64);
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

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} [{bar:30.cyan/blue}] {pos:>3}/{len:3} {prefix} {msg:.36}",
    )
    .expect("static progress template is valid")
    .progress_chars("=> ")
    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
}

impl Drop for TerminalProgress {
    fn drop(&mut self) {
        self.finish();
    }
}

fn command_exists(command: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|path| {
            let candidate = path.join(command);
            candidate.is_file()
        })
    })
}
