# Semafuse

Semafuse is a local, coding-agent-agnostic storage and retrieval layer for conversational memory. It currently supports
Amp, Codex, Claude Code, and OpenCode, and explicitly distinguishes between two ingestion paths:

- One-time history import: reads existing sessions, normalizes them, and writes them idempotently; repeated imports do not produce duplicate events.
- Real-time capture: agent lifecycle hooks write JSON events to Semafuse; when the database is temporarily unavailable, events are written to a local spool and replayed later.

Search projections are stored in `sqlx + SQLite + FTS5`. Parsed raw records are the immutable source of truth; memory strategies compiled into
Semafuse subscribe to new events and maintain rebuildable derived memories in separate namespaces.
Strategy failures neither roll back nor block raw history capture. Binary attachments such as images, audio, and PDFs are stored through
OpenDAL and deduplicated using SHA-256 content addressing. Agent adapters project native records into the same formal event model,
enabling retrieval across agents and sessions.

## Formal Archive Contract

History ingestion has three explicit layers:

1. `RawRecord` retains the complete parsed native JSON record, its source position, digest, and agent without assigning canonical meaning.
2. An agent adapter parses records with its own native history schema and projects them into a versioned `EventEnvelope` whose `data` is a closed tagged union: session lifecycle, message, tool call, tool result, compaction, error, or unknown.
3. Validation either accepts the projection or emits a durable `ConversionDiagnostic`. Unknown native records remain available through raw provenance even when their payload is deliberately excluded from searchable canonical data.

There is intentionally no shared native import record type across adapters. The
shared contract begins at `RawRecord` for provenance and `EventEnvelope` /
`CanonicalArchive` for normalized output; each adapter owns the schema needed to
understand its own persisted history and hook payloads.

SQLite writes raw records, canonical projections, provenance links, and diagnostics in one transaction. Full-text retrieval indexes only the explicit `search_text` projection; it never treats arbitrary native JSON as conversational text. The Rust types are the implementation source of truth, while [the checked-in JSON Schema](schemas/canonical-archive-v1.schema.json) is the language-neutral contract using JSON Schema Draft 2020-12.

```bash
# Print the authoritative schema
semafuse schema

# Retrieve the native source record(s) for a canonical event
semafuse raw evt_0123456789abcdef01234567 --db /data/semafuse.db --json
```

Canonical event JSON uses `spec_version: "semafuse.event/v1"`. The archive schema uses `spec_version: "semafuse.archive/v1"`. Contract changes that are not backward compatible require a new version instead of silently changing these values.

The event model and each agent's native history and real-time hook mappings are documented next
to their Rust implementations. Generate and open the documentation with:

```bash
cargo doc --no-deps --open
```

## Build and Install

Install a prebuilt release with mise:

```bash
# Latest release
mise use -g github:reonokiy/semafuse

# A specific release
mise use -g github:reonokiy/semafuse@0.1.0
```

Because this repository is private, authenticate mise through the GitHub CLI or set
`MISE_GITHUB_TOKEN` to a token with read access to the repository. The release workflow
publishes platform-specific archives and a `SHA256SUMS` file that mise uses to select and
verify the download.

To build and install from source instead:

```bash
cargo install --path . --locked
```

All examples below assume that `semafuse` is installed in `PATH`.

```bash
semafuse --help
semafuse <command> --help
```

## Default Paths and Environment Variables

The default data root is `semafuse` under the operating system's local data directory. On Linux, this is usually
`~/.local/share/semafuse`, though it may be affected by `XDG_DATA_HOME`.

| Item | Default | Override |
| --- | --- | --- |
| SQLite database | `<data-dir>/semafuse.db` | `SEMAFUSE_DB` or each command's `--db` |
| OpenDAL blob storage | `fs`, in `<data-dir>/blobs` | `[blob]` in the configuration file; the default directory can also be overridden with `SEMAFUSE_BLOB_DIR` |
| Hook failure queue | `<data-dir>/spool` | `SEMAFUSE_SPOOL_DIR` or `--spool-dir` |
| Entire data root | `semafuse` under the OS local data directory | `SEMAFUSE_DATA_DIR` |
| Strategy configuration | `<data-dir>/config.toml` | `SEMAFUSE_CONFIG` |

Explicit command arguments take precedence over environment variables. For example:

```bash
SEMAFUSE_DATA_DIR=/data/semafuse semafuse init
semafuse search "authentication decision" --db /data/memory.db
```

## Output Format

By default, all user-facing commands output headings, fields, and lists suitable for terminal reading. The global `--json` argument can
be placed before or after the subcommand to obtain stable, machine-readable JSON:

```bash
semafuse --json doctor
semafuse import --agent codex --dry-run --json
semafuse search "database decision" --json | jq '.results[]'
```

Successful results use a consistent JSON object containing:

- `ok: true`;
- a `command` identifying the result type;
- the command's own structured fields, such as `inserted`, `results`, `events`, or `installed`.

Runtime failures return a nonzero exit code and write JSON containing `ok: false` and `error` to stderr.
If the command-line arguments themselves cannot be parsed, the CLI parser still outputs its standard help text. Progress animations are written only to stderr
and are automatically hidden in non-interactive environments, so they do not contaminate JSON on stdout.

`mcp` is the exception: its stdin/stdout are themselves standard JSON-RPC/MCP streams, so no
Semafuse envelope is added. Generated real-time agent hooks likewise use a separate internal protocol mode; when users invoke
`hook` directly, it still follows the human-readable output/`--json` rules above.

### Working Directory and Repository Fields

Canonical events store the agent's working directory and repository separately under `source`:

| Field | Meaning |
| --- | --- |
| `source.cwd` | The working directory reported by the coding agent. It is preserved whether or not the directory belongs to a Git repository. |
| `source.repository` | The explicit worktree reported by the agent, or the nearest ancestor of `source.cwd` containing `.git`. It is `null` when no worktree is supplied and the working directory is not inside a locally detectable Git repository. |

For example, if `cwd` is `/work/api/src` and `/work/api/.git` exists, the event stores
`cwd: "/work/api/src"` and `repository: "/work/api"`. If `/work/scratch` is not in a Git
repository, it stores `cwd: "/work/scratch"` and `repository: null`. An explicit worktree is
retained even when that path is no longer available locally, which keeps imported or remote
events filterable by their source worktree.

## Quick Start

```bash
# Create or open the database and complete schema migrations
semafuse init

# Idempotently import the local history of all four agents
semafuse import --agent amp
semafuse import --agent codex
semafuse import --agent claude-code
semafuse import --agent opencode

# Search across agents
semafuse search "previous database decision"

# Check whether the database and local agent commands are available
semafuse doctor
```

Valid values for `--agent` are `amp`, `codex`, `claude-code`, and `opencode`.

## Command Reference

### `init`: Initialize the Database

```bash
semafuse init [--db PATH]
```

Creates the database's parent directory, the SQLite database, event tables, and FTS5 indexes. This command can be run repeatedly.

### `import`: Import Historical Sessions Once

```bash
semafuse import --agent AGENT [--source PATH] [--db PATH] [--dry-run]
```

Arguments:

- `--agent`: source agent; required.
- `--source`: explicitly specifies a single JSON/JSONL file or a directory containing such files. Directories are scanned recursively.
- `--db`: overrides the target SQLite database.
- `--dry-run`: parses and reports statistics only, without writing to the database or storing binary attachments.

When `--source` is omitted, each adapter uses its own default source:

| Agent | Default Import Method |
| --- | --- |
| Amp | Recursively reads thread JSON from `amp/threads` under the OS local data directory; on Linux this is usually `~/.local/share/amp/threads` |
| Codex | Reads through the Codex App Server's `thread/list` and `thread/read`; falls back to `$CODEX_HOME/sessions` or `~/.codex/sessions` when unavailable |
| Claude Code | Recursively reads JSON/JSONL history under `~/.claude/projects` |
| OpenCode | Temporarily starts a local `opencode serve` and reads through `/session` and `/session/:id/message` |

A dry run is recommended first to confirm the number of parsed items and errors:

```bash
semafuse import --agent codex --dry-run
semafuse import --agent claude-code --source ~/.claude/projects --dry-run
semafuse import --agent opencode --db /data/semafuse.db
```

Normal import output includes raw-record, canonical-event, diagnostic, deduplication, and error counts. Identical events have stable fingerprints,
so rerunning the same import counts them under `deduplicated` rather than writing duplicates.

In an interactive terminal, the import process displays a progress bar, current progress, and the file or
session being processed on stderr. Final statistics are written to stdout. When output is redirected or the command is invoked by a script, the progress bar is hidden automatically,
preventing terminal control characters from entering machine-readable output.
The progress bar is implemented with `indicatif`, which handles refresh frequency, terminal width, and TTY detection.
A spinner is displayed while enumerating import sources; the read phase displays a spinner and bounded progress bar; after reading finishes,
it switches to a write spinner with the event count while writing to SQLite and updating the full-text index.

### `hook`: Receive a Real-Time Event

```bash
semafuse hook \
  --agent AGENT \
  [--event EVENT] \
  [--db PATH] \
  [--recall] \
  [--spool-dir PATH]
```

This command reads a JSON object from standard input, normalizes it, and writes it to the database. It is primarily invoked by generated agent
hook configurations, but can also be debugged manually:

```bash
printf '%s\n' '{"session_id":"demo","cwd":"/work/repo","prompt":"Why was SQLite chosen?"}' | \
  semafuse hook \
    --agent claude-code \
    --event UserPromptSubmit \
    --recall
```

Arguments:

- `--event`: the agent's native event name; defaults to `auto`, for integrations whose payload includes the event type.
- `--recall`: retrieves relevant history after writing the event. By default, it outputs human-readable match hints; with `--json`,
  the agent-native recall data is in the `recall` field. Generated agent configurations automatically use the internal
  protocol mode to return native hook responses.
- `--spool-dir`: overrides the failure queue directory.

If the database cannot be opened or event processing fails, the hook appends the raw payload to the spool and attempts to exit
successfully, preventing memory-layer failures from blocking the coding agent. Diagnostic information is written to stderr.

### `search`: Retrieve Historical Memories

```bash
semafuse search QUERY \
  [--repository REPOSITORY] \
  [--agent AGENT] \
  [--mode hybrid|lexical|recent] \
  [--since TIME] \
  [--until TIME] \
  [--limit N] \
  [--json] \
  [--db PATH]
```

- `hybrid`: the default basic query mode, combining full-text relevance with recent events.
- `lexical`: uses only SQLite FTS5 full-text relevance.
- `recent`: returns the most recent events matching the filters; query text does not affect ranking.
- `--repository`: searches only events whose normalized `repository` exactly matches the specified Git repository/worktree identifier; it does not filter by `cwd`.
- `--agent`: searches only events from a specific agent.
- `--since`: includes only events at or after the given bound. It accepts relative durations such as `48h`, `2d`, or `1w`, RFC3339 timestamps, or Unix milliseconds.
- `--until`: includes only events at or before the given bound, using the same formats as `--since`.
- `--limit`: defaults to 10; constrained by the program to 1 through 100.
- `--json`: outputs the standard envelope; complete matches are in `results`, including sources, fingerprints, attachments, and scores.

```bash
semafuse search "Cloudflare authentication" --mode lexical --limit 20
semafuse search "deployment" --agent codex --repository /work/api --json
semafuse search "ignored for recent" --mode recent --since 2d
```

`--mode` controls only the basic query method for raw history; it does not select a memory strategy. The built-in `history`
strategy is enabled by default and provides agent hooks with recall consistent with the basic hybrid query. It can be configured in the configuration file:

```toml
[strategies]
enabled = ["history"]
recall = "history"
```

Configuration is loaded by `config-rs`, with precedence ordered from code defaults to the TOML file and then environment variables. Nested environment variables
use double underscores as separators, for example `SEMAFUSE__STRATEGIES__RECALL=history`; lists use commas as separators,
for example `SEMAFUSE__STRATEGIES__ENABLED=history,another`.

Blob storage can use any OpenDAL service enabled at build time. By default, Semafuse enables only
`services-fs`; other backends can be enabled as needed through Cargo features of the same name, for example
`cargo install --path . --features services-s3`. `scheme` is the OpenDAL service name, and key-value pairs under
`config` are passed directly to OpenDAL; see the corresponding service's OpenDAL configuration documentation for specific fields.
When not configured, local `fs` storage is used:

```toml
[blob]
scheme = "s3"

[blob.config]
bucket = "semafuse"
region = "us-east-1"
endpoint = "https://s3.example.com"
access_key_id = "..."
secret_access_key = "..."
```

Nested environment variables can also be used, such as `SEMAFUSE__BLOB__SCHEME=s3` and
`SEMAFUSE__BLOB__CONFIG__BUCKET=semafuse`. Write OpenDAL configuration values in TOML as strings.

General strategy operations commands do not contain specific retrieval algorithms:

```bash
semafuse strategy list
semafuse strategy status
semafuse strategy retry
semafuse strategy rebuild history
```

Each import, hook, or spool replay creates persistent deliveries only for newly added raw events. Each strategy
consumes events in its own checkpoint order and retains a retryable state on failure. `rebuild` reconstructs the target strategy's derived memories
from the complete raw event history without modifying other strategies or raw events.

Memory strategies are organized by file under `src/strategy/`. Shared contracts, the registry, and the event dispatcher are in
`mod.rs`; each concrete strategy uses a separate file, with the built-in strategy in `history.rs`, for example. To add a strategy, implement
`MemoryStrategy`, then register only its name and constructor in the registry.

### `session`: Read a Complete Session

```bash
semafuse session SESSION_ID [--json] [--db PATH]
```

Lists all events in a normalized session in chronological order. Standard output is suitable for terminal reading, while `--json`
is suitable for scripts and includes attachment metadata.

### `hooks print`: Preview Agent Integration Files

```bash
semafuse hooks print \
  --agent AGENT \
  [--binary /absolute/path/to/semafuse] \
  [--db PATH]
```

Prints only the files to be generated and their contents to stdout, without modifying the project. `--binary` should use the absolute path to Semafuse
that the agent can actually execute; when omitted, the path of the currently running executable is used.

Each agent generates:

| Agent | Files |
| --- | --- |
| Amp | `.amp/plugins/semafuse.ts`, `.amp/settings.json` |
| Codex | `.codex/hooks.json`, `.codex/config.toml` |
| Claude Code | `.claude/settings.json`, `.mcp.json` |
| OpenCode | `.opencode/plugins/semafuse.js`, `opencode.json` |

All agent configurations include an entry point for real-time capture; Amp, Codex, and Claude Code also recall relevant history when a new prompt
begins. All supported integrations configure the Semafuse MCP or an equivalent search tool.
When running Amp with `--execute`, also pass `--plugin-ready-timeout` to ensure that the plugin is loaded before the first turn
begins; this argument is not needed in normal interactive mode.

### `hooks install`: Write Agent Integration Files

```bash
semafuse hooks install \
  --agent AGENT \
  --target /path/to/project \
  [--binary /absolute/path/to/semafuse] \
  [--db PATH]
```

Installs the files shown by `hooks print` into the specified project directory. Installation is conflict-safe: if any target
file already exists, the entire operation is rejected, no files are overwritten, and no partial installation is created.

If agent configuration already exists, first run `hooks print`, then manually merge the relevant fragments:

```bash
semafuse hooks print --agent codex \
  --binary "$(command -v semafuse)" \
  --db "$HOME/.local/share/semafuse/semafuse.db"
```

### `spool replay`: Replay Failed Real-Time Events

```bash
semafuse spool replay [--db PATH] [--spool-dir PATH]
```

After the database recovers, reads and reprocesses the hook failure queue. Successfully processed records are not written again
during subsequent replays; the event layer itself also deduplicates by fingerprint.

```bash
semafuse doctor
semafuse spool replay
```

### `mcp`: Start the stdio MCP Service

```bash
semafuse mcp [--db PATH]
```

Starts an MCP server that communicates over stdin/stdout. It is not an HTTP service and is usually started automatically by MCP clients such as Codex and
Claude Code according to the generated configuration. Currently exposed tools include:

- searching historical events;
- reading events by session;
- reading recent events.

Do not run this command directly for an extended period in a normal interactive terminal unless debugging the JSON-RPC/MCP protocol.

### `doctor`: Check Local Status

```bash
semafuse doctor [--db PATH]
```

Opens the database and outputs the current event count while checking whether the `amp`, `codex`, `claude`, and `opencode` commands can be
found in the current `PATH`. It checks only basic local availability and does not indicate that all agent hooks have been installed.

## Recommended Workflow

Initial deployment:

```bash
cargo install --path . --locked
semafuse init
semafuse doctor

semafuse import --agent codex --dry-run
semafuse import --agent codex

semafuse hooks print --agent codex
semafuse hooks install \
  --agent codex \
  --target /path/to/project \
  --binary "$(command -v semafuse)"

semafuse search "previous design decision"
```

During normal operation, agent hooks capture events automatically. After a transient database failure, run:

```bash
semafuse spool replay
```

## Development Verification

The database schema is managed by `migrations/`; unapplied migrations run automatically when the application starts.
When modifying the schema or SQL queries, install the CLI matching the project's `sqlx 0.8`, update the offline query metadata, and then verify:

```bash
cargo install sqlx-cli --version '^0.8' --no-default-features --features sqlite
export DATABASE_URL=sqlite://./target/sqlx-dev.db
cargo sqlx database create
cargo sqlx migrate run
cargo sqlx prepare
```

`.sqlx/` must be committed together with queries; it enables `query!` / `query_as!` to perform compile-time SQL validation
in build environments without a database connection.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo doc --no-deps
cargo build --release
```

Tests cover import and deduplication for all four agents, cross-agent hook recall, OpenDAL image storage, spool
replay, the MCP protocol, modular database/retrieval contracts, and integration-file installation safeguards.
