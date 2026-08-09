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

Install a prebuilt release with mise after a matching GitHub release has been published:

```bash
# Latest release
mise use -g github:reonokiy/semafuse

# A specific release
mise use -g github:reonokiy/semafuse@0.1.0
```

Because this repository is private, authenticate mise through the GitHub CLI or set
`MISE_GITHUB_TOKEN` to a token with read access to the repository. The release workflow
accepts only a `v<version>` tag matching the Rust package, reruns the same Rust and Web
quality checks used for pushes and pull requests, and smoke-tests every supported binary.
Each Linux/macOS archive contains `semafuse`, a models-candle-enabled `semafused`, the
daemon example config, and the license; the Windows archive contains `semafuse.exe` and
the license because the daemon transport is Unix-only. Mise exposes the executables found
in the selected archive and verifies it using the published `SHA256SUMS.txt`.

To build and install from source instead:

```bash
cargo install --path . --locked
cargo install --path . --locked --bin semafused --no-default-features --features models-candle
```

Rust, Node.js, and pnpm are pinned in `rust-toolchain.toml`, `mise.toml`, and
`web/package.json`; `mise install` installs those managed tools. Native Rust dependencies
also require a C/C++ compiler and `pkg-config` (Xcode Command Line Tools on macOS, or the
distribution build toolchain on Linux). On Unix, reproduce CI locally with:

```bash
mise install

cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --no-default-features --features daemon -- -D warnings
cargo clippy --locked --all-targets --no-default-features --features models-candle -- -D warnings
cargo test --locked --all-targets
cargo test --locked --all-targets --features daemon
cargo test --locked --all-targets --features models-candle
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --no-default-features --features models-candle

pnpm --dir web install --frozen-lockfile
pnpm --dir web check
pnpm --dir web build
```

All examples below assume that `semafuse` is installed in `PATH`.

```bash
semafuse --help
semafuse <command> --help
```

## Process Boundary

`semafuse` is the one-shot command. Imports, hooks, searches, maintenance,
strategy operations, benchmark requests, and `daemon status` connect to their
dependencies and then exit. The stdio MCP command is the explicit exception
because its protocol owns stdin/stdout for the lifetime of one client session.

`semafused` is a separate, all-Rust foreground process for services that
benefit from staying warm. The first built-in service hosts local embedding and
reranking models, while its versioned service envelope and status registry allow
other long-lived services to be added later. It has no daemonization,
`start`, `stop`, or `restart` command; systemd or another supervisor owns
its lifecycle.

The model service selects its Candle device automatically: CUDA when compiled
with `models-cuda`, Metal on macOS, and CPU otherwise. macOS builds include an
Accelerate CPU fallback; explicit `cpu`, `cuda`, and `metal` configuration is
also available when startup must fail rather than fall back.

Official `semafused` builds target Linux and macOS. Windows keeps the one-shot
`semafuse` CLI, but its Unix-socket daemon client returns an unsupported error.

The two processes communicate through a same-user Unix-domain socket. The
`semafuse` client never starts `semafused`, never removes its socket, and
never falls back from a failed local request to HTTP. Install the optional
models-enabled daemon separately:

```bash
cargo install --path . --locked --bin semafused --no-default-features --features models-candle
```

See [the semafused guide](docs/semafused.md) for strict local model layout,
configuration, foreground execution, and supervisor examples. Inspect an
already-running process without changing it with:

```bash
semafuse daemon status --socket "$XDG_RUNTIME_DIR/semafuse/semafused.sock"
```

## License

Semafuse is available under the [MIT License](LICENSE).

## Default Paths and Environment Variables

The default data root is `semafuse` under the operating system's local data directory. On Linux, this is usually
`~/.local/share/semafuse`, though it may be affected by `XDG_DATA_HOME`.

| Item | Default | Override |
| --- | --- | --- |
| SQLite database and location identity | `<data-dir>/semafuse.db` and `<data-dir>/semafuse.db.location-id` | `SEMAFUSE_DB` or each command's `--db` |
| OpenDAL blob storage | `fs`, in `<data-dir>/blobs` | `[blob]` in the configuration file; the default directory can also be overridden with `SEMAFUSE_BLOB_DIR` |
| Hook failure queue | `<data-dir>/spool` | `SEMAFUSE_SPOOL_DIR` or `--spool-dir` |
| Entire data root | `semafuse` under the OS local data directory | `SEMAFUSE_DATA_DIR` |
| Strategy configuration | `<data-dir>/config.toml` | `SEMAFUSE_CONFIG` |

Explicit command arguments take precedence over environment variables. For example:

```bash
SEMAFUSE_DATA_DIR=/data/semafuse semafuse init
semafuse search "authentication decision" --db /data/memory.db
```

## Data Safety and Lifecycle

By default, Semafuse stores SQLite records, native raw payloads, and attachments as plaintext
on the local machine. On Unix, newly created database/spool files use mode `0600` and newly
created spool/blob roots use `0700`. Semafuse does not silently change permissions on an
existing user-selected directory. Full-disk encryption is still recommended for sensitive
histories; application-level database encryption and automatic secret redaction are not
currently provided.

The governance commands are deliberately explicit:

```bash
# Export JSON and checksum-verified attachments; OUTPUT must be a new directory
semafuse export --output OUTPUT

# Preview matching deletions, then repeat with --confirm to apply them
semafuse forget --session SESSION_ID
semafuse forget --session SESSION_ID --confirm

# Include every non-default hook spool location that has been used; repeat as needed
semafuse forget --session SESSION_ID --confirm \
  --spool-dir /path/to/project-a-spool \
  --spool-dir /path/to/project-b-spool

# Retry queued attachment deletion and SQLite/WAL cleanup
semafuse gc
```

`forget` accepts bounded session, repository, agent, and/or `--before` selectors. A preview is
read from one SQLite snapshot. A confirmed operation records hashed deletion rules, removes
matching canonical events and explicitly owned raw records, rebuilds derived memories, and
prevents the same data from returning through a later history import or spool replay. The
deletion is closed over the provenance graph: if one native record produced events on both
sides of a time cutoff, every projection of that record is included in the preview and removed.
This avoids retaining either the sensitive native payload or an event with broken provenance. The
configured default hook spool is always previewed and cleaned; every custom `--spool-dir` used
by an integration must also be supplied, and the option may be repeated. Live, pending, and
claimed replay generations are filtered under the same append/replay locks used by hooks. A
missing spool directory is ignored without creating it, while a locked, malformed, or otherwise
unreadable spool makes confirmed cleanup incomplete and the command exits nonzero. SQLite
uses `secure_delete`; after deletion Semafuse rebuilds FTS indexes, truncates the WAL, and vacuums
the database. If a reader temporarily blocks that scrub, Semafuse records a durable pending marker,
continues spool, attachment, and derived-memory cleanup, then exits nonzero. Repeating the same
confirmed forget command or running `semafuse gc` retries the physical scrub.

New attachment keys live in a database-specific namespace. Writes are registered before blob
I/O, committed events maintain a normalized reference index, and GC first claims each deletion;
new writes and references are rejected while that claim is active. The cross-process lifecycle
lock is derived from the canonical database path, so a symlink cannot create a second lock.
Independently created databases therefore cannot delete each other's keys. The database is bound
to both its canonical path and the adjacent `.location-id` sidecar, so copying only the SQLite file
to another path—or to the same path on another host—is rejected before it can reuse the original
namespace. Preserve the sidecar when restoring an offline backup at its original location.
This guard is not a distributed lease: deliberately copying both the database and its sidecar
cannot be distinguished from a restore, so never run those copies independently or concurrently.
Use `semafuse export` for portable copies. Legacy
unnamespaced attachment keys are logically forgotten but deliberately not deleted; the command
records them as auditable residue and reports cleanup as incomplete instead of risking another
store's data. `semafuse doctor` reports the accumulated residue count.

Physical erasure can only cover the active SQLite database/WAL and the configured non-versioned
blob object. Filesystem snapshots, backups, SSD wear-leveling, replicated or versioned object
stores, and previously exported archives must be governed separately. Hook payloads and history
imports never open attachment references that contain local filesystem paths. Binary attachments
must be embedded as supported base64 bodies or `data:` URLs.

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
An import with unreadable sources or error-severity record diagnostics instead emits its complete
report on stdout with `ok: false`, `complete: false`, `failed_records`, and `source_errors`, then
exits nonzero. Valid records from a partial import may already have been committed, so those
counters are the authoritative result rather than a blanket success signal.
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
- `--recall`: retrieves relevant history before writing the current event, so the prompt cannot recall itself. By default, it outputs human-readable match hints; with `--json`,
  the agent-native recall data is in the `recall` field. Generated agent configurations automatically use the internal
  protocol mode to return native hook responses.
- `--spool-dir`: overrides the failure queue directory.

Every hook durably stages its raw payload before processing. The write-ahead record is removed on success, discarded when a durable forget rule matches, and otherwise remains for replay after a failure. The hook attempts to exit
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

`--mode` controls only the basic query method for raw history; it does not select a memory strategy. Memory strategies are independent modules behind a shared recall contract:

| Strategy | Construction | Recall output | Retrieval | Intended use |
| --- | --- | --- | --- | --- |
| `history` | None; reads immutable events directly | `event` items | Configurable `hybrid`, `lexical`, or `recent` | Cheap control/baseline |
| `lazymem` | Paper-v2 query-time MPM construction | One chronological `memory` item with source IDs and a redacted protocol trace | Qwen3 dense + BM25, RRF, BGE cross-encoder, Top-50, radius-2 windows | Full model-backed LazyMem path and paper protocol |

`lazymem` implements the complete model-backed structure: raw message storage,
durable event embeddings, parallel dense/BM25 retrieval, RRF, cross-encoder
reranking, Top-50 radius-2 history restoration, transitive span merging, 8/7
overlapping MPM windows, strict positional KEEP/DROP JSON, KEEP-wins overlap
resolution, and chronological construction. The MPM request uses the complete
Appendix G.1 `previous_bridge`/`core_messages`/`next_bridge` prompt. Its response
must be the positional JSON array, either directly or after exactly one closed
leading `<think>...</think>` block; prose, fences, nested/multiple thinking
blocks, unknown fields, and count mismatches fail closed.

`paper-v2` locks the paper equation's anchor-gap boundary (`<= 2w`), while
`public-repro` locks the released repository implementation (`<= 2w + 1`). Both
also lock dense/sparse/fused limits `100/100/100`, rerank Top-50, radius 2,
window length/stride `8/7`, and MPM generation at temperature `0.1`, top-p
`0.95`, 3,072 tokens, 64 concurrent windows, with thinking enabled. These named
profiles require the `openai_compatible` processor so they cannot accidentally
label a coding-agent run as paper-comparable. Use `protocol = "custom"` for a
Codex processor or another explicit ablation. Embeddings use an OpenAI-compatible
endpoint or the local Unix-socket semafused service; reranking likewise uses an
explicit TEI/Cohere-compatible or semafused backend. There is no automatic
fallback between local and remote transports, and credentials are referenced
only by environment-variable name.

To route through the full path with Codex itself as the prompt-only memory
processor, configure the shared embedding/reranker services and the isolated
`codex exec` backend, then rebuild the vector namespace once for existing
events:

```toml
[strategies]
enabled = ["history", "lazymem"]
recall = "lazymem"

[strategies.settings.lazymem]
protocol = "custom"
embedding_batch_size = 32

[strategies.settings.lazymem.embedding]
backend = "openai_compatible"

[strategies.settings.lazymem.embedding.endpoint]
base_url = "http://127.0.0.1:8001/v1"
model = "Qwen/Qwen3-Embedding-8B"
api_key_env = "LAZYMEM_EMBED_API_KEY"

[strategies.settings.lazymem.reranker]
backend = "openai_compatible"

[strategies.settings.lazymem.reranker.endpoint]
base_url = "http://127.0.0.1:8002"
model = "BAAI/bge-reranker-v2-m3"
protocol = "tei"
api_key_env = "LAZYMEM_RERANK_API_KEY"

[strategies.settings.lazymem.processor]
backend = "codex_cli"
# Choose a model listed by `codex debug models`; `gpt-5.6-sol` is the
# validated Codex CLI 0.145.0 choice.
model = "gpt-5.6-sol"
command = "codex"
timeout_secs = 600
max_output_bytes = 1048576

[strategies.settings.lazymem.generation]
max_concurrency = 1
```

Each window gets a fresh, ephemeral Codex process in a private temporary
directory. Semafuse passes the prompt on stdin, requires a structured result,
runs read-only, disables hooks/tools/web/memory features, bounds all output and
wall time, and removes the temporary files. The CLI must already be installed
and authenticated. Codex's own instructions mean this backend is a practical
local ablation, not the paper's exact MPM environment; its sampling fields are
recorded but not claimed to control the Codex model. The embedding and reranker
services are still required. Recall metadata records the processor transport
version; preserve the output of `codex --version` with benchmark runs as well.

For the paper's training-free `LazyMem + Qwen3-32B` route, keep the same shared
settings, switch `protocol` to `paper-v2`, and replace the processor tables with:

```toml
[strategies.settings.lazymem.processor]
backend = "openai_compatible"

[strategies.settings.lazymem.processor.endpoint]
base_url = "http://127.0.0.1:8003/v1"
model = "Qwen/Qwen3-32B"
api_key_env = "LAZYMEM_MPM_API_KEY"
```

```bash
semafuse strategy rebuild lazymem
```

The local vector repository uses exact cosine ranking so deletion governance
and deterministic correctness remain inside the same SQLite transaction domain;
forgetting an event cascades to its vector. For very large installations, place
the same `VectorRepository` contract behind a mature ANN service rather than
weakening lifecycle isolation. Recall fails closed when the strategy checkpoint
lags the event high-water mark, when the selected model's vectors do not exactly
cover the scoped message corpus, or when a namespace/model changes vector
dimension. Rebuild after changing models or indexed data. Recall metadata keeps
IDs, ranks, selected operations, configuration, and index consistency fields,
but redacts source content, compressed content, and model reasons from the
processing trace so dropped history is not re-exposed. The prompt-only paper
benchmark harness and real-data validation record live under
[`benchmarks/lazymem`](benchmarks/lazymem/README.md); no SFT/GRPO training stack
is included.

`enabled` controls event delivery and lifecycle operations; `recall` selects the one module serving hooks and `semafuse_recall`. Each module owns and strictly validates only the object under `strategies.settings.<name>`, so adding a strategy does not add branches to hooks, MCP, or the CLI.

Configuration is loaded by `config-rs`, with precedence ordered from code defaults to the TOML file and then environment variables. Nested environment variables
use double underscores as separators, for example `SEMAFUSE__STRATEGIES__RECALL=lazymem`; lists use commas as separators,
for example `SEMAFUSE__STRATEGIES__ENABLED=history,lazymem`.

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
semafuse strategy rebuild lazymem
```

Each import, hook, or spool replay creates persistent deliveries only for newly added raw events. Each strategy
consumes events in its own checkpoint order and retains a retryable state on failure. `rebuild` reconstructs the target strategy's derived memories
from the complete raw event history without modifying other strategies or raw events.

Memory strategies are organized as peers under `src/strategy/`. `mod.rs` is the stable façade for dispatch, recall, and rebuild; `registry.rs` contains the ordered module registry; `recall.rs` defines the strategy-neutral `event | memory` output; and each strategy owns its constructor and private typed settings. The model-backed `lazymem` strategy lives in `lazymem.rs`, with retrieval, windowing, processing, profile validation, and prompt construction isolated in `lazymem/` submodules. Model I/O is similarly separated behind the small `inference.rs` façade into HTTP transport, OpenAI-compatible inference, reranking, processor dispatch, and isolated Codex execution. To add a strategy, implement `MemoryStrategy` and add its registration to the registry. Callers remain unchanged.

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

Opens the database and reports the event count, pending attachment GC, auditable unmanaged
attachment residue, and whether the `amp`, `codex`, `claude`, and `opencode` commands can be found
in the current `PATH`. It checks only basic local availability and does not indicate that all agent
hooks have been installed.

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

Run the Unix package and models-candle gates in [Build and Install](#build-and-install).
After they pass, build the same two Rust executables shipped on Unix:

```bash
cargo build --locked --release -p semafuse --bin semafuse
cargo build --locked --release -p semafuse --bin semafused --no-default-features --features models-candle
```

Tests cover import and deduplication for all four agents, cross-agent hook recall, OpenDAL image storage, spool
replay, the MCP protocol, modular database/retrieval contracts, and integration-file installation safeguards.
