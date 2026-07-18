# Operations reference

Read this file only for Semafuse installation, initialization, imports, live integration, diagnosis, recovery, or strategy maintenance.

## Install and initialize

When this repository is available locally, install from its root only when the user requests installation:

```bash
cargo install --path . --locked
semafuse init --json
semafuse doctor --json
```

`doctor` reports the database, event count, and locally available agent commands. It does not prove hooks are installed.

## Import history

Supported agents are `amp`, `codex`, `claude-code`, and `opencode`. Dry-run an unfamiliar import before writing:

```bash
semafuse import --agent codex --dry-run --json
semafuse import --agent codex --json
```

Imports are idempotent: repeated events are reported as `deduplicated`. Add `--source PATH` only for a specific JSON/JSONL file or recursively scanned directory; otherwise Semafuse discovers the agent's default local history.

## Configure live integration

Preview generated files before installing them:

```bash
semafuse hooks print --agent codex \
  --binary "$(command -v semafuse)" \
  --json

semafuse hooks install --agent codex \
  --target /absolute/path/to/project \
  --binary "$(command -v semafuse)" \
  --json
```

`hooks install` writes hook and MCP configuration into the target project, so obtain approval first. It refuses the entire operation when any target file exists. In that case, use `hooks print` and propose a manual merge; do not delete or overwrite existing configuration.

## Diagnose and recover

```bash
semafuse doctor --json
semafuse spool replay --json
semafuse strategy status --json
semafuse strategy retry --json
```

- `spool replay` reprocesses hook payloads saved while the database was unavailable.
- `strategy status` shows delivery checkpoints, attempts, and failures.
- `strategy retry [NAME]` retries pending events for all strategies or one named strategy.

## Rebuild derived memory

```bash
semafuse strategy rebuild NAME --json
```

Rebuild deletes and reconstructs one strategy's derived memory from immutable raw events. It does not modify other strategies or raw history, but it is still a destructive maintenance action; obtain explicit user approval first.

For database and configuration overrides, read [retrieval.md](retrieval.md#paths-and-overrides) only if a non-default path is involved.
