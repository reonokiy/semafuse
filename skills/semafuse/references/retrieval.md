# Retrieval reference

Read this file only when using the Semafuse CLI for memory retrieval or when exact search behavior, output, filters, or storage paths are needed.

## Machine-readable output

Pass global `--json` for agent-driven CLI calls. Successful commands emit a JSON object containing `ok: true` and a command-specific field such as `results` or `events`. Runtime failures return a non-zero exit code and emit an error object to stderr.

## Search history

```bash
semafuse search "authentication token rotation" \
  --repository "$PWD" \
  --limit 10 \
  --json
```

Options:

- `--repository REPOSITORY`: restrict to a repository/worktree identifier. Retry without it only when cross-project context is plausible.
- `--agent amp|codex|claude-code|opencode`: restrict to one source agent.
- `--mode hybrid|lexical|recent`: `hybrid` is the default; `lexical` uses only FTS relevance; `recent` sorts by recency and ignores query text for ranking.
- `--limit N`: result limit, clamped to 1 through 100.
- `--db PATH`: use a non-default SQLite database.

Prefer distinctive technical phrases over long natural-language questions. Search results include source event data, agent, session ID, score, fingerprint, and attachment metadata.

## Inspect a full session

```bash
semafuse session SESSION_ID --json
```

This returns normalized events in chronological order. Read the session when surrounding turns may qualify or reverse a search snippet.

## Inspect recent events

Prefer MCP `semafuse_recent` when available. The CLI equivalent is:

```bash
semafuse search "recent" --mode recent --repository "$PWD" --limit 20 --json
```

The placeholder query is required syntactically but does not affect ranking in `recent` mode.

## Paths and overrides

Semafuse normally uses the OS local data directory. On Linux this is commonly `~/.local/share/semafuse`.

- Database: `SEMAFUSE_DB` or `--db PATH`
- Entire data root: `SEMAFUSE_DATA_DIR`
- Hook failure queue: `SEMAFUSE_SPOOL_DIR` or `--spool-dir PATH`
- Blob directory: `SEMAFUSE_BLOB_DIR`
- Configuration file: `SEMAFUSE_CONFIG`

Prefer the default shared database unless the user provides another path. For a custom database, pass the same `--db` consistently to retrieval and operations commands.
