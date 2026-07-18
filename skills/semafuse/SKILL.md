---
name: semafuse
description: Uses Semafuse to search and inspect local coding-agent conversation memory, recover prior decisions and sessions, import agent history, diagnose storage, and configure live integrations. Use when a task depends on previous Amp, Codex, Claude Code, or OpenCode conversations, or when asked to set up or operate Semafuse.
license: MIT
compatibility: Requires the semafuse executable on PATH; jq is optional for inspecting CLI JSON.
metadata:
  author: semafuse
  version: "1.0"
---

# Using Semafuse CLI and MCP

Semafuse is a local, agent-neutral memory layer for coding-agent conversations. Use it to retrieve prior context with provenance instead of guessing what happened in earlier sessions.

## Load only what the task needs

- For searching, recent activity, full sessions, CLI output, filters, and paths, read [references/retrieval.md](references/retrieval.md).
- For installation, initialization, imports, live agent integration, diagnosis, spool recovery, or strategy maintenance, read [references/operations.md](references/operations.md).
- Do not read an operations reference for an ordinary memory lookup.

## Choose the interface

1. Prefer available Semafuse MCP tools:
   - `semafuse_recall` for general memory recall through the configured strategy.
   - `semafuse_search` for raw history, exact terms, provenance, or query modes.
   - `semafuse_session` for all events from a known session.
   - `semafuse_recent` for recent events.
2. Otherwise, verify `command -v semafuse`, read the relevant reference, and use the CLI with `--json`.
3. If neither interface exists, state that Semafuse is unavailable. Do not install it unless the user requests setup.

## Retrieval workflow

Use this workflow when prior agent context can materially affect the task:

1. Search with a short phrase containing distinctive technical terms.
2. Restrict to the current repository when known.
3. Start with a small result limit and the default hybrid search.
4. If no useful result appears, vary terms, use lexical search for exact evidence, or remove the repository restriction when cross-project context is plausible.
5. When a hit affects the answer, inspect its full session before treating a snippet as the final decision.
6. Summarize only the relevant context and retain provenance such as source agent and session ID.

## Guardrails

- Do not query Semafuse for unrelated tasks merely because it is available; use it when prior agent context could materially affect the result.
- Do not expose entire stored conversations when a focused summary answers the request. Conversation history may contain sensitive local data.
- Do not edit the SQLite database, spool files, or blob storage directly. Use Semafuse commands.
- Only perform setup or data-changing operations when the user asks for them.
- Do not run `semafuse mcp` interactively as a normal query command; it is a long-lived stdio JSON-RPC server intended for MCP clients.
- If command syntax is uncertain, run `semafuse <command> --help` rather than inventing flags.
