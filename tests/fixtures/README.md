# History fixtures

The `*-complete` fixtures are synthetic, privacy-safe transcripts modeled on real
agent histories rather than copies of user conversations. They retain native IDs,
timestamps, metadata nesting, tool lifecycle fields, compaction records, and
otherwise-unrecognized records so adapter tests exercise realistic files.

The shapes were checked on 2026-07-18 against local Codex 0.144.4 rollouts,
Claude transcript JSONL, OpenCode 1.18.3 SQLite/API records, and Amp thread JSON.
Claude project-session JSONL and the other public contracts were cross-checked
against:

- <https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md>
- <https://github.com/anthropics/claude-code/issues/24304>
- <https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/message-v2.ts>
- <https://ampcode.com/manual>

Do not replace these files with raw local histories. Add a redacted structural
case when a native format gains a field or event variant.
