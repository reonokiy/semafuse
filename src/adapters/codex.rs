use super::{
    AgentAdapter, ImportContext, ImportedEvent, event_metadata, recall_text, scalar_string,
    text_content, timestamp_ms, value_text,
};
use crate::model::{AgentKind, CanonicalEvent, EventKind, SearchHit};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

/// Normalizes Codex history records and real-time hook events.
///
/// # Real-time hooks
///
/// Event names are matched case-insensitively with CamelCase and snake_case
/// aliases.
///
/// | Native event | [`EventKind`] | Role | Content | Tool name |
/// | --- | --- | --- | --- | --- |
/// | `SessionStart` | [`SessionStart`](EventKind::SessionStart) | — | Empty | — |
/// | `SessionEnd` | [`SessionEnd`](EventKind::SessionEnd) | — | `reason` | — |
/// | `UserPromptSubmit` | [`UserMessage`](EventKind::UserMessage) | `user` | `prompt` | — |
/// | `PreToolUse` | [`ToolCall`](EventKind::ToolCall) | `assistant` | `tool_input` as text or JSON | `tool_name` |
/// | `PostToolUse` | [`ToolResult`](EventKind::ToolResult) | `tool` | `tool_response`, then `tool_output` | `tool_name` |
/// | `PostToolUseFailure` | [`Error`](EventKind::Error) | `tool` | `error`, then `tool_response` | `tool_name` |
/// | `Stop` | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | `last_assistant_message` | — |
/// | `StopFailure` | [`Error`](EventKind::Error) | `assistant` | `error` | — |
/// | `PreCompact` / `PostCompact` | [`Compaction`](EventKind::Compaction) | — | `compact_summary`, then `custom_instructions` | — |
/// | Other | [`Other`](EventKind::Other) | — | Complete payload JSON | — |
///
/// `session_id` uses `session_id`, or `codex-unknown`; `turn_id` uses `turn_id`;
/// `native_id` tries `tool_use_id`, `agent_id`, and `id`. The generated hook
/// configuration registers `SessionStart`, `UserPromptSubmit`, `PreToolUse`,
/// `PostToolUse`, `Stop`, `PreCompact`, and `PostCompact`. Other aliases remain
/// accepted for manual input and protocol evolution.
///
/// # History import
///
/// | Native item `type` | [`EventKind`] | Default role | Content | Tool name |
/// | --- | --- | --- | --- | --- |
/// | `message` | User role: [`UserMessage`](EventKind::UserMessage); otherwise: [`AssistantMessage`](EventKind::AssistantMessage) | Native role | `content` text | — |
/// | `userMessage` | [`UserMessage`](EventKind::UserMessage) | `user` | `content` text | — |
/// | `agentMessage` / `plan` | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | `text` | — |
/// | `function_call` | [`ToolCall`](EventKind::ToolCall) | `assistant` | `arguments` as text or JSON | `name` |
/// | `custom_tool_call` | [`ToolCall`](EventKind::ToolCall) | `assistant` | `input`, falling back to `arguments`, as text or JSON | `name` |
/// | `function_call_output` / `custom_tool_call_output` | [`ToolResult`](EventKind::ToolResult) | `tool` | `output` text | — |
/// | `commandExecution` | [`ToolResult`](EventKind::ToolResult) | `tool` | Command plus `aggregatedOutput` | `commandExecution` |
/// | `mcpToolCall` | [`ToolResult`](EventKind::ToolResult) | `tool` | Serialized `result`, `error`, or `arguments` | `{server}/{tool}` |
/// | `dynamicToolCall` / `collabToolCall` | [`ToolResult`](EventKind::ToolResult) | `tool` | Complete payload JSON | `tool` |
/// | `fileChange` | [`ToolResult`](EventKind::ToolResult) | `tool` | Serialized `changes` | `fileChange` |
/// | `contextCompaction` | [`Compaction`](EventKind::Compaction) | — | Complete payload JSON | — |
/// | `imageView` | [`ToolResult`](EventKind::ToolResult) | `tool` | `path` | `imageView` |
/// | Other | [`Other`](EventKind::Other) | Native role | Complete payload JSON | — |
///
/// Empty mapped content is dropped except for tool calls and message records
/// that may carry attachments. `native_id` prefers `id`, then `call_id`.
/// Rollout `session_meta` records produce an empty [`EventKind::SessionStart`];
/// `turn_context` only updates the turn ID for subsequent events.
pub struct CodexAdapter;

fn event_from_item(
    item: &Value,
    session: &str,
    turn: Option<String>,
    cwd: Option<String>,
    raw_record_id: &str,
) -> Option<ImportedEvent> {
    let payload = item.get("payload").unwrap_or(item);
    let item_type = payload
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| item.get("type").and_then(Value::as_str))
        .unwrap_or_default();
    let role = payload
        .get("role")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let (kind, content, tool) = match item_type {
        "message" => {
            let content = text_content(payload.get("content").unwrap_or(&Value::Null));
            let kind = if role.as_deref() == Some("user") {
                EventKind::UserMessage
            } else {
                EventKind::AssistantMessage
            };
            (kind, content, None)
        }
        "userMessage" => (
            EventKind::UserMessage,
            text_content(payload.get("content").unwrap_or(&Value::Null)),
            None,
        ),
        "agentMessage" | "plan" => (
            EventKind::AssistantMessage,
            scalar_string(payload.get("text")).unwrap_or_default(),
            None,
        ),
        "function_call" => (
            EventKind::ToolCall,
            value_text(payload.get("arguments")),
            scalar_string(payload.get("name")),
        ),
        "custom_tool_call" => (
            EventKind::ToolCall,
            value_text(payload.get("input").or_else(|| payload.get("arguments"))),
            scalar_string(payload.get("name")),
        ),
        "function_call_output" | "custom_tool_call_output" => (
            EventKind::ToolResult,
            value_text(payload.get("output")),
            None,
        ),
        "commandExecution" => (
            EventKind::ToolResult,
            format!(
                "$ {}\n{}",
                scalar_string(payload.get("command")).unwrap_or_default(),
                scalar_string(payload.get("aggregatedOutput")).unwrap_or_default()
            ),
            Some("commandExecution".into()),
        ),
        "mcpToolCall" => (
            EventKind::ToolResult,
            payload
                .get("result")
                .or_else(|| payload.get("error"))
                .or_else(|| payload.get("arguments"))
                .map(Value::to_string)
                .unwrap_or_default(),
            Some(format!(
                "{}/{}",
                scalar_string(payload.get("server")).unwrap_or_default(),
                scalar_string(payload.get("tool")).unwrap_or_default()
            )),
        ),
        "dynamicToolCall" | "collabToolCall" => (
            EventKind::ToolResult,
            payload.to_string(),
            scalar_string(payload.get("tool")),
        ),
        "fileChange" => (
            EventKind::ToolResult,
            payload
                .get("changes")
                .map(Value::to_string)
                .unwrap_or_default(),
            Some("fileChange".into()),
        ),
        "contextCompaction" | "compacted" => (EventKind::Compaction, payload.to_string(), None),
        "imageView" => (
            EventKind::ToolResult,
            scalar_string(payload.get("path")).unwrap_or_default(),
            Some("imageView".into()),
        ),
        _ => (EventKind::Other, payload.to_string(), None),
    };
    if content.is_empty()
        && !matches!(kind, EventKind::ToolCall)
        && !matches!(item_type, "message" | "userMessage")
    {
        return None;
    }
    let canonical_role = role.or_else(|| match kind {
        EventKind::UserMessage => Some("user".into()),
        EventKind::AssistantMessage => Some("assistant".into()),
        EventKind::ToolCall => Some("assistant".into()),
        EventKind::ToolResult => Some("tool".into()),
        _ => None,
    });
    let attachment_source = match item_type {
        "message" | "userMessage" => payload.get("content").unwrap_or(&Value::Null),
        "function_call" => payload.get("arguments").unwrap_or(&Value::Null),
        "custom_tool_call" => payload
            .get("input")
            .or_else(|| payload.get("arguments"))
            .unwrap_or(&Value::Null),
        "function_call_output" | "custom_tool_call_output" => {
            payload.get("output").unwrap_or(&Value::Null)
        }
        "commandExecution" => payload.get("aggregatedOutput").unwrap_or(&Value::Null),
        "mcpToolCall" => payload
            .get("result")
            .or_else(|| payload.get("error"))
            .unwrap_or(&Value::Null),
        _ => payload,
    };
    Some(ImportedEvent::new(
        CanonicalEvent::new(
            AgentKind::Codex,
            session,
            turn,
            scalar_string(payload.get("id")).or_else(|| scalar_string(payload.get("call_id"))),
            kind,
            canonical_role,
            content,
            tool,
            cwd,
            None,
            timestamp_ms(payload).or_else(|| timestamp_ms(item)),
            event_metadata(item_type),
        ),
        attachment_source,
        raw_record_id,
    ))
}

impl AgentAdapter for CodexAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Codex
    }

    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ImportedEvent>> {
        // Supports Codex rollout JSONL and app-server thread/read responses.
        if values.len() == 1
            && let Some(thread) = values[0]
                .get("result")
                .and_then(|v| v.get("thread"))
                .or_else(|| values[0].get("thread"))
        {
            let session = scalar_string(thread.get("id"))
                .unwrap_or_else(|| context.fallback_session_id.clone());
            let cwd = scalar_string(thread.get("cwd")).or_else(|| context.cwd.clone());
            let mut out = Vec::new();
            for turn_value in thread
                .get("turns")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let turn = scalar_string(turn_value.get("id"));
                for item in turn_value
                    .get("items")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(event) = event_from_item(
                        item,
                        &session,
                        turn.clone(),
                        cwd.clone(),
                        context.raw_record_id(0),
                    ) {
                        out.push(event);
                    }
                }
            }
            return Ok(out);
        }
        let mut session = context.fallback_session_id.clone();
        let mut cwd = context.cwd.clone();
        let mut turn = None;
        let mut out = Vec::new();
        for (record_index, record) in values.iter().enumerate() {
            let record_type = record
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let payload = record.get("payload").unwrap_or(record);
            match record_type {
                "session_meta" => {
                    session = scalar_string(payload.get("id")).unwrap_or(session);
                    cwd = scalar_string(payload.get("cwd")).or(cwd);
                    out.push(ImportedEvent::new(
                        CanonicalEvent::new(
                            AgentKind::Codex,
                            &session,
                            None,
                            None,
                            EventKind::SessionStart,
                            None,
                            "",
                            None,
                            cwd.clone(),
                            None,
                            timestamp_ms(payload).or_else(|| timestamp_ms(record)),
                            event_metadata("session_meta"),
                        ),
                        payload,
                        context.raw_record_id(record_index),
                    ));
                }
                "turn_context" => {
                    turn = scalar_string(payload.get("turn_id"))
                        .or_else(|| scalar_string(payload.get("id")))
                }
                "response_item" => {
                    if let Some(event) = event_from_item(
                        record,
                        &session,
                        turn.clone(),
                        cwd.clone(),
                        context.raw_record_id(record_index),
                    ) {
                        out.push(event);
                    }
                }
                _ => {
                    if let Some(event) = event_from_item(
                        record,
                        &session,
                        turn.clone(),
                        cwd.clone(),
                        context.raw_record_id(record_index),
                    ) {
                        out.push(event);
                    }
                }
            }
        }
        Ok(out)
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let session = scalar_string(p.get("session_id")).unwrap_or_else(|| "codex-unknown".into());
        let turn = scalar_string(p.get("turn_id"));
        let cwd = scalar_string(p.get("cwd"));
        let mut native_id = scalar_string(p.get("tool_use_id"))
            .or_else(|| scalar_string(p.get("agent_id")))
            .or_else(|| scalar_string(p.get("id")));
        let normalized_event = event_name.to_ascii_lowercase();
        let (kind, role, content, tool) = match normalized_event.as_str() {
            "sessionstart" | "session_start" => {
                (EventKind::SessionStart, None, String::new(), None)
            }
            "sessionend" | "session_end" => (
                EventKind::SessionEnd,
                None,
                scalar_string(p.get("reason")).unwrap_or_default(),
                None,
            ),
            "userpromptsubmit" | "user_prompt_submit" => (
                EventKind::UserMessage,
                Some("user".into()),
                scalar_string(p.get("prompt")).unwrap_or_default(),
                None,
            ),
            "pretooluse" | "pre_tool_use" => (
                EventKind::ToolCall,
                Some("assistant".into()),
                value_text(p.get("tool_input")),
                scalar_string(p.get("tool_name")),
            ),
            "posttooluse" | "post_tool_use" => (
                EventKind::ToolResult,
                Some("tool".into()),
                value_text(p.get("tool_response").or_else(|| p.get("tool_output"))),
                scalar_string(p.get("tool_name")),
            ),
            "posttoolusefailure" | "post_tool_use_failure" => (
                EventKind::Error,
                Some("tool".into()),
                value_text(p.get("error").or_else(|| p.get("tool_response"))),
                scalar_string(p.get("tool_name")),
            ),
            "stop" => (
                EventKind::AssistantMessage,
                Some("assistant".into()),
                scalar_string(p.get("last_assistant_message")).unwrap_or_default(),
                None,
            ),
            "stopfailure" | "stop_failure" => (
                EventKind::Error,
                Some("assistant".into()),
                value_text(p.get("error")),
                None,
            ),
            "precompact" | "pre_compact" | "postcompact" | "post_compact" => (
                EventKind::Compaction,
                None,
                scalar_string(p.get("compact_summary"))
                    .or_else(|| scalar_string(p.get("custom_instructions")))
                    .unwrap_or_default(),
                None,
            ),
            _ => (EventKind::Other, None, p.to_string(), None),
        };
        if matches!(kind, EventKind::Compaction) && native_id.is_none() {
            native_id = Some(format!(
                "{}:{normalized_event}",
                turn.as_deref().unwrap_or(&session)
            ));
        }
        Ok(vec![CanonicalEvent::new(
            AgentKind::Codex,
            session,
            turn,
            native_id,
            kind,
            role,
            content,
            tool,
            cwd.clone(),
            None,
            timestamp_ms(p),
            event_metadata(event_name),
        )])
    }

    fn recall_response(&self, event_name: &str, hits: &[SearchHit]) -> Option<Value> {
        if !matches!(
            event_name.to_ascii_lowercase().as_str(),
            "sessionstart" | "session_start" | "userpromptsubmit" | "user_prompt_submit"
        ) {
            return None;
        }
        let text = recall_text(hits);
        if text.is_empty() {
            None
        } else {
            Some(
                json!({"hookSpecificOutput":{"hookEventName":event_name,"additionalContext":text}}),
            )
        }
    }

    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)> {
        let command = [
            binary.display().to_string(),
            "hook".into(),
            "--agent".into(),
            "codex".into(),
            "--event".into(),
            "auto".into(),
            "--db".into(),
            database.display().to_string(),
            "--recall".into(),
            "--protocol".into(),
        ]
        .iter()
        .map(|value| shell_escape::unix::escape(value.into()))
        .collect::<Vec<_>>()
        .join(" ");
        let mut hooks = serde_json::Map::new();
        for name in [
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "PreCompact",
            "PostCompact",
        ] {
            hooks.insert(
                name.into(),
                json!([{"hooks":[{"type":"command","command":command}]}]),
            );
        }
        let config = json!({"hooks":hooks});
        vec![(
            ".codex/hooks.json".into(),
            serde_json::to_string_pretty(&config).unwrap(),
        )]
    }
}
