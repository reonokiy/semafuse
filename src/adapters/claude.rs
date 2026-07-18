use super::{
    AgentAdapter, ImportContext, ImportedEvent, event_metadata, recall_text, scalar_string,
    text_content, timestamp_ms, value_text,
};
use crate::model::{AgentKind, CanonicalEvent, EventKind, SearchHit};
use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

/// Normalizes Claude Code history records and real-time hook events.
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
/// | `PostToolUse` | [`ToolResult`](EventKind::ToolResult) | `tool` | `tool_response` as text or JSON | `tool_name` |
/// | `PostToolUseFailure` | [`Error`](EventKind::Error) | `tool` | `error` | `tool_name` |
/// | `Stop` | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | `last_assistant_message` | — |
/// | `StopFailure` | [`Error`](EventKind::Error) | `assistant` | `error` | — |
/// | `PreCompact` / `PostCompact` | [`Compaction`](EventKind::Compaction) | — | `compact_summary`, then `custom_instructions` | — |
/// | Other | [`Other`](EventKind::Other) | — | Complete payload JSON | — |
///
/// `session_id` uses `session_id`, or `claude-unknown`; `turn_id` uses
/// `turn_id`; `native_id` tries `tool_use_id`, `agent_id`, `id`, and `prompt_id`.
/// A compaction hook without a native ID receives a synthetic ID containing its
/// phase so `PreCompact` and `PostCompact` do not deduplicate each other.
///
/// # History import
///
/// | Native record | Condition | [`EventKind`] | Role | Content |
/// | --- | --- | --- | --- | --- |
/// | Message/record | Role is `user` | [`UserMessage`](EventKind::UserMessage) | `user` | `message.content` text |
/// | Message/record | Role is `assistant` | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | `message.content` text |
/// | Message/record | No role and record type contains `tool` | [`ToolResult`](EventKind::ToolResult) | — | `message.content` text |
/// | `tool_use` block | — | [`ToolCall`](EventKind::ToolCall) | `assistant` | `input` as text or JSON |
/// | `tool_result` block | `is_error != true` | [`ToolResult`](EventKind::ToolResult) | `tool` | `content` as text or JSON |
/// | `tool_result` block | `is_error == true` | [`Error`](EventKind::Error) | `tool` | `content` as text or JSON |
/// | Top-level transcript `tool_use` | — | [`ToolCall`](EventKind::ToolCall) | `assistant` | `tool_input` as text or JSON |
/// | Top-level transcript `tool_result` | — | [`ToolResult`](EventKind::ToolResult) | `tool` | `tool_output` as text or JSON |
/// | `summary` or a record with `isCompactSummary == true` | — | [`Compaction`](EventKind::Compaction) | — | Summary text |
/// | Other top-level record | — | [`Other`](EventKind::Other) | Native role | Complete record JSON |
///
/// Empty recognized message text does not produce a message event; a wholly
/// unrecognized record produces [`EventKind::Other`]. `session_id` prefers
/// `sessionId`, then `session_id`; `turn_id` uses `parentUuid`; `native_id` uses
/// `uuid`; `cwd` uses the record's `cwd` or import context, and `repository` is
/// detected independently from that path.
pub struct ClaudeAdapter;

impl AgentAdapter for ClaudeAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::ClaudeCode
    }

    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ImportedEvent>> {
        let file_session = values
            .iter()
            .find_map(|record| {
                scalar_string(record.get("sessionId"))
                    .or_else(|| scalar_string(record.get("session_id")))
            })
            .unwrap_or_else(|| context.fallback_session_id.clone());
        let file_cwd = values
            .iter()
            .find_map(|record| scalar_string(record.get("cwd")))
            .or_else(|| context.cwd.clone());
        let mut out = Vec::new();
        for (record_index, record) in values.iter().enumerate() {
            let session = scalar_string(record.get("sessionId"))
                .or_else(|| scalar_string(record.get("session_id")))
                .unwrap_or_else(|| file_session.clone());
            let record_type = record
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let message = record.get("message").unwrap_or(record);
            let role = scalar_string(message.get("role")).or_else(|| match record_type {
                "user" => Some("user".into()),
                "assistant" => Some("assistant".into()),
                _ => None,
            });
            let content_value = message.get("content").unwrap_or(&Value::Null);
            let content = if let Some(parts) = content_value.as_array() {
                parts
                    .iter()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                    .map(text_content)
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                text_content(content_value)
            };
            let cwd = scalar_string(record.get("cwd")).or_else(|| file_cwd.clone());
            let turn = scalar_string(record.get("parentUuid"));
            let native_id = scalar_string(record.get("uuid"));
            let timestamp = timestamp_ms(record).or_else(|| timestamp_ms(message));
            if record.get("message").is_none() && matches!(record_type, "tool_use" | "tool_result")
            {
                let source = if record_type == "tool_use" {
                    record.get("tool_input").unwrap_or(&Value::Null)
                } else {
                    record.get("tool_output").unwrap_or(&Value::Null)
                };
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::ClaudeCode,
                        &session,
                        turn,
                        native_id,
                        if record_type == "tool_use" {
                            EventKind::ToolCall
                        } else {
                            EventKind::ToolResult
                        },
                        Some(if record_type == "tool_use" {
                            "assistant".into()
                        } else {
                            "tool".into()
                        }),
                        value_text(Some(source)),
                        scalar_string(record.get("tool_name")),
                        cwd,
                        None,
                        timestamp,
                        event_metadata(record_type),
                    ),
                    source,
                    context.raw_record_id(record_index),
                ));
                continue;
            }
            let compact = record_type == "summary"
                || record.get("isCompactSummary").and_then(Value::as_bool) == Some(true);
            let content = if record_type == "summary" {
                scalar_string(record.get("summary")).unwrap_or_default()
            } else {
                content
            };
            let mut recognized = false;
            if !content.is_empty()
                && let Some(kind) = if compact {
                    Some(EventKind::Compaction)
                } else {
                    match role.as_deref() {
                        Some("user") => Some(EventKind::UserMessage),
                        Some("assistant") => Some(EventKind::AssistantMessage),
                        _ if record_type.contains("tool") => Some(EventKind::ToolResult),
                        _ => None,
                    }
                }
            {
                recognized = true;
                let attachment_source = content_value
                    .as_array()
                    .map(|parts| {
                        Value::Array(
                            parts
                                .iter()
                                .filter(|part| {
                                    !matches!(
                                        part.get("type").and_then(Value::as_str),
                                        Some("tool_use" | "tool_result")
                                    )
                                })
                                .cloned()
                                .collect(),
                        )
                    })
                    .unwrap_or_else(|| content_value.clone());
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::ClaudeCode,
                        &session,
                        turn.clone(),
                        native_id.clone(),
                        kind,
                        (!compact).then(|| role.clone()).flatten(),
                        content,
                        None,
                        cwd.clone(),
                        None,
                        timestamp,
                        event_metadata(record_type),
                    ),
                    &attachment_source,
                    context.raw_record_id(record_index),
                ));
            }
            for part in message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let (kind, canonical_role, content, tool, part_id) =
                    match part.get("type").and_then(Value::as_str) {
                        Some("tool_use") => (
                            EventKind::ToolCall,
                            Some("assistant".into()),
                            value_text(part.get("input")),
                            scalar_string(part.get("name")),
                            scalar_string(part.get("id")),
                        ),
                        Some("tool_result") => (
                            if part.get("is_error").and_then(Value::as_bool) == Some(true) {
                                EventKind::Error
                            } else {
                                EventKind::ToolResult
                            },
                            Some("tool".into()),
                            value_text(part.get("content")),
                            None,
                            scalar_string(part.get("tool_use_id")),
                        ),
                        _ => continue,
                    };
                recognized = true;
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::ClaudeCode,
                        &session,
                        turn.clone(),
                        part_id,
                        kind,
                        canonical_role,
                        content,
                        tool,
                        cwd.clone(),
                        None,
                        timestamp_ms(part).or(timestamp),
                        event_metadata(part.get("type").and_then(Value::as_str).unwrap_or("part")),
                    ),
                    match part.get("type").and_then(Value::as_str) {
                        Some("tool_use") => part.get("input").unwrap_or(&Value::Null),
                        Some("tool_result") => part.get("content").unwrap_or(&Value::Null),
                        _ => &Value::Null,
                    },
                    context.raw_record_id(record_index),
                ));
            }
            if !recognized {
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::ClaudeCode,
                        session,
                        turn,
                        native_id,
                        EventKind::Other,
                        role,
                        record.to_string(),
                        None,
                        cwd,
                        None,
                        timestamp,
                        event_metadata(record_type),
                    ),
                    record,
                    context.raw_record_id(record_index),
                ));
            }
        }
        Ok(out)
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let session = scalar_string(p.get("session_id")).unwrap_or_else(|| "claude-unknown".into());
        let cwd = scalar_string(p.get("cwd"));
        let mut native_id = scalar_string(p.get("tool_use_id"))
            .or_else(|| scalar_string(p.get("agent_id")))
            .or_else(|| scalar_string(p.get("id")))
            .or_else(|| scalar_string(p.get("prompt_id")));
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
                value_text(p.get("tool_response")),
                scalar_string(p.get("tool_name")),
            ),
            "posttoolusefailure" | "post_tool_use_failure" => (
                EventKind::Error,
                Some("tool".into()),
                scalar_string(p.get("error")).unwrap_or_default(),
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
                p.get("turn_id").and_then(Value::as_str).unwrap_or(&session)
            ));
        }
        Ok(vec![CanonicalEvent::new(
            AgentKind::ClaudeCode,
            session,
            scalar_string(p.get("turn_id")),
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
        let command = binary.display().to_string();
        let args = [
            "hook",
            "--agent",
            "claude-code",
            "--event",
            "auto",
            "--db",
            database.to_str().unwrap_or_default(),
            "--recall",
            "--protocol",
        ];
        let mut hooks = serde_json::Map::new();
        for name in [
            "SessionStart",
            "SessionEnd",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "StopFailure",
            "PreCompact",
            "PostCompact",
        ] {
            hooks.insert(
                name.into(),
                json!([{"hooks":[{"type":"command","command":command,"args":args}]}]),
            );
        }
        vec![(
            ".claude/settings.json".into(),
            serde_json::to_string_pretty(&json!({"hooks":hooks})).unwrap(),
        )]
    }
}
