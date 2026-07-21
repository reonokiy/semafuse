use super::{
    HistoryImporter, HookInstaller, HookProjector, ImportContext, NativeHistoryProjection,
    ProjectedEvent, event_source, message_role, non_null_sources, project_native_history,
    recall_text, scalar_string, text_content, timestamp_ms, value_text,
};
use crate::{
    blob::BlobStore,
    importer::{ImportReport, NativeSource, json_source_files, read_json_values},
    model::{AgentKind, CanonicalEvent, EventKind, MessageRole, SearchHit, ToolStatus},
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
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

struct ClaudeHistory {
    records: Vec<ClaudeRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeRecord {
    #[serde(rename = "sessionId")]
    session_id_camel: Option<Value>,
    session_id: Option<Value>,
    cwd: Option<Value>,
    #[serde(rename = "parentUuid")]
    parent_uuid: Option<Value>,
    uuid: Option<Value>,
    #[serde(rename = "type")]
    record_type: Option<String>,
    message: Option<ClaudeMessage>,
    content: Option<Value>,
    summary: Option<Value>,
    #[serde(rename = "isCompactSummary")]
    is_compact_summary: Option<bool>,
    tool_input: Option<Value>,
    tool_output: Option<Value>,
    tool_name: Option<Value>,
    #[serde(skip)]
    content_parts: Vec<ClaudeContentPart>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeMessage {
    role: Option<Value>,
    content: Option<Value>,
    #[serde(skip)]
    content_parts: Vec<ClaudeContentPart>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeContentPart {
    #[serde(default, rename = "type")]
    part_type: String,
    input: Option<Value>,
    name: Option<Value>,
    id: Option<Value>,
    content: Option<Value>,
    tool_use_id: Option<Value>,
    is_error: Option<bool>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Deserialize)]
struct ClaudeHookPayload {
    session_id: Option<Value>,
    turn_id: Option<Value>,
    cwd: Option<Value>,
    tool_use_id: Option<Value>,
    agent_id: Option<Value>,
    id: Option<Value>,
    prompt_id: Option<Value>,
    reason: Option<Value>,
    prompt: Option<Value>,
    tool_input: Option<Value>,
    tool_name: Option<Value>,
    tool_response: Option<Value>,
    error: Option<Value>,
    last_assistant_message: Option<Value>,
    compact_summary: Option<Value>,
    custom_instructions: Option<Value>,
    content: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

impl TryFrom<Value> for ClaudeRecord {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut record: Self = serde_json::from_value(value.clone())?;
        if let Some(raw_message) = value.get("message")
            && let Some(message) = record.message.as_mut()
        {
            message.attach_raw(raw_message.clone())?;
        }
        record.content_parts = value
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .cloned()
            .map(ClaudeContentPart::try_from)
            .collect::<Result<Vec<_>>>()?;
        record.raw = value;
        Ok(record)
    }
}

impl ClaudeRecord {
    fn session_id(&self) -> Option<String> {
        scalar_string(self.session_id_camel.as_ref())
            .or_else(|| scalar_string(self.session_id.as_ref()))
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.cwd.as_ref())
    }

    fn record_type(&self) -> &str {
        self.record_type.as_deref().unwrap_or_default()
    }

    fn role(&self) -> Option<String> {
        self.message
            .as_ref()
            .and_then(ClaudeMessage::role)
            .or_else(|| match self.record_type() {
                "user" => Some("user".into()),
                "assistant" => Some("assistant".into()),
                _ => None,
            })
    }

    fn content_value(&self) -> &Value {
        self.message
            .as_ref()
            .and_then(ClaudeMessage::content_value)
            .or(self.content.as_ref())
            .unwrap_or(&Value::Null)
    }

    fn content_parts(&self) -> &[ClaudeContentPart] {
        self.message
            .as_ref()
            .map(ClaudeMessage::content_parts)
            .unwrap_or(&self.content_parts)
    }

    fn content_text(&self) -> String {
        let content = self.content_value();
        if content.as_array().is_some() {
            self.content_parts()
                .iter()
                .filter(|part| part.part_type() == Some("text"))
                .map(|part| text_content(part.raw_value()))
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            text_content(content)
        }
    }

    fn attachment_source(&self) -> Value {
        if self.content_value().as_array().is_some() {
            Value::Array(
                self.content_parts()
                    .iter()
                    .filter(|part| !matches!(part.part_type(), Some("tool_use" | "tool_result")))
                    .map(|part| part.raw_value().clone())
                    .collect(),
            )
        } else {
            self.content_value().clone()
        }
    }

    fn turn(&self) -> Option<String> {
        scalar_string(self.parent_uuid.as_ref())
    }

    fn native_id(&self) -> Option<String> {
        scalar_string(self.uuid.as_ref())
    }

    fn timestamp_ms(&self) -> Option<i64> {
        timestamp_ms(&self.raw).or_else(|| {
            self.message
                .as_ref()
                .and_then(|message| timestamp_ms(message.raw_value()))
        })
    }

    fn is_top_level_tool(&self) -> bool {
        self.message.is_none() && matches!(self.record_type(), "tool_use" | "tool_result")
    }

    fn top_level_tool_source(&self) -> &Value {
        if self.record_type() == "tool_use" {
            self.tool_input.as_ref()
        } else {
            self.tool_output.as_ref()
        }
        .unwrap_or(&Value::Null)
    }

    fn tool_name(&self) -> Option<String> {
        scalar_string(self.tool_name.as_ref())
    }

    fn is_compaction(&self) -> bool {
        self.record_type() == "summary" || self.is_compact_summary == Some(true)
    }

    fn summary_text(&self) -> String {
        scalar_string(self.summary.as_ref()).unwrap_or_default()
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl ClaudeMessage {
    fn attach_raw(&mut self, value: Value) -> Result<()> {
        self.content_parts = value
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .cloned()
            .map(ClaudeContentPart::try_from)
            .collect::<Result<Vec<_>>>()?;
        self.raw = value;
        Ok(())
    }

    fn role(&self) -> Option<String> {
        scalar_string(self.role.as_ref())
    }

    fn content_value(&self) -> Option<&Value> {
        self.content.as_ref()
    }

    fn content_parts(&self) -> &[ClaudeContentPart] {
        &self.content_parts
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<Value> for ClaudeContentPart {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut part: Self = serde_json::from_value(value.clone())?;
        part.raw = value;
        Ok(part)
    }
}

impl ClaudeContentPart {
    fn part_type(&self) -> Option<&str> {
        (!self.part_type.is_empty()).then_some(self.part_type.as_str())
    }

    fn input(&self) -> Option<&Value> {
        self.input.as_ref()
    }

    fn name(&self) -> Option<String> {
        scalar_string(self.name.as_ref())
    }

    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn content(&self) -> Option<&Value> {
        self.content.as_ref()
    }

    fn tool_use_id(&self) -> Option<String> {
        scalar_string(self.tool_use_id.as_ref())
    }

    fn is_error(&self) -> bool {
        self.is_error == Some(true)
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<&Value> for ClaudeHookPayload {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let mut payload: Self = serde_json::from_value(value.clone())?;
        payload.raw = value.clone();
        Ok(payload)
    }
}

impl ClaudeHookPayload {
    fn raw_value(&self) -> &Value {
        &self.raw
    }

    fn session(&self) -> String {
        scalar_string(self.session_id.as_ref()).unwrap_or_else(|| "claude-unknown".into())
    }

    fn turn(&self) -> Option<String> {
        scalar_string(self.turn_id.as_ref())
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.cwd.as_ref())
    }

    fn native_id(&self) -> Option<String> {
        scalar_string(self.tool_use_id.as_ref())
            .or_else(|| scalar_string(self.agent_id.as_ref()))
            .or_else(|| scalar_string(self.id.as_ref()))
            .or_else(|| scalar_string(self.prompt_id.as_ref()))
    }

    fn synthetic_compaction_id(&self, session: &str, normalized_event: &str) -> String {
        format!(
            "{}:{normalized_event}",
            self.turn().as_deref().unwrap_or(session)
        )
    }
}

impl NativeHistoryProjection for ClaudeHistory {
    fn decode_native(values: &[Value], _context: &ImportContext) -> Result<Self> {
        Ok(Self {
            records: values
                .iter()
                .cloned()
                .map(ClaudeRecord::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn project_native(&self, context: &ImportContext) -> Result<Vec<ProjectedEvent>> {
        let file_session = self
            .records
            .iter()
            .find_map(ClaudeRecord::session_id)
            .unwrap_or_else(|| context.fallback_session_id.clone());
        let file_cwd = self
            .records
            .iter()
            .find_map(ClaudeRecord::cwd)
            .or_else(|| context.cwd.clone());
        let mut out = Vec::new();
        for (record_index, record) in self.records.iter().enumerate() {
            let session = record.session_id().unwrap_or_else(|| file_session.clone());
            let record_type = record.record_type();
            let role = record.role();
            let content = record.content_text();
            let cwd = record.cwd().or_else(|| file_cwd.clone());
            let turn = record.turn();
            let native_id = record.native_id();
            let timestamp = record.timestamp_ms();
            if record.is_top_level_tool() {
                let source = record.top_level_tool_source();
                let event_source = event_source(
                    AgentKind::ClaudeCode,
                    &session,
                    turn,
                    native_id,
                    record_type,
                    cwd,
                    None,
                );
                let event = if record_type == "tool_use" {
                    CanonicalEvent::tool_call_from_content(
                        event_source,
                        record.tool_name().unwrap_or_else(|| "unknown".into()),
                        value_text(Some(source)),
                        timestamp,
                    )
                } else {
                    CanonicalEvent::tool_result_from_content(
                        event_source,
                        record.tool_name(),
                        ToolStatus::Succeeded,
                        value_text(Some(source)),
                        None,
                        timestamp,
                    )
                };
                out.push(ProjectedEvent::new(
                    event,
                    source,
                    context.raw_record_id(record_index),
                ));
                continue;
            }
            let compact = record.is_compaction();
            let content = if record_type == "summary" {
                record.summary_text()
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
                let attachment_source = record.attachment_source();
                let source = event_source(
                    AgentKind::ClaudeCode,
                    &session,
                    turn.clone(),
                    native_id.clone(),
                    record_type,
                    cwd.clone(),
                    None,
                );
                let event = if compact {
                    CanonicalEvent::compaction_from_content(source, content, timestamp)
                } else {
                    let role = message_role(
                        role.as_deref(),
                        match kind {
                            EventKind::UserMessage => MessageRole::User,
                            EventKind::AssistantMessage => MessageRole::Assistant,
                            EventKind::ToolResult => MessageRole::Tool,
                            _ => MessageRole::Unknown,
                        },
                    );
                    if matches!(kind, EventKind::ToolResult) {
                        CanonicalEvent::tool_result_from_content(
                            source,
                            None,
                            ToolStatus::Succeeded,
                            content,
                            None,
                            timestamp,
                        )
                    } else {
                        CanonicalEvent::message_from_text(source, role, content, timestamp)
                    }
                };
                out.push(ProjectedEvent::new(
                    event,
                    &attachment_source,
                    context.raw_record_id(record_index),
                ));
            }
            for part in record.content_parts() {
                let (kind, content, tool, part_id): (
                    EventKind,
                    String,
                    Option<String>,
                    Option<String>,
                ) = match part.part_type() {
                    Some("tool_use") => (
                        EventKind::ToolCall,
                        value_text(part.input()),
                        part.name(),
                        part.id(),
                    ),
                    Some("tool_result") => (
                        if part.is_error() {
                            EventKind::Error
                        } else {
                            EventKind::ToolResult
                        },
                        value_text(part.content()),
                        None,
                        part.tool_use_id(),
                    ),
                    _ => continue,
                };
                recognized = true;
                let native_event = part.part_type().unwrap_or("part");
                let source = event_source(
                    AgentKind::ClaudeCode,
                    &session,
                    turn.clone(),
                    part_id,
                    native_event,
                    cwd.clone(),
                    None,
                );
                let timestamp = timestamp_ms(part.raw_value()).or(timestamp);
                let event = match &kind {
                    EventKind::ToolCall => CanonicalEvent::tool_call_from_content(
                        source,
                        tool.unwrap_or_else(|| "unknown".into()),
                        content,
                        timestamp,
                    ),
                    EventKind::ToolResult => CanonicalEvent::tool_result_from_content(
                        source,
                        tool,
                        ToolStatus::Succeeded,
                        content,
                        None,
                        timestamp,
                    ),
                    EventKind::Error => {
                        CanonicalEvent::failed_tool_result(source, tool, content, timestamp)
                    }
                    _ => CanonicalEvent::unknown(source, native_event, timestamp),
                };
                let attachment_source = match part.part_type() {
                    Some("tool_use") => part.input().unwrap_or(&Value::Null),
                    Some("tool_result") => part.content().unwrap_or(&Value::Null),
                    _ => &Value::Null,
                };
                out.push(ProjectedEvent::new(
                    event,
                    attachment_source,
                    context.raw_record_id(record_index),
                ));
            }
            if !recognized {
                let source = event_source(
                    AgentKind::ClaudeCode,
                    session,
                    turn,
                    native_id,
                    record_type,
                    cwd,
                    None,
                );
                out.push(ProjectedEvent::new(
                    CanonicalEvent::unknown(source, record_type, timestamp),
                    record.raw_value(),
                    context.raw_record_id(record_index),
                ));
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl HistoryImporter for ClaudeAdapter {
    async fn import_default(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport> {
        let home = dirs::home_dir().context("home directory not found")?;
        self.import_path(&home.join(".claude/projects"), blobs)
            .await
    }

    async fn import_path(
        &self,
        path: &Path,
        blobs: Option<&dyn BlobStore>,
    ) -> Result<ImportReport> {
        let mut report = ImportReport::default();
        let files = json_source_files(path);
        for file in files {
            match read_json_values(&file) {
                Ok(values) => {
                    let fallback = file
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_owned();
                    report
                        .add_native_source(
                            NativeSource {
                                agent: AgentKind::ClaudeCode,
                                source: file.display().to_string(),
                                fallback_session_id: fallback,
                                cwd: None,
                                values,
                            },
                            blobs,
                            |values, context| self.import_values(values, context),
                        )
                        .await;
                }
                Err(error) => report.errors.push(format!("{}: {error:#}", file.display())),
            }
        }
        Ok(report)
    }

    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ProjectedEvent>> {
        project_native_history::<ClaudeHistory>(values, context)
    }
}

impl HookProjector for ClaudeAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::ClaudeCode
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let payload: ClaudeHookPayload = p.try_into()?;
        let session = payload.session();
        let cwd = payload.cwd();
        let mut native_id = payload.native_id();
        let normalized_event = event_name.to_ascii_lowercase();
        let is_compaction = matches!(
            normalized_event.as_str(),
            "precompact" | "pre_compact" | "postcompact" | "post_compact"
        );
        if is_compaction && native_id.is_none() {
            native_id = Some(payload.synthetic_compaction_id(&session, normalized_event.as_str()));
        }
        let source = event_source(
            AgentKind::ClaudeCode,
            session,
            payload.turn(),
            native_id,
            event_name,
            cwd.clone(),
            None,
        );
        let timestamp = timestamp_ms(payload.raw_value());
        let event = match normalized_event.as_str() {
            "sessionstart" | "session_start" => CanonicalEvent::session_started(source, timestamp),
            "sessionend" | "session_end" => CanonicalEvent::session_ended(
                source,
                scalar_string(payload.reason.as_ref()).filter(|value| !value.trim().is_empty()),
                timestamp,
            ),
            "userpromptsubmit" | "user_prompt_submit" => CanonicalEvent::message_from_text(
                source,
                MessageRole::User,
                scalar_string(payload.prompt.as_ref()).unwrap_or_default(),
                timestamp,
            ),
            "pretooluse" | "pre_tool_use" => CanonicalEvent::tool_call_from_content(
                source,
                scalar_string(payload.tool_name.as_ref()).unwrap_or_else(|| "unknown".into()),
                value_text(payload.tool_input.as_ref()),
                timestamp,
            ),
            "posttooluse" | "post_tool_use" => CanonicalEvent::tool_result_from_content(
                source,
                scalar_string(payload.tool_name.as_ref()),
                ToolStatus::Succeeded,
                value_text(payload.tool_response.as_ref()),
                None,
                timestamp,
            ),
            "posttoolusefailure" | "post_tool_use_failure" => CanonicalEvent::failed_tool_result(
                source,
                scalar_string(payload.tool_name.as_ref()),
                value_text(payload.error.as_ref()),
                timestamp,
            ),
            "stop" => CanonicalEvent::message_from_text(
                source,
                MessageRole::Assistant,
                scalar_string(payload.last_assistant_message.as_ref()).unwrap_or_default(),
                timestamp,
            ),
            "stopfailure" | "stop_failure" => {
                CanonicalEvent::error_message(source, value_text(payload.error.as_ref()), timestamp)
            }
            "precompact" | "pre_compact" | "postcompact" | "post_compact" => {
                CanonicalEvent::compaction_from_content(
                    source,
                    scalar_string(payload.compact_summary.as_ref())
                        .or_else(|| scalar_string(payload.custom_instructions.as_ref()))
                        .unwrap_or_default(),
                    timestamp,
                )
            }
            _ => CanonicalEvent::unknown(source, event_name, timestamp),
        };
        Ok(vec![event])
    }

    fn hook_attachment_sources(&self, event_name: &str, p: &Value) -> Vec<Value> {
        let Ok(payload) = ClaudeHookPayload::try_from(p) else {
            return Vec::new();
        };
        match event_name.to_ascii_lowercase().as_str() {
            "userpromptsubmit" | "user_prompt_submit" => {
                non_null_sources([payload.content.as_ref(), payload.prompt.as_ref()])
            }
            "pretooluse" | "pre_tool_use" => {
                non_null_sources([payload.tool_input.as_ref(), payload.content.as_ref()])
            }
            "posttooluse" | "post_tool_use" => non_null_sources([payload.tool_response.as_ref()]),
            "posttoolusefailure" | "post_tool_use_failure" => {
                non_null_sources([payload.error.as_ref()])
            }
            "stop" => non_null_sources([
                payload.last_assistant_message.as_ref(),
                payload.content.as_ref(),
            ]),
            _ => Vec::new(),
        }
    }

    fn supports_recall(&self, event_name: &str) -> bool {
        matches!(
            event_name.to_ascii_lowercase().as_str(),
            "sessionstart" | "session_start" | "userpromptsubmit" | "user_prompt_submit"
        )
    }

    fn recall_response(&self, event_name: &str, hits: &[SearchHit]) -> Option<Value> {
        if !self.supports_recall(event_name) {
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
}

impl HookInstaller for ClaudeAdapter {
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)> {
        let command = binary.display().to_string();
        let base_args = [
            "hook",
            "--agent",
            "claude-code",
            "--event",
            "auto",
            "--db",
            database.to_str().unwrap_or_default(),
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
            let mut args = base_args.to_vec();
            if self.supports_recall(name) {
                args.push("--recall");
            }
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
