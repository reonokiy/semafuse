use super::{
    HistoryImporter, HookInstaller, HookProjector, ImportContext, NativeHistoryProjection,
    ProjectedEvent, event_source, message_role, non_null_sources, project_native_history,
    scalar_string, text_content, timestamp_at, timestamp_ms, value_text,
};
use crate::{
    blob::BlobStore,
    importer::{ImportReport, NativeSource, json_source_files, read_json_values},
    model::{AgentKind, CanonicalEvent, EventKind, MessageRole, SearchHit, ToolStatus},
};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::{
    net::TcpListener,
    path::Path,
    process::{Command as StdCommand, Stdio},
    time::Duration,
};

/// Normalizes OpenCode history records and real-time bus events.
///
/// # Real-time hooks
///
/// OpenCode payloads normally use `{ event, directory, worktree }`. With
/// `--event auto`, the adapter uses `event.type` and reads message information
/// from `event.properties.info`, `event.properties.message`, or the properties
/// object itself.
///
/// | Native `event.type` | Condition | [`EventKind`] | Role | Content |
/// | --- | --- | --- | --- | --- |
/// | `session.created` | — | [`SessionStart`](EventKind::SessionStart) | Native role | Empty |
/// | `session.deleted` | — | [`SessionEnd`](EventKind::SessionEnd) | Native role | Properties JSON |
/// | `session.error` | — | [`Error`](EventKind::Error) | Native role | `error` or properties JSON |
/// | `session.compacted` | — | [`Compaction`](EventKind::Compaction) | Native role | Properties JSON |
/// | `message.part.updated` | User text part | [`UserMessage`](EventKind::UserMessage) | `user` | Part text |
/// | `message.part.updated` | Other text part | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | Part text |
/// | `message.part.updated` | Pending/running tool part | [`ToolCall`](EventKind::ToolCall) | `assistant` | `state.input` |
/// | `message.part.updated` | Completed tool part | [`ToolResult`](EventKind::ToolResult) | `tool` | `state.output` |
/// | `message.part.updated` | Failed tool part | [`Error`](EventKind::Error) | `tool` | `state.error` |
/// | `message.part.updated` | Compaction part | [`Compaction`](EventKind::Compaction) | — | Properties JSON |
/// | `tool.execute.before` | — | [`ToolCall`](EventKind::ToolCall) | `assistant` | `properties.input` |
/// | `tool.execute.after` | — | [`ToolResult`](EventKind::ToolResult) | `tool` | `properties.output` |
/// | Other | — | [`Other`](EventKind::Other) | Native role | Properties JSON |
///
/// Tool names come from `part.tool`, then `properties.tool`. The generated plugin
/// caches roles from `message.updated` without persisting those events. It keeps
/// only the latest assistant text-part snapshot and writes it when the session
/// becomes idle; unrelated high-frequency bus events are not forwarded.
/// `session_id` prefers message, part, then properties/session IDs. Tool
/// `native_id` prefers `callID`; other events use part/message IDs. `cwd` uses
/// `directory`, while `repository` prefers `worktree`.
///
/// # History import
///
/// | Native message/part | Condition | [`EventKind`] | Role | Content |
/// | --- | --- | --- | --- | --- |
/// | Message text parts | `info.role == "user"` | [`UserMessage`](EventKind::UserMessage) | `user` | All text parts joined with newlines |
/// | Message text parts | Other role | [`AssistantMessage`](EventKind::AssistantMessage) | Native role | All text parts joined with newlines |
/// | Tool part | Any status | [`ToolCall`](EventKind::ToolCall) | `assistant` | `state.input` |
/// | Tool part | Status is `completed` | [`ToolResult`](EventKind::ToolResult) | `tool` | `state.output` |
/// | Tool part | Status is `error` | [`Error`](EventKind::Error) | `tool` | `state.error` |
/// | Compaction part | — | [`Compaction`](EventKind::Compaction) | — | Complete part JSON |
/// | Message with no recognized content | — | [`Other`](EventKind::Other) | Native role | Complete message JSON |
///
/// `session_id` prefers message `info.sessionID`, then root
/// `info.id`/`info.sessionID`. Message IDs supply both `turn_id` and `native_id`;
/// tool events use `callID` as `native_id`.
pub struct OpenCodeAdapter;

struct OpenCodeHistory {
    root_session: String,
    cwd: Option<String>,
    messages: Vec<OpenCodeMessage>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodeRoot {
    #[serde(default)]
    info: OpenCodeInfo,
    #[serde(default)]
    messages: Option<Vec<OpenCodeMessage>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OpenCodeInfo {
    id: Option<Value>,
    #[serde(rename = "sessionID")]
    session_id: Option<Value>,
    #[serde(rename = "parentID")]
    parent_id: Option<Value>,
    role: Option<Value>,
    directory: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodeMessage {
    #[serde(default)]
    info: OpenCodeInfo,
    #[serde(default)]
    parts: Vec<OpenCodePart>,
    #[serde(default)]
    content: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodePart {
    id: Option<Value>,
    #[serde(rename = "callID")]
    call_id: Option<Value>,
    #[serde(rename = "messageID")]
    message_id: Option<Value>,
    #[serde(rename = "sessionID")]
    session_id: Option<Value>,
    tool: Option<Value>,
    #[serde(default, rename = "type")]
    part_type: String,
    #[serde(default)]
    state: OpenCodeToolState,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OpenCodeToolState {
    status: Option<String>,
    input: Option<Value>,
    output: Option<Value>,
    error: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone)]
struct OpenCodeHookPayload {
    event: OpenCodeHookEvent,
    directory: Option<Value>,
    worktree: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodeHookEnvelope {
    event: Option<Value>,
    directory: Option<Value>,
    worktree: Option<Value>,
}

#[derive(Debug, Clone)]
struct OpenCodeHookEvent {
    event_type: Option<String>,
    properties: OpenCodeHookProperties,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodeHookEventFields {
    #[serde(rename = "type")]
    event_type: Option<String>,
    properties: Option<Value>,
}

#[derive(Debug, Clone)]
struct OpenCodeHookProperties {
    info: OpenCodeInfo,
    part: Option<OpenCodePart>,
    session_id: Option<Value>,
    message_role: Option<Value>,
    input: Option<Value>,
    output: Option<Value>,
    error: Option<Value>,
    tool: Option<Value>,
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenCodeHookPropertyFields {
    info: Option<Value>,
    message: Option<Value>,
    part: Option<Value>,
    #[serde(rename = "sessionID")]
    session_id: Option<Value>,
    #[serde(rename = "messageRole")]
    message_role: Option<Value>,
    input: Option<Value>,
    output: Option<Value>,
    error: Option<Value>,
    tool: Option<Value>,
}

impl TryFrom<&[Value]> for OpenCodeRoot {
    type Error = anyhow::Error;

    fn try_from(values: &[Value]) -> Result<Self, Self::Error> {
        let root_value = values.first().cloned().unwrap_or(Value::Null);
        let mut root: Self = serde_json::from_value(root_value.clone())?;
        if let Some(raw_info) = root_value.get("info") {
            root.info.raw = raw_info.clone();
        } else if root.messages.is_none() {
            root.info = root_value.clone().try_into()?;
        }
        if let Some(raw_messages) = root_value.get("messages").and_then(Value::as_array)
            && let Some(messages) = root.messages.as_mut()
        {
            for (message, raw_message) in messages.iter_mut().zip(raw_messages) {
                message.attach_raw(raw_message.clone())?;
            }
        }
        Ok(root)
    }
}

impl OpenCodeRoot {
    fn session(&self, fallback: &str) -> String {
        self.info
            .id()
            .or_else(|| self.info.session_id())
            .unwrap_or_else(|| fallback.into())
    }

    fn cwd(&self) -> Option<String> {
        self.info.directory()
    }

    fn into_messages(self, values: &[Value]) -> Result<Vec<OpenCodeMessage>> {
        if let Some(messages) = self.messages {
            Ok(messages)
        } else {
            values
                .iter()
                .cloned()
                .map(OpenCodeMessage::try_from)
                .collect::<Result<Vec<_>>>()
        }
    }
}

impl TryFrom<Value> for OpenCodeInfo {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut info: Self = serde_json::from_value(value.clone())?;
        info.raw = value;
        Ok(info)
    }
}

impl OpenCodeInfo {
    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn session_id(&self) -> Option<String> {
        scalar_string(self.session_id.as_ref())
    }

    fn parent_id(&self) -> Option<String> {
        scalar_string(self.parent_id.as_ref())
    }

    fn role(&self) -> Option<String> {
        scalar_string(self.role.as_ref())
    }

    fn directory(&self) -> Option<String> {
        scalar_string(self.directory.as_ref())
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<Value> for OpenCodeMessage {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut message: Self = serde_json::from_value(value.clone())?;
        message.attach_raw(value)?;
        Ok(message)
    }
}

impl OpenCodeMessage {
    fn attach_raw(&mut self, value: Value) -> Result<()> {
        if value.get("info").is_none() {
            self.info = value.clone().try_into()?;
        } else {
            self.info.raw = value.get("info").cloned().unwrap_or(Value::Null);
        }
        if let Some(raw_parts) = value.get("parts").and_then(Value::as_array) {
            for (part, raw_part) in self.parts.iter_mut().zip(raw_parts) {
                part.attach_raw(raw_part.clone());
            }
        }
        self.raw = value;
        Ok(())
    }

    fn timestamp_ms(&self) -> Option<i64> {
        timestamp_ms(self.info.raw_value()).or_else(|| timestamp_ms(&self.raw))
    }

    fn text(&self) -> String {
        let content = self
            .parts
            .iter()
            .filter(|part| part.part_type() == Some("text"))
            .map(|part| text_content(part.raw_value()))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if content.is_empty() {
            self.content.as_ref().map(text_content).unwrap_or_default()
        } else {
            content
        }
    }

    fn attachment_source(&self) -> Value {
        Value::Array(
            self.parts
                .iter()
                .filter(|part| !matches!(part.part_type(), Some("tool" | "compaction")))
                .map(|part| part.raw_value().clone())
                .collect(),
        )
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<Value> for OpenCodePart {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut part: Self = serde_json::from_value(value.clone())?;
        part.attach_raw(value);
        Ok(part)
    }
}

impl OpenCodePart {
    fn attach_raw(&mut self, value: Value) {
        if let Some(raw_state) = value.get("state") {
            self.state.raw = raw_state.clone();
        }
        self.raw = value;
    }

    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn call_id(&self) -> Option<String> {
        scalar_string(self.call_id.as_ref())
    }

    fn message_id(&self) -> Option<String> {
        scalar_string(self.message_id.as_ref())
    }

    fn session_id(&self) -> Option<String> {
        scalar_string(self.session_id.as_ref())
    }

    fn tool(&self) -> Option<String> {
        scalar_string(self.tool.as_ref())
    }

    fn part_type(&self) -> Option<&str> {
        (!self.part_type.is_empty()).then_some(self.part_type.as_str())
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl OpenCodeToolState {
    fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    fn input_value(&self) -> Option<&Value> {
        self.input.as_ref()
    }

    fn output_value(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    fn error_value(&self) -> Option<&Value> {
        self.error.as_ref()
    }

    fn input(&self) -> &Value {
        self.input.as_ref().unwrap_or(&Value::Null)
    }

    fn output_or_error(&self) -> &Value {
        self.output
            .as_ref()
            .or(self.error.as_ref())
            .unwrap_or(&Value::Null)
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<&Value> for OpenCodeHookPayload {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let envelope: OpenCodeHookEnvelope = serde_json::from_value(value.clone())?;
        let event = envelope.event.unwrap_or_else(|| value.clone()).try_into()?;
        Ok(Self {
            event,
            directory: envelope.directory,
            worktree: envelope.worktree,
        })
    }
}

impl OpenCodeHookPayload {
    fn event_type<'a>(&'a self, event_name: &'a str) -> &'a str {
        if event_name == "auto" {
            self.event.event_type.as_deref().unwrap_or("other")
        } else {
            event_name
        }
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.directory.as_ref()).or_else(|| scalar_string(self.worktree.as_ref()))
    }

    fn worktree(&self) -> Option<String> {
        scalar_string(self.worktree.as_ref())
    }
}

impl TryFrom<Value> for OpenCodeHookEvent {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let fields: OpenCodeHookEventFields = serde_json::from_value(value.clone())?;
        let properties = fields
            .properties
            .unwrap_or_else(|| value.clone())
            .try_into()?;
        Ok(Self {
            event_type: fields.event_type,
            properties,
        })
    }
}

impl TryFrom<Value> for OpenCodeHookProperties {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let fields: OpenCodeHookPropertyFields = serde_json::from_value(value.clone())?;
        let info = fields
            .info
            .clone()
            .or(fields.message.clone())
            .unwrap_or_else(|| value.clone())
            .try_into()?;
        let part = fields.part.map(OpenCodePart::try_from).transpose()?;
        Ok(Self {
            info,
            part,
            session_id: fields.session_id,
            message_role: fields.message_role,
            input: fields.input,
            output: fields.output,
            error: fields.error,
            tool: fields.tool,
            raw: value,
        })
    }
}

impl OpenCodeHookProperties {
    fn session_id(&self) -> Option<String> {
        scalar_string(self.session_id.as_ref())
    }

    fn message_role(&self) -> Option<String> {
        scalar_string(self.message_role.as_ref())
    }

    fn tool(&self) -> Option<String> {
        scalar_string(self.tool.as_ref())
    }

    fn input(&self) -> Option<&Value> {
        self.input.as_ref()
    }

    fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    fn error(&self) -> Option<&Value> {
        self.error.as_ref()
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl NativeHistoryProjection for OpenCodeHistory {
    fn decode_native(values: &[Value], context: &ImportContext) -> Result<Self> {
        let root: OpenCodeRoot = values.try_into()?;
        let root_session = root.session(&context.fallback_session_id);
        let cwd = root.cwd().or_else(|| context.cwd.clone());
        let messages = root.into_messages(values)?;
        Ok(Self {
            root_session,
            cwd,
            messages,
        })
    }

    fn project_native(&self, context: &ImportContext) -> Result<Vec<ProjectedEvent>> {
        let mut out = Vec::new();
        for message in &self.messages {
            let session = message
                .info
                .session_id()
                .unwrap_or_else(|| self.root_session.clone());
            let role = message.info.role();
            let message_id = message.info.id();
            let timestamp = message.timestamp_ms();
            let content = message.text();
            let mut recognized = false;
            if !content.is_empty() {
                recognized = true;
                let attachment_source = message.attachment_source();
                let source = event_source(
                    AgentKind::OpenCode,
                    &session,
                    message_id.clone(),
                    message_id.clone(),
                    "message",
                    self.cwd.clone(),
                    None,
                );
                let role = message_role(
                    role.as_deref(),
                    if role.as_deref() == Some("user") {
                        MessageRole::User
                    } else {
                        MessageRole::Assistant
                    },
                );
                out.push(ProjectedEvent::new(
                    CanonicalEvent::message_from_text(source, role, content, timestamp),
                    &attachment_source,
                    context.raw_record_id(0),
                ));
            }
            for part in &message.parts {
                match part.part_type() {
                    Some("tool") => {
                        recognized = true;
                        let call_id = part.call_id();
                        let tool = part.tool();
                        let source = event_source(
                            AgentKind::OpenCode,
                            &session,
                            message_id.clone(),
                            call_id.clone(),
                            "tool",
                            self.cwd.clone(),
                            None,
                        );
                        out.push(ProjectedEvent::new(
                            CanonicalEvent::tool_call_from_content(
                                source,
                                tool.clone().unwrap_or_else(|| "unknown".into()),
                                value_text(Some(part.state.input())),
                                timestamp_at(part.state.raw_value(), &["/time/start"])
                                    .or(timestamp),
                            ),
                            part.state.input(),
                            context.raw_record_id(0),
                        ));
                        if matches!(part.state.status(), Some("completed" | "error")) {
                            let source = event_source(
                                AgentKind::OpenCode,
                                &session,
                                message_id.clone(),
                                call_id,
                                "tool",
                                self.cwd.clone(),
                                None,
                            );
                            let content = value_text(Some(part.state.output_or_error()));
                            let timestamp =
                                timestamp_at(part.state.raw_value(), &["/time/end", "/time/start"])
                                    .or(timestamp);
                            let event = if part.state.status() == Some("error") {
                                CanonicalEvent::failed_tool_result(source, tool, content, timestamp)
                            } else {
                                CanonicalEvent::tool_result_from_content(
                                    source,
                                    tool,
                                    ToolStatus::Succeeded,
                                    content,
                                    None,
                                    timestamp,
                                )
                            };
                            out.push(ProjectedEvent::new(
                                event,
                                part.state.output_or_error(),
                                context.raw_record_id(0),
                            ));
                        }
                    }
                    Some("compaction") => {
                        recognized = true;
                        let source = event_source(
                            AgentKind::OpenCode,
                            &session,
                            message_id.clone(),
                            part.id(),
                            "compaction",
                            self.cwd.clone(),
                            None,
                        );
                        out.push(ProjectedEvent::new(
                            CanonicalEvent::compaction_from_content(
                                source,
                                part.raw_value().to_string(),
                                timestamp_ms(part.raw_value()).or(timestamp),
                            ),
                            part.raw_value(),
                            context.raw_record_id(0),
                        ));
                    }
                    _ => {}
                }
            }
            if !recognized {
                let source = event_source(
                    AgentKind::OpenCode,
                    session,
                    message_id.clone(),
                    message_id,
                    "message",
                    self.cwd.clone(),
                    None,
                );
                out.push(ProjectedEvent::new(
                    CanonicalEvent::unknown(source, "message", timestamp),
                    message.raw_value(),
                    context.raw_record_id(0),
                ));
            }
        }
        Ok(out)
    }
}

struct ChildGuard(std::process::Child);

impl ChildGuard {
    fn kill(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

#[async_trait]
impl HistoryImporter for OpenCodeAdapter {
    async fn import_default(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        let child = StdCommand::new("opencode")
            .args([
                "serve",
                "--hostname",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("start opencode server")?;
        let mut child = ChildGuard(child);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        let base = format!("http://127.0.0.1:{port}");
        let sessions = async {
            for _ in 0..100 {
                if let Ok(response) = client.get(format!("{base}/session")).send().await
                    && response.status().is_success()
                {
                    return Ok::<Value, anyhow::Error>(response.json().await?);
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            bail!("OpenCode server did not become ready")
        }
        .await?;
        let mut report = ImportReport::default();
        let sessions = sessions.as_array().cloned().unwrap_or_default();
        for item in &sessions {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            let mut request = client.get(format!("{base}/session/{id}/message"));
            if let Some(directory) = item.get("directory").and_then(Value::as_str) {
                request = request.header("x-opencode-directory", directory);
            }
            let response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    report
                        .errors
                        .push(format!("opencode messages {id} failed: {error}"));
                    continue;
                }
            };
            if !response.status().is_success() {
                report.errors.push(format!(
                    "opencode messages {id} failed: {}",
                    response.status()
                ));
                continue;
            }
            let messages: Value = response.json().await?;
            let value = serde_json::json!({"info":item,"messages":messages});
            report
                .add_native_source(
                    NativeSource {
                        agent: AgentKind::OpenCode,
                        source: format!("opencode:{id}"),
                        fallback_session_id: id.to_owned(),
                        cwd: None,
                        values: vec![value],
                    },
                    blobs,
                    |values, context| self.import_values(values, context),
                )
                .await;
        }
        child.kill();
        Ok(report)
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
                                agent: AgentKind::OpenCode,
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
        project_native_history::<OpenCodeHistory>(values, context)
    }
}

impl HookProjector for OpenCodeAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::OpenCode
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let payload: OpenCodeHookPayload = p.try_into()?;
        let props = &payload.event.properties;
        let info = &props.info;
        let part = props.part.as_ref();
        let event_type = payload.event_type(event_name);
        let session = info
            .session_id()
            .or_else(|| props.session_id())
            .or_else(|| part.and_then(OpenCodePart::session_id))
            .or_else(|| info.id())
            .unwrap_or_else(|| "opencode-unknown".into());
        let cwd = payload.cwd();
        let role = info.role().or_else(|| props.message_role());
        let part_type = part.and_then(OpenCodePart::part_type);
        let tool_state = part.map(|part| &part.state);
        let kind = match event_type {
            "session.created" => EventKind::SessionStart,
            "session.deleted" => EventKind::SessionEnd,
            "session.error" => EventKind::Error,
            "session.compacted" => EventKind::Compaction,
            "message.part.updated"
                if part_type == Some("text") && role.as_deref() == Some("user") =>
            {
                EventKind::UserMessage
            }
            "message.part.updated" if part_type == Some("text") => EventKind::AssistantMessage,
            "message.part.updated"
                if part_type == Some("tool")
                    && tool_state.and_then(|state| state.status()) == Some("completed") =>
            {
                EventKind::ToolResult
            }
            "message.part.updated"
                if part_type == Some("tool")
                    && tool_state.and_then(|state| state.status()) == Some("error") =>
            {
                EventKind::Error
            }
            "message.part.updated" if part_type == Some("tool") => EventKind::ToolCall,
            "message.part.updated" if part_type == Some("compaction") => EventKind::Compaction,
            "tool.execute.after" => EventKind::ToolResult,
            "tool.execute.before" => EventKind::ToolCall,
            _ => EventKind::Other,
        };
        let canonical_role = match &kind {
            EventKind::UserMessage => Some("user".into()),
            EventKind::AssistantMessage | EventKind::ToolCall => Some("assistant".into()),
            EventKind::ToolResult | EventKind::Error
                if part_type == Some("tool") || event_type.starts_with("tool.execute.") =>
            {
                Some("tool".into())
            }
            _ => role,
        };
        let content = match &kind {
            EventKind::UserMessage | EventKind::AssistantMessage => {
                value_text(part.map(OpenCodePart::raw_value))
            }
            EventKind::ToolCall => value_text(
                tool_state
                    .and_then(OpenCodeToolState::input_value)
                    .or_else(|| props.input()),
            ),
            EventKind::ToolResult => value_text(
                tool_state
                    .and_then(OpenCodeToolState::output_value)
                    .or_else(|| props.output()),
            ),
            EventKind::Error => value_text(
                tool_state
                    .and_then(OpenCodeToolState::error_value)
                    .or_else(|| props.error()),
            ),
            EventKind::SessionStart => String::new(),
            _ => props.raw_value().to_string(),
        };
        let turn = part
            .and_then(OpenCodePart::message_id)
            .or_else(|| info.parent_id());
        let native_id = if matches!(kind, EventKind::UserMessage | EventKind::AssistantMessage) {
            turn.clone()
        } else {
            part.and_then(OpenCodePart::call_id)
                .or_else(|| part.and_then(OpenCodePart::id))
                .or_else(|| info.id())
        };
        let tool = part.and_then(OpenCodePart::tool).or_else(|| props.tool());
        let worktree = payload.worktree();
        let timestamp = part
            .and_then(|part| match &kind {
                EventKind::ToolResult | EventKind::Error if part_type == Some("tool") => {
                    timestamp_at(part.raw_value(), &["/state/time/end", "/state/time/start"])
                }
                EventKind::ToolCall => timestamp_at(part.raw_value(), &["/state/time/start"]),
                _ => timestamp_ms(part.raw_value()),
            })
            .or_else(|| timestamp_ms(info.raw_value()))
            .or_else(|| timestamp_ms(props.raw_value()));
        let source = event_source(
            AgentKind::OpenCode,
            session,
            turn,
            native_id,
            event_type,
            cwd.clone(),
            worktree,
        );
        let error_as_tool_result = part_type == Some("tool") || event_type.starts_with("tool.");
        let event = match &kind {
            EventKind::SessionStart => CanonicalEvent::session_started(source, timestamp),
            EventKind::SessionEnd => CanonicalEvent::session_ended(
                source,
                (!content.trim().is_empty()).then_some(content),
                timestamp,
            ),
            EventKind::UserMessage | EventKind::AssistantMessage => {
                let role = message_role(
                    canonical_role.as_deref(),
                    if matches!(kind, EventKind::UserMessage) {
                        MessageRole::User
                    } else {
                        MessageRole::Assistant
                    },
                );
                CanonicalEvent::message_from_text(source, role, content, timestamp)
            }
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
            EventKind::Compaction => {
                CanonicalEvent::compaction_from_content(source, content, timestamp)
            }
            EventKind::Error
                if error_as_tool_result
                    || canonical_role.as_deref() == Some("tool")
                    || tool.is_some() =>
            {
                CanonicalEvent::failed_tool_result(source, tool, content, timestamp)
            }
            EventKind::Error => CanonicalEvent::error_message(source, content, timestamp),
            EventKind::Other => CanonicalEvent::unknown(source, event_type, timestamp),
        };
        Ok(vec![event])
    }

    fn hook_attachment_sources(&self, event_name: &str, p: &Value) -> Vec<Value> {
        let Ok(payload) = OpenCodeHookPayload::try_from(p) else {
            return Vec::new();
        };
        let props = &payload.event.properties;
        let part = props.part.as_ref();
        let event_type = payload.event_type(event_name);
        match event_type {
            "message.part.updated" => match part.and_then(OpenCodePart::part_type) {
                Some("tool") => non_null_sources([
                    part.and_then(|part| part.state.input_value()),
                    part.and_then(|part| part.state.output_value()),
                    part.and_then(|part| part.state.error_value()),
                ]),
                Some("text") => non_null_sources([part.map(OpenCodePart::raw_value)]),
                _ => Vec::new(),
            },
            "tool.execute.before" => non_null_sources([props.input()]),
            "tool.execute.after" => non_null_sources([props.output()]),
            _ => Vec::new(),
        }
    }

    fn recall_response(&self, _event_name: &str, _hits: &[SearchHit]) -> Option<Value> {
        None
    }
}

impl HookInstaller for OpenCodeAdapter {
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)> {
        let code = format!(
            r#"// Generated by Semafuse. OpenCode loads this plugin automatically.
import {{ tool }} from "@opencode-ai/plugin";
const BIN = {binary:?};
const DB = {database:?};
const messageRoles = new Map();
const pendingTextParts = new Map();
const run = async (payload) => {{
  const proc = Bun.spawn([BIN, "hook", "--agent", "opencode", "--event", "auto", "--db", DB, "--protocol"], {{ stdin: JSON.stringify(payload), stdout: "ignore", stderr: "ignore" }});
  await proc.exited;
}};
const search = async (query, repository) => {{
  const proc = Bun.spawn([BIN, "search", query, "--db", DB, "--repository", repository || "", "--json"], {{ stdout: "pipe", stderr: "pipe" }});
  return await new Response(proc.stdout).text();
}};
const flushTextParts = async (sessionID) => {{
  for (const [partID, payload] of pendingTextParts) {{
    if (payload.event.properties.part?.sessionID !== sessionID) continue;
    pendingTextParts.delete(partID);
    await run(payload);
  }}
}};
export const Semafuse = async ({{ directory, worktree }}) => ({{
  event: async ({{ event }}) => {{
    const properties = event.properties || {{}};
    if (event.type === "message.updated" && properties.info?.id) {{
      messageRoles.set(properties.info.id, properties.info.role);
      return;
    }}
    const sessionID = properties.sessionID || properties.info?.id;
    const idle = event.type === "session.idle"
      || (event.type === "session.status" && properties.status?.type === "idle");
    if (idle) {{
      if (sessionID) await flushTextParts(sessionID);
      return;
    }}
    const allowed = new Set([
      "session.created", "session.deleted", "session.error", "session.compacted",
      "message.part.updated", "tool.execute.before", "tool.execute.after",
    ]);
    if (!allowed.has(event.type)) return;
    const messageID = properties.part?.messageID;
    const payload = {{
      event: {{ ...event, properties: {{ ...properties, messageRole: messageID ? messageRoles.get(messageID) : undefined }} }},
      directory,
      worktree,
    }};
    if (event.type === "message.part.updated"
      && properties.part?.type === "text"
      && payload.event.properties.messageRole !== "user") {{
      pendingTextParts.set(properties.part.id, payload);
      return;
    }}
    await run(payload);
  }},
  tool: {{
    semafuse_search: tool({{
      description: "Search prior coding-agent sessions with provenance",
      args: {{ query: tool.schema.string() }},
      async execute(args) {{ return search(args.query, worktree || directory); }}
    }})
  }},
  "experimental.session.compacting": async (_input, output) => {{
    const context = await search("decision constraint workaround", worktree || directory);
    if (context.trim()) output.context.push("Semafuse prior-session context:\n" + context);
  }}
}});
"#,
            binary = binary.display().to_string(),
            database = database.display().to_string()
        );
        vec![(".opencode/plugins/semafuse.js".into(), code)]
    }
}
