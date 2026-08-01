use super::{
    HistoryImporter, HookInstaller, HookProjector, ImportContext, NativeHistoryProjection,
    ProjectedEvent, event_source, message_role, non_null_sources, project_native_history,
    recall_text, scalar_string, text_content, timestamp_ms, value_text,
};
use crate::{
    blob::BlobStore,
    importer::{ImportReport, NativeSource, json_source_files, read_json_values},
    model::{AgentKind, CanonicalEvent, EventKind, MessageRole, ToolStatus},
    strategy::RecallItem,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::HashMap, path::Path};

/// Normalizes Amp history records and real-time hook events.
///
/// # Real-time hooks
///
/// | Native event | Condition | [`EventKind`] | Role | Content | Tool name |
/// | --- | --- | --- | --- | --- | --- |
/// | `session.start` | — | [`SessionStart`](EventKind::SessionStart) | — | Empty | — |
/// | `agent.start` | — | [`UserMessage`](EventKind::UserMessage) | `user` | `message` | — |
/// | `agent.end` | `status == "error"` | [`Error`](EventKind::Error) | `assistant` | Last nonempty assistant text in `messages` | — |
/// | `agent.end` | Other status | [`AssistantMessage`](EventKind::AssistantMessage) | `assistant` | Last nonempty assistant text in `messages` | — |
/// | `tool.call` | — | [`ToolCall`](EventKind::ToolCall) | `assistant` | `input` as text or JSON | `tool` |
/// | `tool.result` | `status == "error"` | [`Error`](EventKind::Error) | `tool` | `output`, then `error` | `tool` |
/// | `tool.result` | Other status | [`ToolResult`](EventKind::ToolResult) | `tool` | `output`, then `error` | `tool` |
/// | Other | — | [`Other`](EventKind::Other) | — | Complete payload JSON | — |
///
/// `session_id` uses `thread.id`, or `amp-unknown` if absent. `turn_id` uses
/// `id`; `native_id` prefers `toolUseID`, then `id`. An `agent.end` instead uses
/// the last assistant message's `id`/`messageId` so it can share identity with
/// the imported history record.
///
/// # History import
///
/// | Native record | Condition | [`EventKind`] | Role | Content | Tool name |
/// | --- | --- | --- | --- | --- | --- |
/// | Message text parts | Native role is `user` | [`UserMessage`](EventKind::UserMessage) | Native role | All text parts joined with newlines | — |
/// | Message text parts | Other role | [`AssistantMessage`](EventKind::AssistantMessage) | Native role | All text parts joined with newlines | — |
/// | `tool_use` part | — | [`ToolCall`](EventKind::ToolCall) | `assistant` | `input` as text or JSON | `name` |
/// | `tool_result` part | `run.status` is `error`, `cancelled`, or `rejected-by-user` | [`Error`](EventKind::Error) | `tool` | `run.result`, `run.error`, or `run.reason` | Prior `tool_use.name`, found by `toolUseID` |
/// | `tool_result` part | Other status | [`ToolResult`](EventKind::ToolResult) | `tool` | `run.result`, `run.error`, or `run.reason` | Prior `tool_use.name`, found by `toolUseID` |
/// | Message with only unrecognized parts | — | [`Other`](EventKind::Other) | Native role | Complete message JSON | — |
///
/// The session ID uses thread `id`. Ordinary messages use `messageId` for both
/// `turn_id` and `native_id`; tool events use their tool-call ID as `native_id`.
/// The working directory comes from `env.initial.trees[0].uri` with `file://`
/// removed. Message time is preferred over thread `created`.
pub struct AmpAdapter;

struct AmpHistory {
    thread: AmpThread,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpThread {
    id: Option<Value>,
    created: Option<Value>,
    env: Option<AmpThreadEnv>,
    #[serde(default)]
    messages: Vec<AmpHistoryMessage>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpThreadEnv {
    initial: Option<AmpInitialEnv>,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpInitialEnv {
    #[serde(default)]
    trees: Vec<AmpTree>,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpTree {
    uri: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHistoryMessage {
    #[serde(rename = "messageId")]
    message_id: Option<Value>,
    role: Option<Value>,
    #[serde(skip)]
    parts: Vec<AmpHistoryPart>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHistoryPart {
    #[serde(default, rename = "type")]
    part_type: String,
    id: Option<Value>,
    name: Option<Value>,
    #[serde(rename = "toolUseID")]
    tool_use_id: Option<Value>,
    input: Option<Value>,
    run: Option<AmpHistoryRun>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHistoryRun {
    status: Option<String>,
    result: Option<Value>,
    error: Option<Value>,
    reason: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHookPayload {
    thread: Option<AmpHookThread>,
    id: Option<Value>,
    #[serde(rename = "toolUseID")]
    tool_use_id: Option<Value>,
    cwd: Option<Value>,
    message: Option<Value>,
    #[serde(default)]
    messages: Vec<AmpHookMessage>,
    status: Option<Value>,
    input: Option<Value>,
    output: Option<Value>,
    error: Option<Value>,
    tool: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHookThread {
    id: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct AmpHookMessage {
    id: Option<Value>,
    #[serde(rename = "messageId")]
    message_id: Option<Value>,
    role: Option<Value>,
    #[serde(default)]
    content: Value,
    #[serde(skip)]
    raw: Value,
}

impl TryFrom<&Value> for AmpHookPayload {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let mut payload: Self = serde_json::from_value(value.clone())?;
        if let Some(raw_messages) = value.get("messages").and_then(Value::as_array) {
            for (message, raw_message) in payload.messages.iter_mut().zip(raw_messages) {
                message.raw = raw_message.clone();
            }
        }
        payload.raw = value.clone();
        Ok(payload)
    }
}

impl AmpHookPayload {
    fn session(&self) -> String {
        self.thread
            .as_ref()
            .and_then(|thread| scalar_string(thread.id.as_ref()))
            .unwrap_or_else(|| "amp-unknown".into())
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.cwd.as_ref())
    }

    fn turn(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn native_id(&self) -> Option<String> {
        scalar_string(self.tool_use_id.as_ref()).or_else(|| self.turn())
    }

    fn message_text(&self) -> String {
        scalar_string(self.message.as_ref()).unwrap_or_default()
    }

    fn tool(&self) -> Option<String> {
        scalar_string(self.tool.as_ref())
    }

    fn input(&self) -> Option<&Value> {
        self.input.as_ref()
    }

    fn output_or_error(&self) -> Option<&Value> {
        self.output.as_ref().or(self.error.as_ref())
    }

    fn is_error(&self) -> bool {
        self.status.as_ref().and_then(Value::as_str) == Some("error")
    }

    fn last_assistant_message(&self) -> Option<&AmpHookMessage> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role() == Some("assistant"))
    }

    fn last_assistant_message_id(&self) -> Option<String> {
        self.last_assistant_message()
            .and_then(AmpHookMessage::native_id)
    }

    fn last_assistant_text(&self) -> String {
        self.last_assistant_message()
            .map(AmpHookMessage::content_text)
            .filter(|text| !text.is_empty())
            .unwrap_or_default()
    }

    fn messages_source(&self) -> Vec<Value> {
        (!self.messages.is_empty())
            .then(|| {
                Value::Array(
                    self.messages
                        .iter()
                        .map(|message| message.raw.clone())
                        .collect(),
                )
            })
            .into_iter()
            .collect()
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl AmpHookMessage {
    fn role(&self) -> Option<&str> {
        self.role.as_ref().and_then(Value::as_str)
    }

    fn native_id(&self) -> Option<String> {
        scalar_string(self.id.as_ref()).or_else(|| scalar_string(self.message_id.as_ref()))
    }

    fn content_text(&self) -> String {
        text_content(&self.content)
    }
}

impl TryFrom<Value> for AmpThread {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut thread: Self = serde_json::from_value(value.clone())?;
        if let Some(raw_messages) = value.get("messages").and_then(Value::as_array) {
            for (message, raw_message) in thread.messages.iter_mut().zip(raw_messages) {
                message.attach_raw(raw_message.clone())?;
            }
        }
        thread.raw = value;
        Ok(thread)
    }
}

impl AmpThread {
    fn session(&self, fallback: &str) -> String {
        scalar_string(self.id.as_ref()).unwrap_or_else(|| fallback.into())
    }

    fn cwd(&self) -> Option<String> {
        self.env
            .as_ref()
            .and_then(|env| env.initial.as_ref())
            .and_then(|initial| initial.trees.first())
            .and_then(|tree| scalar_string(tree.uri.as_ref()))
            .and_then(|uri| uri.strip_prefix("file://").map(str::to_owned).or(Some(uri)))
    }

    fn created_at_ms(&self) -> Option<i64> {
        self.created
            .as_ref()
            .and_then(timestamp_ms)
            .or_else(|| timestamp_ms(&self.raw))
    }
}

impl AmpHistoryMessage {
    fn attach_raw(&mut self, value: Value) -> Result<()> {
        self.parts = value
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .cloned()
            .map(AmpHistoryPart::try_from)
            .collect::<Result<Vec<_>>>()?;
        self.raw = value;
        Ok(())
    }

    fn role(&self) -> Option<String> {
        scalar_string(self.role.as_ref())
    }

    fn message_id(&self) -> Option<String> {
        scalar_string(self.message_id.as_ref())
    }

    fn timestamp_ms(&self, thread_created: Option<i64>) -> Option<i64> {
        timestamp_ms(&self.raw).or(thread_created)
    }

    fn text(&self) -> String {
        self.parts
            .iter()
            .filter(|part| part.part_type() == Some("text"))
            .map(|part| text_content(part.raw_value()))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn attachment_source(&self) -> Value {
        Value::Array(
            self.parts
                .iter()
                .filter(|part| !matches!(part.part_type(), Some("tool_use" | "tool_result")))
                .map(|part| part.raw_value().clone())
                .collect(),
        )
    }

    fn parts(&self) -> &[AmpHistoryPart] {
        &self.parts
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<Value> for AmpHistoryPart {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut part: Self = serde_json::from_value(value.clone())?;
        part.raw = value;
        Ok(part)
    }
}

impl AmpHistoryPart {
    fn part_type(&self) -> Option<&str> {
        (!self.part_type.is_empty()).then_some(self.part_type.as_str())
    }

    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn name(&self) -> Option<String> {
        scalar_string(self.name.as_ref())
    }

    fn tool_use_id(&self) -> Option<String> {
        scalar_string(self.tool_use_id.as_ref())
    }

    fn input(&self) -> Option<&Value> {
        self.input.as_ref()
    }

    fn run(&self) -> Option<&AmpHistoryRun> {
        self.run.as_ref()
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl AmpHistoryRun {
    fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    fn result(&self) -> Option<&Value> {
        self.result
            .as_ref()
            .or(self.error.as_ref())
            .or(self.reason.as_ref())
    }
}

impl NativeHistoryProjection for AmpHistory {
    fn decode_native(values: &[Value], _context: &ImportContext) -> Result<Self> {
        Ok(Self {
            thread: values.first().cloned().unwrap_or(Value::Null).try_into()?,
        })
    }

    fn project_native(&self, context: &ImportContext) -> Result<Vec<ProjectedEvent>> {
        let thread = &self.thread;
        let session = thread.session(&context.fallback_session_id);
        let cwd = thread.cwd().or_else(|| context.cwd.clone());
        let created = thread.created_at_ms();
        let mut tool_names = HashMap::new();
        let mut out = Vec::new();

        for message in &thread.messages {
            let mut recognized = false;
            let role = message.role();
            let message_id = message.message_id();
            let timestamp = message.timestamp_ms(created);
            let text = message.text();
            if !text.is_empty() {
                recognized = true;
                let attachment_source = message.attachment_source();
                let kind = if role.as_deref() == Some("user") {
                    EventKind::UserMessage
                } else {
                    EventKind::AssistantMessage
                };
                let source = event_source(
                    AgentKind::Amp,
                    &session,
                    message_id.clone(),
                    message_id.clone(),
                    "message",
                    cwd.clone(),
                    None,
                );
                let role = message_role(
                    role.as_deref(),
                    if matches!(kind, EventKind::UserMessage) {
                        MessageRole::User
                    } else {
                        MessageRole::Assistant
                    },
                );
                out.push(ProjectedEvent::new(
                    CanonicalEvent::message_from_text(source, role, text, timestamp),
                    &attachment_source,
                    context.raw_record_id(0),
                ));
            }

            for part in message.parts() {
                match part.part_type() {
                    Some("tool_use") => {
                        recognized = true;
                        let native_id = part.id();
                        let tool = part.name();
                        if let (Some(id), Some(name)) = (&native_id, &tool) {
                            tool_names.insert(id.clone(), name.clone());
                        }
                        let source = event_source(
                            AgentKind::Amp,
                            &session,
                            message_id.clone(),
                            native_id,
                            "tool_use",
                            cwd.clone(),
                            None,
                        );
                        out.push(ProjectedEvent::new(
                            CanonicalEvent::tool_call_from_content(
                                source,
                                tool.unwrap_or_else(|| "unknown".into()),
                                value_text(part.input()),
                                timestamp,
                            ),
                            &part.input().cloned().unwrap_or(Value::Null),
                            context.raw_record_id(0),
                        ));
                    }
                    Some("tool_result") => {
                        recognized = true;
                        let native_id = part.tool_use_id();
                        let tool = native_id
                            .as_ref()
                            .and_then(|id| tool_names.get(id))
                            .cloned();
                        let run = part.run();
                        let status = run.and_then(AmpHistoryRun::status);
                        let result = run
                            .and_then(AmpHistoryRun::result)
                            .cloned()
                            .unwrap_or(Value::Null);
                        let source = event_source(
                            AgentKind::Amp,
                            &session,
                            message_id.clone(),
                            native_id,
                            "tool_result",
                            cwd.clone(),
                            None,
                        );
                        let content = value_text(Some(&result));
                        let event =
                            if matches!(status, Some("error" | "cancelled" | "rejected-by-user")) {
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
                            &result,
                            context.raw_record_id(0),
                        ));
                    }
                    _ => {}
                }
            }
            if !recognized {
                let source = event_source(
                    AgentKind::Amp,
                    &session,
                    message_id.clone(),
                    message_id,
                    "message",
                    cwd.clone(),
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

#[async_trait]
impl HistoryImporter for AmpAdapter {
    async fn import_default(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport> {
        let root = dirs::data_local_dir()
            .context("local data directory not found")?
            .join("amp/threads");
        self.import_path(&root, blobs).await
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
                                agent: AgentKind::Amp,
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
        project_native_history::<AmpHistory>(values, context)
    }
}

impl HookProjector for AmpAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Amp
    }

    fn hook_event(&self, event_name: &str, payload: &Value) -> Result<Vec<CanonicalEvent>> {
        let payload: AmpHookPayload = payload.try_into()?;
        let session = payload.session();
        let cwd = payload.cwd();
        let mut turn = payload.turn();
        let mut native_id = payload.native_id();
        if event_name == "agent.end"
            && let Some(message_id) = payload.last_assistant_message_id()
        {
            turn = Some(message_id.clone());
            native_id = Some(message_id);
        }
        let source = event_source(
            AgentKind::Amp,
            session,
            turn,
            native_id,
            event_name,
            cwd.clone(),
            None,
        );
        let timestamp = timestamp_ms(payload.raw_value());
        let event = match event_name {
            "session.start" => CanonicalEvent::session_started(source, timestamp),
            "agent.start" => CanonicalEvent::message_from_text(
                source,
                MessageRole::User,
                payload.message_text(),
                timestamp,
            ),
            "agent.end" if payload.is_error() => {
                CanonicalEvent::error_message(source, payload.last_assistant_text(), timestamp)
            }
            "agent.end" => CanonicalEvent::message_from_text(
                source,
                MessageRole::Assistant,
                payload.last_assistant_text(),
                timestamp,
            ),
            "tool.call" => CanonicalEvent::tool_call_from_content(
                source,
                payload.tool().unwrap_or_else(|| "unknown".into()),
                value_text(payload.input()),
                timestamp,
            ),
            "tool.result" if payload.is_error() => CanonicalEvent::failed_tool_result(
                source,
                payload.tool(),
                value_text(payload.output_or_error()),
                timestamp,
            ),
            "tool.result" => CanonicalEvent::tool_result_from_content(
                source,
                payload.tool(),
                ToolStatus::Succeeded,
                value_text(payload.output_or_error()),
                None,
                timestamp,
            ),
            _ => CanonicalEvent::unknown(source, event_name, timestamp),
        };
        Ok(vec![event])
    }

    fn hook_attachment_sources(&self, event_name: &str, payload: &Value) -> Vec<Value> {
        let Ok(payload) = AmpHookPayload::try_from(payload) else {
            return Vec::new();
        };
        match event_name {
            "agent.start" | "agent.end" => payload.messages_source(),
            "tool.call" => non_null_sources([payload.input()]),
            "tool.result" => non_null_sources([payload.output_or_error()]),
            _ => Vec::new(),
        }
    }
    fn supports_recall(&self, event_name: &str) -> bool {
        event_name == "agent.start"
    }

    fn recall_response(&self, event_name: &str, hits: &[RecallItem]) -> Option<Value> {
        if event_name != "agent.start" {
            return None;
        }
        let text = recall_text(hits);
        (!text.is_empty()).then(|| json!({"additionalContext":text}))
    }
}

impl HookInstaller for AmpAdapter {
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)> {
        let code = format!(
            r#"// Generated by Semafuse. Amp loads project plugins automatically.
import type {{ PluginAPI }} from "@ampcode/plugin";

const BIN = {binary:?};
const DB = {database:?};

export default function (amp: PluginAPI) {{
  const cwd = amp.system.workspaceRoot
    ? amp.helpers.filePathFromURI(amp.system.workspaceRoot)
    : undefined;
  const record = async (eventName: string, event: object, recall = false) => {{
    try {{
      const args = ["hook", "--agent", "amp", "--event", eventName, "--db", DB, "--protocol"];
      if (recall) args.push("--recall");
      const proc = Bun.spawn([BIN, ...args], {{
        stdin: JSON.stringify({{ ...event, cwd }}),
        stdout: "pipe",
        stderr: "ignore",
      }});
      const output = await new Response(proc.stdout).text();
      await proc.exited;
      return output.trim() ? JSON.parse(output) : undefined;
    }} catch {{
      return undefined;
    }}
  }};

  amp.on("session.start", async event => {{
    await record("session.start", event);
  }});
  amp.on("agent.start", async event => {{
    const response = await record("agent.start", event, true);
    if (response?.additionalContext) {{
      return {{ message: {{ content: response.additionalContext, display: false }} }};
    }}
  }});
  amp.on("tool.call", async event => {{
    await record("tool.call", event);
    return {{ action: "allow" }};
  }});
  amp.on("tool.result", async event => {{
    await record("tool.result", event);
  }});
  amp.on("agent.end", async event => {{
    await record("agent.end", event);
  }});
}}
"#,
            binary = binary.display().to_string(),
            database = database.display().to_string(),
        );
        vec![(".amp/plugins/semafuse.ts".into(), code)]
    }
}
