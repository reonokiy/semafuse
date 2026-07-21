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
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::Command as TokioCommand,
};

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

enum CodexHistory {
    Thread(CodexThread),
    Rollout(Vec<CodexRolloutRecord>),
}

#[derive(Debug, Clone, Deserialize)]
struct CodexThreadEnvelope {
    result: Option<CodexThreadResult>,
    thread: Option<CodexThread>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexThreadResult {
    thread: Option<CodexThread>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexThread {
    id: Option<Value>,
    cwd: Option<Value>,
    #[serde(default)]
    turns: Vec<CodexTurn>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexTurn {
    id: Option<Value>,
    #[serde(default)]
    items: Vec<CodexItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexRolloutRecord {
    #[serde(rename = "type")]
    record_type: Option<String>,
    payload: Option<CodexPayload>,
    #[serde(flatten)]
    direct: CodexPayload,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexItem {
    #[serde(default)]
    payload: Option<CodexPayload>,
    #[serde(flatten)]
    direct: CodexPayload,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CodexPayload {
    #[serde(rename = "type")]
    item_type: Option<String>,
    id: Option<Value>,
    call_id: Option<Value>,
    turn_id: Option<Value>,
    role: Option<Value>,
    content: Option<Value>,
    text: Option<Value>,
    arguments: Option<Value>,
    input: Option<Value>,
    output: Option<Value>,
    command: Option<Value>,
    #[serde(rename = "aggregatedOutput")]
    aggregated_output: Option<Value>,
    result: Option<Value>,
    error: Option<Value>,
    server: Option<Value>,
    tool: Option<Value>,
    name: Option<Value>,
    changes: Option<Value>,
    path: Option<Value>,
    cwd: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

#[derive(Debug, Deserialize)]
struct CodexHookPayload {
    session_id: Option<Value>,
    turn_id: Option<Value>,
    cwd: Option<Value>,
    tool_use_id: Option<Value>,
    agent_id: Option<Value>,
    id: Option<Value>,
    reason: Option<Value>,
    prompt: Option<Value>,
    tool_input: Option<Value>,
    tool_name: Option<Value>,
    tool_response: Option<Value>,
    tool_output: Option<Value>,
    error: Option<Value>,
    last_assistant_message: Option<Value>,
    compact_summary: Option<Value>,
    custom_instructions: Option<Value>,
    content: Option<Value>,
    #[serde(skip)]
    raw: Value,
}

impl CodexThreadEnvelope {
    fn into_thread(self, value: &Value) -> Result<Option<CodexThread>> {
        let mut thread = self.result.and_then(|result| result.thread).or(self.thread);
        if let Some(thread) = thread.as_mut() {
            let raw_thread = value
                .pointer("/result/thread")
                .or_else(|| value.get("thread"))
                .unwrap_or(&Value::Null);
            thread.attach_raw(raw_thread)?;
        }
        Ok(thread)
    }
}

impl CodexThread {
    fn attach_raw(&mut self, value: &Value) -> Result<()> {
        if let Some(raw_turns) = value.get("turns").and_then(Value::as_array) {
            for (turn, raw_turn) in self.turns.iter_mut().zip(raw_turns) {
                turn.attach_raw(raw_turn)?;
            }
        }
        Ok(())
    }

    fn session(&self, fallback: &str) -> String {
        scalar_string(self.id.as_ref()).unwrap_or_else(|| fallback.into())
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.cwd.as_ref())
    }
}

impl CodexTurn {
    fn attach_raw(&mut self, value: &Value) -> Result<()> {
        if let Some(raw_items) = value.get("items").and_then(Value::as_array) {
            for (item, raw_item) in self.items.iter_mut().zip(raw_items) {
                item.attach_raw(raw_item.clone())?;
            }
        }
        Ok(())
    }

    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }
}

impl TryFrom<Value> for CodexRolloutRecord {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let mut record: Self = serde_json::from_value(value.clone())?;
        if let Some(raw_payload) = value.get("payload")
            && let Some(payload) = record.payload.as_mut()
        {
            payload.raw = raw_payload.clone();
        }
        record.direct.raw = value.clone();
        record.raw = value;
        Ok(record)
    }
}

impl CodexRolloutRecord {
    fn record_type(&self) -> &str {
        self.record_type.as_deref().unwrap_or_default()
    }

    fn payload_or_record(&self) -> CodexItemRef<'_> {
        CodexItemRef {
            item_type_override: self.record_type.as_deref(),
            payload: self.payload.as_ref().unwrap_or(&self.direct),
            item_raw: &self.raw,
        }
    }

    fn payload_fields(&self) -> &CodexPayload {
        self.payload.as_ref().unwrap_or(&self.direct)
    }

    fn payload_source(&self) -> &Value {
        self.payload
            .as_ref()
            .map(CodexPayload::raw_value)
            .unwrap_or(&self.raw)
    }

    fn timestamp_ms(&self) -> Option<i64> {
        timestamp_ms(&self.raw)
    }
}

impl CodexItem {
    fn attach_raw(&mut self, value: Value) -> Result<()> {
        if let Some(raw_payload) = value.get("payload")
            && let Some(payload) = self.payload.as_mut()
        {
            payload.raw = raw_payload.clone();
        }
        self.direct.raw = value.clone();
        self.raw = value;
        Ok(())
    }

    fn as_ref(&self) -> CodexItemRef<'_> {
        CodexItemRef {
            item_type_override: None,
            payload: self.payload.as_ref().unwrap_or(&self.direct),
            item_raw: &self.raw,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CodexItemRef<'a> {
    item_type_override: Option<&'a str>,
    payload: &'a CodexPayload,
    item_raw: &'a Value,
}

impl<'a> CodexItemRef<'a> {
    fn item_type(&self) -> &'a str {
        self.payload
            .item_type()
            .or(self.item_type_override)
            .unwrap_or_default()
    }

    fn role(&self) -> Option<String> {
        self.payload.role()
    }

    fn native_id(&self) -> Option<String> {
        self.payload.id().or_else(|| self.payload.call_id())
    }

    fn timestamp_ms(&self) -> Option<i64> {
        self.payload
            .timestamp_ms()
            .or_else(|| timestamp_ms(self.item_raw))
    }
}

impl CodexPayload {
    fn item_type(&self) -> Option<&str> {
        self.item_type.as_deref()
    }

    fn id(&self) -> Option<String> {
        scalar_string(self.id.as_ref())
    }

    fn call_id(&self) -> Option<String> {
        scalar_string(self.call_id.as_ref())
    }

    fn turn_id(&self) -> Option<String> {
        scalar_string(self.turn_id.as_ref()).or_else(|| self.id())
    }

    fn role(&self) -> Option<String> {
        scalar_string(self.role.as_ref())
    }

    fn cwd(&self) -> Option<String> {
        scalar_string(self.cwd.as_ref())
    }

    fn text_content(&self) -> String {
        text_content(self.content.as_ref().unwrap_or(&Value::Null))
    }

    fn text(&self) -> String {
        scalar_string(self.text.as_ref()).unwrap_or_default()
    }

    fn arguments(&self) -> Option<&Value> {
        self.arguments.as_ref()
    }

    fn input_or_arguments(&self) -> Option<&Value> {
        self.input.as_ref().or(self.arguments.as_ref())
    }

    fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    fn command(&self) -> String {
        scalar_string(self.command.as_ref()).unwrap_or_default()
    }

    fn aggregated_output(&self) -> String {
        scalar_string(self.aggregated_output.as_ref()).unwrap_or_default()
    }

    fn result_or_error_or_arguments(&self) -> Option<&Value> {
        self.result
            .as_ref()
            .or(self.error.as_ref())
            .or(self.arguments.as_ref())
    }

    fn result_or_error(&self) -> Option<&Value> {
        self.result.as_ref().or(self.error.as_ref())
    }

    fn server(&self) -> String {
        scalar_string(self.server.as_ref()).unwrap_or_default()
    }

    fn tool(&self) -> Option<String> {
        scalar_string(self.tool.as_ref())
    }

    fn tool_or_default(&self) -> String {
        scalar_string(self.tool.as_ref()).unwrap_or_default()
    }

    fn name(&self) -> Option<String> {
        scalar_string(self.name.as_ref())
    }

    fn changes(&self) -> Option<&Value> {
        self.changes.as_ref()
    }

    fn path(&self) -> String {
        scalar_string(self.path.as_ref()).unwrap_or_default()
    }

    fn content_value(&self) -> &Value {
        self.content.as_ref().unwrap_or(&Value::Null)
    }

    fn timestamp_ms(&self) -> Option<i64> {
        timestamp_ms(&self.raw)
    }

    fn raw_value(&self) -> &Value {
        &self.raw
    }
}

impl TryFrom<&Value> for CodexHookPayload {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        let mut payload: Self = serde_json::from_value(value.clone())?;
        payload.raw = value.clone();
        Ok(payload)
    }
}

impl CodexHookPayload {
    fn raw_value(&self) -> &Value {
        &self.raw
    }

    fn session(&self) -> String {
        scalar_string(self.session_id.as_ref()).unwrap_or_else(|| "codex-unknown".into())
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
    }

    fn synthetic_compaction_id(&self, session: &str, normalized_event: &str) -> String {
        format!(
            "{}:{normalized_event}",
            self.turn().as_deref().unwrap_or(session)
        )
    }
}

fn event_from_item(
    item: CodexItemRef<'_>,
    session: &str,
    turn: Option<String>,
    cwd: Option<String>,
    raw_record_id: &str,
) -> Option<ProjectedEvent> {
    let payload = item.payload;
    let item_type = item.item_type();
    let role = item.role();
    let (kind, content, tool) = match item_type {
        "message" => {
            let content = payload.text_content();
            let kind = if role.as_deref() == Some("user") {
                EventKind::UserMessage
            } else {
                EventKind::AssistantMessage
            };
            (kind, content, None)
        }
        "userMessage" => (EventKind::UserMessage, payload.text_content(), None),
        "agentMessage" | "plan" => (EventKind::AssistantMessage, payload.text(), None),
        "function_call" => (
            EventKind::ToolCall,
            value_text(payload.arguments()),
            payload.name(),
        ),
        "custom_tool_call" => (
            EventKind::ToolCall,
            value_text(payload.input_or_arguments()),
            payload.name(),
        ),
        "function_call_output" | "custom_tool_call_output" => {
            (EventKind::ToolResult, value_text(payload.output()), None)
        }
        "commandExecution" => (
            EventKind::ToolResult,
            format!("$ {}\n{}", payload.command(), payload.aggregated_output()),
            Some("commandExecution".into()),
        ),
        "mcpToolCall" => (
            EventKind::ToolResult,
            payload
                .result_or_error_or_arguments()
                .map(Value::to_string)
                .unwrap_or_default(),
            Some(format!(
                "{}/{}",
                payload.server(),
                payload.tool_or_default()
            )),
        ),
        "dynamicToolCall" | "collabToolCall" => (
            EventKind::ToolResult,
            payload.raw_value().to_string(),
            payload.tool(),
        ),
        "fileChange" => (
            EventKind::ToolResult,
            payload.changes().map(Value::to_string).unwrap_or_default(),
            Some("fileChange".into()),
        ),
        "contextCompaction" | "compacted" => {
            (EventKind::Compaction, payload.raw_value().to_string(), None)
        }
        "imageView" => (
            EventKind::ToolResult,
            payload.path(),
            Some("imageView".into()),
        ),
        _ => (EventKind::Other, payload.raw_value().to_string(), None),
    };
    if content.is_empty()
        && !matches!(kind, EventKind::ToolCall)
        && !matches!(item_type, "message" | "userMessage")
    {
        return None;
    }
    let attachment_source = match item_type {
        "message" | "userMessage" => payload.content_value(),
        "function_call" => payload.arguments().unwrap_or(&Value::Null),
        "custom_tool_call" => payload.input_or_arguments().unwrap_or(&Value::Null),
        "function_call_output" | "custom_tool_call_output" => {
            payload.output().unwrap_or(&Value::Null)
        }
        "commandExecution" => payload.aggregated_output.as_ref().unwrap_or(&Value::Null),
        "mcpToolCall" => payload.result_or_error().unwrap_or(&Value::Null),
        _ => payload.raw_value(),
    };
    let source = event_source(
        AgentKind::Codex,
        session,
        turn,
        item.native_id(),
        item_type,
        cwd,
        None,
    );
    let timestamp = item.timestamp_ms();
    let event = match &kind {
        EventKind::UserMessage | EventKind::AssistantMessage => {
            let role = message_role(
                role.as_deref(),
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
        _ => CanonicalEvent::unknown(source, item_type, timestamp),
    };
    Some(ProjectedEvent::new(event, attachment_source, raw_record_id))
}

impl NativeHistoryProjection for CodexHistory {
    fn decode_native(values: &[Value], _context: &ImportContext) -> Result<Self> {
        // Supports Codex rollout JSONL and app-server thread/read responses.
        if values.len() == 1 {
            let envelope: CodexThreadEnvelope = serde_json::from_value(values[0].clone())?;
            if let Some(thread) = envelope.into_thread(&values[0])? {
                return Ok(Self::Thread(thread));
            }
        }
        Ok(Self::Rollout(
            values
                .iter()
                .cloned()
                .map(CodexRolloutRecord::try_from)
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    fn project_native(&self, context: &ImportContext) -> Result<Vec<ProjectedEvent>> {
        match self {
            Self::Thread(thread) => {
                let session = thread.session(&context.fallback_session_id);
                let cwd = thread.cwd().or_else(|| context.cwd.clone());
                let mut out = Vec::new();
                for turn_value in &thread.turns {
                    let turn = turn_value.id();
                    for item in &turn_value.items {
                        if let Some(event) = event_from_item(
                            item.as_ref(),
                            &session,
                            turn.clone(),
                            cwd.clone(),
                            context.raw_record_id(0),
                        ) {
                            out.push(event);
                        }
                    }
                }
                Ok(out)
            }
            Self::Rollout(values) => {
                let mut session = context.fallback_session_id.clone();
                let mut cwd = context.cwd.clone();
                let mut turn = None;
                let mut out = Vec::new();
                for (record_index, record) in values.iter().enumerate() {
                    let record_type = record.record_type();
                    match record_type {
                        "session_meta" => {
                            let payload = record.payload_fields();
                            session = payload.id().unwrap_or(session);
                            cwd = payload.cwd().or(cwd);
                            let source = event_source(
                                AgentKind::Codex,
                                &session,
                                None,
                                None,
                                "session_meta",
                                cwd.clone(),
                                None,
                            );
                            out.push(ProjectedEvent::new(
                                CanonicalEvent::session_started(
                                    source,
                                    payload.timestamp_ms().or_else(|| record.timestamp_ms()),
                                ),
                                record.payload_source(),
                                context.raw_record_id(record_index),
                            ));
                        }
                        "turn_context" => {
                            turn = Some(record.payload_fields()).and_then(CodexPayload::turn_id)
                        }
                        "response_item" => {
                            if let Some(event) = event_from_item(
                                record.payload_or_record(),
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
                                record.payload_or_record(),
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
        }
    }
}

impl CodexAdapter {
    async fn import_app_server(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport> {
        let mut command = TokioCommand::new("codex");
        command
            .args(["app-server", "--listen", "stdio://"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        command.kill_on_drop(true);
        let mut child = command.spawn().context("start Codex App Server")?;
        let Some(stdin) = child.stdin.take() else {
            let _ = child.start_kill();
            let _ = child.wait().await;
            bail!("Codex App Server stdin unavailable");
        };
        let Some(stdout) = child.stdout.take() else {
            let _ = child.start_kill();
            let _ = child.wait().await;
            bail!("Codex App Server stdout unavailable");
        };
        let result = self.run_app_server_protocol(stdin, stdout, blobs).await;
        let _ = child.start_kill();
        let cleanup = child.wait().await;
        match result {
            Ok(report) => {
                cleanup.context("wait for Codex App Server shutdown")?;
                Ok(report)
            }
            Err(error) => {
                let _ = cleanup;
                Err(error)
            }
        }
    }

    async fn run_app_server_protocol(
        &self,
        mut stdin: tokio::process::ChildStdin,
        stdout: tokio::process::ChildStdout,
        blobs: Option<&dyn BlobStore>,
    ) -> Result<ImportReport> {
        let mut reader = BufReader::new(stdout);
        send_json(
            &mut stdin,
            &json!({"method":"initialize","id":0,"params":{"clientInfo":{"name":"semafuse","title":"Semafuse","version":env!("CARGO_PKG_VERSION")}}}),
        )
        .await?;
        let _ = read_response(&mut reader, 0).await?;
        send_json(&mut stdin, &json!({"method":"initialized","params":{}})).await?;
        let mut report = ImportReport::default();
        let mut request_id = 1_i64;
        let mut thread_ids = Vec::new();
        let mut seen = HashSet::new();
        for archived in [false, true] {
            let mut cursor: Option<String> = None;
            loop {
                send_json(
                    &mut stdin,
                    &json!({"method":"thread/list","id":request_id,"params":{"cursor":cursor,"limit":100,"archived":archived}}),
                )
                .await?;
                let response = read_response(&mut reader, request_id).await?;
                request_id += 1;
                if let Some(error) = response.get("error") {
                    bail!("Codex thread/list error: {error}");
                }
                let result = response.get("result").cloned().unwrap_or(Value::Null);
                for summary in result
                    .get("data")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let Some(id) = summary.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    if seen.insert(id.to_owned()) {
                        thread_ids.push(id.to_owned());
                    }
                }
                cursor = result
                    .get("nextCursor")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                if cursor.is_none() {
                    break;
                }
            }
        }
        for id in thread_ids {
            send_json(
                &mut stdin,
                &json!({"method":"thread/read","id":request_id,"params":{"threadId":id,"includeTurns":true}}),
            )
            .await?;
            let response = read_response(&mut reader, request_id).await?;
            request_id += 1;
            report
                .add_native_source(
                    NativeSource {
                        agent: AgentKind::Codex,
                        source: format!("codex-app-server:{id}"),
                        fallback_session_id: id.clone(),
                        cwd: None,
                        values: vec![response],
                    },
                    blobs,
                    |values, context| self.import_values(values, context),
                )
                .await;
        }
        Ok(report)
    }
}

async fn send_json(writer: &mut (impl AsyncWrite + Unpin), value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    tokio::time::timeout(Duration::from_secs(30), async {
        writer.write_all(&bytes).await?;
        writer.flush().await?;
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("timed out writing agent protocol request")?
}

async fn read_response(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
    id: i64,
) -> Result<Value> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).await? == 0 {
                bail!("agent protocol closed before response id {id}");
            }
            let value: Value = serde_json::from_str(&line)?;
            if value.get("id").and_then(Value::as_i64) == Some(id) {
                return Ok(value);
            }
        }
    })
    .await
    .with_context(|| format!("timed out waiting for agent response id {id}"))?
}

#[async_trait]
impl HistoryImporter for CodexAdapter {
    async fn import_default(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport> {
        let home = dirs::home_dir().context("home directory not found")?;
        let root = std::env::var("CODEX_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".codex"));
        match self.import_app_server(blobs).await {
            Ok(report) => Ok(report),
            Err(error) => {
                eprintln!(
                    "warning: Codex App Server import unavailable ({error:#}); falling back to rollout JSONL"
                );
                self.import_path(&root.join("sessions"), blobs).await
            }
        }
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
                                agent: AgentKind::Codex,
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
        project_native_history::<CodexHistory>(values, context)
    }
}

impl HookProjector for CodexAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Codex
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let payload: CodexHookPayload = p.try_into()?;
        let session = payload.session();
        let turn = payload.turn();
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
            AgentKind::Codex,
            session,
            turn,
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
                value_text(
                    payload
                        .tool_response
                        .as_ref()
                        .or(payload.tool_output.as_ref()),
                ),
                None,
                timestamp,
            ),
            "posttoolusefailure" | "post_tool_use_failure" => CanonicalEvent::failed_tool_result(
                source,
                scalar_string(payload.tool_name.as_ref()),
                value_text(payload.error.as_ref().or(payload.tool_response.as_ref())),
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
        let Ok(payload) = CodexHookPayload::try_from(p) else {
            return Vec::new();
        };
        match event_name.to_ascii_lowercase().as_str() {
            "userpromptsubmit" | "user_prompt_submit" => {
                non_null_sources([payload.content.as_ref(), payload.prompt.as_ref()])
            }
            "pretooluse" | "pre_tool_use" => {
                non_null_sources([payload.tool_input.as_ref(), payload.content.as_ref()])
            }
            "posttooluse" | "post_tool_use" => non_null_sources([payload
                .tool_response
                .as_ref()
                .or(payload.tool_output.as_ref())]),
            "posttoolusefailure" | "post_tool_use_failure" => {
                non_null_sources([payload.error.as_ref().or(payload.tool_response.as_ref())])
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

impl HookInstaller for CodexAdapter {
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)> {
        let base_command = [
            binary.display().to_string(),
            "hook".into(),
            "--agent".into(),
            "codex".into(),
            "--event".into(),
            "auto".into(),
            "--db".into(),
            database.display().to_string(),
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
            let command = if self.supports_recall(name) {
                format!("{base_command} --recall")
            } else {
                base_command.clone()
            };
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
