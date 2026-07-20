use super::{
    AgentAdapter, ImportContext, ImportedEvent, event_metadata, non_null_sources, recall_text,
    scalar_string, text_content, timestamp_ms, value_text,
};
use crate::model::{AgentKind, CanonicalEvent, EventKind, SearchHit};
use anyhow::Result;
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

impl AgentAdapter for AmpAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::Amp
    }

    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ImportedEvent>> {
        let thread = values.first().unwrap_or(&Value::Null);
        let session =
            scalar_string(thread.get("id")).unwrap_or_else(|| context.fallback_session_id.clone());
        let tree = thread
            .pointer("/env/initial/trees/0")
            .unwrap_or(&Value::Null);
        let cwd = scalar_string(tree.get("uri"))
            .and_then(|uri| uri.strip_prefix("file://").map(str::to_owned).or(Some(uri)))
            .or_else(|| context.cwd.clone());
        let created = timestamp_ms(thread);
        let mut tool_names = HashMap::new();
        let mut out = Vec::new();

        for message in thread
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let mut recognized = false;
            let role = scalar_string(message.get("role"));
            let message_id = scalar_string(message.get("messageId"));
            let timestamp = timestamp_ms(message).or(created);
            let parts = message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten();

            let text = parts
                .clone()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .map(text_content)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                recognized = true;
                let attachment_source = Value::Array(
                    parts
                        .clone()
                        .filter(|part| {
                            !matches!(
                                part.get("type").and_then(Value::as_str),
                                Some("tool_use" | "tool_result")
                            )
                        })
                        .cloned()
                        .collect(),
                );
                let kind = if role.as_deref() == Some("user") {
                    EventKind::UserMessage
                } else {
                    EventKind::AssistantMessage
                };
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::Amp,
                        &session,
                        message_id.clone(),
                        message_id.clone(),
                        kind,
                        role.clone(),
                        text,
                        None,
                        cwd.clone(),
                        None,
                        timestamp,
                        event_metadata("message"),
                    ),
                    &attachment_source,
                    context.raw_record_id(0),
                ));
            }

            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        recognized = true;
                        let native_id = scalar_string(part.get("id"));
                        let tool = scalar_string(part.get("name"));
                        if let (Some(id), Some(name)) = (&native_id, &tool) {
                            tool_names.insert(id.clone(), name.clone());
                        }
                        out.push(ImportedEvent::new(
                            CanonicalEvent::new(
                                AgentKind::Amp,
                                &session,
                                message_id.clone(),
                                native_id,
                                EventKind::ToolCall,
                                Some("assistant".into()),
                                value_text(part.get("input")),
                                tool,
                                cwd.clone(),
                                None,
                                timestamp,
                                event_metadata("tool_use"),
                            ),
                            part.get("input").unwrap_or(&Value::Null),
                            context.raw_record_id(0),
                        ));
                    }
                    Some("tool_result") => {
                        recognized = true;
                        let native_id = scalar_string(part.get("toolUseID"));
                        let tool = native_id
                            .as_ref()
                            .and_then(|id| tool_names.get(id))
                            .cloned();
                        let run = part.get("run").unwrap_or(&Value::Null);
                        let status = run.get("status").and_then(Value::as_str);
                        let result = run
                            .get("result")
                            .or_else(|| run.get("error"))
                            .or_else(|| run.get("reason"))
                            .unwrap_or(&Value::Null);
                        out.push(ImportedEvent::new(
                            CanonicalEvent::new(
                                AgentKind::Amp,
                                &session,
                                message_id.clone(),
                                native_id,
                                if matches!(
                                    status,
                                    Some("error" | "cancelled" | "rejected-by-user")
                                ) {
                                    EventKind::Error
                                } else {
                                    EventKind::ToolResult
                                },
                                Some("tool".into()),
                                value_text(Some(result)),
                                tool,
                                cwd.clone(),
                                None,
                                timestamp,
                                event_metadata("tool_result"),
                            ),
                            result,
                            context.raw_record_id(0),
                        ));
                    }
                    _ => {}
                }
            }
            if !recognized {
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::Amp,
                        &session,
                        message_id.clone(),
                        message_id,
                        EventKind::Other,
                        role,
                        message.to_string(),
                        None,
                        cwd.clone(),
                        None,
                        timestamp,
                        event_metadata("message"),
                    ),
                    message,
                    context.raw_record_id(0),
                ));
            }
        }
        Ok(out)
    }

    fn hook_event(&self, event_name: &str, payload: &Value) -> Result<Vec<CanonicalEvent>> {
        let session =
            scalar_string(payload.pointer("/thread/id")).unwrap_or_else(|| "amp-unknown".into());
        let cwd = scalar_string(payload.get("cwd"));
        let mut turn = scalar_string(payload.get("id"));
        let mut native_id = scalar_string(payload.get("toolUseID")).or_else(|| turn.clone());
        if event_name == "agent.end"
            && let Some(message) =
                payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .and_then(|messages| {
                        messages.iter().rev().find(|message| {
                            message.get("role").and_then(Value::as_str) == Some("assistant")
                        })
                    })
            && let Some(message_id) =
                scalar_string(message.get("id")).or_else(|| scalar_string(message.get("messageId")))
        {
            turn = Some(message_id.clone());
            native_id = Some(message_id);
        }
        let (kind, role, content, tool) = match event_name {
            "session.start" => (EventKind::SessionStart, None, String::new(), None),
            "agent.start" => (
                EventKind::UserMessage,
                Some("user".into()),
                scalar_string(payload.get("message")).unwrap_or_default(),
                None,
            ),
            "agent.end" => {
                let content = payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .and_then(|messages| {
                        messages.iter().rev().find_map(|message| {
                            (message.get("role").and_then(Value::as_str) == Some("assistant"))
                                .then(|| {
                                    text_content(message.get("content").unwrap_or(&Value::Null))
                                })
                                .filter(|text| !text.is_empty())
                        })
                    })
                    .unwrap_or_default();
                (
                    if payload.get("status").and_then(Value::as_str) == Some("error") {
                        EventKind::Error
                    } else {
                        EventKind::AssistantMessage
                    },
                    Some("assistant".into()),
                    content,
                    None,
                )
            }
            "tool.call" => (
                EventKind::ToolCall,
                Some("assistant".into()),
                value_text(payload.get("input")),
                scalar_string(payload.get("tool")),
            ),
            "tool.result" => (
                if payload.get("status").and_then(Value::as_str) == Some("error") {
                    EventKind::Error
                } else {
                    EventKind::ToolResult
                },
                Some("tool".into()),
                value_text(payload.get("output").or_else(|| payload.get("error"))),
                scalar_string(payload.get("tool")),
            ),
            _ => (EventKind::Other, None, payload.to_string(), None),
        };
        Ok(vec![CanonicalEvent::new(
            AgentKind::Amp,
            session,
            turn,
            native_id,
            kind,
            role,
            content,
            tool,
            cwd.clone(),
            None,
            timestamp_ms(payload),
            event_metadata(event_name),
        )])
    }

    fn hook_attachment_sources(&self, event_name: &str, payload: &Value) -> Vec<Value> {
        match event_name {
            "agent.start" | "agent.end" => non_null_sources([payload.get("messages")]),
            "tool.call" => non_null_sources([payload.get("input")]),
            "tool.result" => {
                non_null_sources([payload.get("output").or_else(|| payload.get("error"))])
            }
            _ => Vec::new(),
        }
    }

    fn supports_recall(&self, event_name: &str) -> bool {
        event_name == "agent.start"
    }

    fn recall_response(&self, event_name: &str, hits: &[SearchHit]) -> Option<Value> {
        if event_name != "agent.start" {
            return None;
        }
        let text = recall_text(hits);
        (!text.is_empty()).then(|| json!({"additionalContext":text}))
    }

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
