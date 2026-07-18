use super::{
    AgentAdapter, ImportContext, ImportedEvent, event_metadata, scalar_string, text_content,
    timestamp_at, timestamp_ms, value_text,
};
use crate::model::{AgentKind, CanonicalEvent, EventKind, SearchHit};
use anyhow::Result;
use serde_json::Value;
use std::path::Path;

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

impl AgentAdapter for OpenCodeAdapter {
    fn kind(&self) -> AgentKind {
        AgentKind::OpenCode
    }

    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ImportedEvent>> {
        let root = values.first().unwrap_or(&Value::Null);
        let info = root.get("info").unwrap_or(root);
        let root_session = scalar_string(info.get("id"))
            .or_else(|| scalar_string(info.get("sessionID")))
            .unwrap_or_else(|| context.fallback_session_id.clone());
        let cwd = scalar_string(info.get("directory")).or_else(|| context.cwd.clone());
        let messages = root
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| values.to_vec());
        let mut out = Vec::new();
        for message in messages {
            let mi = message.get("info").unwrap_or(&message);
            let session =
                scalar_string(mi.get("sessionID")).unwrap_or_else(|| root_session.clone());
            let role = scalar_string(mi.get("role"));
            let message_id = scalar_string(mi.get("id"));
            let timestamp = timestamp_ms(mi).or_else(|| timestamp_ms(&message));
            let parts = message.get("parts").and_then(Value::as_array);
            let content = parts
                .into_iter()
                .flatten()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .map(text_content)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            let content = if content.is_empty() {
                message.get("content").map(text_content).unwrap_or_default()
            } else {
                content
            };
            let mut recognized = false;
            if !content.is_empty() {
                recognized = true;
                let attachment_source = Value::Array(
                    parts
                        .into_iter()
                        .flatten()
                        .filter(|part| {
                            !matches!(
                                part.get("type").and_then(Value::as_str),
                                Some("tool" | "compaction")
                            )
                        })
                        .cloned()
                        .collect(),
                );
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::OpenCode,
                        &session,
                        message_id.clone(),
                        message_id.clone(),
                        if role.as_deref() == Some("user") {
                            EventKind::UserMessage
                        } else {
                            EventKind::AssistantMessage
                        },
                        role.clone(),
                        content,
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
            for part in parts.into_iter().flatten() {
                match part.get("type").and_then(Value::as_str) {
                    Some("tool") => {
                        recognized = true;
                        let state = part.get("state").unwrap_or(&Value::Null);
                        let call_id = scalar_string(part.get("callID"));
                        let tool = scalar_string(part.get("tool"));
                        out.push(ImportedEvent::new(
                            CanonicalEvent::new(
                                AgentKind::OpenCode,
                                &session,
                                message_id.clone(),
                                call_id.clone(),
                                EventKind::ToolCall,
                                Some("assistant".into()),
                                value_text(state.get("input")),
                                tool.clone(),
                                cwd.clone(),
                                None,
                                timestamp_at(state, &["/time/start"]).or(timestamp),
                                event_metadata("tool"),
                            ),
                            state.get("input").unwrap_or(&Value::Null),
                            context.raw_record_id(0),
                        ));
                        let status = state.get("status").and_then(Value::as_str);
                        if matches!(status, Some("completed" | "error")) {
                            out.push(ImportedEvent::new(
                                CanonicalEvent::new(
                                    AgentKind::OpenCode,
                                    &session,
                                    message_id.clone(),
                                    call_id,
                                    if status == Some("error") {
                                        EventKind::Error
                                    } else {
                                        EventKind::ToolResult
                                    },
                                    Some("tool".into()),
                                    value_text(state.get("output").or_else(|| state.get("error"))),
                                    tool,
                                    cwd.clone(),
                                    None,
                                    timestamp_at(state, &["/time/end", "/time/start"])
                                        .or(timestamp),
                                    event_metadata("tool"),
                                ),
                                state
                                    .get("output")
                                    .or_else(|| state.get("error"))
                                    .unwrap_or(&Value::Null),
                                context.raw_record_id(0),
                            ));
                        }
                    }
                    Some("compaction") => {
                        recognized = true;
                        out.push(ImportedEvent::new(
                            CanonicalEvent::new(
                                AgentKind::OpenCode,
                                &session,
                                message_id.clone(),
                                scalar_string(part.get("id")),
                                EventKind::Compaction,
                                None,
                                part.to_string(),
                                None,
                                cwd.clone(),
                                None,
                                timestamp_ms(part).or(timestamp),
                                event_metadata("compaction"),
                            ),
                            part,
                            context.raw_record_id(0),
                        ));
                    }
                    _ => {}
                }
            }
            if !recognized {
                out.push(ImportedEvent::new(
                    CanonicalEvent::new(
                        AgentKind::OpenCode,
                        session,
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
                    &message,
                    context.raw_record_id(0),
                ));
            }
        }
        Ok(out)
    }

    fn hook_event(&self, event_name: &str, p: &Value) -> Result<Vec<CanonicalEvent>> {
        let event = p.get("event").unwrap_or(p);
        let props = event.get("properties").unwrap_or(event);
        let info = props
            .get("info")
            .or_else(|| props.get("message"))
            .unwrap_or(props);
        let session = scalar_string(info.get("sessionID"))
            .or_else(|| scalar_string(props.get("sessionID")))
            .or_else(|| {
                props
                    .get("part")
                    .and_then(|part| scalar_string(part.get("sessionID")))
            })
            .or_else(|| scalar_string(info.get("id")))
            .unwrap_or_else(|| "opencode-unknown".into());
        let cwd = scalar_string(p.get("directory")).or_else(|| scalar_string(p.get("worktree")));
        let event_type = if event_name == "auto" {
            event.get("type").and_then(Value::as_str).unwrap_or("other")
        } else {
            event_name
        };
        let part = props.get("part");
        let role =
            scalar_string(info.get("role")).or_else(|| scalar_string(props.get("messageRole")));
        let part_type = part
            .and_then(|part| part.get("type"))
            .and_then(Value::as_str);
        let tool_state = part.and_then(|part| part.get("state"));
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
                    && tool_state
                        .and_then(|state| state.get("status"))
                        .and_then(Value::as_str)
                        == Some("completed") =>
            {
                EventKind::ToolResult
            }
            "message.part.updated"
                if part_type == Some("tool")
                    && tool_state
                        .and_then(|state| state.get("status"))
                        .and_then(Value::as_str)
                        == Some("error") =>
            {
                EventKind::Error
            }
            "message.part.updated" if part_type == Some("tool") => EventKind::ToolCall,
            "message.part.updated" if part_type == Some("compaction") => EventKind::Compaction,
            "tool.execute.after" => EventKind::ToolResult,
            "tool.execute.before" => EventKind::ToolCall,
            _ => EventKind::Other,
        };
        let canonical_role = match kind {
            EventKind::UserMessage => Some("user".into()),
            EventKind::AssistantMessage | EventKind::ToolCall => Some("assistant".into()),
            EventKind::ToolResult | EventKind::Error
                if part_type == Some("tool") || event_type.starts_with("tool.execute.") =>
            {
                Some("tool".into())
            }
            _ => role,
        };
        let content = match kind {
            EventKind::UserMessage | EventKind::AssistantMessage => value_text(part),
            EventKind::ToolCall => value_text(
                tool_state
                    .and_then(|state| state.get("input"))
                    .or_else(|| props.get("input")),
            ),
            EventKind::ToolResult => value_text(
                tool_state
                    .and_then(|state| state.get("output"))
                    .or_else(|| props.get("output")),
            ),
            EventKind::Error => value_text(
                tool_state
                    .and_then(|state| state.get("error"))
                    .or_else(|| props.get("error")),
            ),
            EventKind::SessionStart => String::new(),
            _ => props.to_string(),
        };
        let turn = part
            .and_then(|part| scalar_string(part.get("messageID")))
            .or_else(|| scalar_string(info.get("parentID")));
        let native_id = if matches!(kind, EventKind::UserMessage | EventKind::AssistantMessage) {
            turn.clone()
        } else {
            part.and_then(|part| scalar_string(part.get("callID")))
                .or_else(|| part.and_then(|part| scalar_string(part.get("id"))))
                .or_else(|| scalar_string(info.get("id")))
        };
        let tool = part
            .and_then(|part| scalar_string(part.get("tool")))
            .or_else(|| scalar_string(props.get("tool")));
        let worktree = scalar_string(p.get("worktree"));
        let timestamp = part
            .and_then(|part| match kind {
                EventKind::ToolResult | EventKind::Error if part_type == Some("tool") => {
                    timestamp_at(part, &["/state/time/end", "/state/time/start"])
                }
                EventKind::ToolCall => timestamp_at(part, &["/state/time/start"]),
                _ => timestamp_ms(part),
            })
            .or_else(|| timestamp_ms(info))
            .or_else(|| timestamp_ms(props));
        Ok(vec![CanonicalEvent::new(
            AgentKind::OpenCode,
            session,
            turn,
            native_id,
            kind,
            canonical_role,
            content,
            tool,
            cwd.clone(),
            worktree,
            timestamp,
            event_metadata(event_type),
        )])
    }

    fn recall_response(&self, _event_name: &str, _hits: &[SearchHit]) -> Option<Value> {
        None
    }

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
