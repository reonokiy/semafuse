//! Agent-specific capabilities for native history, hooks, and installation.
//!
//! Each concrete adapter can implement the capabilities it supports. History
//! import, hook projection, and hook artifact generation are separate contracts
//! so their source-specific details do not collapse into one class-like API.

mod amp;
mod claude;
mod codex;
pub mod executable;
mod history;
mod opencode;

pub use amp::AmpAdapter;
pub use claude::ClaudeAdapter;
pub use codex::CodexAdapter;
pub use history::{ImportContext, NativeHistoryProjection, ProjectedEvent, project_native_history};
pub use opencode::OpenCodeAdapter;

use crate::{
    blob::BlobStore,
    importer::ImportReport,
    model::{AgentKind, CanonicalEvent, EventSource, MessageRole, normalize_repository},
    strategy::RecallItem,
};
use anyhow::Result;
use async_trait::async_trait;
use chrono::DateTime;
use serde_json::Value;
use std::path::Path;

/// Imports one agent's native persisted history into canonical projections.
#[async_trait]
pub trait HistoryImporter: Send + Sync {
    /// Imports this adapter's conventional history source.
    async fn import_default(&self, blobs: Option<&dyn BlobStore>) -> Result<ImportReport>;
    /// Imports one explicit native source path using this adapter's source semantics.
    async fn import_path(&self, path: &Path, blobs: Option<&dyn BlobStore>)
    -> Result<ImportReport>;
    /// Projects native history values using source-level fallback context.
    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ProjectedEvent>>;
}

/// Projects one agent's real-time hook payloads into canonical events.
///
/// Missing real-time session IDs use `<agent>-unknown`. A missing timestamp is
/// replaced with event construction time. Native IDs are preferred over turn IDs
/// for deduplication, but history and hook records can remain distinct when
/// their sources expose no common ID.
pub trait HookProjector: Send + Sync {
    /// Returns the agent protocol implemented by this hook projector.
    fn kind(&self) -> AgentKind;
    /// Maps one named real-time hook payload into zero or more events.
    fn hook_event(&self, event_name: &str, payload: &Value) -> Result<Vec<CanonicalEvent>>;
    /// Returns the trusted native subtrees whose attachments belong to the hook event.
    fn hook_attachment_sources(&self, _event_name: &str, _payload: &Value) -> Vec<Value> {
        Vec::new()
    }
    /// Returns true when this hook event can consume recall output.
    fn supports_recall(&self, _event_name: &str) -> bool {
        false
    }
    /// Formats strategy recall items for the named native hook, if supported.
    fn recall_response(&self, event_name: &str, items: &[RecallItem]) -> Option<Value>;
}

/// Generates installation artifacts for one agent's hooks and integrations.
pub trait HookInstaller: Send + Sync {
    /// Returns installable hook artifact names and contents for this agent.
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)>;
}

/// Constructs the history importer for an agent kind.
pub fn history_importer(kind: AgentKind) -> Box<dyn HistoryImporter> {
    match kind {
        AgentKind::Amp => Box::new(AmpAdapter),
        AgentKind::Codex => Box::new(CodexAdapter),
        AgentKind::ClaudeCode => Box::new(ClaudeAdapter),
        AgentKind::OpenCode => Box::new(OpenCodeAdapter),
    }
}

/// Constructs the hook projector for an agent kind.
pub fn hook_projector(kind: AgentKind) -> Box<dyn HookProjector> {
    match kind {
        AgentKind::Amp => Box::new(AmpAdapter),
        AgentKind::Codex => Box::new(CodexAdapter),
        AgentKind::ClaudeCode => Box::new(ClaudeAdapter),
        AgentKind::OpenCode => Box::new(OpenCodeAdapter),
    }
}

/// Constructs the hook installer for an agent kind.
pub fn hook_installer(kind: AgentKind) -> Box<dyn HookInstaller> {
    match kind {
        AgentKind::Amp => Box::new(AmpAdapter),
        AgentKind::Codex => Box::new(CodexAdapter),
        AgentKind::ClaudeCode => Box::new(ClaudeAdapter),
        AgentKind::OpenCode => Box::new(OpenCodeAdapter),
    }
}

pub(crate) fn scalar_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

pub(crate) fn text_content(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    Some(text.to_owned())
                } else {
                    item.get("content").map(text_content)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => map
            .get("content")
            .map(text_content)
            .or_else(|| map.get("text").map(text_content))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(crate) fn value_text(value: Option<&Value>) -> String {
    value
        .map(|value| {
            let text = text_content(value);
            if text.is_empty() && !value.is_null() {
                value.to_string()
            } else {
                text
            }
        })
        .unwrap_or_default()
}

pub(crate) fn non_null_sources<'a>(
    values: impl IntoIterator<Item = Option<&'a Value>>,
) -> Vec<Value> {
    values
        .into_iter()
        .flatten()
        .filter(|value| !value.is_null())
        .cloned()
        .collect()
}

pub(crate) fn timestamp_ms(value: &Value) -> Option<i64> {
    timestamp_at(
        value,
        &[
            "/timestamp_ms",
            "/timestamp",
            "/created_at",
            "/createdAt",
            "/time/created",
            "/time/start",
            "/time/end",
        ],
    )
}

/// Deletion-relevant fields that can be recovered without fully decoding a
/// native history record. This deliberately uses agent-specific paths so a
/// message or tool `id` can never be mistaken for a session identifier.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NativeRawContext {
    pub(crate) session_id: Option<String>,
    pub(crate) repository: Option<String>,
    pub(crate) occurred_at_ms: Option<i64>,
}

pub(crate) fn native_raw_context(agent: AgentKind, value: &Value) -> NativeRawContext {
    let common_session = [
        "/session_id",
        "/sessionId",
        "/conversation_id",
        "/thread_id",
        "/session/id",
    ];
    let agent_sessions: &[&str] = match agent {
        AgentKind::Amp => &["/thread/id", "/id"],
        AgentKind::Codex => &["/result/thread/id", "/thread/id", "/payload/session_id"],
        AgentKind::ClaudeCode => &[],
        AgentKind::OpenCode => &[
            "/info/id",
            "/event/properties/info/sessionID",
            "/event/properties/sessionID",
            "/event/properties/part/sessionID",
        ],
    };
    let session_id = common_session
        .iter()
        .chain(agent_sessions)
        .find_map(|pointer| nonempty_pointer(value, pointer))
        .or_else(|| {
            (agent == AgentKind::Codex
                && value.get("type").and_then(Value::as_str) == Some("session_meta"))
            .then(|| nonempty_pointer(value, "/payload/id"))
            .flatten()
        });

    let common_cwds = ["/cwd", "/worktree", "/directory"];
    let agent_cwds: &[&str] = match agent {
        AgentKind::Amp => &[
            "/env/initial/trees/0/uri",
            "/thread/env/initial/trees/0/uri",
        ],
        AgentKind::Codex => &["/result/thread/cwd", "/thread/cwd", "/payload/cwd"],
        AgentKind::ClaudeCode => &[],
        AgentKind::OpenCode => &["/info/worktree", "/info/directory"],
    };
    let cwd = common_cwds
        .iter()
        .chain(agent_cwds)
        .find_map(|pointer| nonempty_pointer(value, pointer))
        .map(|value| value.strip_prefix("file://").unwrap_or(&value).to_owned());
    // These values come from an agent's explicit source/worktree metadata, so
    // retain unavailable paths instead of requiring the repository to exist on
    // the machine performing an import or governance operation.
    let repository = cwd
        .as_deref()
        .and_then(|cwd| normalize_repository(None, Some(cwd)));

    let occurred_at_ms = match agent {
        AgentKind::Amp => timestamp_at(value, &["/created"]),
        AgentKind::Codex => timestamp_at(
            value,
            &[
                "/result/thread/createdAt",
                "/thread/createdAt",
                "/payload/timestamp",
            ],
        ),
        AgentKind::ClaudeCode => None,
        AgentKind::OpenCode => timestamp_at(value, &["/info/time/created", "/info/time/updated"]),
    }
    .or_else(|| timestamp_ms(value));

    NativeRawContext {
        session_id,
        repository,
        occurred_at_ms,
    }
}

fn nonempty_pointer(value: &Value, pointer: &str) -> Option<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
}

pub(crate) fn timestamp_at(value: &Value, pointers: &[&str]) -> Option<i64> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(parse_timestamp))
}

fn parse_timestamp(value: &Value) -> Option<i64> {
    let numeric = value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()));
    if let Some(value) = numeric {
        let magnitude = value.unsigned_abs();
        return Some(if magnitude < 10_000_000_000 {
            value.saturating_mul(1_000)
        } else if magnitude > 10_000_000_000_000 {
            value / 1_000
        } else {
            value
        });
    }
    value
        .as_str()
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.timestamp_millis())
}

pub(crate) fn event_source(
    agent: AgentKind,
    session_id: impl Into<String>,
    turn_id: Option<String>,
    native_id: Option<String>,
    native_event: &str,
    cwd: Option<String>,
    repository: Option<String>,
) -> EventSource {
    EventSource::new(
        agent,
        session_id,
        turn_id,
        native_id,
        native_event,
        cwd,
        repository,
    )
}

pub(crate) fn message_role(role: Option<&str>, default: MessageRole) -> MessageRole {
    match role {
        Some("user") => MessageRole::User,
        Some("assistant") => MessageRole::Assistant,
        Some("system") => MessageRole::System,
        Some("developer") => MessageRole::Developer,
        Some("tool") => MessageRole::Tool,
        _ => default,
    }
}

pub(crate) fn recall_text(items: &[RecallItem]) -> String {
    if items.is_empty() {
        return String::new();
    }
    let lines = items
        .iter()
        .map(|item| match item {
            RecallItem::Event { event, .. } => format!(
                "- [sem:{} agent={} session={}] {}",
                event.id,
                event.source.agent,
                event.source.session_id,
                event.search_text().replace('\n', " ")
            ),
            RecallItem::Memory {
                namespace,
                id,
                content,
                source_event_ids,
                ..
            } => {
                let sources = source_event_ids.join(",");
                format!(
                    "- [sem-memory:{namespace}/{id} sources={sources}] {}",
                    content.replace('\n', "\n  ")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Relevant history from Semafuse (verify against the cited session before relying on it):\n{lines}"
    )
}

#[cfg(test)]
mod tests {
    use super::native_raw_context;
    use crate::model::AgentKind;
    use serde_json::json;

    #[test]
    fn native_raw_context_uses_only_agent_specific_root_identifiers() {
        let amp = native_raw_context(
            AgentKind::Amp,
            &json!({
                "id":"T-amp",
                "created":1_784_350_800_000_i64,
                "env":{"initial":{"trees":[{"uri":"file:///workspace/amp"}]}},
                "messages":[{"id":"not-the-session"}]
            }),
        );
        assert_eq!(amp.session_id.as_deref(), Some("T-amp"));
        assert_eq!(amp.repository.as_deref(), Some("/workspace/amp"));
        assert_eq!(amp.occurred_at_ms, Some(1_784_350_800_000));
        let amp_hook =
            native_raw_context(AgentKind::Amp, &json!({"thread":{"id":"T-hook"},"id":7}));
        assert_eq!(amp_hook.session_id.as_deref(), Some("T-hook"));

        let opencode = native_raw_context(
            AgentKind::OpenCode,
            &json!({"id":"wrong","info":{"id":"open-session","directory":"/workspace/open"}}),
        );
        assert_eq!(opencode.session_id.as_deref(), Some("open-session"));
        assert_eq!(opencode.repository.as_deref(), Some("/workspace/open"));
        let opencode_hook = native_raw_context(
            AgentKind::OpenCode,
            &json!({"event":{"properties":{"part":{"sessionID":"open-hook"}}}}),
        );
        assert_eq!(opencode_hook.session_id.as_deref(), Some("open-hook"));

        let codex = native_raw_context(
            AgentKind::Codex,
            &json!({"id":19,"result":{"thread":{"id":"codex-session","cwd":"/workspace/codex"}}}),
        );
        assert_eq!(codex.session_id.as_deref(), Some("codex-session"));
        assert_eq!(codex.repository.as_deref(), Some("/workspace/codex"));

        let codex_session_meta = native_raw_context(
            AgentKind::Codex,
            &json!({"type":"session_meta","payload":{"id":"codex-rollout"}}),
        );
        assert_eq!(
            codex_session_meta.session_id.as_deref(),
            Some("codex-rollout")
        );
        let codex_message = native_raw_context(
            AgentKind::Codex,
            &json!({"type":"response_item","payload":{"id":"message-id"}}),
        );
        assert_eq!(codex_message.session_id, None);
    }
}
