//! Agent-specific capabilities for native history, hooks, and installation.
//!
//! Each concrete adapter can implement the capabilities it supports. History
//! import, hook projection, and hook artifact generation are separate contracts
//! so their source-specific details do not collapse into one class-like API.

mod amp;
mod claude;
mod codex;
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
    model::{AgentKind, CanonicalEvent, EventSource, MessageRole, SearchHit},
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
    /// Formats retrieval hits for the named native hook, if it supports recall injection.
    fn recall_response(&self, event_name: &str, hits: &[SearchHit]) -> Option<Value>;
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

pub(crate) fn recall_text(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    let lines = hits
        .iter()
        .map(|hit| {
            format!(
                "- [sem:{} agent={} session={}] {}",
                hit.event.id,
                hit.event.source.agent,
                hit.event.source.session_id,
                hit.event.search_text().replace('\n', " ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Relevant history from Semafuse (verify against the cited session before relying on it):\n{lines}"
    )
}
