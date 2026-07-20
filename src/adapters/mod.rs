//! Conversion of agent-native history and hook payloads into canonical events.
//!
//! [`AgentAdapter`] is the common contract. Each concrete adapter documents its
//! native-to-canonical mapping on the adapter type itself.

mod amp;
mod claude;
mod codex;
mod opencode;

pub use amp::AmpAdapter;
pub use claude::ClaudeAdapter;
pub use codex::CodexAdapter;
pub use opencode::OpenCodeAdapter;

use crate::model::{AgentKind, CanonicalEvent, SearchHit};
use anyhow::Result;
use chrono::DateTime;
use serde_json::{Map, Value};
use std::path::Path;

/// Source context supplied while importing an agent history file or session.
#[derive(Debug, Clone)]
pub struct ImportContext {
    /// Human-readable source identifier, normally the input path or remote session.
    pub source: String,
    /// Session ID to use when the native records provide none.
    pub fallback_session_id: String,
    /// Working directory supplied by the source as session-level context.
    pub cwd: Option<String>,
    /// Raw record IDs in the same order as values supplied to the adapter.
    pub raw_record_ids: Vec<String>,
}

impl ImportContext {
    pub(crate) fn raw_record_id(&self, index: usize) -> &str {
        self.raw_record_ids
            .get(index)
            .or_else(|| self.raw_record_ids.first())
            .map(String::as_str)
            .unwrap_or("raw_missing")
    }
}

/// A canonical imported event paired with the exact native value from which
/// attachments must be extracted.
///
/// Keeping attachment ownership at this boundary prevents attachments from all
/// being assigned to the final event in a session. If attachment processing
/// fails, the importer rejects that source instead of writing an event with
/// silently missing attachments.
#[derive(Debug, Clone)]
pub struct ImportedEvent {
    /// Normalized event to ingest after attachment extraction.
    pub event: CanonicalEvent,
    /// Native subtree whose attachments belong specifically to `event`.
    pub attachment_source: Value,
}

impl ImportedEvent {
    pub(crate) fn new(
        mut event: CanonicalEvent,
        attachment_source: &Value,
        raw_record_id: &str,
    ) -> Self {
        event.attach_raw_record(raw_record_id);
        Self {
            event,
            attachment_source: attachment_source.clone(),
        }
    }
}

/// Converts one agent's native protocols into [`CanonicalEvent`] values.
///
/// History imports and real-time hooks share the canonical model but have
/// separate mapping rules because agents persist different structures from the
/// payloads emitted by lifecycle hooks. Unrecognized input is generally kept as
/// [`crate::model::EventKind::Other`] with serialized native JSON in `content`
/// and the original sanitized payload in metadata.
///
/// Missing real-time session IDs use `<agent>-unknown`; import callers supply a
/// fallback derived from the source. A missing timestamp is replaced with event
/// construction time. Native IDs are preferred over turn IDs for deduplication,
/// but history and hook records can remain distinct when their sources expose no
/// common ID.
pub trait AgentAdapter: Send + Sync {
    /// Returns the agent protocol implemented by this adapter.
    fn kind(&self) -> AgentKind;
    /// Maps native history values using source-level fallback context.
    fn import_values(
        &self,
        values: &[Value],
        context: &ImportContext,
    ) -> Result<Vec<ImportedEvent>>;
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
    /// Returns installable hook artifact names and contents for this agent.
    fn hook_artifacts(&self, binary: &Path, database: &Path) -> Vec<(String, String)>;
}

/// Constructs the adapter for an agent kind.
pub fn adapter(kind: AgentKind) -> Box<dyn AgentAdapter> {
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

pub(crate) fn event_metadata(native_event: &str) -> Map<String, Value> {
    let mut metadata = Map::new();
    metadata.insert("native_event".into(), Value::String(native_event.into()));
    metadata
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
