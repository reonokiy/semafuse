use super::{EventSource, now_ms};
use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{fmt, str::FromStr};

/// Identifies the coding agent that produced an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum AgentKind {
    /// An Amp event.
    Amp,
    /// A Codex event.
    Codex,
    /// A Claude Code event.
    ClaudeCode,
    /// An OpenCode event.
    #[serde(rename = "opencode")]
    OpenCode,
}

impl AgentKind {
    /// Returns the stable kebab-case protocol name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Amp => "amp",
            Self::Codex => "codex",
            Self::ClaudeCode => "claude-code",
            Self::OpenCode => "opencode",
        }
    }
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AgentKind {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "amp" => Ok(Self::Amp),
            "codex" => Ok(Self::Codex),
            "claude" | "claude-code" | "claudecode" => Ok(Self::ClaudeCode),
            "opencode" | "open-code" => Ok(Self::OpenCode),
            _ => bail!("unsupported agent: {value}"),
        }
    }
}

/// The agent-neutral category of a [`CanonicalEvent`].
///
/// Values serialize as the documented `snake_case` names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A session started (`session_start`).
    SessionStart,
    /// A session ended or was deleted (`session_end`).
    SessionEnd,
    /// User input (`user_message`).
    UserMessage,
    /// Agent or assistant output (`assistant_message`).
    AssistantMessage,
    /// A tool call initiated by an agent (`tool_call`).
    ToolCall,
    /// A tool execution result or related artifact (`tool_result`).
    ToolResult,
    /// A context compaction event (`compaction`).
    Compaction,
    /// An agent, session, or tool error (`error`).
    Error,
    /// An event not recognized by its adapter (`other`).
    Other,
}

impl EventKind {
    /// Returns the stable snake-case serialized name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::SessionEnd => "session_end",
            Self::UserMessage => "user_message",
            Self::AssistantMessage => "assistant_message",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Compaction => "compaction",
            Self::Error => "error",
            Self::Other => "other",
        }
    }
}

/// A content-addressed reference to a binary attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttachmentRef {
    /// The backend-specific object key.
    pub object_key: String,
    /// The SHA-256 digest of the attachment contents.
    pub digest: String,
    /// The attachment's media type.
    pub media_type: String,
    /// The attachment size in bytes.
    pub size: u64,
    /// The source file name, when available.
    pub name: Option<String>,
}

/// Version of the stable Semafuse canonical-event contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum EventSpecVersion {
    /// The first formal event schema.
    #[serde(rename = "semafuse.event/v1")]
    V1,
}

/// Closed set of message authors represented by the canonical schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// Human input.
    User,
    /// Model output.
    Assistant,
    /// System instruction.
    System,
    /// Developer instruction.
    Developer,
    /// Tool-authored message.
    Tool,
    /// A native role that cannot be represented more precisely.
    Unknown,
}

/// Structured content retained without flattening every value into text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentPart {
    /// Plain searchable text.
    Text {
        /// Text body.
        text: String,
    },
    /// Structured JSON such as tool arguments or results.
    Json {
        /// JSON value.
        value: Value,
    },
    /// Content-addressed binary or file attachment.
    Attachment {
        /// Durable attachment reference.
        attachment: AttachmentRef,
    },
    /// A reasoning summary intentionally exposed by the native agent.
    ReasoningSummary {
        /// Summary text; encrypted or hidden reasoning is never copied here.
        text: String,
    },
}

/// Terminal or intermediate outcome of a tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    /// Tool has not started.
    Pending,
    /// Tool is executing.
    Running,
    /// Tool completed successfully.
    Succeeded,
    /// Tool failed.
    Failed,
    /// Tool was cancelled.
    Cancelled,
    /// Tool execution was rejected.
    Rejected,
    /// Native status was unavailable.
    Unknown,
}

/// Structured error information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ErrorInfo {
    /// Human-readable error message.
    pub message: String,
    /// Optional stable native error code.
    pub code: Option<String>,
    /// Optional structured details that are safe to retain canonically.
    pub details: Option<Value>,
}

/// Fully typed payload of a canonical conversation event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventData {
    /// Session lifecycle start.
    SessionStarted,
    /// Session lifecycle end.
    SessionEnded {
        /// Native termination reason.
        reason: Option<String>,
    },
    /// User, assistant, system, developer, or tool message.
    Message {
        /// Formal message author.
        role: MessageRole,
        /// Ordered message content.
        content: Vec<ContentPart>,
    },
    /// Tool invocation.
    ToolCall {
        /// Native call identifier when exposed.
        call_id: Option<String>,
        /// Tool name.
        tool: String,
        /// Structured native arguments.
        arguments: Value,
        /// Attachments supplied with the invocation.
        #[serde(default)]
        attachments: Vec<AttachmentRef>,
    },
    /// Tool execution outcome.
    ToolResult {
        /// Native call identifier when exposed.
        call_id: Option<String>,
        /// Tool name when exposed by the source or correlated call.
        tool: Option<String>,
        /// Formal execution status.
        status: ToolStatus,
        /// Ordered result content.
        output: Vec<ContentPart>,
        /// Structured failure when status is not successful.
        error: Option<ErrorInfo>,
    },
    /// Context compaction or summary boundary.
    Compaction {
        /// Searchable compacted summary when available.
        summary: Option<String>,
        /// Structured compaction details when available.
        details: Option<Value>,
    },
    /// Non-tool failure.
    Error {
        /// Structured failure.
        error: ErrorInfo,
    },
    /// Native record deliberately retained only through raw provenance.
    Unknown {
        /// Native discriminator; raw payload lives in [`crate::model::RawRecord`].
        native_type: String,
    },
}

/// Versioned, agent-neutral event produced by normalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventEnvelope {
    /// Canonical contract version.
    pub spec_version: EventSpecVersion,
    /// `evt_` followed by the first 24 fingerprint characters.
    pub id: String,
    /// SHA-256 of normalized identity fields.
    pub fingerprint: String,
    /// Native source identity and workspace context.
    pub source: EventSource,
    /// Native occurrence time; absent when the source does not provide one.
    pub occurred_at_ms: Option<i64>,
    /// Time this canonical projection was built.
    pub ingested_at_ms: i64,
    /// Fully typed event payload.
    pub data: EventData,
    /// Raw records from which this event was projected.
    #[serde(default)]
    pub raw_record_ids: Vec<String>,
}

/// Concise name for the formal event envelope used throughout Semafuse.
pub type CanonicalEvent = EventEnvelope;

impl EventEnvelope {
    /// Constructs a session-start event from normalized source provenance.
    pub fn session_started(source: EventSource, occurred_at_ms: Option<i64>) -> Self {
        Self::from_data(source, occurred_at_ms, EventData::SessionStarted)
    }

    /// Constructs a session-end event from normalized source provenance.
    pub fn session_ended(
        source: EventSource,
        reason: Option<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::from_data(source, occurred_at_ms, EventData::SessionEnded { reason })
    }

    /// Constructs a message event from already-typed content parts.
    pub fn message(
        source: EventSource,
        role: MessageRole,
        content: Vec<ContentPart>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::from_data(source, occurred_at_ms, EventData::Message { role, content })
    }

    /// Constructs a message event from plain text.
    pub fn message_from_text(
        source: EventSource,
        role: MessageRole,
        content: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::message(source, role, text_parts(content.into()), occurred_at_ms)
    }

    /// Constructs a tool-call event from structured arguments.
    pub fn tool_call(
        source: EventSource,
        tool: impl Into<String>,
        arguments: Value,
        attachments: Vec<AttachmentRef>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        let call_id = source.native_id.clone();
        Self::from_data(
            source,
            occurred_at_ms,
            EventData::ToolCall {
                call_id,
                tool: tool.into(),
                arguments,
                attachments,
            },
        )
    }

    /// Constructs a tool-call event from native text or JSON content.
    pub fn tool_call_from_content(
        source: EventSource,
        tool: impl Into<String>,
        content: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::tool_call(
            source,
            tool,
            parse_json_or_string(content.into()),
            Vec::new(),
            occurred_at_ms,
        )
    }

    /// Constructs a tool-result event from already-typed output and status.
    pub fn tool_result(
        source: EventSource,
        tool: Option<String>,
        status: ToolStatus,
        output: Vec<ContentPart>,
        error: Option<ErrorInfo>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        let call_id = source.native_id.clone();
        Self::from_data(
            source,
            occurred_at_ms,
            EventData::ToolResult {
                call_id,
                tool,
                status,
                output,
                error,
            },
        )
    }

    /// Constructs a tool-result event from native text or JSON output.
    pub fn tool_result_from_content(
        source: EventSource,
        tool: Option<String>,
        status: ToolStatus,
        content: impl Into<String>,
        error: Option<ErrorInfo>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::tool_result(
            source,
            tool,
            status,
            value_parts(content.into()),
            error,
            occurred_at_ms,
        )
    }

    /// Constructs a failed tool-result event from an error message.
    pub fn failed_tool_result(
        source: EventSource,
        tool: Option<String>,
        message: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        let message = message.into();
        Self::tool_result(
            source,
            tool,
            ToolStatus::Failed,
            Vec::new(),
            Some(ErrorInfo {
                message: if message.is_empty() {
                    "tool failed".into()
                } else {
                    message
                },
                code: None,
                details: None,
            }),
            occurred_at_ms,
        )
    }

    /// Constructs a compaction event from explicit summary and details fields.
    pub fn compaction(
        source: EventSource,
        summary: Option<String>,
        details: Option<Value>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::from_data(
            source,
            occurred_at_ms,
            EventData::Compaction { summary, details },
        )
    }

    /// Constructs a compaction event from native content text.
    pub fn compaction_from_content(
        source: EventSource,
        content: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        let content = content.into();
        let details = serde_json::from_str::<Value>(&content)
            .ok()
            .map(|value| canonical_json_projection(&value));
        let summary = details
            .as_ref()
            .and_then(|value| {
                value
                    .get("summary")
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
            })
            .map(str::to_owned)
            .or_else(|| {
                details
                    .is_none()
                    .then(|| nonempty_string(content))
                    .flatten()
            });
        Self::compaction(source, summary, details, occurred_at_ms)
    }

    /// Constructs a non-tool error event from structured error information.
    pub fn error(source: EventSource, error: ErrorInfo, occurred_at_ms: Option<i64>) -> Self {
        Self::from_data(source, occurred_at_ms, EventData::Error { error })
    }

    /// Constructs a non-tool error event from an error message.
    pub fn error_message(
        source: EventSource,
        message: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        let message = message.into();
        Self::error(
            source,
            ErrorInfo {
                message: if message.is_empty() {
                    "agent error".into()
                } else {
                    message
                },
                code: None,
                details: None,
            },
            occurred_at_ms,
        )
    }

    /// Constructs an event for an unrecognized native record.
    pub fn unknown(
        source: EventSource,
        native_type: impl Into<String>,
        occurred_at_ms: Option<i64>,
    ) -> Self {
        Self::from_data(
            source,
            occurred_at_ms,
            EventData::Unknown {
                native_type: native_type.into(),
            },
        )
    }

    fn from_data(source: EventSource, occurred_at_ms: Option<i64>, data: EventData) -> Self {
        let mut event = Self {
            spec_version: EventSpecVersion::V1,
            id: String::new(),
            fingerprint: String::new(),
            source,
            occurred_at_ms,
            ingested_at_ms: now_ms(),
            data,
            raw_record_ids: Vec::new(),
        };
        event.refresh_identity();
        event
    }

    /// Validates invariants that JSON Schema alone cannot express.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.source.session_id.trim().is_empty(),
            "empty session_id"
        );
        anyhow::ensure!(
            !self.source.native_event.trim().is_empty(),
            "empty native_event"
        );
        match &self.data {
            EventData::Message { content, .. } => {
                anyhow::ensure!(!content.is_empty(), "message content is empty")
            }
            EventData::ToolCall { tool, .. } => {
                anyhow::ensure!(!tool.trim().is_empty(), "tool name is empty")
            }
            EventData::ToolResult { status, error, .. } => {
                anyhow::ensure!(
                    !matches!(
                        status,
                        ToolStatus::Failed | ToolStatus::Cancelled | ToolStatus::Rejected
                    ) || error.is_some(),
                    "unsuccessful tool result is missing error"
                )
            }
            EventData::Error { error } => {
                anyhow::ensure!(!error.message.trim().is_empty(), "error message is empty")
            }
            EventData::Unknown { native_type } => {
                anyhow::ensure!(
                    !native_type.trim().is_empty(),
                    "unknown native type is empty"
                )
            }
            EventData::SessionStarted
            | EventData::SessionEnded { .. }
            | EventData::Compaction { .. } => {}
        }
        let fingerprint = self.identity_fingerprint();
        anyhow::ensure!(
            self.fingerprint == fingerprint,
            "event fingerprint mismatch"
        );
        anyhow::ensure!(
            self.id == format!("evt_{}", &fingerprint[..24]),
            "event ID mismatch"
        );
        Ok(())
    }

    /// Returns the stable event discriminator.
    pub fn event_type(&self) -> &'static str {
        match &self.data {
            EventData::SessionStarted => "session_started",
            EventData::SessionEnded { .. } => "session_ended",
            EventData::Message { .. } => "message",
            EventData::ToolCall { .. } => "tool_call",
            EventData::ToolResult { .. } => "tool_result",
            EventData::Compaction { .. } => "compaction",
            EventData::Error { .. } => "error",
            EventData::Unknown { .. } => "unknown",
        }
    }

    /// Produces the only text projection indexed by retrieval.
    pub fn search_text(&self) -> String {
        match &self.data {
            EventData::SessionStarted => String::new(),
            EventData::SessionEnded { reason } => reason.clone().unwrap_or_default(),
            EventData::Message { content, .. } => parts_text(content),
            EventData::ToolCall {
                tool, arguments, ..
            } => {
                format!("{tool}\n{}", value_text_projection(arguments))
            }
            EventData::ToolResult {
                tool,
                output,
                error,
                ..
            } => [
                tool.clone().unwrap_or_default(),
                parts_text(output),
                error
                    .as_ref()
                    .map(|e| e.message.clone())
                    .unwrap_or_default(),
            ]
            .into_iter()
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
            EventData::Compaction { summary, .. } => summary.clone().unwrap_or_default(),
            EventData::Error { error } => error.message.clone(),
            EventData::Unknown { .. } => String::new(),
        }
    }

    /// Effective timestamp for ordering when native time is absent.
    pub fn ordering_timestamp_ms(&self) -> i64 {
        self.occurred_at_ms.unwrap_or(self.ingested_at_ms)
    }

    /// Adds raw provenance without changing canonical identity.
    pub fn attach_raw_record(&mut self, raw_record_id: impl Into<String>) {
        let id = raw_record_id.into();
        if !self.raw_record_ids.contains(&id) {
            self.raw_record_ids.push(id);
        }
    }

    /// Adds extracted attachments as structured content and refreshes identity.
    pub fn add_attachments(&mut self, attachments: Vec<AttachmentRef>) {
        let existing = self.search_text();
        let mut digests = match &self.data {
            EventData::Message { content, .. }
            | EventData::ToolResult {
                output: content, ..
            } => content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Attachment { attachment } => Some(attachment.digest.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            EventData::ToolCall { attachments, .. } => attachments
                .iter()
                .map(|attachment| attachment.digest.clone())
                .collect(),
            _ => Vec::new(),
        };
        let parts = attachments
            .into_iter()
            .filter(|attachment| {
                if digests.contains(&attachment.digest) {
                    false
                } else {
                    digests.push(attachment.digest.clone());
                    true
                }
            })
            .map(|attachment| ContentPart::Attachment { attachment })
            .collect::<Vec<_>>();
        match &mut self.data {
            EventData::Message { content, .. } => content.extend(parts),
            EventData::ToolResult { output, .. } => output.extend(parts),
            EventData::ToolCall { attachments, .. } => {
                attachments.extend(parts.into_iter().map(|part| match part {
                    ContentPart::Attachment { attachment } => attachment,
                    _ => unreachable!("constructed only from attachments"),
                }))
            }
            EventData::SessionStarted
            | EventData::SessionEnded { .. }
            | EventData::Compaction { .. }
            | EventData::Error { .. }
            | EventData::Unknown { .. } => {
                if !parts.is_empty() {
                    self.data = EventData::Message {
                        role: MessageRole::Unknown,
                        content: text_parts(existing).into_iter().chain(parts).collect(),
                    };
                }
            }
        }
        self.refresh_identity();
    }

    /// Returns every content-addressed attachment referenced by this event.
    pub fn attachment_refs(&self) -> Vec<&AttachmentRef> {
        match &self.data {
            EventData::Message { content, .. }
            | EventData::ToolResult {
                output: content, ..
            } => content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Attachment { attachment } => Some(attachment),
                    _ => None,
                })
                .collect(),
            EventData::ToolCall { attachments, .. } => attachments.iter().collect(),
            EventData::SessionStarted
            | EventData::SessionEnded { .. }
            | EventData::Compaction { .. }
            | EventData::Error { .. }
            | EventData::Unknown { .. } => Vec::new(),
        }
    }

    /// Recomputes the fingerprint after canonical identity content changes.
    pub fn refresh_identity(&mut self) {
        self.fingerprint = self.identity_fingerprint();
        self.id = format!("evt_{}", &self.fingerprint[..24]);
    }

    fn identity_fingerprint(&self) -> String {
        let logical_id = self
            .source
            .native_id
            .as_ref()
            .or(self.source.turn_id.as_ref());
        let canonical = serde_json::json!({
            "spec_version": self.spec_version,
            "agent": self.source.agent,
            "session_id": self.source.session_id,
            "logical_id": logical_id,
            "data": self.data,
        });
        hex::encode(Sha256::digest(canonical.to_string().as_bytes()))
    }
}

fn nonempty_string(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn parse_json_or_string(value: String) -> Value {
    if value.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        canonical_json_projection(&serde_json::from_str(&value).unwrap_or(Value::String(value)))
    }
}

fn text_parts(value: String) -> Vec<ContentPart> {
    nonempty_string(value)
        .map(|text| vec![ContentPart::Text { text }])
        .unwrap_or_default()
}

fn value_parts(value: String) -> Vec<ContentPart> {
    if value.trim().is_empty() {
        Vec::new()
    } else if let Ok(value) = serde_json::from_str(&value) {
        vec![ContentPart::Json {
            value: canonical_json_projection(&value),
        }]
    } else {
        vec![ContentPart::Text { text: value }]
    }
}

fn canonical_json_projection(value: &Value) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.iter().map(canonical_json_projection).collect())
        }
        Value::Object(values) => {
            let kind = values.get("type").and_then(Value::as_str);
            let media_type = values
                .get("media_type")
                .or_else(|| values.get("mimeType"))
                .or_else(|| values.get("mime"))
                .and_then(Value::as_str);
            let binary = matches!(
                kind,
                Some("base64" | "image" | "input_image" | "audio" | "document" | "pdf")
            ) || media_type.is_some_and(|value| {
                value.starts_with("image/")
                    || value.starts_with("audio/")
                    || value == "application/pdf"
            });
            Value::Object(
                values
                    .iter()
                    .map(|(key, value)| {
                        if binary && matches!(key.as_str(), "data" | "url" | "image_url") {
                            (
                                key.clone(),
                                Value::String("[binary omitted from canonical event]".into()),
                            )
                        } else {
                            (key.clone(), canonical_json_projection(value))
                        }
                    })
                    .collect(),
            )
        }
        Value::String(value) if value.starts_with("data:") && value.contains(";base64,") => {
            Value::String("[binary omitted from canonical event]".into())
        }
        _ => value.clone(),
    }
}

fn parts_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } | ContentPart::ReasoningSummary { text } => {
                Some(text.clone())
            }
            ContentPart::Json { value } => Some(value_text_projection(value)),
            ContentPart::Attachment { attachment } => attachment.name.clone(),
        })
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn value_text_projection(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => value.to_string(),
    }
}
