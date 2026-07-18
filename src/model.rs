//! Agent-neutral event and memory data model.

use anyhow::{Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    path::Path,
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

/// Monotonically increasing position of a committed domain event.
///
/// Sequence zero represents the position before the first domain event. Values
/// persisted in `domain_events.seq` start at one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct DomainEventSequence(i64);

impl DomainEventSequence {
    /// Position before any committed domain event.
    pub const ORIGIN: Self = Self(0);

    /// Returns the durable integer position.
    pub const fn get(self) -> i64 {
        self.0
    }

    pub(crate) const fn from_database(value: i64) -> Self {
        Self(value)
    }
}

impl fmt::Display for DomainEventSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

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

/// Version of the persisted raw-to-canonical archive contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum ArchiveSpecVersion {
    /// The first formal archive schema.
    #[serde(rename = "semafuse.archive/v1")]
    V1,
}

/// Provenance fields shared by every canonical event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventSource {
    /// Coding agent that emitted the native record.
    pub agent: AgentKind,
    /// Native session identifier or an explicit import fallback.
    pub session_id: String,
    /// Native turn or message grouping identifier.
    pub turn_id: Option<String>,
    /// Native event, message, or tool-call identifier.
    pub native_id: Option<String>,
    /// Native event discriminator used by the adapter.
    pub native_event: String,
    /// Working directory reported by the source.
    pub cwd: Option<String>,
    /// Explicit or locally detected repository/worktree.
    pub repository: Option<String>,
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
        /// Native discriminator; raw payload lives in [`RawRecord`].
        native_type: String,
    },
}

/// Immutable parsed native input retained independently from normalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawRecord {
    /// Stable content-and-source-derived raw record ID.
    pub id: String,
    /// SHA-256 digest of the normalized JSON representation.
    pub digest: String,
    /// Agent owning the native format.
    pub agent: AgentKind,
    /// File, protocol, or hook source identifier.
    pub source: String,
    /// Zero-based record position within the source.
    pub record_index: u64,
    /// Best-effort native discriminator.
    pub native_type: String,
    /// Parsed JSON record with every representable native field retained.
    pub payload: Value,
    /// Local observation time in Unix milliseconds.
    pub ingested_at_ms: i64,
}

impl RawRecord {
    /// Constructs a stable raw record without interpreting its payload.
    pub fn new(
        agent: AgentKind,
        source: impl Into<String>,
        record_index: u64,
        payload: Value,
    ) -> Self {
        let source = source.into();
        let payload_bytes = serde_json::to_vec(&payload).expect("serialize JSON value");
        let digest = hex::encode(Sha256::digest(&payload_bytes));
        let identity = serde_json::json!({
            "agent": agent,
            "source": source,
            "record_index": record_index,
            "digest": digest,
        });
        let id_digest = hex::encode(Sha256::digest(identity.to_string().as_bytes()));
        let native_type = payload
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| payload.pointer("/event/type").and_then(Value::as_str))
            .unwrap_or("unknown")
            .to_owned();
        Self {
            id: format!("raw_{}", &id_digest[..24]),
            digest,
            agent,
            source,
            record_index,
            native_type,
            payload,
            ingested_at_ms: now_ms(),
        }
    }

    /// Validates the raw payload digest and stable source identity.
    pub fn validate(&self) -> Result<()> {
        let expected_digest = hex::encode(Sha256::digest(serde_json::to_vec(&self.payload)?));
        anyhow::ensure!(
            self.digest == expected_digest,
            "raw payload digest mismatch"
        );
        let identity = serde_json::json!({
            "agent": self.agent,
            "source": self.source,
            "record_index": self.record_index,
            "digest": self.digest,
        });
        let id_digest = hex::encode(Sha256::digest(identity.to_string().as_bytes()));
        anyhow::ensure!(
            self.id == format!("raw_{}", &id_digest[..24]),
            "raw record ID mismatch"
        );
        Ok(())
    }
}

/// Severity of a normalization diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    /// Conversion retained a usable event with degraded information.
    Warning,
    /// Conversion could not produce a valid canonical event.
    Error,
}

/// Durable validation or normalization problem linked to one raw record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConversionDiagnostic {
    /// Stable diagnostic ID.
    pub id: String,
    /// Raw record that caused the diagnostic.
    pub raw_record_id: String,
    /// Diagnostic severity.
    pub severity: DiagnosticSeverity,
    /// Stable machine-readable code.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// Creation time in Unix milliseconds.
    pub created_at_ms: i64,
}

/// Self-contained formal output of an external-history conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalArchive {
    /// Archive contract version.
    pub spec_version: ArchiveSpecVersion,
    /// Immutable parsed native input records.
    pub raw_records: Vec<RawRecord>,
    /// Valid canonical projections.
    pub events: Vec<EventEnvelope>,
    /// Records that could not be projected without violating the contract.
    pub diagnostics: Vec<ConversionDiagnostic>,
}

impl CanonicalArchive {
    /// Validates record identities, canonical events, and provenance references.
    pub fn validate(&self) -> Result<()> {
        let raw_ids = self
            .raw_records
            .iter()
            .map(|raw| {
                raw.validate()?;
                Ok(raw.id.as_str())
            })
            .collect::<Result<std::collections::HashSet<_>>>()?;
        anyhow::ensure!(
            raw_ids.len() == self.raw_records.len(),
            "duplicate raw record IDs"
        );
        for event in &self.events {
            event.validate()?;
            anyhow::ensure!(
                !event.raw_record_ids.is_empty(),
                "canonical event {} has no raw provenance",
                event.id
            );
            for raw_id in &event.raw_record_ids {
                anyhow::ensure!(
                    raw_ids.contains(raw_id.as_str()),
                    "unknown raw record {raw_id}"
                );
            }
        }
        for diagnostic in &self.diagnostics {
            anyhow::ensure!(
                raw_ids.contains(diagnostic.raw_record_id.as_str()),
                "diagnostic {} references unknown raw record",
                diagnostic.id
            );
        }
        Ok(())
    }
}

/// Generates the authoritative JSON Schema for [`CanonicalArchive`].
pub fn canonical_archive_schema() -> Value {
    let mut schema = serde_json::to_value(schemars::schema_for!(CanonicalArchive))
        .expect("JSON Schema is serializable");
    schema
        .as_object_mut()
        .expect("root schema is an object")
        .insert(
            "$id".into(),
            Value::String("https://semafuse.dev/schemas/canonical-archive-v1.schema.json".into()),
        );
    schema
}

impl ConversionDiagnostic {
    /// Constructs a stable diagnostic.
    pub fn new(
        raw_record_id: impl Into<String>,
        severity: DiagnosticSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let raw_record_id = raw_record_id.into();
        let code = code.into();
        let message = message.into();
        let digest = hex::encode(Sha256::digest(
            format!("{raw_record_id}\0{code}\0{message}").as_bytes(),
        ));
        Self {
            id: format!("diag_{}", &digest[..24]),
            raw_record_id,
            severity,
            code,
            message,
            created_at_ms: now_ms(),
        }
    }
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
    /// Converts the adapters' normalized fields into the formal v1 payload.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent: AgentKind,
        session_id: impl Into<String>,
        turn_id: Option<String>,
        native_id: Option<String>,
        kind: EventKind,
        role: Option<String>,
        content: impl Into<String>,
        tool_name: Option<String>,
        cwd: Option<String>,
        repository: Option<String>,
        occurred_at_ms: Option<i64>,
        metadata: Map<String, Value>,
    ) -> Self {
        let session_id = session_id.into();
        let content = content.into();
        let native_event = metadata
            .get("native_event")
            .and_then(Value::as_str)
            .unwrap_or_else(|| kind.as_str())
            .to_owned();
        let data = match kind {
            EventKind::SessionStart => EventData::SessionStarted,
            EventKind::SessionEnd => EventData::SessionEnded {
                reason: nonempty_string(content),
            },
            EventKind::UserMessage | EventKind::AssistantMessage => EventData::Message {
                role: parse_role(role.as_deref(), &kind),
                content: text_parts(content),
            },
            EventKind::ToolCall => EventData::ToolCall {
                call_id: native_id.clone(),
                tool: tool_name.unwrap_or_else(|| "unknown".into()),
                arguments: parse_json_or_string(content),
                attachments: Vec::new(),
            },
            EventKind::ToolResult => EventData::ToolResult {
                call_id: native_id.clone(),
                tool: tool_name,
                status: ToolStatus::Succeeded,
                output: value_parts(content),
                error: None,
            },
            EventKind::Compaction => {
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
                EventData::Compaction { summary, details }
            }
            EventKind::Error if role.as_deref() == Some("tool") || tool_name.is_some() => {
                let error = ErrorInfo {
                    message: if content.is_empty() {
                        "tool failed".into()
                    } else {
                        content
                    },
                    code: None,
                    details: None,
                };
                EventData::ToolResult {
                    call_id: native_id.clone(),
                    tool: tool_name,
                    status: ToolStatus::Failed,
                    output: Vec::new(),
                    error: Some(error),
                }
            }
            EventKind::Error => EventData::Error {
                error: ErrorInfo {
                    message: if content.is_empty() {
                        "agent error".into()
                    } else {
                        content
                    },
                    code: None,
                    details: None,
                },
            },
            EventKind::Other => EventData::Unknown {
                native_type: native_event.clone(),
            },
        };
        let source = EventSource {
            agent,
            session_id,
            turn_id,
            native_id,
            native_event,
            cwd: cwd.clone(),
            repository: normalize_repository(cwd.as_deref(), repository.as_deref()),
        };
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

fn parse_role(role: Option<&str>, kind: &EventKind) -> MessageRole {
    match role {
        Some("user") => MessageRole::User,
        Some("assistant") => MessageRole::Assistant,
        Some("system") => MessageRole::System,
        Some("developer") => MessageRole::Developer,
        Some("tool") => MessageRole::Tool,
        _ if matches!(kind, EventKind::UserMessage) => MessageRole::User,
        _ if matches!(kind, EventKind::AssistantMessage) => MessageRole::Assistant,
        _ => MessageRole::Unknown,
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

/// Resolve a stable repository/worktree identifier independently from the event cwd.
///
/// An explicit worktree wins and is retained even when it is no longer available
/// locally. Otherwise, the nearest ancestor of `cwd` containing `.git` is used.
/// A working directory outside a Git repository does not become a repository ID.
pub fn normalize_repository(cwd: Option<&str>, repository: Option<&str>) -> Option<String> {
    if let Some(repository) = repository {
        let path = Path::new(repository);
        return path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_str()
            .map(str::to_owned)
            .or_else(|| Some(repository.to_owned()));
    }

    let value = cwd?;
    let path = Path::new(value);
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let start = if normalized.is_file() {
        normalized.parent().unwrap_or(&normalized)
    } else {
        normalized.as_path()
    };
    start
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists())?
        .to_str()
        .map(str::to_owned)
}

/// Filters and bounds a search over canonical events.
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Text interpreted by the selected retrieval strategy.
    pub text: String,
    /// Exact normalized repository filter, or all repositories when absent.
    pub repository: Option<String>,
    /// Exact source-agent filter, or all agents when absent.
    pub agent: Option<AgentKind>,
    /// Maximum number of results requested.
    pub limit: usize,
}

/// Algorithm used to retrieve canonical events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryMode {
    /// Full-text or substring matching, depending on the store.
    Lexical,
    /// Newest events first without considering query text.
    Recent,
    /// Rank fusion of lexical and recent results.
    #[default]
    Hybrid,
}

impl std::str::FromStr for QueryMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lexical" | "fts" => Ok(Self::Lexical),
            "recent" => Ok(Self::Recent),
            "hybrid" => Ok(Self::Hybrid),
            _ => anyhow::bail!("unknown query mode: {value}"),
        }
    }
}

impl QueryMode {
    /// Returns the stable snake-case mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Recent => "recent",
            Self::Hybrid => "hybrid",
        }
    }
}

/// A strategy-owned, derived memory with canonical-event provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    /// Ownership boundary; strategies may mutate only their own namespace.
    pub namespace: String,
    /// Stable identifier unique within the namespace.
    pub id: String,
    /// Searchable derived-memory text.
    pub content: String,
    /// Strategy-defined structured attributes.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// Canonical event IDs from which this memory was derived.
    #[serde(default)]
    pub source_event_ids: Vec<String>,
    /// Creation time in Unix milliseconds.
    pub created_at_ms: i64,
    /// Last update time in Unix milliseconds.
    pub updated_at_ms: i64,
}

/// Namespace-scoped search over derived memories.
#[derive(Debug, Clone, Default)]
pub struct MemoryQuery {
    /// Required strategy namespace.
    pub namespace: String,
    /// Full-text query; empty text requests the most recently updated records.
    pub text: String,
    /// Maximum number of records requested.
    pub limit: usize,
}

/// Atomic change requested by a memory strategy.
#[derive(Debug, Clone)]
pub enum MemoryMutation {
    /// Inserts or updates a record identified by namespace and ID.
    Upsert(MemoryRecord),
    /// Deletes one strategy-owned record.
    Delete {
        /// Namespace that owns the record.
        namespace: String,
        /// Identifier within the namespace.
        id: String,
    },
}

/// A canonical event paired with a strategy-dependent relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Retrieved canonical event, including its provenance fields.
    pub event: CanonicalEvent,
    /// Relative ranking score; meaningful within one result set.
    pub score: f64,
}

/// Returns the current Unix time in milliseconds.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::normalize_repository;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn repository_is_detected_separately_from_cwd() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("repo");
        let cwd = repo.join("src/nested");
        let outside = std::path::PathBuf::from(format!(
            "/semafuse-non-repository-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        assert_eq!(
            normalize_repository(cwd.to_str(), None),
            Some(repo.canonicalize().unwrap().to_string_lossy().into_owned())
        );
        assert_eq!(normalize_repository(outside.to_str(), None), None);
    }

    #[test]
    fn explicit_worktree_is_preserved_when_unavailable_locally() {
        assert_eq!(
            normalize_repository(Some("/current/cwd"), Some("/remote/worktree")),
            Some("/remote/worktree".into())
        );
    }
}
