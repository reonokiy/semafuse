use crate::model::CanonicalEvent;
use anyhow::Result;
use serde_json::Value;

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

/// A canonical event projected from adapter-native history, paired with the
/// exact native value from which attachments must be extracted.
///
/// Keeping attachment ownership at this boundary prevents attachments from all
/// being assigned to the final event in a session. If attachment processing
/// fails, the importer rejects that source instead of writing an event with
/// silently missing attachments.
#[derive(Debug, Clone)]
pub struct ProjectedEvent {
    /// Normalized event to ingest after attachment extraction.
    pub event: CanonicalEvent,
    /// Native subtree whose attachments belong specifically to `event`.
    pub attachment_source: Value,
}

impl ProjectedEvent {
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

/// Decodes one adapter-owned native history schema and projects it into canonical events.
///
/// This trait deliberately does not define a shared import record type. Each
/// adapter owns its persisted history structs and converts them into the common
/// canonical schema only after native parsing succeeds.
pub trait NativeHistoryProjection: Sized {
    /// Decodes raw JSON values from a history source into this adapter's native schema.
    fn decode_native(values: &[Value], context: &ImportContext) -> Result<Self>;

    /// Converts this adapter's native history schema into canonical projected events.
    fn project_native(&self, context: &ImportContext) -> Result<Vec<ProjectedEvent>>;
}

/// Runs one adapter's native history parser and canonical projection.
pub fn project_native_history<P: NativeHistoryProjection>(
    values: &[Value],
    context: &ImportContext,
) -> Result<Vec<ProjectedEvent>> {
    let history = P::decode_native(values, context)?;
    history.project_native(context)
}
