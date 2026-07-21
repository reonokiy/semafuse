use super::{AgentKind, EventEnvelope, now_ms};
use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Version of the persisted raw-to-canonical archive contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum ArchiveSpecVersion {
    /// The first formal archive schema.
    #[serde(rename = "semafuse.archive/v1")]
    V1,
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
