use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt};

pub const MAGIC: [u8; 4] = *b"SFSD";
pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 0;
pub const HEADER_LEN: usize = 24;
pub const MAX_REQUEST_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_RESPONSE_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ITEMS: usize = 256;
pub const MAX_TEXT_BYTES: usize = 256 * 1024;
pub const MAX_TOTAL_TEXT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_FINGERPRINT_BYTES: usize = 256;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 8_192;
pub const MAX_EMBEDDING_VALUES: usize = 2_097_152;
pub const MAX_ERROR_MESSAGE_BYTES: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrameKind {
    Request = 1,
    Response = 2,
}

impl TryFrom<u8> for FrameKind {
    type Error = HeaderError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Request),
            2 => Ok(Self::Response),
            other => Err(HeaderError::InvalidKind(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameHeader {
    pub major: u16,
    pub minor: u16,
    pub kind: FrameKind,
    pub payload_len: u32,
    pub request_id: u64,
}

impl FrameHeader {
    pub fn new(kind: FrameKind, payload_len: usize, request_id: u64) -> Result<Self, HeaderError> {
        let payload_len = u32::try_from(payload_len).map_err(|_| HeaderError::PayloadTooLarge {
            actual: payload_len,
            maximum: u32::MAX as usize,
        })?;
        Ok(Self {
            major: PROTOCOL_MAJOR,
            minor: PROTOCOL_MINOR,
            kind,
            payload_len,
            request_id,
        })
    }

    pub fn encode(self) -> [u8; HEADER_LEN] {
        let mut bytes = [0_u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4..6].copy_from_slice(&self.major.to_be_bytes());
        bytes[6..8].copy_from_slice(&self.minor.to_be_bytes());
        bytes[8] = self.kind as u8;
        bytes[9] = 0;
        bytes[10..12].copy_from_slice(&0_u16.to_be_bytes());
        bytes[12..16].copy_from_slice(&self.payload_len.to_be_bytes());
        bytes[16..24].copy_from_slice(&self.request_id.to_be_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8; HEADER_LEN]) -> Result<Self, HeaderError> {
        if bytes[0..4] != MAGIC {
            return Err(HeaderError::InvalidMagic);
        }
        if bytes[9] != 0 {
            return Err(HeaderError::InvalidFlags(bytes[9]));
        }
        let reserved = u16::from_be_bytes([bytes[10], bytes[11]]);
        if reserved != 0 {
            return Err(HeaderError::InvalidReserved(reserved));
        }
        Ok(Self {
            major: u16::from_be_bytes([bytes[4], bytes[5]]),
            minor: u16::from_be_bytes([bytes[6], bytes[7]]),
            kind: FrameKind::try_from(bytes[8])?,
            payload_len: u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            request_id: u64::from_be_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22],
                bytes[23],
            ]),
        })
    }

    pub fn require_supported_version(self) -> Result<Self, HeaderError> {
        if self.major != PROTOCOL_MAJOR || self.minor != PROTOCOL_MINOR {
            return Err(HeaderError::UnsupportedVersion {
                major: self.major,
                minor: self.minor,
            });
        }
        Ok(self)
    }

    pub fn require_kind(self, expected: FrameKind) -> Result<Self, HeaderError> {
        if self.kind != expected {
            return Err(HeaderError::UnexpectedKind {
                actual: self.kind,
                expected,
            });
        }
        Ok(self)
    }

    pub fn require_payload_limit(self, maximum: usize) -> Result<Self, HeaderError> {
        let actual = self.payload_len as usize;
        if actual > maximum {
            return Err(HeaderError::PayloadTooLarge { actual, maximum });
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeaderError {
    InvalidMagic,
    InvalidKind(u8),
    InvalidFlags(u8),
    InvalidReserved(u16),
    UnsupportedVersion {
        major: u16,
        minor: u16,
    },
    UnexpectedKind {
        actual: FrameKind,
        expected: FrameKind,
    },
    PayloadTooLarge {
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => formatter.write_str("invalid semafused frame magic"),
            Self::InvalidKind(value) => {
                write!(formatter, "invalid semafused frame kind {value}")
            }
            Self::InvalidFlags(value) => {
                write!(formatter, "unsupported semafused frame flags {value:#04x}")
            }
            Self::InvalidReserved(value) => write!(
                formatter,
                "non-zero semafused reserved header field {value:#06x}"
            ),
            Self::UnsupportedVersion { major, minor } => write!(
                formatter,
                "unsupported semafused protocol version {major}.{minor}"
            ),
            Self::UnexpectedKind { actual, expected } => write!(
                formatter,
                "unexpected semafused frame kind {actual:?}; expected {expected:?}"
            ),
            Self::PayloadTooLarge { actual, maximum } => write!(
                formatter,
                "semafused payload is {actual} bytes; maximum is {maximum}"
            ),
        }
    }
}

impl std::error::Error for HeaderError {}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(
    tag = "service",
    content = "request",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RequestEnvelope {
    Daemon(DaemonRequest),
    Models(ModelRequest),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum DaemonRequest {
    Status {},
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelRequest {
    Embed {
        expected_model_fingerprint: String,
        input: Vec<String>,
    },
    Rerank {
        expected_model_fingerprint: String,
        query: String,
        documents: Vec<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResponseEnvelope {
    Ok { response: ServiceResponse },
    Error { error: ProtocolError },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(
    tag = "service",
    content = "response",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ServiceResponse {
    Daemon(DaemonResponse),
    Models(ModelResponse),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum DaemonResponse {
    Status { status: DaemonStatus },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelResponse {
    Embed {
        model_fingerprint: String,
        data: Vec<IndexedEmbedding>,
    },
    Rerank {
        model_fingerprint: String,
        results: Vec<IndexedScore>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonStatus {
    pub state: DaemonState,
    pub daemon_version: String,
    pub services: BTreeMap<String, serde_json::Value>,
}

impl DaemonStatus {
    pub fn models(&self) -> Result<Option<ModelServiceStatus>, serde_json::Error> {
        self.services
            .get("models")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelServiceStatus {
    pub state: ServiceState,
    pub service_version: String,
    pub embedding: Option<ModelIdentity>,
    pub reranker: Option<ModelIdentity>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    Ready,
    Degraded,
    Failed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceState {
    Ready,
    Degraded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub fingerprint: String,
    pub model: String,
    pub dimensions: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexedEmbedding {
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndexedScore {
    pub index: u32,
    pub score: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    UnsupportedVersion,
    MalformedRequest,
    PayloadTooLarge,
    InvalidRequest,
    Unauthorized,
    NotReady,
    Overloaded,
    DeadlineExceeded,
    ModelMismatch,
    InferenceFailed,
    Internal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_has_stable_golden_layout() {
        let header =
            FrameHeader::new(FrameKind::Request, 0x0102_0304, 0x0102_0304_0506_0708).unwrap();
        assert_eq!(
            header.encode(),
            [
                b'S', b'F', b'S', b'D', 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8
            ]
        );
        assert_eq!(FrameHeader::decode(&header.encode()).unwrap(), header);
    }

    #[test]
    fn header_rejects_reserved_fields_and_wrong_version() {
        let mut bytes = FrameHeader::new(FrameKind::Response, 1, 9)
            .unwrap()
            .encode();
        bytes[9] = 1;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(HeaderError::InvalidFlags(1))
        ));
        bytes[9] = 0;
        bytes[10] = 1;
        assert!(matches!(
            FrameHeader::decode(&bytes),
            Err(HeaderError::InvalidReserved(0x0100))
        ));
        let mut header = FrameHeader::new(FrameKind::Response, 1, 9).unwrap();
        header.major = 2;
        assert!(matches!(
            header.require_supported_version(),
            Err(HeaderError::UnsupportedVersion { major: 2, minor: 0 })
        ));
    }

    #[test]
    fn json_schema_routes_requests_by_service_and_rejects_unknown_fields() {
        let request = RequestEnvelope::Models(ModelRequest::Embed {
            expected_model_fingerprint: "sha256:abc".into(),
            input: vec!["hello".into()],
        });
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["service"], "models");
        assert_eq!(json["request"]["op"], "embed");
        assert_eq!(json["request"]["input"][0], "hello");
        assert!(
            serde_json::from_str::<RequestEnvelope>(
                r#"{"service":"daemon","request":{"op":"status","unexpected":true}}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<RequestEnvelope>(
                r#"{"service":"daemon","request":{"op":"status"},"unexpected":true}"#
            )
            .is_err()
        );
    }
}
