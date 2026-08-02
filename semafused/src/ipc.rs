//! Async server-side framing for the shared `semafuse::daemon_ipc` protocol.

use std::io;

use semafuse::daemon_ipc::protocol::{
    ErrorCode, FrameHeader, FrameKind, HEADER_LEN, HeaderError, MAX_REQUEST_PAYLOAD_BYTES,
    MAX_RESPONSE_PAYLOAD_BYTES, ProtocolError, RequestEnvelope, ResponseEnvelope,
};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub use semafuse::daemon_ipc::protocol;

#[derive(Debug)]
pub struct IncomingRequest {
    pub request_id: u64,
    pub request: RequestEnvelope,
}

#[derive(Debug, Error)]
pub enum ReadFrameError {
    #[error("I/O error while reading frame: {source}")]
    Io {
        request_id: Option<u64>,
        #[source]
        source: io::Error,
    },
    #[error("invalid frame header: {source}")]
    Header {
        request_id: Option<u64>,
        #[source]
        source: HeaderError,
    },
    #[error("request payload is empty")]
    EmptyPayload { request_id: u64 },
    #[error("request JSON is malformed: {source}")]
    MalformedPayload {
        request_id: u64,
        #[source]
        source: serde_json::Error,
    },
}

impl ReadFrameError {
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Io { request_id, .. } | Self::Header { request_id, .. } => *request_id,
            Self::EmptyPayload { request_id } | Self::MalformedPayload { request_id, .. } => {
                Some(*request_id)
            }
        }
    }

    pub fn to_protocol_error(&self) -> ProtocolError {
        let code = match self {
            Self::Header {
                source: HeaderError::UnsupportedVersion { .. },
                ..
            } => ErrorCode::UnsupportedVersion,
            Self::Header {
                source: HeaderError::PayloadTooLarge { .. },
                ..
            } => ErrorCode::PayloadTooLarge,
            _ => ErrorCode::MalformedRequest,
        };
        ProtocolError {
            code,
            message: truncate_error_message(self.to_string()),
            retryable: false,
            retry_after_ms: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum WriteFrameError {
    #[error("failed to encode response JSON: {0}")]
    Encode(#[source] serde_json::Error),
    #[error("invalid response frame header: {0}")]
    Header(#[from] HeaderError),
    #[error("I/O error while writing response: {0}")]
    Io(#[from] io::Error),
}

/// Reads one shared-protocol request. `Ok(None)` means a clean EOF before any
/// header byte; partial headers and bodies remain protocol errors.
pub async fn read_request<R>(reader: &mut R) -> Result<Option<IncomingRequest>, ReadFrameError>
where
    R: AsyncRead + Unpin,
{
    let mut bytes = [0_u8; HEADER_LEN];
    let count = reader
        .read(&mut bytes[..1])
        .await
        .map_err(|source| ReadFrameError::Io {
            request_id: None,
            source,
        })?;
    if count == 0 {
        return Ok(None);
    }
    reader
        .read_exact(&mut bytes[1..])
        .await
        .map_err(|source| ReadFrameError::Io {
            request_id: None,
            source,
        })?;

    let header = FrameHeader::decode(&bytes).map_err(|source| ReadFrameError::Header {
        request_id: None,
        source,
    })?;
    let request_id = header.request_id;
    let header = header
        .require_supported_version()
        .and_then(|value| value.require_kind(FrameKind::Request))
        .and_then(|value| value.require_payload_limit(MAX_REQUEST_PAYLOAD_BYTES))
        .map_err(|source| ReadFrameError::Header {
            request_id: Some(request_id),
            source,
        })?;

    if header.payload_len == 0 {
        return Err(ReadFrameError::EmptyPayload { request_id });
    }
    let mut payload = vec![0_u8; header.payload_len as usize];
    reader
        .read_exact(&mut payload)
        .await
        .map_err(|source| ReadFrameError::Io {
            request_id: Some(request_id),
            source,
        })?;
    let request = serde_json::from_slice(&payload)
        .map_err(|source| ReadFrameError::MalformedPayload { request_id, source })?;
    Ok(Some(IncomingRequest {
        request_id,
        request,
    }))
}

pub async fn write_response<W>(
    writer: &mut W,
    request_id: u64,
    response: &ResponseEnvelope,
) -> Result<(), WriteFrameError>
where
    W: AsyncWrite + Unpin,
{
    let payload = serde_json::to_vec(response).map_err(WriteFrameError::Encode)?;
    let header = FrameHeader::new(FrameKind::Response, payload.len(), request_id)?
        .require_payload_limit(MAX_RESPONSE_PAYLOAD_BYTES)?;
    writer.write_all(&header.encode()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

fn truncate_error_message(mut message: String) -> String {
    use semafuse::daemon_ipc::protocol::MAX_ERROR_MESSAGE_BYTES;

    if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        return message;
    }
    let mut boundary = MAX_ERROR_MESSAGE_BYTES;
    while boundary > 0 && !message.is_char_boundary(boundary) {
        boundary -= 1;
    }
    message.truncate(boundary);
    message
}

#[cfg(test)]
mod tests {
    use semafuse::daemon_ipc::protocol::{DaemonRequest, RequestEnvelope};

    use super::*;

    #[tokio::test]
    async fn request_round_trip_uses_shared_header() {
        let expected = RequestEnvelope::Daemon(DaemonRequest::Status {});
        let payload = serde_json::to_vec(&expected).unwrap();
        let header = FrameHeader::new(FrameKind::Request, payload.len(), 42).unwrap();
        let (mut client, mut server) = tokio::io::duplex(4096);

        let send = tokio::spawn(async move {
            client.write_all(&header.encode()).await.unwrap();
            client.write_all(&payload).await.unwrap();
        });
        let actual = read_request(&mut server).await.unwrap().unwrap();
        send.await.unwrap();

        assert_eq!(actual.request_id, 42);
        assert_eq!(actual.request, expected);
    }

    #[tokio::test]
    async fn payload_limit_is_checked_before_allocation() {
        let mut header = FrameHeader::new(FrameKind::Request, 1, 9).unwrap();
        header.payload_len = (MAX_REQUEST_PAYLOAD_BYTES + 1) as u32;
        let (mut client, mut server) = tokio::io::duplex(HEADER_LEN);
        client.write_all(&header.encode()).await.unwrap();

        let error = read_request(&mut server).await.unwrap_err();
        assert!(matches!(
            error,
            ReadFrameError::Header {
                source: HeaderError::PayloadTooLarge { .. },
                request_id: Some(9)
            }
        ));
    }
}
