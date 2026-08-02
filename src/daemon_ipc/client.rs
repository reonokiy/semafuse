use super::protocol::{
    DaemonRequest, DaemonResponse, DaemonStatus, FrameHeader, FrameKind, HEADER_LEN, HeaderError,
    IndexedEmbedding, IndexedScore, MAX_EMBEDDING_DIMENSIONS, MAX_EMBEDDING_VALUES,
    MAX_ERROR_MESSAGE_BYTES, MAX_FINGERPRINT_BYTES, MAX_ITEMS, MAX_REQUEST_PAYLOAD_BYTES,
    MAX_RESPONSE_PAYLOAD_BYTES, MAX_TEXT_BYTES, MAX_TOTAL_TEXT_BYTES, ModelIdentity, ModelRequest,
    ModelResponse, ModelServiceStatus, ProtocolError, RequestEnvelope, ResponseEnvelope,
    ServiceResponse,
};
use rustix::process::geteuid;
use std::{
    fmt, io,
    os::unix::{
        ffi::OsStrExt,
        fs::{FileTypeExt, MetadataExt},
    },
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    time::timeout,
};

const MAX_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
#[cfg(any(target_os = "linux", target_os = "android"))]
const MAX_SOCKET_PATH_BYTES: usize = 107;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const MAX_SOCKET_PATH_BYTES: usize = 103;
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct DaemonClient {
    inner: ModelServiceClient,
}

impl DaemonClient {
    pub fn new(
        socket_path: impl Into<PathBuf>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Self {
        Self {
            inner: ModelServiceClient::new(
                socket_path,
                "daemon-status",
                connect_timeout,
                request_timeout,
            ),
        }
    }

    pub fn socket_path(&self) -> &Path {
        self.inner.socket_path()
    }

    pub fn connect_timeout(&self) -> Duration {
        self.inner.connect_timeout()
    }

    pub fn request_timeout(&self) -> Duration {
        self.inner.request_timeout()
    }

    pub fn validate(&self) -> Result<(), DaemonClientError> {
        self.inner.validate()
    }

    pub async fn status_at(
        socket_path: impl Into<PathBuf>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<DaemonStatus, DaemonClientError> {
        Self::new(socket_path, connect_timeout, request_timeout)
            .status()
            .await
    }

    pub async fn status(&self) -> Result<DaemonStatus, DaemonClientError> {
        self.inner.status().await
    }
}

#[derive(Clone, Debug)]
pub struct ModelServiceClient {
    socket_path: PathBuf,
    model_fingerprint: String,
    connect_timeout: Duration,
    request_timeout: Duration,
}

impl ModelServiceClient {
    pub fn new(
        socket_path: impl Into<PathBuf>,
        model_fingerprint: impl Into<String>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Self {
        Self {
            socket_path: socket_path.into(),
            model_fingerprint: model_fingerprint.into(),
            connect_timeout,
            request_timeout,
        }
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn validate(&self) -> Result<(), DaemonClientError> {
        if !self.socket_path.is_absolute() {
            return Err(DaemonClientError::InvalidConfig(
                "semafused socket path must be absolute".into(),
            ));
        }
        let bytes = self.socket_path.as_os_str().as_bytes();
        if bytes.is_empty() || bytes.len() > MAX_SOCKET_PATH_BYTES || bytes.contains(&0) {
            return Err(DaemonClientError::InvalidConfig(format!(
                "semafused socket path must contain 1..={MAX_SOCKET_PATH_BYTES} non-NUL bytes"
            )));
        }
        validate_fingerprint(&self.model_fingerprint)?;
        if self.connect_timeout.is_zero() || self.connect_timeout > MAX_CONNECT_TIMEOUT {
            return Err(DaemonClientError::InvalidConfig(format!(
                "semafused connect timeout must be greater than zero and at most {} seconds",
                MAX_CONNECT_TIMEOUT.as_secs()
            )));
        }
        if self.request_timeout.is_zero() || self.request_timeout > MAX_REQUEST_TIMEOUT {
            return Err(DaemonClientError::InvalidConfig(format!(
                "semafused request timeout must be greater than zero and at most {} seconds",
                MAX_REQUEST_TIMEOUT.as_secs()
            )));
        }
        Ok(())
    }

    pub async fn status_at(
        socket_path: impl Into<PathBuf>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<DaemonStatus, DaemonClientError> {
        Self::new(
            socket_path,
            "status-probe",
            connect_timeout,
            request_timeout,
        )
        .status()
        .await
    }

    pub async fn status(&self) -> Result<DaemonStatus, DaemonClientError> {
        match self
            .request(RequestEnvelope::Daemon(DaemonRequest::Status {}))
            .await?
        {
            ServiceResponse::Daemon(DaemonResponse::Status { status }) => {
                validate_status(&status)?;
                Ok(status)
            }
            response => Err(unexpected_response("status", &response)),
        }
    }

    pub async fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, DaemonClientError> {
        self.validate()?;
        validate_text_collection("embedding input", input)?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let response = self
            .request(RequestEnvelope::Models(ModelRequest::Embed {
                expected_model_fingerprint: self.model_fingerprint.clone(),
                input: input.to_vec(),
            }))
            .await?;
        let (fingerprint, data) = match response {
            ServiceResponse::Models(ModelResponse::Embed {
                model_fingerprint,
                data,
            }) => (model_fingerprint, data),
            other => return Err(unexpected_response("embed", &other)),
        };
        self.require_expected_fingerprint(&fingerprint)?;
        validate_embeddings(data, input.len())
    }

    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<f64>, DaemonClientError> {
        self.validate()?;
        validate_text("rerank query", query)?;
        validate_text_collection("rerank documents", documents)?;
        let total = query
            .len()
            .checked_add(documents.iter().map(String::len).sum::<usize>())
            .ok_or_else(|| DaemonClientError::InvalidInput("rerank text size overflow".into()))?;
        if total > MAX_TOTAL_TEXT_BYTES {
            return Err(DaemonClientError::InvalidInput(format!(
                "rerank request contains {total} text bytes; maximum is {MAX_TOTAL_TEXT_BYTES}"
            )));
        }
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let response = self
            .request(RequestEnvelope::Models(ModelRequest::Rerank {
                expected_model_fingerprint: self.model_fingerprint.clone(),
                query: query.into(),
                documents: documents.to_vec(),
            }))
            .await?;
        let (fingerprint, results) = match response {
            ServiceResponse::Models(ModelResponse::Rerank {
                model_fingerprint,
                results,
            }) => (model_fingerprint, results),
            other => return Err(unexpected_response("rerank", &other)),
        };
        self.require_expected_fingerprint(&fingerprint)?;
        validate_scores(results, documents.len())
    }

    async fn request(
        &self,
        request: RequestEnvelope,
    ) -> Result<ServiceResponse, DaemonClientError> {
        self.validate()?;
        self.verify_socket_file()?;
        let payload = serde_json::to_vec(&request).map_err(|error| {
            DaemonClientError::Protocol(format!("could not encode request: {error}"))
        })?;
        if payload.len() > MAX_REQUEST_PAYLOAD_BYTES {
            return Err(DaemonClientError::InvalidInput(format!(
                "encoded semafused request is {} bytes; maximum is {MAX_REQUEST_PAYLOAD_BYTES}",
                payload.len()
            )));
        }
        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let header = FrameHeader::new(FrameKind::Request, payload.len(), request_id)
            .map_err(DaemonClientError::Header)?;
        let mut stream = timeout(self.connect_timeout, UnixStream::connect(&self.socket_path))
            .await
            .map_err(|_| DaemonClientError::ConnectTimeout(self.connect_timeout))?
            .map_err(|source| DaemonClientError::Io {
                context: "connect to semafused",
                source,
            })?;
        self.verify_peer(&stream)?;
        let response_payload = timeout(
            self.request_timeout,
            exchange(&mut stream, header, &payload, request_id),
        )
        .await
        .map_err(|_| DaemonClientError::RequestTimeout(self.request_timeout))??;
        let envelope: ResponseEnvelope =
            serde_json::from_slice(&response_payload).map_err(|error| {
                DaemonClientError::Protocol(format!("could not decode semafused response: {error}"))
            })?;
        match envelope {
            ResponseEnvelope::Ok { response } => Ok(response),
            ResponseEnvelope::Error { error } => {
                if error.message.len() > MAX_ERROR_MESSAGE_BYTES {
                    return Err(DaemonClientError::Protocol(format!(
                        "semafused error message exceeds {MAX_ERROR_MESSAGE_BYTES} bytes"
                    )));
                }
                Err(DaemonClientError::Remote(error))
            }
        }
    }

    fn require_expected_fingerprint(&self, actual: &str) -> Result<(), DaemonClientError> {
        if actual != self.model_fingerprint {
            return Err(DaemonClientError::ModelMismatch {
                expected: self.model_fingerprint.clone(),
                actual: actual.into(),
            });
        }
        Ok(())
    }

    fn verify_socket_file(&self) -> Result<(), DaemonClientError> {
        let metadata = std::fs::symlink_metadata(&self.socket_path).map_err(|source| {
            DaemonClientError::Io {
                context: "inspect semafused socket",
                source,
            }
        })?;
        if !metadata.file_type().is_socket() {
            return Err(DaemonClientError::SocketSecurity(
                "semafused path is not a Unix socket".into(),
            ));
        }
        let expected = geteuid().as_raw();
        if metadata.uid() != expected {
            return Err(DaemonClientError::SocketSecurity(format!(
                "semafused socket is owned by uid {}, expected {expected}",
                metadata.uid()
            )));
        }
        if metadata.mode() & 0o077 != 0 {
            return Err(DaemonClientError::SocketSecurity(
                "semafused socket must not grant group or other permissions".into(),
            ));
        }
        Ok(())
    }

    fn verify_peer(&self, stream: &UnixStream) -> Result<(), DaemonClientError> {
        let credentials = stream.peer_cred().map_err(|source| DaemonClientError::Io {
            context: "inspect semafused peer credentials",
            source,
        })?;
        let expected = geteuid().as_raw();
        let actual = credentials.uid();
        if actual != expected {
            return Err(DaemonClientError::SocketSecurity(format!(
                "semafused peer uid is {actual}, expected {expected}"
            )));
        }
        Ok(())
    }
}

async fn exchange(
    stream: &mut UnixStream,
    header: FrameHeader,
    payload: &[u8],
    request_id: u64,
) -> Result<Vec<u8>, DaemonClientError> {
    stream
        .write_all(&header.encode())
        .await
        .map_err(|source| DaemonClientError::Io {
            context: "write semafused request header",
            source,
        })?;
    stream
        .write_all(payload)
        .await
        .map_err(|source| DaemonClientError::Io {
            context: "write semafused request payload",
            source,
        })?;
    stream
        .shutdown()
        .await
        .map_err(|source| DaemonClientError::Io {
            context: "finish semafused request",
            source,
        })?;
    let mut bytes = [0_u8; HEADER_LEN];
    stream
        .read_exact(&mut bytes)
        .await
        .map_err(|source| DaemonClientError::Io {
            context: "read semafused response header",
            source,
        })?;
    let header = FrameHeader::decode(&bytes)
        .and_then(FrameHeader::require_supported_version)
        .and_then(|header| header.require_kind(FrameKind::Response))
        .and_then(|header| header.require_payload_limit(MAX_RESPONSE_PAYLOAD_BYTES))
        .map_err(DaemonClientError::Header)?;
    if header.request_id != request_id {
        return Err(DaemonClientError::Protocol(format!(
            "semafused response request id {} does not match {request_id}",
            header.request_id
        )));
    }
    if header.payload_len == 0 {
        return Err(DaemonClientError::Protocol(
            "semafused returned an empty response payload".into(),
        ));
    }
    let mut response = vec![0_u8; header.payload_len as usize];
    stream
        .read_exact(&mut response)
        .await
        .map_err(|source| DaemonClientError::Io {
            context: "read semafused response payload",
            source,
        })?;
    Ok(response)
}

fn validate_fingerprint(value: &str) -> Result<(), DaemonClientError> {
    if value.is_empty()
        || value.len() > MAX_FINGERPRINT_BYTES
        || !value.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(DaemonClientError::InvalidConfig(format!(
            "model fingerprint must contain 1..={MAX_FINGERPRINT_BYTES} printable ASCII bytes"
        )));
    }
    Ok(())
}

fn validate_text_collection(
    label: &'static str,
    texts: &[String],
) -> Result<(), DaemonClientError> {
    if texts.len() > MAX_ITEMS {
        return Err(DaemonClientError::InvalidInput(format!(
            "{label} contains {} items; maximum is {MAX_ITEMS}",
            texts.len()
        )));
    }
    let mut total = 0_usize;
    for text in texts {
        validate_text(label, text)?;
        total = total
            .checked_add(text.len())
            .ok_or_else(|| DaemonClientError::InvalidInput(format!("{label} size overflow")))?;
    }
    if total > MAX_TOTAL_TEXT_BYTES {
        return Err(DaemonClientError::InvalidInput(format!(
            "{label} contains {total} text bytes; maximum is {MAX_TOTAL_TEXT_BYTES}"
        )));
    }
    Ok(())
}

fn validate_text(label: &'static str, text: &str) -> Result<(), DaemonClientError> {
    if text.is_empty() || text.len() > MAX_TEXT_BYTES {
        return Err(DaemonClientError::InvalidInput(format!(
            "each {label} must contain 1..={MAX_TEXT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_embeddings(
    data: Vec<IndexedEmbedding>,
    expected_count: usize,
) -> Result<Vec<Vec<f32>>, DaemonClientError> {
    if data.len() != expected_count {
        return Err(DaemonClientError::InvalidEmbedding(format!(
            "semafused returned {} embeddings for {expected_count} inputs",
            data.len()
        )));
    }
    let mut ordered: Vec<Option<Vec<f32>>> = std::iter::repeat_with(|| None)
        .take(expected_count)
        .collect();
    let mut dimensions = None;
    let mut total_values = 0_usize;
    for item in data {
        let slot = ordered.get_mut(item.index as usize).ok_or_else(|| {
            DaemonClientError::InvalidEmbedding(format!(
                "semafused returned out-of-range embedding index {}",
                item.index
            ))
        })?;
        if slot.is_some() {
            return Err(DaemonClientError::InvalidEmbedding(format!(
                "semafused returned duplicate embedding index {}",
                item.index
            )));
        }
        if item.embedding.is_empty() || item.embedding.len() > MAX_EMBEDDING_DIMENSIONS {
            return Err(DaemonClientError::InvalidEmbedding(format!(
                "embedding {} has {} dimensions; expected 1..={MAX_EMBEDDING_DIMENSIONS}",
                item.index,
                item.embedding.len()
            )));
        }
        if item.embedding.iter().any(|value| !value.is_finite()) {
            return Err(DaemonClientError::InvalidEmbedding(format!(
                "embedding {} contains a non-finite value",
                item.index
            )));
        }
        if let Some(expected) = dimensions {
            if item.embedding.len() != expected {
                return Err(DaemonClientError::InvalidEmbedding(format!(
                    "embedding {} has {} dimensions; expected {expected}",
                    item.index,
                    item.embedding.len()
                )));
            }
        } else {
            dimensions = Some(item.embedding.len());
        }
        total_values = total_values
            .checked_add(item.embedding.len())
            .ok_or_else(|| {
                DaemonClientError::InvalidEmbedding("embedding value count overflow".into())
            })?;
        if total_values > MAX_EMBEDDING_VALUES {
            return Err(DaemonClientError::InvalidEmbedding(format!(
                "embedding response contains {total_values} values; maximum is {MAX_EMBEDDING_VALUES}"
            )));
        }
        *slot = Some(item.embedding);
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                DaemonClientError::InvalidEmbedding(format!(
                    "semafused omitted embedding index {index}"
                ))
            })
        })
        .collect()
}

fn validate_scores(
    results: Vec<IndexedScore>,
    expected_count: usize,
) -> Result<Vec<f64>, DaemonClientError> {
    if results.len() != expected_count {
        return Err(DaemonClientError::InvalidRerank(format!(
            "semafused returned {} scores for {expected_count} documents",
            results.len()
        )));
    }
    let mut ordered = vec![None; expected_count];
    for item in results {
        if !item.score.is_finite() {
            return Err(DaemonClientError::InvalidRerank(format!(
                "rerank score {} is not finite",
                item.index
            )));
        }
        let slot = ordered.get_mut(item.index as usize).ok_or_else(|| {
            DaemonClientError::InvalidRerank(format!(
                "semafused returned out-of-range rerank index {}",
                item.index
            ))
        })?;
        if slot.replace(item.score).is_some() {
            return Err(DaemonClientError::InvalidRerank(format!(
                "semafused returned duplicate rerank index {}",
                item.index
            )));
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                DaemonClientError::InvalidRerank(format!("semafused omitted rerank index {index}"))
            })
        })
        .collect()
}

fn validate_status(status: &DaemonStatus) -> Result<(), DaemonClientError> {
    if status.daemon_version.is_empty() || status.daemon_version.len() > MAX_TEXT_BYTES {
        return Err(DaemonClientError::Protocol(
            "semafused returned an invalid daemon version".into(),
        ));
    }
    for name in status.services.keys() {
        if name.is_empty()
            || name.len() > 64
            || !name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
            })
        {
            return Err(DaemonClientError::Protocol(
                "semafused returned an invalid service name".into(),
            ));
        }
    }
    if let Some(models) = status.models().map_err(|error| {
        DaemonClientError::Protocol(format!("invalid models service status: {error}"))
    })? {
        validate_model_status(&models)?;
    }
    Ok(())
}

fn validate_model_status(status: &ModelServiceStatus) -> Result<(), DaemonClientError> {
    if status.service_version.is_empty() || status.service_version.len() > MAX_TEXT_BYTES {
        return Err(DaemonClientError::Protocol(
            "semafused returned an invalid models service version".into(),
        ));
    }
    for identity in [&status.embedding, &status.reranker].into_iter().flatten() {
        validate_identity(identity)?;
    }
    Ok(())
}

fn validate_identity(identity: &ModelIdentity) -> Result<(), DaemonClientError> {
    validate_fingerprint(&identity.fingerprint)
        .map_err(|error| DaemonClientError::Protocol(format!("invalid model identity: {error}")))?;
    if identity.model.is_empty() || identity.model.len() > MAX_TEXT_BYTES {
        return Err(DaemonClientError::Protocol(
            "semafused returned an invalid model name".into(),
        ));
    }
    if let Some(dimensions) = identity.dimensions
        && (dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS)
    {
        return Err(DaemonClientError::Protocol(format!(
            "semafused returned invalid model dimensions {dimensions}"
        )));
    }
    Ok(())
}

fn unexpected_response(expected: &'static str, response: &ServiceResponse) -> DaemonClientError {
    let actual = match response {
        ServiceResponse::Daemon(DaemonResponse::Status { .. }) => "status",
        ServiceResponse::Models(ModelResponse::Embed { .. }) => "embed",
        ServiceResponse::Models(ModelResponse::Rerank { .. }) => "rerank",
    };
    DaemonClientError::UnexpectedResponse { expected, actual }
}

#[derive(Debug)]
pub enum DaemonClientError {
    InvalidConfig(String),
    InvalidInput(String),
    SocketSecurity(String),
    ConnectTimeout(Duration),
    RequestTimeout(Duration),
    Io {
        context: &'static str,
        source: io::Error,
    },
    Header(HeaderError),
    Protocol(String),
    Remote(ProtocolError),
    UnexpectedResponse {
        expected: &'static str,
        actual: &'static str,
    },
    ModelMismatch {
        expected: String,
        actual: String,
    },
    InvalidEmbedding(String),
    InvalidRerank(String),
}

impl DaemonClientError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Remote(error) if error.retryable)
    }
}

impl fmt::Display for DaemonClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(value) => write!(formatter, "invalid semafused config: {value}"),
            Self::InvalidInput(value) => write!(formatter, "invalid semafused input: {value}"),
            Self::SocketSecurity(value) => {
                write!(formatter, "semafused socket security check failed: {value}")
            }
            Self::ConnectTimeout(value) => {
                write!(
                    formatter,
                    "timed out connecting to semafused after {value:?}"
                )
            }
            Self::RequestTimeout(value) => {
                write!(formatter, "semafused request timed out after {value:?}")
            }
            Self::Io { context, source } => write!(formatter, "{context}: {source}"),
            Self::Header(value) => write!(formatter, "invalid semafused frame: {value}"),
            Self::Protocol(value) => write!(formatter, "invalid semafused protocol: {value}"),
            Self::Remote(value) => {
                write!(
                    formatter,
                    "semafused returned {:?}: {}",
                    value.code, value.message
                )
            }
            Self::UnexpectedResponse { expected, actual } => write!(
                formatter,
                "semafused returned {actual} response to {expected} request"
            ),
            Self::ModelMismatch { expected, actual } => write!(
                formatter,
                "semafused used fingerprint {actual}; expected {expected}"
            ),
            Self::InvalidEmbedding(value) => {
                write!(formatter, "invalid semafused embedding response: {value}")
            }
            Self::InvalidRerank(value) => {
                write!(formatter, "invalid semafused rerank response: {value}")
            }
        }
    }
}

impl std::error::Error for DaemonClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Header(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::Permissions, os::unix::fs::PermissionsExt};
    use tempfile::TempDir;
    use tokio::{net::UnixListener, task::JoinHandle};

    fn endpoint(path: &Path, fingerprint: &str) -> ModelServiceClient {
        ModelServiceClient::new(
            path,
            fingerprint,
            Duration::from_secs(1),
            Duration::from_secs(2),
        )
    }

    fn serve_once(response: ResponseEnvelope) -> (TempDir, PathBuf, JoinHandle<RequestEnvelope>) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("semafused.sock");
        let listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, Permissions::from_mode(0o600)).unwrap();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = [0_u8; HEADER_LEN];
            stream.read_exact(&mut bytes).await.unwrap();
            let header = FrameHeader::decode(&bytes)
                .and_then(FrameHeader::require_supported_version)
                .and_then(|header| header.require_kind(FrameKind::Request))
                .and_then(|header| header.require_payload_limit(MAX_REQUEST_PAYLOAD_BYTES))
                .unwrap();
            let mut payload = vec![0; header.payload_len as usize];
            stream.read_exact(&mut payload).await.unwrap();
            let request: RequestEnvelope = serde_json::from_slice(&payload).unwrap();
            let payload = serde_json::to_vec(&response).unwrap();
            let response_header =
                FrameHeader::new(FrameKind::Response, payload.len(), header.request_id).unwrap();
            stream.write_all(&response_header.encode()).await.unwrap();
            stream.write_all(&payload).await.unwrap();
            request
        });
        (directory, path, task)
    }

    #[tokio::test]
    async fn embed_reorders_and_validates_indexed_vectors() {
        let response = ResponseEnvelope::Ok {
            response: ServiceResponse::Models(ModelResponse::Embed {
                model_fingerprint: "sha256:test".into(),
                data: vec![
                    IndexedEmbedding {
                        index: 1,
                        embedding: vec![3.0, 4.0],
                    },
                    IndexedEmbedding {
                        index: 0,
                        embedding: vec![1.0, 2.0],
                    },
                ],
            }),
        };
        let (_directory, path, server) = serve_once(response);
        let result = endpoint(&path, "sha256:test")
            .embed(&["first".into(), "second".into()])
            .await
            .unwrap();
        assert_eq!(result, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(matches!(
            server.await.unwrap(),
            RequestEnvelope::Models(ModelRequest::Embed { .. })
        ));
    }

    #[tokio::test]
    async fn rerank_reorders_scores_and_rejects_mismatch() {
        let response = ResponseEnvelope::Ok {
            response: ServiceResponse::Models(ModelResponse::Rerank {
                model_fingerprint: "sha256:reranker".into(),
                results: vec![
                    IndexedScore {
                        index: 1,
                        score: 0.2,
                    },
                    IndexedScore {
                        index: 0,
                        score: 0.8,
                    },
                ],
            }),
        };
        let (_directory, path, server) = serve_once(response);
        let scores = endpoint(&path, "sha256:reranker")
            .rerank("query", &["first".into(), "second".into()])
            .await
            .unwrap();
        assert_eq!(scores, vec![0.8, 0.2]);
        assert!(matches!(
            server.await.unwrap(),
            RequestEnvelope::Models(ModelRequest::Rerank { .. })
        ));

        let response = ResponseEnvelope::Ok {
            response: ServiceResponse::Models(ModelResponse::Rerank {
                model_fingerprint: "sha256:other".into(),
                results: vec![IndexedScore {
                    index: 0,
                    score: 0.5,
                }],
            }),
        };
        let (_directory, path, server) = serve_once(response);
        let error = endpoint(&path, "sha256:expected")
            .rerank("query", &["document".into()])
            .await
            .unwrap_err();
        assert!(matches!(error, DaemonClientError::ModelMismatch { .. }));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_insecure_socket_permissions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("semafused.sock");
        let _listener = UnixListener::bind(&path).unwrap();
        std::fs::set_permissions(&path, Permissions::from_mode(0o666)).unwrap();
        let error = endpoint(&path, "sha256:test").status().await.unwrap_err();
        assert!(matches!(error, DaemonClientError::SocketSecurity(_)));
    }

    #[test]
    fn validates_path_fingerprint_and_timeouts() {
        let value = ModelServiceClient::new(
            "relative.sock",
            "sha256:test",
            Duration::from_secs(1),
            Duration::from_secs(2),
        );
        assert!(matches!(
            value.validate(),
            Err(DaemonClientError::InvalidConfig(_))
        ));
        let value = ModelServiceClient::new(
            "/tmp/semafused.sock",
            "contains whitespace",
            Duration::from_secs(1),
            Duration::from_secs(2),
        );
        assert!(matches!(
            value.validate(),
            Err(DaemonClientError::InvalidConfig(_))
        ));
    }
}
