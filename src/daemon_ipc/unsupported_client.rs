use super::protocol::DaemonStatus;
use std::{
    fmt,
    path::{Path, PathBuf},
    time::Duration,
};

const UNSUPPORTED: &str = "semafused Unix-domain socket transport is unavailable on this platform";

#[derive(Clone, Debug)]
pub struct DaemonClient {
    socket_path: PathBuf,
    connect_timeout: Duration,
    request_timeout: Duration,
}

impl DaemonClient {
    pub fn new(
        socket_path: impl Into<PathBuf>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Self {
        Self {
            socket_path: socket_path.into(),
            connect_timeout,
            request_timeout,
        }
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    pub fn validate(&self) -> Result<(), DaemonClientError> {
        Err(DaemonClientError)
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
        Err(DaemonClientError)
    }
}

#[derive(Clone, Debug)]
pub struct ModelServiceClient {
    daemon: DaemonClient,
    model_fingerprint: String,
}

impl ModelServiceClient {
    pub fn new(
        socket_path: impl Into<PathBuf>,
        model_fingerprint: impl Into<String>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Self {
        Self {
            daemon: DaemonClient::new(socket_path, connect_timeout, request_timeout),
            model_fingerprint: model_fingerprint.into(),
        }
    }

    pub fn socket_path(&self) -> &Path {
        self.daemon.socket_path()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub fn connect_timeout(&self) -> Duration {
        self.daemon.connect_timeout()
    }

    pub fn request_timeout(&self) -> Duration {
        self.daemon.request_timeout()
    }

    pub async fn status_at(
        socket_path: impl Into<PathBuf>,
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<DaemonStatus, DaemonClientError> {
        DaemonClient::status_at(socket_path, connect_timeout, request_timeout).await
    }

    pub fn validate(&self) -> Result<(), DaemonClientError> {
        Err(DaemonClientError)
    }

    pub async fn status(&self) -> Result<DaemonStatus, DaemonClientError> {
        Err(DaemonClientError)
    }

    pub async fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, DaemonClientError> {
        if input.is_empty() {
            Ok(Vec::new())
        } else {
            Err(DaemonClientError)
        }
    }

    pub async fn rerank(
        &self,
        _query: &str,
        documents: &[String],
    ) -> Result<Vec<f64>, DaemonClientError> {
        if documents.is_empty() {
            Ok(Vec::new())
        } else {
            Err(DaemonClientError)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DaemonClientError;

impl DaemonClientError {
    pub fn is_retryable(&self) -> bool {
        false
    }
}

impl fmt::Display for DaemonClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(UNSUPPORTED)
    }
}

impl std::error::Error for DaemonClientError {}
