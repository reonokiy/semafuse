use std::{io, path::PathBuf};

use thiserror::Error;

use crate::config::ConfigLoadError;

pub type Result<T, E = DaemonError> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum DaemonError {
    #[error(transparent)]
    ConfigLoad(#[from] ConfigLoadError),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("socket path is already owned by a running daemon: {0}")]
    AlreadyRunning(PathBuf),

    #[error("refusing to replace an unsafe Unix socket path: {0}")]
    UnsafeSocketPath(PathBuf),

    #[error("socket directory must be a private mode-0700 directory owned by this uid: {0}")]
    UnsafeSocketDirectory(PathBuf),

    #[error("I/O error for {context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: io::Error,
    },

    #[error("service registration failed: {0}")]
    ServiceRegistration(String),

    #[error("signal handler error: {0}")]
    Signal(#[source] io::Error),
}

impl DaemonError {
    pub(crate) fn io(context: &'static str, source: io::Error) -> Self {
        Self::Io { context, source }
    }
}
