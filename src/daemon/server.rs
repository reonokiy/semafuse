use std::{
    fs,
    future::Future,
    io,
    os::unix::{
        fs::{FileTypeExt, MetadataExt, PermissionsExt},
        net::UnixStream as StdUnixStream,
    },
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use crate::daemon_ipc::protocol::ResponseEnvelope;
use rustix::process::geteuid;
use tokio::{
    net::{UnixListener, UnixStream},
    sync::Semaphore,
    task::JoinSet,
    time::{Instant, timeout},
};
use tracing::{debug, error, warn};

use crate::daemon::{
    config::Config,
    error::{DaemonError, Result},
    ipc::{read_request, write_response},
    models,
    router::ServiceRouter,
    services::ModelsService,
};

const CONNECTION_IO_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Server {
    config: Config,
    listener: UnixListener,
    router: Arc<ServiceRouter>,
    socket_guard: SocketGuard,
}

impl Server {
    pub async fn from_config(config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|error| DaemonError::Config(error.to_string()))?;

        let mut router = ServiceRouter::new(env!("CARGO_PKG_VERSION"));
        if let Some(model_config) = &config.services.models {
            let engines = models::load_engines(model_config).map_err(DaemonError::Config)?;
            let service = ModelsService::new(engines, model_config.max_concurrent_inferences)
                .map_err(DaemonError::ServiceRegistration)?;
            router.register(Arc::new(service))?;
        }
        Self::with_router(config, router).await
    }

    /// Bind a server around an explicitly constructed registry. This is useful
    /// for tests and for adding future local services without coupling them to
    /// process or socket lifecycle code.
    pub async fn with_router(config: Config, router: ServiceRouter) -> Result<Self> {
        config
            .validate()
            .map_err(|error| DaemonError::Config(error.to_string()))?;
        let (listener, socket_guard) = bind_socket(&config.daemon.socket_path)?;
        Ok(Self {
            config,
            listener,
            router: Arc::new(router),
            socket_guard,
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_guard.path
    }

    pub async fn run_until_shutdown_signal(self) -> Result<()> {
        self.run_until(shutdown_signal()).await
    }

    pub async fn run_until<F>(self, shutdown: F) -> Result<()>
    where
        F: Future<Output = Result<()>>,
    {
        let Server {
            config,
            listener,
            router,
            socket_guard,
        } = self;
        let permits = Arc::new(Semaphore::new(config.daemon.max_connections));
        let mut tasks = JoinSet::new();
        tokio::pin!(shutdown);

        loop {
            let permit = tokio::select! {
                signal = &mut shutdown => {
                    signal?;
                    break;
                }
                permit = permits.clone().acquire_owned() => {
                    permit.map_err(|_| DaemonError::Config("connection limiter closed".to_owned()))?
                }
            };

            let accepted = tokio::select! {
                signal = &mut shutdown => {
                    signal?;
                    drop(permit);
                    break;
                }
                accepted = listener.accept() => accepted,
            };
            match accepted {
                Ok((stream, _address)) => {
                    let router = router.clone();
                    tasks.spawn(async move {
                        let _permit = permit;
                        handle_connection(stream, router).await;
                    });
                }
                Err(error) if is_transient_accept_error(&error) => {
                    warn!(%error, "transient Unix socket accept error");
                    drop(permit);
                }
                Err(source) => {
                    return Err(DaemonError::io("accepting Unix socket connection", source));
                }
            }
            while let Some(result) = tasks.try_join_next() {
                if let Err(error) = result {
                    warn!(%error, "connection task failed");
                }
            }
        }

        drop(listener);
        let grace = Duration::from_millis(config.daemon.shutdown_grace_ms);
        let deadline = Instant::now() + grace;
        while !tasks.is_empty() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match timeout(remaining, tasks.join_next()).await {
                Ok(Some(Err(error))) => warn!(%error, "connection task failed during shutdown"),
                Ok(Some(Ok(()))) => {}
                Ok(None) => break,
                Err(_) => break,
            }
        }
        if !tasks.is_empty() {
            warn!(
                remaining = tasks.len(),
                "aborting connections after shutdown grace period"
            );
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
        }
        drop(socket_guard);
        Ok(())
    }
}

async fn handle_connection(mut stream: UnixStream, router: Arc<ServiceRouter>) {
    match stream.peer_cred() {
        Ok(credentials) if credentials.uid() != geteuid().as_raw() => {
            warn!(
                peer_uid = credentials.uid(),
                "rejected Unix socket peer with a different uid"
            );
            return;
        }
        Err(error) => {
            warn!(%error, "could not authenticate Unix socket peer");
            return;
        }
        Ok(_) => {}
    }

    let incoming = match timeout(CONNECTION_IO_TIMEOUT, read_request(&mut stream)).await {
        Err(_) => {
            warn!("connection timed out while reading a request");
            return;
        }
        Ok(Ok(Some(incoming))) => incoming,
        Ok(Ok(None)) => return,
        Ok(Err(error)) => {
            if let Some(request_id) = error.request_id() {
                let response = ResponseEnvelope::Error {
                    error: error.to_protocol_error(),
                };
                if let Err(write_error) = write_response(&mut stream, request_id, &response).await {
                    debug!(%write_error, "failed to return malformed-request response");
                }
            } else {
                debug!(%error, "closed connection with an unreadable frame header");
            }
            return;
        }
    };

    let request_id = incoming.request_id;
    let response = router.route(incoming.request).await;
    match timeout(
        CONNECTION_IO_TIMEOUT,
        write_response(&mut stream, request_id, &response),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(write_error)) => debug!(%write_error, "failed to write response"),
        Err(_) => warn!(request_id, "connection timed out while writing a response"),
    }
}

fn bind_socket(path: &Path) -> Result<(UnixListener, SocketGuard)> {
    prepare_socket_directory(path)?;
    remove_stale_socket(path)?;

    let listener = UnixListener::bind(path)
        .map_err(|source| DaemonError::io("binding Unix socket", source))?;
    if let Err(source) = fs::set_permissions(path, fs::Permissions::from_mode(0o600)) {
        let _ = fs::remove_file(path);
        return Err(DaemonError::io("setting Unix socket permissions", source));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|source| DaemonError::io("inspecting bound Unix socket", source))?;
    let guard = SocketGuard {
        path: path.to_path_buf(),
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    Ok((listener, guard))
}

fn prepare_socket_directory(socket_path: &Path) -> Result<()> {
    let parent = socket_path.parent().ok_or_else(|| {
        DaemonError::Config("daemon.socket_path must have a parent directory".to_owned())
    })?;
    match fs::symlink_metadata(parent) {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(parent)
                .map_err(|source| DaemonError::io("creating socket directory", source))?;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
                .map_err(|source| DaemonError::io("securing socket directory", source))?;
        }
        Err(source) => {
            return Err(DaemonError::io("inspecting socket directory", source));
        }
    }

    let metadata = fs::symlink_metadata(parent)
        .map_err(|source| DaemonError::io("inspecting socket directory", source))?;
    let owner_matches = metadata.uid() == geteuid().as_raw();
    let private_mode = metadata.mode() & 0o077 == 0;
    let owner_can_access = metadata.mode() & 0o700 == 0o700;
    if !metadata.file_type().is_dir() || !owner_matches || !private_mode || !owner_can_access {
        return Err(DaemonError::UnsafeSocketDirectory(parent.to_path_buf()));
    }
    Ok(())
}

fn remove_stale_socket(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(DaemonError::io("inspecting Unix socket path", source)),
    };
    if !is_private_owned_socket(&metadata) {
        return Err(DaemonError::UnsafeSocketPath(path.to_path_buf()));
    }

    match StdUnixStream::connect(path) {
        Ok(_) => Err(DaemonError::AlreadyRunning(path.to_path_buf())),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
            ) =>
        {
            remove_socket_if_unchanged(path, &metadata)
        }
        Err(source) => Err(DaemonError::io("probing existing Unix socket", source)),
    }
}

fn remove_socket_if_unchanged(path: &Path, original: &fs::Metadata) -> Result<()> {
    let current = fs::symlink_metadata(path)
        .map_err(|source| DaemonError::io("revalidating stale Unix socket", source))?;
    if current.dev() != original.dev()
        || current.ino() != original.ino()
        || !is_private_owned_socket(&current)
    {
        return Err(DaemonError::UnsafeSocketPath(path.to_path_buf()));
    }
    fs::remove_file(path).map_err(|source| DaemonError::io("removing stale Unix socket", source))
}

fn is_private_owned_socket(metadata: &fs::Metadata) -> bool {
    metadata.file_type().is_socket()
        && metadata.uid() == geteuid().as_raw()
        && metadata.mode() & 0o777 == 0o600
}

fn is_transient_accept_error(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::Interrupted
    )
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map_err(DaemonError::Signal)?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.map_err(DaemonError::Signal),
            _ = terminate.recv() => Ok(()),
        }
    }
}

struct SocketGuard {
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let Ok(metadata) = fs::symlink_metadata(&self.path) else {
            return;
        };
        if metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
            && let Err(error) = fs::remove_file(&self.path)
        {
            error!(path = %self.path.display(), %error, "failed to remove daemon socket");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener as StdUnixListener;

    use tempfile::TempDir;

    use super::*;

    fn test_config(temp: &TempDir) -> Config {
        let socket_dir = temp.path().join("runtime");
        fs::create_dir(&socket_dir).unwrap();
        fs::set_permissions(&socket_dir, fs::Permissions::from_mode(0o700)).unwrap();
        let mut config = Config::default();
        config.daemon.socket_path = socket_dir.join("semafused.sock");
        config
    }

    #[tokio::test]
    async fn socket_is_private_and_removed_on_drop() {
        let temp = TempDir::new().unwrap();
        let config = test_config(&temp);
        let path = config.daemon.socket_path.clone();
        let server = Server::with_router(config, ServiceRouter::new("test"))
            .await
            .unwrap();
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        drop(server);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_live_socket_is_not_replaced() {
        let temp = TempDir::new().unwrap();
        let config = test_config(&temp);
        let first = Server::with_router(config.clone(), ServiceRouter::new("first"))
            .await
            .unwrap();
        let second = Server::with_router(config, ServiceRouter::new("second")).await;
        assert!(matches!(second, Err(DaemonError::AlreadyRunning(_))));
        drop(first);
    }

    #[test]
    fn replacement_between_probe_and_remove_is_preserved() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("semafused.sock");
        let replacement_path = temp.path().join("replacement.sock");

        let original_listener = StdUnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let original = fs::symlink_metadata(&path).unwrap();

        fs::remove_file(&path).unwrap();
        let replacement_listener = StdUnixListener::bind(&replacement_path).unwrap();
        fs::set_permissions(&replacement_path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::rename(&replacement_path, &path).unwrap();

        let result = remove_socket_if_unchanged(&path, &original);
        assert!(matches!(result, Err(DaemonError::UnsafeSocketPath(_))));
        assert!(path.exists());

        drop(replacement_listener);
        drop(original_listener);
    }
}
