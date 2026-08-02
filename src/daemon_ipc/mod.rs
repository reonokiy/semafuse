#[cfg(unix)]
mod client;
pub mod protocol;
#[cfg(not(unix))]
mod unsupported_client;

#[cfg(unix)]
pub use client::{DaemonClient, DaemonClientError, ModelServiceClient};
pub use protocol::{DaemonState, DaemonStatus, ModelIdentity, ModelServiceStatus, ServiceState};
#[cfg(not(unix))]
pub use unsupported_client::{DaemonClient, DaemonClientError, ModelServiceClient};
