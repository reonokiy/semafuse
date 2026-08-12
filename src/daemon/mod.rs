//! `semafuse daemon serve` is the foreground, long-lived process mode for local
//! Semafuse services.
//!
//! It intentionally contains no daemonization or process-management commands.
//! A supervisor owns the process lifecycle; clients communicate over a local
//! Unix-domain socket.
#![deny(unsafe_code)]

pub mod config;
pub mod error;
pub mod ipc;
pub mod models;
pub mod router;
pub mod server;
pub mod services;

pub use config::Config;
pub use error::{DaemonError, Result};
pub use router::ServiceRouter;
pub use server::Server;
