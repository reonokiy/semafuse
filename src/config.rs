//! Configuration paths and layered application settings.
//!
//! [`crate::config::load`] starts with built-in defaults, overlays the optional TOML file, and
//! finally applies `SEMAFUSE__...` environment variables. Environment values
//! therefore have highest precedence.

use anyhow::{Result, ensure};
use config::{Environment, File};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

/// Returns the data directory, honoring `SEMAFUSE_DATA_DIR`.
pub fn default_data_dir() -> PathBuf {
    if let Ok(path) = std::env::var("SEMAFUSE_DATA_DIR") {
        return PathBuf::from(path);
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("semafuse")
}

/// Returns the database path, honoring `SEMAFUSE_DB`.
pub fn default_db_path() -> PathBuf {
    std::env::var("SEMAFUSE_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_data_dir().join("semafuse.db"))
}

/// Returns the hook spool directory, honoring `SEMAFUSE_SPOOL_DIR`.
pub fn default_spool_dir() -> PathBuf {
    std::env::var("SEMAFUSE_SPOOL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_data_dir().join("spool"))
}

/// Returns the attachment directory, honoring `SEMAFUSE_BLOB_DIR`.
pub fn default_blob_dir() -> PathBuf {
    std::env::var("SEMAFUSE_BLOB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_data_dir().join("blobs"))
}

/// Returns the TOML path, honoring `SEMAFUSE_CONFIG`.
pub fn default_config_path() -> PathBuf {
    std::env::var("SEMAFUSE_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_data_dir().join("config.toml"))
}

/// Selection of derived-memory strategies.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyConfig {
    /// Strategies that receive committed domain events.
    #[serde(default = "default_enabled")]
    pub enabled: Vec<String>,
    /// Enabled strategy through which recall requests are routed.
    #[serde(default = "default_recall")]
    pub recall: String,
    /// Strategy-owned configuration keyed by registered strategy name.
    ///
    /// Keeping these values opaque at the application layer lets each strategy
    /// module deserialize and validate its own settings without growing a
    /// central configuration enum whenever a module is added.
    #[serde(default)]
    pub settings: HashMap<String, Value>,
}

/// Complete Semafuse application configuration.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    /// Derived-memory strategy selection.
    #[serde(default)]
    pub strategies: StrategyConfig,
    /// Attachment backend configuration.
    #[serde(default)]
    pub blob: BlobConfig,
}

/// OpenDAL attachment backend configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct BlobConfig {
    /// OpenDAL service scheme, defaulting to `fs`.
    #[serde(default = "default_blob_scheme")]
    pub scheme: String,
    /// Service-specific OpenDAL options.
    #[serde(default)]
    pub config: HashMap<String, String>,
}

impl Default for BlobConfig {
    fn default() -> Self {
        Self {
            scheme: default_blob_scheme(),
            config: HashMap::new(),
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            recall: default_recall(),
            settings: HashMap::new(),
        }
    }
}

fn default_enabled() -> Vec<String> {
    vec!["history".into()]
}
fn default_recall() -> String {
    "history".into()
}

fn default_blob_scheme() -> String {
    "fs".into()
}

/// Loads layered configuration and validates that the recall strategy is enabled.
///
/// For the default `fs` blob scheme, an absent `root` is filled from
/// [`default_blob_dir`]. Strategy implementation names are validated separately
/// by [`crate::strategy::validate`].
pub fn load() -> Result<Config> {
    let path = default_config_path();
    load_with_path(&path, false)
}

/// Loads configuration from an explicit TOML file, then applies `SEMAFUSE__...`
/// environment overrides.
///
/// Unlike [`load`], this function requires the supplied file to exist. It is
/// intended for isolated tools, such as benchmark runners, that must never
/// silently fall back to a user's ambient default configuration.
pub fn load_from_path(path: impl AsRef<Path>) -> Result<Config> {
    let path = path.as_ref();
    ensure!(
        path.is_file(),
        "configuration file does not exist or is not a regular file: {}",
        path.display()
    );
    load_with_path(path, true)
}

fn load_with_path(path: &Path, required: bool) -> Result<Config> {
    let mut builder = config::Config::builder()
        .set_default("strategies.enabled", default_enabled())?
        .set_default("strategies.recall", default_recall())?;
    if path.exists() {
        builder = builder.add_source(File::from(path.to_path_buf()));
    } else {
        ensure!(
            !required,
            "configuration file does not exist: {}",
            path.display()
        );
    }
    let mut config: Config = builder
        .add_source(
            Environment::with_prefix("SEMAFUSE")
                .prefix_separator("__")
                .separator("__")
                .try_parsing(true)
                .list_separator(",")
                .with_list_parse_key("strategies.enabled"),
        )
        .build()?
        .try_deserialize()?;
    if config.blob.scheme == "fs" && !config.blob.config.contains_key("root") {
        config.blob.config.insert(
            "root".into(),
            default_blob_dir().to_string_lossy().into_owned(),
        );
    }
    ensure!(
        config
            .strategies
            .enabled
            .iter()
            .any(|name| name == &config.strategies.recall),
        "recall strategy {} is not enabled",
        config.strategies.recall
    );
    Ok(config)
}
