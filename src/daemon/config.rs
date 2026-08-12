use serde::Deserialize;
use std::{
    error::Error,
    fmt, fs, io,
    os::unix::ffi::OsStrExt,
    path::{Component, Path, PathBuf},
};

pub const DEFAULT_MAX_CONNECTIONS: usize = 64;
pub const DEFAULT_SHUTDOWN_GRACE_MS: u64 = 5_000;
pub const DEFAULT_MAX_CONCURRENT_INFERENCES: usize = 1;
pub const DEFAULT_DEVICE_ORDINAL: usize = 0;
pub const DEFAULT_EMBEDDING_MAX_LENGTH: usize = 2_048;
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 1;
pub const DEFAULT_RERANKER_MAX_LENGTH: usize = 1_024;
pub const DEFAULT_RERANKER_BATCH_SIZE: usize = 1;

pub const MAX_CONNECTIONS_LIMIT: usize = 4_096;
pub const MAX_SHUTDOWN_GRACE_MS_LIMIT: u64 = 300_000;
pub const MAX_CONCURRENT_INFERENCES_LIMIT: usize = 64;
pub const MAX_DEVICE_ORDINAL: usize = 15;
pub const MIN_MODEL_LENGTH: usize = 128;
pub const MAX_MODEL_LENGTH_LIMIT: usize = 32_768;
pub const MAX_MODEL_BATCH_SIZE_LIMIT: usize = 64;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub const MAX_SOCKET_PATH_BYTES: usize = 107;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub const MAX_SOCKET_PATH_BYTES: usize = 103;

/// Strict top-level configuration for `semafuse daemon serve`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub services: ServicesConfig,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigLoadError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| ConfigLoadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config =
            toml::from_str::<Self>(&contents).map_err(|source| ConfigLoadError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        config.validate().map_err(ConfigLoadError::Invalid)?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.daemon.validate()?;
        self.services.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonConfig {
    #[serde(default = "default_socket_path")]
    pub socket_path: PathBuf,
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_shutdown_grace_ms")]
    pub shutdown_grace_ms: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            shutdown_grace_ms: DEFAULT_SHUTDOWN_GRACE_MS,
        }
    }
}

impl DaemonConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        validate_socket_path(&self.socket_path)?;
        validate_limit(
            self.max_connections,
            1,
            MAX_CONNECTIONS_LIMIT,
            "daemon.max_connections",
        )?;
        validate_limit(
            self.shutdown_grace_ms,
            1,
            MAX_SHUTDOWN_GRACE_MS_LIMIT,
            "daemon.shutdown_grace_ms",
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServicesConfig {
    #[serde(default)]
    pub models: Option<ModelsConfig>,
}

impl ServicesConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.models.as_ref().map_or(Ok(()), ModelsConfig::validate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    #[serde(default = "default_max_concurrent_inferences")]
    pub max_concurrent_inferences: usize,
    #[serde(default)]
    pub device: ModelDevice,
    #[serde(default = "default_device_ordinal")]
    pub device_ordinal: usize,
    #[serde(default)]
    pub embedding: Option<EmbeddingModelConfig>,
    #[serde(default)]
    pub reranker: Option<RerankerModelConfig>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            max_concurrent_inferences: DEFAULT_MAX_CONCURRENT_INFERENCES,
            device: ModelDevice::Auto,
            device_ordinal: DEFAULT_DEVICE_ORDINAL,
            embedding: None,
            reranker: None,
        }
    }
}

impl ModelsConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        validate_limit(
            self.max_concurrent_inferences,
            1,
            MAX_CONCURRENT_INFERENCES_LIMIT,
            "services.models.max_concurrent_inferences",
        )?;
        validate_limit(
            self.device_ordinal,
            0,
            MAX_DEVICE_ORDINAL,
            "services.models.device_ordinal",
        )?;
        if self.device == ModelDevice::Cpu && self.device_ordinal != 0 {
            return Err(ConfigError::new(
                "services.models.device_ordinal",
                "must be zero when device is `cpu`",
            ));
        }
        if self.embedding.is_none() && self.reranker.is_none() {
            return Err(ConfigError::new(
                "services.models",
                "must configure at least one of embedding or reranker",
            ));
        }
        if let Some(embedding) = &self.embedding {
            embedding.validate()?;
        }
        if let Some(reranker) = &self.reranker {
            reranker.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelDevice {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingModelConfig {
    pub model_dir: PathBuf,
    #[serde(default = "default_embedding_max_length")]
    pub max_length: usize,
    #[serde(default = "default_embedding_batch_size")]
    pub batch_size: usize,
}

impl EmbeddingModelConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            max_length: DEFAULT_EMBEDDING_MAX_LENGTH,
            batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        validate_model(
            &self.model_dir,
            self.max_length,
            self.batch_size,
            "services.models.embedding.model_dir",
            "services.models.embedding.max_length",
            "services.models.embedding.batch_size",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankerModelConfig {
    pub model_dir: PathBuf,
    #[serde(default = "default_reranker_max_length")]
    pub max_length: usize,
    #[serde(default = "default_reranker_batch_size")]
    pub batch_size: usize,
}

impl RerankerModelConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            max_length: DEFAULT_RERANKER_MAX_LENGTH,
            batch_size: DEFAULT_RERANKER_BATCH_SIZE,
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        validate_model(
            &self.model_dir,
            self.max_length,
            self.batch_size,
            "services.models.reranker.model_dir",
            "services.models.reranker.max_length",
            "services.models.reranker.batch_size",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    field: &'static str,
    message: String,
}

impl ConfigError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }

    pub fn field(&self) -> &'static str {
        self.field
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.message)
    }
}

impl Error for ConfigError {}

#[derive(Debug)]
pub enum ConfigLoadError {
    Read {
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    Invalid(ConfigError),
}

impl fmt::Display for ConfigLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, .. } => {
                write!(formatter, "failed to read config {}", path.display())
            }
            Self::Parse { path, .. } => {
                write!(formatter, "failed to parse config {}", path.display())
            }
            Self::Invalid(error) => error.fmt(formatter),
        }
    }
}

impl Error for ConfigLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            Self::Invalid(error) => Some(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_model(
    model_dir: &Path,
    max_length: usize,
    batch_size: usize,
    model_dir_field: &'static str,
    max_length_field: &'static str,
    batch_size_field: &'static str,
) -> Result<(), ConfigError> {
    validate_absolute_path(model_dir, model_dir_field)?;
    validate_limit(
        max_length,
        MIN_MODEL_LENGTH,
        MAX_MODEL_LENGTH_LIMIT,
        max_length_field,
    )?;
    validate_limit(batch_size, 1, MAX_MODEL_BATCH_SIZE_LIMIT, batch_size_field)
}

fn validate_socket_path(path: &Path) -> Result<(), ConfigError> {
    const FIELD: &str = "daemon.socket_path";

    validate_absolute_path(path, FIELD)?;
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() > MAX_SOCKET_PATH_BYTES || bytes.contains(&0) {
        return Err(ConfigError::new(
            FIELD,
            format!("must contain 1..={MAX_SOCKET_PATH_BYTES} non-NUL bytes"),
        ));
    }
    Ok(())
}

fn validate_absolute_path(path: &Path, field: &'static str) -> Result<(), ConfigError> {
    if path.as_os_str().is_empty() {
        return Err(ConfigError::new(field, "must not be empty"));
    }
    if !path.is_absolute() {
        return Err(ConfigError::new(field, "must be an absolute path"));
    }
    if path.parent().is_none() {
        return Err(ConfigError::new(field, "must not be a filesystem root"));
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ConfigError::new(field, "must not contain `..` components"));
    }
    Ok(())
}

fn validate_limit<T>(
    value: T,
    minimum: T,
    maximum: T,
    field: &'static str,
) -> Result<(), ConfigError>
where
    T: Copy + Ord + fmt::Display,
{
    if value < minimum || value > maximum {
        return Err(ConfigError::new(
            field,
            format!("must be between {minimum} and {maximum}, got {value}"),
        ));
    }
    Ok(())
}

fn default_socket_path() -> PathBuf {
    if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
    {
        return runtime_dir.join("semafuse").join("semafused.sock");
    }
    PathBuf::from(format!(
        "/tmp/semafuse-{}/semafused.sock",
        rustix::process::geteuid().as_raw()
    ))
}

const fn default_max_connections() -> usize {
    DEFAULT_MAX_CONNECTIONS
}
const fn default_shutdown_grace_ms() -> u64 {
    DEFAULT_SHUTDOWN_GRACE_MS
}
const fn default_max_concurrent_inferences() -> usize {
    DEFAULT_MAX_CONCURRENT_INFERENCES
}
const fn default_device_ordinal() -> usize {
    DEFAULT_DEVICE_ORDINAL
}
const fn default_embedding_max_length() -> usize {
    DEFAULT_EMBEDDING_MAX_LENGTH
}
const fn default_embedding_batch_size() -> usize {
    DEFAULT_EMBEDDING_BATCH_SIZE
}
const fn default_reranker_max_length() -> usize {
    DEFAULT_RERANKER_MAX_LENGTH
}
const fn default_reranker_batch_size() -> usize {
    DEFAULT_RERANKER_BATCH_SIZE
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};

    use super::*;

    #[test]
    fn defaults_are_safe_and_valid() {
        let config = Config::default();
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(DEFAULT_EMBEDDING_MAX_LENGTH, 2_048);
        assert_eq!(DEFAULT_EMBEDDING_BATCH_SIZE, 1);
        assert_eq!(DEFAULT_RERANKER_BATCH_SIZE, 1);
        assert_eq!(
            ModelsConfig::default().max_concurrent_inferences,
            DEFAULT_MAX_CONCURRENT_INFERENCES
        );
        assert_eq!(ModelsConfig::default().device, ModelDevice::Auto);
        assert_eq!(ModelsConfig::default().device_ordinal, 0);
    }

    #[test]
    fn unknown_and_removed_backend_fields_are_rejected() {
        assert!(serde_json::from_str::<Config>(r#"{"unexpected":true}"#).is_err());
        assert!(
            serde_json::from_str::<Config>(
                r#"{"services":{"models":{"embedding":{"model_dir":"/models/qwen","backend":"ort"}}}}"#
            )
            .is_err()
        );
    }

    #[test]
    fn socket_paths_match_the_client_platform_limit() {
        let mut config = Config::default();
        config.daemon.socket_path =
            PathBuf::from(format!("/{}", "a".repeat(MAX_SOCKET_PATH_BYTES)));
        assert_eq!(config.validate().unwrap_err().field(), "daemon.socket_path");

        let mut bytes = b"/tmp/semafuse".to_vec();
        bytes.push(0);
        bytes.extend_from_slice(b"socket");
        config.daemon.socket_path = PathBuf::from(OsString::from_vec(bytes));
        assert_eq!(config.validate().unwrap_err().field(), "daemon.socket_path");
    }

    #[test]
    fn invalid_model_paths_and_lengths_are_rejected() {
        let mut embedding = EmbeddingModelConfig::new("relative/model");
        let mut config = Config::default();
        config.services.models = Some(ModelsConfig {
            embedding: Some(embedding.clone()),
            ..ModelsConfig::default()
        });
        assert_eq!(
            config.validate().unwrap_err().field(),
            "services.models.embedding.model_dir"
        );

        embedding.model_dir = "/models/qwen".into();
        embedding.max_length = MIN_MODEL_LENGTH - 1;
        config.services.models = Some(ModelsConfig {
            embedding: Some(embedding),
            ..ModelsConfig::default()
        });
        assert_eq!(
            config.validate().unwrap_err().field(),
            "services.models.embedding.max_length"
        );
    }

    #[test]
    fn empty_models_and_excessive_limits_are_rejected() {
        let mut config = Config::default();
        config.services.models = Some(ModelsConfig::default());
        assert_eq!(config.validate().unwrap_err().field(), "services.models");

        config.services.models = None;
        config.daemon.max_connections = MAX_CONNECTIONS_LIMIT + 1;
        assert_eq!(
            config.validate().unwrap_err().field(),
            "daemon.max_connections"
        );
    }

    #[test]
    fn valid_models_configuration_passes_validation() {
        let mut config = Config::default();
        config.services.models = Some(ModelsConfig {
            embedding: Some(EmbeddingModelConfig::new("/models/qwen3-embedding")),
            reranker: Some(RerankerModelConfig::new("/models/bge-reranker")),
            ..ModelsConfig::default()
        });
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn model_device_configuration_is_strict_and_bounded() {
        let parsed: ModelsConfig = serde_json::from_str(
            r#"{"device":"metal","device_ordinal":1,"embedding":{"model_dir":"/models/qwen"}}"#,
        )
        .unwrap();
        assert_eq!(parsed.device, ModelDevice::Metal);
        assert_eq!(parsed.device_ordinal, 1);

        assert!(
            serde_json::from_str::<ModelsConfig>(
                r#"{"device":"gpu","embedding":{"model_dir":"/models/qwen"}}"#
            )
            .is_err()
        );

        let mut config = Config::default();
        config.services.models = Some(ModelsConfig {
            device: ModelDevice::Cpu,
            device_ordinal: 1,
            embedding: Some(EmbeddingModelConfig::new("/models/qwen")),
            ..ModelsConfig::default()
        });
        assert_eq!(
            config.validate().unwrap_err().field(),
            "services.models.device_ordinal"
        );
    }
}
