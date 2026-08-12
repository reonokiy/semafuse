//! Shared LoCoMo benchmark configuration, method resolution, and artifact types.

use crate::{
    adapters::executable::resolve_executable_in,
    benchmark::{BenchmarkMessage as RunnerMessage, BenchmarkQuestion as RunnerQuestion},
    benchmark_baselines::Message as BaselineMessage,
    config::{self as semafuse_config, Config},
    locomo::{LocomoTrack, PreparedDataset, PreparedQuestion},
};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};
use walkdir::WalkDir;

/// Fixed evidence metrics for one cutoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMetricsAtCutoff {
    /// Maximum number of ranked source messages inspected.
    pub cutoff: usize,
    /// Number of gold source messages found within the cutoff.
    pub hits: usize,
    /// Hits divided by the fixed cutoff.
    pub precision: f64,
    /// One when at least one gold source message was found.
    pub recall_any: f64,
    /// One when every gold source message was found.
    pub recall_all: f64,
    /// Fraction of gold source messages found.
    pub mean_recall: f64,
    /// Reciprocal rank of the first gold source message.
    pub mrr: f64,
    /// Conventional binary-relevance nDCG.
    pub ndcg: f64,
}

/// One label-aware evaluator record. No question, answer, or message text is
/// persisted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoQuestionRecord {
    /// Output schema version.
    pub schema_version: u32,
    /// Stable identity for the complete suite execution.
    pub run_key: String,
    /// Stable baseline, Semafuse stage, or external method identifier.
    pub method: String,
    /// Opaque question identifier supplied by the label-isolated adapter.
    pub question_id: String,
    /// Stable paper category label.
    pub category: String,
    /// Numeric official category in `1..=4`.
    pub category_number: u8,
    /// Number of gold evidence messages for this question.
    pub gold_count: usize,
    /// Ranked opaque source-message identifiers retained for rescoring.
    pub ranked_message_ids: Vec<String>,
    /// Evidence metrics at exactly 1, 5, 10, and 50.
    pub evidence: Vec<EvidenceMetricsAtCutoff>,
    /// Wall-clock measurements with explicit shared-execution semantics.
    pub efficiency: LocomoEfficiency,
    /// Secret-free method metadata.
    pub metadata: Map<String, Value>,
}

/// Per-record runtime measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoEfficiency {
    /// Complete execution wall time for this baseline call, production call,
    /// or external search.
    pub query_wall_s: f64,
    /// Retrieval time reported by Semafuse, when available.
    pub retrieval_s: Option<f64>,
    /// Once-per-conversation history construction shared by its queries.
    pub shared_construction_s: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SuiteConfig {
    pub(crate) schema_version: u32,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) track: LocomoTrack,
    pub(crate) baseline_method: String,
    pub(crate) methods: Vec<MethodConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum MethodConfig {
    Bm25 {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Recent {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Semafuse {
        id: String,
        config: PathBuf,
        #[serde(default = "default_limit")]
        limit: usize,
        stages: Vec<SemafuseStage>,
    },
    ExternalJsonl {
        id: String,
        worker_config: PathBuf,
        command: Vec<String>,
        #[serde(default)]
        working_directory: Option<PathBuf>,
        #[serde(default)]
        python_path: Option<PathBuf>,
        #[serde(default)]
        source_root: Option<PathBuf>,
        #[serde(default)]
        dependency_lock: Option<PathBuf>,
        #[serde(default = "default_limit")]
        top_k: usize,
        #[serde(default = "default_context_tokens")]
        max_context_tokens: usize,
        #[serde(default = "default_timeout_seconds")]
        timeout_seconds: u64,
    },
}

impl MethodConfig {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Bm25 { id, .. }
            | Self::Recent { id, .. }
            | Self::Semafuse { id, .. }
            | Self::ExternalJsonl { id, .. } => id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SemafuseStage {
    Dense,
    Bm25,
    Rrf,
    Retrieved,
}

impl SemafuseStage {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Bm25 => "bm25",
            Self::Rrf => "rrf",
            Self::Retrieved => "retrieved",
        }
    }
}

fn default_limit() -> usize {
    50
}

fn default_context_tokens() -> usize {
    16_384
}

fn default_timeout_seconds() -> u64 {
    600
}

#[derive(Debug)]
pub(crate) enum ResolvedMethod {
    Bm25 {
        id: String,
        limit: usize,
    },
    Recent {
        id: String,
        limit: usize,
    },
    Semafuse {
        id: String,
        config: Config,
        config_sha256: String,
        resolved_config_sha256: String,
        embedding_model: Option<String>,
        reranker_model: Option<String>,
        limit: usize,
        stages: Vec<SemafuseStage>,
    },
    ExternalJsonl(ResolvedExternalMethod),
}

impl ResolvedMethod {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Bm25 { id, .. } | Self::Recent { id, .. } | Self::Semafuse { id, .. } => id,
            Self::ExternalJsonl(method) => &method.id,
        }
    }

    pub(crate) fn output_ids(&self) -> Vec<String> {
        match self {
            Self::Semafuse { id, stages, .. } => stages
                .iter()
                .map(|stage| format!("{id}::{}", stage.as_str()))
                .collect(),
            _ => vec![self.id().to_owned()],
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedExternalMethod {
    pub(crate) id: String,
    pub(crate) worker_config: PathBuf,
    pub(crate) worker_config_sha256: String,
    pub(crate) adapter_kind: String,
    pub(crate) track: String,
    pub(crate) upstream_revision: Option<String>,
    pub(crate) model_revisions: BTreeMap<String, String>,
    pub(crate) reader_tokenizer_model: Option<String>,
    pub(crate) reader_tokenizer_revision: Option<String>,
    pub(crate) command: Vec<String>,
    pub(crate) command_sha256: String,
    pub(crate) executable_invocation: PathBuf,
    pub(crate) executable_sha256: String,
    pub(crate) working_directory: Option<PathBuf>,
    pub(crate) python_path: Option<PathBuf>,
    pub(crate) python_tree_sha256: Option<String>,
    pub(crate) dependency_lock_sha256: Option<String>,
    pub(crate) python_environment_sha256: Option<String>,
    pub(crate) dependency_pins_verified: bool,
    pub(crate) top_k: usize,
    pub(crate) max_context_tokens: usize,
    pub(crate) timeout: Duration,
}

/// Validates method-local invariants without imposing a comparison cohort or
/// requiring the baseline to be one of these methods. Answer suites use this
/// when additional outputs come from pinned memory-context journals.
pub(crate) fn validate_method_configs(methods: &[MethodConfig]) -> Result<()> {
    let mut output_ids = HashSet::new();
    for method in methods {
        ensure!(
            !method.id().trim().is_empty() && !method.id().contains("::"),
            "method IDs must be non-empty and must not contain ::"
        );
        match method {
            MethodConfig::Bm25 { id, limit } | MethodConfig::Recent { id, limit } => {
                ensure!(
                    *limit >= 50,
                    "method {id} must return at least 50 items for fixed @50 scoring"
                );
                ensure!(output_ids.insert(id.clone()), "duplicate output id {id}");
            }
            MethodConfig::Semafuse {
                id, limit, stages, ..
            } => {
                ensure!(
                    *limit >= 50,
                    "method {id} must return at least 50 items for fixed @50 scoring"
                );
                ensure!(output_ids.insert(id.clone()), "duplicate output id {id}");
                let unique = stages.iter().copied().collect::<BTreeSet<_>>();
                ensure!(
                    unique.len() == stages.len(),
                    "method {id} has duplicate stages"
                );
                for stage in stages {
                    let output = format!("{id}::{}", stage.as_str());
                    ensure!(
                        output_ids.insert(output.clone()),
                        "duplicate output id {output}"
                    );
                }
            }
            MethodConfig::ExternalJsonl {
                id,
                command,
                source_root,
                dependency_lock,
                top_k,
                max_context_tokens,
                timeout_seconds,
                ..
            } => {
                ensure!(
                    source_root.is_some() == dependency_lock.is_some(),
                    "external method {id} must configure source_root and dependency_lock together"
                );
                ensure!(
                    *top_k >= 50,
                    "external method {id} must request at least 50 native items for fixed @50 scoring"
                );
                ensure!(
                    *top_k <= 1_000,
                    "external method {id} top_k exceeds protocol maximum 1000"
                );
                ensure!(
                    (1..=10_000_000).contains(max_context_tokens),
                    "external method {id} max_context_tokens is outside protocol limits"
                );
                ensure!(
                    *timeout_seconds > 0,
                    "external method {id} timeout must be positive"
                );
                ensure!(
                    !command.is_empty() && command.iter().all(|part| !part.trim().is_empty()),
                    "external method {id} command must contain non-empty arguments"
                );
                ensure!(output_ids.insert(id.clone()), "duplicate output id {id}");
            }
        }
    }
    Ok(())
}

pub(crate) fn resolve_methods(
    suite: &SuiteConfig,
    suite_dir: &Path,
) -> Result<Vec<ResolvedMethod>> {
    suite
        .methods
        .iter()
        .map(|method| match method {
            MethodConfig::Bm25 { id, limit } => Ok(ResolvedMethod::Bm25 {
                id: id.clone(),
                limit: *limit,
            }),
            MethodConfig::Recent { id, limit } => Ok(ResolvedMethod::Recent {
                id: id.clone(),
                limit: *limit,
            }),
            MethodConfig::Semafuse {
                id,
                config,
                limit,
                stages,
            } => {
                let path = resolve_path(suite_dir, config)?;
                let config_sha256 = sha256_file(&path)?;
                let config = semafuse_config::load_from_path(&path)?;
                let recall_strategy = config.strategies.recall.as_str();
                ensure!(
                    config
                        .strategies
                        .enabled
                        .iter()
                        .any(|name| name == recall_strategy),
                    "LoCoMo Semafuse method {id} requires configured recall strategy {recall_strategy} to be enabled"
                );
                let resolved_config_sha256 =
                    hex::encode(Sha256::digest(format!("{config:?}").as_bytes()));
                let embedding_model =
                    recall_strategy_setting_string(&config, &["embedding", "endpoint", "model"]);
                let reranker_model =
                    recall_strategy_setting_string(&config, &["reranker", "endpoint", "model"]);
                Ok(ResolvedMethod::Semafuse {
                    id: id.clone(),
                    config,
                    config_sha256,
                    resolved_config_sha256,
                    embedding_model,
                    reranker_model,
                    limit: *limit,
                    stages: stages.clone(),
                })
            }
            MethodConfig::ExternalJsonl {
                id,
                worker_config,
                command,
                working_directory,
                python_path,
                source_root,
                dependency_lock,
                top_k,
                max_context_tokens,
                timeout_seconds,
            } => {
                let worker_config = resolve_path(suite_dir, worker_config)?;
                let worker_config_sha256 = sha256_file(&worker_config)?;
                let identity = external_config_identity(&worker_config)?;
                let working_directory = working_directory
                    .as_ref()
                    .map(|path| resolve_path(suite_dir, path))
                    .transpose()?;
                let python_path = python_path
                    .as_ref()
                    .map(|path| resolve_path(suite_dir, path))
                    .transpose()?;
                let command_directory = match &working_directory {
                    Some(path) => path.clone(),
                    None => env::current_dir().context("resolve current command directory")?,
                };
                let program = command
                    .first()
                    .with_context(|| format!("external method {id} has no command"))?;
                let search_path = env::var_os("PATH");
                let executable =
                    resolve_executable(program, &command_directory, search_path.as_deref())?;
                let executable_sha256 = sha256_file(&executable.identity_target)?;
                let (
                    python_tree_sha256,
                    dependency_lock_sha256,
                    python_environment_sha256,
                    dependency_pins_verified,
                ) =
                    match (source_root, dependency_lock) {
                        (Some(source_root), Some(dependency_lock)) => {
                            let source_root = resolve_source_root(suite_dir, source_root)?;
                            let dependency_lock =
                                resolve_dependency_lock(suite_dir, dependency_lock)?;
                            let environment = if identity.track == "contract-test" {
                                None
                            } else {
                                Some(
                                    inspect_python_environment(
                                        &executable.invocation_path,
                                        working_directory.as_deref(),
                                    )
                                    .with_context(|| {
                                        format!(
                                            "inspect Python environment for external method {id}"
                                        )
                                    })?,
                                )
                            };
                            let dependency_pins_verified =
                                if is_strict_python_dependency_lock(&dependency_lock) {
                                    let environment = environment.as_ref().with_context(|| {
                                        format!(
                                            "external method {id} cannot verify a strict dependency lock on the contract-test track"
                                        )
                                    })?;
                                    validate_dependency_pins(
                                        &dependency_lock,
                                        &environment.distributions,
                                    )
                                    .with_context(|| {
                                        format!(
                                            "verify Python dependencies for external method {id}"
                                        )
                                    })?;
                                    true
                                } else {
                                    false
                                };
                            (
                                Some(python_tree_sha256(&source_root)?),
                                Some(sha256_file(&dependency_lock)?),
                                environment
                                    .map(|environment| environment.python_environment_sha256),
                                dependency_pins_verified,
                            )
                        }
                        (None, None) => {
                            ensure!(
                                identity.track == "contract-test",
                                "external method {id} must configure source_root and dependency_lock outside contract-test"
                            );
                            (None, None, None, false)
                        }
                        _ => bail!(
                            "external method {id} must configure source_root and dependency_lock together"
                        ),
                    };
                let command_sha256 = hash_strings(command);
                Ok(ResolvedMethod::ExternalJsonl(ResolvedExternalMethod {
                    id: id.clone(),
                    worker_config,
                    worker_config_sha256,
                    adapter_kind: identity.adapter_kind,
                    track: identity.track,
                    upstream_revision: identity.upstream_revision,
                    model_revisions: identity.model_revisions,
                    reader_tokenizer_model: identity.reader_tokenizer_model,
                    reader_tokenizer_revision: identity.reader_tokenizer_revision,
                    command: command.clone(),
                    command_sha256,
                    executable_invocation: executable.invocation_path,
                    executable_sha256,
                    working_directory,
                    python_path,
                    python_tree_sha256,
                    dependency_lock_sha256,
                    python_environment_sha256,
                    dependency_pins_verified,
                    top_k: *top_k,
                    max_context_tokens: *max_context_tokens,
                    timeout: Duration::from_secs(*timeout_seconds),
                }))
            }
        })
        .collect()
}

fn recall_strategy_setting_string(config: &Config, path: &[&str]) -> Option<String> {
    let mut value = config
        .strategies
        .settings
        .get(config.strategies.recall.as_str())?;
    for component in path {
        value = value.get(*component)?;
    }
    value.as_str().map(str::to_owned)
}

struct ExternalConfigIdentity {
    adapter_kind: String,
    track: String,
    upstream_revision: Option<String>,
    model_revisions: BTreeMap<String, String>,
    reader_tokenizer_model: Option<String>,
    reader_tokenizer_revision: Option<String>,
}

fn external_config_identity(path: &Path) -> Result<ExternalConfigIdentity> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read worker config {}", path.display()))?,
    )
    .with_context(|| format!("parse worker config {}", path.display()))?;
    let adapter_kind = value
        .pointer("/adapter/kind")
        .and_then(Value::as_str)
        .context("external worker config has no adapter.kind string")?;
    let track = value
        .pointer("/run_policy/track")
        .and_then(Value::as_str)
        .context("external worker config has no run_policy.track string")?;
    ensure!(
        !adapter_kind.trim().is_empty() && !track.trim().is_empty(),
        "external worker adapter kind and track must be non-empty"
    );
    let upstream_revision = value
        .pointer("/run_policy/upstream/revision")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let model_revisions = value
        .pointer("/run_policy/models")
        .and_then(Value::as_object)
        .map(|models| {
            models
                .iter()
                .filter_map(|(role, lock)| {
                    lock.get("revision")
                        .and_then(Value::as_str)
                        .map(|revision| (role.clone(), revision.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default();
    let reader_tokenizer_model = value
        .pointer("/tokenizer/model")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let reader_tokenizer_revision = value
        .pointer("/tokenizer/revision")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(ExternalConfigIdentity {
        adapter_kind: adapter_kind.to_owned(),
        track: track.to_owned(),
        upstream_revision,
        model_revisions,
        reader_tokenizer_model,
        reader_tokenizer_revision,
    })
}

fn resolve_path(base: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = configured_path(base, path);
    candidate
        .canonicalize()
        .with_context(|| format!("resolve configured path {}", candidate.display()))
}

fn configured_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedExecutable {
    pub(crate) invocation_path: PathBuf,
    pub(crate) identity_target: PathBuf,
}

pub(crate) fn resolve_executable(
    program: &str,
    relative_base: &Path,
    search_path: Option<&std::ffi::OsStr>,
) -> Result<ResolvedExecutable> {
    resolve_executable_for_platform(program, relative_base, search_path, cfg!(windows))
}

fn resolve_executable_for_platform(
    program: &str,
    relative_base: &Path,
    search_path: Option<&std::ffi::OsStr>,
    windows: bool,
) -> Result<ResolvedExecutable> {
    ensure!(!program.trim().is_empty(), "external command[0] is empty");
    // LoCoMo hashes and later launches this path directly. Windows command
    // scripts require a cmd.exe wrapper, so they are deliberately excluded
    // from this native-worker contract.
    let path_ext = windows.then_some(std::ffi::OsStr::new(".EXE"));
    let relative_base = if relative_base.is_absolute() {
        relative_base.to_path_buf()
    } else {
        env::current_dir()
            .context("resolve current directory for external executable")?
            .join(relative_base)
    };
    let program_path = Path::new(program);
    let candidate = if program_path.is_absolute() || program_path.components().count() > 1 {
        let configured = configured_path(&relative_base, program_path);
        resolve_executable_in(&configured.to_string_lossy(), None, path_ext, windows).with_context(
            || format!("external executable {} was not found", configured.display()),
        )?
    } else {
        let search_path = search_path
            .with_context(|| format!("PATH is not set while resolving executable {program}"))?;
        let normalized_search_path =
            env::join_paths(env::split_paths(search_path).map(|directory| {
                if directory.is_absolute() {
                    directory
                } else {
                    relative_base.join(directory)
                }
            }))
            .context("normalize PATH while resolving external executable")?;
        resolve_executable_in(program, Some(&normalized_search_path), path_ext, windows)
            .with_context(|| format!("executable {program} was not found on PATH"))?
    };
    ensure!(
        !windows
            || candidate
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|extension| extension.eq_ignore_ascii_case("exe")),
        "external executable must be a native .exe file on Windows"
    );
    ensure!(
        is_executable_file(&candidate, windows),
        "external executable must be an executable regular file"
    );
    let identity_target = candidate
        .canonicalize()
        .with_context(|| format!("resolve external executable {}", candidate.display()))?;
    ensure!(
        is_executable_file(&identity_target, windows),
        "external executable must be an executable regular file"
    );
    Ok(ResolvedExecutable {
        invocation_path: candidate,
        identity_target,
    })
}

fn is_executable_file(path: &Path, windows: bool) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    if windows {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(test)]
mod executable_resolution_tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn windows_resolution_selects_native_exe_from_relative_search_paths() {
        let directory = tempdir().unwrap();
        let bin = directory.path().join("bin");
        fs::create_dir(&bin).unwrap();
        fs::write(bin.join("python.CMD"), b"command script").unwrap();
        let executable = bin.join("PyThOn.ExE");
        fs::write(&executable, b"test executable").unwrap();
        let search_path = env::join_paths([Path::new("bin")]).unwrap();

        let resolved =
            resolve_executable_for_platform("python", directory.path(), Some(&search_path), true)
                .unwrap();

        assert_eq!(resolved.invocation_path, executable);
        assert_eq!(resolved.identity_target, executable.canonicalize().unwrap());
    }

    #[test]
    fn windows_resolution_rejects_an_explicit_command_script() {
        let directory = tempdir().unwrap();
        let script = directory.path().join("python.CMD");
        fs::write(&script, b"command script").unwrap();

        let error =
            resolve_executable_for_platform(script.to_str().unwrap(), directory.path(), None, true)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("must be a native .exe file on Windows")
        );
    }
}

fn resolve_source_root(base: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = configured_path(base, path);
    let metadata = fs::symlink_metadata(&candidate)
        .with_context(|| format!("inspect Python source root {}", candidate.display()))?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "Python source root must not be a symlink"
    );
    ensure!(metadata.is_dir(), "Python source root must be a directory");
    candidate
        .canonicalize()
        .with_context(|| format!("resolve Python source root {}", candidate.display()))
}

pub(crate) fn resolve_dependency_lock(base: &Path, path: &Path) -> Result<PathBuf> {
    let candidate = configured_path(base, path);
    let metadata = fs::symlink_metadata(&candidate)
        .with_context(|| format!("inspect dependency lock {}", candidate.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_file(),
        "dependency lock must be a regular non-symlink file"
    );
    candidate
        .canonicalize()
        .with_context(|| format!("resolve dependency lock {}", candidate.display()))
}

#[derive(Debug)]
pub(crate) struct PythonEnvironmentIdentity {
    pub(crate) distributions: BTreeMap<String, String>,
    pub(crate) python_environment_sha256: String,
}

const PYTHON_DISTRIBUTION_INVENTORY_SCRIPT: &str = r#"
import importlib.metadata
import json
import sys

rows = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not isinstance(name, str) or not isinstance(version, str):
        raise RuntimeError("installed distribution has no string Name or Version metadata")
    rows.append([name, version])
sys.stdout.write(json.dumps(rows, ensure_ascii=True, separators=(",", ":")))
"#;

pub(crate) fn inspect_python_environment(
    invocation_interpreter: &Path,
    working_directory: Option<&Path>,
) -> Result<PythonEnvironmentIdentity> {
    // Use the invocation path rather than its canonical target: for venv
    // interpreters, Python discovers pyvenv.cfg from the invoked path. Isolated
    // mode keeps this metadata-only preflight independent of PYTHONPATH and the
    // benchmark source tree while retaining the venv's site-packages.
    let mut command = std::process::Command::new(invocation_interpreter);
    command.args(["-I", "-c", PYTHON_DISTRIBUTION_INVENTORY_SCRIPT]);
    if let Some(directory) = working_directory {
        command.current_dir(directory);
    }
    let output = command.output().with_context(|| {
        format!(
            "run Python distribution preflight with {}",
            invocation_interpreter.display()
        )
    })?;
    ensure!(
        output.status.success(),
        "Python distribution preflight failed with status {}: {}",
        output.status,
        bounded_process_error(&output.stderr)
    );
    let rows: Vec<(String, String)> = serde_json::from_slice(&output.stdout)
        .context("Python distribution preflight returned invalid JSON")?;
    let mut distributions = BTreeMap::new();
    for (raw_name, version) in rows {
        let name = normalize_python_distribution_name(&raw_name)?;
        ensure!(
            !version.is_empty()
                && version.trim() == version
                && !version.chars().any(char::is_control),
            "installed Python distribution {name} has an invalid version"
        );
        if let Some(existing) = distributions.insert(name.clone(), version.clone()) {
            ensure!(
                existing == version,
                "installed Python distribution {name} has conflicting versions {existing} and {version}"
            );
        }
    }
    ensure!(
        !distributions.is_empty(),
        "Python distribution preflight found no installed distributions"
    );

    Ok(PythonEnvironmentIdentity {
        python_environment_sha256: installed_distributions_sha256(&distributions),
        distributions,
    })
}

pub(crate) fn installed_distributions_sha256(distributions: &BTreeMap<String, String>) -> String {
    let mut digest = Sha256::new();
    digest.update(b"semafuse-python-installed-distributions/v1");
    for (name, version) in distributions {
        digest.update((name.len() as u64).to_be_bytes());
        digest.update(name.as_bytes());
        digest.update((version.len() as u64).to_be_bytes());
        digest.update(version.as_bytes());
    }
    hex::encode(digest.finalize())
}

fn bounded_process_error(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .chars()
        .take(2_000)
        .collect::<String>()
        .trim()
        .to_owned()
}

fn normalize_python_distribution_name(raw: &str) -> Result<String> {
    ensure!(
        !raw.is_empty() && raw.trim() == raw,
        "Python distribution name must be non-empty without surrounding whitespace"
    );
    ensure!(
        raw.is_ascii()
            && raw
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
        "invalid Python distribution name {raw:?}"
    );
    ensure!(
        raw.as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
            && raw.as_bytes().last().is_some_and(u8::is_ascii_alphanumeric),
        "invalid Python distribution name {raw:?}"
    );

    let mut normalized = String::with_capacity(raw.len());
    let mut previous_separator = false;
    for byte in raw.bytes() {
        if matches!(byte, b'-' | b'_' | b'.') {
            if !previous_separator {
                normalized.push('-');
                previous_separator = true;
            }
        } else {
            normalized.push((byte as char).to_ascii_lowercase());
            previous_separator = false;
        }
    }
    Ok(normalized)
}

pub(crate) fn is_strict_python_dependency_lock(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name == "requirements.lock.txt")
}

pub(crate) fn validate_dependency_pins(
    dependency_lock: &Path,
    installed: &BTreeMap<String, String>,
) -> Result<()> {
    let contents = fs::read_to_string(dependency_lock)
        .with_context(|| format!("read dependency lock {}", dependency_lock.display()))?;
    let mut pins = BTreeMap::new();
    for (index, raw_line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(url) = line.strip_prefix("--extra-index-url=") {
            ensure!(
                !url.is_empty() && !url.chars().any(char::is_whitespace),
                "dependency lock line {line_number} has an invalid --extra-index-url"
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("--extra-index-url") {
            ensure!(
                rest.chars().next().is_some_and(char::is_whitespace),
                "dependency lock line {line_number} has an unrecognized requirement option"
            );
            let mut fields = rest.split_whitespace();
            let url = fields.next().unwrap_or_default();
            ensure!(
                !url.is_empty() && fields.next().is_none(),
                "dependency lock line {line_number} has an invalid --extra-index-url"
            );
            continue;
        }

        let (raw_name, version) = line.split_once("==").with_context(|| {
            format!("dependency lock line {line_number} is not an exact name==version pin")
        })?;
        ensure!(
            !version.is_empty()
                && !version.contains("==")
                && !version.chars().any(char::is_whitespace)
                && !version.chars().any(char::is_control),
            "dependency lock line {line_number} has an invalid pinned version"
        );
        let name = normalize_python_distribution_name(raw_name).with_context(|| {
            format!("dependency lock line {line_number} has an invalid package name")
        })?;
        ensure!(
            pins.insert(name.clone(), version.to_owned()).is_none(),
            "dependency lock contains duplicate pin for {name}"
        );
    }
    ensure!(
        !pins.is_empty(),
        "strict dependency lock contains no exact pins"
    );
    for (name, expected) in pins {
        let actual = installed
            .get(&name)
            .with_context(|| format!("pinned Python distribution {name} is not installed"))?;
        ensure!(
            actual == &expected,
            "pinned Python distribution {name} has version {actual}, expected {expected}"
        );
    }
    Ok(())
}

pub(crate) fn python_tree_sha256(root: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("inspect Python source tree {}", root.display()))?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        "Python source tree root must be a regular directory"
    );

    let mut sources = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.with_context(|| format!("walk Python source tree {}", root.display()))?;
        if entry.path() == root {
            continue;
        }
        ensure!(
            !entry.file_type().is_symlink(),
            "Python source tree contains a symlink"
        );
        if entry.file_type().is_dir() {
            continue;
        }
        ensure!(
            entry.file_type().is_file(),
            "Python source tree contains a non-regular file"
        );
        if entry.path().extension().and_then(|value| value.to_str()) != Some("py") {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .context("Python source escaped its configured root")?;
        let relative = relative
            .to_str()
            .context("Python source tree contains a non-UTF-8 path")?
            .replace(std::path::MAIN_SEPARATOR, "/");
        sources.push((relative, entry.into_path()));
    }
    ensure!(
        !sources.is_empty(),
        "Python source tree contains no .py files"
    );
    sources.sort_by(|left, right| left.0.cmp(&right.0));

    let mut digest = Sha256::new();
    digest.update(b"semafuse-external-memory-python-tree/v1");
    for (relative, path) in sources {
        let bytes = fs::read(&path)
            .with_context(|| format!("read Python source {} for hashing", path.display()))?;
        digest.update((relative.len() as u64).to_be_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    Ok(hex::encode(digest.finalize()))
}

pub(crate) fn validate_track_dataset(dataset: &PreparedDataset, track: LocomoTrack) -> Result<()> {
    ensure!(
        dataset.summary.conversation_count == track.conversation_ids().len(),
        "LoCoMo {} conversation denominator mismatch",
        track.name()
    );
    ensure!(
        dataset.summary.qa_count == track.qa_count(),
        "LoCoMo {} QA denominator mismatch",
        track.name()
    );
    ensure!(
        dataset.summary.retrieval_count == track.retrieval_count(),
        "LoCoMo {} retrieval denominator mismatch",
        track.name()
    );
    Ok(())
}

pub(crate) fn projected_corpus(question: &PreparedQuestion) -> Vec<BaselineMessage> {
    question
        .input
        .messages
        .iter()
        .map(|message| BaselineMessage {
            id: message.id.clone(),
            role: message.role.clone(),
            content: message.content.clone(),
            session_id: message.session_id.clone(),
            corpus_index: message.index,
        })
        .collect()
}

pub(crate) fn runner_question(question: &PreparedQuestion) -> RunnerQuestion {
    RunnerQuestion {
        id: question.input.id.clone(),
        dataset: question.input.dataset.clone(),
        question_type: String::new(),
        query: question.input.query.clone(),
        query_time: question.input.query_time.clone(),
        messages: question
            .input
            .messages
            .iter()
            .map(|message| RunnerMessage {
                id: message.id.clone(),
                index: Some(message.index),
                role: message.role.clone(),
                content: message.content.clone(),
                timestamp: message.timestamp.clone(),
                session_id: message.session_id.clone(),
            })
            .collect(),
    }
}

pub(crate) fn method_manifest(methods: &[ResolvedMethod]) -> Vec<Value> {
    let mut manifest = Vec::new();
    for method in methods {
        match method {
            ResolvedMethod::Bm25 { id, limit } => manifest.push(json!({
                "id": id,
                "kind": "bm25",
                "limit": limit,
            })),
            ResolvedMethod::Recent { id, limit } => manifest.push(json!({
                "id": id,
                "kind": "recent",
                "limit": limit,
            })),
            ResolvedMethod::Semafuse {
                id,
                config_sha256,
                resolved_config_sha256,
                embedding_model,
                reranker_model,
                limit,
                stages,
                ..
            } => manifest.push(json!({
                "id": id,
                "kind": "semafuse_production_retrieval",
                "config_sha256": config_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "embedding_model": embedding_model,
                "reranker_model": reranker_model,
                "model_revisions_verified": false,
                "limit": limit,
                "stages": stages,
                "outputs": method.output_ids(),
                "shared_execution": true,
                "state_scope": "one_ingest_and_index_per_source_conversation",
                "query_order": "sequential",
            })),
            ResolvedMethod::ExternalJsonl(external) => manifest.push(json!({
                "id": external.id,
                "kind": "external_jsonl",
                "adapter_kind": external.adapter_kind,
                "track": external.track,
                "upstream_revision": external.upstream_revision,
                "model_revisions": external.model_revisions,
                "reader_tokenizer_model": external.reader_tokenizer_model,
                "reader_tokenizer_revision": external.reader_tokenizer_revision,
                "protected_identity_complete": external.track != "development"
                    && external.upstream_revision.is_some()
                    && !external.model_revisions.is_empty()
                    && external.python_tree_sha256.is_some()
                    && external.dependency_lock_sha256.is_some()
                    && external.python_environment_sha256.is_some()
                    && external.dependency_pins_verified,
                "worker_config_sha256": external.worker_config_sha256,
                "command_sha256": external.command_sha256,
                "executable_sha256": external.executable_sha256,
                "python_tree_sha256": external.python_tree_sha256,
                "dependency_lock_sha256": external.dependency_lock_sha256,
                "python_environment_sha256": external.python_environment_sha256,
                "dependency_pins_verified": external.dependency_pins_verified,
                "top_k": external.top_k,
                "max_context_tokens": external.max_context_tokens,
                "timeout_seconds": external.timeout.as_secs(),
                "state_scope": "one_reset_and_construction_per_source_conversation",
            })),
        }
    }
    manifest.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    manifest
}

pub(crate) fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {} for hashing", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn hash_strings(values: &[String]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update((value.len() as u64).to_be_bytes());
        digest.update(value.as_bytes());
    }
    hex::encode(digest.finalize())
}
