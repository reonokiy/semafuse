//! Retrieval-only LoCoMo paper-v2 comparison suite.
//!
//! This runner fixes the LazyMem paper-v2 selection (two conversations, 314
//! category-1--4 QA items, and 312 questions with evidence) while keeping all
//! answer and evidence annotations outside memory-system inputs. It scores
//! deterministic BM25/recent controls, the production Semafuse retrieval
//! stages, and process-isolated external memory workers through one evaluator.
//!
//! LoCoMo is CC BY-NC 4.0. Runs fail closed without an explicit
//! non-commercial-use acknowledgement and accept only the pinned official data
//! file hash. The dataset itself is never copied into benchmark artifacts.

use crate::{
    benchmark::{
        BenchmarkMessage as RunnerMessage, BenchmarkQuestion as RunnerQuestion, BenchmarkResponse,
        StrategyRetrievalSession,
    },
    benchmark_baselines::{BaselineHit, Message as BaselineMessage, bm25, recent},
    benchmark_journal::{
        JournalRecord, JournalRecordKey, JsonlJournal, SummaryEnvelope, write_summary_atomic,
    },
    benchmark_metrics::{RankingMetricsAtCutoff, evaluate_ranking},
    benchmark_report::{
        MetricDirection, ScalarBenchmarkReport, ScalarMetricRow, build_scalar_report,
    },
    config::{self as semafuse_config, Config},
    locomo::{
        DATA_LICENSE, LocomoMode, PAPER_V2_CONVERSATION_IDS, PAPER_V2_QA_COUNT,
        PAPER_V2_RETRIEVAL_COUNT, PreparedDataset, PreparedQuestion, UPSTREAM_COMMIT,
        UPSTREAM_DATA_SHA256, load_locomo,
    },
};
use anyhow::{Context, Result, bail, ensure};
use config::{Config as LayeredConfig, File};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    time::timeout,
};
use walkdir::WalkDir;

const SUITE_SCHEMA_VERSION: u32 = 1;
const RECORD_SCHEMA_VERSION: u32 = 1;
const EXTERNAL_PROTOCOL: &str = "semafuse.external-memory.v1";
const EXTERNAL_MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
const RECORD_KIND: &str = "locomo_retrieval";
const EVIDENCE_CUTOFFS: [usize; 4] = [1, 5, 10, 50];

/// Filesystem and license inputs for one LoCoMo retrieval run.
#[derive(Debug, Clone)]
pub struct LocomoSuiteRunOptions {
    /// Modular suite TOML path.
    pub suite_path: PathBuf,
    /// Official pinned `data/locomo10.json` path.
    pub dataset_path: PathBuf,
    /// Explicit acknowledgement that this is permitted non-commercial use.
    pub acknowledge_noncommercial: bool,
    /// Optional crash-safe output directory.
    pub output_dir: Option<PathBuf>,
}

/// Complete in-memory artifacts returned by [`run_locomo_retrieval_suite`].
#[derive(Debug, Serialize)]
pub struct LocomoSuiteArtifacts {
    /// Secret-free run identity and protocol manifest.
    pub manifest: Value,
    /// One evaluator record per method and retrieval-eligible question.
    pub records: Vec<LocomoQuestionRecord>,
    /// Paired aggregate and per-category reports.
    pub summary: Value,
}

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
struct SuiteConfig {
    schema_version: u32,
    name: String,
    baseline_method: String,
    methods: Vec<MethodConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum MethodConfig {
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
    fn id(&self) -> &str {
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
enum SemafuseStage {
    Dense,
    Bm25,
    Rrf,
    Retrieved,
}

impl SemafuseStage {
    const fn as_str(self) -> &'static str {
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
enum ResolvedMethod {
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
    fn id(&self) -> &str {
        match self {
            Self::Bm25 { id, .. } | Self::Recent { id, .. } | Self::Semafuse { id, .. } => id,
            Self::ExternalJsonl(method) => &method.id,
        }
    }

    fn output_ids(&self) -> Vec<String> {
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
struct ResolvedExternalMethod {
    id: String,
    worker_config: PathBuf,
    worker_config_sha256: String,
    adapter_kind: String,
    track: String,
    upstream_revision: Option<String>,
    model_revisions: BTreeMap<String, String>,
    reader_tokenizer_model: Option<String>,
    reader_tokenizer_revision: Option<String>,
    command: Vec<String>,
    command_sha256: String,
    executable_invocation: PathBuf,
    executable_sha256: String,
    working_directory: Option<PathBuf>,
    python_path: Option<PathBuf>,
    python_tree_sha256: Option<String>,
    dependency_lock_sha256: Option<String>,
    python_environment_sha256: Option<String>,
    dependency_pins_verified: bool,
    top_k: usize,
    max_context_tokens: usize,
    timeout: Duration,
}

#[derive(Debug)]
struct MethodOutput {
    method: String,
    ranking: Vec<String>,
    efficiency: LocomoEfficiency,
    metadata: Map<String, Value>,
}

/// Executes the pinned LoCoMo paper-v2 retrieval protocol.
pub async fn run_locomo_retrieval_suite(
    options: &LocomoSuiteRunOptions,
) -> Result<LocomoSuiteArtifacts> {
    ensure!(
        options.acknowledge_noncommercial,
        "LoCoMo is CC BY-NC 4.0; pass --acknowledge-noncommercial only after confirming this run is permitted non-commercial use"
    );
    let suite_bytes = fs::read(&options.suite_path)
        .with_context(|| format!("read suite {}", options.suite_path.display()))?;
    let suite = load_suite(&options.suite_path)?;
    validate_suite(&suite)?;
    let suite_dir = options
        .suite_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let methods = resolve_methods(&suite, suite_dir)?;

    let dataset_sha256 = sha256_file(&options.dataset_path)?;
    ensure!(
        dataset_sha256 == UPSTREAM_DATA_SHA256,
        "LoCoMo paper-v2 requires pinned locomo10.json SHA-256 {UPSTREAM_DATA_SHA256}; got {dataset_sha256}"
    );
    let dataset = load_locomo(
        &options.dataset_path,
        options.acknowledge_noncommercial,
        &LocomoMode::PaperV2,
    )?;
    validate_paper_dataset(&dataset)?;

    let cargo_lock_sha256 = cargo_lock_sha256();
    let code_sha256 = benchmark_code_sha256();
    let run_key = run_key(
        &suite_bytes,
        &dataset_sha256,
        &methods,
        &cargo_lock_sha256,
        &code_sha256,
    );
    let mut journal = options
        .output_dir
        .as_ref()
        .map(|directory| {
            JsonlJournal::<LocomoQuestionRecord>::open(
                directory.join("per-question.jsonl"),
                run_key.clone(),
            )
        })
        .transpose()?;
    let mut records = journal
        .as_ref()
        .map(|journal| {
            journal
                .records()
                .iter()
                .map(|record| record.payload.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    for question in dataset
        .questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
    {
        let corpus = projected_corpus(question);
        for method in methods.iter().filter(|method| {
            matches!(
                method,
                ResolvedMethod::Bm25 { .. } | ResolvedMethod::Recent { .. }
            )
        }) {
            let expected = method.output_ids();
            if expected
                .iter()
                .all(|method_id| journal_has(&journal, &run_key, method_id, &question.input.id))
            {
                continue;
            }
            let outputs = execute_native_method(method, question, &corpus)
                .await
                .with_context(|| {
                    format!(
                        "execute method {} on opaque question {}",
                        method.id(),
                        question.input.id
                    )
                })?;
            persist_outputs(
                &mut journal,
                &mut records,
                &run_key,
                question,
                &corpus,
                outputs,
            )?;
        }
    }

    for method in methods
        .iter()
        .filter(|method| matches!(method, ResolvedMethod::Semafuse { .. }))
    {
        execute_semafuse_method(method, &dataset, &run_key, &mut journal, &mut records)
            .await
            .with_context(|| format!("execute persistent Semafuse method {}", method.id()))?;
    }

    for method in methods.iter().filter_map(|method| match method {
        ResolvedMethod::ExternalJsonl(method) => Some(method),
        _ => None,
    }) {
        execute_external_method(method, &dataset, &run_key, &mut journal, &mut records)
            .await
            .with_context(|| format!("execute external method {}", method.id))?;
    }

    validate_record_completeness(&records, &dataset, &methods, &run_key)?;
    let summary = build_locomo_summary_from_records(&suite.baseline_method, &records)?;
    let method_manifest = method_manifest(&methods);
    let mut limitations = vec![
        "retrieval-only evidence metrics are not LoCoMo answer F1 and must not be compared with published answer scores",
        "stateful Semafuse and external memory methods construct once per source conversation and serve that conversation's missing queries sequentially under the same lifecycle",
        "a resumed run reconstructs only conversations with missing records; construction timing therefore describes the attempt that produced each record and is not additive across records",
    ];
    if methods
        .iter()
        .any(|method| matches!(method, ResolvedMethod::Semafuse { .. }))
    {
        limitations.push(
            "Semafuse dense, bm25, rrf, and retrieved rows share one production execution per question",
        );
    }
    let manifest = json!({
        "schema_version": 1,
        "run_key": run_key,
        "suite": suite.name,
        "track": "paper-v2-selection/retrieval-only-common-evaluator",
        "paper_comparable": false,
        "benchmark": {
            "name": "LoCoMo",
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_sha256": dataset_sha256,
            "selection": {
                "conversation_ids": PAPER_V2_CONVERSATION_IDS,
                "categories": [1, 2, 3, 4],
                "qa_n": PAPER_V2_QA_COUNT,
                "retrieval_eval_n": PAPER_V2_RETRIEVAL_COUNT,
            },
        },
        "license": {
            "spdx": DATA_LICENSE,
            "noncommercial_use_acknowledged": true,
            "dataset_redistributed": false,
            "dataset_path_serialized": false,
        },
        "protocol": {
            "history_roles": "all",
            "evidence_unit": "source_message",
            "cutoffs": EVIDENCE_CUTOFFS,
            "external_source_projection": "ordered_unique_source_ids_flattened_from_native_item_rank_then_capped_at_50",
            "query_concurrency": 1,
            "memory_lifecycle": "one_reset_ingest_and_index_per_source_conversation_then_sequential_queries",
            "construction_accounting": "shared_construction_s is the once-per-conversation construction attempt repeated on that conversation's query records; divide by metadata.construction_amortization_n for an amortized per-query value and never sum the repeated field",
            "answer_generation": false,
            "official_token_f1": {
                "value": null,
                "status": "not_applicable_without_answer_predictions",
                "separate_evaluator": "src/locomo_eval.rs",
            },
        },
        "denominators": {
            "qa_n": dataset.summary.qa_count,
            "retrieval_eval_n": dataset.summary.retrieval_count,
            "qa_by_category": qa_category_counts(&dataset),
            "retrieval_by_category": eligible_category_counts(&dataset),
        },
        "baseline_method": suite.baseline_method,
        "methods": method_manifest,
        "implementation": {
            "crate_version": env!("CARGO_PKG_VERSION"),
            "cargo_lock_sha256": cargo_lock_sha256,
            "benchmark_code_sha256": code_sha256,
        },
        "limitations": limitations,
        "credential_handling": "worker and model configs are represented only by hashes and non-secret identities; environment values are never serialized",
    });
    let artifacts = LocomoSuiteArtifacts {
        manifest,
        records,
        summary,
    };
    if let Some(output_dir) = &options.output_dir {
        write_summary_atomic(
            output_dir.join("summary.json"),
            &SummaryEnvelope {
                run_key: run_key.clone(),
                summary: json!({
                    "manifest": &artifacts.manifest,
                    "summary": &artifacts.summary,
                }),
            },
        )?;
    }
    Ok(artifacts)
}

fn load_suite(path: &Path) -> Result<SuiteConfig> {
    LayeredConfig::builder()
        .add_source(File::from(path.to_path_buf()).required(true))
        .build()
        .with_context(|| format!("parse LoCoMo suite TOML {}", path.display()))?
        .try_deserialize()
        .with_context(|| format!("deserialize LoCoMo suite TOML {}", path.display()))
}

fn validate_suite(suite: &SuiteConfig) -> Result<()> {
    ensure!(
        suite.schema_version == SUITE_SCHEMA_VERSION,
        "unsupported LoCoMo suite schema_version {}",
        suite.schema_version
    );
    ensure!(
        !suite.name.trim().is_empty(),
        "suite name must be non-empty"
    );
    ensure!(
        !suite.baseline_method.trim().is_empty(),
        "baseline_method must be non-empty"
    );
    ensure!(
        suite.methods.len() >= 2,
        "LoCoMo comparison requires at least two configured methods"
    );
    let mut output_ids = HashSet::new();
    let mut direct_baselines = HashSet::new();
    for method in &suite.methods {
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
                direct_baselines.insert(id.clone());
            }
            MethodConfig::Semafuse {
                id, limit, stages, ..
            } => {
                ensure!(
                    *limit >= 50,
                    "method {id} must return at least 50 items for fixed @50 scoring"
                );
                ensure!(!stages.is_empty(), "method {id} must expose stages");
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
    ensure!(
        direct_baselines.contains(&suite.baseline_method),
        "baseline_method must name a configured BM25 or recent control"
    );
    Ok(())
}

fn resolve_methods(suite: &SuiteConfig, suite_dir: &Path) -> Result<Vec<ResolvedMethod>> {
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
                ensure!(
                    config.strategies.recall == "lazymem"
                        && config
                            .strategies
                            .enabled
                            .iter()
                            .any(|name| name == "lazymem"),
                    "LoCoMo Semafuse method {id} requires enabled recall strategy lazymem"
                );
                let resolved_config_sha256 =
                    hex::encode(Sha256::digest(format!("{config:?}").as_bytes()));
                let embedding_model =
                    lazymem_setting_string(&config, &["embedding", "endpoint", "model"]);
                let reranker_model =
                    lazymem_setting_string(&config, &["reranker", "endpoint", "model"]);
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

fn lazymem_setting_string(config: &Config, path: &[&str]) -> Option<String> {
    let mut value = config.strategies.settings.get("lazymem")?;
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

struct ResolvedExecutable {
    invocation_path: PathBuf,
    identity_target: PathBuf,
}

fn resolve_executable(
    program: &str,
    relative_base: &Path,
    search_path: Option<&std::ffi::OsStr>,
) -> Result<ResolvedExecutable> {
    ensure!(!program.trim().is_empty(), "external command[0] is empty");
    let relative_base = if relative_base.is_absolute() {
        relative_base.to_path_buf()
    } else {
        env::current_dir()
            .context("resolve current directory for external executable")?
            .join(relative_base)
    };
    let program_path = Path::new(program);
    let candidate = if program_path.is_absolute() || program_path.components().count() > 1 {
        configured_path(&relative_base, program_path)
    } else {
        let search_path = search_path
            .with_context(|| format!("PATH is not set while resolving executable {program}"))?;
        env::split_paths(search_path)
            .map(|directory| {
                let directory = if directory.is_absolute() {
                    directory
                } else {
                    relative_base.join(directory)
                };
                directory.join(program_path)
            })
            .find(|path| is_executable_file(path))
            .with_context(|| format!("executable {program} was not found on PATH"))?
    };
    ensure!(
        is_executable_file(&candidate),
        "external executable must be an executable regular file"
    );
    let identity_target = candidate
        .canonicalize()
        .with_context(|| format!("resolve external executable {}", candidate.display()))?;
    ensure!(
        is_executable_file(&identity_target),
        "external executable must be an executable regular file"
    );
    Ok(ResolvedExecutable {
        invocation_path: candidate,
        identity_target,
    })
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
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

fn resolve_dependency_lock(base: &Path, path: &Path) -> Result<PathBuf> {
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
struct PythonEnvironmentIdentity {
    distributions: BTreeMap<String, String>,
    python_environment_sha256: String,
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

fn inspect_python_environment(
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

fn installed_distributions_sha256(distributions: &BTreeMap<String, String>) -> String {
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

fn is_strict_python_dependency_lock(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name == "requirements.lock.txt")
}

fn validate_dependency_pins(
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

fn python_tree_sha256(root: &Path) -> Result<String> {
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

fn validate_paper_dataset(dataset: &PreparedDataset) -> Result<()> {
    ensure!(
        dataset.summary.conversation_count == PAPER_V2_CONVERSATION_IDS.len(),
        "LoCoMo paper-v2 requires exactly two conversations"
    );
    ensure!(
        dataset.summary.qa_count == PAPER_V2_QA_COUNT,
        "LoCoMo paper-v2 QA denominator mismatch"
    );
    ensure!(
        dataset.summary.retrieval_count == PAPER_V2_RETRIEVAL_COUNT,
        "LoCoMo paper-v2 retrieval denominator mismatch"
    );
    Ok(())
}

fn projected_corpus(question: &PreparedQuestion) -> Vec<BaselineMessage> {
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

async fn execute_native_method(
    method: &ResolvedMethod,
    question: &PreparedQuestion,
    corpus: &[BaselineMessage],
) -> Result<Vec<MethodOutput>> {
    match method {
        ResolvedMethod::Bm25 { id, limit } => {
            let started = Instant::now();
            let ranking = hit_ids(&bm25(corpus, &question.input.query, *limit)?);
            Ok(vec![baseline_output(
                id,
                "bm25",
                ranking,
                started.elapsed(),
            )])
        }
        ResolvedMethod::Recent { id, limit } => {
            let started = Instant::now();
            let ranking = hit_ids(&recent(corpus, *limit)?);
            Ok(vec![baseline_output(
                id,
                "recent",
                ranking,
                started.elapsed(),
            )])
        }
        ResolvedMethod::Semafuse { .. } => {
            unreachable!("Semafuse methods are grouped by conversation")
        }
        ResolvedMethod::ExternalJsonl(_) => {
            unreachable!("external methods are grouped by conversation")
        }
    }
}

async fn execute_semafuse_method(
    method: &ResolvedMethod,
    dataset: &PreparedDataset,
    run_key: &str,
    journal: &mut Option<JsonlJournal<LocomoQuestionRecord>>,
    records: &mut Vec<LocomoQuestionRecord>,
) -> Result<()> {
    let ResolvedMethod::Semafuse {
        id,
        config,
        config_sha256,
        resolved_config_sha256,
        limit,
        stages,
        ..
    } = method
    else {
        bail!("execute_semafuse_method requires a Semafuse method")
    };
    let output_ids = method.output_ids();
    for questions in grouped_eligible_questions(dataset).values() {
        let missing = questions
            .iter()
            .copied()
            .filter(|question| {
                output_ids
                    .iter()
                    .any(|output| !journal_has(journal, run_key, output, &question.input.id))
            })
            .collect::<Vec<_>>();
        if missing.is_empty() {
            continue;
        }
        let exemplar = questions[0];
        validate_shared_conversation(questions, exemplar)?;
        let exemplar_input = runner_question(exemplar);
        let mut session = StrategyRetrievalSession::new(config, &exemplar_input).await?;
        let construction_s = session.construction_metrics().total_s;
        let history_identity = session.history_identity_sha256().to_owned();
        let amortization_n = questions.len();
        for question in missing {
            let started = Instant::now();
            let response = session.query(runner_question(question), *limit).await?;
            let query_wall_s = started.elapsed().as_secs_f64();
            let outputs = semafuse_stage_outputs(
                id,
                config_sha256,
                resolved_config_sha256,
                stages,
                &response,
                query_wall_s,
                construction_s,
                &history_identity,
                amortization_n,
            )?;
            let corpus = projected_corpus(question);
            persist_outputs(journal, records, run_key, question, &corpus, outputs)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn semafuse_stage_outputs(
    id: &str,
    config_sha256: &str,
    resolved_config_sha256: &str,
    stages: &[SemafuseStage],
    response: &BenchmarkResponse,
    query_wall_s: f64,
    construction_s: f64,
    history_identity_sha256: &str,
    construction_amortization_n: usize,
) -> Result<Vec<MethodOutput>> {
    stages
        .iter()
        .map(|stage| {
            let stage_name = stage.as_str();
            let ranking = response
                .stage_rankings
                .get(stage_name)
                .with_context(|| format!("Semafuse method {id} emitted no {stage_name} ranking"))?
                .iter()
                .take(50)
                .cloned()
                .collect::<Vec<_>>();
            Ok(MethodOutput {
                method: format!("{id}::{stage_name}"),
                ranking,
                efficiency: LocomoEfficiency {
                    query_wall_s,
                    retrieval_s: Some(response.retrieval_latency_s),
                    shared_construction_s: Some(construction_s),
                },
                metadata: Map::from_iter([
                    ("kind".into(), json!("semafuse_shared_stage")),
                    ("source_method".into(), json!(id)),
                    ("stage".into(), json!(stage_name)),
                    ("shared_execution".into(), json!(true)),
                    ("config_sha256".into(), json!(config_sha256)),
                    (
                        "resolved_config_sha256".into(),
                        json!(resolved_config_sha256),
                    ),
                    (
                        "history_identity_sha256".into(),
                        json!(history_identity_sha256),
                    ),
                    ("construction_scope".into(), json!("source_conversation")),
                    (
                        "construction_amortization_n".into(),
                        json!(construction_amortization_n),
                    ),
                    ("construction_value_repeated_per_query".into(), json!(true)),
                ]),
            })
        })
        .collect()
}

fn hit_ids(hits: &[BaselineHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.message_id.clone()).collect()
}

fn baseline_output(id: &str, kind: &str, ranking: Vec<String>, elapsed: Duration) -> MethodOutput {
    MethodOutput {
        method: id.to_owned(),
        ranking,
        efficiency: LocomoEfficiency {
            query_wall_s: elapsed.as_secs_f64(),
            ..LocomoEfficiency::default()
        },
        metadata: Map::from_iter([
            ("kind".into(), json!(kind)),
            ("shared_execution".into(), json!(false)),
        ]),
    }
}

fn runner_question(question: &PreparedQuestion) -> RunnerQuestion {
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

fn journal_has(
    journal: &Option<JsonlJournal<LocomoQuestionRecord>>,
    run_key: &str,
    method: &str,
    question_id: &str,
) -> bool {
    journal.as_ref().is_some_and(|journal| {
        journal.completed_keys().contains(&JournalRecordKey {
            run_key: run_key.to_owned(),
            method: method.to_owned(),
            question_id: question_id.to_owned(),
            record_kind: RECORD_KIND.to_owned(),
        })
    })
}

fn persist_outputs(
    journal: &mut Option<JsonlJournal<LocomoQuestionRecord>>,
    records: &mut Vec<LocomoQuestionRecord>,
    run_key: &str,
    question: &PreparedQuestion,
    corpus: &[BaselineMessage],
    outputs: Vec<MethodOutput>,
) -> Result<()> {
    for output in outputs {
        if journal_has(journal, run_key, &output.method, &question.input.id) {
            continue;
        }
        validate_ranking(corpus, &output.ranking)?;
        let record = record_output(run_key, question, output)?;
        if let Some(journal) = journal {
            journal.append(JournalRecord {
                run_key: run_key.to_owned(),
                method: record.method.clone(),
                question_id: record.question_id.clone(),
                record_kind: RECORD_KIND.to_owned(),
                payload: record.clone(),
            })?;
        }
        records.push(record);
    }
    Ok(())
}

fn record_output(
    run_key: &str,
    question: &PreparedQuestion,
    output: MethodOutput,
) -> Result<LocomoQuestionRecord> {
    let gold = &question.evaluator.retrieval.gold_message_ids;
    ensure!(!gold.is_empty(), "cannot score a question without evidence");
    let evidence = score_evidence(&output.ranking, gold)?;
    Ok(LocomoQuestionRecord {
        schema_version: RECORD_SCHEMA_VERSION,
        run_key: run_key.to_owned(),
        method: output.method,
        question_id: question.input.id.clone(),
        category: question.evaluator.category.name().to_owned(),
        category_number: question.evaluator.category.number(),
        gold_count: gold.len(),
        ranked_message_ids: output.ranking,
        evidence,
        efficiency: output.efficiency,
        metadata: output.metadata,
    })
}

fn score_evidence(ranking: &[String], gold: &[String]) -> Result<Vec<EvidenceMetricsAtCutoff>> {
    let ranking_metrics = evaluate_ranking(ranking, gold)?;
    let mut evidence = Vec::with_capacity(EVIDENCE_CUTOFFS.len());
    for cutoff in EVIDENCE_CUTOFFS {
        let metrics = ranking_metrics
            .at(cutoff)
            .with_context(|| format!("shared metric module did not emit cutoff {cutoff}"))?;
        evidence.push(evidence_metrics(metrics, ranking, gold));
    }
    Ok(evidence)
}

fn evidence_metrics(
    metrics: &RankingMetricsAtCutoff,
    ranking: &[String],
    gold: &[String],
) -> EvidenceMetricsAtCutoff {
    let gold = gold.iter().map(String::as_str).collect::<HashSet<_>>();
    let hits = ranking
        .iter()
        .take(metrics.cutoff)
        .filter(|id| gold.contains(id.as_str()))
        .count();
    EvidenceMetricsAtCutoff {
        cutoff: metrics.cutoff,
        hits,
        precision: hits as f64 / metrics.cutoff as f64,
        recall_any: metrics.recall_any,
        recall_all: metrics.recall_all,
        mean_recall: metrics.mean_recall,
        mrr: metrics.mrr,
        ndcg: metrics.ndcg,
    }
}

fn validate_ranking(corpus: &[BaselineMessage], ranking: &[String]) -> Result<()> {
    let corpus_ids = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    for (rank, id) in ranking.iter().enumerate() {
        ensure!(
            !id.trim().is_empty(),
            "ranking has an empty ID at rank {}",
            rank + 1
        );
        ensure!(seen.insert(id), "ranking contains duplicate ID {id}");
        ensure!(
            corpus_ids.contains(id.as_str()),
            "ranking contains unknown ID {id}"
        );
    }
    Ok(())
}

async fn execute_external_method(
    method: &ResolvedExternalMethod,
    dataset: &PreparedDataset,
    run_key: &str,
    journal: &mut Option<JsonlJournal<LocomoQuestionRecord>>,
    records: &mut Vec<LocomoQuestionRecord>,
) -> Result<()> {
    for questions in grouped_eligible_questions(dataset).values() {
        let missing = questions
            .iter()
            .copied()
            .filter(|question| !journal_has(journal, run_key, &method.id, &question.input.id))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            continue;
        }
        let exemplar = questions[0];
        validate_shared_conversation(questions, exemplar)?;
        let construction_started = Instant::now();
        let mut worker = ExternalWorker::spawn(method).await?;
        worker.reset().await?;
        worker.add_history(&exemplar.input.messages).await?;
        let construction_s = construction_started.elapsed().as_secs_f64();
        for question in missing {
            let started = Instant::now();
            let search = worker
                .search(
                    &question.input.query,
                    method.top_k,
                    method.max_context_tokens,
                )
                .await?;
            let query_wall_s = started.elapsed().as_secs_f64();
            let corpus = projected_corpus(question);
            let ranking = project_external_source_ids(&search.items, &corpus, 50)?;
            persist_outputs(
                journal,
                records,
                run_key,
                question,
                &corpus,
                vec![MethodOutput {
                    method: method.id.clone(),
                    ranking,
                    efficiency: LocomoEfficiency {
                        query_wall_s,
                        retrieval_s: None,
                        shared_construction_s: Some(construction_s),
                    },
                    metadata: Map::from_iter([
                        ("kind".into(), json!("external_jsonl")),
                        ("adapter_kind".into(), json!(method.adapter_kind)),
                        ("track".into(), json!(method.track)),
                        (
                            "worker_config_sha256".into(),
                            json!(method.worker_config_sha256),
                        ),
                        ("native_items".into(), json!(search.items.len())),
                        ("context_tokens".into(), json!(search.context_tokens)),
                        ("truncated".into(), json!(search.truncated)),
                        ("shared_execution".into(), json!(false)),
                        ("construction_scope".into(), json!("source_conversation")),
                        ("construction_amortization_n".into(), json!(questions.len())),
                        ("construction_value_repeated_per_query".into(), json!(true)),
                    ]),
                }],
            )?;
        }
        worker.close().await?;
    }
    Ok(())
}

fn grouped_eligible_questions(
    dataset: &PreparedDataset,
) -> BTreeMap<String, Vec<&PreparedQuestion>> {
    let mut histories = BTreeMap::<String, Vec<&PreparedQuestion>>::new();
    for question in dataset
        .questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
    {
        histories
            .entry(question.evaluator.source_conversation_id.clone())
            .or_default()
            .push(question);
    }
    histories
}

fn validate_shared_conversation(
    questions: &[&PreparedQuestion],
    exemplar: &PreparedQuestion,
) -> Result<()> {
    ensure!(
        !questions.is_empty(),
        "shared conversation has no questions"
    );
    for question in questions.iter().skip(1) {
        ensure!(
            question.input.dataset == exemplar.input.dataset
                && question.input.messages == exemplar.input.messages,
            "LoCoMo conversation history differs across questions"
        );
    }
    Ok(())
}

fn project_external_source_ids(
    items: &[ExternalSearchItem],
    corpus: &[BaselineMessage],
    limit: usize,
) -> Result<Vec<String>> {
    let known = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    let mut ranking = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        ensure!(
            !item.source_ids.is_empty(),
            "external item has no source provenance"
        );
        for source_id in &item.source_ids {
            ensure!(
                known.contains(source_id.as_str()),
                "external worker returned unknown source ID"
            );
            if seen.insert(source_id.as_str()) {
                ranking.push(source_id.clone());
                if ranking.len() == limit {
                    return Ok(ranking);
                }
            }
        }
    }
    Ok(ranking)
}

struct ExternalWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    method_id: String,
    request_sequence: u64,
    timeout: Duration,
}

impl ExternalWorker {
    async fn spawn(method: &ResolvedExternalMethod) -> Result<Self> {
        let mut command = Command::new(&method.executable_invocation);
        command
            .args(&method.command[1..])
            .arg("--config")
            .arg(&method.worker_config)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Some(directory) = &method.working_directory {
            command.current_dir(directory);
        }
        if let Some(path) = &method.python_path {
            command.env("PYTHONPATH", path);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("start external worker for {}", method.id))?;
        let stdin = child.stdin.take().context("external worker has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("external worker has no stdout")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            method_id: method.id.clone(),
            request_sequence: 0,
            timeout: method.timeout,
        })
    }

    async fn reset(&mut self) -> Result<()> {
        let result = self.rpc("reset", Map::new()).await?;
        let result: ExternalResetResult =
            serde_json::from_value(result).context("external reset result violates protocol")?;
        ensure!(result.reset, "external reset did not confirm reset=true");
        Ok(())
    }

    async fn add_history(
        &mut self,
        messages: &[crate::longmemeval::BenchmarkMessage],
    ) -> Result<()> {
        let mut sessions = Vec::<(String, String, Vec<Value>)>::new();
        for message in messages {
            ensure!(
                matches!(
                    message.role.as_str(),
                    "user" | "assistant" | "system" | "tool"
                ),
                "external protocol does not support role {}",
                message.role
            );
            ensure!(
                !message.content.trim().is_empty(),
                "external protocol rejects empty messages"
            );
            if sessions
                .last()
                .is_none_or(|(session, _, _)| session != &message.session_id)
            {
                sessions.push((
                    message.session_id.clone(),
                    message.timestamp.clone(),
                    Vec::new(),
                ));
            }
            let session_messages = &mut sessions
                .last_mut()
                .context("external session grouping lost its inserted session")?
                .2;
            session_messages.push(json!({
                "role": message.role,
                "content": message.content,
                "source_id": message.id,
            }));
        }
        for (session, timestamp, messages) in sessions {
            ensure!(
                messages.len() <= 4_096,
                "external add exceeds protocol message limit"
            );
            let expected = messages.len();
            let result = self
                .rpc(
                    "add",
                    Map::from_iter([
                        ("messages".into(), Value::Array(messages)),
                        ("session".into(), json!(session)),
                        (
                            "timestamp".into(),
                            if timestamp.trim().is_empty() {
                                Value::Null
                            } else {
                                json!(timestamp)
                            },
                        ),
                    ]),
                )
                .await?;
            let result: ExternalAddResult =
                serde_json::from_value(result).context("external add result violates protocol")?;
            ensure!(
                result.accepted == expected,
                "external worker accepted only {} of {expected} messages",
                result.accepted
            );
        }
        Ok(())
    }

    async fn search(
        &mut self,
        query: &str,
        top_k: usize,
        max_context_tokens: usize,
    ) -> Result<ExternalSearchResult> {
        let result = self
            .rpc(
                "search",
                Map::from_iter([
                    ("query".into(), json!(query)),
                    ("top_k".into(), json!(top_k)),
                    ("max_context_tokens".into(), json!(max_context_tokens)),
                ]),
            )
            .await?;
        let result: ExternalSearchResult =
            serde_json::from_value(result).context("external search result violates protocol")?;
        ensure!(
            result.items.len() <= top_k,
            "external response exceeds top_k"
        );
        ensure!(
            result.context_tokens <= max_context_tokens,
            "external response exceeds context cap"
        );
        ensure!(
            result.context
                == result
                    .items
                    .iter()
                    .map(|item| item.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            "external context does not match ranked item texts"
        );
        ensure!(
            result.items.is_empty() == (result.context_tokens == 0),
            "external context token count is inconsistent"
        );
        for item in &result.items {
            ensure!(!item.id.trim().is_empty(), "external item id is empty");
            ensure!(!item.text.trim().is_empty(), "external item text is empty");
            ensure!(item.token_count > 0, "external item token count is zero");
            ensure!(
                !item.source_ids.is_empty()
                    && item.source_ids.iter().all(|id| !id.trim().is_empty()),
                "external item source provenance is empty"
            );
            ensure!(
                item.session
                    .as_ref()
                    .is_none_or(|value| !value.trim().is_empty()),
                "external item session is empty"
            );
            ensure!(
                item.timestamp
                    .as_ref()
                    .is_none_or(|value| !value.trim().is_empty()),
                "external item timestamp is empty"
            );
            if let Some(score) = item.score {
                ensure!(score.is_finite(), "external item score is not finite");
            }
        }
        Ok(result)
    }

    async fn close(mut self) -> Result<()> {
        let result = self.rpc("close", Map::new()).await?;
        let result: ExternalCloseResult =
            serde_json::from_value(result).context("external close result violates protocol")?;
        ensure!(result.closed, "external close did not confirm closed=true");
        let status = timeout(self.timeout, self.child.wait())
            .await
            .context("external worker did not exit after close")??;
        ensure!(
            status.success(),
            "external worker exited unsuccessfully after close"
        );
        Ok(())
    }

    async fn rpc(&mut self, op: &str, fields: Map<String, Value>) -> Result<Value> {
        self.request_sequence += 1;
        let request_id = format!("{}-{}", self.method_id, self.request_sequence);
        let mut request = Map::from_iter([
            ("protocol".into(), json!(EXTERNAL_PROTOCOL)),
            ("request_id".into(), json!(request_id)),
            ("op".into(), json!(op)),
        ]);
        for (key, value) in fields {
            ensure!(
                request.insert(key, value).is_none(),
                "duplicate protocol field"
            );
        }
        let mut encoded = serde_json::to_vec(&request).context("serialize external request")?;
        ensure!(
            encoded.len() <= EXTERNAL_MAX_LINE_BYTES,
            "external request exceeds protocol line limit"
        );
        encoded.push(b'\n');
        timeout(self.timeout, async {
            self.stdin.write_all(&encoded).await?;
            self.stdin.flush().await
        })
        .await
        .with_context(|| format!("external {op} write timed out"))??;

        let mut response = Vec::new();
        let read = timeout(self.timeout, self.stdout.read_until(b'\n', &mut response))
            .await
            .with_context(|| format!("external {op} response timed out"))??;
        ensure!(read > 0, "external worker exited before {op} response");
        ensure!(
            response.ends_with(b"\n") && response.len() <= EXTERNAL_MAX_LINE_BYTES + 1,
            "external response violates protocol line framing"
        );
        let envelope: ExternalEnvelope = serde_json::from_slice(&response)
            .context("external response is not a strict protocol envelope")?;
        ensure!(
            envelope.protocol == EXTERNAL_PROTOCOL,
            "external protocol mismatch"
        );
        ensure!(
            envelope.request_id == request_id,
            "external request_id mismatch"
        );
        ensure!(envelope.op == op, "external response op mismatch");
        if envelope.ok {
            ensure!(
                envelope.error.is_none(),
                "successful external response contains error"
            );
            envelope
                .result
                .context("successful external response has no result")
        } else {
            ensure!(
                envelope.result.is_none(),
                "failed external response contains result"
            );
            let error = envelope
                .error
                .context("failed external response has no error")?;
            bail!(
                "external worker {} failed with {} (retryable={}): {}",
                self.method_id,
                error.code,
                error.retryable,
                error.message
            )
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalEnvelope {
    protocol: String,
    request_id: String,
    op: String,
    ok: bool,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<ExternalError>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalError {
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalResetResult {
    reset: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalAddResult {
    accepted: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalCloseResult {
    closed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalSearchResult {
    items: Vec<ExternalSearchItem>,
    context: String,
    context_tokens: usize,
    truncated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalSearchItem {
    id: String,
    text: String,
    score: Option<f64>,
    source_ids: Vec<String>,
    session: Option<String>,
    timestamp: Option<String>,
    token_count: usize,
}

/// Builds paired accuracy reports from complete persisted records.
pub fn build_locomo_summary_from_records(
    baseline_method: &str,
    records: &[LocomoQuestionRecord],
) -> Result<Value> {
    ensure!(!records.is_empty(), "cannot summarize empty LoCoMo records");
    let run_key = &records[0].run_key;
    ensure!(
        records.iter().all(|record| record.run_key == *run_key),
        "cannot summarize records from mixed run identities"
    );
    let mut reports = BTreeMap::<String, ScalarBenchmarkReport>::new();
    for cutoff in EVIDENCE_CUTOFFS {
        for metric in [
            "precision",
            "recall_any",
            "recall_all",
            "mean_recall",
            "mrr",
            "ndcg",
        ] {
            let rows = records
                .iter()
                .map(|record| {
                    let value = metric_value(record, cutoff, metric)?;
                    Ok(ScalarMetricRow {
                        method: record.method.clone(),
                        question_id: record.question_id.clone(),
                        question_type: record.category.clone(),
                        value,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let name = format!("evidence_{metric}@{cutoff}");
            reports.insert(
                name.clone(),
                build_scalar_report(
                    &name,
                    baseline_method,
                    MetricDirection::HigherIsBetter,
                    &rows,
                )?,
            );
        }
    }
    let method_ids = records
        .iter()
        .map(|record| record.method.clone())
        .collect::<BTreeSet<_>>();
    Ok(json!({
        "schema_version": 1,
        "run_key": run_key,
        "benchmark": "locomo",
        "protocol": "paper-v2-retrieval-only",
        "baseline_method": baseline_method,
        "qa_n": PAPER_V2_QA_COUNT,
        "retrieval_eval_n": PAPER_V2_RETRIEVAL_COUNT,
        "cutoffs": EVIDENCE_CUTOFFS,
        "methods": method_ids,
        "reports": reports,
        "official_token_f1": {
            "value": null,
            "status": "not_applicable_without_answer_predictions",
        },
    }))
}

fn metric_value(record: &LocomoQuestionRecord, cutoff: usize, metric: &str) -> Result<f64> {
    let value = record
        .evidence
        .iter()
        .find(|value| value.cutoff == cutoff)
        .with_context(|| format!("record lacks cutoff {cutoff}"))?;
    match metric {
        "precision" => Ok(value.precision),
        "recall_any" => Ok(value.recall_any),
        "recall_all" => Ok(value.recall_all),
        "mean_recall" => Ok(value.mean_recall),
        "mrr" => Ok(value.mrr),
        "ndcg" => Ok(value.ndcg),
        _ => bail!("unknown LoCoMo metric {metric}"),
    }
}

fn validate_record_completeness(
    records: &[LocomoQuestionRecord],
    dataset: &PreparedDataset,
    methods: &[ResolvedMethod],
    run_key: &str,
) -> Result<()> {
    let questions = dataset
        .questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
        .map(|question| (question.input.id.as_str(), question))
        .collect::<HashMap<_, _>>();
    let outputs = methods
        .iter()
        .flat_map(ResolvedMethod::output_ids)
        .collect::<BTreeSet<_>>();
    let mut pairs = HashSet::new();
    for record in records {
        ensure!(
            record.schema_version == RECORD_SCHEMA_VERSION,
            "record schema mismatch"
        );
        ensure!(record.run_key == run_key, "records mix run identities");
        ensure!(
            outputs.contains(&record.method),
            "record has unknown method {}",
            record.method
        );
        let question = questions
            .get(record.question_id.as_str())
            .with_context(|| format!("record has unknown question {}", record.question_id))?;
        ensure!(
            record.category == question.evaluator.category.name()
                && record.category_number == question.evaluator.category.number(),
            "record category does not match evaluator"
        );
        ensure!(
            record.gold_count == question.evaluator.retrieval.gold_message_ids.len(),
            "record gold denominator does not match evaluator"
        );
        ensure!(
            record
                .evidence
                .iter()
                .map(|metric| metric.cutoff)
                .collect::<Vec<_>>()
                == EVIDENCE_CUTOFFS,
            "record does not contain exact fixed cutoffs"
        );
        ensure!(
            record.efficiency.query_wall_s.is_finite()
                && record.efficiency.query_wall_s >= 0.0
                && record
                    .efficiency
                    .retrieval_s
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
                && record
                    .efficiency
                    .shared_construction_s
                    .is_none_or(|value| value.is_finite() && value >= 0.0),
            "record contains invalid efficiency timing"
        );
        validate_ranking(&projected_corpus(question), &record.ranked_message_ids)?;
        let expected_evidence = score_evidence(
            &record.ranked_message_ids,
            &question.evaluator.retrieval.gold_message_ids,
        )?;
        ensure!(
            evidence_matches(&record.evidence, &expected_evidence),
            "record evidence metrics do not match its persisted ranking for {} / {}",
            record.method,
            record.question_id
        );
        ensure!(
            pairs.insert((record.method.as_str(), record.question_id.as_str())),
            "duplicate method/question record"
        );
    }
    ensure!(
        records.len() == questions.len() * outputs.len(),
        "incomplete LoCoMo records: expected {}, got {}",
        questions.len() * outputs.len(),
        records.len()
    );
    for question_id in questions.keys() {
        for method in &outputs {
            ensure!(
                pairs.contains(&(method.as_str(), *question_id)),
                "missing method {method} / question {question_id}"
            );
        }
    }
    Ok(())
}

fn evidence_matches(
    actual: &[EvidenceMetricsAtCutoff],
    expected: &[EvidenceMetricsAtCutoff],
) -> bool {
    const EPSILON: f64 = 1e-12;
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual.cutoff == expected.cutoff
                && actual.hits == expected.hits
                && [
                    (actual.precision, expected.precision),
                    (actual.recall_any, expected.recall_any),
                    (actual.recall_all, expected.recall_all),
                    (actual.mean_recall, expected.mean_recall),
                    (actual.mrr, expected.mrr),
                    (actual.ndcg, expected.ndcg),
                ]
                .into_iter()
                .all(|(left, right)| (left - right).abs() <= EPSILON)
        })
}

fn eligible_category_counts(dataset: &PreparedDataset) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for question in dataset
        .questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
    {
        *counts
            .entry(question.evaluator.category.name().to_owned())
            .or_default() += 1;
    }
    counts
}

fn qa_category_counts(dataset: &PreparedDataset) -> BTreeMap<String, usize> {
    dataset
        .summary
        .category_counts
        .iter()
        .map(|(category, count)| (category.name().to_owned(), *count))
        .collect()
}

fn method_manifest(methods: &[ResolvedMethod]) -> Vec<Value> {
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

fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
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

fn cargo_lock_sha256() -> String {
    hex::encode(Sha256::digest(include_bytes!("../Cargo.lock")))
}

fn benchmark_code_sha256() -> String {
    let mut digest = Sha256::new();
    for (path, bytes) in [
        (
            "src/locomo_suite.rs",
            include_bytes!("locomo_suite.rs").as_slice(),
        ),
        ("src/locomo.rs", include_bytes!("locomo.rs").as_slice()),
        (
            "src/benchmark.rs",
            include_bytes!("benchmark.rs").as_slice(),
        ),
        (
            "src/benchmark_baselines.rs",
            include_bytes!("benchmark_baselines.rs").as_slice(),
        ),
        (
            "src/benchmark_metrics.rs",
            include_bytes!("benchmark_metrics.rs").as_slice(),
        ),
        (
            "src/benchmark_report.rs",
            include_bytes!("benchmark_report.rs").as_slice(),
        ),
        (
            "src/benchmark_journal.rs",
            include_bytes!("benchmark_journal.rs").as_slice(),
        ),
    ] {
        digest.update((path.len() as u64).to_be_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

fn run_key(
    suite_bytes: &[u8],
    dataset_sha256: &str,
    methods: &[ResolvedMethod],
    cargo_lock_sha256: &str,
    code_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"semafuse-locomo-retrieval-run/v2");
    digest.update((suite_bytes.len() as u64).to_be_bytes());
    digest.update(suite_bytes);
    digest.update(dataset_sha256.as_bytes());
    digest.update(cargo_lock_sha256.as_bytes());
    digest.update(code_sha256.as_bytes());
    for method in methods {
        digest.update(method.id().as_bytes());
        match method {
            ResolvedMethod::Semafuse {
                config_sha256,
                resolved_config_sha256,
                ..
            } => {
                digest.update(config_sha256.as_bytes());
                digest.update(resolved_config_sha256.as_bytes());
            }
            ResolvedMethod::ExternalJsonl(method) => {
                digest.update(method.worker_config_sha256.as_bytes());
                digest.update(method.command_sha256.as_bytes());
                digest.update(method.executable_sha256.as_bytes());
                update_optional_sha256(&mut digest, &method.python_tree_sha256);
                update_optional_sha256(&mut digest, &method.dependency_lock_sha256);
                update_optional_sha256(&mut digest, &method.python_environment_sha256);
                digest.update([u8::from(method.dependency_pins_verified)]);
            }
            _ => {}
        }
    }
    hex::encode(digest.finalize())
}

fn update_optional_sha256(digest: &mut Sha256, value: &Option<String>) {
    if let Some(value) = value {
        digest.update([1]);
        digest.update(value.as_bytes());
    } else {
        digest.update([0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn metric(cutoff: usize, value: f64) -> EvidenceMetricsAtCutoff {
        EvidenceMetricsAtCutoff {
            cutoff,
            hits: usize::from(value > 0.0),
            precision: value,
            recall_any: value,
            recall_all: value,
            mean_recall: value,
            mrr: value,
            ndcg: value,
        }
    }

    fn record(method: &str, question: &str, category: &str, value: f64) -> LocomoQuestionRecord {
        LocomoQuestionRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            run_key: "run".into(),
            method: method.into(),
            question_id: question.into(),
            category: category.into(),
            category_number: if category == "multi-hop" { 1 } else { 4 },
            gold_count: 1,
            ranked_message_ids: vec!["m".into()],
            evidence: EVIDENCE_CUTOFFS
                .into_iter()
                .map(|cutoff| metric(cutoff, value))
                .collect(),
            efficiency: LocomoEfficiency::default(),
            metadata: Map::new(),
        }
    }

    #[test]
    fn suite_requires_fixed_depth_and_a_direct_baseline() {
        let invalid = SuiteConfig {
            schema_version: 1,
            name: "test".into(),
            baseline_method: "semafuse::retrieved".into(),
            methods: vec![
                MethodConfig::Bm25 {
                    id: "bm25".into(),
                    limit: 10,
                },
                MethodConfig::Semafuse {
                    id: "semafuse".into(),
                    config: "unused".into(),
                    limit: 50,
                    stages: vec![SemafuseStage::Retrieved],
                },
            ],
        };
        let error = validate_suite(&invalid).unwrap_err();
        assert!(format!("{error:#}").contains("at least 50"));
    }

    #[test]
    fn bundled_suites_parse_and_resolve_without_model_calls() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for name in [
            "paper_v2.baselines.toml",
            "paper_v2.local-retrieval.toml",
            "paper_v2.openrouter.toml",
            "paper_v2.external-fake.contract-test.toml",
        ] {
            let path = root.join("benchmarks/locomo/configs").join(name);
            let suite = load_suite(&path).unwrap();
            validate_suite(&suite).unwrap();
            let methods = resolve_methods(&suite, path.parent().unwrap()).unwrap();
            assert_eq!(methods.len(), suite.methods.len());
        }
    }

    #[test]
    fn summary_has_exact_cutoffs_and_per_category_reports() {
        let records = vec![
            record("bm25", "q1", "multi-hop", 0.0),
            record("candidate", "q1", "multi-hop", 1.0),
            record("bm25", "q2", "single-hop", 1.0),
            record("candidate", "q2", "single-hop", 1.0),
        ];
        let summary = build_locomo_summary_from_records("bm25", &records).unwrap();
        assert_eq!(summary["cutoffs"], json!([1, 5, 10, 50]));
        assert_eq!(summary["reports"].as_object().unwrap().len(), 24);
        let report = &summary["reports"]["evidence_recall_all@10"];
        assert_eq!(report["denominator"], 2);
        assert_eq!(
            report["methods"]["candidate"]["by_question_type"]["multi-hop"]["mean"],
            1.0
        );
        assert_eq!(summary["official_token_f1"]["value"], Value::Null);
    }

    #[test]
    fn external_projection_is_ranked_unique_and_fail_closed() {
        let corpus = vec![
            BaselineMessage {
                id: "a".into(),
                role: "user".into(),
                content: "a".into(),
                session_id: "s".into(),
                corpus_index: 0,
            },
            BaselineMessage {
                id: "b".into(),
                role: "assistant".into(),
                content: "b".into(),
                session_id: "s".into(),
                corpus_index: 1,
            },
        ];
        let item = |ids: Vec<&str>| ExternalSearchItem {
            id: "item".into(),
            text: "text".into(),
            score: Some(1.0),
            source_ids: ids.into_iter().map(str::to_owned).collect(),
            session: None,
            timestamp: None,
            token_count: 1,
        };
        assert_eq!(
            project_external_source_ids(&[item(vec!["b", "a"]), item(vec!["b"])], &corpus, 50)
                .unwrap(),
            ["b", "a"]
        );
        assert!(project_external_source_ids(&[item(vec!["unknown"])], &corpus, 50).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn executable_resolution_uses_path_and_binds_file_bytes() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let bin = directory.path().join("bin");
        fs::create_dir(&bin).unwrap();
        let executable = bin.join("worker");
        let bytes = b"#!/bin/sh\nexit 0\n";
        fs::write(&executable, bytes).unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();

        let resolved = resolve_executable(
            "worker",
            directory.path(),
            Some(std::ffi::OsStr::new("bin/../bin")),
        )
        .unwrap();
        assert_eq!(
            resolved.invocation_path,
            directory.path().join("bin/../bin/worker")
        );
        assert_ne!(resolved.invocation_path, resolved.identity_target);
        assert_eq!(resolved.identity_target, executable.canonicalize().unwrap());
        assert_eq!(
            sha256_file(&resolved.identity_target).unwrap(),
            hex::encode(Sha256::digest(bytes))
        );
    }

    #[test]
    fn python_tree_hash_is_path_stable_and_content_bound() {
        let first = tempdir().unwrap();
        fs::create_dir(first.path().join("pkg")).unwrap();
        fs::write(first.path().join("pkg/b.py"), b"B\n").unwrap();
        fs::write(first.path().join("a.py"), b"A\n").unwrap();
        fs::write(first.path().join("ignored.txt"), b"ignored").unwrap();

        let second = tempdir().unwrap();
        fs::create_dir(second.path().join("pkg")).unwrap();
        fs::write(second.path().join("a.py"), b"A\n").unwrap();
        fs::write(second.path().join("pkg/b.py"), b"B\n").unwrap();

        let expected = "21e0e22606a49f4a3813dc462fe749bd499ae3cccf69acb434903ef7e8b6fbf9";
        assert_eq!(python_tree_sha256(first.path()).unwrap(), expected);
        assert_eq!(python_tree_sha256(second.path()).unwrap(), expected);

        fs::write(second.path().join("pkg/b.py"), b"changed\n").unwrap();
        assert_ne!(python_tree_sha256(second.path()).unwrap(), expected);
    }

    #[test]
    fn external_identity_inputs_fail_closed() {
        let directory = tempdir().unwrap();
        let worker_config = directory.path().join("worker.json");
        fs::write(
            &worker_config,
            br#"{"adapter":{"kind":"fake"},"run_policy":{"track":"development"}}"#,
        )
        .unwrap();
        let search_path = env::var_os("PATH");
        let command = resolve_executable("python3", directory.path(), search_path.as_deref())
            .unwrap()
            .invocation_path;
        let suite = SuiteConfig {
            schema_version: 1,
            name: "identity-test".into(),
            baseline_method: "bm25".into(),
            methods: vec![
                MethodConfig::Bm25 {
                    id: "bm25".into(),
                    limit: 50,
                },
                MethodConfig::ExternalJsonl {
                    id: "external".into(),
                    worker_config: worker_config.clone(),
                    command: vec![command.to_string_lossy().into_owned()],
                    working_directory: None,
                    python_path: None,
                    source_root: None,
                    dependency_lock: None,
                    top_k: 50,
                    max_context_tokens: 16_384,
                    timeout_seconds: 10,
                },
            ],
        };
        let error = resolve_methods(&suite, directory.path()).unwrap_err();
        assert!(format!("{error:#}").contains("outside contract-test"));

        let source_root = directory.path().join("source-root");
        fs::create_dir(&source_root).unwrap();
        fs::write(source_root.join("module.py"), b"module\n").unwrap();
        let environment = inspect_python_environment(&command, None).unwrap();
        let (package, version) = environment.distributions.first_key_value().unwrap();
        let lock_contents = format!("{package}=={version}\n");
        let dependency_lock = directory.path().join("requirements.lock.txt");
        fs::write(&dependency_lock, &lock_contents).unwrap();
        let mut configured = suite.clone();
        let MethodConfig::ExternalJsonl {
            source_root: configured_source_root,
            dependency_lock: configured_dependency_lock,
            ..
        } = &mut configured.methods[1]
        else {
            unreachable!()
        };
        *configured_source_root = Some("source-root".into());
        *configured_dependency_lock = Some("requirements.lock.txt".into());
        validate_suite(&configured).unwrap();
        let resolved = resolve_methods(&configured, directory.path()).unwrap();
        let ResolvedMethod::ExternalJsonl(resolved) = &resolved[1] else {
            unreachable!()
        };
        assert_eq!(
            resolved.python_tree_sha256.as_deref(),
            Some(python_tree_sha256(&source_root).unwrap().as_str())
        );
        assert_eq!(
            resolved.dependency_lock_sha256.as_deref(),
            Some(hex::encode(Sha256::digest(lock_contents.as_bytes())).as_str())
        );
        assert!(resolved.python_environment_sha256.is_some());
        assert!(resolved.dependency_pins_verified);

        let mut incomplete = configured.clone();
        let MethodConfig::ExternalJsonl {
            dependency_lock, ..
        } = &mut incomplete.methods[1]
        else {
            unreachable!()
        };
        *dependency_lock = None;
        let error = validate_suite(&incomplete).unwrap_err();
        assert!(
            format!("{error:#}").contains("configure source_root and dependency_lock together")
        );

        let empty_source = directory.path().join("empty-source");
        fs::create_dir(&empty_source).unwrap();
        let error = python_tree_sha256(&empty_source).unwrap_err();
        assert!(format!("{error:#}").contains("contains no .py files"));

        let source_file = directory.path().join("source.py");
        fs::write(&source_file, b"source\n").unwrap();
        let error = python_tree_sha256(&source_file).unwrap_err();
        assert!(format!("{error:#}").contains("regular directory"));

        let dependency_directory = directory.path().join("dependency-directory");
        fs::create_dir(&dependency_directory).unwrap();
        let error = resolve_dependency_lock(directory.path(), &dependency_directory).unwrap_err();
        assert!(format!("{error:#}").contains("regular non-symlink file"));
    }

    #[cfg(unix)]
    #[test]
    fn python_tree_rejects_symlinks_when_supported() {
        use std::os::unix::fs::symlink;

        let directory = tempdir().unwrap();
        fs::write(directory.path().join("source.py"), b"source\n").unwrap();
        match symlink("source.py", directory.path().join("linked.py")) {
            Ok(()) => {
                let error = python_tree_sha256(directory.path()).unwrap_err();
                assert!(format!("{error:#}").contains("contains a symlink"));
            }
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {}
            Err(error) => panic!("create test symlink: {error}"),
        }
    }

    #[test]
    fn installed_distribution_fingerprint_captures_complete_package_state() {
        let first = BTreeMap::from([
            ("demo".to_owned(), "1.0".to_owned()),
            ("shared".to_owned(), "2.0".to_owned()),
        ]);
        let changed_version = BTreeMap::from([
            ("demo".to_owned(), "1.1".to_owned()),
            ("shared".to_owned(), "2.0".to_owned()),
        ]);
        let added_distribution = BTreeMap::from([
            ("demo".to_owned(), "1.0".to_owned()),
            ("extra".to_owned(), "3.0".to_owned()),
            ("shared".to_owned(), "2.0".to_owned()),
        ]);
        assert_ne!(
            installed_distributions_sha256(&first),
            installed_distributions_sha256(&changed_version)
        );
        assert_ne!(
            installed_distributions_sha256(&first),
            installed_distributions_sha256(&added_distribution)
        );
    }

    #[test]
    fn strict_dependency_lock_rejects_pin_mismatch_and_unknown_syntax() {
        let directory = tempdir().unwrap();
        let lock = directory.path().join("requirements.lock.txt");
        let installed = BTreeMap::from([
            ("demo-pkg".to_owned(), "1.0+cpu".to_owned()),
            ("other".to_owned(), "2.0".to_owned()),
        ]);

        fs::write(
            &lock,
            b"# exact lock\n--extra-index-url https://example.invalid/simple\nDemo_Pkg==1.0+cpu\nother==2.0\n",
        )
        .unwrap();
        assert!(is_strict_python_dependency_lock(&lock));
        validate_dependency_pins(&lock, &installed).unwrap();

        fs::write(&lock, b"demo-pkg==9.0\n").unwrap();
        let error = validate_dependency_pins(&lock, &installed).unwrap_err();
        assert!(format!("{error:#}").contains("has version 1.0+cpu, expected 9.0"));

        fs::write(&lock, b"demo-pkg>=1.0\n").unwrap();
        let error = validate_dependency_pins(&lock, &installed).unwrap_err();
        assert!(format!("{error:#}").contains("not an exact name==version pin"));

        fs::write(&lock, b"--extra-index-url-typo https://example.invalid\n").unwrap();
        let error = validate_dependency_pins(&lock, &installed).unwrap_err();
        assert!(format!("{error:#}").contains("unrecognized requirement option"));
    }

    #[test]
    fn external_identity_hashes_are_manifested_and_bound_into_run_key() {
        let method = |executable_sha256: &str,
                      python_tree_sha256: &str,
                      dependency_lock_sha256: &str,
                      python_environment_sha256: &str| {
            let model_revisions = BTreeMap::from([("embedding".into(), "model-revision".into())]);
            vec![ResolvedMethod::ExternalJsonl(ResolvedExternalMethod {
                id: "external".into(),
                worker_config: "worker.json".into(),
                worker_config_sha256: "worker-config".into(),
                adapter_kind: "paper".into(),
                track: "paper-reproduction".into(),
                upstream_revision: Some("upstream-revision".into()),
                model_revisions,
                reader_tokenizer_model: None,
                reader_tokenizer_revision: None,
                command: vec!["worker".into()],
                command_sha256: "command".into(),
                executable_invocation: "/resolved/worker".into(),
                executable_sha256: executable_sha256.into(),
                working_directory: None,
                python_path: None,
                python_tree_sha256: Some(python_tree_sha256.into()),
                dependency_lock_sha256: Some(dependency_lock_sha256.into()),
                python_environment_sha256: Some(python_environment_sha256.into()),
                dependency_pins_verified: true,
                top_k: 50,
                max_context_tokens: 16_384,
                timeout: Duration::from_secs(10),
            })]
        };
        let baseline = method("executable-a", "python-a", "lock-a", "environment-a");
        let baseline_key = run_key(b"suite", "dataset", &baseline, "cargo", "code");
        assert_ne!(
            baseline_key,
            run_key(
                b"suite",
                "dataset",
                &method("executable-b", "python-a", "lock-a", "environment-a"),
                "cargo",
                "code"
            )
        );
        assert_ne!(
            baseline_key,
            run_key(
                b"suite",
                "dataset",
                &method("executable-a", "python-b", "lock-a", "environment-a"),
                "cargo",
                "code"
            )
        );
        assert_ne!(
            baseline_key,
            run_key(
                b"suite",
                "dataset",
                &method("executable-a", "python-a", "lock-b", "environment-a"),
                "cargo",
                "code"
            )
        );
        assert_ne!(
            baseline_key,
            run_key(
                b"suite",
                "dataset",
                &method("executable-a", "python-a", "lock-a", "environment-b"),
                "cargo",
                "code"
            )
        );

        let manifest = method_manifest(&baseline);
        assert_eq!(manifest[0]["executable_sha256"], "executable-a");
        assert_eq!(manifest[0]["python_tree_sha256"], "python-a");
        assert_eq!(manifest[0]["dependency_lock_sha256"], "lock-a");
        assert_eq!(manifest[0]["python_environment_sha256"], "environment-a");
        assert_eq!(manifest[0]["dependency_pins_verified"], true);
        assert_eq!(manifest[0]["protected_identity_complete"], true);
    }

    #[tokio::test]
    async fn existing_fake_worker_protocol_round_trips() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let external = root.join("benchmarks/external_memory");
        if !external
            .join("src/semafuse_external_memory/worker.py")
            .is_file()
        {
            return;
        }
        let search_path = env::var_os("PATH");
        let executable = resolve_executable("python3", &external, search_path.as_deref()).unwrap();
        let executable_sha256 = sha256_file(&executable.identity_target).unwrap();
        let method = ResolvedExternalMethod {
            id: "fake".into(),
            worker_config: external.join("configs/fake.contract-test.json"),
            worker_config_sha256: "hash".into(),
            adapter_kind: "fake".into(),
            track: "contract-test".into(),
            upstream_revision: None,
            model_revisions: BTreeMap::new(),
            reader_tokenizer_model: None,
            reader_tokenizer_revision: None,
            command: vec![
                "python3".into(),
                "-m".into(),
                "semafuse_external_memory.worker".into(),
            ],
            command_sha256: "command".into(),
            executable_invocation: executable.invocation_path,
            executable_sha256,
            working_directory: Some(external.clone()),
            python_path: Some(external.join("src")),
            python_tree_sha256: None,
            dependency_lock_sha256: None,
            python_environment_sha256: None,
            dependency_pins_verified: false,
            top_k: 50,
            max_context_tokens: 16_384,
            timeout: Duration::from_secs(10),
        };
        let mut worker = ExternalWorker::spawn(&method).await.unwrap();
        worker.reset().await.unwrap();
        worker
            .add_history(&[
                crate::longmemeval::BenchmarkMessage {
                    id: "m1".into(),
                    index: 0,
                    role: "user".into(),
                    content: "alpha fact".into(),
                    timestamp: "2026-01-01".into(),
                    session_id: "s1".into(),
                },
                crate::longmemeval::BenchmarkMessage {
                    id: "m2".into(),
                    index: 1,
                    role: "assistant".into(),
                    content: "beta fact".into(),
                    timestamp: "2026-01-01".into(),
                    session_id: "s1".into(),
                },
            ])
            .await
            .unwrap();
        let result = worker.search("alpha", 50, 16_384).await.unwrap();
        assert_eq!(result.items[0].source_ids, ["m1"]);
        worker.close().await.unwrap();
    }

    #[test]
    fn journal_resume_key_is_method_and_opaque_question() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("per-question.jsonl");
        let payload = record("bm25", "q-opaque", "single-hop", 1.0);
        {
            let mut journal = JsonlJournal::open(&path, "run").unwrap();
            journal
                .append(JournalRecord {
                    run_key: "run".into(),
                    method: "bm25".into(),
                    question_id: "q-opaque".into(),
                    record_kind: RECORD_KIND.into(),
                    payload,
                })
                .unwrap();
        }
        let journal = Some(JsonlJournal::open(&path, "run").unwrap());
        assert!(journal_has(&journal, "run", "bm25", "q-opaque"));
    }
}
