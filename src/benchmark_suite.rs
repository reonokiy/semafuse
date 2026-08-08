//! Native, fail-closed LongMemEval comparison runner.
//!
//! The runner executes deployable baselines and production Semafuse strategies
//! against the same label-isolated cases. Retrieval and answer evaluation are
//! separate protocols: official user-turn retrieval excludes abstention and
//! assistant-only evidence, all-role retrieval excludes abstention only, while
//! full answer evaluation includes every selected question.

use crate::{
    benchmark::{
        BenchmarkMessage, BenchmarkQuestion, run_strategy_question, run_strategy_retrieval_question,
    },
    benchmark_baselines::{
        BaselineHit, Message as BaselineMessage, bm25, bm25_positive, full_history,
        longmemeval_official_bm25, oracle_session, oracle_turn, recent,
    },
    benchmark_journal::{
        JournalRecord, JournalRecordKey, JsonlJournal, SummaryEnvelope, write_summary_atomic,
    },
    benchmark_metrics::{RankingMetrics, evaluate_ranking},
    benchmark_report::{
        MetricDirection, ScalarBenchmarkReport, ScalarMetricRow, build_scalar_report,
    },
    config::{self as semafuse_config, Config},
    inference::{GenerationOptions, OpenAiEndpoint},
    longmemeval::{
        BenchmarkInput, PreparedQuestion, RetrievalProfileGold, SplitIdFilter, load_longmemeval,
    },
    longmemeval_eval::{
        AnswerMemoryKind, AnswerPromptInput, LongMemEvalQuestionType, PredictionRecord,
        render_answer_prompt, render_judge_prompt, score_judge_response,
    },
};
use anyhow::{Context, Result, ensure};
use config::{Config as LayeredConfig, File};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const SUITE_SCHEMA_VERSION: u32 = 1;
const RECORD_SCHEMA_VERSION: u32 = 2;

/// Which LongMemEval retrieval protocol the suite scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalProfile {
    /// Official user-message projection; 419 eligible questions on full data.
    OfficialUserTurn,
    /// Semafuse/LazyMem all-role message projection; 470 eligible questions.
    AllRoleMessage,
}

/// Whether to run retrieval only or also invoke the common answerer and judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    /// Evaluate retrieval without invoking the shared answerer or judge.
    Retrieval,
    /// Evaluate retrieval and invoke the shared answerer and judge.
    Full,
}

/// How one benchmark output may be used in aggregate result displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportRole {
    /// A deployable reference method suitable for the paired baseline.
    DeployableBaseline,
    /// A complete system under evaluation.
    Candidate,
    /// A diagnostic variant or shared pipeline stage.
    Ablation,
    /// A non-retrieval lower bound or context-capacity control.
    Control,
    /// An evaluator-label-dependent upper bound that is not deployable.
    Oracle,
}

impl ReportRole {
    fn is_paired(self) -> bool {
        matches!(
            self,
            Self::DeployableBaseline | Self::Candidate | Self::Ablation
        )
    }

    fn is_diagnostic(self) -> bool {
        matches!(self, Self::Control | Self::Oracle)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StrategyExecution {
    #[default]
    FullMemory,
    RetrievalOnly,
}

/// Filesystem and mode inputs for one benchmark execution.
#[derive(Debug, Clone)]
pub struct SuiteRunOptions {
    /// Modular suite TOML path.
    pub suite_path: PathBuf,
    /// Official `longmemeval_s_cleaned.json` path.
    pub dataset_path: PathBuf,
    /// Optional frozen split manifest path.
    pub split_manifest: Option<PathBuf>,
    /// Split key inside the manifest, normally `test`.
    pub split_name: String,
    /// Retrieval-only or complete answer/judge execution.
    pub mode: BenchmarkMode,
    /// Optional crash-safe output directory for journal and summary artifacts.
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteConfig {
    schema_version: u32,
    name: String,
    retrieval_profile: RetrievalProfile,
    baseline_method: String,
    methods: Vec<MethodConfig>,
    #[serde(default)]
    answerer: Option<ChatModelConfig>,
    #[serde(default)]
    judge: Option<ChatModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatModelConfig {
    endpoint: OpenAiEndpoint,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    max_tokens: usize,
    #[serde(default)]
    enable_thinking: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum MethodConfig {
    Bm25 {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Bm25Positive {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    OfficialBm25 {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Recent {
        id: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    FullHistory {
        id: String,
    },
    NoMemory {
        id: String,
    },
    OracleTurn {
        id: String,
    },
    OracleSession {
        id: String,
    },
    Strategy {
        id: String,
        config: PathBuf,
        #[serde(default)]
        execution: StrategyExecution,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default)]
        expose_stages: Vec<String>,
        #[serde(default = "default_stage_context_limit")]
        stage_context_limit: usize,
    },
}

impl MethodConfig {
    fn id(&self) -> &str {
        match self {
            Self::Bm25 { id, .. }
            | Self::Bm25Positive { id, .. }
            | Self::OfficialBm25 { id, .. }
            | Self::Recent { id, .. }
            | Self::FullHistory { id }
            | Self::NoMemory { id }
            | Self::OracleTurn { id }
            | Self::OracleSession { id }
            | Self::Strategy { id, .. } => id,
        }
    }

    fn report_role(&self) -> ReportRole {
        match self {
            Self::Bm25 { .. } | Self::OfficialBm25 { .. } | Self::Recent { .. } => {
                ReportRole::DeployableBaseline
            }
            Self::Bm25Positive { .. } => ReportRole::Ablation,
            Self::FullHistory { .. } | Self::NoMemory { .. } => ReportRole::Control,
            Self::OracleTurn { .. } | Self::OracleSession { .. } => ReportRole::Oracle,
            Self::Strategy { .. } => ReportRole::Candidate,
        }
    }

    fn metadata_kind(&self) -> &'static str {
        match self {
            Self::Bm25 { .. } => "bm25_current",
            Self::Bm25Positive { .. } => "bm25_positive",
            Self::OfficialBm25 { .. } => "longmemeval_official_bm25",
            Self::Recent { .. } => "recent",
            Self::FullHistory { .. } => "full_history",
            Self::NoMemory { .. } => "no_memory",
            Self::OracleTurn { .. } => "oracle_turn_evaluator_upper_bound",
            Self::OracleSession { .. } => "oracle_session_evaluator_upper_bound",
            Self::Strategy { .. } => "production_strategy",
        }
    }
}

fn default_limit() -> usize {
    50
}

fn default_stage_context_limit() -> usize {
    50
}

#[derive(Debug, Clone)]
enum ResolvedMethod {
    ModelFree(MethodConfig),
    Strategy {
        id: String,
        config: Config,
        config_sha256: String,
        execution: StrategyExecution,
        limit: usize,
        expose_stages: Vec<String>,
        stage_context_limit: usize,
    },
}

impl ResolvedMethod {
    fn id(&self) -> &str {
        match self {
            Self::ModelFree(method) => method.id(),
            Self::Strategy { id, .. } => id,
        }
    }
}

/// Per-question long-form result written by the native runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionMethodRecord {
    /// Output schema version.
    pub schema_version: u32,
    /// Reproducible run identity.
    pub run_key: String,
    /// Stable method or stage identifier.
    pub method: String,
    /// Display role for curated aggregate comparisons.
    ///
    /// Version-one journals omit this field; aggregation infers their role
    /// fail-closed from the runner-owned `metadata.kind` value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report_role: Option<ReportRole>,
    /// Official evaluator-side question identifier.
    pub question_id: String,
    /// Dataset question category.
    pub question_type: String,
    /// Whether this is an abstention question.
    pub is_abstention: bool,
    /// Retrieval profile for this run.
    pub retrieval_profile: RetrievalProfile,
    /// Whether this question contributes retrieval metrics.
    pub retrieval_eligible: bool,
    /// Ranked message metrics, absent for answer-only abstention records.
    pub message_ranking: Option<RankingMetrics>,
    /// Ranked unique-session metrics.
    pub session_ranking: Option<RankingMetrics>,
    /// Unordered neighbor-expansion evidence coverage.
    pub expanded_coverage: Option<EvidenceCoverage>,
    /// Unordered post-construction evidence retention.
    pub retained_coverage: Option<EvidenceCoverage>,
    /// Ranked opaque message identifiers for audit/rescoring.
    pub ranked_message_ids: Vec<String>,
    /// Expanded opaque message identifiers.
    pub covered_message_ids: Vec<String>,
    /// Retained opaque message identifiers.
    pub retained_message_ids: Vec<String>,
    /// Exact memory supplied to the answer model in full mode.
    pub context: Option<String>,
    /// Model answer in full mode.
    pub prediction: Option<String>,
    /// Official LongMemEval LJ score in full mode.
    pub lj_score: Option<f64>,
    /// Raw judge response in full mode.
    pub judge_raw: Option<String>,
    /// Stage latency and token usage.
    pub efficiency: Efficiency,
    /// Secret-free method trace.
    pub metadata: Map<String, Value>,
}

/// Coverage metrics for an unordered evidence set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCoverage {
    /// At least one gold evidence item is present.
    pub recall_any: f64,
    /// Every gold evidence item is present.
    pub recall_all: f64,
    /// Fraction of gold evidence items present.
    pub mean_recall: f64,
    /// Number of unique covered identifiers.
    pub covered_count: usize,
}

/// Per-question efficiency measurements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Efficiency {
    /// Ingest/index/query/model construction wall time.
    pub memory_total_s: f64,
    /// Retrieval stage reported by the method.
    pub retrieval_s: f64,
    /// Window/processor stage reported by the method.
    pub processing_s: f64,
    /// Answer generation wall time.
    pub answer_s: f64,
    /// Judge wall time.
    pub judge_s: f64,
    /// Answer prompt tokens returned by the provider.
    pub answer_prompt_tokens: Option<u64>,
    /// Answer completion tokens returned by the provider.
    pub answer_completion_tokens: Option<u64>,
    /// Judge prompt tokens returned by the provider.
    pub judge_prompt_tokens: Option<u64>,
    /// Judge completion tokens returned by the provider.
    pub judge_completion_tokens: Option<u64>,
}

/// Complete in-memory artifact returned for persistence by the CLI.
#[derive(Debug, Serialize)]
pub struct SuiteArtifacts {
    /// Reproducibility and completeness manifest.
    pub manifest: Value,
    /// Long-form per-question records.
    pub records: Vec<QuestionMethodRecord>,
    /// Fail-closed aggregate comparisons.
    pub summary: Value,
}

#[derive(Debug)]
struct MethodOutput {
    method: String,
    report_role: ReportRole,
    ranking: Vec<String>,
    covered: Vec<String>,
    retained: Vec<String>,
    context: String,
    memory_kind: AnswerMemoryKind,
    efficiency: Efficiency,
    metadata: Map<String, Value>,
}

/// Executes one complete native LongMemEval suite.
pub async fn run_longmemeval_suite(options: &SuiteRunOptions) -> Result<SuiteArtifacts> {
    let suite_bytes = fs::read(&options.suite_path)
        .with_context(|| format!("read suite {}", options.suite_path.display()))?;
    let suite = load_suite(&options.suite_path)?;
    validate_suite(&suite, options.mode)?;
    let suite_dir = options
        .suite_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let methods = resolve_methods(&suite, suite_dir)?;

    let split_filter = options
        .split_manifest
        .as_ref()
        .map(|path| {
            let file = fs::File::open(path)
                .with_context(|| format!("open split manifest {}", path.display()))?;
            SplitIdFilter::from_manifest_reader(file, &options.split_name)
        })
        .transpose()?;
    let questions = load_longmemeval(&options.dataset_path, split_filter.as_ref())?;
    ensure!(
        !questions.is_empty(),
        "benchmark selection contains no questions"
    );

    let dataset_sha256 = sha256_file(&options.dataset_path)?;
    let split_sha256 = options
        .split_manifest
        .as_ref()
        .map(sha256_file)
        .transpose()?;
    let cargo_lock_sha256 = cargo_lock_sha256();
    let benchmark_code_sha256 = benchmark_code_sha256();
    let run_key = run_key(
        &suite_bytes,
        &dataset_sha256,
        split_sha256.as_deref(),
        options.mode,
        &methods,
        &cargo_lock_sha256,
        &benchmark_code_sha256,
    );

    let record_kind = match options.mode {
        BenchmarkMode::Retrieval => "retrieval",
        BenchmarkMode::Full => "full",
    };
    let mut journal = options
        .output_dir
        .as_ref()
        .map(|directory| {
            JsonlJournal::<QuestionMethodRecord>::open(
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
    for question in &questions {
        let gold = retrieval_gold(question, suite.retrieval_profile);
        if options.mode == BenchmarkMode::Retrieval && !gold.eligible {
            continue;
        }
        let corpus = projected_corpus(&question.input, suite.retrieval_profile);
        for method in &methods {
            let expected_outputs = method_output_ids(method);
            if journal.as_ref().is_some_and(|journal| {
                expected_outputs.iter().all(|method_id| {
                    journal.completed_keys().contains(&JournalRecordKey {
                        run_key: run_key.clone(),
                        method: method_id.clone(),
                        question_id: question.evaluator.source_question_id.clone(),
                        record_kind: record_kind.into(),
                    })
                })
            }) {
                continue;
            }
            let outputs = execute_method(method, question, &corpus, gold)
                .await
                .with_context(|| {
                    format!(
                        "execute method {} on question {}",
                        method.id(),
                        question.evaluator.source_question_id
                    )
                })?;
            for output in outputs {
                let key = JournalRecordKey {
                    run_key: run_key.clone(),
                    method: output.method.clone(),
                    question_id: question.evaluator.source_question_id.clone(),
                    record_kind: record_kind.into(),
                };
                if journal
                    .as_ref()
                    .is_some_and(|journal| journal.completed_keys().contains(&key))
                {
                    continue;
                }
                let record = record_output(
                    &run_key,
                    &suite,
                    options.mode,
                    question,
                    &corpus,
                    gold,
                    output,
                )
                .await?;
                if let Some(journal) = &mut journal {
                    journal.append(JournalRecord {
                        run_key: run_key.clone(),
                        method: record.method.clone(),
                        question_id: record.question_id.clone(),
                        record_kind: record_kind.into(),
                        payload: record.clone(),
                    })?;
                }
                records.push(record);
            }
        }
    }
    ensure!(!records.is_empty(), "benchmark produced no records");
    validate_record_completeness(&records, &suite, options.mode, &questions, &methods)?;
    let summary = build_summary_from_records(
        &suite.name,
        &suite.baseline_method,
        options.mode,
        suite.retrieval_profile,
        &records,
    )?;
    let methods_manifest = method_manifest(&methods);
    ensure!(
        summary["methods"] == json!(methods_manifest),
        "configured method manifest differs from record-derived manifest"
    );
    let manifest = json!({
        "schema_version": 1,
        "run_key": run_key,
        "suite": suite.name,
        "mode": options.mode,
        "retrieval_profile": suite.retrieval_profile,
        "paper_comparable": false,
        "dataset": {
            "path": options.dataset_path,
            "sha256": dataset_sha256,
            "selected_questions": questions.len(),
        },
        "split": {
            "path": options.split_manifest,
            "name": options.split_name,
            "sha256": split_sha256,
        },
        "denominators": {
            "answer_eval_n": if options.mode == BenchmarkMode::Full { questions.len() } else { 0 },
            "retrieval_eval_n": questions.iter().filter(|question| retrieval_gold(question, suite.retrieval_profile).eligible).count(),
        },
        "implementation": {
            "crate_version": env!("CARGO_PKG_VERSION"),
            "cargo_lock_sha256": cargo_lock_sha256,
            "benchmark_code_sha256": benchmark_code_sha256,
        },
        "methods": methods_manifest,
        "limitations": [
            "LongMemEval public reconstruction; not the paper's private split",
            "official_user_turn and all_role_message are separate retrieval protocols",
            "cold replay rebuilds an isolated store per question and is not online warm-query latency"
        ],
        "credential_handling": "only environment-variable names are configured; values are never serialized",
    });
    let artifacts = SuiteArtifacts {
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
        .with_context(|| format!("parse suite TOML {}", path.display()))?
        .try_deserialize()
        .with_context(|| format!("deserialize suite TOML {}", path.display()))
}

fn validate_suite(suite: &SuiteConfig, mode: BenchmarkMode) -> Result<()> {
    ensure!(
        suite.schema_version == SUITE_SCHEMA_VERSION,
        "unsupported suite schema_version {}",
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
    ensure!(!suite.methods.is_empty(), "suite must configure methods");
    let mut ids = HashSet::<String>::new();
    for method in &suite.methods {
        ensure!(
            !method.id().trim().is_empty(),
            "method id must be non-empty"
        );
        ensure!(
            ids.insert(method.id().to_owned()),
            "duplicate method id {}",
            method.id()
        );
        match method {
            MethodConfig::Bm25 { limit, .. }
            | MethodConfig::Bm25Positive { limit, .. }
            | MethodConfig::OfficialBm25 { limit, .. }
            | MethodConfig::Recent { limit, .. }
            | MethodConfig::Strategy { limit, .. } => {
                ensure!(*limit > 0, "method {} limit must be positive", method.id());
            }
            _ => {}
        }
        if let MethodConfig::Strategy {
            execution,
            expose_stages,
            stage_context_limit,
            ..
        } = method
        {
            ensure!(
                mode != BenchmarkMode::Full || *execution == StrategyExecution::FullMemory,
                "strategy {} uses execution=retrieval_only, which cannot run in full mode",
                method.id()
            );
            ensure!(
                *stage_context_limit > 0,
                "strategy {} stage_context_limit must be positive",
                method.id()
            );
            let mut seen = HashSet::new();
            for stage in expose_stages {
                ensure!(
                    matches!(stage.as_str(), "dense" | "bm25" | "rrf" | "retrieved"),
                    "strategy {} exposes unknown ranked stage {stage}",
                    method.id()
                );
                ensure!(
                    seen.insert(stage),
                    "strategy {} exposes duplicate stage {stage}",
                    method.id()
                );
                let virtual_id = format!("{}::{stage}", method.id());
                ensure!(ids.insert(virtual_id), "duplicate expanded method id");
            }
        }
    }
    ensure!(
        ids.contains(&suite.baseline_method),
        "baseline_method {} is not a configured method or exposed stage",
        suite.baseline_method
    );
    let baseline_role = configured_output_role(&suite.methods, &suite.baseline_method)
        .context("resolve configured baseline report role")?;
    ensure!(
        baseline_role == ReportRole::DeployableBaseline,
        "baseline_method {} must have report role deployable_baseline, found {:?}",
        suite.baseline_method,
        baseline_role
    );
    if mode == BenchmarkMode::Full {
        ensure!(
            suite.answerer.is_some() && suite.judge.is_some(),
            "full mode requires both [answerer] and [judge]"
        );
    }
    for (label, model) in [("answerer", &suite.answerer), ("judge", &suite.judge)] {
        if let Some(model) = model {
            model.endpoint.validate(label)?;
            ensure!(model.max_tokens > 0, "{label}.max_tokens must be positive");
            let temperature = model.temperature.unwrap_or(0.0);
            let top_p = model.top_p.unwrap_or(1.0);
            ensure!(
                temperature.is_finite() && temperature >= 0.0,
                "{label}.temperature must be finite and non-negative"
            );
            ensure!(
                top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
                "{label}.top_p must be in (0, 1]"
            );
        }
    }
    Ok(())
}

fn configured_output_role(methods: &[MethodConfig], output_id: &str) -> Option<ReportRole> {
    for method in methods {
        if method.id() == output_id {
            return Some(method.report_role());
        }
        if let MethodConfig::Strategy {
            id, expose_stages, ..
        } = method
            && expose_stages
                .iter()
                .any(|stage| output_id == format!("{id}::{stage}"))
        {
            return Some(ReportRole::Ablation);
        }
    }
    None
}

fn resolve_methods(suite: &SuiteConfig, suite_dir: &Path) -> Result<Vec<ResolvedMethod>> {
    suite
        .methods
        .iter()
        .map(|method| match method {
            MethodConfig::Strategy {
                id,
                config,
                execution,
                limit,
                expose_stages,
                stage_context_limit,
            } => {
                let path = if config.is_absolute() {
                    config.clone()
                } else {
                    suite_dir.join(config)
                };
                let config_sha256 = sha256_file(&path)?;
                let config = semafuse_config::load_from_path(&path)?;
                Ok(ResolvedMethod::Strategy {
                    id: id.clone(),
                    config,
                    config_sha256,
                    execution: *execution,
                    limit: *limit,
                    expose_stages: expose_stages.clone(),
                    stage_context_limit: *stage_context_limit,
                })
            }
            _ => Ok(ResolvedMethod::ModelFree(method.clone())),
        })
        .collect()
}

fn retrieval_gold(question: &PreparedQuestion, profile: RetrievalProfile) -> &RetrievalProfileGold {
    match profile {
        RetrievalProfile::OfficialUserTurn => &question.evaluator.official_user_turn,
        RetrievalProfile::AllRoleMessage => &question.evaluator.all_role_non_abstention,
    }
}

fn projected_corpus(input: &BenchmarkInput, profile: RetrievalProfile) -> Vec<BaselineMessage> {
    input
        .messages
        .iter()
        .filter(|message| {
            profile == RetrievalProfile::AllRoleMessage || message.role.eq_ignore_ascii_case("user")
        })
        .map(|message| BaselineMessage {
            id: message.id.clone(),
            role: message.role.clone(),
            content: message.content.clone(),
            session_id: message.session_id.clone(),
            corpus_index: message.index,
        })
        .collect()
}

async fn execute_method(
    method: &ResolvedMethod,
    question: &PreparedQuestion,
    corpus: &[BaselineMessage],
    gold: &RetrievalProfileGold,
) -> Result<Vec<MethodOutput>> {
    match method {
        ResolvedMethod::ModelFree(method) => {
            let started = Instant::now();
            let hits = match method {
                MethodConfig::Bm25 { limit, .. } => bm25(corpus, &question.input.query, *limit)?,
                MethodConfig::Bm25Positive { limit, .. } => {
                    bm25_positive(corpus, &question.input.query, *limit)?
                }
                MethodConfig::OfficialBm25 { limit, .. } => {
                    longmemeval_official_bm25(corpus, &question.input.query, *limit)?
                }
                MethodConfig::Recent { limit, .. } => recent(corpus, *limit)?,
                MethodConfig::FullHistory { .. } => full_history(corpus)?,
                MethodConfig::NoMemory { .. } => Vec::new(),
                MethodConfig::OracleTurn { .. } => oracle_turn(corpus, &gold.gold_message_ids)?,
                MethodConfig::OracleSession { .. } => {
                    oracle_session(corpus, &gold.gold_session_ids)?
                }
                MethodConfig::Strategy { .. } => unreachable!("resolved separately"),
            };
            let ranking = hit_ids(&hits);
            let context = render_ranked_context(corpus, &ranking)?;
            Ok(vec![MethodOutput {
                method: method.id().to_owned(),
                report_role: method.report_role(),
                ranking: ranking.clone(),
                covered: ranking.clone(),
                retained: ranking,
                context,
                memory_kind: AnswerMemoryKind::Chronological,
                efficiency: Efficiency {
                    memory_total_s: started.elapsed().as_secs_f64(),
                    retrieval_s: started.elapsed().as_secs_f64(),
                    ..Efficiency::default()
                },
                metadata: Map::from_iter([("kind".into(), json!(method.metadata_kind()))]),
            }])
        }
        ResolvedMethod::Strategy {
            id,
            config,
            config_sha256,
            execution,
            limit,
            expose_stages,
            stage_context_limit,
        } => {
            let runner_question = strategy_question(&question.input, corpus);
            let started = Instant::now();
            let response = match execution {
                StrategyExecution::FullMemory => {
                    run_strategy_question(config, runner_question, *limit).await
                }
                StrategyExecution::RetrievalOnly => {
                    run_strategy_retrieval_question(config, runner_question, *limit).await
                }
            }
            .with_context(|| {
                format!(
                    "run strategy method {id} ({execution:?}) on {}",
                    question.evaluator.source_question_id
                )
            })?;
            let memory_total_s = started.elapsed().as_secs_f64();
            let main_context =
                if *execution == StrategyExecution::FullMemory && response.strategy == "lazymem" {
                    response.context.clone()
                } else {
                    render_ranked_context(corpus, &response.retrieved_message_ids)?
                };
            let mut outputs = vec![MethodOutput {
                method: id.clone(),
                report_role: ReportRole::Candidate,
                ranking: response.retrieved_message_ids.clone(),
                covered: response.covered_message_ids.clone(),
                retained: response.retained_message_ids.clone(),
                context: main_context,
                memory_kind: if *execution == StrategyExecution::FullMemory
                    && response.strategy == "lazymem"
                {
                    AnswerMemoryKind::Compressed
                } else {
                    AnswerMemoryKind::Chronological
                },
                efficiency: Efficiency {
                    memory_total_s,
                    retrieval_s: response.retrieval_latency_s,
                    processing_s: response.processing_latency_s,
                    ..Efficiency::default()
                },
                metadata: Map::from_iter([
                    ("kind".into(), json!("production_strategy")),
                    ("strategy".into(), json!(response.strategy)),
                    ("execution".into(), json!(execution)),
                    ("config_sha256".into(), json!(config_sha256)),
                    ("trace".into(), Value::Object(response.metadata.clone())),
                ]),
            }];
            for stage in expose_stages {
                let ranking = response.stage_rankings.get(stage).with_context(|| {
                    format!("strategy method {id} did not emit requested stage {stage}")
                })?;
                let ranking = ranking
                    .iter()
                    .take(*stage_context_limit)
                    .cloned()
                    .collect::<Vec<_>>();
                outputs.push(MethodOutput {
                    method: format!("{id}::{stage}"),
                    report_role: ReportRole::Ablation,
                    covered: ranking.clone(),
                    retained: ranking.clone(),
                    context: render_ranked_context(corpus, &ranking)?,
                    ranking,
                    memory_kind: AnswerMemoryKind::Chronological,
                    efficiency: Efficiency {
                        memory_total_s,
                        retrieval_s: response.retrieval_latency_s,
                        ..Efficiency::default()
                    },
                    metadata: Map::from_iter([
                        ("kind".into(), json!("shared_strategy_stage")),
                        ("source_method".into(), json!(id)),
                        ("stage".into(), json!(stage)),
                        ("execution".into(), json!(execution)),
                        ("shared_model_calls".into(), json!(true)),
                        ("config_sha256".into(), json!(config_sha256)),
                    ]),
                });
            }
            Ok(outputs)
        }
    }
}

fn hit_ids(hits: &[BaselineHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.message_id.clone()).collect()
}

fn strategy_question(input: &BenchmarkInput, corpus: &[BaselineMessage]) -> BenchmarkQuestion {
    let selected = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    let messages = input
        .messages
        .iter()
        .filter(|message| selected.contains(message.id.as_str()))
        .enumerate()
        .map(|(index, message)| BenchmarkMessage {
            id: message.id.clone(),
            index: Some(index),
            role: message.role.clone(),
            content: message.content.clone(),
            timestamp: message.timestamp.clone(),
            session_id: message.session_id.clone(),
        })
        .collect();
    BenchmarkQuestion {
        id: input.id.clone(),
        dataset: input.dataset.clone(),
        question_type: String::new(),
        query: input.query.clone(),
        query_time: input.query_time.clone(),
        messages,
    }
}

fn render_ranked_context(corpus: &[BaselineMessage], ranking: &[String]) -> Result<String> {
    let rank_set = ranking.iter().map(String::as_str).collect::<HashSet<_>>();
    ensure!(
        rank_set.len() == ranking.len(),
        "ranked context contains duplicate message IDs"
    );
    let by_id = corpus
        .iter()
        .map(|message| (message.id.as_str(), message))
        .collect::<HashMap<_, _>>();
    for id in &rank_set {
        ensure!(
            by_id.contains_key(id),
            "ranking contains unknown message ID {id}"
        );
    }
    let mut selected = corpus
        .iter()
        .filter(|message| rank_set.contains(message.id.as_str()))
        .collect::<Vec<_>>();
    selected.sort_by_key(|message| message.corpus_index);
    Ok(selected
        .into_iter()
        .map(|message| {
            format!(
                "[message_id={} role={} session_id={}]\n{}",
                message.id, message.role, message.session_id, message.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n"))
}

async fn record_output(
    run_key: &str,
    suite: &SuiteConfig,
    mode: BenchmarkMode,
    question: &PreparedQuestion,
    corpus: &[BaselineMessage],
    gold: &RetrievalProfileGold,
    mut output: MethodOutput,
) -> Result<QuestionMethodRecord> {
    let (message_ranking, session_ranking, expanded_coverage, retained_coverage) = if gold.eligible
    {
        let message_ranking = evaluate_ranking(&output.ranking, &gold.gold_message_ids)
            .context("score message ranking")?;
        let session_ids = message_ranking_to_sessions(corpus, &output.ranking)?;
        let session_ranking = evaluate_ranking(&session_ids, &gold.gold_session_ids)
            .context("score session ranking")?;
        (
            Some(message_ranking),
            Some(session_ranking),
            Some(evidence_coverage(&output.covered, &gold.gold_message_ids)?),
            Some(evidence_coverage(&output.retained, &gold.gold_message_ids)?),
        )
    } else {
        (None, None, None, None)
    };

    let mut prediction = None;
    let mut lj_score = None;
    let mut judge_raw = None;
    if mode == BenchmarkMode::Full {
        let answerer = suite.answerer.as_ref().context("full mode answerer")?;
        let judge = suite.judge.as_ref().context("full mode judge")?;
        let answer_prompt = render_answer_prompt(&AnswerPromptInput {
            question: question.input.query.clone(),
            query_time: question.input.query_time.clone(),
            memory: output.context.clone(),
            memory_kind: output.memory_kind,
        });
        let answer_started = Instant::now();
        let answer = answerer
            .endpoint
            .chat(
                &answer_prompt.system,
                &answer_prompt.user,
                answerer.options(),
            )
            .await
            .with_context(|| {
                format!(
                    "answer {} with method {}",
                    question.evaluator.source_question_id, output.method
                )
            })?;
        output.efficiency.answer_s = answer_started.elapsed().as_secs_f64();
        output.efficiency.answer_prompt_tokens = answer.prompt_tokens;
        output.efficiency.answer_completion_tokens = answer.completion_tokens;
        let question_type = question
            .evaluator
            .question_type
            .parse::<LongMemEvalQuestionType>()?;
        let prediction_record = PredictionRecord::new(
            question.evaluator.source_question_id.clone(),
            question_type,
            question.evaluator.is_abstention,
            answer.content.trim(),
        );
        let reference = answer_text(&question.evaluator.answer)?;
        let judge_prompt =
            render_judge_prompt(&prediction_record, &question.input.query, &reference);
        let judge_started = Instant::now();
        let judged = judge
            .endpoint
            .chat_user_only(&judge_prompt.text, judge.options())
            .await
            .with_context(|| {
                format!(
                    "judge {} for method {}",
                    question.evaluator.source_question_id, output.method
                )
            })?;
        output.efficiency.judge_s = judge_started.elapsed().as_secs_f64();
        output.efficiency.judge_prompt_tokens = judged.prompt_tokens;
        output.efficiency.judge_completion_tokens = judged.completion_tokens;
        let score = score_judge_response(
            question.evaluator.source_question_id.clone(),
            judge_prompt.kind,
            judged.content.clone(),
        );
        output.metadata.insert(
            "answer_response_model".into(),
            answer.model.map_or(Value::Null, Value::String),
        );
        output.metadata.insert(
            "judge_response_model".into(),
            judged.model.map_or(Value::Null, Value::String),
        );
        prediction = Some(prediction_record.prediction);
        lj_score = Some(score.score);
        judge_raw = Some(score.raw_response);
    }

    Ok(QuestionMethodRecord {
        schema_version: RECORD_SCHEMA_VERSION,
        run_key: run_key.to_owned(),
        method: output.method,
        report_role: Some(output.report_role),
        question_id: question.evaluator.source_question_id.clone(),
        question_type: question.evaluator.question_type.clone(),
        is_abstention: question.evaluator.is_abstention,
        retrieval_profile: suite.retrieval_profile,
        retrieval_eligible: gold.eligible,
        message_ranking,
        session_ranking,
        expanded_coverage,
        retained_coverage,
        ranked_message_ids: output.ranking,
        covered_message_ids: output.covered,
        retained_message_ids: output.retained,
        context: (mode == BenchmarkMode::Full).then_some(output.context),
        prediction,
        lj_score,
        judge_raw,
        efficiency: output.efficiency,
        metadata: output.metadata,
    })
}

impl ChatModelConfig {
    fn options(&self) -> GenerationOptions {
        GenerationOptions {
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
        }
    }
}

fn answer_text(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        _ => serde_json::to_string(value).context("serialize LongMemEval reference answer"),
    }
}

fn message_ranking_to_sessions(
    corpus: &[BaselineMessage],
    ranking: &[String],
) -> Result<Vec<String>> {
    let sessions = corpus
        .iter()
        .map(|message| (message.id.as_str(), message.session_id.as_str()))
        .collect::<HashMap<_, _>>();
    let mut ranked_sessions = Vec::new();
    let mut seen = HashSet::new();
    for message_id in ranking {
        let session_id = sessions
            .get(message_id.as_str())
            .with_context(|| format!("ranking contains unknown message ID {message_id}"))?;
        if seen.insert(*session_id) {
            ranked_sessions.push((*session_id).to_owned());
        }
    }
    Ok(ranked_sessions)
}

fn evidence_coverage(ids: &[String], gold_ids: &[String]) -> Result<EvidenceCoverage> {
    ensure!(!gold_ids.is_empty(), "coverage requires non-empty gold");
    let covered = ids.iter().map(String::as_str).collect::<HashSet<_>>();
    ensure!(
        covered.len() == ids.len(),
        "coverage IDs must not contain duplicates"
    );
    let gold = gold_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    ensure!(
        gold.len() == gold_ids.len(),
        "gold IDs must not contain duplicates"
    );
    let found = gold.intersection(&covered).count();
    Ok(EvidenceCoverage {
        recall_any: f64::from(found > 0),
        recall_all: f64::from(found == gold.len()),
        mean_recall: found as f64 / gold.len() as f64,
        covered_count: covered.len(),
    })
}

fn validate_record_completeness(
    records: &[QuestionMethodRecord],
    suite: &SuiteConfig,
    mode: BenchmarkMode,
    questions: &[PreparedQuestion],
    methods: &[ResolvedMethod],
) -> Result<()> {
    let expected_methods = expanded_method_ids(methods);
    ensure!(
        expected_methods.len() >= 2,
        "comparison suite must produce at least two method IDs"
    );
    ensure!(
        expected_methods.contains(&suite.baseline_method),
        "baseline method {} is absent after method expansion",
        suite.baseline_method
    );
    let expected_questions = questions
        .iter()
        .filter(|question| {
            mode == BenchmarkMode::Full
                || retrieval_gold(question, suite.retrieval_profile).eligible
        })
        .map(|question| question.evaluator.source_question_id.as_str())
        .collect::<HashSet<_>>();
    let mut pairs = HashSet::new();
    let mut actual_by_method = HashMap::<&str, HashSet<&str>>::new();
    let run_key = &records[0].run_key;
    for record in records {
        ensure!(record.run_key == *run_key, "records mix run identities");
        ensure!(
            expected_methods.contains(&record.method),
            "unexpected method record {}",
            record.method
        );
        ensure!(
            pairs.insert((record.method.as_str(), record.question_id.as_str())),
            "duplicate method/question record {} / {}",
            record.method,
            record.question_id
        );
        actual_by_method
            .entry(record.method.as_str())
            .or_default()
            .insert(record.question_id.as_str());
    }
    for method in &expected_methods {
        let actual = actual_by_method
            .get(method.as_str())
            .with_context(|| format!("method {method} produced no records"))?;
        ensure!(
            *actual == expected_questions,
            "method {method} has an incomplete question set (expected {}, got {})",
            expected_questions.len(),
            actual.len()
        );
    }
    if mode == BenchmarkMode::Full {
        for record in records {
            ensure!(
                record.prediction.is_some()
                    && record.lj_score.is_some()
                    && record.judge_raw.is_some(),
                "full record {} / {} lacks answer or judge output",
                record.method,
                record.question_id
            );
        }
    }
    Ok(())
}

fn expanded_method_ids(methods: &[ResolvedMethod]) -> HashSet<String> {
    let mut ids = HashSet::new();
    for method in methods {
        ids.insert(method.id().to_owned());
        if let ResolvedMethod::Strategy {
            id, expose_stages, ..
        } = method
        {
            ids.extend(expose_stages.iter().map(|stage| format!("{id}::{stage}")));
        }
    }
    ids
}

fn method_output_ids(method: &ResolvedMethod) -> Vec<String> {
    let mut ids = vec![method.id().to_owned()];
    if let ResolvedMethod::Strategy {
        id, expose_stages, ..
    } = method
    {
        ids.extend(expose_stages.iter().map(|stage| format!("{id}::{stage}")));
    }
    ids
}

fn method_manifest(methods: &[ResolvedMethod]) -> Vec<Value> {
    let mut outputs = Vec::new();
    for method in methods {
        match method {
            ResolvedMethod::ModelFree(config) => {
                outputs.push(output_manifest_entry(OutputManifestEntry {
                    id: config.id(),
                    source_method: config.id(),
                    role: config.report_role(),
                    shared_execution: false,
                    kind: config.metadata_kind(),
                    config_sha256: None,
                    execution: "model_free",
                    stage: None,
                }))
            }
            ResolvedMethod::Strategy {
                id,
                config_sha256,
                execution,
                expose_stages,
                ..
            } => {
                let execution = strategy_execution_name(*execution);
                outputs.push(output_manifest_entry(OutputManifestEntry {
                    id,
                    source_method: id,
                    role: ReportRole::Candidate,
                    shared_execution: false,
                    kind: "production_strategy",
                    config_sha256: Some(config_sha256),
                    execution,
                    stage: None,
                }));
                for stage in expose_stages {
                    outputs.push(output_manifest_entry(OutputManifestEntry {
                        id: &format!("{id}::{stage}"),
                        source_method: id,
                        role: ReportRole::Ablation,
                        shared_execution: true,
                        kind: "shared_strategy_stage",
                        config_sha256: Some(config_sha256),
                        execution,
                        stage: Some(stage),
                    }));
                }
            }
        }
    }
    outputs.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    outputs
}

fn strategy_execution_name(execution: StrategyExecution) -> &'static str {
    match execution {
        StrategyExecution::FullMemory => "full_memory",
        StrategyExecution::RetrievalOnly => "retrieval_only",
    }
}

struct OutputManifestEntry<'a> {
    id: &'a str,
    source_method: &'a str,
    role: ReportRole,
    shared_execution: bool,
    kind: &'a str,
    config_sha256: Option<&'a str>,
    execution: &'a str,
    stage: Option<&'a str>,
}

fn output_manifest_entry(entry: OutputManifestEntry<'_>) -> Value {
    json!({
        "id": entry.id,
        "source_method": entry.source_method,
        "role": entry.role,
        "shared_execution": entry.shared_execution,
        "kind": entry.kind,
        "config_sha256": entry.config_sha256,
        "execution": entry.execution,
        "stage": entry.stage,
    })
}

fn inferred_report_role(record: &QuestionMethodRecord) -> Result<ReportRole> {
    let kind = record
        .metadata
        .get("kind")
        .and_then(Value::as_str)
        .with_context(|| {
            format!(
                "record {} / {} has no runner-owned metadata.kind",
                record.method, record.question_id
            )
        })?;
    match kind {
        "bm25_current" | "longmemeval_official_bm25" | "recent" => {
            Ok(ReportRole::DeployableBaseline)
        }
        "bm25_positive" | "shared_strategy_stage" => Ok(ReportRole::Ablation),
        "full_history" | "no_memory" => Ok(ReportRole::Control),
        "oracle_turn_evaluator_upper_bound" | "oracle_session_evaluator_upper_bound" => {
            Ok(ReportRole::Oracle)
        }
        "production_strategy" => Ok(ReportRole::Candidate),
        other => anyhow::bail!(
            "record {} / {} has unsupported metadata.kind {other:?}",
            record.method,
            record.question_id
        ),
    }
}

fn resolved_report_role(record: &QuestionMethodRecord) -> Result<ReportRole> {
    ensure!(
        matches!(record.schema_version, 1 | RECORD_SCHEMA_VERSION),
        "record {} / {} has unsupported schema_version {}",
        record.method,
        record.question_id,
        record.schema_version
    );
    ensure!(
        record.schema_version == 1 || record.report_role.is_some(),
        "version-two record {} / {} is missing report_role",
        record.method,
        record.question_id
    );
    let inferred = inferred_report_role(record)?;
    let role = record.report_role.unwrap_or(inferred);
    ensure!(
        role == inferred,
        "record {} / {} report_role {:?} conflicts with metadata.kind-derived {:?}",
        record.method,
        record.question_id,
        role,
        inferred
    );
    Ok(role)
}

fn record_method_manifest(records: &[QuestionMethodRecord]) -> Result<Vec<Value>> {
    let mut outputs = BTreeMap::<String, Value>::new();
    for record in records {
        let role = resolved_report_role(record)?;
        let kind = record.metadata["kind"]
            .as_str()
            .context("validated metadata.kind must be a string")?;
        let (source_method, shared_execution, config_sha256, execution, stage) = match kind {
            "production_strategy" => (
                record.method.as_str(),
                false,
                record.metadata["config_sha256"].as_str(),
                record.metadata["execution"].as_str().with_context(|| {
                    format!("strategy record {} has no execution", record.method)
                })?,
                None,
            ),
            "shared_strategy_stage" => (
                record.metadata["source_method"].as_str().with_context(|| {
                    format!("shared stage record {} has no source_method", record.method)
                })?,
                true,
                record.metadata["config_sha256"].as_str(),
                record.metadata["execution"].as_str().with_context(|| {
                    format!("shared stage record {} has no execution", record.method)
                })?,
                Some(record.metadata["stage"].as_str().with_context(|| {
                    format!("shared stage record {} has no stage", record.method)
                })?),
            ),
            _ => (record.method.as_str(), false, None, "model_free", None),
        };
        let entry = output_manifest_entry(OutputManifestEntry {
            id: &record.method,
            source_method,
            role,
            shared_execution,
            kind,
            config_sha256,
            execution,
            stage,
        });
        if let Some(previous) = outputs.insert(record.method.clone(), entry.clone()) {
            ensure!(
                previous == entry,
                "method {} has inconsistent report metadata across records",
                record.method
            );
        }
    }
    Ok(outputs.into_values().collect())
}

fn is_shared_strategy_stage(record: &QuestionMethodRecord) -> bool {
    record.metadata.get("kind").and_then(Value::as_str) == Some("shared_strategy_stage")
}

/// Rebuilds a curated aggregate summary from persisted per-question records.
///
/// Version-one records without `report_role` are accepted only when their
/// runner-owned `metadata.kind` maps unambiguously to a known role. Controls
/// and oracles are emitted as unpaired diagnostics; deployable baselines,
/// candidates, and ablations are eligible for paired reports.
pub fn build_summary_from_records(
    suite_name: &str,
    baseline_method: &str,
    mode: BenchmarkMode,
    retrieval_profile: RetrievalProfile,
    records: &[QuestionMethodRecord],
) -> Result<Value> {
    ensure!(
        !suite_name.trim().is_empty(),
        "suite name must be non-empty"
    );
    ensure!(
        !baseline_method.trim().is_empty(),
        "baseline method must be non-empty"
    );
    ensure!(!records.is_empty(), "cannot summarize empty records");
    let run_key = &records[0].run_key;
    let mut roles = BTreeMap::<String, ReportRole>::new();
    let mut question_sets = BTreeMap::<String, HashSet<String>>::new();
    let mut pairs = HashSet::new();
    for record in records {
        ensure!(record.run_key == *run_key, "records mix run identities");
        ensure!(
            record.retrieval_profile == retrieval_profile,
            "record {} / {} uses retrieval profile {:?}, expected {:?}",
            record.method,
            record.question_id,
            record.retrieval_profile,
            retrieval_profile
        );
        ensure!(
            pairs.insert((record.method.as_str(), record.question_id.as_str())),
            "duplicate method/question record {} / {}",
            record.method,
            record.question_id
        );
        if mode == BenchmarkMode::Full {
            ensure!(
                record.prediction.is_some()
                    && record.lj_score.is_some()
                    && record.judge_raw.is_some(),
                "full record {} / {} lacks answer or judge output",
                record.method,
                record.question_id
            );
        }
        let role = resolved_report_role(record)?;
        if let Some(previous) = roles.insert(record.method.clone(), role) {
            ensure!(
                previous == role,
                "method {} mixes report roles {:?} and {:?}",
                record.method,
                previous,
                role
            );
        }
        question_sets
            .entry(record.method.clone())
            .or_default()
            .insert(record.question_id.clone());
    }
    let baseline_role = roles
        .get(baseline_method)
        .with_context(|| format!("baseline method {baseline_method:?} is absent"))?;
    ensure!(
        *baseline_role == ReportRole::DeployableBaseline,
        "baseline method {baseline_method:?} must have report role deployable_baseline, found {:?}",
        baseline_role
    );
    let baseline_questions = &question_sets[baseline_method];
    for (method, questions) in &question_sets {
        ensure!(
            questions == baseline_questions,
            "method {method} has a different question set from baseline {baseline_method}"
        );
    }

    let mut rows = BTreeMap::<String, Vec<ScalarMetricRow>>::new();
    for record in records {
        if let (Some(message), Some(session)) = (&record.message_ranking, &record.session_ranking) {
            for (message_cutoff, session_cutoff) in message.cutoffs.iter().zip(&session.cutoffs) {
                ensure!(
                    message_cutoff.cutoff == session_cutoff.cutoff,
                    "message/session cutoff mismatch"
                );
                let cutoff = message_cutoff.cutoff;
                for (scope, metrics) in [("message", message_cutoff), ("session", session_cutoff)] {
                    push_scalar(
                        &mut rows,
                        format!("{scope}_recall_any@{cutoff}"),
                        record,
                        metrics.recall_any,
                    );
                    push_scalar(
                        &mut rows,
                        format!("{scope}_recall_all@{cutoff}"),
                        record,
                        metrics.recall_all,
                    );
                    push_scalar(
                        &mut rows,
                        format!("{scope}_mean_recall@{cutoff}"),
                        record,
                        metrics.mean_recall,
                    );
                    push_scalar(
                        &mut rows,
                        format!("{scope}_ndcg_standard@{cutoff}"),
                        record,
                        metrics.ndcg,
                    );
                    push_scalar(
                        &mut rows,
                        format!("{scope}_ndcg_official@{cutoff}"),
                        record,
                        metrics.official_ndcg_any,
                    );
                    if cutoff == 50 {
                        push_scalar(&mut rows, format!("{scope}_mrr@50"), record, metrics.mrr);
                    }
                }
            }
        }
        if let Some(coverage) = &record.expanded_coverage {
            push_scalar(
                &mut rows,
                "context_recall_any".into(),
                record,
                coverage.recall_any,
            );
            push_scalar(
                &mut rows,
                "context_recall_all".into(),
                record,
                coverage.recall_all,
            );
            push_scalar(
                &mut rows,
                "context_mean_recall".into(),
                record,
                coverage.mean_recall,
            );
        }
        if let Some(score) = record.lj_score {
            push_scalar(&mut rows, "longmemeval_official_lj".into(), record, score);
        }
        if !is_shared_strategy_stage(record) {
            for (metric, value) in [
                ("latency_memory_total_s", record.efficiency.memory_total_s),
                ("latency_retrieval_s", record.efficiency.retrieval_s),
                ("latency_processing_s", record.efficiency.processing_s),
            ] {
                push_scalar(&mut rows, metric.into(), record, value);
            }
        }
        if mode == BenchmarkMode::Full {
            push_scalar(
                &mut rows,
                "latency_answer_s".into(),
                record,
                record.efficiency.answer_s,
            );
            push_scalar(
                &mut rows,
                "latency_judge_s".into(),
                record,
                record.efficiency.judge_s,
            );
        }
    }
    let mut reports = BTreeMap::<String, ScalarBenchmarkReport>::new();
    let mut diagnostics = BTreeMap::<String, Value>::new();
    for (metric, metric_rows) in rows {
        let direction = if metric.starts_with("latency_") {
            MetricDirection::LowerIsBetter
        } else {
            MetricDirection::HigherIsBetter
        };
        let paired_rows = metric_rows
            .iter()
            .filter(|row| roles[&row.method].is_paired())
            .cloned()
            .collect::<Vec<_>>();
        let paired_method_count = paired_rows
            .iter()
            .map(|row| row.method.as_str())
            .collect::<HashSet<_>>()
            .len();
        if paired_method_count >= 2 {
            reports.insert(
                metric.clone(),
                build_scalar_report(&metric, baseline_method, direction, &paired_rows)
                    .with_context(|| format!("build paired report for {metric}"))?,
            );
        }

        let diagnostic_methods = metric_rows
            .iter()
            .filter(|row| roles[&row.method].is_diagnostic())
            .map(|row| row.method.as_str())
            .collect::<HashSet<_>>();
        if !diagnostic_methods.is_empty() {
            let diagnostic_rows = metric_rows
                .iter()
                .filter(|row| {
                    row.method == baseline_method
                        || diagnostic_methods.contains(row.method.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            let aggregate =
                build_scalar_report(&metric, baseline_method, direction, &diagnostic_rows)
                    .with_context(|| format!("build unpaired diagnostics for {metric}"))?;
            let methods = aggregate
                .methods
                .into_iter()
                .filter(|(method, _)| diagnostic_methods.contains(method.as_str()))
                .collect::<BTreeMap<_, _>>();
            diagnostics.insert(
                metric.clone(),
                json!({
                    "schema_version": 1,
                    "metric": metric,
                    "direction": direction,
                    "bootstrap": aggregate.bootstrap,
                    "methods": methods,
                }),
            );
        }
    }
    serde_json::to_value(json!({
        "schema_version": 2,
        "suite": suite_name,
        "mode": mode,
        "retrieval_profile": retrieval_profile,
        "baseline_method": baseline_method,
        "methods": record_method_manifest(records)?,
        "answer_eval_n": if mode == BenchmarkMode::Full {
            records.iter().filter(|record| record.method == baseline_method).count()
        } else { 0 },
        "retrieval_eval_n": records.iter().filter(|record| record.method == baseline_method && record.retrieval_eligible).count(),
        "reports": reports,
        "diagnostics": diagnostics,
    }))
    .context("serialize benchmark summary")
}

fn push_scalar(
    rows: &mut BTreeMap<String, Vec<ScalarMetricRow>>,
    metric: String,
    record: &QuestionMethodRecord,
    value: f64,
) {
    rows.entry(metric).or_default().push(ScalarMetricRow {
        method: record.method.clone(),
        question_id: record.question_id.clone(),
        question_type: record.question_type.clone(),
        value,
    });
}

fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {} for hashing", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn cargo_lock_sha256() -> String {
    hex::encode(Sha256::digest(include_bytes!("../Cargo.lock")))
}

fn benchmark_code_sha256() -> String {
    let mut digest = Sha256::new();
    for (path, bytes) in [
        (
            "src/benchmark_suite.rs",
            include_bytes!("benchmark_suite.rs").as_slice(),
        ),
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
            "src/longmemeval.rs",
            include_bytes!("longmemeval.rs").as_slice(),
        ),
        (
            "src/longmemeval_eval.rs",
            include_bytes!("longmemeval_eval.rs").as_slice(),
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
    split_sha256: Option<&str>,
    mode: BenchmarkMode,
    methods: &[ResolvedMethod],
    cargo_lock_sha256: &str,
    benchmark_code_sha256: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"semafuse-memory-benchmark-run/v1");
    digest.update((suite_bytes.len() as u64).to_be_bytes());
    digest.update(suite_bytes);
    digest.update(dataset_sha256.as_bytes());
    digest.update(split_sha256.unwrap_or("full").as_bytes());
    digest.update(cargo_lock_sha256.as_bytes());
    digest.update(benchmark_code_sha256.as_bytes());
    digest.update(match mode {
        BenchmarkMode::Retrieval => b"retrieval".as_slice(),
        BenchmarkMode::Full => b"full".as_slice(),
    });
    for method in methods {
        digest.update(method.id().as_bytes());
        if let ResolvedMethod::Strategy {
            config_sha256,
            execution,
            ..
        } = method
        {
            digest.update(config_sha256.as_bytes());
            digest.update(match execution {
                StrategyExecution::FullMemory => b"full_memory".as_slice(),
                StrategyExecution::RetrievalOnly => b"retrieval_only".as_slice(),
            });
        }
    }
    hex::encode(digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn aggregate_record(
        method: &str,
        kind: &str,
        report_role: Option<ReportRole>,
    ) -> QuestionMethodRecord {
        let mut metadata = Map::from_iter([("kind".into(), json!(kind))]);
        if kind == "production_strategy" {
            metadata.insert("config_sha256".into(), json!("strategy-config"));
            metadata.insert("execution".into(), json!("retrieval_only"));
        } else if kind == "shared_strategy_stage" {
            metadata.insert("source_method".into(), json!("candidate"));
            metadata.insert("stage".into(), json!("dense"));
            metadata.insert("config_sha256".into(), json!("strategy-config"));
            metadata.insert("execution".into(), json!("retrieval_only"));
        }
        QuestionMethodRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            run_key: "run".into(),
            method: method.into(),
            report_role,
            question_id: "q1".into(),
            question_type: "single-session-user".into(),
            is_abstention: false,
            retrieval_profile: RetrievalProfile::AllRoleMessage,
            retrieval_eligible: true,
            message_ranking: Some(evaluate_ranking(&["gold"], &["gold"]).unwrap()),
            session_ranking: Some(evaluate_ranking(&["session"], &["session"]).unwrap()),
            expanded_coverage: Some(EvidenceCoverage {
                recall_any: 1.0,
                recall_all: 1.0,
                mean_recall: 1.0,
                covered_count: 1,
            }),
            retained_coverage: Some(EvidenceCoverage {
                recall_any: 0.0,
                recall_all: 0.0,
                mean_recall: 0.0,
                covered_count: 0,
            }),
            ranked_message_ids: vec!["gold".into()],
            covered_message_ids: vec!["gold".into()],
            retained_message_ids: Vec::new(),
            context: None,
            prediction: None,
            lj_score: None,
            judge_raw: None,
            efficiency: Efficiency {
                memory_total_s: 3.0,
                retrieval_s: 2.0,
                processing_s: 1.0,
                ..Efficiency::default()
            },
            metadata,
        }
    }

    #[tokio::test]
    async fn model_free_suite_is_complete_scored_and_resumable() {
        let directory = tempdir().unwrap();
        let dataset_path = directory.path().join("longmemeval.json");
        let suite_path = directory.path().join("suite.toml");
        let output_dir = directory.path().join("output");
        let dataset = json!([
            {
                "question_id": "q1",
                "question_type": "single-session-user",
                "question": "What color was selected?",
                "answer": "blue",
                "question_date": "2023/06/01 (Thu) 12:00",
                "haystack_session_ids": ["answer_session"],
                "haystack_dates": ["2023/05/30 (Tue) 23:40"],
                "haystack_sessions": [[
                    {"role": "user", "content": "The selected color was blue.", "has_answer": true},
                    {"role": "assistant", "content": "Noted."}
                ]],
                "answer_session_ids": ["answer_session"]
            },
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "question": "Which fruit was purchased?",
                "answer": "apple",
                "question_date": "2023/06/02 (Fri) 12:00",
                "haystack_session_ids": ["noise", "answer_fruit"],
                "haystack_dates": ["2023/05/30 (Tue) 10:00", "2023/06/01 (Thu) 10:00"],
                "haystack_sessions": [
                    [{"role": "user", "content": "It was sunny."}],
                    [{"role": "assistant", "content": "An apple was purchased.", "has_answer": true}]
                ],
                "answer_session_ids": ["answer_fruit"]
            },
            {
                "question_id": "q3_abs",
                "question_type": "single-session-user",
                "question": "What is unavailable?",
                "answer": "not in history",
                "question_date": "2023/06/03 (Sat) 12:00",
                "haystack_session_ids": ["noise"],
                "haystack_dates": ["2023/06/01 (Thu) 10:00"],
                "haystack_sessions": [[{"role": "user", "content": "No answer here."}]],
                "answer_session_ids": []
            }
        ]);
        fs::write(&dataset_path, serde_json::to_vec(&dataset).unwrap()).unwrap();
        fs::write(
            &suite_path,
            r#"
schema_version = 1
name = "fixture-all-role"
retrieval_profile = "all_role_message"
baseline_method = "bm25"

[[methods]]
kind = "bm25"
id = "bm25"
limit = 50

[[methods]]
kind = "recent"
id = "recent"
limit = 50

[[methods]]
kind = "oracle_turn"
id = "oracle-turn"
"#,
        )
        .unwrap();
        let options = SuiteRunOptions {
            suite_path,
            dataset_path,
            split_manifest: None,
            split_name: "test".into(),
            mode: BenchmarkMode::Retrieval,
            output_dir: Some(output_dir.clone()),
        };

        let first = run_longmemeval_suite(&options).await.unwrap();
        assert_eq!(first.manifest["denominators"]["retrieval_eval_n"], 2);
        assert_eq!(first.records.len(), 6);
        assert!(first.records.iter().all(|record| !record.is_abstention));
        assert!(
            first
                .records
                .iter()
                .all(|record| record.schema_version == RECORD_SCHEMA_VERSION
                    && record.report_role.is_some())
        );
        assert_eq!(first.summary["retrieval_eval_n"], Value::Number(2.into()));
        let manifest_methods = first.manifest["methods"].as_array().unwrap();
        assert_eq!(manifest_methods.len(), 3);
        assert_eq!(manifest_methods[0]["id"], "bm25");
        assert_eq!(manifest_methods[0]["role"], "deployable_baseline");
        assert_eq!(manifest_methods[0]["shared_execution"], false);
        assert_eq!(manifest_methods[0]["execution"], "model_free");
        assert!(manifest_methods[0]["config_sha256"].is_null());
        assert_eq!(first.summary["methods"], first.manifest["methods"]);
        let paired = &first.summary["reports"]["message_recall_all@1"]["methods"];
        assert!(paired.get("bm25").is_some());
        assert!(paired.get("recent").is_some());
        assert!(paired.get("oracle-turn").is_none());
        let diagnostics = &first.summary["diagnostics"]["message_recall_all@1"]["methods"];
        assert!(diagnostics.get("oracle-turn").is_some());
        assert!(diagnostics.get("bm25").is_none());
        let report_keys = first.summary["reports"].as_object().unwrap();
        assert!(report_keys.contains_key("context_recall_all"));
        assert!(
            report_keys
                .keys()
                .all(|key| !key.starts_with("expanded_") && !key.starts_with("retained_"))
        );
        assert_eq!(
            first.manifest["implementation"]["cargo_lock_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            first.manifest["implementation"]["benchmark_code_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert!(output_dir.join("per-question.jsonl").is_file());
        assert!(output_dir.join("summary.json").is_file());

        let second = run_longmemeval_suite(&options).await.unwrap();
        assert_eq!(second.records.len(), first.records.len());
        assert_eq!(second.manifest["run_key"], first.manifest["run_key"]);
        assert_eq!(
            fs::read_to_string(output_dir.join("per-question.jsonl"))
                .unwrap()
                .lines()
                .count(),
            6
        );
    }

    #[test]
    fn strategy_execution_defaults_to_full_memory_and_retrieval_only_fails_closed_in_full_mode() {
        let directory = tempdir().unwrap();
        let default_suite_path = directory.path().join("default.toml");
        fs::write(
            &default_suite_path,
            r#"
schema_version = 1
name = "execution-default"
retrieval_profile = "all_role_message"
baseline_method = "bm25"

[[methods]]
kind = "bm25"
id = "bm25"

[[methods]]
kind = "strategy"
id = "semafuse"
config = "strategy.toml"
"#,
        )
        .unwrap();
        let default_suite = load_suite(&default_suite_path).unwrap();
        assert!(matches!(
            &default_suite.methods[1],
            MethodConfig::Strategy {
                execution: StrategyExecution::FullMemory,
                ..
            }
        ));
        validate_suite(&default_suite, BenchmarkMode::Retrieval).unwrap();

        let retrieval_suite_path = directory.path().join("retrieval-only.toml");
        fs::write(
            &retrieval_suite_path,
            r#"
schema_version = 1
name = "execution-retrieval-only"
retrieval_profile = "all_role_message"
baseline_method = "bm25"

[[methods]]
kind = "bm25"
id = "bm25"

[[methods]]
kind = "strategy"
id = "semafuse"
config = "strategy.toml"
execution = "retrieval_only"
expose_stages = ["dense", "rrf"]
"#,
        )
        .unwrap();
        let retrieval_suite = load_suite(&retrieval_suite_path).unwrap();
        assert!(matches!(
            &retrieval_suite.methods[1],
            MethodConfig::Strategy {
                execution: StrategyExecution::RetrievalOnly,
                ..
            }
        ));
        validate_suite(&retrieval_suite, BenchmarkMode::Retrieval).unwrap();
        let error = validate_suite(&retrieval_suite, BenchmarkMode::Full).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("execution=retrieval_only, which cannot run in full mode")
        );
    }

    #[test]
    fn paired_baseline_must_be_a_deployable_baseline() {
        let suite = SuiteConfig {
            schema_version: SUITE_SCHEMA_VERSION,
            name: "invalid-oracle-baseline".into(),
            retrieval_profile: RetrievalProfile::AllRoleMessage,
            baseline_method: "oracle".into(),
            methods: vec![
                MethodConfig::Bm25 {
                    id: "bm25".into(),
                    limit: 50,
                },
                MethodConfig::OracleTurn {
                    id: "oracle".into(),
                },
            ],
            answerer: None,
            judge: None,
        };
        let error = validate_suite(&suite, BenchmarkMode::Retrieval).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must have report role deployable_baseline")
        );
    }

    #[test]
    fn curated_summary_separates_roles_context_and_shared_stage_latency() {
        let records = vec![
            aggregate_record(
                "baseline",
                "bm25_current",
                Some(ReportRole::DeployableBaseline),
            ),
            aggregate_record(
                "candidate",
                "production_strategy",
                Some(ReportRole::Candidate),
            ),
            aggregate_record(
                "candidate::dense",
                "shared_strategy_stage",
                Some(ReportRole::Ablation),
            ),
            aggregate_record("full-history", "full_history", Some(ReportRole::Control)),
            aggregate_record(
                "oracle-turn",
                "oracle_turn_evaluator_upper_bound",
                Some(ReportRole::Oracle),
            ),
        ];
        let summary = build_summary_from_records(
            "suite",
            "baseline",
            BenchmarkMode::Retrieval,
            RetrievalProfile::AllRoleMessage,
            &records,
        )
        .unwrap();

        let ranked_methods = summary["reports"]["message_recall_all@1"]["methods"]
            .as_object()
            .unwrap();
        assert_eq!(ranked_methods.len(), 3);
        assert!(ranked_methods.contains_key("baseline"));
        assert!(ranked_methods.contains_key("candidate"));
        assert!(ranked_methods.contains_key("candidate::dense"));
        assert!(!ranked_methods.contains_key("full-history"));
        assert!(!ranked_methods.contains_key("oracle-turn"));

        let diagnostic_methods = summary["diagnostics"]["message_recall_all@1"]["methods"]
            .as_object()
            .unwrap();
        assert_eq!(diagnostic_methods.len(), 2);
        assert!(diagnostic_methods.contains_key("full-history"));
        assert!(diagnostic_methods.contains_key("oracle-turn"));

        let report_keys = summary["reports"].as_object().unwrap();
        assert!(report_keys.contains_key("context_recall_any"));
        assert!(report_keys.contains_key("context_recall_all"));
        assert!(report_keys.contains_key("context_mean_recall"));
        assert!(
            report_keys
                .keys()
                .all(|key| !key.starts_with("expanded_") && !key.starts_with("retained_"))
        );
        let latency_methods = summary["reports"]["latency_retrieval_s"]["methods"]
            .as_object()
            .unwrap();
        assert!(latency_methods.contains_key("baseline"));
        assert!(latency_methods.contains_key("candidate"));
        assert!(!latency_methods.contains_key("candidate::dense"));

        let methods = summary["methods"].as_array().unwrap();
        let stage = methods
            .iter()
            .find(|method| method["id"] == "candidate::dense")
            .unwrap();
        assert_eq!(stage["role"], "ablation");
        assert_eq!(stage["shared_execution"], true);
        let oracle = methods
            .iter()
            .find(|method| method["id"] == "oracle-turn")
            .unwrap();
        assert_eq!(oracle["role"], "oracle");
        assert_eq!(oracle["shared_execution"], false);
    }

    #[test]
    fn version_one_records_rebuild_by_strict_metadata_kind_inference() {
        let mut baseline = aggregate_record("baseline", "bm25_current", None);
        baseline.schema_version = 1;
        let encoded = serde_json::to_value(&baseline).unwrap();
        assert!(encoded.get("report_role").is_none());
        let baseline: QuestionMethodRecord = serde_json::from_value(encoded).unwrap();

        let mut candidate = aggregate_record("candidate", "production_strategy", None);
        candidate.schema_version = 1;
        let summary = build_summary_from_records(
            "legacy-suite",
            "baseline",
            BenchmarkMode::Retrieval,
            RetrievalProfile::AllRoleMessage,
            &[baseline.clone(), candidate],
        )
        .unwrap();
        assert!(
            summary["reports"]["message_recall_all@1"]["methods"]
                .get("candidate")
                .is_some()
        );

        let mut unknown = baseline;
        unknown.metadata.insert("kind".into(), json!("unknown"));
        let error = build_summary_from_records(
            "legacy-suite",
            "baseline",
            BenchmarkMode::Retrieval,
            RetrievalProfile::AllRoleMessage,
            &[unknown],
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported metadata.kind"));
    }

    #[test]
    fn run_identity_is_bound_to_lock_and_benchmark_code_hashes() {
        let methods = vec![ResolvedMethod::ModelFree(MethodConfig::Bm25 {
            id: "bm25".into(),
            limit: 50,
        })];
        let base = run_key(
            b"suite",
            "dataset",
            None,
            BenchmarkMode::Retrieval,
            &methods,
            "lock-a",
            "code-a",
        );
        let different_lock = run_key(
            b"suite",
            "dataset",
            None,
            BenchmarkMode::Retrieval,
            &methods,
            "lock-b",
            "code-a",
        );
        let different_code = run_key(
            b"suite",
            "dataset",
            None,
            BenchmarkMode::Retrieval,
            &methods,
            "lock-a",
            "code-b",
        );
        assert_ne!(base, different_lock);
        assert_ne!(base, different_code);

        let strategy = |execution| {
            vec![ResolvedMethod::Strategy {
                id: "semafuse".into(),
                config: Config::default(),
                config_sha256: "strategy-config".into(),
                execution,
                limit: 50,
                expose_stages: vec!["dense".into(), "rrf".into()],
                stage_context_limit: 50,
            }]
        };
        let full_memory_methods = strategy(StrategyExecution::FullMemory);
        let retrieval_only_methods = strategy(StrategyExecution::RetrievalOnly);
        let full_memory_key = run_key(
            b"suite",
            "dataset",
            None,
            BenchmarkMode::Retrieval,
            &full_memory_methods,
            "lock",
            "code",
        );
        let retrieval_only_key = run_key(
            b"suite",
            "dataset",
            None,
            BenchmarkMode::Retrieval,
            &retrieval_only_methods,
            "lock",
            "code",
        );
        assert_ne!(full_memory_key, retrieval_only_key);

        let manifest = method_manifest(&retrieval_only_methods);
        assert_eq!(manifest.len(), 3);
        assert_eq!(manifest[0]["id"], "semafuse");
        assert_eq!(manifest[0]["config_sha256"], "strategy-config");
        assert_eq!(manifest[0]["execution"], "retrieval_only");
        assert_eq!(manifest[0]["role"], "candidate");
        assert_eq!(manifest[0]["shared_execution"], false);
        assert_eq!(manifest[1]["id"], "semafuse::dense");
        assert_eq!(manifest[1]["role"], "ablation");
        assert_eq!(manifest[1]["shared_execution"], true);
        assert_eq!(manifest[2]["id"], "semafuse::rrf");
        assert_eq!(manifest[2]["role"], "ablation");
        assert_eq!(manifest[2]["shared_execution"], true);
    }
}
