//! End-to-end LoCoMo memory construction and answer evaluation.
//!
//! This runner complements [`crate::locomo_suite`]'s retrieval-only protocol.
//! It constructs one query-conditioned context per method, sends every context
//! through the same frozen answer model, and applies LoCoMo's released
//! category-aware token-F1 scorer. Expensive memory construction and answer
//! generation use separate append-and-sync journals. Once a complete memory
//! record commits, an interrupted answer run does not rerun its processor.

use crate::{
    benchmark::{BenchmarkResponse, StrategyBenchmarkSession},
    benchmark_baselines::{BaselineHit, Message as BaselineMessage, bm25, recent},
    benchmark_journal::{
        JournalRecord, JournalRecordKey, JsonlJournal, SummaryEnvelope, write_summary_atomic,
    },
    benchmark_metrics::evaluate_ranking,
    benchmark_report::{MetricDirection, ScalarMetricRow, build_scalar_report},
    inference::{GenerationOptions, OpenAiEndpoint},
    locomo::{
        DATA_LICENSE, LocomoTrack, PreparedDataset, PreparedQuestion, UPSTREAM_COMMIT,
        UPSTREAM_DATA_SHA256, load_locomo,
    },
    locomo_answer_protocol::{AnswerPromptInput, LocomoAnswerProtocol},
    locomo_eval::{
        LOCOMO_OFFICIAL_F1_METRIC, LOCOMO_UPSTREAM_COMMIT, LOCOMO_UPSTREAM_SCORER_SHA256,
        LocomoF1Score, score_locomo_answer,
    },
    locomo_suite_common::{
        LocomoQuestionRecord, MethodConfig, ResolvedMethod, SemafuseStage, SuiteConfig,
        method_manifest, projected_corpus, resolve_methods, runner_question, sha256_file,
        validate_method_configs, validate_track_dataset,
    },
};
use anyhow::{Context, Result, bail, ensure};
use config::{Config as LayeredConfig, File};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const SUITE_SCHEMA_VERSION: u32 = 1;
const MEMORY_RECORD_SCHEMA_VERSION: u32 = 1;
const ANSWER_RECORD_SCHEMA_VERSION: u32 = 1;
const MEMORY_RECORD_KIND: &str = "locomo_answer_memory";
const ANSWER_RECORD_KIND: &str = "locomo_answer_prediction";
const SEMAFUSE_MEMORY_KIND: &str = "semafuse_strategy_full";
const SEMAFUSE_FULL_MEMORY_MANIFEST_KIND: &str = "semafuse_full_memory_and_retrieval";
const LABEL_ISOLATION_DECLARATION: &str = "question and history may enter model requests; reference answers and categories never enter any model request and are used only for scheduling, metadata validation, and evaluation";

/// Scheduling and filesystem inputs for an end-to-end LoCoMo run.
#[derive(Debug, Clone)]
pub struct LocomoAnswerSuiteRunOptions {
    /// Modular full-suite TOML path.
    pub suite_path: PathBuf,
    /// Official pinned `data/locomo10.json` path.
    pub dataset_path: PathBuf,
    /// Explicit acknowledgement that this is permitted non-commercial use.
    pub acknowledge_noncommercial: bool,
    /// Crash-safe output directory. Raw contexts remain local and gitignored.
    pub output_dir: Option<PathBuf>,
    /// Optional deterministic pilot size. It changes scheduling, not run identity.
    pub question_limit: Option<usize>,
    /// Optional exact opaque question IDs for targeted failure reproduction. It
    /// changes scheduling, not run identity, and is mutually exclusive with a
    /// stratified question limit.
    pub question_ids: BTreeSet<String>,
    /// Optional source-method filter used for a cheap pilot. Semafuse stages are
    /// scheduled with their source method.
    pub methods: BTreeSet<String>,
    /// Resolve and validate every identity without constructing memory or calling models.
    pub validate_only: bool,
}

/// Complete or partial artifacts returned by [`run_locomo_answer_suite`].
#[derive(Debug, Serialize)]
pub struct LocomoAnswerSuiteArtifacts {
    /// Secret-free protocol and implementation identity.
    pub manifest: Value,
    /// Raw local memory contexts. Do not publish without reviewing the dataset license.
    pub memory_records: Vec<LocomoMemoryRecord>,
    /// Predictions and official per-question F1 values.
    pub answer_records: Vec<LocomoAnswerRecord>,
    /// Complete paired report, or an explicitly partial progress summary.
    pub summary: Value,
    /// Whether every configured output covers its declared track cohort.
    pub complete: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnswerSuiteConfig {
    schema_version: u32,
    name: String,
    #[serde(default)]
    track: LocomoTrack,
    baseline_method: String,
    methods: Vec<MethodConfig>,
    #[serde(default)]
    imported_rankings: Vec<ImportedRankingConfig>,
    /// Previously committed full memory contexts that may be replayed through
    /// a different common answer protocol without rebuilding the memory.
    #[serde(default)]
    imported_memories: Vec<ImportedMemoryConfig>,
    answerer: AnswerModelConfig,
    pricing: PricingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedRankingConfig {
    id: String,
    source_method: String,
    journal: PathBuf,
    journal_sha256: String,
    summary: PathBuf,
    summary_sha256: String,
    source_run_key: String,
    limit: usize,
    cohort: ImportedCohort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ImportedCohort {
    RetrievalEligible,
}

#[derive(Debug)]
struct ResolvedImportedRanking {
    id: String,
    source_method: String,
    source_run_key: String,
    journal_sha256: String,
    summary_sha256: String,
    limit: usize,
    cohort: ImportedCohort,
    rankings: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedMemoryConfig {
    /// Destination method identifier in the new answer run.
    id: String,
    /// Exact method identifier selected from the source memory journal.
    source_method: String,
    journal: PathBuf,
    journal_sha256: String,
    /// Source full-answer summary whose manifest proves the journal's
    /// dataset, memory-run, method kind, and label-isolation identity.
    summary: PathBuf,
    summary_sha256: String,
    source_run_key: String,
    cohort: QuestionCohort,
}

#[derive(Debug, Clone)]
struct ResolvedImportedMemory {
    id: String,
    source_method: String,
    source_run_key: String,
    journal_sha256: String,
    summary_sha256: String,
    source_kind: String,
    source_label_isolation_proven: bool,
    cohort: QuestionCohort,
    records: BTreeMap<String, LocomoMemoryRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum QuestionCohort {
    AllQa,
    RetrievalEligible,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AnswerModelConfig {
    endpoint: OpenAiEndpoint,
    expected_provider: String,
    /// Frozen prompt, response contract, and answer extraction protocol. The
    /// default preserves the free-text reader used by v3 benchmark configs.
    #[serde(default)]
    protocol: LocomoAnswerProtocol,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    max_tokens: usize,
    #[serde(default)]
    enable_thinking: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PricingConfig {
    input_usd_per_million: f64,
    output_usd_per_million: f64,
    /// Stops before the next question-level processor operation or answer call
    /// once the local estimate reaches this value. One processor operation can
    /// contain concurrent responses; transport retries and retrieval costs are
    /// outside the estimate.
    max_reported_usage_usd: f64,
}

impl AnswerModelConfig {
    fn validate(&self) -> Result<()> {
        let options = self.options();
        self.endpoint.validate_generation("answerer", &options)?;
        let provider = self.expected_provider.as_bytes();
        ensure!(
            !provider.is_empty()
                && provider.len() <= 64
                && provider.first().is_some_and(u8::is_ascii_alphanumeric)
                && provider.last().is_some_and(u8::is_ascii_alphanumeric)
                && provider.iter().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b' ' | b'.' | b'_' | b'+' | b'-')
                }),
            "answerer.expected_provider must be a safe provider name of at most 64 ASCII bytes"
        );
        ensure!(
            options.temperature.is_finite() && options.temperature >= 0.0,
            "answerer.temperature must be finite and non-negative"
        );
        ensure!(
            options.top_p.is_finite() && options.top_p > 0.0 && options.top_p <= 1.0,
            "answerer.top_p must be in (0, 1]"
        );
        ensure!(
            (1..=4096).contains(&options.max_tokens),
            "answerer.max_tokens must be between 1 and 4096"
        );
        Ok(())
    }

    fn options(&self) -> GenerationOptions {
        GenerationOptions {
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(1.0),
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
        }
    }
}

impl PricingConfig {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("input_usd_per_million", self.input_usd_per_million),
            ("output_usd_per_million", self.output_usd_per_million),
            ("max_reported_usage_usd", self.max_reported_usage_usd),
        ] {
            ensure!(
                value.is_finite() && value > 0.0,
                "pricing.{name} must be finite and positive"
            );
        }
        Ok(())
    }

    fn estimate(&self, usage: &ModelUsage) -> f64 {
        usage.cost.unwrap_or_else(|| {
            usage.prompt_tokens as f64 * self.input_usd_per_million / 1_000_000.0
                + usage.completion_tokens as f64 * self.output_usd_per_million / 1_000_000.0
        })
    }
}

/// Token and optional gateway-cost accounting for one logical operation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelUsage {
    /// Input tokens reported across successful logical responses and validation retries.
    pub prompt_tokens: u64,
    /// Output tokens reported across successful logical responses and validation retries.
    pub completion_tokens: u64,
    /// Provider-reported USD cost when every successful logical response reported it.
    pub cost: Option<f64>,
    /// Number of successful logical completion responses represented here.
    /// Transport-level retries without a response usage object are not visible.
    pub calls: usize,
}

impl ModelUsage {
    fn add_assign(&mut self, other: &Self) {
        let was_empty = self.calls == 0
            && self.prompt_tokens == 0
            && self.completion_tokens == 0
            && self.cost.is_none();
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.calls = self.calls.saturating_add(other.calls);
        self.cost = match (self.cost, other.cost) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(left), None) if other.calls == 0 => Some(left),
            (None, Some(right)) if was_empty => Some(right),
            (None, None) => None,
            _ => None,
        };
    }
}

/// One resumable memory-construction artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoMemoryRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Identity excluding the common answer model.
    pub run_key: String,
    /// Configured method or shared Semafuse stage identifier.
    pub method: String,
    /// Opaque evaluator question ID.
    pub question_id: String,
    /// Stable LoCoMo category label.
    pub category: String,
    /// Official category number in `1..=4`.
    pub category_number: u8,
    /// Whether source-message retrieval evidence exists for this question.
    pub retrieval_eligible: bool,
    /// Ranked source-message IDs, when the method exposes provenance.
    pub ranked_message_ids: Vec<String>,
    /// Expanded source-message coverage.
    pub covered_message_ids: Vec<String>,
    /// Messages retained by query-conditioned memory processing.
    pub retained_message_ids: Vec<String>,
    /// The actual context supplied verbatim to the common answer prompt.
    pub context: String,
    /// SHA-256 of `context` for answer-journal binding.
    pub context_sha256: String,
    /// UTF-8 context size.
    pub context_bytes: usize,
    /// Exact tokenizer count when supplied by an external worker.
    pub context_tokens: Option<usize>,
    /// Retrieval RecallAll@10 where evidence exists.
    pub retrieval_recall_all_at_10: Option<f64>,
    /// Retrieval nDCG@10 where evidence exists.
    pub retrieval_ndcg_at_10: Option<f64>,
    /// End-to-end memory call wall time.
    pub memory_wall_s: f64,
    /// Retrieval latency reported by Semafuse.
    pub retrieval_s: Option<f64>,
    /// Query-time window construction and processing latency.
    pub processing_s: Option<f64>,
    /// Once-per-conversation history construction latency.
    pub shared_construction_s: Option<f64>,
    /// Processor token/cost accounting. Shared stage rows keep this zero.
    pub processor_usage: ModelUsage,
    /// Secret-free method and execution metadata.
    pub metadata: Map<String, Value>,
}

/// One prediction scored by the released LoCoMo evaluator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoAnswerRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Identity including the common answer model and prompt.
    pub run_key: String,
    /// Memory journal identity from which this answer was generated.
    pub memory_run_key: String,
    /// Memory method identifier.
    pub method: String,
    /// Opaque evaluator question ID.
    pub question_id: String,
    /// Stable LoCoMo category label.
    pub category: String,
    /// Official category number in `1..=4`.
    pub category_number: u8,
    /// Memory context digest checked before generation.
    pub context_sha256: String,
    /// Model prediction retained locally for audit and rescoring.
    pub prediction: String,
    /// Prediction digest used by compact receipts.
    pub prediction_sha256: String,
    /// Released category-aware LoCoMo token-F1 result.
    pub official_f1: LocomoF1Score,
    /// Common answer-call wall time.
    pub answer_wall_s: f64,
    /// Answer-model token and cost accounting.
    pub answer_usage: ModelUsage,
    /// Secret-free response metadata.
    pub metadata: Map<String, Value>,
}

/// Runs context construction, common answer generation, and official F1.
pub async fn run_locomo_answer_suite(
    options: &LocomoAnswerSuiteRunOptions,
) -> Result<LocomoAnswerSuiteArtifacts> {
    ensure!(
        options.acknowledge_noncommercial,
        "LoCoMo is CC BY-NC 4.0; acknowledge non-commercial use explicitly"
    );
    ensure!(
        options.question_limit.is_none_or(|limit| limit > 0),
        "question_limit must be positive"
    );
    ensure!(
        options.question_limit.is_none() || options.question_ids.is_empty(),
        "question_limit and question_ids are mutually exclusive"
    );
    let suite = load_answer_suite(&options.suite_path)?;
    let compatibility = SuiteConfig {
        schema_version: suite.schema_version,
        name: suite.name.clone(),
        track: suite.track,
        baseline_method: suite.baseline_method.clone(),
        methods: suite.methods.clone(),
    };
    ensure!(
        !suite.name.trim().is_empty(),
        "suite name must be non-empty"
    );
    ensure!(
        !suite.baseline_method.trim().is_empty(),
        "baseline_method must be non-empty"
    );
    validate_method_configs(&suite.methods)?;
    suite.answerer.validate()?;
    suite.pricing.validate()?;
    let suite_dir = options
        .suite_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let methods = resolve_methods(&compatibility, suite_dir)?;

    let dataset_sha256 = sha256_file(&options.dataset_path)?;
    ensure!(
        dataset_sha256 == UPSTREAM_DATA_SHA256,
        "LoCoMo full suite requires pinned dataset SHA-256 {UPSTREAM_DATA_SHA256}; got {dataset_sha256}"
    );
    let dataset = load_locomo(
        &options.dataset_path,
        options.acknowledge_noncommercial,
        &suite.track.mode(),
    )?;
    validate_track_dataset(&dataset, suite.track)?;
    let imported = resolve_imported_rankings(
        &suite.imported_rankings,
        suite_dir,
        &dataset,
        &dataset_sha256,
    )?;
    let imported_memories = resolve_imported_memories(
        &suite.imported_memories,
        suite_dir,
        &dataset,
        &dataset_sha256,
    )?;
    validate_answer_outputs(&suite, &methods, &imported, &imported_memories)?;
    validate_method_filter(&options.methods, &methods, &imported, &imported_memories)?;
    validate_question_filter(&options.question_ids, &dataset)?;

    let implementation_sha256 = crate::benchmark_identity::source_sha256();
    let method_manifest = answer_method_manifest(&methods, &imported, &imported_memories);
    let memory_run_key = memory_run_key(
        &dataset_sha256,
        suite.track,
        &dataset,
        &method_manifest,
        &suite.pricing,
        &implementation_sha256,
        &cargo_lock_sha256(),
    )?;
    let answer_run_key = answer_run_key(
        &memory_run_key,
        &suite.answerer,
        &suite.pricing,
        &suite.baseline_method,
        &implementation_sha256,
        &cargo_lock_sha256(),
    )?;
    let scheduled = scheduled_questions(&dataset, options.question_limit, &options.question_ids);
    if options.validate_only {
        let expected_outputs = output_specs(&methods, &imported, &imported_memories)?;
        let summary = build_answer_summary(&suite, &[], &[], &expected_outputs, &dataset, false)?;
        let manifest = build_manifest(
            &suite,
            &methods,
            &imported,
            &imported_memories,
            &dataset,
            &dataset_sha256,
            &memory_run_key,
            &answer_run_key,
            &implementation_sha256,
            options,
            scheduled.len(),
            false,
        );
        let artifacts = LocomoAnswerSuiteArtifacts {
            manifest,
            memory_records: Vec::new(),
            answer_records: Vec::new(),
            summary,
            complete: false,
        };
        if let Some(output_dir) = &options.output_dir {
            write_summary_atomic(
                output_dir.join("validation.json"),
                &SummaryEnvelope {
                    run_key: answer_run_key,
                    summary: json!({
                        "manifest": &artifacts.manifest,
                        "summary": &artifacts.summary,
                    }),
                },
            )?;
        }
        return Ok(artifacts);
    }
    let scheduled_ids = scheduled
        .iter()
        .map(|question| question.input.id.as_str())
        .collect::<HashSet<_>>();

    let mut memory_journal = options
        .output_dir
        .as_ref()
        .map(|directory| {
            JsonlJournal::<LocomoMemoryRecord>::open(
                directory.join("memory-contexts.jsonl"),
                memory_run_key.clone(),
            )
        })
        .transpose()?;
    let mut memory_records = memory_journal
        .as_ref()
        .map(|journal| {
            journal
                .records()
                .iter()
                .map(|record| record.payload.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut answer_journal = options
        .output_dir
        .as_ref()
        .map(|directory| {
            JsonlJournal::<LocomoAnswerRecord>::open(
                directory.join("answers.jsonl"),
                answer_run_key.clone(),
            )
        })
        .transpose()?;
    let mut answer_records = answer_journal
        .as_ref()
        .map(|journal| {
            journal
                .records()
                .iter()
                .map(|record| record.payload.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Resume artifacts are untrusted persisted state. Validate both journals
    // before any operation that can issue a paid processor or answer call.
    validate_memory_records(
        &memory_records,
        &dataset,
        &methods,
        &imported,
        &imported_memories,
        &memory_run_key,
    )?;
    validate_answer_records(
        &answer_records,
        &memory_records,
        &dataset,
        &methods,
        &imported,
        &imported_memories,
        &suite.answerer,
        &answer_run_key,
        &memory_run_key,
    )?;

    execute_native_memories(
        &methods,
        &scheduled,
        &options.methods,
        &memory_run_key,
        &mut memory_journal,
        &mut memory_records,
    )
    .await?;
    execute_semafuse_memories(
        &methods,
        &scheduled,
        &options.methods,
        &memory_run_key,
        &suite.pricing,
        &answer_records,
        &mut memory_journal,
        &mut memory_records,
    )
    .await?;
    execute_imported_memories(
        &imported,
        &scheduled,
        &options.methods,
        &memory_run_key,
        &mut memory_journal,
        &mut memory_records,
    )?;
    execute_imported_memory_contexts(
        &imported_memories,
        &scheduled,
        &options.methods,
        &memory_run_key,
        &mut memory_journal,
        &mut memory_records,
    )?;

    let questions = dataset
        .questions
        .iter()
        .map(|question| (question.input.id.as_str(), question))
        .collect::<HashMap<_, _>>();
    let mut pending_memories = memory_records
        .iter()
        .filter(|record| {
            scheduled_ids.contains(record.question_id.as_str())
                && output_selected(
                    &options.methods,
                    &methods,
                    &imported,
                    &imported_memories,
                    &record.method,
                )
        })
        .cloned()
        .collect::<Vec<_>>();
    pending_memories.sort_by(|left, right| {
        left.question_id
            .cmp(&right.question_id)
            .then_with(|| left.method.cmp(&right.method))
    });
    for memory in pending_memories {
        if journal_has(
            &answer_journal,
            &answer_run_key,
            &memory.method,
            &memory.question_id,
            ANSWER_RECORD_KIND,
        ) {
            continue;
        }
        enforce_budget(&suite.pricing, &memory_records, &answer_records)?;
        let question = questions
            .get(memory.question_id.as_str())
            .with_context(|| format!("unknown memory question {}", memory.question_id))?;
        let record = answer_memory(
            &suite.answerer,
            &answer_run_key,
            &memory_run_key,
            question,
            &memory,
        )
        .await?;
        persist_answer(
            &mut answer_journal,
            &mut answer_records,
            &answer_run_key,
            record,
        )?;
    }

    validate_memory_records(
        &memory_records,
        &dataset,
        &methods,
        &imported,
        &imported_memories,
        &memory_run_key,
    )?;
    validate_answer_records(
        &answer_records,
        &memory_records,
        &dataset,
        &methods,
        &imported,
        &imported_memories,
        &suite.answerer,
        &answer_run_key,
        &memory_run_key,
    )?;
    let expected_outputs = output_specs(&methods, &imported, &imported_memories)?;
    let complete = options.question_limit.is_none()
        && options.question_ids.is_empty()
        && options.methods.is_empty()
        && is_complete(
            &answer_records,
            &answer_run_key,
            &expected_outputs,
            &dataset,
        );
    let summary = build_answer_summary(
        &suite,
        &memory_records,
        &answer_records,
        &expected_outputs,
        &dataset,
        complete,
    )?;
    let manifest = build_manifest(
        &suite,
        &methods,
        &imported,
        &imported_memories,
        &dataset,
        &dataset_sha256,
        &memory_run_key,
        &answer_run_key,
        &implementation_sha256,
        options,
        scheduled.len(),
        complete,
    );
    let artifacts = LocomoAnswerSuiteArtifacts {
        manifest,
        memory_records,
        answer_records,
        summary,
        complete,
    };
    if let Some(output_dir) = &options.output_dir {
        write_summary_atomic(
            output_dir.join("summary.json"),
            &SummaryEnvelope {
                run_key: answer_run_key,
                summary: json!({
                    "manifest": &artifacts.manifest,
                    "summary": &artifacts.summary,
                }),
            },
        )?;
    }
    Ok(artifacts)
}

fn load_answer_suite(path: &Path) -> Result<AnswerSuiteConfig> {
    let suite: AnswerSuiteConfig = LayeredConfig::builder()
        .add_source(File::from(path.to_path_buf()).required(true))
        .build()
        .with_context(|| format!("parse LoCoMo answer suite TOML {}", path.display()))?
        .try_deserialize()
        .with_context(|| format!("deserialize LoCoMo answer suite TOML {}", path.display()))?;
    ensure!(
        suite.schema_version == SUITE_SCHEMA_VERSION,
        "unsupported LoCoMo answer suite schema_version {}",
        suite.schema_version
    );
    Ok(suite)
}

fn resolve_imported_rankings(
    configs: &[ImportedRankingConfig],
    suite_dir: &Path,
    dataset: &PreparedDataset,
    dataset_sha256: &str,
) -> Result<Vec<ResolvedImportedRanking>> {
    let eligible = dataset
        .questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
        .map(|question| (question.input.id.as_str(), question))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        eligible.len() == dataset.summary.retrieval_count,
        "pinned LoCoMo retrieval cohort size drifted"
    );
    let mut destination_ids = BTreeSet::new();
    configs
        .iter()
        .map(|config| {
            ensure!(
                destination_ids.insert(config.id.clone()),
                "duplicate imported ranking id {}",
                config.id
            );
            ensure!(
                config.id.ends_with("_source_ranking_rag"),
                "imported ranking id {} must end in _source_ranking_rag so it cannot be mistaken for a native memory context",
                config.id
            );
            ensure!(
                !config.source_method.trim().is_empty()
                    && !config.source_run_key.trim().is_empty(),
                "imported ranking source identity cannot be empty"
            );
            ensure!(
                (1..=50).contains(&config.limit),
                "imported ranking {} limit must be between 1 and 50",
                config.id
            );
            let journal_path = resolve_answer_path(suite_dir, &config.journal)?;
            let summary_path = resolve_answer_path(suite_dir, &config.summary)?;
            let journal_bytes = fs::read(&journal_path).with_context(|| {
                format!("read imported ranking journal {}", journal_path.display())
            })?;
            let actual_journal_sha256 = hex::encode(Sha256::digest(&journal_bytes));
            ensure!(
                actual_journal_sha256 == config.journal_sha256,
                "imported ranking journal SHA-256 mismatch for {}",
                config.id
            );
            let summary_bytes = fs::read(&summary_path).with_context(|| {
                format!("read imported ranking summary {}", summary_path.display())
            })?;
            let actual_summary_sha256 = hex::encode(Sha256::digest(&summary_bytes));
            ensure!(
                actual_summary_sha256 == config.summary_sha256,
                "imported ranking summary SHA-256 mismatch for {}",
                config.id
            );
            validate_imported_summary(
                &summary_bytes,
                config,
                dataset_sha256,
                dataset.summary.retrieval_count,
            )?;
            let rankings = parse_imported_rankings(&journal_bytes, config, &eligible)?;
            Ok(ResolvedImportedRanking {
                id: config.id.clone(),
                source_method: config.source_method.clone(),
                source_run_key: config.source_run_key.clone(),
                journal_sha256: actual_journal_sha256,
                summary_sha256: actual_summary_sha256,
                limit: config.limit,
                cohort: config.cohort,
                rankings,
            })
        })
        .collect()
}

fn resolve_imported_memories(
    configs: &[ImportedMemoryConfig],
    suite_dir: &Path,
    dataset: &PreparedDataset,
    dataset_sha256: &str,
) -> Result<Vec<ResolvedImportedMemory>> {
    let mut destination_ids = BTreeSet::new();
    configs
        .iter()
        .map(|config| {
            ensure!(
                !config.id.trim().is_empty() && !config.id.contains("::"),
                "imported memory destination IDs must be non-empty and must not contain ::"
            );
            ensure!(
                destination_ids.insert(config.id.clone()),
                "duplicate imported memory destination id {}",
                config.id
            );
            ensure!(
                !config.source_method.trim().is_empty() && !config.source_run_key.trim().is_empty(),
                "imported memory source identity cannot be empty"
            );
            let journal_path = resolve_answer_path(suite_dir, &config.journal)?;
            let journal_bytes = fs::read(&journal_path).with_context(|| {
                format!("read imported memory journal {}", journal_path.display())
            })?;
            let actual_journal_sha256 = hex::encode(Sha256::digest(&journal_bytes));
            ensure!(
                actual_journal_sha256 == config.journal_sha256,
                "imported memory journal SHA-256 mismatch for {}",
                config.id
            );
            let summary_path = resolve_answer_path(suite_dir, &config.summary)?;
            let summary_bytes = fs::read(&summary_path).with_context(|| {
                format!("read imported memory summary {}", summary_path.display())
            })?;
            let actual_summary_sha256 = hex::encode(Sha256::digest(&summary_bytes));
            ensure!(
                actual_summary_sha256 == config.summary_sha256,
                "imported memory summary SHA-256 mismatch for {}",
                config.id
            );
            let source_kind =
                validate_imported_memory_summary(&summary_bytes, config, dataset_sha256)?;
            let records = parse_imported_memory_journal(&journal_bytes, config, dataset)?;
            Ok(ResolvedImportedMemory {
                id: config.id.clone(),
                source_method: config.source_method.clone(),
                source_run_key: config.source_run_key.clone(),
                journal_sha256: actual_journal_sha256,
                summary_sha256: actual_summary_sha256,
                source_kind,
                source_label_isolation_proven: true,
                cohort: config.cohort,
                records,
            })
        })
        .collect()
}

fn validate_imported_memory_summary(
    bytes: &[u8],
    config: &ImportedMemoryConfig,
    dataset_sha256: &str,
) -> Result<String> {
    let summary: Value = serde_json::from_slice(bytes)
        .with_context(|| format!("parse imported memory summary for {}", config.id))?;
    let manifest = summary
        .pointer("/summary/manifest")
        .and_then(Value::as_object)
        .context("imported memory summary omitted its manifest envelope")?;
    ensure!(
        manifest.get("memory_run_key").and_then(Value::as_str)
            == Some(config.source_run_key.as_str()),
        "imported memory summary memory run key mismatch for {}",
        config.id
    );
    ensure!(
        manifest
            .get("benchmark")
            .and_then(|benchmark| benchmark.get("dataset_sha256"))
            .and_then(Value::as_str)
            == Some(dataset_sha256),
        "imported memory summary dataset identity mismatch for {}",
        config.id
    );
    ensure!(
        manifest
            .get("protocol")
            .and_then(|protocol| protocol.get("label_isolation"))
            .and_then(Value::as_str)
            == Some(LABEL_ISOLATION_DECLARATION),
        "imported memory summary does not prove the current label-isolation protocol for {}",
        config.id
    );
    let methods = manifest
        .get("methods")
        .and_then(Value::as_array)
        .context("imported memory summary omitted method manifest")?;
    let matching = methods
        .iter()
        .filter(|method| {
            method.get("id").and_then(Value::as_str) == Some(config.source_method.as_str())
        })
        .collect::<Vec<_>>();
    ensure!(
        matching.len() == 1,
        "imported memory summary must declare source method {} exactly once",
        config.source_method
    );
    let source_kind = matching[0]
        .get("kind")
        .and_then(Value::as_str)
        .filter(|kind| !kind.trim().is_empty())
        .context("imported memory source method omitted its kind")?;
    Ok(source_kind.to_owned())
}

fn parse_imported_memory_journal(
    bytes: &[u8],
    config: &ImportedMemoryConfig,
    dataset: &PreparedDataset,
) -> Result<BTreeMap<String, LocomoMemoryRecord>> {
    ensure!(
        bytes.is_empty() || bytes.ends_with(b"\n"),
        "imported memory journal for {} has a truncated final line",
        config.id
    );
    let questions = dataset
        .questions
        .iter()
        .map(|question| (question.input.id.as_str(), question))
        .collect::<BTreeMap<_, _>>();
    let expected_ids = dataset
        .questions
        .iter()
        .filter(|question| question_in_cohort(question, config.cohort))
        .map(|question| question.input.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut envelope_keys = BTreeSet::new();
    let mut records = BTreeMap::new();
    let mut lines = bytes.split(|byte| *byte == b'\n').peekable();
    for index in 0.. {
        let Some(raw_line) = lines.next() else {
            break;
        };
        if lines.peek().is_none() && raw_line.is_empty() {
            continue;
        }
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        ensure!(
            !line.is_empty(),
            "imported memory journal {} has an empty JSONL record at line {}",
            config.id,
            index + 1
        );
        let envelope: JournalRecord<LocomoMemoryRecord> = serde_json::from_slice(line)
            .with_context(|| {
                format!(
                    "parse imported memory journal {} line {}",
                    config.id,
                    index + 1
                )
            })?;
        ensure!(
            envelope.run_key == config.source_run_key
                && envelope.payload.run_key == config.source_run_key,
            "imported memory journal {} has a mixed envelope or payload run key",
            config.id
        );
        ensure!(
            envelope.record_kind == MEMORY_RECORD_KIND
                && envelope.method == envelope.payload.method
                && envelope.question_id == envelope.payload.question_id,
            "imported memory journal {} has inconsistent envelope identity",
            config.id
        );
        ensure!(
            envelope_keys.insert((
                envelope.method.clone(),
                envelope.question_id.clone(),
                envelope.record_kind.clone(),
            )),
            "imported memory journal {} has a duplicate envelope key",
            config.id
        );
        if envelope.method != config.source_method {
            continue;
        }
        let question = questions
            .get(envelope.question_id.as_str())
            .with_context(|| {
                format!(
                    "imported source method {} contains unknown question {}",
                    config.source_method, envelope.question_id
                )
            })?;
        validate_imported_memory_payload(&envelope.payload, question, config)?;
        if question_in_cohort(question, config.cohort) {
            ensure!(
                records
                    .insert(envelope.question_id.clone(), envelope.payload)
                    .is_none(),
                "imported source method {} has a duplicate question {}",
                config.source_method,
                envelope.question_id
            );
        }
    }
    ensure!(
        records.len() == expected_ids.len()
            && expected_ids
                .iter()
                .all(|question_id| records.contains_key(*question_id)),
        "imported source method {} must completely cover the configured {:?} cohort of {} questions",
        config.source_method,
        config.cohort,
        expected_ids.len()
    );
    Ok(records)
}

fn validate_imported_memory_payload(
    record: &LocomoMemoryRecord,
    question: &PreparedQuestion,
    config: &ImportedMemoryConfig,
) -> Result<()> {
    ensure!(
        record.schema_version == MEMORY_RECORD_SCHEMA_VERSION,
        "imported memory {} has an unsupported payload schema",
        config.id
    );
    ensure!(
        record.method == config.source_method,
        "imported memory {} payload source method mismatch",
        config.id
    );
    ensure!(
        record.category == question.evaluator.category.name()
            && record.category_number == question.evaluator.category.number()
            && record.retrieval_eligible == question.evaluator.retrieval.eligible,
        "imported memory evaluator metadata drift for {}",
        record.question_id
    );
    ensure!(
        !record.context.contains('\0')
            && record.context_bytes == record.context.len()
            && record.context_sha256 == sha256_text(&record.context),
        "imported memory context identity mismatch for {}",
        record.question_id
    );
    ensure!(
        record.context_tokens.is_none_or(|tokens| tokens > 0),
        "imported memory context token count is invalid for {}",
        record.question_id
    );
    ensure!(
        record.memory_wall_s.is_finite()
            && record.memory_wall_s >= 0.0
            && record
                .retrieval_s
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && record
                .processing_s
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && record
                .shared_construction_s
                .is_none_or(|value| value.is_finite() && value >= 0.0)
            && record
                .processor_usage
                .cost
                .is_none_or(|value| value.is_finite() && value >= 0.0),
        "imported memory execution metadata is invalid for {}",
        record.question_id
    );
    let corpus = projected_corpus(question);
    validate_ids(&corpus, &record.ranked_message_ids, "imported ranking")?;
    validate_ids(&corpus, &record.covered_message_ids, "imported coverage")?;
    validate_ids(
        &corpus,
        &record.retained_message_ids,
        "imported retained IDs",
    )?;
    let expected_scores = retrieval_scores(question, &record.ranked_message_ids)?;
    ensure!(
        persisted_retrieval_score_matches(record.retrieval_recall_all_at_10, expected_scores.0)
            && persisted_retrieval_score_matches(record.retrieval_ndcg_at_10, expected_scores.1),
        "imported memory retrieval score mismatch for {}",
        record.question_id
    );
    Ok(())
}

fn question_in_cohort(question: &PreparedQuestion, cohort: QuestionCohort) -> bool {
    match cohort {
        QuestionCohort::AllQa => true,
        QuestionCohort::RetrievalEligible => question.evaluator.retrieval.eligible,
    }
}

fn resolve_answer_path(base: &Path, configured: &Path) -> Result<PathBuf> {
    let candidate = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        base.join(configured)
    };
    candidate
        .canonicalize()
        .with_context(|| format!("resolve configured path {}", candidate.display()))
}

fn validate_imported_summary(
    bytes: &[u8],
    config: &ImportedRankingConfig,
    dataset_sha256: &str,
    retrieval_count: usize,
) -> Result<()> {
    let summary: Value = serde_json::from_slice(bytes)
        .with_context(|| format!("parse imported ranking summary for {}", config.id))?;
    ensure!(
        summary.get("run_key").and_then(Value::as_str) == Some(&config.source_run_key)
            && summary
                .pointer("/summary/manifest/run_key")
                .and_then(Value::as_str)
                == Some(&config.source_run_key),
        "imported summary run key mismatch for {}",
        config.id
    );
    ensure!(
        summary
            .pointer("/summary/manifest/benchmark/dataset_sha256")
            .and_then(Value::as_str)
            == Some(dataset_sha256),
        "imported summary dataset identity mismatch for {}",
        config.id
    );
    ensure!(
        summary
            .pointer("/summary/manifest/denominators/retrieval_eval_n")
            .and_then(Value::as_u64)
            == Some(retrieval_count as u64),
        "imported summary retrieval denominator mismatch for {}",
        config.id
    );
    let methods = summary
        .pointer("/summary/manifest/methods")
        .and_then(Value::as_array)
        .context("imported summary omitted method manifest")?;
    ensure!(
        methods.iter().any(|method| {
            method.get("id").and_then(Value::as_str) == Some(&config.source_method)
                || method
                    .get("outputs")
                    .and_then(Value::as_array)
                    .is_some_and(|outputs| {
                        outputs
                            .iter()
                            .any(|output| output.as_str() == Some(config.source_method.as_str()))
                    })
        }),
        "imported summary does not declare source method {}",
        config.source_method
    );
    Ok(())
}

fn parse_imported_rankings(
    bytes: &[u8],
    config: &ImportedRankingConfig,
    eligible: &BTreeMap<&str, &PreparedQuestion>,
) -> Result<BTreeMap<String, Vec<String>>> {
    ensure!(
        bytes.is_empty() || bytes.ends_with(b"\n"),
        "imported ranking journal for {} has a truncated final line",
        config.id
    );
    let mut envelope_keys = BTreeSet::new();
    let mut rankings = BTreeMap::new();
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let envelope: JournalRecord<LocomoQuestionRecord> = serde_json::from_slice(line)
            .with_context(|| {
                format!(
                    "parse imported ranking journal {} line {}",
                    config.id,
                    index + 1
                )
            })?;
        ensure!(
            envelope.run_key == config.source_run_key
                && envelope.payload.run_key == config.source_run_key,
            "imported ranking journal {} has a mixed run key",
            config.id
        );
        ensure!(
            envelope.record_kind == "locomo_retrieval"
                && envelope.method == envelope.payload.method
                && envelope.question_id == envelope.payload.question_id,
            "imported ranking journal {} has inconsistent envelope identity",
            config.id
        );
        ensure!(
            envelope_keys.insert((
                envelope.method.clone(),
                envelope.question_id.clone(),
                envelope.record_kind.clone(),
            )),
            "imported ranking journal {} has a duplicate envelope key",
            config.id
        );
        if envelope.method != config.source_method {
            continue;
        }
        let question = eligible
            .get(envelope.question_id.as_str())
            .with_context(|| {
                format!(
                    "imported source method {} contains non-cohort question {}",
                    config.source_method, envelope.question_id
                )
            })?;
        ensure!(
            envelope.payload.category == question.evaluator.category.name()
                && envelope.payload.category_number == question.evaluator.category.number(),
            "imported ranking evaluator metadata drift for {}",
            envelope.question_id
        );
        ensure!(
            envelope.payload.ranked_message_ids.len() >= config.limit,
            "imported ranking {} has fewer than {} items for question {}",
            config.id,
            config.limit,
            envelope.question_id
        );
        let ranking = envelope
            .payload
            .ranked_message_ids
            .into_iter()
            .take(config.limit)
            .collect::<Vec<_>>();
        validate_ids(&projected_corpus(question), &ranking, "imported ranking")?;
        ensure!(
            rankings
                .insert(envelope.question_id.clone(), ranking)
                .is_none(),
            "imported source method {} has a duplicate question {}",
            config.source_method,
            envelope.question_id
        );
    }
    ensure!(
        rankings.len() == eligible.len()
            && eligible
                .keys()
                .all(|question_id| rankings.contains_key(*question_id)),
        "imported source method {} must cover exactly {} retrieval-eligible questions",
        config.source_method,
        eligible.len()
    );
    Ok(rankings)
}

fn validate_answer_outputs(
    suite: &AnswerSuiteConfig,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
) -> Result<()> {
    ensure!(
        methods
            .iter()
            .all(|method| !matches!(method, ResolvedMethod::ExternalJsonl(_))),
        "full answer runs refuse live external workers because their construction cost and resume state are not bounded; import a pinned retrieval journal instead"
    );
    let outputs = output_specs(methods, imported, imported_memories)?;
    ensure!(
        outputs.len() >= 2,
        "LoCoMo answer comparison requires at least two outputs"
    );
    ensure!(
        outputs.contains_key(suite.baseline_method.as_str()),
        "baseline_method must name a complete answer output"
    );
    ensure!(
        outputs.get(suite.baseline_method.as_str()) == Some(&QuestionCohort::AllQa),
        "baseline_method must cover every question in the configured track"
    );
    ensure!(
        methods.iter().any(|method| matches!(
            method,
            ResolvedMethod::Bm25 { id, .. } | ResolvedMethod::Recent { id, .. }
                if id == &suite.baseline_method
        )),
        "baseline_method must name a native configured BM25 or recent all-QA control"
    );
    ensure!(
        methods
            .iter()
            .any(|method| matches!(method, ResolvedMethod::Semafuse { .. }))
            || imported_memories.iter().any(|method| {
                method.source_kind == SEMAFUSE_FULL_MEMORY_MANIFEST_KIND
                    && method.source_label_isolation_proven
            }),
        "full memory suite must include a live or imported Semafuse memory method"
    );
    Ok(())
}

fn validate_method_filter(
    filter: &BTreeSet<String>,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
) -> Result<()> {
    if filter.is_empty() {
        return Ok(());
    }
    let mut configured = methods
        .iter()
        .map(ResolvedMethod::id)
        .collect::<BTreeSet<_>>();
    configured.extend(imported.iter().map(|method| method.id.as_str()));
    configured.extend(imported_memories.iter().map(|method| method.id.as_str()));
    let unknown = filter
        .iter()
        .filter(|method| !configured.contains(method.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ensure!(unknown.is_empty(), "unknown method filters: {unknown:?}");
    Ok(())
}

fn validate_question_filter(filter: &BTreeSet<String>, dataset: &PreparedDataset) -> Result<()> {
    let known = dataset
        .questions
        .iter()
        .map(|question| question.input.id.as_str())
        .collect::<HashSet<_>>();
    let unknown = filter
        .iter()
        .filter(|question_id| !known.contains(question_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    ensure!(unknown.is_empty(), "unknown question filters: {unknown:?}");
    Ok(())
}

fn all_output_ids(methods: &[ResolvedMethod]) -> BTreeSet<String> {
    let mut outputs = BTreeSet::new();
    for method in methods {
        match method {
            ResolvedMethod::Semafuse { id, stages, .. } => {
                outputs.insert(id.clone());
                outputs.extend(
                    stages
                        .iter()
                        .map(|stage| format!("{id}::{}", stage.as_str())),
                );
            }
            _ => {
                outputs.insert(method.id().to_owned());
            }
        }
    }
    outputs
}

fn output_specs(
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
) -> Result<BTreeMap<String, QuestionCohort>> {
    let mut outputs = all_output_ids(methods)
        .into_iter()
        .map(|id| (id, QuestionCohort::AllQa))
        .collect::<BTreeMap<_, _>>();
    for method in imported {
        let cohort = match method.cohort {
            ImportedCohort::RetrievalEligible => QuestionCohort::RetrievalEligible,
        };
        ensure!(
            outputs.insert(method.id.clone(), cohort).is_none(),
            "duplicate answer output id {}",
            method.id
        );
    }
    for method in imported_memories {
        ensure!(
            outputs.insert(method.id.clone(), method.cohort).is_none(),
            "duplicate answer output id {}",
            method.id
        );
    }
    Ok(outputs)
}

fn output_selected(
    selected_sources: &BTreeSet<String>,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
    output: &str,
) -> bool {
    methods.iter().any(|method| {
        source_selected(selected_sources, method.id())
            && match method {
                ResolvedMethod::Semafuse { id, stages, .. } => {
                    output == id
                        || stages
                            .iter()
                            .any(|stage| output == format!("{id}::{}", stage.as_str()))
                }
                _ => output == method.id(),
            }
    }) || imported
        .iter()
        .any(|method| source_selected(selected_sources, &method.id) && output == method.id)
        || imported_memories
            .iter()
            .any(|method| source_selected(selected_sources, &method.id) && output == method.id)
}

fn source_selected(selected_sources: &BTreeSet<String>, source: &str) -> bool {
    selected_sources.is_empty() || selected_sources.contains(source)
}

fn scheduled_questions<'a>(
    dataset: &'a PreparedDataset,
    limit: Option<usize>,
    question_ids: &BTreeSet<String>,
) -> Vec<&'a PreparedQuestion> {
    if !question_ids.is_empty() {
        return dataset
            .questions
            .iter()
            .filter(|question| question_ids.contains(&question.input.id))
            .collect();
    }
    let Some(limit) = limit else {
        return dataset.questions.iter().collect();
    };
    let mut groups = BTreeMap::<(String, u8), VecDeque<&PreparedQuestion>>::new();
    for question in &dataset.questions {
        groups
            .entry((
                question.evaluator.source_conversation_id.clone(),
                question.evaluator.category.number(),
            ))
            .or_default()
            .push_back(question);
    }
    let mut selected = Vec::with_capacity(limit.min(dataset.questions.len()));
    while selected.len() < limit && groups.values().any(|questions| !questions.is_empty()) {
        for questions in groups.values_mut() {
            if let Some(question) = questions.pop_front() {
                selected.push(question);
                if selected.len() == limit {
                    break;
                }
            }
        }
    }
    selected
}

fn grouped_questions<'a>(
    questions: &[&'a PreparedQuestion],
) -> BTreeMap<String, Vec<&'a PreparedQuestion>> {
    let mut grouped = BTreeMap::new();
    for question in questions {
        grouped
            .entry(question.evaluator.source_conversation_id.clone())
            .or_insert_with(Vec::new)
            .push(*question);
    }
    grouped
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

fn sha256_text(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

fn hit_ids(hits: &[BaselineHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.message_id.clone()).collect()
}

fn render_ranked_context(corpus: &[BaselineMessage], ranking: &[String]) -> Result<String> {
    let selected_ids = ranking.iter().map(String::as_str).collect::<HashSet<_>>();
    ensure!(
        selected_ids.len() == ranking.len(),
        "ranked context contains duplicate message IDs"
    );
    let known = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    for id in &selected_ids {
        ensure!(
            known.contains(id),
            "ranked context contains unknown ID {id}"
        );
    }
    let mut selected = corpus
        .iter()
        .filter(|message| selected_ids.contains(message.id.as_str()))
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

fn retrieval_scores(
    question: &PreparedQuestion,
    ranking: &[String],
) -> Result<(Option<f64>, Option<f64>)> {
    if !question.evaluator.retrieval.eligible {
        return Ok((None, None));
    }
    let metrics = evaluate_ranking(ranking, &question.evaluator.retrieval.gold_message_ids)?;
    let at_10 = metrics.at(10).context("LoCoMo metrics omitted cutoff 10")?;
    Ok((Some(at_10.recall_all), Some(at_10.ndcg)))
}

async fn execute_native_memories(
    methods: &[ResolvedMethod],
    questions: &[&PreparedQuestion],
    selected_sources: &BTreeSet<String>,
    run_key: &str,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    for question in questions {
        let corpus = projected_corpus(question);
        for method in methods {
            if !source_selected(selected_sources, method.id()) {
                continue;
            }
            let (id, limit, kind) = match method {
                ResolvedMethod::Bm25 { id, limit } => (id, limit, "bm25"),
                ResolvedMethod::Recent { id, limit } => (id, limit, "recent"),
                _ => continue,
            };
            if journal_has(journal, run_key, id, &question.input.id, MEMORY_RECORD_KIND) {
                continue;
            }
            let started = Instant::now();
            let ranking = match method {
                ResolvedMethod::Bm25 { .. } => {
                    hit_ids(&bm25(&corpus, &question.input.query, *limit)?)
                }
                ResolvedMethod::Recent { .. } => hit_ids(&recent(&corpus, *limit)?),
                _ => unreachable!(),
            };
            let context = render_ranked_context(&corpus, &ranking)?;
            let wall_s = started.elapsed().as_secs_f64();
            let record = build_memory_record(
                run_key,
                id,
                question,
                ranking.clone(),
                ranking.clone(),
                ranking,
                context,
                None,
                wall_s,
                Some(wall_s),
                Some(0.0),
                None,
                ModelUsage::default(),
                Map::from_iter([
                    ("kind".into(), json!(kind)),
                    ("context_order".into(), json!("source_chronological")),
                    ("shared_execution".into(), json!(false)),
                ]),
            )?;
            persist_memory(journal, records, run_key, record)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn execute_semafuse_memories(
    methods: &[ResolvedMethod],
    questions: &[&PreparedQuestion],
    selected_sources: &BTreeSet<String>,
    run_key: &str,
    pricing: &PricingConfig,
    answer_records: &[LocomoAnswerRecord],
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    for method in methods {
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
            continue;
        };
        if !source_selected(selected_sources, id) {
            continue;
        }
        let outputs = std::iter::once(id.clone())
            .chain(
                stages
                    .iter()
                    .map(|stage| format!("{id}::{}", stage.as_str())),
            )
            .collect::<Vec<_>>();
        for grouped in grouped_questions(questions).values() {
            for question in grouped {
                recover_semafuse_stages(
                    run_key,
                    id,
                    config_sha256,
                    resolved_config_sha256,
                    stages,
                    question,
                    journal,
                    records,
                )?;
            }
            let missing = grouped
                .iter()
                .copied()
                .filter(|question| {
                    outputs.iter().any(|output| {
                        !journal_has(
                            journal,
                            run_key,
                            output,
                            &question.input.id,
                            MEMORY_RECORD_KIND,
                        )
                    })
                })
                .collect::<Vec<_>>();
            if missing.is_empty() {
                continue;
            }
            let exemplar = grouped[0];
            validate_shared_conversation(grouped, exemplar)?;
            enforce_budget(pricing, records, answer_records)?;
            let exemplar_input = runner_question(exemplar);
            let mut session = StrategyBenchmarkSession::new(config, &exemplar_input).await?;
            let construction_s = session.construction_metrics().total_s;
            let history_identity_sha256 = session.history_identity_sha256().to_owned();
            let construction_amortization_n = missing.len();
            for question in missing {
                ensure!(
                    outputs.iter().all(|output| !journal_has(
                        journal,
                        run_key,
                        output,
                        &question.input.id,
                        MEMORY_RECORD_KIND,
                    )),
                    "partial Semafuse output set could not be recovered without another paid processor call"
                );
                enforce_budget(pricing, records, answer_records)?;
                let started = Instant::now();
                let response = session
                    .query_full(runner_question(question), *limit)
                    .await?;
                let wall_s = started.elapsed().as_secs_f64();
                persist_semafuse_response(
                    run_key,
                    id,
                    config_sha256,
                    resolved_config_sha256,
                    stages,
                    question,
                    &response,
                    wall_s,
                    construction_s,
                    &history_identity_sha256,
                    construction_amortization_n,
                    journal,
                    records,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn persist_semafuse_response(
    run_key: &str,
    id: &str,
    config_sha256: &str,
    resolved_config_sha256: &str,
    stages: &[SemafuseStage],
    question: &PreparedQuestion,
    response: &BenchmarkResponse,
    wall_s: f64,
    construction_s: f64,
    history_identity_sha256: &str,
    construction_amortization_n: usize,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    let common_metadata = semafuse_common_metadata(
        id,
        config_sha256,
        resolved_config_sha256,
        history_identity_sha256,
        construction_amortization_n,
    );
    let processor_usage = processor_usage(&response.metadata)?;
    if !journal_has(journal, run_key, id, &question.input.id, MEMORY_RECORD_KIND) {
        let mut metadata = common_metadata.clone();
        metadata.insert("kind".into(), json!(SEMAFUSE_MEMORY_KIND));
        metadata.insert("strategy".into(), json!(response.strategy));
        metadata.insert(
            "strategy_trace".into(),
            Value::Object(response.metadata.clone()),
        );
        metadata.insert(
            "stage_rankings".into(),
            serde_json::to_value(&response.stage_rankings)?,
        );
        let record = build_memory_record(
            run_key,
            id,
            question,
            response.retrieved_message_ids.clone(),
            response.covered_message_ids.clone(),
            response.retained_message_ids.clone(),
            response.context.clone(),
            None,
            wall_s,
            Some(response.retrieval_latency_s),
            Some(response.processing_latency_s),
            Some(construction_s),
            processor_usage,
            metadata,
        )?;
        persist_memory(journal, records, run_key, record)?;
    }
    persist_semafuse_stage_records(
        run_key,
        id,
        stages,
        question,
        &response.stage_rankings,
        wall_s,
        Some(response.retrieval_latency_s),
        Some(construction_s),
        common_metadata,
        journal,
        records,
    )
}

fn semafuse_common_metadata(
    id: &str,
    config_sha256: &str,
    resolved_config_sha256: &str,
    history_identity_sha256: &str,
    construction_amortization_n: usize,
) -> Map<String, Value> {
    Map::from_iter([
        ("source_method".into(), json!(id)),
        ("config_sha256".into(), json!(config_sha256)),
        (
            "resolved_config_sha256".into(),
            json!(resolved_config_sha256),
        ),
        (
            "history_identity_sha256".into(),
            json!(history_identity_sha256),
        ),
        (
            "construction_amortization_n".into(),
            json!(construction_amortization_n),
        ),
        ("shared_execution".into(), json!(true)),
    ])
}

#[allow(clippy::too_many_arguments)]
fn recover_semafuse_stages(
    run_key: &str,
    id: &str,
    config_sha256: &str,
    resolved_config_sha256: &str,
    stages: &[SemafuseStage],
    question: &PreparedQuestion,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    let Some(main) = records
        .iter()
        .find(|record| record.method == id && record.question_id == question.input.id)
        .cloned()
    else {
        let orphaned = stages.iter().any(|stage| {
            journal_has(
                journal,
                run_key,
                &format!("{id}::{}", stage.as_str()),
                &question.input.id,
                MEMORY_RECORD_KIND,
            )
        });
        ensure!(
            !orphaned,
            "Semafuse stage exists without its paid main response; refusing an ambiguous resume"
        );
        return Ok(());
    };
    if stages.iter().all(|stage| {
        journal_has(
            journal,
            run_key,
            &format!("{id}::{}", stage.as_str()),
            &question.input.id,
            MEMORY_RECORD_KIND,
        )
    }) {
        return Ok(());
    }
    ensure!(
        main.metadata.get("config_sha256").and_then(Value::as_str) == Some(config_sha256)
            && main
                .metadata
                .get("resolved_config_sha256")
                .and_then(Value::as_str)
                == Some(resolved_config_sha256),
        "persisted Semafuse main response has different configuration identity"
    );
    let stage_rankings = main
        .metadata
        .get("stage_rankings")
        .cloned()
        .context("persisted Semafuse main response omitted recoverable stage rankings")?;
    let stage_rankings = serde_json::from_value::<BTreeMap<String, Vec<String>>>(stage_rankings)
        .context("persisted Semafuse stage rankings are malformed")?;
    let history_identity_sha256 = main
        .metadata
        .get("history_identity_sha256")
        .and_then(Value::as_str)
        .context("persisted Semafuse main response omitted history identity")?;
    let construction_amortization_n = main
        .metadata
        .get("construction_amortization_n")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .context("persisted Semafuse main response omitted construction amortization count")?;
    let common_metadata = semafuse_common_metadata(
        id,
        config_sha256,
        resolved_config_sha256,
        history_identity_sha256,
        construction_amortization_n,
    );
    persist_semafuse_stage_records(
        run_key,
        id,
        stages,
        question,
        &stage_rankings,
        main.memory_wall_s,
        main.retrieval_s,
        main.shared_construction_s,
        common_metadata,
        journal,
        records,
    )
}

#[allow(clippy::too_many_arguments)]
fn persist_semafuse_stage_records(
    run_key: &str,
    id: &str,
    stages: &[SemafuseStage],
    question: &PreparedQuestion,
    stage_rankings: &BTreeMap<String, Vec<String>>,
    wall_s: f64,
    retrieval_s: Option<f64>,
    construction_s: Option<f64>,
    common_metadata: Map<String, Value>,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    let corpus = projected_corpus(question);
    for stage in stages {
        let output = format!("{id}::{}", stage.as_str());
        if journal_has(
            journal,
            run_key,
            &output,
            &question.input.id,
            MEMORY_RECORD_KIND,
        ) {
            continue;
        }
        let ranking = stage_rankings
            .get(stage.as_str())
            .with_context(|| format!("Semafuse response omitted stage {}", stage.as_str()))?
            .iter()
            .take(50)
            .cloned()
            .collect::<Vec<_>>();
        let context = render_ranked_context(&corpus, &ranking)?;
        let mut metadata = common_metadata.clone();
        metadata.insert("kind".into(), json!("shared_semafuse_stage_context"));
        metadata.insert("stage".into(), json!(stage.as_str()));
        metadata.insert("usage_shared_with".into(), json!(id));
        let record = build_memory_record(
            run_key,
            &output,
            question,
            ranking.clone(),
            ranking.clone(),
            ranking,
            context,
            None,
            wall_s,
            retrieval_s,
            Some(0.0),
            construction_s,
            ModelUsage::default(),
            metadata,
        )?;
        persist_memory(journal, records, run_key, record)?;
    }
    Ok(())
}

fn execute_imported_memories(
    methods: &[ResolvedImportedRanking],
    questions: &[&PreparedQuestion],
    selected_sources: &BTreeSet<String>,
    run_key: &str,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    for method in methods {
        if !source_selected(selected_sources, &method.id) {
            continue;
        }
        for question in questions
            .iter()
            .copied()
            .filter(|question| question.evaluator.retrieval.eligible)
        {
            if journal_has(
                journal,
                run_key,
                &method.id,
                &question.input.id,
                MEMORY_RECORD_KIND,
            ) {
                continue;
            }
            let ranking = method
                .rankings
                .get(&question.input.id)
                .with_context(|| {
                    format!(
                        "validated imported ranking {} omitted question {}",
                        method.id, question.input.id
                    )
                })?
                .clone();
            let started = Instant::now();
            let context = render_ranked_context(&projected_corpus(question), &ranking)?;
            let wall_s = started.elapsed().as_secs_f64();
            let metadata = Map::from_iter([
                ("kind".into(), json!("imported_source_ranking_rag")),
                ("cohort".into(), json!("retrieval_eligible")),
                ("source_method".into(), json!(method.source_method)),
                ("source_run_key".into(), json!(method.source_run_key)),
                ("source_journal_sha256".into(), json!(method.journal_sha256)),
                ("source_summary_sha256".into(), json!(method.summary_sha256)),
                ("source_ranking_limit".into(), json!(method.limit)),
                ("native_memory_reused".into(), json!(false)),
                ("context_order".into(), json!("source_chronological")),
                ("shared_execution".into(), json!(false)),
            ]);
            let record = build_memory_record(
                run_key,
                &method.id,
                question,
                ranking.clone(),
                ranking.clone(),
                ranking,
                context,
                None,
                wall_s,
                None,
                None,
                None,
                ModelUsage::default(),
                metadata,
            )?;
            persist_memory(journal, records, run_key, record)?;
        }
    }
    Ok(())
}

/// Materializes already-validated memory contexts under the current run
/// identity. This path is deliberately synchronous and has no strategy,
/// embedding, reranker, processor, or inference dependency.
fn execute_imported_memory_contexts(
    methods: &[ResolvedImportedMemory],
    questions: &[&PreparedQuestion],
    selected_sources: &BTreeSet<String>,
    run_key: &str,
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
) -> Result<()> {
    for method in methods {
        if !source_selected(selected_sources, &method.id) {
            continue;
        }
        for question in questions
            .iter()
            .copied()
            .filter(|question| question_in_cohort(question, method.cohort))
        {
            if journal_has(
                journal,
                run_key,
                &method.id,
                &question.input.id,
                MEMORY_RECORD_KIND,
            ) {
                continue;
            }
            let source = method.records.get(&question.input.id).with_context(|| {
                format!(
                    "validated imported memory {} omitted question {}",
                    method.id, question.input.id
                )
            })?;
            let source_payload_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(source)?));
            let record = build_memory_record(
                run_key,
                &method.id,
                question,
                source.ranked_message_ids.clone(),
                source.covered_message_ids.clone(),
                source.retained_message_ids.clone(),
                source.context.clone(),
                source.context_tokens,
                0.0,
                None,
                None,
                None,
                ModelUsage::default(),
                Map::from_iter([
                    ("kind".into(), json!("imported_memory_context")),
                    ("cohort".into(), json!(method.cohort)),
                    ("source_method".into(), json!(method.source_method)),
                    ("source_run_key".into(), json!(method.source_run_key)),
                    ("source_journal_sha256".into(), json!(method.journal_sha256)),
                    ("source_summary_sha256".into(), json!(method.summary_sha256)),
                    ("source_kind".into(), json!(method.source_kind)),
                    (
                        "source_label_isolation_proven".into(),
                        json!(method.source_label_isolation_proven),
                    ),
                    ("source_context_sha256".into(), json!(source.context_sha256)),
                    ("source_payload_sha256".into(), json!(source_payload_sha256)),
                    ("context_reused_verbatim".into(), json!(true)),
                    ("source_metadata_copied".into(), json!(false)),
                    ("live_memory_construction".into(), json!(false)),
                    ("processor_usage_replayed".into(), json!(false)),
                ]),
            )?;
            ensure!(
                record.context_sha256 == source.context_sha256,
                "imported memory context changed while cloning {}",
                question.input.id
            );
            persist_memory(journal, records, run_key, record)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_memory_record(
    run_key: &str,
    method: &str,
    question: &PreparedQuestion,
    ranked_message_ids: Vec<String>,
    covered_message_ids: Vec<String>,
    retained_message_ids: Vec<String>,
    context: String,
    context_tokens: Option<usize>,
    memory_wall_s: f64,
    retrieval_s: Option<f64>,
    processing_s: Option<f64>,
    shared_construction_s: Option<f64>,
    processor_usage: ModelUsage,
    metadata: Map<String, Value>,
) -> Result<LocomoMemoryRecord> {
    ensure!(
        !method.trim().is_empty() && !context.contains('\0'),
        "memory output is malformed"
    );
    let corpus = projected_corpus(question);
    validate_ids(&corpus, &ranked_message_ids, "ranking")?;
    validate_ids(&corpus, &covered_message_ids, "coverage")?;
    validate_ids(&corpus, &retained_message_ids, "retained")?;
    let (retrieval_recall_all_at_10, retrieval_ndcg_at_10) =
        retrieval_scores(question, &ranked_message_ids)?;
    let context_sha256 = sha256_text(&context);
    let context_bytes = context.len();
    Ok(LocomoMemoryRecord {
        schema_version: MEMORY_RECORD_SCHEMA_VERSION,
        run_key: run_key.to_owned(),
        method: method.to_owned(),
        question_id: question.input.id.clone(),
        category: question.evaluator.category.name().to_owned(),
        category_number: question.evaluator.category.number(),
        retrieval_eligible: question.evaluator.retrieval.eligible,
        ranked_message_ids,
        covered_message_ids,
        retained_message_ids,
        context,
        context_sha256,
        context_bytes,
        context_tokens,
        retrieval_recall_all_at_10,
        retrieval_ndcg_at_10,
        memory_wall_s,
        retrieval_s,
        processing_s,
        shared_construction_s,
        processor_usage,
        metadata,
    })
}

fn validate_ids(corpus: &[BaselineMessage], ids: &[String], label: &str) -> Result<()> {
    let known = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    for id in ids {
        ensure!(seen.insert(id), "{label} contains duplicate ID {id}");
        ensure!(
            known.contains(id.as_str()),
            "{label} contains unknown ID {id}"
        );
    }
    Ok(())
}

fn persist_memory(
    journal: &mut Option<JsonlJournal<LocomoMemoryRecord>>,
    records: &mut Vec<LocomoMemoryRecord>,
    run_key: &str,
    record: LocomoMemoryRecord,
) -> Result<()> {
    if journal_has(
        journal,
        run_key,
        &record.method,
        &record.question_id,
        MEMORY_RECORD_KIND,
    ) {
        return Ok(());
    }
    if let Some(journal) = journal {
        journal.append(JournalRecord {
            run_key: run_key.to_owned(),
            method: record.method.clone(),
            question_id: record.question_id.clone(),
            record_kind: MEMORY_RECORD_KIND.to_owned(),
            payload: record.clone(),
        })?;
    }
    records.push(record);
    Ok(())
}

fn persist_answer(
    journal: &mut Option<JsonlJournal<LocomoAnswerRecord>>,
    records: &mut Vec<LocomoAnswerRecord>,
    run_key: &str,
    record: LocomoAnswerRecord,
) -> Result<()> {
    if journal_has(
        journal,
        run_key,
        &record.method,
        &record.question_id,
        ANSWER_RECORD_KIND,
    ) {
        return Ok(());
    }
    if let Some(journal) = journal {
        journal.append(JournalRecord {
            run_key: run_key.to_owned(),
            method: record.method.clone(),
            question_id: record.question_id.clone(),
            record_kind: ANSWER_RECORD_KIND.to_owned(),
            payload: record.clone(),
        })?;
    }
    records.push(record);
    Ok(())
}

fn journal_has<T>(
    journal: &Option<JsonlJournal<T>>,
    run_key: &str,
    method: &str,
    question_id: &str,
    record_kind: &str,
) -> bool
where
    T: Serialize + serde::de::DeserializeOwned,
{
    journal.as_ref().is_some_and(|journal| {
        journal.completed_keys().contains(&JournalRecordKey {
            run_key: run_key.to_owned(),
            method: method.to_owned(),
            question_id: question_id.to_owned(),
            record_kind: record_kind.to_owned(),
        })
    })
}

fn processor_usage(metadata: &Map<String, Value>) -> Result<ModelUsage> {
    let processing = metadata
        .get("strategy")
        .and_then(Value::as_object)
        .and_then(|strategy| strategy.get("processing"))
        .context("full Semafuse response omitted strategy.processing cost trace")?;
    let usage = processing
        .get("usage")
        .context("full Semafuse response omitted strategy.processing.usage")?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cost = usage.get("cost").and_then(Value::as_f64);
    if let Some(cost) = cost {
        ensure!(cost.is_finite() && cost >= 0.0, "invalid processor cost");
    }
    let calls = usage
        .get("calls")
        .and_then(Value::as_u64)
        .context("processor usage omitted logical response count")?;
    let calls = usize::try_from(calls).context("processor call count does not fit usize")?;
    let traced_calls = processing
        .get("processor_calls")
        .and_then(Value::as_array)
        .context("processor trace omitted processor_calls")?
        .iter()
        .map(|window| {
            window
                .get("calls")
                .and_then(Value::as_array)
                .map(Vec::len)
                .context("processor window trace omitted calls")
        })
        .sum::<Result<usize>>()?;
    ensure!(
        calls == traced_calls,
        "processor usage call count {calls} disagrees with trace count {traced_calls}"
    );
    let missing_prompt = usage
        .get("calls_missing_prompt_tokens")
        .and_then(Value::as_u64)
        .context("processor usage omitted missing prompt-token count")?;
    let missing_completion = usage
        .get("calls_missing_completion_tokens")
        .and_then(Value::as_u64)
        .context("processor usage omitted missing completion-token count")?;
    let missing_cost = usage
        .get("calls_missing_cost")
        .and_then(Value::as_u64)
        .context("processor usage omitted missing cost count")?;
    ensure!(
        missing_prompt <= calls as u64
            && missing_completion <= calls as u64
            && missing_cost <= calls as u64,
        "processor missing-usage counters exceed logical response count"
    );
    ensure!(
        calls == 0 || cost.is_some() || (missing_prompt == 0 && missing_completion == 0),
        "processor provider reported neither complete token usage nor complete cost for successful responses; refusing an unbounded paid run"
    );
    ensure!(
        cost.is_none() || missing_cost == 0,
        "processor aggregate cost is present despite missing per-call cost"
    );
    Ok(ModelUsage {
        prompt_tokens,
        completion_tokens,
        cost,
        calls,
    })
}

async fn answer_memory(
    answerer: &AnswerModelConfig,
    run_key: &str,
    memory_run_key: &str,
    question: &PreparedQuestion,
    memory: &LocomoMemoryRecord,
) -> Result<LocomoAnswerRecord> {
    ensure!(
        sha256_text(&memory.context) == memory.context_sha256,
        "memory context digest changed before answer generation"
    );
    let prompt = answerer.protocol.render(AnswerPromptInput {
        query_time: &question.input.query_time,
        query: &question.input.query,
        memory: &memory.context,
    });
    let started = Instant::now();
    let completion = match answerer.protocol.response_contract() {
        Some(contract) => {
            answerer
                .endpoint
                .complete_json_contract(prompt.system, &prompt.user, answerer.options(), contract)
                .await
        }
        None => {
            answerer
                .endpoint
                .chat(prompt.system, &prompt.user, answerer.options())
                .await
        }
    }
    .with_context(|| {
        format!(
            "answer opaque LoCoMo question {} with {}",
            question.input.id, memory.method
        )
    })?;
    let answer_wall_s = started.elapsed().as_secs_f64();
    ensure!(
        completion.model.as_deref() == Some(answerer.endpoint.model.as_str()),
        "answer response model mismatch: expected {:?}, got {:?}",
        answerer.endpoint.model,
        completion.model
    );
    ensure!(
        completion.provider.as_deref() == Some(answerer.expected_provider.as_str()),
        "answer response provider mismatch: expected {:?}, got {:?}",
        answerer.expected_provider,
        completion.provider
    );
    // The concise JSON protocol discards its `reasoning` field here. Only the
    // returned answer text may enter the journal or official scorer.
    let prediction = answerer.protocol.parse_answer(&completion.content)?;
    ensure!(
        completion.finish_reason.as_deref() == Some("stop"),
        "answer model did not finish normally (finish_reason={:?})",
        completion.finish_reason
    );
    let reference = match &question.evaluator.answer {
        Value::String(answer) => answer.as_str(),
        _ => bail!("LoCoMo reference answer is not a string"),
    };
    let official_f1 = score_locomo_answer(question.evaluator.category, &prediction, reference);
    ensure!(
        completion.cost.is_some()
            || (completion.prompt_tokens.is_some() && completion.completion_tokens.is_some()),
        "answer provider reported neither complete token usage nor cost; refusing an unbounded paid run"
    );
    if let Some(cost) = completion.cost {
        ensure!(cost.is_finite() && cost >= 0.0, "invalid answer cost");
    }
    let answer_usage = ModelUsage {
        prompt_tokens: completion.prompt_tokens.unwrap_or(0),
        completion_tokens: completion.completion_tokens.unwrap_or(0),
        cost: completion.cost,
        calls: 1,
    };
    Ok(LocomoAnswerRecord {
        schema_version: ANSWER_RECORD_SCHEMA_VERSION,
        run_key: run_key.to_owned(),
        memory_run_key: memory_run_key.to_owned(),
        method: memory.method.clone(),
        question_id: question.input.id.clone(),
        category: question.evaluator.category.name().to_owned(),
        category_number: question.evaluator.category.number(),
        context_sha256: memory.context_sha256.clone(),
        prediction_sha256: sha256_text(&prediction),
        prediction,
        official_f1,
        answer_wall_s,
        answer_usage,
        metadata: Map::from_iter([
            ("prompt_version".into(), json!(answerer.protocol.version())),
            (
                "answer_protocol".into(),
                serde_json::to_value(answerer.protocol.identity())?,
            ),
            ("response_model".into(), json!(completion.model)),
            ("response_provider".into(), json!(completion.provider)),
            (
                "response_finish_reason".into(),
                json!(completion.finish_reason),
            ),
            (
                "response_reasoning_tokens".into(),
                json!(completion.reasoning_tokens),
            ),
            (
                "response_reasoning_present".into(),
                json!(completion.reasoning_present),
            ),
        ]),
    })
}

fn enforce_budget(
    pricing: &PricingConfig,
    memories: &[LocomoMemoryRecord],
    answers: &[LocomoAnswerRecord],
) -> Result<()> {
    let estimated = estimated_reported_usage_cost(pricing, memories, answers);
    ensure!(
        estimated < pricing.max_reported_usage_usd,
        "reported-usage budget fuse reached: estimated ${estimated:.6} >= configured ${:.6}; journals are resumable",
        pricing.max_reported_usage_usd
    );
    Ok(())
}

fn estimated_reported_usage_cost(
    pricing: &PricingConfig,
    memories: &[LocomoMemoryRecord],
    answers: &[LocomoAnswerRecord],
) -> f64 {
    memories
        .iter()
        .map(|record| pricing.estimate(&record.processor_usage))
        .chain(
            answers
                .iter()
                .map(|record| pricing.estimate(&record.answer_usage)),
        )
        .sum()
}

fn validate_imported_destination_memory(
    record: &LocomoMemoryRecord,
    method: &ResolvedImportedMemory,
) -> Result<()> {
    let source = method
        .records
        .get(&record.question_id)
        .context("imported destination memory has no pinned source payload")?;
    ensure!(
        record.context == source.context
            && record.context_bytes == source.context_bytes
            && record.context_sha256 == source.context_sha256
            && record.context_tokens == source.context_tokens,
        "imported destination context does not match its pinned source payload"
    );
    ensure!(
        record.ranked_message_ids == source.ranked_message_ids
            && record.covered_message_ids == source.covered_message_ids
            && record.retained_message_ids == source.retained_message_ids,
        "imported destination retrieval provenance does not match its pinned source payload"
    );
    ensure!(
        persisted_retrieval_score_matches(
            record.retrieval_recall_all_at_10,
            source.retrieval_recall_all_at_10,
        ) && persisted_retrieval_score_matches(
            record.retrieval_ndcg_at_10,
            source.retrieval_ndcg_at_10,
        ),
        "imported destination retrieval scores do not match its pinned source payload"
    );
    ensure!(
        record.memory_wall_s == 0.0
            && record.retrieval_s.is_none()
            && record.processing_s.is_none()
            && record.shared_construction_s.is_none()
            && record.processor_usage == ModelUsage::default(),
        "imported destination memory must remain a zero-cost read-only clone"
    );
    let source_payload_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(source)?));
    let expected_cohort = serde_json::to_value(method.cohort)?;
    ensure!(
        record.metadata.get("kind").and_then(Value::as_str) == Some("imported_memory_context")
            && record.metadata.get("cohort") == Some(&expected_cohort)
            && record.metadata.get("source_method").and_then(Value::as_str)
                == Some(method.source_method.as_str())
            && record
                .metadata
                .get("source_run_key")
                .and_then(Value::as_str)
                == Some(method.source_run_key.as_str())
            && record
                .metadata
                .get("source_journal_sha256")
                .and_then(Value::as_str)
                == Some(method.journal_sha256.as_str())
            && record
                .metadata
                .get("source_summary_sha256")
                .and_then(Value::as_str)
                == Some(method.summary_sha256.as_str())
            && record.metadata.get("source_kind").and_then(Value::as_str)
                == Some(method.source_kind.as_str())
            && record
                .metadata
                .get("source_label_isolation_proven")
                .and_then(Value::as_bool)
                == Some(method.source_label_isolation_proven)
            && record
                .metadata
                .get("source_context_sha256")
                .and_then(Value::as_str)
                == Some(source.context_sha256.as_str())
            && record
                .metadata
                .get("source_payload_sha256")
                .and_then(Value::as_str)
                == Some(source_payload_sha256.as_str())
            && record
                .metadata
                .get("context_reused_verbatim")
                .and_then(Value::as_bool)
                == Some(true)
            && record
                .metadata
                .get("source_metadata_copied")
                .and_then(Value::as_bool)
                == Some(false)
            && record
                .metadata
                .get("live_memory_construction")
                .and_then(Value::as_bool)
                == Some(false)
            && record
                .metadata
                .get("processor_usage_replayed")
                .and_then(Value::as_bool)
                == Some(false),
        "imported destination provenance does not match its pinned source payload"
    );
    Ok(())
}

fn validate_memory_records(
    records: &[LocomoMemoryRecord],
    dataset: &PreparedDataset,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
    run_key: &str,
) -> Result<()> {
    let questions = dataset
        .questions
        .iter()
        .map(|question| (question.input.id.as_str(), question))
        .collect::<HashMap<_, _>>();
    let outputs = output_specs(methods, imported, imported_memories)?;
    let mut pairs = HashSet::new();
    for record in records {
        ensure!(record.schema_version == MEMORY_RECORD_SCHEMA_VERSION);
        ensure!(record.run_key == run_key, "mixed memory run keys");
        let cohort = outputs
            .get(&record.method)
            .context("unknown memory method")?;
        let question = questions
            .get(record.question_id.as_str())
            .context("memory record has unknown question")?;
        ensure!(
            question_in_cohort(question, *cohort),
            "memory record is outside its declared output cohort"
        );
        ensure!(
            record.category == question.evaluator.category.name()
                && record.category_number == question.evaluator.category.number()
                && record.retrieval_eligible == question.evaluator.retrieval.eligible,
            "memory evaluator metadata drift"
        );
        ensure!(
            record.context_bytes == record.context.len()
                && record.context_sha256 == sha256_text(&record.context),
            "memory context identity mismatch"
        );
        ensure!(
            record.memory_wall_s.is_finite()
                && record.memory_wall_s >= 0.0
                && record
                    .retrieval_s
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
                && record
                    .processing_s
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
                && record
                    .shared_construction_s
                    .is_none_or(|value| value.is_finite() && value >= 0.0),
            "memory timing is invalid"
        );
        let corpus = projected_corpus(question);
        validate_ids(&corpus, &record.ranked_message_ids, "ranking")?;
        validate_ids(&corpus, &record.covered_message_ids, "coverage")?;
        validate_ids(&corpus, &record.retained_message_ids, "retained")?;
        let expected = retrieval_scores(question, &record.ranked_message_ids)?;
        ensure!(
            persisted_retrieval_score_matches(record.retrieval_recall_all_at_10, expected.0)
                && persisted_retrieval_score_matches(record.retrieval_ndcg_at_10, expected.1),
            "persisted retrieval score mismatch"
        );
        if let Some(method) = imported_memories
            .iter()
            .find(|method| method.id == record.method)
        {
            validate_imported_destination_memory(record, method)?;
        }
        ensure!(
            pairs.insert((record.method.as_str(), record.question_id.as_str())),
            "duplicate memory method/question record"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_answer_records(
    records: &[LocomoAnswerRecord],
    memories: &[LocomoMemoryRecord],
    dataset: &PreparedDataset,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
    answerer: &AnswerModelConfig,
    run_key: &str,
    memory_run_key: &str,
) -> Result<()> {
    let questions = dataset
        .questions
        .iter()
        .map(|question| (question.input.id.as_str(), question))
        .collect::<HashMap<_, _>>();
    let memory_by_pair = memories
        .iter()
        .map(|memory| {
            (
                (memory.method.as_str(), memory.question_id.as_str()),
                memory,
            )
        })
        .collect::<HashMap<_, _>>();
    let outputs = output_specs(methods, imported, imported_memories)?;
    let mut pairs = HashSet::new();
    for record in records {
        ensure!(record.schema_version == ANSWER_RECORD_SCHEMA_VERSION);
        ensure!(record.run_key == run_key, "mixed answer run keys");
        ensure!(
            record.memory_run_key == memory_run_key,
            "answer record refers to a different memory run"
        );
        let cohort = outputs
            .get(&record.method)
            .context("unknown answer method")?;
        let question = questions
            .get(record.question_id.as_str())
            .context("answer record has unknown question")?;
        ensure!(
            question_in_cohort(question, *cohort),
            "answer record is outside its declared output cohort"
        );
        let memory = memory_by_pair
            .get(&(record.method.as_str(), record.question_id.as_str()))
            .context("answer record has no matching memory")?;
        ensure!(
            record.context_sha256 == memory.context_sha256
                && record.prediction_sha256 == sha256_text(&record.prediction),
            "answer content identity mismatch"
        );
        ensure!(
            record.category == question.evaluator.category.name()
                && record.category_number == question.evaluator.category.number(),
            "answer evaluator metadata drift"
        );
        let reference = question
            .evaluator
            .answer
            .as_str()
            .context("LoCoMo reference is not text")?;
        let expected_f1 =
            score_locomo_answer(question.evaluator.category, &record.prediction, reference);
        let mut persisted_f1 = record.official_f1.clone();
        let score_matches = persisted_score_matches(persisted_f1.score, expected_f1.score);
        persisted_f1.score = expected_f1.score;
        ensure!(
            score_matches && persisted_f1 == expected_f1,
            "answer official F1 mismatch"
        );
        ensure!(
            record.answer_wall_s.is_finite() && record.answer_wall_s >= 0.0,
            "invalid answer timing"
        );
        ensure!(
            record
                .metadata
                .get("response_model")
                .and_then(Value::as_str)
                == Some(answerer.endpoint.model.as_str()),
            "persisted answer response model mismatch"
        );
        ensure!(
            record
                .metadata
                .get("response_provider")
                .and_then(Value::as_str)
                == Some(answerer.expected_provider.as_str()),
            "persisted answer response provider mismatch"
        );
        ensure!(
            record
                .metadata
                .get("prompt_version")
                .and_then(Value::as_str)
                == Some(answerer.protocol.version()),
            "persisted answer prompt version mismatch"
        );
        ensure!(
            record.metadata.get("answer_protocol")
                == Some(&serde_json::to_value(answerer.protocol.identity())?),
            "persisted answer protocol identity mismatch"
        );
        ensure!(
            record
                .metadata
                .get("response_finish_reason")
                .and_then(Value::as_str)
                == Some("stop"),
            "persisted answer did not finish normally"
        );
        if !answerer.enable_thinking {
            ensure!(
                record
                    .metadata
                    .get("response_reasoning_present")
                    .and_then(Value::as_bool)
                    == Some(false)
                    && record
                        .metadata
                        .get("response_reasoning_tokens")
                        .and_then(Value::as_u64)
                        == Some(0),
                "persisted no-thinking answer contains reasoning"
            );
        }
        ensure!(
            record.answer_usage.calls == 1
                && (record.answer_usage.cost.is_some()
                    || (record.answer_usage.prompt_tokens > 0
                        && record.answer_usage.completion_tokens > 0)),
            "persisted answer usage is incomplete"
        );
        ensure!(
            pairs.insert((record.method.as_str(), record.question_id.as_str())),
            "duplicate answer method/question record"
        );
    }
    Ok(())
}

fn persisted_retrieval_score_matches(actual: Option<f64>, expected: Option<f64>) -> bool {
    match (actual, expected) {
        (None, None) => true,
        (Some(actual), Some(expected)) => persisted_score_matches(actual, expected),
        _ => false,
    }
}

fn persisted_score_matches(actual: f64, expected: f64) -> bool {
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    if actual == expected {
        return true;
    }
    let ordered_bits = |value: f64| {
        let bits = value.to_bits();
        if bits & (1_u64 << 63) == 0 {
            bits | (1_u64 << 63)
        } else {
            !bits
        }
    };
    ordered_bits(actual).abs_diff(ordered_bits(expected)) <= 1
}

fn is_complete(
    records: &[LocomoAnswerRecord],
    run_key: &str,
    outputs: &BTreeMap<String, QuestionCohort>,
    dataset: &PreparedDataset,
) -> bool {
    let completed = records
        .iter()
        .filter(|record| record.run_key == run_key)
        .map(|record| (record.method.as_str(), record.question_id.as_str()))
        .collect::<HashSet<_>>();
    let expected_n = outputs
        .values()
        .map(|cohort| match cohort {
            QuestionCohort::AllQa => dataset.questions.len(),
            QuestionCohort::RetrievalEligible => dataset.summary.retrieval_count,
        })
        .sum::<usize>();
    records.len() == expected_n
        && outputs.iter().all(|(method, cohort)| {
            dataset
                .questions
                .iter()
                .filter(|question| match cohort {
                    QuestionCohort::AllQa => true,
                    QuestionCohort::RetrievalEligible => question.evaluator.retrieval.eligible,
                })
                .all(|question| completed.contains(&(method.as_str(), question.input.id.as_str())))
        })
}

fn build_answer_summary(
    suite: &AnswerSuiteConfig,
    memories: &[LocomoMemoryRecord],
    answers: &[LocomoAnswerRecord],
    outputs: &BTreeMap<String, QuestionCohort>,
    dataset: &PreparedDataset,
    complete: bool,
) -> Result<Value> {
    let mut progress = BTreeMap::<String, Value>::new();
    for (output, cohort) in outputs {
        let method_answers = answers
            .iter()
            .filter(|record| record.method == *output)
            .collect::<Vec<_>>();
        let score_mean = (!method_answers.is_empty()).then(|| {
            method_answers
                .iter()
                .map(|record| record.official_f1.score)
                .sum::<f64>()
                / method_answers.len() as f64
        });
        let method_memories = memories
            .iter()
            .filter(|record| record.method == *output)
            .collect::<Vec<_>>();
        let mean_context_bytes = (!method_memories.is_empty()).then(|| {
            method_memories
                .iter()
                .map(|record| record.context_bytes as f64)
                .sum::<f64>()
                / method_memories.len() as f64
        });
        progress.insert(
            output.clone(),
            json!({
                "memory_n": method_memories.len(),
                "answer_n": method_answers.len(),
                "required_n": match cohort {
                    QuestionCohort::AllQa => dataset.questions.len(),
                    QuestionCohort::RetrievalEligible => dataset.summary.retrieval_count,
                },
                "cohort": cohort,
                "partial_mean_f1": score_mean,
                "mean_context_bytes": mean_context_bytes,
            }),
        );
    }
    let official_f1_by_cohort = if complete {
        let eligible_ids = dataset
            .questions
            .iter()
            .filter(|question| question.evaluator.retrieval.eligible)
            .map(|question| question.input.id.as_str())
            .collect::<HashSet<_>>();
        let all_qa_methods = outputs
            .iter()
            .filter_map(|(method, cohort)| {
                (*cohort == QuestionCohort::AllQa).then_some(method.as_str())
            })
            .collect::<HashSet<_>>();
        let all_rows = scalar_rows(
            answers
                .iter()
                .filter(|record| all_qa_methods.contains(record.method.as_str())),
        );
        let eligible_rows = scalar_rows(answers.iter().filter(|record| {
            eligible_ids.contains(record.question_id.as_str())
                && outputs.contains_key(record.method.as_str())
        }));
        let mut cohorts = Map::new();
        cohorts.insert(
            format!("all_qa_{}", dataset.questions.len()),
            json!({
                "question_n": dataset.questions.len(),
                "methods": all_qa_methods.into_iter().collect::<BTreeSet<_>>(),
                "official_f1": build_scalar_report(
                    LOCOMO_OFFICIAL_F1_METRIC,
                    &suite.baseline_method,
                    MetricDirection::HigherIsBetter,
                    &all_rows,
                )?,
            }),
        );
        cohorts.insert(
            format!("retrieval_eligible_{}", eligible_ids.len()),
            json!({
                "question_n": eligible_ids.len(),
                "methods": outputs.keys().collect::<Vec<_>>(),
                "official_f1": build_scalar_report(
                    LOCOMO_OFFICIAL_F1_METRIC,
                    &suite.baseline_method,
                    MetricDirection::HigherIsBetter,
                    &eligible_rows,
                )?,
            }),
        );
        Some(Value::Object(cohorts))
    } else {
        None
    };
    let estimated_usage_cost = estimated_reported_usage_cost(&suite.pricing, memories, answers);
    let processor_usage = aggregate_usage(memories.iter().map(|record| &record.processor_usage));
    let answer_usage = aggregate_usage(answers.iter().map(|record| &record.answer_usage));
    let processor_completion_tokens_per_call = (processor_usage.calls > 0)
        .then(|| processor_usage.completion_tokens as f64 / processor_usage.calls as f64);
    let answer_completion_tokens_per_call = (answer_usage.calls > 0)
        .then(|| answer_usage.completion_tokens as f64 / answer_usage.calls as f64);
    Ok(json!({
        "metric": LOCOMO_OFFICIAL_F1_METRIC,
        "complete": complete,
        "method_specific_denominators": true,
        "no_single_all_methods_denominator": true,
        "progress": progress,
        "official_f1_by_cohort": official_f1_by_cohort,
        "usage": {
            "processor": processor_usage,
            "processor_completion_tokens_per_call": processor_completion_tokens_per_call,
            "answerer": answer_usage,
            "answer_completion_tokens_per_call": answer_completion_tokens_per_call,
            "estimated_or_reported_usd": estimated_usage_cost,
            "configured_fuse_usd": suite.pricing.max_reported_usage_usd,
            "scope": "semafuse_processor_and_common_answerer_only",
            "untracked_costs": ["embedding", "reranker", "failed or uncommitted completion calls whose usage was not journaled"],
        },
    }))
}

fn scalar_rows<'a>(records: impl Iterator<Item = &'a LocomoAnswerRecord>) -> Vec<ScalarMetricRow> {
    records
        .map(|record| ScalarMetricRow {
            method: record.method.clone(),
            question_id: record.question_id.clone(),
            question_type: record.category.clone(),
            value: record.official_f1.score,
        })
        .collect()
}

fn aggregate_usage<'a>(usage: impl Iterator<Item = &'a ModelUsage>) -> ModelUsage {
    let mut aggregate = ModelUsage::default();
    for item in usage {
        aggregate.add_assign(item);
    }
    aggregate
}

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    suite: &AnswerSuiteConfig,
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
    dataset: &PreparedDataset,
    dataset_sha256: &str,
    memory_run_key: &str,
    answer_run_key: &str,
    implementation_sha256: &str,
    options: &LocomoAnswerSuiteRunOptions,
    scheduled_questions: usize,
    complete: bool,
) -> Value {
    json!({
        "schema_version": 1,
        "run_key": answer_run_key,
        "memory_run_key": memory_run_key,
        "suite": suite.name,
        "track": format!("{}-selection/full-answer-development", suite.track.name()),
        "complete": complete,
        "paper_comparable": false,
        "paper_comparability_reason": "uses released LoCoMo category-aware token F1, not the LazyMem paper's binary DeepSeek-V4-Pro judge",
        "benchmark": {
            "name": "LoCoMo",
            "upstream_commit": UPSTREAM_COMMIT,
            "dataset_sha256": dataset_sha256,
            "selection": memory_selection_identity(suite.track, dataset),
        },
        "license": {
            "spdx": DATA_LICENSE,
            "noncommercial_use_acknowledged": true,
            "dataset_redistributed": false,
            "dataset_path_serialized": false,
            "raw_context_artifacts_publishable": false,
        },
        "protocol": {
            "question_denominator": format!(
                "method_specific_cohorts: native_and_semafuse=all_{}; imported_source_rankings=retrieval_eligible_{}",
                dataset.summary.qa_count,
                dataset.summary.retrieval_count,
            ),
            "memory_lifecycle": "one_in_memory_construction_per_source_conversation_per_process_attempt_then_sequential_queries; resume may rebuild the shared index",
            "common_answer_model": suite.answerer.endpoint.model,
            "common_answer_expected_provider": suite.answerer.expected_provider,
            "common_answer_endpoint": suite.answerer.endpoint,
            "answer_protocol": suite.answerer.protocol.identity(),
            "answer_generation_options": {
                "temperature": suite.answerer.options().temperature,
                "top_p": suite.answerer.options().top_p,
                "max_tokens": suite.answerer.options().max_tokens,
                "enable_thinking": suite.answerer.options().enable_thinking,
            },
            "official_f1": {
                "metric": LOCOMO_OFFICIAL_F1_METRIC,
                "upstream_commit": LOCOMO_UPSTREAM_COMMIT,
                "upstream_scorer_sha256": LOCOMO_UPSTREAM_SCORER_SHA256,
            },
            "label_isolation": LABEL_ISOLATION_DECLARATION,
            "imported_memory_source_label_isolation": {
                "source_n": imported_memories.len(),
                "all_proven": imported_memories
                    .iter()
                    .all(|method| method.source_label_isolation_proven),
                "required_declaration": LABEL_ISOLATION_DECLARATION,
            },
            "journaling": "committed memory contexts and answers are independently append-and-sync resumable; calls that fail before a complete response is committed may repeat",
        },
        "execution": {
            "validation_only": options.validate_only,
            "scheduled_question_n": scheduled_questions,
            "question_limit": options.question_limit,
            "question_id_filter_n": options.question_ids.len(),
            "question_id_filter_sha256": (!options.question_ids.is_empty()).then(|| {
                sha256_text(&options.question_ids.iter().cloned().collect::<Vec<_>>().join("\n"))
            }),
            "source_method_filter": options.methods,
            "partial_runs_cannot_set_complete": true,
        },
        "denominators": {
            "qa_n": dataset.summary.qa_count,
            "retrieval_eval_n": dataset.summary.retrieval_count,
        },
        "baseline_method": suite.baseline_method,
        "methods": answer_method_manifest(methods, imported, imported_memories),
        "answer_outputs": answer_output_manifest(methods, imported, imported_memories),
        "pricing_guard": {
            "input_usd_per_million": suite.pricing.input_usd_per_million,
            "output_usd_per_million": suite.pricing.output_usd_per_million,
            "max_reported_usage_usd": suite.pricing.max_reported_usage_usd,
            "fallback_rates_in_run_identity": true,
            "max_reported_usage_usd_policy": "mutable execution fuse excluded from run identity",
            "embedding_reranker_costs_included": false,
            "live_external_workers_allowed": false,
        },
        "implementation": {
            "crate_version": env!("CARGO_PKG_VERSION"),
            "cargo_lock_sha256": cargo_lock_sha256(),
            "benchmark_code_sha256": implementation_sha256,
        },
        "limitations": [
            "The common F1 scorer is official LoCoMo code but is not the binary LLM judge reported by LazyMem.",
            "Embedding, reranker, memory-construction, and answer backends are development substitutions unless a method manifest explicitly proves otherwise.",
            "The local fuse covers Semafuse processor and common-answer calls; embedding and reranker costs are outside it and require reserved account headroom.",
            "Imported Mem0/A-MEM rows reuse only pinned source-message rankings and render raw source messages; they are source-ranking RAG comparisons, not native memory-context or full-method scores.",
            "Imported memory-context rows reuse a byte-identical, SHA-pinned context from a prior run; they do not rerun or reprice that run's memory construction.",
            "A process restart may repeat shared history embedding/index construction and paid processor calls from a question that failed before its main memory record was committed.",
            "Raw memory and prediction journals contain dataset-derived text and remain gitignored local artifacts.",
        ],
        "credential_handling": "only environment-variable names and secret-free endpoint identities are serialized",
    })
}

fn answer_output_manifest(
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
) -> Vec<Value> {
    let mut outputs = all_output_ids(methods)
        .into_iter()
        .map(|id| json!({"id": id, "cohort": QuestionCohort::AllQa}))
        .collect::<Vec<_>>();
    outputs.extend(imported.iter().map(|method| {
        json!({
            "id": method.id,
            "cohort": QuestionCohort::RetrievalEligible,
        })
    }));
    outputs.extend(imported_memories.iter().map(|method| {
        json!({
            "id": method.id,
            "cohort": method.cohort,
        })
    }));
    outputs.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    outputs
}

fn answer_method_manifest(
    methods: &[ResolvedMethod],
    imported: &[ResolvedImportedRanking],
    imported_memories: &[ResolvedImportedMemory],
) -> Vec<Value> {
    let mut manifest = method_manifest(methods);
    let methods_by_id = methods
        .iter()
        .map(|method| (method.id(), method))
        .collect::<HashMap<_, _>>();
    for entry in &mut manifest {
        let Some(method) = entry["id"].as_str().and_then(|id| methods_by_id.get(id)) else {
            continue;
        };
        if let ResolvedMethod::Semafuse { id, stages, .. } = method {
            let stage_outputs = stages
                .iter()
                .map(|stage| format!("{id}::{}", stage.as_str()))
                .collect::<Vec<_>>();
            let mut outputs = vec![id.clone()];
            outputs.extend(stage_outputs.iter().cloned());
            entry["kind"] = json!(SEMAFUSE_FULL_MEMORY_MANIFEST_KIND);
            entry["main_output"] = json!(id);
            entry["stage_outputs"] = json!(stage_outputs);
            entry["outputs"] = json!(outputs);
        }
    }
    manifest.extend(imported.iter().map(|method| {
        json!({
            "id": method.id,
            "kind": "imported_source_ranking_rag",
            "cohort": method.cohort,
            "source_method": method.source_method,
            "source_run_key": method.source_run_key,
            "source_journal_sha256": method.journal_sha256,
            "source_summary_sha256": method.summary_sha256,
            "limit": method.limit,
            "native_memory_reused": false,
            "context_order": "source_chronological",
        })
    }));
    manifest.extend(imported_memories.iter().map(|method| {
        json!({
            "id": method.id,
            "kind": "imported_memory_context",
            "cohort": method.cohort,
            "source_method": method.source_method,
            "source_run_key": method.source_run_key,
            "source_journal_sha256": method.journal_sha256,
            "source_summary_sha256": method.summary_sha256,
            "source_kind": method.source_kind,
            "source_label_isolation_proven": method.source_label_isolation_proven,
            "source_record_n": method.records.len(),
            "context_reused_verbatim": true,
            "source_metadata_copied": false,
            "live_memory_construction": false,
            "processor_usage_replayed": false,
        })
    }));
    manifest.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    manifest
}

fn cargo_lock_sha256() -> String {
    hex::encode(Sha256::digest(include_bytes!("../Cargo.lock")))
}

fn selected_question_ids_sha256(dataset: &PreparedDataset) -> String {
    let mut question_ids = dataset
        .questions
        .iter()
        .map(|question| question.input.id.as_str())
        .collect::<Vec<_>>();
    question_ids.sort_unstable();
    let mut digest = Sha256::new();
    digest.update(b"semafuse-locomo-selected-question-ids/v1");
    digest.update((question_ids.len() as u64).to_be_bytes());
    for question_id in question_ids {
        digest.update((question_id.len() as u64).to_be_bytes());
        digest.update(question_id.as_bytes());
    }
    hex::encode(digest.finalize())
}

fn memory_selection_identity(track: LocomoTrack, dataset: &PreparedDataset) -> Value {
    json!({
        "track": track.name(),
        "conversation_ids": track.conversation_ids(),
        "categories": [1, 2, 3, 4],
        "qa_n": dataset.summary.qa_count,
        "retrieval_eval_n": dataset.summary.retrieval_count,
        "selected_question_ids_sha256": selected_question_ids_sha256(dataset),
    })
}

fn memory_run_key(
    dataset_sha256: &str,
    track: LocomoTrack,
    dataset: &PreparedDataset,
    method_manifest: &[Value],
    pricing: &PricingConfig,
    implementation_sha256: &str,
    cargo_lock_sha256: &str,
) -> Result<String> {
    let manifest = serde_json::to_vec(method_manifest)?;
    let selection = serde_json::to_vec(&memory_selection_identity(track, dataset))?;
    let mut digest = Sha256::new();
    digest.update(b"semafuse-locomo-answer-memory/v4");
    digest.update(dataset_sha256.as_bytes());
    digest.update((selection.len() as u64).to_be_bytes());
    digest.update(selection);
    digest.update(cargo_lock_sha256.as_bytes());
    digest.update(implementation_sha256.as_bytes());
    update_pricing_fallback_identity(&mut digest, pricing);
    digest.update((manifest.len() as u64).to_be_bytes());
    digest.update(manifest);
    Ok(hex::encode(digest.finalize()))
}

fn answer_run_key(
    memory_run_key: &str,
    answerer: &AnswerModelConfig,
    pricing: &PricingConfig,
    baseline_method: &str,
    implementation_sha256: &str,
    cargo_lock_sha256: &str,
) -> Result<String> {
    let protocol = serde_json::to_vec(&answerer.protocol.identity())?;
    let answerer = serde_json::to_vec(answerer)?;
    let mut digest = Sha256::new();
    digest.update(b"semafuse-locomo-common-answer/v3");
    digest.update(memory_run_key.as_bytes());
    digest.update((answerer.len() as u64).to_be_bytes());
    digest.update(answerer);
    update_pricing_fallback_identity(&mut digest, pricing);
    digest.update((baseline_method.len() as u64).to_be_bytes());
    digest.update(baseline_method.as_bytes());
    digest.update(implementation_sha256.as_bytes());
    digest.update(cargo_lock_sha256.as_bytes());
    digest.update((protocol.len() as u64).to_be_bytes());
    digest.update(protocol);
    Ok(hex::encode(digest.finalize()))
}

fn update_pricing_fallback_identity(digest: &mut Sha256, pricing: &PricingConfig) {
    digest.update(b"semafuse-pricing-fallback/v1");
    digest.update(pricing.input_usd_per_million.to_bits().to_be_bytes());
    digest.update(pricing.output_usd_per_million.to_bits().to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config as semafuse_config,
        locomo::{DatasetSummary, EvaluatorOnly, LocomoCategory},
        locomo_suite_common::SemafuseStage,
        longmemeval::{BenchmarkInput, BenchmarkMessage, RetrievalProfileGold},
    };
    use tempfile::tempdir;

    const FINAL_NO_THINKING_SUITE: &str =
        "paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml";

    fn import_test_question(id: &str, message_id: &str, eligible: bool) -> PreparedQuestion {
        PreparedQuestion {
            input: BenchmarkInput {
                id: id.into(),
                dataset: "locomo".into(),
                query: format!("question {id}"),
                query_time: "today".into(),
                messages: vec![BenchmarkMessage {
                    id: message_id.into(),
                    index: 0,
                    role: "user".into(),
                    content: format!("memory {id}"),
                    timestamp: "yesterday".into(),
                    session_id: format!("session-{id}"),
                }],
            },
            evaluator: EvaluatorOnly {
                source_question_id: format!("source-{id}"),
                source_conversation_id: "source-conversation".into(),
                source_qa_index: 0,
                category: LocomoCategory::SingleHop,
                answer: json!(format!("reference {id}")),
                raw_evidence: Vec::new(),
                source_evidence_ids: Vec::new(),
                retrieval: RetrievalProfileGold {
                    eligible,
                    gold_message_ids: if eligible {
                        vec![message_id.into()]
                    } else {
                        Vec::new()
                    },
                    gold_session_ids: if eligible {
                        vec![format!("session-{id}")]
                    } else {
                        Vec::new()
                    },
                },
                sessions: Vec::new(),
                messages: Vec::new(),
                speaker_a: "A".into(),
                speaker_b: "B".into(),
            },
        }
    }

    fn import_test_dataset() -> PreparedDataset {
        PreparedDataset {
            questions: vec![
                import_test_question("eligible", "message-eligible", true),
                import_test_question("not-eligible", "message-not-eligible", false),
            ],
            summary: DatasetSummary {
                conversation_count: 1,
                qa_count: 2,
                retrieval_count: 1,
                category_counts: BTreeMap::from([(LocomoCategory::SingleHop, 2)]),
            },
        }
    }

    fn source_memory(question: &PreparedQuestion, method: &str) -> LocomoMemoryRecord {
        let message_id = question.input.messages[0].id.clone();
        let mut record = build_memory_record(
            "source-run",
            method,
            question,
            vec![message_id.clone()],
            vec![message_id.clone()],
            vec![message_id],
            format!("source context for {}", question.input.id),
            Some(5),
            1.0,
            Some(0.25),
            Some(0.75),
            None,
            ModelUsage {
                prompt_tokens: 10,
                completion_tokens: 2,
                cost: Some(0.01),
                calls: 1,
            },
            Map::from_iter([("private_source_note".into(), json!("must not be copied"))]),
        )
        .unwrap();
        record.metadata.insert(
            "source_reference_marker".into(),
            json!("metadata must remain behind the import boundary"),
        );
        record
    }

    fn source_envelope(record: LocomoMemoryRecord) -> JournalRecord<LocomoMemoryRecord> {
        JournalRecord {
            run_key: record.run_key.clone(),
            method: record.method.clone(),
            question_id: record.question_id.clone(),
            record_kind: MEMORY_RECORD_KIND.into(),
            payload: record,
        }
    }

    fn encode_source_journal(records: &[JournalRecord<LocomoMemoryRecord>]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for record in records {
            bytes.extend(serde_json::to_vec(record).unwrap());
            bytes.push(b'\n');
        }
        bytes
    }

    fn import_memory_config(cohort: QuestionCohort) -> ImportedMemoryConfig {
        ImportedMemoryConfig {
            id: "reused_context".into(),
            source_method: "source-memory".into(),
            journal: "source.jsonl".into(),
            journal_sha256: "will-be-replaced".into(),
            summary: "source-summary.json".into(),
            summary_sha256: "will-be-replaced".into(),
            source_run_key: "source-run".into(),
            cohort,
        }
    }

    fn source_summary_bytes(
        dataset_sha256: &str,
        source_method: &str,
        source_kind: &str,
        label_isolation: &str,
    ) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "run_key": "source-answer-run",
            "summary": {
                "manifest": {
                    "run_key": "source-answer-run",
                    "memory_run_key": "source-run",
                    "benchmark": {"dataset_sha256": dataset_sha256},
                    "protocol": {"label_isolation": label_isolation},
                    "methods": [{
                        "id": source_method,
                        "kind": source_kind,
                        "outputs": [source_method],
                    }],
                },
                "summary": {},
            },
        }))
        .unwrap()
    }

    #[test]
    fn bundled_full_suites_parse_and_bind_prompt_settings() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for (name, expected_track, expected_imports) in [
            ("paper_v2.full.openrouter.toml", LocomoTrack::PaperV2, 2),
            (FINAL_NO_THINKING_SUITE, LocomoTrack::PaperV2, 2),
            (
                "community_full.full-no-thinking.openrouter-low-cost.toml",
                LocomoTrack::CommunityFull,
                0,
            ),
        ] {
            let suite = load_answer_suite(&root.join("benchmarks/locomo/configs").join(name))
                .expect("parse full suite");
            assert_eq!(suite.schema_version, 1);
            assert_eq!(suite.track, expected_track);
            assert_eq!(
                suite.answerer.endpoint.model,
                "qwen/qwen3-30b-a3b-instruct-2507"
            );
            assert_eq!(suite.answerer.expected_provider, "CoreWeave");
            assert_eq!(
                suite.answerer.protocol,
                LocomoAnswerProtocol::PlainV1,
                "existing v3 configs must retain their frozen plain-text reader"
            );
            assert_eq!(suite.methods.len(), 2);
            assert_eq!(suite.imported_rankings.len(), expected_imports);
            assert!(
                suite
                    .imported_rankings
                    .iter()
                    .all(|method| method.id.ends_with("_source_ranking_rag"))
            );
            suite.answerer.validate().unwrap();
            suite.pricing.validate().unwrap();
        }
    }

    #[test]
    fn reconstructive_v4_suite_is_bundled_and_resolves_without_model_calls() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = root
            .join("benchmarks/locomo/configs")
            .join("paper_v2.reconstructive-v4.openrouter.toml");
        let suite = load_answer_suite(&path).expect("parse reconstructive v4 suite");
        assert_eq!(suite.track, LocomoTrack::PaperV2);
        assert_eq!(suite.answerer.protocol, LocomoAnswerProtocol::ConciseJsonV2);
        assert_eq!(suite.methods.len(), 4);
        assert!(suite.imported_rankings.is_empty());
        validate_method_configs(&suite.methods).unwrap();
        suite.answerer.validate().unwrap();
        suite.pricing.validate().unwrap();

        let compatibility = SuiteConfig {
            schema_version: suite.schema_version,
            name: suite.name,
            track: suite.track,
            baseline_method: suite.baseline_method,
            methods: suite.methods,
        };
        let methods = resolve_methods(&compatibility, path.parent().unwrap()).unwrap();
        assert_eq!(methods.len(), 4);
        for method in methods.iter().skip(1) {
            let ResolvedMethod::Semafuse { config, .. } = method else {
                panic!("v4 candidate methods must use Semafuse strategies");
            };
            assert_eq!(config.strategies.recall, "reconstructive");
        }
    }

    #[test]
    fn imported_memory_parser_selects_one_method_and_strict_cohort() {
        let dataset = import_test_dataset();
        let eligible = source_envelope(source_memory(&dataset.questions[0], "source-memory"));
        let ineligible = source_envelope(source_memory(&dataset.questions[1], "source-memory"));
        let other_method = source_envelope(source_memory(&dataset.questions[0], "other-method"));
        let bytes = encode_source_journal(&[other_method, eligible.clone(), ineligible.clone()]);

        let all = parse_imported_memory_journal(
            &bytes,
            &import_memory_config(QuestionCohort::AllQa),
            &dataset,
        )
        .unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.values().all(|record| record.method == "source-memory"));

        let retrieval_only = parse_imported_memory_journal(
            &bytes,
            &import_memory_config(QuestionCohort::RetrievalEligible),
            &dataset,
        )
        .unwrap();
        assert_eq!(retrieval_only.keys().collect::<Vec<_>>(), [&"eligible"]);

        let incomplete = encode_source_journal(&[eligible]);
        let error = parse_imported_memory_journal(
            &incomplete,
            &import_memory_config(QuestionCohort::AllQa),
            &dataset,
        )
        .unwrap_err();
        assert!(error.to_string().contains("completely cover"));

        let duplicate = encode_source_journal(&[ineligible.clone(), ineligible]);
        let error = parse_imported_memory_journal(
            &duplicate,
            &import_memory_config(QuestionCohort::AllQa),
            &dataset,
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate envelope"));
    }

    #[test]
    fn imported_memory_parser_rejects_truncation_and_payload_tampering() {
        let dataset = import_test_dataset();
        let records = dataset
            .questions
            .iter()
            .map(|question| source_envelope(source_memory(question, "source-memory")))
            .collect::<Vec<_>>();
        let config = import_memory_config(QuestionCohort::AllQa);

        let mut truncated = encode_source_journal(&records);
        truncated.pop();
        assert!(
            parse_imported_memory_journal(&truncated, &config, &dataset)
                .unwrap_err()
                .to_string()
                .contains("truncated")
        );

        let reject = |mutated: Vec<JournalRecord<LocomoMemoryRecord>>, needle: &str| {
            let error =
                parse_imported_memory_journal(&encode_source_journal(&mutated), &config, &dataset)
                    .unwrap_err();
            assert!(
                error.to_string().contains(needle),
                "expected {needle:?}, got {error:#}"
            );
        };
        let mut mixed_run = records.clone();
        mixed_run[0].payload.run_key = "tampered-run".into();
        reject(mixed_run, "run key");

        let mut wrong_category = records.clone();
        wrong_category[0].payload.category_number = 1;
        reject(wrong_category, "evaluator metadata");

        let mut wrong_context = records.clone();
        wrong_context[0].payload.context_sha256 = "tampered".into();
        reject(wrong_context, "context identity");

        let mut wrong_id = records.clone();
        wrong_id[0].payload.ranked_message_ids = vec!["unknown-message".into()];
        reject(wrong_id, "unknown ID");

        let mut wrong_score = records;
        wrong_score[0].payload.retrieval_recall_all_at_10 = Some(0.0);
        reject(wrong_score, "retrieval score");
    }

    #[test]
    fn imported_memory_resolver_pins_sha_and_reads_source_read_only() {
        let directory = tempdir().unwrap();
        let dataset = import_test_dataset();
        let dataset_sha256 = "pinned-dataset-sha";
        let records = dataset
            .questions
            .iter()
            .map(|question| source_envelope(source_memory(question, "source-memory")))
            .collect::<Vec<_>>();
        let bytes = encode_source_journal(&records);
        let path = directory.path().join("source.jsonl");
        fs::write(&path, &bytes).unwrap();
        let summary_bytes = source_summary_bytes(
            dataset_sha256,
            "source-memory",
            SEMAFUSE_FULL_MEMORY_MANIFEST_KIND,
            LABEL_ISOLATION_DECLARATION,
        );
        let summary_path = directory.path().join("source-summary.json");
        fs::write(&summary_path, &summary_bytes).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o400)).unwrap();
            fs::set_permissions(&summary_path, fs::Permissions::from_mode(0o400)).unwrap();
        }
        let mut config = import_memory_config(QuestionCohort::AllQa);
        config.journal_sha256 = hex::encode(Sha256::digest(&bytes));
        config.summary_sha256 = hex::encode(Sha256::digest(&summary_bytes));
        let resolved = resolve_imported_memories(
            &[config.clone()],
            directory.path(),
            &dataset,
            dataset_sha256,
        )
        .expect("read-only sources should resolve without being opened for append");
        assert_eq!(resolved[0].records.len(), 2);
        assert_eq!(resolved[0].source_kind, SEMAFUSE_FULL_MEMORY_MANIFEST_KIND);
        assert_eq!(resolved[0].summary_sha256, config.summary_sha256);
        assert!(resolved[0].source_label_isolation_proven);

        config.journal_sha256 = "0".repeat(64);
        let error = resolve_imported_memories(
            &[config.clone()],
            directory.path(),
            &dataset,
            dataset_sha256,
        )
        .unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));

        config.journal_sha256 = hex::encode(Sha256::digest(&bytes));
        config.summary_sha256 = "0".repeat(64);
        let error =
            resolve_imported_memories(&[config], directory.path(), &dataset, dataset_sha256)
                .unwrap_err();
        assert!(error.to_string().contains("summary SHA-256 mismatch"));
    }

    #[test]
    fn imported_memory_summary_binds_dataset_run_method_kind_and_label_isolation() {
        let config = import_memory_config(QuestionCohort::AllQa);
        let dataset_sha256 = "pinned-dataset-sha";
        let valid = source_summary_bytes(
            dataset_sha256,
            "source-memory",
            SEMAFUSE_FULL_MEMORY_MANIFEST_KIND,
            LABEL_ISOLATION_DECLARATION,
        );
        assert_eq!(
            validate_imported_memory_summary(&valid, &config, dataset_sha256).unwrap(),
            SEMAFUSE_FULL_MEMORY_MANIFEST_KIND
        );

        let reject = |mutated: Value, needle: &str| {
            let bytes = serde_json::to_vec(&mutated).unwrap();
            let error =
                validate_imported_memory_summary(&bytes, &config, dataset_sha256).unwrap_err();
            assert!(
                error.to_string().contains(needle),
                "expected {needle:?}, got {error:#}"
            );
        };
        let valid_value: Value = serde_json::from_slice(&valid).unwrap();

        let mut wrong_dataset = valid_value.clone();
        wrong_dataset["summary"]["manifest"]["benchmark"]["dataset_sha256"] =
            json!("different-dataset");
        reject(wrong_dataset, "dataset identity");

        let mut wrong_run = valid_value.clone();
        wrong_run["summary"]["manifest"]["memory_run_key"] = json!("different-run");
        reject(wrong_run, "memory run key");

        let mut wrong_label_protocol = valid_value.clone();
        wrong_label_protocol["summary"]["manifest"]["protocol"]["label_isolation"] =
            json!("unproven");
        reject(wrong_label_protocol, "label-isolation");

        let mut missing_source_method = valid_value.clone();
        missing_source_method["summary"]["manifest"]["methods"][0]["id"] =
            json!("different-method");
        missing_source_method["summary"]["manifest"]["methods"][0]["outputs"] = json!([]);
        reject(missing_source_method, "exactly once");

        let mut missing_kind = valid_value;
        missing_kind["summary"]["manifest"]["methods"][0]
            .as_object_mut()
            .unwrap()
            .remove("kind");
        reject(missing_kind, "omitted its kind");
    }

    #[test]
    fn imported_memory_execution_is_a_filtered_safe_clone_without_live_construction() {
        let dataset = import_test_dataset();
        let source = source_memory(&dataset.questions[0], "source-memory");
        let method = ResolvedImportedMemory {
            id: "reused_context".into(),
            source_method: "source-memory".into(),
            source_run_key: "source-run".into(),
            journal_sha256: "journal-sha".into(),
            summary_sha256: "summary-sha".into(),
            source_kind: SEMAFUSE_FULL_MEMORY_MANIFEST_KIND.into(),
            source_label_isolation_proven: true,
            cohort: QuestionCohort::RetrievalEligible,
            records: BTreeMap::from([("eligible".into(), source.clone())]),
        };
        let questions = dataset.questions.iter().collect::<Vec<_>>();
        let mut output = Vec::new();
        execute_imported_memory_contexts(
            std::slice::from_ref(&method),
            &questions,
            &BTreeSet::from(["reused_context".into()]),
            "new-memory-run",
            &mut None,
            &mut output,
        )
        .unwrap();
        assert_eq!(output.len(), 1);
        let cloned = &output[0];
        assert_eq!(cloned.run_key, "new-memory-run");
        assert_eq!(cloned.method, "reused_context");
        assert_eq!(cloned.context, source.context);
        assert_eq!(cloned.context_sha256, source.context_sha256);
        assert_eq!(cloned.processor_usage, ModelUsage::default());
        assert_eq!(cloned.memory_wall_s, 0.0);
        assert_eq!(cloned.metadata["live_memory_construction"], false);
        assert_eq!(cloned.metadata["source_metadata_copied"], false);
        assert!(!cloned.metadata.contains_key("private_source_note"));
        validate_memory_records(
            &output,
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "new-memory-run",
        )
        .unwrap();

        let mut self_consistent_context_tamper = cloned.clone();
        self_consistent_context_tamper.context = "self-consistent forged context".into();
        self_consistent_context_tamper.context_bytes = self_consistent_context_tamper.context.len();
        self_consistent_context_tamper.context_sha256 =
            sha256_text(&self_consistent_context_tamper.context);
        self_consistent_context_tamper.metadata["source_context_sha256"] =
            json!(self_consistent_context_tamper.context_sha256);
        let error = validate_memory_records(
            &[self_consistent_context_tamper],
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "new-memory-run",
        )
        .unwrap_err();
        assert!(error.to_string().contains("pinned source payload"));

        let mut token_count_tamper = cloned.clone();
        token_count_tamper.context_tokens = Some(6);
        let error = validate_memory_records(
            &[token_count_tamper],
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "new-memory-run",
        )
        .unwrap_err();
        assert!(error.to_string().contains("pinned source payload"));

        let mut provenance_tamper = cloned.clone();
        provenance_tamper.covered_message_ids.clear();
        let error = validate_memory_records(
            &[provenance_tamper],
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "new-memory-run",
        )
        .unwrap_err();
        assert!(error.to_string().contains("retrieval provenance"));

        let mut payload_digest_tamper = cloned.clone();
        payload_digest_tamper.metadata["source_payload_sha256"] = json!("forged-payload-sha");
        let error = validate_memory_records(
            &[payload_digest_tamper],
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "new-memory-run",
        )
        .unwrap_err();
        assert!(error.to_string().contains("destination provenance"));

        let mut filtered = Vec::new();
        execute_imported_memory_contexts(
            &[method],
            &questions,
            &BTreeSet::from(["some-other-method".into()]),
            "new-memory-run",
            &mut None,
            &mut filtered,
        )
        .unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn memory_and_answer_validators_reject_rows_outside_declared_output_cohort() {
        let dataset = import_test_dataset();
        let eligible_source = source_memory(&dataset.questions[0], "source-memory");
        let method = ResolvedImportedMemory {
            id: "reused_context".into(),
            source_method: "source-memory".into(),
            source_run_key: "source-run".into(),
            journal_sha256: "journal-sha".into(),
            summary_sha256: "summary-sha".into(),
            source_kind: SEMAFUSE_FULL_MEMORY_MANIFEST_KIND.into(),
            source_label_isolation_proven: true,
            cohort: QuestionCohort::RetrievalEligible,
            records: BTreeMap::from([("eligible".into(), eligible_source)]),
        };
        let ineligible_question = &dataset.questions[1];
        let message_id = ineligible_question.input.messages[0].id.clone();
        let memory = build_memory_record(
            "memory-run",
            "reused_context",
            ineligible_question,
            vec![message_id.clone()],
            vec![message_id.clone()],
            vec![message_id],
            "otherwise valid extra memory row".into(),
            None,
            0.0,
            None,
            None,
            None,
            ModelUsage::default(),
            Map::new(),
        )
        .unwrap();
        let memory_error = validate_memory_records(
            std::slice::from_ref(&memory),
            &dataset,
            &[],
            &[],
            std::slice::from_ref(&method),
            "memory-run",
        )
        .unwrap_err();
        assert!(memory_error.to_string().contains("output cohort"));

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        let prediction = "answer".to_owned();
        let answer = LocomoAnswerRecord {
            schema_version: ANSWER_RECORD_SCHEMA_VERSION,
            run_key: "answer-run".into(),
            memory_run_key: "memory-run".into(),
            method: "reused_context".into(),
            question_id: ineligible_question.input.id.clone(),
            category: ineligible_question.evaluator.category.name().into(),
            category_number: ineligible_question.evaluator.category.number(),
            context_sha256: memory.context_sha256.clone(),
            prediction_sha256: sha256_text(&prediction),
            official_f1: score_locomo_answer(
                ineligible_question.evaluator.category,
                &prediction,
                ineligible_question.evaluator.answer.as_str().unwrap(),
            ),
            prediction,
            answer_wall_s: 0.0,
            answer_usage: ModelUsage::default(),
            metadata: Map::new(),
        };
        let answer_error = validate_answer_records(
            &[answer],
            &[memory],
            &dataset,
            &[],
            &[],
            &[method],
            &suite.answerer,
            "answer-run",
            "memory-run",
        )
        .unwrap_err();
        assert!(answer_error.to_string().contains("output cohort"));
    }

    #[test]
    fn imported_memory_supports_one_baseline_suite_and_binds_output_identity() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dataset = import_test_dataset();
        let mut suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        suite.methods = vec![MethodConfig::Bm25 {
            id: "bm25".into(),
            limit: 50,
        }];
        suite.baseline_method = "bm25".into();
        validate_method_configs(&suite.methods).unwrap();
        let methods = vec![ResolvedMethod::Bm25 {
            id: "bm25".into(),
            limit: 50,
        }];
        let imported = ResolvedImportedMemory {
            id: "reused_context".into(),
            source_method: "source-memory".into(),
            source_run_key: "source-run".into(),
            journal_sha256: "journal-sha".into(),
            summary_sha256: "summary-sha".into(),
            source_kind: SEMAFUSE_FULL_MEMORY_MANIFEST_KIND.into(),
            source_label_isolation_proven: true,
            cohort: QuestionCohort::AllQa,
            records: BTreeMap::new(),
        };
        validate_answer_outputs(&suite, &methods, &[], std::slice::from_ref(&imported)).unwrap();

        suite.baseline_method = "reused_context".into();
        let error = validate_answer_outputs(&suite, &methods, &[], std::slice::from_ref(&imported))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("native configured BM25 or recent")
        );
        suite.baseline_method = "bm25".into();

        let mut non_semafuse_import = imported.clone();
        non_semafuse_import.source_kind = "bm25".into();
        let error = validate_answer_outputs(
            &suite,
            &methods,
            &[],
            std::slice::from_ref(&non_semafuse_import),
        )
        .unwrap_err();
        assert!(error.to_string().contains("Semafuse memory"));

        let semafuse_baseline_methods = vec![
            ResolvedMethod::Bm25 {
                id: "bm25".into(),
                limit: 50,
            },
            ResolvedMethod::Semafuse {
                id: "strategy".into(),
                config: crate::config::Config::default(),
                config_sha256: "config".into(),
                resolved_config_sha256: "resolved".into(),
                embedding_model: None,
                reranker_model: None,
                limit: 50,
                stages: Vec::new(),
            },
        ];
        suite.baseline_method = "strategy".into();
        let error =
            validate_answer_outputs(&suite, &semafuse_baseline_methods, &[], &[]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("native configured BM25 or recent")
        );
        suite.baseline_method = "bm25".into();

        let outputs = output_specs(&methods, &[], std::slice::from_ref(&imported)).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs["reused_context"], QuestionCohort::AllQa);
        assert!(output_selected(
            &BTreeSet::from(["reused_context".into()]),
            &methods,
            &[],
            std::slice::from_ref(&imported),
            "reused_context",
        ));
        let manifest = answer_method_manifest(&methods, &[], std::slice::from_ref(&imported));
        let entry = manifest
            .iter()
            .find(|entry| entry["id"] == "reused_context")
            .unwrap();
        assert_eq!(entry["kind"], "imported_memory_context");
        assert_eq!(entry["live_memory_construction"], false);
        assert_eq!(entry["source_summary_sha256"], "summary-sha");
        assert_eq!(entry["source_kind"], SEMAFUSE_FULL_MEMORY_MANIFEST_KIND);
        assert_eq!(entry["source_label_isolation_proven"], true);

        let first = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &manifest,
            &suite.pricing,
            "implementation",
            "lock",
        )
        .unwrap();
        let mut changed = imported.clone();
        changed.journal_sha256 = "different-journal-sha".into();
        let second = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &answer_method_manifest(&methods, &[], &[changed]),
            &suite.pricing,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(first, second);

        let mut changed_summary = imported.clone();
        changed_summary.summary_sha256 = "different-summary-sha".into();
        let changed_summary_key = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &answer_method_manifest(&methods, &[], &[changed_summary]),
            &suite.pricing,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(first, changed_summary_key);

        let mut changed_kind = imported;
        changed_kind.source_kind = "bm25".into();
        let changed_kind_key = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &answer_method_manifest(&methods, &[], &[changed_kind]),
            &suite.pricing,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(first, changed_kind_key);
    }

    #[test]
    fn semafuse_full_output_includes_memory_and_shared_stage() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config =
            semafuse_config::load_from_path(root.join(
                "benchmarks/lazymem/configs/semafuse_lazymem.openrouter-full-no-thinking.toml",
            ))
            .unwrap();
        let methods = vec![
            ResolvedMethod::Bm25 {
                id: "bm25".into(),
                limit: 50,
            },
            ResolvedMethod::Semafuse {
                id: "lazymem".into(),
                config,
                config_sha256: "config".into(),
                resolved_config_sha256: "resolved".into(),
                embedding_model: None,
                reranker_model: None,
                limit: 50,
                stages: vec![SemafuseStage::Retrieved],
            },
        ];
        assert_eq!(
            all_output_ids(&methods),
            BTreeSet::from([
                "bm25".to_owned(),
                "lazymem".to_owned(),
                "lazymem::retrieved".to_owned(),
            ])
        );
        assert!(output_selected(
            &BTreeSet::new(),
            &methods,
            &[],
            &[],
            "lazymem::retrieved"
        ));
        assert!(output_selected(
            &BTreeSet::from(["lazymem".to_owned()]),
            &methods,
            &[],
            &[],
            "lazymem::retrieved"
        ));
        assert!(!output_selected(
            &BTreeSet::from(["bm25".to_owned()]),
            &methods,
            &[],
            &[],
            "lazymem::retrieved"
        ));
        let manifest = answer_method_manifest(&methods, &[], &[]);
        let semafuse = manifest
            .iter()
            .find(|entry| entry["id"] == "lazymem")
            .unwrap();
        assert_eq!(semafuse["kind"], "semafuse_full_memory_and_retrieval");
        assert_eq!(semafuse["main_output"], "lazymem");
        assert_eq!(semafuse["stage_outputs"], json!(["lazymem::retrieved"]));
        assert_eq!(
            semafuse["outputs"],
            json!(["lazymem", "lazymem::retrieved"])
        );
    }

    #[test]
    fn full_answer_accepts_a_generic_semafuse_method_without_stage_outputs() {
        let methods = vec![
            ResolvedMethod::Bm25 {
                id: "bm25".into(),
                limit: 50,
            },
            ResolvedMethod::Semafuse {
                id: "reconstructive".into(),
                config: crate::config::Config::default(),
                config_sha256: "config".into(),
                resolved_config_sha256: "resolved".into(),
                embedding_model: Some("embedding".into()),
                reranker_model: Some("reranker".into()),
                limit: 50,
                stages: Vec::new(),
            },
        ];
        assert_eq!(
            all_output_ids(&methods),
            BTreeSet::from(["bm25".to_owned(), "reconstructive".to_owned()])
        );
        let manifest = answer_method_manifest(&methods, &[], &[]);
        let semafuse = manifest
            .iter()
            .find(|entry| entry["id"] == "reconstructive")
            .unwrap();
        assert_eq!(semafuse["main_output"], "reconstructive");
        assert_eq!(semafuse["stage_outputs"], json!([]));
        assert_eq!(semafuse["outputs"], json!(["reconstructive"]));
        assert_eq!(SEMAFUSE_MEMORY_KIND, "semafuse_strategy_full");
    }

    #[test]
    fn full_answer_manifest_joins_sorted_entries_to_methods_by_id() {
        let methods = vec![
            ResolvedMethod::Bm25 {
                id: "middle_baseline".into(),
                limit: 50,
            },
            ResolvedMethod::Semafuse {
                id: "z_strategy".into(),
                config: crate::config::Config::default(),
                config_sha256: "z_config".into(),
                resolved_config_sha256: "z_resolved".into(),
                embedding_model: None,
                reranker_model: None,
                limit: 50,
                stages: Vec::new(),
            },
            ResolvedMethod::Semafuse {
                id: "a_strategy".into(),
                config: crate::config::Config::default(),
                config_sha256: "a_config".into(),
                resolved_config_sha256: "a_resolved".into(),
                embedding_model: None,
                reranker_model: None,
                limit: 50,
                stages: vec![SemafuseStage::Retrieved],
            },
        ];
        let manifest = answer_method_manifest(&methods, &[], &[]);
        let a = manifest
            .iter()
            .find(|entry| entry["id"] == "a_strategy")
            .unwrap();
        let z = manifest
            .iter()
            .find(|entry| entry["id"] == "z_strategy")
            .unwrap();
        let baseline = manifest
            .iter()
            .find(|entry| entry["id"] == "middle_baseline")
            .unwrap();
        assert_eq!(a["config_sha256"], "a_config");
        assert_eq!(a["main_output"], "a_strategy");
        assert_eq!(a["stage_outputs"], json!(["a_strategy::retrieved"]));
        assert_eq!(z["config_sha256"], "z_config");
        assert_eq!(z["main_output"], "z_strategy");
        assert_eq!(baseline["kind"], "bm25");
        assert!(baseline.get("main_output").is_none());
    }

    #[test]
    fn ranked_context_is_source_chronological_not_score_order() {
        let corpus = vec![
            BaselineMessage {
                id: "early".into(),
                role: "user".into(),
                content: "first".into(),
                session_id: "s".into(),
                corpus_index: 0,
            },
            BaselineMessage {
                id: "late".into(),
                role: "assistant".into(),
                content: "second".into(),
                session_id: "s".into(),
                corpus_index: 1,
            },
        ];
        let rendered =
            render_ranked_context(&corpus, &["late".to_owned(), "early".to_owned()]).unwrap();
        assert!(rendered.find("first").unwrap() < rendered.find("second").unwrap());
    }

    #[test]
    fn usage_aggregation_does_not_erase_cost_on_shared_zero_rows() {
        let paid = ModelUsage {
            prompt_tokens: 100,
            completion_tokens: 20,
            cost: Some(0.25),
            calls: 2,
        };
        let zero = ModelUsage::default();
        let aggregate = aggregate_usage([&paid, &zero].into_iter());
        assert_eq!(aggregate.prompt_tokens, 100);
        assert_eq!(aggregate.completion_tokens, 20);
        assert_eq!(aggregate.calls, 2);
        assert_eq!(aggregate.cost, Some(0.25));
    }

    #[test]
    fn processor_usage_reads_nested_strategy_trace_and_logical_responses() {
        let metadata = json!({
            "strategy": {
                "processing": {
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "cost": 0.004,
                        "calls": 2,
                        "calls_missing_prompt_tokens": 0,
                        "calls_missing_completion_tokens": 0,
                        "calls_missing_cost": 0
                    },
                    "processor_calls": [
                        {"calls": [{"attempt": 1}, {"attempt": 2}]}
                    ]
                }
            }
        });
        let usage = processor_usage(metadata.as_object().unwrap()).unwrap();
        assert_eq!(usage.prompt_tokens, 120);
        assert_eq!(usage.completion_tokens, 30);
        assert_eq!(usage.cost, Some(0.004));
        assert_eq!(usage.calls, 2);
    }

    #[test]
    fn processor_usage_rejects_unbounded_missing_telemetry() {
        let metadata = json!({
            "strategy": {
                "processing": {
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": null,
                        "calls": 1,
                        "calls_missing_prompt_tokens": 1,
                        "calls_missing_completion_tokens": 1,
                        "calls_missing_cost": 1
                    },
                    "processor_calls": [{"calls": [{}]}]
                }
            }
        });
        assert!(processor_usage(metadata.as_object().unwrap()).is_err());
    }

    #[test]
    fn budget_prefers_gateway_cost_and_falls_back_to_pinned_prices() {
        let pricing = PricingConfig {
            input_usd_per_million: 1.0,
            output_usd_per_million: 2.0,
            max_reported_usage_usd: 3.0,
        };
        let reported = ModelUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            cost: Some(0.5),
            calls: 1,
        };
        assert_eq!(pricing.estimate(&reported), 0.5);
        let estimated = ModelUsage {
            cost: None,
            ..reported
        };
        assert_eq!(pricing.estimate(&estimated), 3.0);
    }

    #[test]
    fn persisted_retrieval_scores_allow_one_ulp_but_preserve_option_shape() {
        let expected = 0.5_f64;
        let one_ulp_above = f64::from_bits(expected.to_bits() + 1);
        let material_drift = f64::from_bits(expected.to_bits() + 2);

        assert!(persisted_retrieval_score_matches(
            Some(one_ulp_above),
            Some(expected)
        ));
        assert!(!persisted_retrieval_score_matches(
            Some(material_drift),
            Some(expected)
        ));
        assert!(persisted_retrieval_score_matches(None, None));
        assert!(!persisted_retrieval_score_matches(Some(0.0), None));
        assert!(!persisted_retrieval_score_matches(None, Some(0.0)));
    }

    #[test]
    fn persisted_answer_scores_allow_only_one_ulp_of_json_roundtrip_drift() {
        let expected = 0.095_238_095_238_095_23_f64;
        let one_ulp_above = f64::from_bits(expected.to_bits() + 1);
        let material_drift = f64::from_bits(expected.to_bits() + 2);

        assert!(persisted_score_matches(expected, expected));
        assert!(persisted_score_matches(one_ulp_above, expected));
        assert!(!persisted_score_matches(material_drift, expected));
        assert!(!persisted_score_matches(f64::NAN, expected));
    }

    #[test]
    fn memory_run_key_separates_paper_v2_community_full_and_selected_question_ids() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        let dataset = import_test_dataset();
        let key = |track, dataset: &PreparedDataset| {
            memory_run_key(
                "same-full-dataset-sha",
                track,
                dataset,
                &[],
                &suite.pricing,
                "same-implementation",
                "same-lock",
            )
            .unwrap()
        };
        let paper_v2 = key(LocomoTrack::PaperV2, &dataset);
        let community_full = key(LocomoTrack::CommunityFull, &dataset);
        assert_ne!(paper_v2, community_full);

        let selection = memory_selection_identity(LocomoTrack::PaperV2, &dataset);
        assert_eq!(selection["track"], "paper_v2");
        assert_eq!(selection["qa_n"], 2);
        assert_eq!(selection["retrieval_eval_n"], 1);
        assert_eq!(selection["conversation_ids"], json!(["conv-49", "conv-50"]));
        assert_eq!(
            selection["selected_question_ids_sha256"],
            selected_question_ids_sha256(&dataset)
        );

        let mut changed_questions = dataset.clone();
        changed_questions.questions[0].input.id = "different-opaque-question-id".into();
        assert_ne!(paper_v2, key(LocomoTrack::PaperV2, &changed_questions));
    }

    #[test]
    fn run_keys_bind_fallback_rates_but_exclude_mutable_budget_fuse() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        let mut changed_rate = suite.pricing.clone();
        changed_rate.input_usd_per_million *= 2.0;
        let mut changed_fuse = suite.pricing.clone();
        changed_fuse.max_reported_usage_usd *= 2.0;
        let dataset = import_test_dataset();

        let memory_key = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &[],
            &suite.pricing,
            "implementation",
            "lock",
        )
        .unwrap();
        let changed_rate_memory_key = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &[],
            &changed_rate,
            "implementation",
            "lock",
        )
        .unwrap();
        let changed_fuse_memory_key = memory_run_key(
            "dataset",
            LocomoTrack::PaperV2,
            &dataset,
            &[],
            &changed_fuse,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(memory_key, changed_rate_memory_key);
        assert_eq!(memory_key, changed_fuse_memory_key);

        let answer_key = answer_run_key(
            &memory_key,
            &suite.answerer,
            &suite.pricing,
            &suite.baseline_method,
            "implementation",
            "lock",
        )
        .unwrap();
        let changed_rate_answer_key = answer_run_key(
            &memory_key,
            &suite.answerer,
            &changed_rate,
            &suite.baseline_method,
            "implementation",
            "lock",
        )
        .unwrap();
        let changed_fuse_answer_key = answer_run_key(
            &memory_key,
            &suite.answerer,
            &changed_fuse,
            &suite.baseline_method,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(answer_key, changed_rate_answer_key);
        assert_eq!(answer_key, changed_fuse_answer_key);

        let changed_baseline_answer_key = answer_run_key(
            &memory_key,
            &suite.answerer,
            &suite.pricing,
            "recent",
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(answer_key, changed_baseline_answer_key);

        let mut concise_answerer = suite.answerer.clone();
        concise_answerer.protocol = LocomoAnswerProtocol::ConciseJsonV2;
        let concise_answer_key = answer_run_key(
            &memory_key,
            &concise_answerer,
            &suite.pricing,
            &suite.baseline_method,
            "implementation",
            "lock",
        )
        .unwrap();
        assert_ne!(answer_key, concise_answer_key);
    }

    #[test]
    fn answerer_rejects_unsafe_expected_provider_names() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        for invalid in [
            "",
            " StreamLake",
            "StreamLake ",
            "bad/provider",
            "bad\nprovider",
        ] {
            let mut answerer = suite.answerer.clone();
            answerer.expected_provider = invalid.to_owned();
            assert!(answerer.validate().is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    fn persisted_answers_must_match_expected_model_and_provider() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let suite = load_answer_suite(
            &root
                .join("benchmarks/locomo/configs")
                .join(FINAL_NO_THINKING_SUITE),
        )
        .unwrap();
        let question = PreparedQuestion {
            input: BenchmarkInput {
                id: "question".into(),
                dataset: "locomo".into(),
                query: "question".into(),
                query_time: "today".into(),
                messages: vec![BenchmarkMessage {
                    id: "message".into(),
                    index: 0,
                    role: "user".into(),
                    content: "memory".into(),
                    timestamp: "yesterday".into(),
                    session_id: "session".into(),
                }],
            },
            evaluator: EvaluatorOnly {
                source_question_id: "source-question".into(),
                source_conversation_id: "source-conversation".into(),
                source_qa_index: 0,
                category: LocomoCategory::SingleHop,
                answer: json!("answer"),
                raw_evidence: Vec::new(),
                source_evidence_ids: Vec::new(),
                retrieval: RetrievalProfileGold {
                    eligible: false,
                    gold_message_ids: Vec::new(),
                    gold_session_ids: Vec::new(),
                },
                sessions: Vec::new(),
                messages: Vec::new(),
                speaker_a: "A".into(),
                speaker_b: "B".into(),
            },
        };
        let dataset = PreparedDataset {
            questions: vec![question.clone()],
            summary: DatasetSummary {
                conversation_count: 1,
                qa_count: 1,
                retrieval_count: 0,
                category_counts: BTreeMap::from([(LocomoCategory::SingleHop, 1)]),
            },
        };
        let methods = vec![ResolvedMethod::Bm25 {
            id: "bm25".into(),
            limit: 1,
        }];
        let memory = build_memory_record(
            "memory-run",
            "bm25",
            &question,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "memory".into(),
            None,
            0.0,
            None,
            None,
            None,
            ModelUsage::default(),
            Map::new(),
        )
        .unwrap();
        let prediction = "answer".to_owned();
        let mut answer = LocomoAnswerRecord {
            schema_version: ANSWER_RECORD_SCHEMA_VERSION,
            run_key: "answer-run".into(),
            memory_run_key: "memory-run".into(),
            method: "bm25".into(),
            question_id: "question".into(),
            category: "single-hop".into(),
            category_number: 4,
            context_sha256: memory.context_sha256.clone(),
            prediction_sha256: sha256_text(&prediction),
            official_f1: score_locomo_answer(LocomoCategory::SingleHop, &prediction, "answer"),
            prediction,
            answer_wall_s: 0.0,
            answer_usage: ModelUsage {
                prompt_tokens: 10,
                completion_tokens: 1,
                cost: Some(0.001),
                calls: 1,
            },
            metadata: Map::from_iter([
                (
                    "prompt_version".into(),
                    json!(suite.answerer.protocol.version()),
                ),
                (
                    "answer_protocol".into(),
                    serde_json::to_value(suite.answerer.protocol.identity()).unwrap(),
                ),
                (
                    "response_model".into(),
                    json!(suite.answerer.endpoint.model),
                ),
                (
                    "response_provider".into(),
                    json!(suite.answerer.expected_provider),
                ),
                ("response_finish_reason".into(), json!("stop")),
                ("response_reasoning_tokens".into(), json!(0)),
                ("response_reasoning_present".into(), json!(false)),
            ]),
        };
        let validate = |answer: &LocomoAnswerRecord| {
            validate_answer_records(
                std::slice::from_ref(answer),
                std::slice::from_ref(&memory),
                &dataset,
                &methods,
                &[],
                &[],
                &suite.answerer,
                "answer-run",
                "memory-run",
            )
        };
        validate(&answer).unwrap();

        answer.metadata["response_provider"] = json!("DifferentProvider");
        assert!(
            validate(&answer)
                .unwrap_err()
                .to_string()
                .contains("provider")
        );
        answer.metadata["response_provider"] = json!(suite.answerer.expected_provider);
        answer.metadata["response_model"] = json!("different/model");
        assert!(validate(&answer).unwrap_err().to_string().contains("model"));
    }
}
