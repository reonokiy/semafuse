//! Optional benchmark commands for Semafuse strategies and datasets.

use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand, ValueEnum};
use semafuse::{
    benchmark::{BenchmarkQuestion, run_strategy_question, run_strategy_retrieval_question},
    benchmark_journal::{JsonlJournal, SummaryEnvelope, read_summary, write_summary_atomic},
    benchmark_suite::{
        BenchmarkMode, QuestionMethodRecord, RetrievalProfile, SuiteRunOptions,
        build_summary_from_records, run_longmemeval_suite,
    },
    config,
    locomo_suite::{LocomoSuiteRunOptions, run_locomo_retrieval_suite},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Subcommand)]
pub(crate) enum BenchmarkCommand {
    /// Stream normalized JSONL questions through the strategy selected by configuration.
    Jsonl(JsonlArgs),
    /// Run the native LongMemEval comparison suite.
    #[command(name = "longmemeval")]
    LongMemEval(LongMemEvalArgs),
    /// Run the native LoCoMo retrieval comparison suite.
    Locomo(LocomoArgs),
    /// Rebuild a curated comparison from an immutable benchmark journal.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
pub(crate) struct JsonlArgs {
    /// Semafuse TOML configuration selecting the strategy under test.
    #[arg(long)]
    config: PathBuf,
    /// Run indexing, retrieval, reranking, and evidence expansion without processing.
    #[arg(long)]
    retrieval_only: bool,
    /// Maximum number of final retrieval anchors.
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    Retrieval,
    Full,
}

impl From<CliMode> for BenchmarkMode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Retrieval => Self::Retrieval,
            CliMode::Full => Self::Full,
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct LongMemEvalArgs {
    /// Modular benchmark suite TOML.
    #[arg(long)]
    suite: PathBuf,
    /// Official LongMemEval cleaned JSON file.
    #[arg(long)]
    dataset: PathBuf,
    /// Optional frozen JSON split manifest; omit for the full 500 questions.
    #[arg(long)]
    split_manifest: Option<PathBuf>,
    /// Split key inside the manifest.
    #[arg(long, default_value = "test")]
    split: String,
    /// Retrieval-only or complete answer plus official judge evaluation.
    #[arg(long, value_enum, default_value = "retrieval")]
    mode: CliMode,
    /// Crash-safe output directory (`per-question.jsonl`, `summary.json`).
    #[arg(long)]
    output_dir: PathBuf,
    /// Do not load an optional `.env` from the current directory.
    #[arg(long)]
    no_dotenv: bool,
}

#[derive(Debug, Args)]
pub(crate) struct LocomoArgs {
    /// Modular LoCoMo suite TOML.
    #[arg(long)]
    suite: PathBuf,
    /// Official pinned `data/locomo10.json` file; it is never copied.
    #[arg(long)]
    dataset: PathBuf,
    /// Confirm that this CC-BY-NC-4.0 run is permitted non-commercial use.
    #[arg(long)]
    acknowledge_noncommercial: bool,
    /// Crash-safe output directory (`per-question.jsonl`, `summary.json`).
    #[arg(long)]
    output_dir: PathBuf,
    /// Do not load an optional `.env` from the current directory.
    #[arg(long)]
    no_dotenv: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ReportArgs {
    /// Directory containing `summary.json` and `per-question.jsonl`.
    #[arg(long)]
    run_dir: PathBuf,
    /// Destination; defaults to `<run-dir>/comparison.json`.
    #[arg(long)]
    output: Option<PathBuf>,
}

pub(crate) async fn run(command: BenchmarkCommand) -> Result<()> {
    match command {
        BenchmarkCommand::Jsonl(args) => run_jsonl(args).await,
        BenchmarkCommand::LongMemEval(args) => run_longmemeval(args).await,
        BenchmarkCommand::Locomo(args) => run_locomo(args).await,
        BenchmarkCommand::Report(args) => run_report(args),
    }
}

async fn run_jsonl(args: JsonlArgs) -> Result<()> {
    let config = config::load_from_path(&args.config)
        .with_context(|| format!("load benchmark configuration {}", args.config.display()))?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for (line_number, line) in stdin.lock().lines().enumerate() {
        let line = line.context("read benchmark JSONL input")?;
        if line.trim().is_empty() {
            continue;
        }
        let question: BenchmarkQuestion = serde_json::from_str(&line)
            .with_context(|| format!("parse benchmark JSONL line {}", line_number + 1))?;
        let response = if args.retrieval_only {
            run_strategy_retrieval_question(&config, question, args.limit).await
        } else {
            run_strategy_question(&config, question, args.limit).await
        }
        .with_context(|| format!("evaluate benchmark JSONL line {}", line_number + 1))?;
        serde_json::to_writer(&mut output, &response)
            .with_context(|| format!("serialize benchmark JSONL line {}", line_number + 1))?;
        output
            .write_all(b"\n")
            .with_context(|| format!("write benchmark JSONL line {}", line_number + 1))?;
        output
            .flush()
            .with_context(|| format!("flush benchmark JSONL line {}", line_number + 1))?;
    }
    Ok(())
}

async fn run_longmemeval(args: LongMemEvalArgs) -> Result<()> {
    if !args.no_dotenv {
        let _ = dotenvy::dotenv();
    }
    let artifacts = run_longmemeval_suite(&SuiteRunOptions {
        suite_path: args.suite,
        dataset_path: args.dataset,
        split_manifest: args.split_manifest,
        split_name: args.split,
        mode: args.mode.into(),
        output_dir: Some(args.output_dir.clone()),
    })
    .await?;
    println!(
        "completed run {}: {} records; summary: {}",
        artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
        artifacts.records.len(),
        args.output_dir.join("summary.json").display()
    );
    Ok(())
}

async fn run_locomo(args: LocomoArgs) -> Result<()> {
    if !args.no_dotenv {
        let _ = dotenvy::dotenv();
    }
    let artifacts = run_locomo_retrieval_suite(&LocomoSuiteRunOptions {
        suite_path: args.suite,
        dataset_path: args.dataset,
        acknowledge_noncommercial: args.acknowledge_noncommercial,
        output_dir: Some(args.output_dir.clone()),
    })
    .await?;
    println!(
        "completed LoCoMo run {}: {} records; summary: {}",
        artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
        artifacts.records.len(),
        args.output_dir.join("summary.json").display()
    );
    Ok(())
}

fn run_report(args: ReportArgs) -> Result<()> {
    let source_summary_path = args.run_dir.join("summary.json");
    let journal_path = args.run_dir.join("per-question.jsonl");
    let output_path = args
        .output
        .unwrap_or_else(|| args.run_dir.join("comparison.json"));
    ensure!(
        output_path != source_summary_path,
        "comparison output must not replace the immutable source summary"
    );
    ensure!(
        journal_path.is_file(),
        "source journal does not exist or is not a regular file"
    );

    let source = read_summary::<Value>(&source_summary_path)
        .with_context(|| format!("read {}", source_summary_path.display()))?;
    let manifest = source
        .summary
        .get("manifest")
        .and_then(Value::as_object)
        .context("source summary has no manifest object")?;
    ensure!(
        manifest.get("run_key").and_then(Value::as_str) == Some(source.run_key.as_str()),
        "source summary envelope and manifest use different run keys"
    );
    let previous_aggregate = source
        .summary
        .get("summary")
        .and_then(Value::as_object)
        .context("source summary has no aggregate object")?;
    let suite_name = manifest
        .get("suite")
        .and_then(Value::as_str)
        .context("source manifest has no suite name")?;
    let baseline_method = previous_aggregate
        .get("baseline_method")
        .and_then(Value::as_str)
        .context("source aggregate has no baseline_method")?;
    let mode: BenchmarkMode = serde_json::from_value(
        manifest
            .get("mode")
            .cloned()
            .context("source manifest has no mode")?,
    )
    .context("source manifest has invalid mode")?;
    let retrieval_profile: RetrievalProfile = serde_json::from_value(
        manifest
            .get("retrieval_profile")
            .cloned()
            .context("source manifest has no retrieval_profile")?,
    )
    .context("source manifest has invalid retrieval_profile")?;

    let journal = JsonlJournal::<QuestionMethodRecord>::open(&journal_path, &source.run_key)
        .with_context(|| format!("validate {}", journal_path.display()))?;
    let expected_record_kind = match mode {
        BenchmarkMode::Retrieval => "retrieval",
        BenchmarkMode::Full => "full",
    };
    let mut records = Vec::with_capacity(journal.records().len());
    for envelope in journal.records() {
        ensure!(
            envelope.record_kind == expected_record_kind,
            "journal record kind does not match source mode"
        );
        ensure!(
            envelope.run_key == envelope.payload.run_key
                && envelope.method == envelope.payload.method
                && envelope.question_id == envelope.payload.question_id,
            "journal envelope and payload identities differ"
        );
        records.push(envelope.payload.clone());
    }

    let expected_methods = manifest
        .get("methods")
        .and_then(Value::as_array)
        .context("source manifest has no methods array")?
        .iter()
        .try_fold(BTreeSet::new(), |mut ids, method| -> Result<_> {
            if let Some(output_ids) = method.get("output_ids").and_then(Value::as_array) {
                for output_id in output_ids {
                    ids.insert(
                        output_id
                            .as_str()
                            .context("source output_id is not a string")?
                            .to_owned(),
                    );
                }
            } else {
                ids.insert(
                    method
                        .get("id")
                        .and_then(Value::as_str)
                        .context("source method has neither output_ids nor id")?
                        .to_owned(),
                );
            }
            Ok(ids)
        })?;
    let actual_methods = records
        .iter()
        .map(|record| record.method.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        !expected_methods.is_empty(),
        "source manifest has no methods"
    );
    ensure!(
        actual_methods == expected_methods,
        "journal method set differs from source manifest"
    );
    let denominator_name = match mode {
        BenchmarkMode::Retrieval => "retrieval_eval_n",
        BenchmarkMode::Full => "answer_eval_n",
    };
    let denominator = manifest
        .get("denominators")
        .and_then(|value| value.get(denominator_name))
        .and_then(Value::as_u64)
        .context("source manifest has no valid evaluation denominator")?
        as usize;
    ensure!(
        records.len() == denominator * expected_methods.len(),
        "journal record count is incomplete for the source manifest"
    );

    let comparison = build_summary_from_records(
        suite_name,
        baseline_method,
        mode,
        retrieval_profile,
        &records,
    )?;
    let source_summary_sha256 = sha256_file(&source_summary_path)?;
    let journal_sha256 = sha256_file(&journal_path)?;
    let report = SummaryEnvelope {
        run_key: source.run_key.clone(),
        summary: json!({
            "schema_version": 1,
            "kind": "curated_comparison",
            "source": {
                "run_key": source.run_key,
                "summary_sha256": source_summary_sha256,
                "journal_sha256": journal_sha256,
            },
            "report_implementation": {
                "crate_version": env!("CARGO_PKG_VERSION"),
                "code_sha256": report_code_sha256(),
            },
            "manifest": manifest,
            "comparison": comparison,
        }),
    };
    write_summary_atomic(&output_path, &report)
        .with_context(|| format!("write {}", output_path.display()))?;
    println!(
        "rebuilt curated comparison for run {} from {} records: {}",
        report.run_key,
        records.len(),
        output_path.display()
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("hash {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn report_code_sha256() -> String {
    let mut digest = Sha256::new();
    for (path, bytes) in [
        (
            "src/benchmark_cli.rs",
            include_bytes!("benchmark_cli.rs").as_slice(),
        ),
        (
            "src/benchmark_suite.rs",
            include_bytes!("benchmark_suite.rs").as_slice(),
        ),
        (
            "src/benchmark_report.rs",
            include_bytes!("benchmark_report.rs").as_slice(),
        ),
    ] {
        digest.update((path.len() as u64).to_be_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}
