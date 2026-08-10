//! Release-mode native LoCoMo paper-v2 retrieval and full-answer runner.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use semafuse::{
    locomo_answer_suite::{LocomoAnswerSuiteRunOptions, run_locomo_answer_suite},
    locomo_suite::{LocomoSuiteRunOptions, run_locomo_retrieval_suite},
};
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Mode {
    #[default]
    Retrieval,
    Full,
}

/// Run pinned LoCoMo retrieval metrics or a common-reader memory comparison.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Modular LoCoMo suite TOML.
    #[arg(long)]
    suite: PathBuf,
    /// Official pinned `data/locomo10.json` file; it is never copied.
    #[arg(long)]
    dataset: PathBuf,
    /// Confirm that this CC-BY-NC-4.0 run is permitted non-commercial use.
    #[arg(long)]
    acknowledge_noncommercial: bool,
    /// Output directory. Retrieval writes per-question.jsonl; full mode writes
    /// memory-contexts.jsonl, answers.jsonl, and summary.json.
    #[arg(long)]
    output_dir: PathBuf,
    /// Retrieval-only evidence metrics or complete memory -> answer -> F1.
    #[arg(long, value_enum, default_value_t = Mode::Retrieval)]
    mode: Mode,
    /// Deterministic stratified pilot size for full mode. A later unrestricted
    /// run resumes the same journals and is the only way to mark them complete.
    #[arg(long)]
    question_limit: Option<usize>,
    /// Full mode only: run exact opaque evaluator question ID(s). Repeatable;
    /// mutually exclusive with --question-limit and excluded from run identity.
    #[arg(long = "question-id")]
    question_ids: Vec<String>,
    /// Source method(s) to schedule in full mode; repeat for a focused pilot.
    #[arg(long = "method")]
    methods: Vec<String>,
    /// Full mode only: validate dataset, configs, imports, and run identities
    /// without constructing memory or calling any model endpoint.
    #[arg(long)]
    validate_only: bool,
    /// Do not load an optional `.env` from the current directory.
    #[arg(long)]
    no_dotenv: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.no_dotenv {
        let _ = dotenvy::dotenv();
    }
    match cli.mode {
        Mode::Retrieval => {
            anyhow::ensure!(
                !cli.validate_only,
                "--validate-only is available only with --mode full"
            );
            let artifacts = run_locomo_retrieval_suite(&LocomoSuiteRunOptions {
                suite_path: cli.suite,
                dataset_path: cli.dataset,
                acknowledge_noncommercial: cli.acknowledge_noncommercial,
                output_dir: Some(cli.output_dir.clone()),
            })
            .await?;
            println!(
                "completed LoCoMo retrieval run {}: {} records; summary: {}",
                artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
                artifacts.records.len(),
                cli.output_dir.join("summary.json").display()
            );
        }
        Mode::Full => {
            let artifacts = run_locomo_answer_suite(&LocomoAnswerSuiteRunOptions {
                suite_path: cli.suite,
                dataset_path: cli.dataset,
                acknowledge_noncommercial: cli.acknowledge_noncommercial,
                output_dir: Some(cli.output_dir.clone()),
                question_limit: cli.question_limit,
                question_ids: cli.question_ids.into_iter().collect::<BTreeSet<_>>(),
                methods: cli.methods.into_iter().collect::<BTreeSet<_>>(),
                validate_only: cli.validate_only,
            })
            .await?;
            println!(
                "completed LoCoMo full run {}: {} memories, {} answers, complete={}; {}: {}",
                artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
                artifacts.memory_records.len(),
                artifacts.answer_records.len(),
                artifacts.complete,
                if cli.validate_only {
                    "validation"
                } else {
                    "summary"
                },
                cli.output_dir
                    .join(if cli.validate_only {
                        "validation.json"
                    } else {
                        "summary.json"
                    })
                    .display()
            );
        }
    }
    Ok(())
}
