//! Release-mode native LongMemEval comparison runner.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use semafuse::benchmark_suite::{BenchmarkMode, SuiteRunOptions, run_longmemeval_suite};
use std::path::PathBuf;

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

/// Run a modular, paired LongMemEval comparison through native Rust.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
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
    /// Retrieval-only or complete answer + official judge evaluation.
    #[arg(long, value_enum, default_value = "retrieval")]
    mode: CliMode,
    /// Crash-safe output directory (`per-question.jsonl`, `summary.json`).
    #[arg(long)]
    output_dir: PathBuf,
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
    let artifacts = run_longmemeval_suite(&SuiteRunOptions {
        suite_path: cli.suite,
        dataset_path: cli.dataset,
        split_manifest: cli.split_manifest,
        split_name: cli.split,
        mode: cli.mode.into(),
        output_dir: Some(cli.output_dir.clone()),
    })
    .await?;
    println!(
        "completed run {}: {} records; summary: {}",
        artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
        artifacts.records.len(),
        cli.output_dir.join("summary.json").display()
    );
    Ok(())
}
