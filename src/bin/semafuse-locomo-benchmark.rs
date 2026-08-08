//! Release-mode native LoCoMo paper-v2 retrieval runner.

use anyhow::Result;
use clap::Parser;
use semafuse::locomo_suite::{LocomoSuiteRunOptions, run_locomo_retrieval_suite};
use std::path::PathBuf;

/// Run the pinned 314-QA / 312-evidence LoCoMo retrieval comparison.
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
    let artifacts = run_locomo_retrieval_suite(&LocomoSuiteRunOptions {
        suite_path: cli.suite,
        dataset_path: cli.dataset,
        acknowledge_noncommercial: cli.acknowledge_noncommercial,
        output_dir: Some(cli.output_dir.clone()),
    })
    .await?;
    println!(
        "completed LoCoMo run {}: {} records; summary: {}",
        artifacts.manifest["run_key"].as_str().unwrap_or("unknown"),
        artifacts.records.len(),
        cli.output_dir.join("summary.json").display()
    );
    Ok(())
}
