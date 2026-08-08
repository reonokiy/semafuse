//! JSONL bridge from the LazyMem benchmark harness to Semafuse's real strategy.

use anyhow::{Context, Result};
use clap::Parser;
use semafuse::{
    benchmark::{BenchmarkQuestion, run_lazymem_question, run_strategy_retrieval_question},
    config,
};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

/// Evaluate normalized benchmark questions through the configured LazyMem strategy.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Semafuse TOML configuration with `strategies.recall = "lazymem"`.
    #[arg(long)]
    config: PathBuf,
    /// Run embedding, hybrid retrieval, reranking, and window expansion only.
    #[arg(long)]
    retrieval_only: bool,
    /// Maximum number of final retrieval anchors.
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::load_from_path(&cli.config)
        .with_context(|| format!("load benchmark configuration {}", cli.config.display()))?;
    if config.strategies.recall != "lazymem" {
        anyhow::bail!("semafuse-lazymem-benchmark requires strategies.recall = \"lazymem\"");
    }

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
        let response = if cli.retrieval_only {
            run_strategy_retrieval_question(&config, question, cli.limit).await
        } else {
            run_lazymem_question(&config, question, cli.limit).await
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
