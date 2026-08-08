//! Rebuild a curated comparison from an immutable Semafuse benchmark journal.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use semafuse::{
    benchmark_journal::{JsonlJournal, SummaryEnvelope, read_summary, write_summary_atomic},
    benchmark_suite::{
        BenchmarkMode, QuestionMethodRecord, RetrievalProfile, build_summary_from_records,
    },
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

/// Build a role-aware report without rerunning retrieval or calling model APIs.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Directory containing `summary.json` and `per-question.jsonl`.
    #[arg(long)]
    run_dir: PathBuf,
    /// Destination; defaults to `<run-dir>/comparison.json`.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let source_summary_path = cli.run_dir.join("summary.json");
    let journal_path = cli.run_dir.join("per-question.jsonl");
    let output_path = cli
        .output
        .unwrap_or_else(|| cli.run_dir.join("comparison.json"));
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
            "src/bin/semafuse-memory-report.rs",
            include_bytes!("semafuse-memory-report.rs").as_slice(),
        ),
        (
            "src/benchmark_suite.rs",
            include_bytes!("../benchmark_suite.rs").as_slice(),
        ),
        (
            "src/benchmark_report.rs",
            include_bytes!("../benchmark_report.rs").as_slice(),
        ),
    ] {
        digest.update((path.len() as u64).to_be_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}
