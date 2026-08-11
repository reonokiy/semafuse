use super::command_exists_in;
use std::ffi::OsStr;
use tempfile::TempDir;

#[cfg(feature = "benchmarks")]
use clap::Parser;

#[test]
fn command_lookup_uses_pathext_with_windows_semantics() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex.cmd"), b"").unwrap();
    std::fs::write(temp.path().join("amp.exe"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE;.CMD")),
        true,
    ));
    assert!(command_exists_in(
        "amp",
        Some(temp.path().as_os_str()),
        Some(OsStr::new("EXE;CMD")),
        true,
    ));
    assert!(!command_exists_in(
        "claude",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE;.CMD")),
        true,
    ));
}

#[test]
fn command_lookup_uses_default_windows_extensions_and_preserves_explicit_ones() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex.EXE"), b"").unwrap();
    std::fs::write(temp.path().join("claude.cmd"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        None,
        true,
    ));
    assert!(command_exists_in(
        "claude.CMD",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        true,
    ));
}

#[test]
fn command_lookup_ignores_pathext_on_posix() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        false,
    ));
    assert!(!command_exists_in(
        "codex.EXE",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        false,
    ));
}

#[cfg(feature = "benchmarks")]
#[test]
fn bench_jsonl_is_a_feature_gated_semafuse_subcommand() {
    let cli = super::Cli::try_parse_from([
        "semafuse",
        "bench",
        "jsonl",
        "--config",
        "/tmp/semafuse.toml",
    ])
    .unwrap();
    assert!(matches!(
        cli.command,
        super::Commands::Bench {
            command: super::benchmark_cli::BenchmarkCommand::Jsonl(_)
        }
    ));
}

#[cfg(feature = "benchmarks")]
#[test]
fn bench_locomo_exposes_the_full_answer_validation_workflow() {
    let cli = super::Cli::try_parse_from([
        "semafuse",
        "bench",
        "locomo",
        "--suite",
        "/tmp/suite.toml",
        "--dataset",
        "/tmp/locomo10.json",
        "--acknowledge-noncommercial",
        "--mode",
        "full",
        "--validate-only",
        "--method",
        "reconstructive_v4",
        "--output-dir",
        "/tmp/validation",
    ])
    .unwrap();
    assert!(matches!(
        cli.command,
        super::Commands::Bench {
            command: super::benchmark_cli::BenchmarkCommand::Locomo(_)
        }
    ));
}
