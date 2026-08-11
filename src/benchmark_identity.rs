//! Complete compile-time source identity for reproducible benchmark artifacts.

use sha2::{Digest, Sha256};

macro_rules! source {
    ($path:literal) => {
        ($path, include_bytes!(concat!("../", $path)).as_slice())
    };
}

// Keep the complete set in one place. The test below scans the repository and
// fails when a Rust source or migration is added without being embedded here.
// Cargo.lock is intentionally reported as a separate identity by each suite.
const IDENTITY_SOURCES: &[(&str, &[u8])] = &[
    source!("Cargo.toml"),
    source!("migrations/20260718000000_initial.sql"),
    source!("migrations/20260801000000_data_governance.sql"),
    source!("migrations/20260801010000_blob_gc_claims.sql"),
    source!("migrations/20260801020000_lazymem_vectors.sql"),
    source!("src/adapters/amp.rs"),
    source!("src/adapters/claude.rs"),
    source!("src/adapters/codex.rs"),
    source!("src/adapters/executable.rs"),
    source!("src/adapters/history.rs"),
    source!("src/adapters/mod.rs"),
    source!("src/adapters/opencode.rs"),
    source!("src/benchmark.rs"),
    source!("src/benchmark_baselines.rs"),
    source!("src/benchmark_cli.rs"),
    source!("src/benchmark_identity.rs"),
    source!("src/benchmark_journal.rs"),
    source!("src/benchmark_metrics.rs"),
    source!("src/benchmark_report.rs"),
    source!("src/benchmark_suite.rs"),
    source!("src/bin/semafused.rs"),
    source!("src/blob.rs"),
    source!("src/config.rs"),
    source!("src/daemon/config.rs"),
    source!("src/daemon/error.rs"),
    source!("src/daemon/ipc.rs"),
    source!("src/daemon/mod.rs"),
    source!("src/daemon/models/candle.rs"),
    source!("src/daemon/models/mod.rs"),
    source!("src/daemon/router.rs"),
    source!("src/daemon/server.rs"),
    source!("src/daemon/services/mod.rs"),
    source!("src/daemon/services/models.rs"),
    source!("src/daemon_ipc/client.rs"),
    source!("src/daemon_ipc/mod.rs"),
    source!("src/daemon_ipc/protocol.rs"),
    source!("src/daemon_ipc/unsupported_client.rs"),
    source!("src/governance.rs"),
    source!("src/hooks.rs"),
    source!("src/importer.rs"),
    source!("src/inference.rs"),
    source!("src/inference/codex.rs"),
    source!("src/inference/http.rs"),
    source!("src/inference/openai.rs"),
    source!("src/inference/processor.rs"),
    source!("src/inference/reranker.rs"),
    source!("src/inference/structured.rs"),
    source!("src/lib.rs"),
    source!("src/locomo.rs"),
    source!("src/locomo_answer_protocol.rs"),
    source!("src/locomo_answer_suite.rs"),
    source!("src/locomo_eval.rs"),
    source!("src/locomo_suite.rs"),
    source!("src/locomo_suite_common.rs"),
    source!("src/longmemeval.rs"),
    source!("src/longmemeval_eval.rs"),
    source!("src/main.rs"),
    source!("src/main_tests.rs"),
    source!("src/mcp.rs"),
    source!("src/model.rs"),
    source!("src/model/archive.rs"),
    source!("src/model/event.rs"),
    source!("src/model/query.rs"),
    source!("src/model/source.rs"),
    source!("src/model/time.rs"),
    source!("src/retrieval.rs"),
    source!("src/storage/memory.rs"),
    source!("src/storage/mod.rs"),
    source!("src/storage/sqlite.rs"),
    source!("src/strategy/history.rs"),
    source!("src/strategy/hybrid.rs"),
    source!("src/strategy/lazymem.rs"),
    source!("src/strategy/lazymem/config.rs"),
    source!("src/strategy/lazymem/processor.rs"),
    source!("src/strategy/lazymem/prompt.rs"),
    source!("src/strategy/lazymem/retrieval.rs"),
    source!("src/strategy/lazymem/types.rs"),
    source!("src/strategy/lazymem/window.rs"),
    source!("src/strategy/mod.rs"),
    source!("src/strategy/recall.rs"),
    source!("src/strategy/reconstructive.rs"),
    source!("src/strategy/reconstructive/config.rs"),
    source!("src/strategy/reconstructive/controller.rs"),
    source!("src/strategy/reconstructive/render.rs"),
    source!("src/strategy/reconstructive/retrieval.rs"),
    source!("src/strategy/registry.rs"),
];

/// Returns a stable digest of every Rust source, migration, and Cargo manifest
/// byte embedded in this build.
pub fn source_sha256() -> String {
    let mut digest = Sha256::new();
    digest.update(b"semafuse-benchmark-source-identity/v1");
    for (path, bytes) in IDENTITY_SOURCES {
        digest.update((path.len() as u64).to_be_bytes());
        digest.update(path.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeSet, path::Path};
    use walkdir::WalkDir;

    #[test]
    fn embedded_identity_covers_every_behavior_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut discovered = BTreeSet::from(["Cargo.toml".to_owned()]);
        if root.join("build.rs").is_file() {
            discovered.insert("build.rs".to_owned());
        }
        for (directory, extension) in [("src", "rs"), ("migrations", "sql")] {
            for entry in WalkDir::new(root.join(directory)) {
                let entry = entry.unwrap();
                if !entry.file_type().is_file()
                    || entry.path().extension().and_then(|value| value.to_str()) != Some(extension)
                {
                    continue;
                }
                let relative = entry.path().strip_prefix(root).unwrap();
                discovered.insert(relative.to_string_lossy().replace('\\', "/"));
            }
        }
        let embedded = IDENTITY_SOURCES
            .iter()
            .map(|(path, _)| (*path).to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(embedded, discovered);
    }

    #[test]
    fn source_identity_is_stable_and_complete() {
        let first = source_sha256();
        assert_eq!(first, source_sha256());
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
}
