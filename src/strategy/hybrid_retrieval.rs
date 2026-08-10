//! Strategy-neutral building blocks for deterministic hybrid retrieval.

use anyhow::{Result, ensure};
use std::collections::{HashMap, HashSet};

/// One item in a dense-first, BM25-fallback union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseBm25UnionCandidate {
    pub(super) message_id: String,
    /// One-based rank in the original dense channel, when present.
    pub(super) dense_rank: Option<usize>,
    /// One-based rank in the original BM25 channel, when present.
    pub(super) bm25_rank: Option<usize>,
}

/// Preserve dense order, then append only BM25-exclusive items in BM25 order.
///
/// The first occurrence in each channel defines its rank. Duplicate identifiers
/// are suppressed stably, and `limit` applies after the two ordered channels
/// are unioned. Inputs are still validated in full when the limit is reached.
pub(super) fn dense_bm25_union(
    dense_ranked_ids: &[String],
    bm25_ranked_ids: &[String],
    limit: usize,
) -> Result<Vec<DenseBm25UnionCandidate>> {
    let dense_ranks = first_ranks("dense", dense_ranked_ids)?;
    let bm25_ranks = first_ranks("BM25", bm25_ranked_ids)?;
    let mut seen = HashSet::new();
    let mut output =
        Vec::with_capacity(limit.min(dense_ranks.len().saturating_add(bm25_ranks.len())));

    for message_id in dense_ranked_ids.iter().chain(bm25_ranked_ids) {
        if !seen.insert(message_id.as_str()) || output.len() == limit {
            continue;
        }
        output.push(DenseBm25UnionCandidate {
            message_id: message_id.clone(),
            dense_rank: dense_ranks.get(message_id.as_str()).copied(),
            bm25_rank: bm25_ranks.get(message_id.as_str()).copied(),
        });
    }
    Ok(output)
}

fn first_ranks<'a>(label: &str, ranked_ids: &'a [String]) -> Result<HashMap<&'a str, usize>> {
    let mut ranks = HashMap::with_capacity(ranked_ids.len());
    for (index, message_id) in ranked_ids.iter().enumerate() {
        ensure!(
            !message_id.trim().is_empty(),
            "{label} retrieval channel contains an empty message id at rank {}",
            index + 1
        );
        ranks.entry(message_id.as_str()).or_insert(index + 1);
    }
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn union_preserves_dense_then_appends_bm25_exclusive_order() {
        let output = dense_bm25_union(
            &ids(&["dense-1", "shared", "dense-2"]),
            &ids(&["shared", "bm25-1", "dense-1", "bm25-2"]),
            10,
        )
        .unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.message_id.as_str())
                .collect::<Vec<_>>(),
            ["dense-1", "shared", "dense-2", "bm25-1", "bm25-2"]
        );
        assert_eq!(
            (output[0].dense_rank, output[0].bm25_rank),
            (Some(1), Some(3))
        );
        assert_eq!(
            (output[1].dense_rank, output[1].bm25_rank),
            (Some(2), Some(1))
        );
        assert_eq!((output[3].dense_rank, output[3].bm25_rank), (None, Some(2)));
    }

    #[test]
    fn union_deduplicates_each_channel_at_its_first_rank() {
        let output =
            dense_bm25_union(&ids(&["a", "a", "b"]), &ids(&["b", "c", "c", "a"]), 10).unwrap();

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.message_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(
            (output[0].dense_rank, output[0].bm25_rank),
            (Some(1), Some(4))
        );
        assert_eq!((output[2].dense_rank, output[2].bm25_rank), (None, Some(2)));
    }

    #[test]
    fn union_applies_limit_after_stable_deduplication() {
        let output = dense_bm25_union(&ids(&["a", "a", "b"]), &ids(&["b", "c", "d"]), 3).unwrap();
        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.message_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
    }
}
