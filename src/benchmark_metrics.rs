//! Deterministic retrieval metrics for one ranked list.
//!
//! This module deliberately exposes both conventional binary-relevance nDCG
//! and LongMemEval's historical `ndcg_any` calculation.  The latter uses the
//! discount implemented in LongMemEval's `src/retrieval/eval_utils.py`: ranks
//! one and two are both undiscounted, while rank `r >= 3` is divided by
//! `log2(r)`.  It is retained for result compatibility and must not be confused
//! with the conventional nDCG field.

use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

/// Fixed retrieval cutoffs emitted for every ranked list.
pub const RANKING_CUTOFFS: [usize; 7] = [1, 3, 5, 10, 20, 30, 50];

/// All retrieval metrics for one ranked list and one non-empty gold set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingMetrics {
    /// Number of unique ranked identifiers supplied by the caller.
    pub ranked_count: usize,
    /// Number of unique gold identifiers supplied by the caller.
    pub gold_count: usize,
    /// Metrics in ascending [`RANKING_CUTOFFS`] order.
    pub cutoffs: Vec<RankingMetricsAtCutoff>,
}

impl RankingMetrics {
    /// Returns the metrics for an emitted cutoff, if present.
    pub fn at(&self, cutoff: usize) -> Option<&RankingMetricsAtCutoff> {
        self.cutoffs.iter().find(|metrics| metrics.cutoff == cutoff)
    }
}

/// Retrieval metrics for one cutoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingMetricsAtCutoff {
    /// Maximum number of ranked identifiers inspected.
    pub cutoff: usize,
    /// One when at least one gold identifier occurs within the cutoff.
    pub recall_any: f64,
    /// One when every gold identifier occurs within the cutoff.
    pub recall_all: f64,
    /// Fraction of all gold identifiers occurring within the cutoff.
    pub mean_recall: f64,
    /// Reciprocal rank of the first gold identifier within the cutoff, or zero.
    pub mrr: f64,
    /// Conventional binary-relevance nDCG using `log2(rank + 1)` discounting.
    pub ndcg: f64,
    /// LongMemEval's historical metric named `ndcg_any` by the official code.
    ///
    /// Despite the historical name, this is not a binary any-hit metric.  It
    /// computes nDCG over every binary-relevant gold item using LongMemEval's
    /// non-standard discount, which leaves ranks one and two undiscounted.
    pub official_ndcg_any: f64,
}

/// Invalid ranked-list input rejected before metric calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankingMetricsError {
    /// At least one gold identifier is required.
    EmptyGold,
    /// A ranked identifier was empty or whitespace-only.
    EmptyRankedId {
        /// One-based position of the invalid ranked identifier.
        rank: usize,
    },
    /// A gold identifier was empty or whitespace-only.
    EmptyGoldId {
        /// Zero-based position of the invalid gold identifier.
        index: usize,
    },
    /// Ranked identifiers must be unique so repeated hits cannot inflate scores.
    DuplicateRankedId {
        /// Repeated identifier.
        id: String,
        /// One-based position of its first occurrence.
        first_rank: usize,
        /// One-based position of its repeated occurrence.
        duplicate_rank: usize,
    },
    /// Gold identifiers must be a set rather than a weighted multiset.
    DuplicateGoldId {
        /// Repeated gold identifier.
        id: String,
    },
}

impl fmt::Display for RankingMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGold => formatter.write_str("gold identifiers must not be empty"),
            Self::EmptyRankedId { rank } => {
                write!(
                    formatter,
                    "ranked identifier at one-based rank {rank} is empty"
                )
            }
            Self::EmptyGoldId { index } => {
                write!(
                    formatter,
                    "gold identifier at zero-based index {index} is empty"
                )
            }
            Self::DuplicateRankedId {
                id,
                first_rank,
                duplicate_rank,
            } => write!(
                formatter,
                "ranked identifier {id:?} is duplicated at one-based ranks {first_rank} and {duplicate_rank}"
            ),
            Self::DuplicateGoldId { id } => {
                write!(formatter, "gold identifier {id:?} is duplicated")
            }
        }
    }
}

impl Error for RankingMetricsError {}

/// Evaluates one ordered list against one non-empty set of gold identifiers.
///
/// Ranked identifiers are required to be unique.  Duplicate rankings are
/// rejected rather than silently deduplicated because deduplication changes
/// ranks and can conceal an invalid retriever response.  Gold identifiers must
/// likewise be unique and non-empty.
///
/// LongMemEval's source implementation receives corpus indices.  Passing the
/// corresponding ranked identifiers here is equivalent when, as required by
/// the benchmark, every gold identifier belongs to the retrieval corpus.
pub fn evaluate_ranking<R, G>(
    ranked_ids: &[R],
    gold_ids: &[G],
) -> Result<RankingMetrics, RankingMetricsError>
where
    R: AsRef<str>,
    G: AsRef<str>,
{
    if gold_ids.is_empty() {
        return Err(RankingMetricsError::EmptyGold);
    }

    let mut gold = HashSet::with_capacity(gold_ids.len());
    for (index, id) in gold_ids.iter().enumerate() {
        let id = id.as_ref();
        if id.trim().is_empty() {
            return Err(RankingMetricsError::EmptyGoldId { index });
        }
        if !gold.insert(id) {
            return Err(RankingMetricsError::DuplicateGoldId { id: id.to_owned() });
        }
    }

    let mut ranked_positions = HashMap::with_capacity(ranked_ids.len());
    for (index, id) in ranked_ids.iter().enumerate() {
        let id = id.as_ref();
        let rank = index + 1;
        if id.trim().is_empty() {
            return Err(RankingMetricsError::EmptyRankedId { rank });
        }
        if let Some(first_rank) = ranked_positions.insert(id, rank) {
            return Err(RankingMetricsError::DuplicateRankedId {
                id: id.to_owned(),
                first_rank,
                duplicate_rank: rank,
            });
        }
    }

    let relevances = ranked_ids
        .iter()
        .map(|id| gold.contains(id.as_ref()))
        .collect::<Vec<_>>();
    let cutoffs = RANKING_CUTOFFS
        .into_iter()
        .map(|cutoff| metrics_at_cutoff(&relevances, gold.len(), cutoff))
        .collect();

    Ok(RankingMetrics {
        ranked_count: ranked_ids.len(),
        gold_count: gold.len(),
        cutoffs,
    })
}

fn metrics_at_cutoff(
    relevances: &[bool],
    gold_count: usize,
    cutoff: usize,
) -> RankingMetricsAtCutoff {
    let prefix = &relevances[..relevances.len().min(cutoff)];
    let found = prefix.iter().filter(|relevant| **relevant).count();
    let first_relevant_rank = prefix
        .iter()
        .position(|relevant| *relevant)
        .map(|index| index + 1);

    RankingMetricsAtCutoff {
        cutoff,
        recall_any: f64::from(found > 0),
        recall_all: f64::from(found == gold_count),
        mean_recall: found as f64 / gold_count as f64,
        mrr: first_relevant_rank.map_or(0.0, |rank| 1.0 / rank as f64),
        ndcg: standard_ndcg(prefix, gold_count, cutoff),
        official_ndcg_any: longmemeval_official_ndcg_any(prefix, gold_count, cutoff),
    }
}

fn standard_ndcg(relevances: &[bool], gold_count: usize, cutoff: usize) -> f64 {
    let actual = relevances
        .iter()
        .enumerate()
        .filter(|(_, relevant)| **relevant)
        .map(|(index, _)| 1.0 / ((index + 2) as f64).log2())
        .sum::<f64>();
    let ideal = (0..gold_count.min(cutoff))
        .map(|index| 1.0 / ((index + 2) as f64).log2())
        .sum::<f64>();
    actual / ideal
}

fn longmemeval_official_ndcg_any(relevances: &[bool], gold_count: usize, cutoff: usize) -> f64 {
    let actual = longmemeval_historical_dcg(relevances);
    let ideal_relevances = vec![true; gold_count.min(cutoff)];
    let ideal = longmemeval_historical_dcg(&ideal_relevances);
    actual / ideal
}

fn longmemeval_historical_dcg(relevances: &[bool]) -> f64 {
    let Some((first, remaining)) = relevances.split_first() else {
        return 0.0;
    };
    f64::from(*first)
        + remaining
            .iter()
            .enumerate()
            .filter(|(_, relevant)| **relevant)
            // `remaining[0]` is rank two.  The official NumPy implementation
            // divides it by log2(2), not the conventional log2(3).
            .map(|(index, _)| 1.0 / ((index + 2) as f64).log2())
            .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected:.15}, got {actual:.15}"
        );
    }

    #[test]
    fn emits_fixed_cutoffs_and_perfect_scores_after_all_gold_is_retrieved() {
        let metrics = evaluate_ranking(&["g1", "g2", "noise"], &["g1", "g2"]).unwrap();
        assert_eq!(metrics.ranked_count, 3);
        assert_eq!(metrics.gold_count, 2);
        assert_eq!(
            metrics
                .cutoffs
                .iter()
                .map(|metrics| metrics.cutoff)
                .collect::<Vec<_>>(),
            RANKING_CUTOFFS
        );

        let at_one = metrics.at(1).unwrap();
        assert_eq!(at_one.recall_any, 1.0);
        assert_eq!(at_one.recall_all, 0.0);
        assert_eq!(at_one.mean_recall, 0.5);
        assert_eq!(at_one.mrr, 1.0);
        assert_eq!(at_one.ndcg, 1.0);
        assert_eq!(at_one.official_ndcg_any, 1.0);

        let at_three = metrics.at(3).unwrap();
        assert_eq!(at_three.recall_any, 1.0);
        assert_eq!(at_three.recall_all, 1.0);
        assert_eq!(at_three.mean_recall, 1.0);
        assert_eq!(at_three.mrr, 1.0);
        assert_eq!(at_three.ndcg, 1.0);
        assert_eq!(at_three.official_ndcg_any, 1.0);
    }

    #[test]
    fn golden_standard_and_longmemeval_ndcg_are_distinct() {
        let metrics = evaluate_ranking(&["noise-1", "g1", "noise-2", "g2"], &["g1", "g2"]).unwrap();
        let at_five = metrics.at(5).unwrap();

        assert_eq!(at_five.recall_any, 1.0);
        assert_eq!(at_five.recall_all, 1.0);
        assert_eq!(at_five.mean_recall, 1.0);
        assert_eq!(at_five.mrr, 0.5);
        close(at_five.ndcg, 0.650_920_929_807_132_6);
        // Historical LongMemEval: rank 2 contributes 1/log2(2), rank 4
        // contributes 1/log2(4), and the two-item ideal DCG is 2.
        close(at_five.official_ndcg_any, 0.75);
    }

    #[test]
    fn cutoff_excludes_late_hits_from_every_metric() {
        let metrics = evaluate_ranking(&["n1", "n2", "n3", "g1"], &["g1", "g2"]).unwrap();
        let at_three = metrics.at(3).unwrap();
        assert_eq!(at_three.recall_any, 0.0);
        assert_eq!(at_three.recall_all, 0.0);
        assert_eq!(at_three.mean_recall, 0.0);
        assert_eq!(at_three.mrr, 0.0);
        assert_eq!(at_three.ndcg, 0.0);
        assert_eq!(at_three.official_ndcg_any, 0.0);

        let at_five = metrics.at(5).unwrap();
        assert_eq!(at_five.recall_any, 1.0);
        assert_eq!(at_five.recall_all, 0.0);
        assert_eq!(at_five.mean_recall, 0.5);
        assert_eq!(at_five.mrr, 0.25);
    }

    #[test]
    fn rejects_empty_gold_and_duplicate_or_empty_identifiers() {
        assert_eq!(
            evaluate_ranking::<&str, &str>(&["one"], &[]),
            Err(RankingMetricsError::EmptyGold)
        );
        assert_eq!(
            evaluate_ranking(&["one", "one"], &["gold"]),
            Err(RankingMetricsError::DuplicateRankedId {
                id: "one".into(),
                first_rank: 1,
                duplicate_rank: 2,
            })
        );
        assert_eq!(
            evaluate_ranking(&["one"], &["gold", "gold"]),
            Err(RankingMetricsError::DuplicateGoldId { id: "gold".into() })
        );
        assert_eq!(
            evaluate_ranking(&[" "], &["gold"]),
            Err(RankingMetricsError::EmptyRankedId { rank: 1 })
        );
        assert_eq!(
            evaluate_ranking(&["one"], &[" "]),
            Err(RankingMetricsError::EmptyGoldId { index: 0 })
        );
    }

    #[test]
    fn serde_output_names_the_official_metric_unambiguously() {
        let metrics = evaluate_ranking(&["gold"], &["gold"]).unwrap();
        let json = serde_json::to_value(metrics).unwrap();
        assert_eq!(json["cutoffs"][0]["cutoff"], 1);
        assert_eq!(json["cutoffs"][0]["ndcg"], 1.0);
        assert_eq!(json["cutoffs"][0]["official_ndcg_any"], 1.0);
        assert!(json["cutoffs"][0].get("ndcg_any").is_none());
    }
}
