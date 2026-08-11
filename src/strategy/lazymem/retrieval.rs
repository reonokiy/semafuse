//! Deterministic hybrid retrieval and cross-encoder reranking.

#[cfg(test)]
use crate::strategy::hybrid::bm25_tokens;
pub(in crate::strategy) use crate::strategy::hybrid::{
    RrfCandidate, RrfInput, RrfOutput, okapi_bm25_ranked_ids, paper_retrieval_text,
    reciprocal_rank_fusion,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Cross-encoder score for one fused candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::strategy) struct RerankScore {
    pub message_id: String,
    pub score: f64,
}

/// Fused candidates plus exactly one cross-encoder score for each candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::strategy) struct RerankInput {
    pub fused_candidates: Vec<RrfCandidate>,
    pub scores: Vec<RerankScore>,
    pub top_n: usize,
}

/// One candidate after cross-encoder reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::strategy) struct RerankedCandidate {
    pub message_id: String,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
    pub rrf_score: f64,
    pub rerank_score: f64,
}

/// Audit data for the reranking stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::strategy) struct RerankTrace {
    pub input_count: usize,
    pub requested_top_n: usize,
    pub output_order: Vec<String>,
}

/// Typed output of the cross-encoder reranking stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::strategy) struct RerankOutput {
    pub candidates: Vec<RerankedCandidate>,
    pub trace: RerankTrace,
}

/// Apply cross-encoder scores and retain the top `N` messages.
///
/// Scores are intentionally not blended with RRF: the paper uses RRF to form
/// the broad pool, then lets the cross-encoder define the final ranking. RRF
/// order is only a deterministic tie-break for equal reranker scores.
pub(in crate::strategy) fn apply_reranker(input: &RerankInput) -> Result<RerankOutput> {
    let mut candidate_ids = HashSet::new();
    for candidate in &input.fused_candidates {
        ensure!(
            !candidate.message_id.trim().is_empty(),
            "fused candidate contains an empty message id"
        );
        ensure!(
            candidate_ids.insert(candidate.message_id.as_str()),
            "duplicate fused candidate id: {}",
            candidate.message_id
        );
        ensure!(
            candidate.rrf_score.is_finite() && candidate.rrf_score > 0.0,
            "invalid RRF score for {}",
            candidate.message_id
        );
    }

    let mut scores = HashMap::new();
    for score in &input.scores {
        ensure!(
            candidate_ids.contains(score.message_id.as_str()),
            "reranker returned unknown candidate id: {}",
            score.message_id
        );
        ensure!(
            score.score.is_finite(),
            "reranker returned non-finite score for {}",
            score.message_id
        );
        ensure!(
            scores
                .insert(score.message_id.as_str(), score.score)
                .is_none(),
            "reranker returned duplicate score for {}",
            score.message_id
        );
    }
    ensure!(
        scores.len() == input.fused_candidates.len(),
        "reranker must return exactly one score for every fused candidate (expected {}, got {})",
        input.fused_candidates.len(),
        scores.len()
    );

    let fused_order = input
        .fused_candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.message_id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut reranked = input
        .fused_candidates
        .iter()
        .map(|candidate| RerankedCandidate {
            message_id: candidate.message_id.clone(),
            dense_rank: candidate.dense_rank,
            sparse_rank: candidate.sparse_rank,
            rrf_score: candidate.rrf_score,
            rerank_score: scores[candidate.message_id.as_str()],
        })
        .collect::<Vec<_>>();
    reranked.sort_by(|left, right| {
        right
            .rerank_score
            .total_cmp(&left.rerank_score)
            .then_with(|| {
                fused_order[left.message_id.as_str()].cmp(&fused_order[right.message_id.as_str()])
            })
    });
    reranked.truncate(input.top_n);
    let output_order = reranked
        .iter()
        .map(|candidate| candidate.message_id.clone())
        .collect();

    Ok(RerankOutput {
        candidates: reranked,
        trace: RerankTrace {
            input_count: input.fused_candidates.len(),
            requested_top_n: input.top_n,
            output_order,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AgentKind, CanonicalEvent, EventSource, MessageRole};

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn event(native_id: &str, content: &str) -> CanonicalEvent {
        CanonicalEvent::message_from_text(
            EventSource::new(
                AgentKind::Codex,
                "session",
                None,
                Some(native_id.to_owned()),
                "message",
                None,
                Some("/repo".into()),
            ),
            MessageRole::User,
            content,
            Some(0),
        )
    }

    #[test]
    fn bm25_uses_the_public_ascii_tokenizer_and_stable_corpus_ties() {
        assert_eq!(
            bm25_tokens("ÄLPHA foo-Bar_baz 中文"),
            ["lpha", "foo-bar_baz"]
        );
        let corpus = vec![
            event("m0", "apple apple"),
            event("m1", "apple"),
            event("m2", "banana"),
            event("m3", "carrot"),
            event("m4", "date"),
        ];
        let ranked = okapi_bm25_ranked_ids("APPLE", &corpus, 5);
        assert_eq!(ranked[0], corpus[0].id);
        assert_eq!(ranked[1], corpus[1].id);
        assert_eq!(
            ranked[2..],
            [
                corpus[2].id.clone(),
                corpus[3].id.clone(),
                corpus[4].id.clone()
            ]
        );
    }

    #[test]
    fn rrf_uses_one_based_ranks_and_fixed_paper_constant() {
        let output = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["both", "dense"]),
            sparse_ranked_ids: ids(&["both", "sparse"]),
            tie_break_ranked_ids: ids(&["both", "dense", "sparse"]),
        })
        .unwrap();

        assert_eq!(output.trace.k, 60);
        assert_eq!(output.trace.fused_order, ids(&["both", "dense", "sparse"]));
        assert_eq!(output.candidates[0].dense_rank, Some(1));
        assert_eq!(output.candidates[0].sparse_rank, Some(1));
        assert_eq!(output.candidates[0].rrf_score, 2.0 / 61.0);
        assert_eq!(output.candidates[1].rrf_score, 1.0 / 62.0);
    }

    #[test]
    fn rrf_ignores_later_in_channel_duplicates_without_changing_first_rank() {
        let output = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["a", "b", "a"]),
            sparse_ranked_ids: ids(&["b"]),
            tie_break_ranked_ids: ids(&["a", "b"]),
        })
        .unwrap();

        assert_eq!(output.trace.dense_duplicates_ignored, ids(&["a"]));
        let a = output
            .candidates
            .iter()
            .find(|candidate| candidate.message_id == "a")
            .unwrap();
        assert_eq!(a.dense_rank, Some(1));
        assert_eq!(a.rrf_score, 1.0 / 61.0);
    }

    #[test]
    fn rrf_has_a_total_order_for_equal_scores() {
        let output = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["z"]),
            sparse_ranked_ids: ids(&["a"]),
            tie_break_ranked_ids: ids(&["z", "a"]),
        })
        .unwrap();

        assert_eq!(output.trace.fused_order, ids(&["z", "a"]));
    }

    #[test]
    fn rrf_rejects_empty_identifiers() {
        let error = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["valid", "  "]),
            sparse_ranked_ids: Vec::new(),
            tie_break_ranked_ids: ids(&["valid"]),
        })
        .unwrap_err();
        assert!(error.to_string().contains("empty message id at rank 2"));
    }

    #[test]
    fn reranker_score_defines_order_and_rrf_breaks_ties() {
        let fused = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["a", "b", "c"]),
            sparse_ranked_ids: ids(&["b", "a", "c"]),
            tie_break_ranked_ids: ids(&["a", "b", "c"]),
        })
        .unwrap();
        let output = apply_reranker(&RerankInput {
            fused_candidates: fused.candidates,
            scores: vec![
                RerankScore {
                    message_id: "c".into(),
                    score: 9.0,
                },
                RerankScore {
                    message_id: "a".into(),
                    score: 8.0,
                },
                RerankScore {
                    message_id: "b".into(),
                    score: 8.0,
                },
            ],
            top_n: 2,
        })
        .unwrap();

        assert_eq!(output.trace.output_order, ids(&["c", "a"]));
        assert_eq!(output.candidates.len(), 2);
    }

    #[test]
    fn reranker_score_input_order_does_not_affect_output() {
        let fused = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: ids(&["a", "b"]),
            sparse_ranked_ids: ids(&["a", "b"]),
            tie_break_ranked_ids: ids(&["a", "b"]),
        })
        .unwrap();
        let left = apply_reranker(&RerankInput {
            fused_candidates: fused.candidates.clone(),
            scores: vec![
                RerankScore {
                    message_id: "a".into(),
                    score: 1.0,
                },
                RerankScore {
                    message_id: "b".into(),
                    score: 2.0,
                },
            ],
            top_n: 50,
        })
        .unwrap();
        let right = apply_reranker(&RerankInput {
            fused_candidates: fused.candidates,
            scores: vec![
                RerankScore {
                    message_id: "b".into(),
                    score: 2.0,
                },
                RerankScore {
                    message_id: "a".into(),
                    score: 1.0,
                },
            ],
            top_n: 50,
        })
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn reranker_requires_exactly_one_finite_score_per_candidate() {
        let candidate = RrfCandidate {
            message_id: "a".into(),
            dense_rank: Some(1),
            sparse_rank: None,
            rrf_score: 1.0 / 61.0,
        };

        let missing = apply_reranker(&RerankInput {
            fused_candidates: vec![candidate.clone()],
            scores: Vec::new(),
            top_n: 1,
        })
        .unwrap_err();
        assert!(missing.to_string().contains("exactly one score"));

        let non_finite = apply_reranker(&RerankInput {
            fused_candidates: vec![candidate],
            scores: vec![RerankScore {
                message_id: "a".into(),
                score: f64::NAN,
            }],
            top_n: 1,
        })
        .unwrap_err();
        assert!(non_finite.to_string().contains("non-finite"));
    }

    #[test]
    fn reranker_zero_top_n_still_validates_then_returns_empty() {
        let output = apply_reranker(&RerankInput {
            fused_candidates: Vec::new(),
            scores: Vec::new(),
            top_n: 0,
        })
        .unwrap();
        assert!(output.candidates.is_empty());
        assert!(output.trace.output_order.is_empty());
    }
}
