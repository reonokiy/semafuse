//! Deterministic hybrid retrieval and cross-encoder reranking.

use super::types::PAPER_RRF_K;
use crate::model::{CanonicalEvent, EventData, MessageRole};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const BM25_EPSILON: f64 = 0.25;

/// Rank the exact scoped message corpus with `rank_bm25.BM25Okapi` semantics.
///
/// LazyMem's public retrieval artifact tokenizes with
/// `[A-Za-z0-9_\\-]+`, lowercases ASCII, and uses the `rank_bm25` defaults.
/// Equal scores retain the input corpus order, matching Python's stable sort.
pub(super) fn okapi_bm25_ranked_ids(
    query: &str,
    corpus: &[CanonicalEvent],
    limit: usize,
) -> Vec<String> {
    if corpus.is_empty() || limit == 0 {
        return Vec::new();
    }

    let documents = corpus
        .iter()
        .map(|event| bm25_tokens(&paper_retrieval_text(event)))
        .collect::<Vec<_>>();
    let mut document_frequency = HashMap::<String, usize>::new();
    for document in &documents {
        let unique = document.iter().map(String::as_str).collect::<HashSet<_>>();
        for token in unique {
            *document_frequency.entry(token.to_owned()).or_default() += 1;
        }
    }

    let corpus_size = documents.len() as f64;
    let average_length = documents.iter().map(Vec::len).sum::<usize>() as f64 / corpus_size;
    let mut idf = HashMap::<String, f64>::new();
    let mut negative = Vec::new();
    let mut idf_sum = 0.0;
    for (token, frequency) in document_frequency {
        let frequency = frequency as f64;
        let value = ((corpus_size - frequency + 0.5) / (frequency + 0.5)).ln();
        if value < 0.0 {
            negative.push(token.clone());
        }
        idf_sum += value;
        idf.insert(token, value);
    }
    if !idf.is_empty() {
        let floor = BM25_EPSILON * idf_sum / idf.len() as f64;
        for token in negative {
            idf.insert(token, floor);
        }
    }

    let query_tokens = bm25_tokens(query);
    let mut scored = documents
        .iter()
        .enumerate()
        .map(|(index, document)| {
            let mut frequencies = HashMap::<&str, usize>::new();
            for token in document {
                *frequencies.entry(token.as_str()).or_default() += 1;
            }
            let mut score = 0.0;
            for token in &query_tokens {
                let frequency = frequencies.get(token.as_str()).copied().unwrap_or(0) as f64;
                if frequency == 0.0 {
                    continue;
                }
                let inverse_document_frequency = idf.get(token).copied().unwrap_or(0.0);
                let length_normalization = if average_length == 0.0 {
                    1.0
                } else {
                    1.0 - BM25_B + BM25_B * document.len() as f64 / average_length
                };
                score += inverse_document_frequency * frequency * (BM25_K1 + 1.0)
                    / (frequency + BM25_K1 * length_normalization);
            }
            (index, score)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored
        .into_iter()
        .take(limit.min(corpus.len()))
        .map(|(index, _)| corpus[index].id.clone())
        .collect()
}

/// Role-aware text used by the paper for dense, sparse, and reranker inputs.
pub(super) fn paper_retrieval_text(event: &CanonicalEvent) -> String {
    let content = event.search_text();
    let label = match &event.data {
        EventData::Message { role, .. } => match role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::System => "System",
            MessageRole::Developer => "Developer",
            MessageRole::Tool => "Tool",
            MessageRole::Unknown => "Unknown",
        },
        EventData::ToolCall { .. } => "Assistant",
        EventData::ToolResult { .. } => "Tool",
        _ => return content,
    };
    format!("{label}: {content}")
}

fn bm25_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (index, byte) in text.as_bytes().iter().copied().enumerate() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') {
            start.get_or_insert(index);
        } else if let Some(token_start) = start.take() {
            tokens.push(text[token_start..index].to_ascii_lowercase());
        }
    }
    if let Some(token_start) = start {
        tokens.push(text[token_start..].to_ascii_lowercase());
    }
    tokens
}

/// Ranked message identifiers produced independently by dense and sparse search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfInput {
    pub dense_ranked_ids: Vec<String>,
    pub sparse_ranked_ids: Vec<String>,
    /// Stable, label-agnostic corpus order used only after score and rank ties.
    pub tie_break_ranked_ids: Vec<String>,
}

/// One candidate in the deterministic RRF order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfCandidate {
    pub message_id: String,
    /// One-based rank in the dense channel, if retrieved there.
    pub dense_rank: Option<usize>,
    /// One-based rank in the sparse channel, if retrieved there.
    pub sparse_rank: Option<usize>,
    pub rrf_score: f64,
}

/// Audit data for duplicate suppression and the resulting fused order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfTrace {
    pub k: u32,
    pub dense_input_count: usize,
    pub sparse_input_count: usize,
    pub dense_duplicates_ignored: Vec<String>,
    pub sparse_duplicates_ignored: Vec<String>,
    pub fused_order: Vec<String>,
}

/// Typed output of the reciprocal-rank-fusion stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfOutput {
    pub candidates: Vec<RrfCandidate>,
    pub trace: RrfTrace,
}

/// Cross-encoder score for one fused candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerankScore {
    pub message_id: String,
    pub score: f64,
}

/// Fused candidates plus exactly one cross-encoder score for each candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerankInput {
    pub fused_candidates: Vec<RrfCandidate>,
    pub scores: Vec<RerankScore>,
    pub top_n: usize,
}

/// One candidate after cross-encoder reranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerankedCandidate {
    pub message_id: String,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
    pub rrf_score: f64,
    pub rerank_score: f64,
}

/// Audit data for the reranking stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerankTrace {
    pub input_count: usize,
    pub requested_top_n: usize,
    pub output_order: Vec<String>,
}

/// Typed output of the cross-encoder reranking stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerankOutput {
    pub candidates: Vec<RerankedCandidate>,
    pub trace: RerankTrace,
}

#[derive(Debug, Default)]
struct PartialCandidate {
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
    rrf_score: f64,
}

/// Merge dense and sparse ranked lists using the paper's fixed `k = 60` RRF.
///
/// A repeated identifier inside one channel is ignored after its first
/// occurrence; its first, original one-based rank is preserved. This makes a
/// malformed retriever harmless without silently improving the duplicate's
/// rank. Empty identifiers are rejected because they cannot be traced back to
/// stored messages.
pub(super) fn reciprocal_rank_fusion(input: &RrfInput) -> Result<RrfOutput> {
    let mut corpus_order = HashMap::with_capacity(input.tie_break_ranked_ids.len());
    for (index, message_id) in input.tie_break_ranked_ids.iter().enumerate() {
        ensure!(
            !message_id.trim().is_empty(),
            "RRF tie-break order contains an empty message id at position {}",
            index + 1
        );
        ensure!(
            corpus_order.insert(message_id.as_str(), index).is_none(),
            "RRF tie-break order contains duplicate message id: {message_id}"
        );
    }
    let mut candidates = HashMap::<String, PartialCandidate>::new();
    let dense_duplicates_ignored = add_channel(
        &mut candidates,
        &input.dense_ranked_ids,
        |candidate, rank, contribution| {
            candidate.dense_rank = Some(rank);
            candidate.rrf_score += contribution;
        },
    )?;
    let sparse_duplicates_ignored = add_channel(
        &mut candidates,
        &input.sparse_ranked_ids,
        |candidate, rank, contribution| {
            candidate.sparse_rank = Some(rank);
            candidate.rrf_score += contribution;
        },
    )?;

    for message_id in candidates.keys() {
        ensure!(
            corpus_order.contains_key(message_id.as_str()),
            "RRF candidate {message_id} is missing from the tie-break corpus order"
        );
    }
    let mut fused = candidates
        .into_iter()
        .map(|(message_id, candidate)| RrfCandidate {
            message_id,
            dense_rank: candidate.dense_rank,
            sparse_rank: candidate.sparse_rank,
            rrf_score: candidate.rrf_score,
        })
        .collect::<Vec<_>>();
    fused.sort_by(|left, right| {
        right
            .rrf_score
            .total_cmp(&left.rrf_score)
            .then_with(|| best_rank(left).cmp(&best_rank(right)))
            .then_with(|| {
                corpus_order[left.message_id.as_str()].cmp(&corpus_order[right.message_id.as_str()])
            })
    });
    let fused_order = fused
        .iter()
        .map(|candidate| candidate.message_id.clone())
        .collect();

    Ok(RrfOutput {
        candidates: fused,
        trace: RrfTrace {
            k: PAPER_RRF_K,
            dense_input_count: input.dense_ranked_ids.len(),
            sparse_input_count: input.sparse_ranked_ids.len(),
            dense_duplicates_ignored,
            sparse_duplicates_ignored,
            fused_order,
        },
    })
}

fn add_channel(
    candidates: &mut HashMap<String, PartialCandidate>,
    ranked_ids: &[String],
    mut apply: impl FnMut(&mut PartialCandidate, usize, f64),
) -> Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    for (index, message_id) in ranked_ids.iter().enumerate() {
        ensure!(
            !message_id.trim().is_empty(),
            "retrieval channel contains an empty message id at rank {}",
            index + 1
        );
        if !seen.insert(message_id.as_str()) {
            duplicates.push(message_id.clone());
            continue;
        }
        let rank = index + 1;
        let contribution = 1.0 / (f64::from(PAPER_RRF_K) + rank as f64);
        apply(
            candidates.entry(message_id.clone()).or_default(),
            rank,
            contribution,
        );
    }
    Ok(duplicates)
}

fn optional_rank(rank: Option<usize>) -> usize {
    rank.unwrap_or(usize::MAX)
}

fn best_rank(candidate: &RrfCandidate) -> usize {
    optional_rank(candidate.dense_rank).min(optional_rank(candidate.sparse_rank))
}

/// Apply cross-encoder scores and retain the top `N` messages.
///
/// Scores are intentionally not blended with RRF: the paper uses RRF to form
/// the broad pool, then lets the cross-encoder define the final ranking. RRF
/// order is only a deterministic tie-break for equal reranker scores.
pub(super) fn apply_reranker(input: &RerankInput) -> Result<RerankOutput> {
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
    use crate::model::{AgentKind, EventSource, MessageRole};

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
