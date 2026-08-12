//! Deterministic hybrid-retrieval primitives shared by memory strategies.

use crate::model::{CanonicalEvent, EventData, MessageRole};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
const BM25_EPSILON: f64 = 0.25;
const PAPER_RRF_K: u32 = 60;

/// Rank the exact scoped message corpus with `rank_bm25.BM25Okapi` semantics.
///
/// The public retrieval artifact tokenizes with `[A-Za-z0-9_\\-]+`,
/// lowercases ASCII, and uses the `rank_bm25` defaults. Equal scores retain
/// input corpus order, matching Python's stable sort.
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

pub(super) fn bm25_tokens(text: &str) -> Vec<String> {
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
    pub(super) dense_ranked_ids: Vec<String>,
    pub(super) sparse_ranked_ids: Vec<String>,
    /// Stable, label-agnostic corpus order used only after score and rank ties.
    pub(super) tie_break_ranked_ids: Vec<String>,
}

/// One candidate in the deterministic RRF order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfCandidate {
    pub(super) message_id: String,
    /// One-based rank in the dense channel, if retrieved there.
    pub(super) dense_rank: Option<usize>,
    /// One-based rank in the sparse channel, if retrieved there.
    pub(super) sparse_rank: Option<usize>,
    pub(super) rrf_score: f64,
}

/// Audit data for duplicate suppression and the resulting fused order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfTrace {
    pub(super) k: u32,
    pub(super) dense_input_count: usize,
    pub(super) sparse_input_count: usize,
    pub(super) dense_duplicates_ignored: Vec<String>,
    pub(super) sparse_duplicates_ignored: Vec<String>,
    pub(super) fused_order: Vec<String>,
}

/// Typed output of the reciprocal-rank-fusion stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RrfOutput {
    pub(super) candidates: Vec<RrfCandidate>,
    pub(super) trace: RrfTrace,
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
/// occurrence; its first, original one-based rank is preserved. Empty
/// identifiers are rejected because they cannot be traced to stored messages.
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
