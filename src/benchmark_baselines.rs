//! Deterministic, model-free baselines for message-retrieval benchmarks.
//!
//! Gold message and session labels are deliberately absent from
//! [`Message`](crate::benchmark_baselines::Message).
//! Oracle labels enter only through evaluator-side function arguments, which
//! prevents them from leaking into BM25 text or another deployable baseline.

use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

/// `rank_bm25.BM25Okapi` 0.2.2 defaults used by the public LazyMem artifact.
pub const BM25_K1: f64 = 1.5;
/// BM25 document-length normalization strength used by `rank_bm25` 0.2.2.
pub const BM25_B: f64 = 0.75;
/// Multiplier applied to average IDF when replacing negative BM25 IDF values.
pub const BM25_EPSILON: f64 = 0.25;

/// One label-free message in the benchmark retrieval corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    /// Unique opaque message identifier used for ranking and evaluation.
    pub id: String,
    /// Source role, such as `user` or `assistant`.
    pub role: String,
    /// Raw message text indexed by the baseline.
    pub content: String,
    /// Opaque identifier of the containing session occurrence.
    pub session_id: String,
    /// Stable zero-based position in the complete retrieval corpus.
    pub corpus_index: usize,
}

/// One result in a baseline's one-based ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineHit {
    /// Identifier of the ranked message.
    pub message_id: String,
    /// Original zero-based corpus position of the ranked message.
    pub corpus_index: usize,
    /// One-based position in this result ranking.
    pub rank: usize,
    /// BM25 emits a score, including explicit zeroes in its tail. Other
    /// deterministic baselines do not invent a numeric relevance score.
    pub score: Option<f64>,
}

/// Invalid corpus or evaluator-only oracle labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaselineError {
    /// A corpus message has an empty identifier.
    EmptyMessageId {
        /// Zero-based position of the invalid message in the input slice.
        position: usize,
    },
    /// Multiple corpus messages use the same identifier.
    DuplicateMessageId {
        /// Repeated message identifier.
        id: String,
    },
    /// Multiple corpus messages use the same stable corpus position.
    DuplicateCorpusIndex {
        /// Repeated zero-based corpus position.
        corpus_index: usize,
    },
    /// An evaluator-supplied oracle label is empty.
    EmptyOracleLabel {
        /// Oracle label category, such as `message` or `session`.
        kind: &'static str,
        /// Zero-based position of the invalid label.
        position: usize,
    },
    /// An evaluator-supplied oracle label occurs more than once.
    DuplicateOracleLabel {
        /// Oracle label category, such as `message` or `session`.
        kind: &'static str,
        /// Repeated oracle label.
        value: String,
    },
    /// A gold message identifier does not occur in the corpus.
    UnknownGoldMessageId {
        /// Missing gold message identifier.
        id: String,
    },
    /// A gold session identifier does not occur in the corpus.
    UnknownGoldSessionId {
        /// Missing gold session identifier.
        session_id: String,
    },
}

impl fmt::Display for BaselineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMessageId { position } => {
                write!(
                    formatter,
                    "message ID at corpus position {position} is empty"
                )
            }
            Self::DuplicateMessageId { id } => {
                write!(formatter, "duplicate message ID {id:?}")
            }
            Self::DuplicateCorpusIndex { corpus_index } => {
                write!(formatter, "duplicate corpus_index {corpus_index}")
            }
            Self::EmptyOracleLabel { kind, position } => write!(
                formatter,
                "empty {kind} oracle label at zero-based position {position}"
            ),
            Self::DuplicateOracleLabel { kind, value } => {
                write!(formatter, "duplicate {kind} oracle label {value:?}")
            }
            Self::UnknownGoldMessageId { id } => {
                write!(
                    formatter,
                    "gold message ID {id:?} is absent from the corpus"
                )
            }
            Self::UnknownGoldSessionId { session_id } => write!(
                formatter,
                "gold session ID {session_id:?} is absent from the corpus"
            ),
        }
    }
}

impl Error for BaselineError {}

/// Prefixes message content with the canonical role label used for dense,
/// sparse, and reranker inputs.
pub fn role_aware_retrieval_text(message: &Message) -> String {
    let trimmed = message.role.trim();
    let lowercase = trimmed.to_ascii_lowercase();
    let label = match lowercase.as_str() {
        "user" | "human" => "User".to_owned(),
        "assistant" | "model" | "bot" => "Assistant".to_owned(),
        "system" => "System".to_owned(),
        "developer" => "Developer".to_owned(),
        "tool" => "Tool".to_owned(),
        "" => "Unknown".to_owned(),
        _ => title_case_role(trimmed),
    };
    format!("{label}: {}", message.content)
}

fn title_case_role(role: &str) -> String {
    let mut capitalize = true;
    let mut value = String::with_capacity(role.len());
    for character in role.chars() {
        if character.is_alphanumeric() {
            if capitalize {
                value.extend(character.to_uppercase());
            } else {
                value.extend(character.to_lowercase());
            }
            capitalize = false;
        } else {
            value.push(character);
            capitalize = true;
        }
    }
    value
}

/// Full `rank_bm25.BM25Okapi` ranking, including its stable zero-score tail.
///
/// Query-token duplicates intentionally contribute repeatedly, matching
/// `BM25Okapi.get_scores`. Equal scores are ordered by [`Message::corpus_index`].
pub fn bm25(
    messages: &[Message],
    query: &str,
    limit: usize,
) -> Result<Vec<BaselineHit>, BaselineError> {
    let corpus = validated_corpus(messages)?;
    if corpus.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let documents = corpus
        .iter()
        .map(|message| bm25_tokens(&role_aware_retrieval_text(message)))
        .collect::<Vec<_>>();
    Ok(bm25_tokenized(corpus, documents, bm25_tokens(query), limit))
}

/// Official LongMemEval flat user-turn BM25 baseline.
///
/// Unlike [`bm25`], this ranks only messages whose normalized role is exactly
/// `user`, indexes their raw content without a role prefix, and tokenizes both
/// documents and the query with Python's case-sensitive `split(" ")`. Empty
/// tokens created by leading, repeated, or trailing ASCII spaces are retained.
/// BM25Okapi parameters, negative-IDF epsilon replacement, duplicate query
/// token contributions, the zero-score tail, and corpus-index tie-breaking are
/// otherwise identical to [`bm25`].
pub fn longmemeval_official_bm25(
    messages: &[Message],
    query: &str,
    limit: usize,
) -> Result<Vec<BaselineHit>, BaselineError> {
    // Validate the complete input before projecting user turns so an invalid
    // assistant message cannot be hidden by the official corpus projection.
    let corpus = validated_corpus(messages)?
        .into_iter()
        .filter(|message| message.role == "user")
        .collect::<Vec<_>>();
    if corpus.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let documents = corpus
        .iter()
        .map(|message| split_ascii_space(&message.content))
        .collect::<Vec<_>>();
    Ok(bm25_tokenized(
        corpus,
        documents,
        split_ascii_space(query),
        limit,
    ))
}

fn bm25_tokenized(
    corpus: Vec<&Message>,
    documents: Vec<Vec<String>>,
    query_tokens: Vec<String>,
    limit: usize,
) -> Vec<BaselineHit> {
    debug_assert!(!corpus.is_empty());
    debug_assert_eq!(corpus.len(), documents.len());
    let mut document_frequencies = HashMap::<String, usize>::new();
    let mut vocabulary_order = Vec::<String>::new();
    for document in &documents {
        let mut seen_in_document = HashSet::<&str>::new();
        for token in document {
            if !seen_in_document.insert(token) {
                continue;
            }
            if !document_frequencies.contains_key(token) {
                vocabulary_order.push(token.clone());
            }
            *document_frequencies.entry(token.clone()).or_default() += 1;
        }
    }

    let corpus_size = documents.len() as f64;
    let average_document_length =
        documents.iter().map(Vec::len).sum::<usize>() as f64 / corpus_size;
    let mut inverse_document_frequencies = HashMap::<String, f64>::new();
    let mut negative_tokens = Vec::<String>::new();
    let mut idf_sum = 0.0;
    for token in &vocabulary_order {
        let frequency = document_frequencies[token] as f64;
        // Preserve rank_bm25's two-log operation rather than algebraically
        // replacing it with log(numerator / denominator).
        let inverse_document_frequency =
            (corpus_size - frequency + 0.5).ln() - (frequency + 0.5).ln();
        if inverse_document_frequency < 0.0 {
            negative_tokens.push(token.clone());
        }
        idf_sum += inverse_document_frequency;
        inverse_document_frequencies.insert(token.clone(), inverse_document_frequency);
    }
    let average_idf = idf_sum / inverse_document_frequencies.len() as f64;
    let epsilon_floor = BM25_EPSILON * average_idf;
    for token in negative_tokens {
        inverse_document_frequencies.insert(token, epsilon_floor);
    }

    let mut scored = corpus
        .iter()
        .zip(&documents)
        .map(|(message, document)| {
            let mut term_frequencies = HashMap::<&str, usize>::new();
            for token in document {
                *term_frequencies.entry(token).or_default() += 1;
            }
            let mut score = 0.0;
            for token in &query_tokens {
                let frequency = term_frequencies.get(token.as_str()).copied().unwrap_or(0) as f64;
                if frequency == 0.0 {
                    continue;
                }
                let inverse_document_frequency = inverse_document_frequencies
                    .get(token)
                    .copied()
                    .unwrap_or(0.0);
                let denominator = frequency
                    + BM25_K1
                        * (1.0 - BM25_B + BM25_B * document.len() as f64 / average_document_length);
                score += inverse_document_frequency * frequency * (BM25_K1 + 1.0) / denominator;
            }
            (message, score)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.corpus_index.cmp(&right.0.corpus_index))
    });
    scored
        .into_iter()
        .take(limit.min(corpus.len()))
        .enumerate()
        .map(|(position, (message, score))| BaselineHit {
            message_id: message.id.clone(),
            corpus_index: message.corpus_index,
            rank: position + 1,
            score: Some(score),
        })
        .collect()
}

/// BM25 control that removes non-positive results after applying the requested
/// channel limit. It must not silently replace the paper/public zero-tail path.
pub fn bm25_positive(
    messages: &[Message],
    query: &str,
    limit: usize,
) -> Result<Vec<BaselineHit>, BaselineError> {
    let mut positive = bm25(messages, query, limit)?
        .into_iter()
        .filter(|hit| hit.score.is_some_and(|score| score > 0.0))
        .collect::<Vec<_>>();
    for (position, hit) in positive.iter_mut().enumerate() {
        hit.rank = position + 1;
    }
    Ok(positive)
}

/// Most-recent-first message baseline.
pub fn recent(messages: &[Message], limit: usize) -> Result<Vec<BaselineHit>, BaselineError> {
    let mut corpus = validated_corpus(messages)?;
    corpus.reverse();
    Ok(unscored_hits(corpus.into_iter().take(limit)))
}

/// Complete chronological history baseline.
pub fn full_history(messages: &[Message]) -> Result<Vec<BaselineHit>, BaselineError> {
    Ok(unscored_hits(validated_corpus(messages)?.into_iter()))
}

/// Evaluator-only oracle containing exactly the labeled gold turns, in
/// chronological corpus order.
pub fn oracle_turn(
    messages: &[Message],
    gold_message_ids: &[String],
) -> Result<Vec<BaselineHit>, BaselineError> {
    let corpus = validated_corpus(messages)?;
    let labels = validated_labels(gold_message_ids, "message")?;
    let corpus_ids = corpus
        .iter()
        .map(|message| message.id.as_str())
        .collect::<HashSet<_>>();
    for label in &labels {
        if !corpus_ids.contains(label) {
            return Err(BaselineError::UnknownGoldMessageId {
                id: (*label).to_owned(),
            });
        }
    }
    Ok(unscored_hits(
        corpus
            .into_iter()
            .filter(|message| labels.contains(message.id.as_str())),
    ))
}

/// Evaluator-only oracle containing every message in a labeled gold session,
/// in chronological corpus order.
pub fn oracle_session(
    messages: &[Message],
    gold_session_ids: &[String],
) -> Result<Vec<BaselineHit>, BaselineError> {
    let corpus = validated_corpus(messages)?;
    let labels = validated_labels(gold_session_ids, "session")?;
    let corpus_sessions = corpus
        .iter()
        .map(|message| message.session_id.as_str())
        .collect::<HashSet<_>>();
    for label in &labels {
        if !corpus_sessions.contains(label) {
            return Err(BaselineError::UnknownGoldSessionId {
                session_id: (*label).to_owned(),
            });
        }
    }
    Ok(unscored_hits(corpus.into_iter().filter(|message| {
        labels.contains(message.session_id.as_str())
    })))
}

fn validated_corpus(messages: &[Message]) -> Result<Vec<&Message>, BaselineError> {
    let mut ids = HashSet::<&str>::with_capacity(messages.len());
    let mut indices = HashSet::<usize>::with_capacity(messages.len());
    for (position, message) in messages.iter().enumerate() {
        if message.id.trim().is_empty() {
            return Err(BaselineError::EmptyMessageId { position });
        }
        if !ids.insert(message.id.as_str()) {
            return Err(BaselineError::DuplicateMessageId {
                id: message.id.clone(),
            });
        }
        if !indices.insert(message.corpus_index) {
            return Err(BaselineError::DuplicateCorpusIndex {
                corpus_index: message.corpus_index,
            });
        }
    }
    let mut corpus = messages.iter().collect::<Vec<_>>();
    corpus.sort_by_key(|message| message.corpus_index);
    Ok(corpus)
}

fn validated_labels<'a>(
    labels: &'a [String],
    kind: &'static str,
) -> Result<HashSet<&'a str>, BaselineError> {
    let mut values = HashSet::with_capacity(labels.len());
    for (position, label) in labels.iter().enumerate() {
        if label.trim().is_empty() {
            return Err(BaselineError::EmptyOracleLabel { kind, position });
        }
        if !values.insert(label.as_str()) {
            return Err(BaselineError::DuplicateOracleLabel {
                kind,
                value: label.clone(),
            });
        }
    }
    Ok(values)
}

fn unscored_hits<'a>(messages: impl Iterator<Item = &'a Message>) -> Vec<BaselineHit> {
    messages
        .enumerate()
        .map(|(position, message)| BaselineHit {
            message_id: message.id.clone(),
            corpus_index: message.corpus_index,
            rank: position + 1,
            score: None,
        })
        .collect()
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

fn split_ascii_space(text: &str) -> Vec<String> {
    text.split(' ').map(str::to_owned).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn message(index: usize, role: &str, content: &str, session: &str) -> Message {
        Message {
            id: format!("m{index}"),
            role: role.to_owned(),
            content: content.to_owned(),
            session_id: session.to_owned(),
            corpus_index: index,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected:.15}, got {actual:.15}"
        );
    }

    #[test]
    fn renders_role_aware_retrieval_text() {
        assert_eq!(
            role_aware_retrieval_text(&message(0, "user", "hello", "s")),
            "User: hello"
        );
        assert_eq!(
            role_aware_retrieval_text(&message(0, "assistant", "hello", "s")),
            "Assistant: hello"
        );
        assert_eq!(
            role_aware_retrieval_text(&message(0, "custom-role", "hello", "s")),
            "Custom-Role: hello"
        );
        assert_eq!(
            bm25_tokens("ÄLPHA foo-Bar_baz 中文"),
            ["lpha", "foo-bar_baz"]
        );
    }

    #[test]
    fn bm25_matches_rank_bm25_0_2_2_golden_scores() {
        // Deliberately scramble the input slice: ties must still use corpus_index.
        let corpus = vec![
            message(2, "user", "common gamma", "s2"),
            message(0, "user", "common alpha", "s1"),
            message(3, "assistant", "delta epsilon", "s2"),
            message(1, "assistant", "common beta", "s1"),
        ];
        let ranked = bm25(&corpus, "common", 4).unwrap();
        assert_eq!(
            ranked
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m1", "m2", "m3"]
        );
        for hit in &ranked[..3] {
            assert_close(hit.score.unwrap(), 0.10591223254840046);
        }
        assert_eq!(ranked[3].score, Some(0.0));

        let repeated_query = bm25(&corpus, "common common", 4).unwrap();
        for hit in &repeated_query[..3] {
            assert_close(hit.score.unwrap(), 0.21182446509680092);
        }
    }

    #[test]
    fn official_longmemeval_bm25_matches_split_space_user_turn_golden() {
        // Python rank_bm25 0.2.2 documents are exactly:
        // ["Alpha", "", "common"], ["alpha", "common"],
        // and ["common", "tail", ""]. The assistant turn is not indexed.
        let corpus = vec![
            message(3, "user", "common tail ", "s2"),
            message(1, "assistant", "Alpha common", "s1"),
            message(0, "user", "Alpha  common", "s1"),
            message(2, "user", "alpha common", "s2"),
        ];
        let ranked = longmemeval_official_bm25(&corpus, "Alpha common", 10).unwrap();
        assert_eq!(
            ranked
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m3", "m2"]
        );
        assert_close(ranked[0].score.unwrap(), 0.4398699916590051);
        assert_close(ranked[1].score.unwrap(), -0.043751900663826365);
        assert_close(ranked[2].score.unwrap(), -0.0520709240294835);

        // `"".split(" ")` is one empty token. Only the leading/repeated or
        // trailing-space documents contain that token; zero outranks negatives.
        let empty_query = longmemeval_official_bm25(&corpus, "", 10).unwrap();
        assert_eq!(
            empty_query
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m2", "m0", "m3"]
        );
        assert_eq!(empty_query[0].score, Some(0.0));
        assert_close(empty_query[1].score.unwrap(), -0.043751900663826365);
        assert_close(empty_query[2].score.unwrap(), -0.043751900663826365);

        let missing = longmemeval_official_bm25(&corpus, "missing", 10).unwrap();
        assert_eq!(
            missing
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m2", "m3"]
        );
        assert!(missing.iter().all(|hit| hit.score == Some(0.0)));
    }

    #[test]
    fn bm25_keeps_zero_tail_while_positive_control_removes_it() {
        let corpus = vec![
            message(0, "user", "apple apple banana", "s"),
            message(1, "assistant", "banana-carrot", "s"),
            message(2, "tool", "snake_case x-y", "s"),
            message(3, "user", "unrelated", "s"),
        ];
        let ranked = bm25(&corpus, "apple", 4).unwrap();
        assert_close(ranked[0].score.unwrap(), 1.0561219789528882);
        assert_eq!(
            ranked[1..].iter().map(|hit| hit.score).collect::<Vec<_>>(),
            [Some(0.0), Some(0.0), Some(0.0)]
        );
        let positive = bm25_positive(&corpus, "apple", 4).unwrap();
        assert_eq!(positive.len(), 1);
        assert_eq!(
            (positive[0].message_id.as_str(), positive[0].rank),
            ("m0", 1)
        );

        let missing = bm25(&corpus, "missing", 4).unwrap();
        assert_eq!(
            missing
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m1", "m2", "m3"]
        );
        assert!(missing.iter().all(|hit| hit.score == Some(0.0)));
        assert!(bm25_positive(&corpus, "missing", 4).unwrap().is_empty());
    }

    #[test]
    fn recent_and_full_history_have_explicit_orders() {
        let corpus = vec![
            message(2, "user", "third", "s2"),
            message(0, "user", "first", "s1"),
            message(1, "assistant", "second", "s1"),
        ];
        assert_eq!(
            full_history(&corpus)
                .unwrap()
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m1", "m2"]
        );
        assert_eq!(
            recent(&corpus, 2)
                .unwrap()
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m2", "m1"]
        );
    }

    #[test]
    fn oracle_labels_are_evaluator_only_arguments() {
        let corpus = vec![
            message(0, "user", "first", "s1"),
            message(1, "assistant", "second", "s1"),
            message(2, "user", "third", "s2"),
        ];
        let turn = oracle_turn(&corpus, &["m2".to_owned(), "m0".to_owned()]).unwrap();
        assert_eq!(
            turn.iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m2"]
        );
        let session = oracle_session(&corpus, &["s1".to_owned()]).unwrap();
        assert_eq!(
            session
                .iter()
                .map(|hit| hit.message_id.as_str())
                .collect::<Vec<_>>(),
            ["m0", "m1"]
        );
        assert!(oracle_turn(&corpus, &[]).unwrap().is_empty());
        assert!(oracle_session(&corpus, &[]).unwrap().is_empty());
    }

    #[test]
    fn every_baseline_rejects_invalid_corpus_ids() {
        let duplicate = vec![
            message(0, "user", "first", "s"),
            Message {
                id: "m0".to_owned(),
                role: "assistant".to_owned(),
                content: "second".to_owned(),
                session_id: "s".to_owned(),
                corpus_index: 1,
            },
        ];
        assert!(matches!(
            bm25(&duplicate, "q", 10),
            Err(BaselineError::DuplicateMessageId { .. })
        ));
        assert!(matches!(
            longmemeval_official_bm25(&duplicate, "q", 10),
            Err(BaselineError::DuplicateMessageId { .. })
        ));
        assert!(matches!(
            recent(&duplicate, 10),
            Err(BaselineError::DuplicateMessageId { .. })
        ));
        assert!(matches!(
            full_history(&duplicate),
            Err(BaselineError::DuplicateMessageId { .. })
        ));
        assert!(matches!(
            oracle_turn(&duplicate, &[]),
            Err(BaselineError::DuplicateMessageId { .. })
        ));
        assert!(matches!(
            oracle_session(&duplicate, &[]),
            Err(BaselineError::DuplicateMessageId { .. })
        ));

        let empty = vec![Message {
            id: "  ".to_owned(),
            role: "user".to_owned(),
            content: "content".to_owned(),
            session_id: "s".to_owned(),
            corpus_index: 0,
        }];
        assert!(matches!(
            bm25_positive(&empty, "q", 10),
            Err(BaselineError::EmptyMessageId { .. })
        ));
    }

    #[test]
    fn oracles_reject_invalid_or_unknown_labels() {
        let corpus = vec![message(0, "user", "first", "s1")];
        assert!(matches!(
            oracle_turn(&corpus, &["missing".to_owned()]),
            Err(BaselineError::UnknownGoldMessageId { .. })
        ));
        assert!(matches!(
            oracle_session(&corpus, &["missing".to_owned()]),
            Err(BaselineError::UnknownGoldSessionId { .. })
        ));
        assert!(matches!(
            oracle_turn(&corpus, &["m0".to_owned(), "m0".to_owned()]),
            Err(BaselineError::DuplicateOracleLabel { .. })
        ));
        assert!(matches!(
            oracle_session(&corpus, &[" ".to_owned()]),
            Err(BaselineError::EmptyOracleLabel { .. })
        ));
    }
}
