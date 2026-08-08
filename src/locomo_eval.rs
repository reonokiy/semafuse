//! Pure, model-free implementation of LoCoMo's released QA F1 protocol.
//!
//! The control flow and normalization rules are pinned to
//! `snap-research/locomo@3eb6f2c`'s `task_eval/evaluation.py`: category 1 is
//! scored by comma-delimited component matching, category 3 truncates the
//! reference at the first semicolon, and categories 2--4 use token F1.  This
//! is the original LoCoMo metric, not the binary LLM judge used in LazyMem's
//! paper-v2 result table.

use crate::locomo::LocomoCategory;
use nltk_porter::{Mode, PorterStemmer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Schema version for serializable LoCoMo score records.
pub const LOCOMO_EVAL_SCHEMA_VERSION: u32 = 1;
/// Stable metric name shared with the Python benchmark harness.
pub const LOCOMO_OFFICIAL_F1_METRIC: &str = "locomo_official_f1";
/// Pinned upstream repository revision whose scorer protocol is implemented.
pub const LOCOMO_UPSTREAM_COMMIT: &str = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376";
/// SHA-256 of upstream `task_eval/evaluation.py` at [`LOCOMO_UPSTREAM_COMMIT`].
pub const LOCOMO_UPSTREAM_SCORER_SHA256: &str =
    "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd";
/// SPDX identifier of the released LoCoMo benchmark data.
pub const LOCOMO_DATA_LICENSE: &str = "CC-BY-NC-4.0";
/// Stemmer implementation recorded in every score for reproducibility.
pub const LOCOMO_STEMMER: &str = "nltk-porter/0.1.0 Mode::Nltk";

/// Auditable provenance attached to every native LoCoMo F1 result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoF1Provenance {
    /// Repository containing the released scorer.
    pub upstream: String,
    /// Frozen upstream revision.
    pub upstream_commit: String,
    /// Digest of the scorer source at the frozen revision.
    pub upstream_scorer_sha256: String,
    /// License of the benchmark data consumed by the scorer.
    pub data_license: String,
    /// Concrete English stemmer used by this native implementation.
    pub stemmer: String,
    /// The LazyMem paper reports a separate, binary LLM-judge metric.
    pub paper_comparable_lj: bool,
}

impl Default for LocomoF1Provenance {
    fn default() -> Self {
        Self {
            upstream: "snap-research/locomo".to_owned(),
            upstream_commit: LOCOMO_UPSTREAM_COMMIT.to_owned(),
            upstream_scorer_sha256: LOCOMO_UPSTREAM_SCORER_SHA256.to_owned(),
            data_license: LOCOMO_DATA_LICENSE.to_owned(),
            stemmer: LOCOMO_STEMMER.to_owned(),
            paper_comparable_lj: false,
        }
    }
}

/// Serializable result of scoring one LoCoMo prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocomoF1Score {
    /// Record schema version.
    pub schema_version: u32,
    /// Metric identifier used by report aggregation.
    pub metric: String,
    /// Numeric LoCoMo score in the inclusive range `[0, 1]`.
    pub score: f64,
    /// Numeric category from the released dataset.
    pub category: u8,
    /// Stable category label used by benchmark reports.
    pub category_name: String,
    /// Frozen evaluator provenance.
    pub provenance: LocomoF1Provenance,
}

/// Applies the released category-aware LoCoMo token-F1 protocol.
///
/// Category 1 splits both strings on commas. Each reference component receives
/// the best token-F1 score from any predicted component, and those maxima are
/// averaged. Category 3 ignores reference text following the first semicolon.
/// Empty normalized inputs score zero, matching upstream's `num_same == 0`
/// branch (including the both-empty case).
pub fn score_locomo_answer(
    category: LocomoCategory,
    prediction: &str,
    reference_answer: &str,
) -> LocomoF1Score {
    let stemmer = PorterStemmer::new(Mode::Nltk);
    let score = match category {
        LocomoCategory::MultiHop => component_f1(prediction, reference_answer, &stemmer),
        LocomoCategory::OpenDomain => {
            let reference_answer = reference_answer
                .split_once(';')
                .map_or(reference_answer, |(answer, _)| answer)
                .trim();
            token_f1(prediction, reference_answer, &stemmer)
        }
        LocomoCategory::Temporal | LocomoCategory::SingleHop => {
            token_f1(prediction, reference_answer, &stemmer)
        }
    };
    LocomoF1Score {
        schema_version: LOCOMO_EVAL_SCHEMA_VERSION,
        metric: LOCOMO_OFFICIAL_F1_METRIC.to_owned(),
        score,
        category: category.number(),
        category_name: category.name().to_owned(),
        provenance: LocomoF1Provenance::default(),
    }
}

fn component_f1(prediction: &str, reference_answer: &str, stemmer: &PorterStemmer) -> f64 {
    let predicted = prediction.split(',').map(str::trim).collect::<Vec<_>>();
    let expected = reference_answer
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    let total = expected
        .iter()
        .map(|expected_component| {
            predicted
                .iter()
                .map(|predicted_component| {
                    token_f1(predicted_component, expected_component, stemmer)
                })
                .fold(0.0_f64, f64::max)
        })
        .sum::<f64>();
    total / expected.len() as f64
}

fn token_f1(prediction: &str, reference_answer: &str, stemmer: &PorterStemmer) -> f64 {
    let predicted = stemmed_tokens(prediction, stemmer);
    let expected = stemmed_tokens(reference_answer, stemmer);
    if predicted.is_empty() || expected.is_empty() {
        return 0.0;
    }

    let mut expected_counts = HashMap::<&str, usize>::new();
    for token in &expected {
        *expected_counts.entry(token).or_default() += 1;
    }
    let mut overlap = 0_usize;
    for token in &predicted {
        if let Some(remaining) = expected_counts.get_mut(token.as_str())
            && *remaining > 0
        {
            overlap += 1;
            *remaining -= 1;
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f64 / predicted.len() as f64;
    let recall = overlap as f64 / expected.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

fn stemmed_tokens(text: &str, stemmer: &PorterStemmer) -> Vec<String> {
    normalize_answer(text)
        .split_whitespace()
        .map(|token| stemmer.stem(token))
        .collect()
}

fn normalize_answer(text: &str) -> String {
    // Upstream first deletes commas and then all `string.punctuation`; the
    // initial replacement is retained here to document that exact sequence.
    let without_commas = text.replace(',', "");
    let lowercase = without_commas.to_lowercase();
    let without_punctuation = lowercase
        .chars()
        .filter(|character| !character.is_ascii_punctuation())
        .collect::<String>();
    let without_articles = remove_articles(&without_punctuation);
    without_articles
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn remove_articles(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while cursor < text.len() {
        let tail = &text[cursor..];
        let article = ["the", "and", "an", "a"].into_iter().find(|article| {
            tail.starts_with(article)
                && is_word_boundary_before(text, cursor)
                && is_word_boundary_after(text, cursor + article.len())
        });
        if let Some(article) = article {
            output.push(' ');
            cursor += article.len();
        } else {
            let character = tail
                .chars()
                .next()
                .expect("cursor is before the UTF-8 string end");
            output.push(character);
            cursor += character.len_utf8();
        }
    }
    output
}

fn is_word_boundary_before(text: &str, byte_index: usize) -> bool {
    byte_index == 0
        || !text[..byte_index]
            .chars()
            .next_back()
            .is_some_and(is_word_character)
}

fn is_word_boundary_after(text: &str, byte_index: usize) -> bool {
    byte_index == text.len()
        || !text[byte_index..]
            .chars()
            .next()
            .is_some_and(is_word_character)
}

fn is_word_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn category_one_uses_component_max_matching_then_mean() {
        let result = score_locomo_answer(
            LocomoCategory::MultiHop,
            "blue car, irrelevant, red apples",
            "red apple, blue cars",
        );
        assert_eq!(result.score, 1.0);

        // Both gold components select the same best prediction, exactly as the
        // released scorer (matching is not one-to-one).
        let reused = score_locomo_answer(LocomoCategory::MultiHop, "red blue", "red, blue");
        assert_close(reused.score, 2.0 / 3.0);
    }

    #[test]
    fn category_two_uses_stemmed_token_f1() {
        let result = score_locomo_answer(
            LocomoCategory::Temporal,
            "The runners finished yesterday",
            "a runner finishes yesterday",
        );
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn category_three_truncates_reference_at_first_semicolon() {
        let result = score_locomo_answer(
            LocomoCategory::OpenDomain,
            "cats sleep",
            "a cat sleeps; this explanatory suffix is deliberately absent",
        );
        assert_eq!(result.score, 1.0);
    }

    #[test]
    fn category_four_preserves_counter_multiplicity() {
        let result = score_locomo_answer(LocomoCategory::SingleHop, "cat cat dog", "cat dog dog");
        assert_close(result.score, 2.0 / 3.0);
    }

    #[test]
    fn empty_normalized_values_score_zero_like_upstream() {
        for (prediction, reference) in [("", ""), ("the and a", "an"), ("", "cat")] {
            assert_eq!(
                score_locomo_answer(LocomoCategory::SingleHop, prediction, reference).score,
                0.0
            );
        }
        assert_eq!(
            score_locomo_answer(LocomoCategory::MultiHop, "", "").score,
            0.0
        );
    }

    #[test]
    fn normalization_matches_upstream_order_and_ascii_punctuation() {
        assert_eq!(normalize_answer(" The, CATS and a dog! "), "cats dog");
        // Punctuation is deleted, not replaced, before article removal.
        assert_eq!(normalize_answer("the-cat"), "thecat");
        // Unicode punctuation is not part of Python's `string.punctuation`;
        // it still creates a regex-style word boundary around the article.
        assert_eq!(normalize_answer("THE—cat"), "—cat");
        assert_eq!(normalize_answer("CafÉ"), "café");
    }

    #[test]
    fn porter_fixture_matches_released_scorer_examples() {
        let stemmer = PorterStemmer::new(Mode::Nltk);
        let fixtures = [
            ("relational", "relat"),
            ("conditional", "condit"),
            ("digitizer", "digit"),
            ("triplicate", "triplic"),
            ("formalize", "formal"),
            ("goodness", "good"),
            // These distinguish NLTK's Porter implementation from Snowball's
            // English/Porter2 implementation.
            ("communication", "commun"),
            ("community", "commun"),
            ("friendly", "friendli"),
            ("weekly", "weekli"),
            ("yes", "ye"),
        ];
        for (word, expected) in fixtures {
            assert_eq!(stemmer.stem(word), expected, "word={word}");
        }
    }

    #[test]
    fn result_exposes_metric_and_frozen_provenance() {
        let result = score_locomo_answer(LocomoCategory::SingleHop, "answer", "answer");
        assert_eq!(result.schema_version, 1);
        assert_eq!(result.metric, "locomo_official_f1");
        assert_eq!(result.category, 4);
        assert_eq!(result.category_name, "single-hop");
        assert_eq!(result.provenance.upstream, "snap-research/locomo");
        assert_eq!(
            result.provenance.upstream_commit,
            "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
        );
        assert_eq!(
            result.provenance.upstream_scorer_sha256,
            "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
        );
        assert_eq!(result.provenance.data_license, "CC-BY-NC-4.0");
        assert!(!result.provenance.paper_comparable_lj);

        let encoded = serde_json::to_value(&result).unwrap();
        assert_eq!(encoded["metric"], "locomo_official_f1");
        assert_eq!(encoded["provenance"]["paper_comparable_lj"], false);
    }
}
