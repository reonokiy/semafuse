//! Deterministic aggregation and paired comparison of scalar benchmark rows.
//!
//! Reports fail closed unless every method contains exactly one value for the
//! same complete set of question identifiers.  Bootstrap intervals use a local
//! SplitMix64 implementation so results do not depend on an external RNG crate
//! or on input row ordering.

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

/// Number of percentile-bootstrap resamples used by every report.
pub const BOOTSTRAP_RESAMPLES: usize = 2_000;
/// Fixed bootstrap seed used by every report.
pub const BOOTSTRAP_SEED: u64 = 42;

/// One method's scalar value for one benchmark question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarMetricRow {
    /// Stable method/configuration identifier.
    pub method: String,
    /// Stable benchmark question identifier.
    pub question_id: String,
    /// Benchmark-defined question category.
    pub question_type: String,
    /// Finite per-question metric value.
    pub value: f64,
}

/// Whether larger or smaller metric values count as a paired win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// A candidate value greater than the baseline is a win.
    HigherIsBetter,
    /// A candidate value less than the baseline is a win.
    LowerIsBetter,
}

/// Complete scalar benchmark report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarBenchmarkReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Caller-supplied metric identifier.
    pub metric: String,
    /// Method used as the paired-comparison baseline.
    pub baseline_method: String,
    /// Direction used only to classify paired wins and losses.
    pub direction: MetricDirection,
    /// Common question denominator enforced across all methods.
    pub denominator: usize,
    /// Reproducible bootstrap protocol.
    pub bootstrap: BootstrapProtocol,
    /// Per-method summaries keyed by stable method identifier.
    pub methods: BTreeMap<String, MethodScalarReport>,
    /// Candidate-minus-baseline paired comparisons; the baseline itself is omitted.
    pub comparisons: BTreeMap<String, PairedDeltaReport>,
}

/// Bootstrap settings recorded with each report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapProtocol {
    /// Number of with-replacement resamples.
    pub resamples: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Central interval confidence level.
    pub confidence_level: f64,
    /// Quantile convention used for interval endpoints.
    pub interval_method: String,
}

/// Overall and per-question-type summaries for one method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodScalarReport {
    /// Summary across the common full question set.
    pub overall: ScalarSummary,
    /// Summaries grouped by question type in lexical key order.
    pub by_question_type: BTreeMap<String, ScalarSummary>,
}

/// Mean, deterministic percentile interval, and denominator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarSummary {
    /// Arithmetic mean of the question-level values.
    pub mean: f64,
    /// Deterministic 95% percentile-bootstrap interval for the mean.
    pub ci95: ConfidenceInterval,
    /// Number of question-level values included.
    pub denominator: usize,
}

/// Inclusive lower and upper confidence bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// 2.5th percentile of bootstrap means.
    pub lower: f64,
    /// 97.5th percentile of bootstrap means.
    pub upper: f64,
}

/// Paired candidate-minus-baseline comparison over the common question set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedDeltaReport {
    /// Candidate method represented by this comparison.
    pub method: String,
    /// Baseline method subtracted from the candidate.
    pub baseline_method: String,
    /// Mean of `candidate_value - baseline_value` over paired questions.
    pub mean_delta: f64,
    /// Deterministic 95% percentile-bootstrap interval over paired deltas.
    pub delta_ci95: ConfidenceInterval,
    /// Common number of paired questions.
    pub denominator: usize,
    /// Number of pairs where the candidate is better under the report direction.
    pub wins: usize,
    /// Number of pairs with exactly equal finite values.
    pub ties: usize,
    /// Number of pairs where the candidate is worse under the report direction.
    pub losses: usize,
}

/// Invalid or non-comparable report input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkReportError {
    /// The metric identifier was empty or whitespace-only.
    EmptyMetric,
    /// No question-level rows were supplied.
    EmptyRows,
    /// At least two methods are required for a comparison report.
    FewerThanTwoMethods,
    /// The selected baseline was empty or absent.
    UnknownBaseline {
        /// Baseline identifier requested by the caller.
        baseline: String,
    },
    /// A method identifier was empty or whitespace-only.
    EmptyMethod {
        /// Zero-based input row containing the empty identifier.
        row: usize,
    },
    /// A question identifier was empty or whitespace-only.
    EmptyQuestionId {
        /// Zero-based input row containing the empty identifier.
        row: usize,
    },
    /// A question type was empty or whitespace-only.
    EmptyQuestionType {
        /// Zero-based input row containing the empty type.
        row: usize,
    },
    /// A question-level value was NaN or infinite.
    NonFiniteValue {
        /// Method whose value was non-finite.
        method: String,
        /// Question whose value was non-finite.
        question_id: String,
    },
    /// A `(method, question_id)` pair occurred more than once.
    DuplicatePair {
        /// Method in the duplicate pair.
        method: String,
        /// Question in the duplicate pair.
        question_id: String,
    },
    /// A method did not contain exactly the baseline question set.
    QuestionSetMismatch {
        /// Method with incomplete or unexpected questions.
        method: String,
        /// Baseline question IDs absent from the method.
        missing: Vec<String>,
        /// Method question IDs absent from the baseline.
        unexpected: Vec<String>,
    },
    /// Methods assigned different types to the same question.
    QuestionTypeMismatch {
        /// Method assigning the conflicting type.
        method: String,
        /// Question with inconsistent type metadata.
        question_id: String,
        /// Question type recorded by the baseline.
        expected: String,
        /// Conflicting question type recorded by the method.
        found: String,
    },
    /// Subtracting two finite values produced a non-finite paired delta.
    NonFinitePairedDelta {
        /// Candidate method whose subtraction overflowed.
        method: String,
        /// Question whose paired subtraction overflowed.
        question_id: String,
    },
}

impl fmt::Display for BenchmarkReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMetric => formatter.write_str("metric identifier must not be empty"),
            Self::EmptyRows => formatter.write_str("scalar benchmark rows must not be empty"),
            Self::FewerThanTwoMethods => {
                formatter.write_str("a comparison report requires at least two methods")
            }
            Self::UnknownBaseline { baseline } => {
                write!(formatter, "baseline method {baseline:?} is absent")
            }
            Self::EmptyMethod { row } => {
                write!(
                    formatter,
                    "method identifier is empty at zero-based row {row}"
                )
            }
            Self::EmptyQuestionId { row } => {
                write!(
                    formatter,
                    "question identifier is empty at zero-based row {row}"
                )
            }
            Self::EmptyQuestionType { row } => {
                write!(formatter, "question type is empty at zero-based row {row}")
            }
            Self::NonFiniteValue {
                method,
                question_id,
            } => write!(
                formatter,
                "metric value for method {method:?}, question {question_id:?} is not finite"
            ),
            Self::DuplicatePair {
                method,
                question_id,
            } => write!(
                formatter,
                "duplicate scalar metric pair for method {method:?}, question {question_id:?}"
            ),
            Self::QuestionSetMismatch {
                method,
                missing,
                unexpected,
            } => write!(
                formatter,
                "method {method:?} has a different question set (missing {missing:?}, unexpected {unexpected:?})"
            ),
            Self::QuestionTypeMismatch {
                method,
                question_id,
                expected,
                found,
            } => write!(
                formatter,
                "method {method:?} assigns question {question_id:?} type {found:?}; expected {expected:?}"
            ),
            Self::NonFinitePairedDelta {
                method,
                question_id,
            } => write!(
                formatter,
                "paired delta for method {method:?}, question {question_id:?} is not finite"
            ),
        }
    }
}

impl Error for BenchmarkReportError {}

#[derive(Debug, Clone)]
struct Observation {
    question_type: String,
    value: f64,
}

/// Builds a deterministic, fail-closed report for one scalar metric.
///
/// Input order does not affect the report.  Methods and questions are sorted by
/// identifier before aggregation.  Every non-baseline method must contain
/// exactly the baseline question set, and every `(method, question_id)` pair
/// must be unique.
pub fn build_scalar_report(
    metric: &str,
    baseline_method: &str,
    direction: MetricDirection,
    rows: &[ScalarMetricRow],
) -> Result<ScalarBenchmarkReport, BenchmarkReportError> {
    if metric.trim().is_empty() {
        return Err(BenchmarkReportError::EmptyMetric);
    }
    if rows.is_empty() {
        return Err(BenchmarkReportError::EmptyRows);
    }

    let mut observations: BTreeMap<String, BTreeMap<String, Observation>> = BTreeMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        if row.method.trim().is_empty() {
            return Err(BenchmarkReportError::EmptyMethod { row: row_index });
        }
        if row.question_id.trim().is_empty() {
            return Err(BenchmarkReportError::EmptyQuestionId { row: row_index });
        }
        if row.question_type.trim().is_empty() {
            return Err(BenchmarkReportError::EmptyQuestionType { row: row_index });
        }
        if !row.value.is_finite() {
            return Err(BenchmarkReportError::NonFiniteValue {
                method: row.method.clone(),
                question_id: row.question_id.clone(),
            });
        }
        let method = observations.entry(row.method.clone()).or_default();
        if method
            .insert(
                row.question_id.clone(),
                Observation {
                    question_type: row.question_type.clone(),
                    value: row.value,
                },
            )
            .is_some()
        {
            return Err(BenchmarkReportError::DuplicatePair {
                method: row.method.clone(),
                question_id: row.question_id.clone(),
            });
        }
    }

    if observations.len() < 2 {
        return Err(BenchmarkReportError::FewerThanTwoMethods);
    }
    let baseline =
        observations
            .get(baseline_method)
            .ok_or_else(|| BenchmarkReportError::UnknownBaseline {
                baseline: baseline_method.to_owned(),
            })?;
    let baseline_questions = baseline.keys().cloned().collect::<BTreeSet<_>>();

    for (method_name, method) in &observations {
        let questions = method.keys().cloned().collect::<BTreeSet<_>>();
        if questions != baseline_questions {
            return Err(BenchmarkReportError::QuestionSetMismatch {
                method: method_name.clone(),
                missing: baseline_questions.difference(&questions).cloned().collect(),
                unexpected: questions.difference(&baseline_questions).cloned().collect(),
            });
        }
        for question_id in &baseline_questions {
            let expected = &baseline[question_id].question_type;
            let found = &method[question_id].question_type;
            if found != expected {
                return Err(BenchmarkReportError::QuestionTypeMismatch {
                    method: method_name.clone(),
                    question_id: question_id.clone(),
                    expected: expected.clone(),
                    found: found.clone(),
                });
            }
        }
    }

    let methods = observations
        .iter()
        .map(|(method_name, method)| {
            let values = method
                .values()
                .map(|observation| observation.value)
                .collect::<Vec<_>>();
            let mut values_by_type: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for observation in method.values() {
                values_by_type
                    .entry(observation.question_type.clone())
                    .or_default()
                    .push(observation.value);
            }
            let by_question_type = values_by_type
                .into_iter()
                .map(|(question_type, values)| (question_type, summarize(&values)))
                .collect();
            (
                method_name.clone(),
                MethodScalarReport {
                    overall: summarize(&values),
                    by_question_type,
                },
            )
        })
        .collect();

    let mut comparisons = BTreeMap::new();
    for (method_name, method) in &observations {
        if method_name == baseline_method {
            continue;
        }
        let mut deltas = Vec::with_capacity(baseline_questions.len());
        let mut wins = 0;
        let mut ties = 0;
        let mut losses = 0;
        for question_id in &baseline_questions {
            let candidate = method[question_id].value;
            let baseline_value = baseline[question_id].value;
            let delta = candidate - baseline_value;
            if !delta.is_finite() {
                return Err(BenchmarkReportError::NonFinitePairedDelta {
                    method: method_name.clone(),
                    question_id: question_id.clone(),
                });
            }
            deltas.push(delta);
            if delta == 0.0 {
                ties += 1;
            } else {
                let candidate_wins = match direction {
                    MetricDirection::HigherIsBetter => delta > 0.0,
                    MetricDirection::LowerIsBetter => delta < 0.0,
                };
                if candidate_wins {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
        }
        let summary = summarize(&deltas);
        comparisons.insert(
            method_name.clone(),
            PairedDeltaReport {
                method: method_name.clone(),
                baseline_method: baseline_method.to_owned(),
                mean_delta: summary.mean,
                delta_ci95: summary.ci95,
                denominator: summary.denominator,
                wins,
                ties,
                losses,
            },
        );
    }

    Ok(ScalarBenchmarkReport {
        schema_version: 1,
        metric: metric.to_owned(),
        baseline_method: baseline_method.to_owned(),
        direction,
        denominator: baseline_questions.len(),
        bootstrap: BootstrapProtocol {
            resamples: BOOTSTRAP_RESAMPLES,
            seed: BOOTSTRAP_SEED,
            confidence_level: 0.95,
            interval_method: "percentile_linear".into(),
        },
        methods,
        comparisons,
    })
}

fn summarize(values: &[f64]) -> ScalarSummary {
    debug_assert!(!values.is_empty());
    ScalarSummary {
        mean: stable_mean(values),
        ci95: bootstrap_mean_ci(values),
        denominator: values.len(),
    }
}

fn stable_mean(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        0.0
    } else {
        scale * (values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64)
    }
}

fn bootstrap_mean_ci(values: &[f64]) -> ConfidenceInterval {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let mut rng = SplitMix64::new(BOOTSTRAP_SEED);
    let mut means = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let normalized_sum = (0..values.len())
            .map(|_| {
                let value = values[rng.index(values.len())];
                if scale == 0.0 { 0.0 } else { value / scale }
            })
            .sum::<f64>();
        means.push(if scale == 0.0 {
            0.0
        } else {
            scale * normalized_sum / values.len() as f64
        });
    }
    means.sort_by(f64::total_cmp);
    ConfidenceInterval {
        lower: linear_percentile(&means, 0.025),
        upper: linear_percentile(&means, 0.975),
    }
}

fn linear_percentile(sorted: &[f64], percentile: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * percentile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

/// Minimal fixed PRNG used only for deterministic bootstrap index sampling.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let bound = upper as u64;
        // Rejection sampling removes modulo bias over the full u64 output range.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn row(method: &str, question_id: &str, question_type: &str, value: f64) -> ScalarMetricRow {
        ScalarMetricRow {
            method: method.into(),
            question_id: question_id.into(),
            question_type: question_type.into(),
            value,
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected:.15}, got {actual:.15}"
        );
    }

    #[test]
    fn rejects_incomplete_method_question_sets() {
        let rows = vec![
            row("baseline", "q1", "single", 0.0),
            row("baseline", "q2", "multi", 1.0),
            row("candidate", "q1", "single", 1.0),
        ];
        assert_eq!(
            build_scalar_report(
                "recall_all_at_50",
                "baseline",
                MetricDirection::HigherIsBetter,
                &rows,
            ),
            Err(BenchmarkReportError::QuestionSetMismatch {
                method: "candidate".into(),
                missing: vec!["q2".into()],
                unexpected: vec![],
            })
        );
    }

    #[test]
    fn rejects_duplicate_method_question_pairs() {
        let rows = vec![
            row("baseline", "q1", "single", 0.0),
            row("baseline", "q1", "single", 1.0),
            row("candidate", "q1", "single", 1.0),
        ];
        assert_eq!(
            build_scalar_report("metric", "baseline", MetricDirection::HigherIsBetter, &rows,),
            Err(BenchmarkReportError::DuplicatePair {
                method: "baseline".into(),
                question_id: "q1".into(),
            })
        );
    }

    #[test]
    fn paired_delta_is_aligned_by_question_id_and_reports_win_tie_loss() {
        let rows = vec![
            // Deliberately shuffled across methods and questions.
            row("candidate", "q3", "multi", 5.0),
            row("baseline", "q2", "single", 2.0),
            row("candidate", "q1", "single", 3.0),
            row("baseline", "q3", "multi", 5.0),
            row("candidate", "q2", "single", 1.0),
            row("baseline", "q1", "single", 1.0),
        ];
        let report =
            build_scalar_report("score", "baseline", MetricDirection::HigherIsBetter, &rows)
                .unwrap();
        assert_eq!(report.denominator, 3);
        assert_eq!(report.bootstrap.resamples, 2_000);
        assert_eq!(report.bootstrap.seed, 42);

        let comparison = &report.comparisons["candidate"];
        close(comparison.mean_delta, 1.0 / 3.0);
        assert_eq!(comparison.denominator, 3);
        assert_eq!(
            (comparison.wins, comparison.ties, comparison.losses),
            (1, 1, 1)
        );
        assert!(comparison.delta_ci95.lower <= comparison.mean_delta);
        assert!(comparison.delta_ci95.upper >= comparison.mean_delta);

        let candidate = &report.methods["candidate"];
        close(candidate.overall.mean, 3.0);
        assert_eq!(candidate.overall.denominator, 3);
        assert_eq!(candidate.by_question_type["single"].denominator, 2);
        close(candidate.by_question_type["single"].mean, 2.0);
        assert_eq!(candidate.by_question_type["multi"].denominator, 1);
        close(candidate.by_question_type["multi"].mean, 5.0);

        let repeated =
            build_scalar_report("score", "baseline", MetricDirection::HigherIsBetter, &rows)
                .unwrap();
        assert_eq!(report, repeated);
    }

    #[test]
    fn direction_changes_wins_without_changing_raw_candidate_minus_baseline_delta() {
        let rows = vec![
            row("baseline", "q1", "latency", 2.0),
            row("candidate", "q1", "latency", 1.0),
        ];
        let report = build_scalar_report(
            "latency_s",
            "baseline",
            MetricDirection::LowerIsBetter,
            &rows,
        )
        .unwrap();
        let comparison = &report.comparisons["candidate"];
        assert_eq!(comparison.mean_delta, -1.0);
        assert_eq!(comparison.delta_ci95.lower, -1.0);
        assert_eq!(comparison.delta_ci95.upper, -1.0);
        assert_eq!(
            (comparison.wins, comparison.ties, comparison.losses),
            (1, 0, 0)
        );
    }

    #[test]
    fn serde_preserves_sorted_report_shape() {
        let rows = vec![
            row("z-baseline", "q1", "single", 0.0),
            row("a-candidate", "q1", "single", 1.0),
        ];
        let report = build_scalar_report(
            "score",
            "z-baseline",
            MetricDirection::HigherIsBetter,
            &rows,
        )
        .unwrap();
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["denominator"], 1);
        assert_eq!(json["methods"]["a-candidate"]["overall"]["mean"], 1.0);
        assert_eq!(json["comparisons"]["a-candidate"]["wins"], 1);
    }
}
