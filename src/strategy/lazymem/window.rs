//! Scope-safe history expansion, transitive span merging, and window splitting.

use super::types::{
    PAPER_CONTEXT_RADIUS, PAPER_WINDOW_LENGTH, PAPER_WINDOW_STRIDE, ScopeKey, StoredMessage,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Controls when expanded anchor spans become one maximal evidence span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SpanMergeRule {
    /// The paper's equation: merge when adjacent anchors differ by `<= 2w`.
    PaperOverlap,
    /// Semafuse's optional extension: also merge directly touching spans,
    /// corresponding to an anchor gap of `<= 2w + 1`.
    RepositoryContiguous,
}

/// Query-time history-window parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HistoryWindowConfig {
    pub radius: usize,
    pub merge_rule: SpanMergeRule,
    pub max_window_messages: usize,
    pub stride: usize,
}

impl Default for HistoryWindowConfig {
    fn default() -> Self {
        Self {
            radius: PAPER_CONTEXT_RADIUS,
            merge_rule: SpanMergeRule::PaperOverlap,
            max_window_messages: PAPER_WINDOW_LENGTH,
            stride: PAPER_WINDOW_STRIDE,
        }
    }
}

/// All messages in one scope. Input order is not trusted; session ordinals are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ScopeTimeline {
    pub scope: ScopeKey,
    pub messages: Vec<StoredMessage>,
}

/// Inputs to paper-style history reconstruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HistoryWindowInput {
    pub timelines: Vec<ScopeTimeline>,
    /// Message IDs in reranker order. Window construction itself reorders them
    /// chronologically within each scope.
    pub anchor_ids: Vec<String>,
    pub config: HistoryWindowConfig,
}

/// A maximal, unsplit context span after transitive anchor merging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvidenceSpan {
    pub span_order: usize,
    pub scope: ScopeKey,
    pub start_session_ordinal: u64,
    pub end_session_ordinal: u64,
    pub core_start_session_ordinal: u64,
    pub core_end_session_ordinal: u64,
    pub anchor_ids: Vec<String>,
    pub messages: Vec<StoredMessage>,
}

/// One bounded model input window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvidenceWindow {
    pub window_order: usize,
    pub span_order: usize,
    pub scope: ScopeKey,
    pub anchor_ids: Vec<String>,
    /// Every dense-span message in this slice, including non-retrieved messages
    /// between the first and last anchor.
    pub core_message_ids: Vec<String>,
    /// One-based cross-encoder retrieval ranks for directly retrieved messages.
    pub retrieval_ranks: BTreeMap<String, usize>,
    pub messages: Vec<StoredMessage>,
}

/// Per-anchor expansion recorded in original reranker order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AnchorExpansionTrace {
    pub retrieval_order: usize,
    pub message_id: String,
    pub scope: ScopeKey,
    pub session_position: usize,
    pub expanded_message_ids: Vec<String>,
}

/// Trace of one maximal span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SpanTrace {
    pub span_order: usize,
    pub scope: ScopeKey,
    pub anchor_ids: Vec<String>,
    pub message_ids: Vec<String>,
}

/// Trace of one split model window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WindowTrace {
    pub window_order: usize,
    pub span_order: usize,
    pub scope: ScopeKey,
    pub message_ids: Vec<String>,
}

/// Fully typed audit data for the history-windowing stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HistoryWindowTrace {
    pub config: HistoryWindowConfig,
    pub anchors: Vec<AnchorExpansionTrace>,
    pub spans: Vec<SpanTrace>,
    pub windows: Vec<WindowTrace>,
}

/// Typed output of history reconstruction and bounded window splitting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HistoryWindowOutput {
    pub spans: Vec<EvidenceSpan>,
    pub windows: Vec<EvidenceWindow>,
    pub trace: HistoryWindowTrace,
}

#[derive(Debug, Clone)]
struct AnchorLocation {
    timeline_index: usize,
    position: usize,
}

#[derive(Debug)]
struct PendingWindow {
    span_order: usize,
    scope: ScopeKey,
    anchor_ids: Vec<String>,
    core_message_ids: Vec<String>,
    retrieval_ranks: BTreeMap<String, usize>,
    messages: Vec<StoredMessage>,
}

/// Restore local context around reranked anchors and produce bounded windows.
///
/// Every validation and sort is deterministic. In particular, scopes never
/// share context, transitive merging is based on adjacent anchor positions,
/// and model-window order has a total tie-break independent of hash iteration.
pub(super) fn construct_history_windows(input: &HistoryWindowInput) -> Result<HistoryWindowOutput> {
    validate_config(input.config)?;
    let (timelines, locations) = normalize_timelines(&input.timelines)?;
    validate_anchors(&input.anchor_ids, &locations)?;

    let anchor_trace = input
        .anchor_ids
        .iter()
        .enumerate()
        .map(|(retrieval_order, message_id)| {
            let location = &locations[message_id.as_str()];
            let timeline = &timelines[location.timeline_index];
            let start = location.position.saturating_sub(input.config.radius);
            let end = location
                .position
                .saturating_add(input.config.radius)
                .saturating_add(1)
                .min(timeline.messages.len());
            AnchorExpansionTrace {
                retrieval_order,
                message_id: message_id.clone(),
                scope: timeline.scope.clone(),
                session_position: location.position,
                expanded_message_ids: timeline.messages[start..end]
                    .iter()
                    .map(|message| message.message_id.clone())
                    .collect(),
            }
        })
        .collect::<Vec<_>>();

    let merge_gap = merge_gap(input.config)?;
    let anchor_set = input
        .anchor_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let retrieval_ranks = input
        .anchor_ids
        .iter()
        .enumerate()
        .map(|(index, message_id)| (message_id.as_str(), index + 1))
        .collect::<HashMap<_, _>>();
    let mut spans = Vec::new();
    for timeline in &timelines {
        let anchors = timeline
            .messages
            .iter()
            .enumerate()
            .filter(|(_, message)| anchor_set.contains(message.message_id.as_str()))
            .map(|(position, message)| (position, message.message_id.clone()))
            .collect::<Vec<_>>();
        if anchors.is_empty() {
            continue;
        }

        let mut group_start = 0;
        for index in 1..=anchors.len() {
            let starts_new_group = index == anchors.len()
                || anchors[index].0.saturating_sub(anchors[index - 1].0) > merge_gap;
            if starts_new_group {
                spans.push(make_span(
                    timeline,
                    &anchors[group_start..index],
                    input.config.radius,
                ));
                group_start = index;
            }
        }
    }

    spans.sort_by(|left, right| {
        first_global(&left.messages)
            .cmp(&first_global(&right.messages))
            .then_with(|| left.scope.stable_key().cmp(&right.scope.stable_key()))
            .then_with(|| left.start_session_ordinal.cmp(&right.start_session_ordinal))
    });
    for (span_order, span) in spans.iter_mut().enumerate() {
        span.span_order = span_order;
    }

    let mut pending_windows = Vec::new();
    for span in &spans {
        let mut start: usize = 0;
        loop {
            let end = start
                .saturating_add(input.config.max_window_messages)
                .min(span.messages.len());
            let messages = span.messages[start..end].to_vec();
            let message_ids = messages
                .iter()
                .map(|message| message.message_id.as_str())
                .collect::<HashSet<_>>();
            let core_message_ids = messages
                .iter()
                .filter(|message| {
                    (span.core_start_session_ordinal..=span.core_end_session_ordinal)
                        .contains(&message.session_ordinal)
                })
                .map(|message| message.message_id.clone())
                .collect::<Vec<_>>();
            if core_message_ids.is_empty() {
                if end == span.messages.len() {
                    break;
                }
                start += input.config.stride;
                continue;
            }
            let window_anchor_ids = span
                .anchor_ids
                .iter()
                .filter(|message_id| message_ids.contains(message_id.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            pending_windows.push(PendingWindow {
                span_order: span.span_order,
                scope: span.scope.clone(),
                retrieval_ranks: window_anchor_ids
                    .iter()
                    .map(|message_id| (message_id.clone(), retrieval_ranks[message_id.as_str()]))
                    .collect(),
                anchor_ids: window_anchor_ids,
                core_message_ids,
                messages,
            });
            if end == span.messages.len() {
                break;
            }
            start += input.config.stride;
        }
    }
    pending_windows.sort_by(|left, right| {
        first_global(&left.messages)
            .cmp(&first_global(&right.messages))
            .then_with(|| left.scope.stable_key().cmp(&right.scope.stable_key()))
            .then_with(|| {
                left.messages[0]
                    .session_ordinal
                    .cmp(&right.messages[0].session_ordinal)
            })
            .then_with(|| left.span_order.cmp(&right.span_order))
    });
    let windows = pending_windows
        .into_iter()
        .enumerate()
        .map(|(window_order, window)| EvidenceWindow {
            window_order,
            span_order: window.span_order,
            scope: window.scope,
            anchor_ids: window.anchor_ids,
            core_message_ids: window.core_message_ids,
            retrieval_ranks: window.retrieval_ranks,
            messages: window.messages,
        })
        .collect::<Vec<_>>();

    let span_trace = spans
        .iter()
        .map(|span| SpanTrace {
            span_order: span.span_order,
            scope: span.scope.clone(),
            anchor_ids: span.anchor_ids.clone(),
            message_ids: span
                .messages
                .iter()
                .map(|message| message.message_id.clone())
                .collect(),
        })
        .collect();
    let window_trace = windows
        .iter()
        .map(|window| WindowTrace {
            window_order: window.window_order,
            span_order: window.span_order,
            scope: window.scope.clone(),
            message_ids: window
                .messages
                .iter()
                .map(|message| message.message_id.clone())
                .collect(),
        })
        .collect();

    Ok(HistoryWindowOutput {
        spans,
        windows,
        trace: HistoryWindowTrace {
            config: input.config,
            anchors: anchor_trace,
            spans: span_trace,
            windows: window_trace,
        },
    })
}

fn validate_config(config: HistoryWindowConfig) -> Result<()> {
    ensure!(
        config.max_window_messages > 0,
        "max_window_messages must be positive"
    );
    ensure!(config.stride > 0, "window stride must be positive");
    ensure!(
        config.stride <= config.max_window_messages,
        "window stride must not exceed max_window_messages"
    );
    merge_gap(config)?;
    Ok(())
}

fn merge_gap(config: HistoryWindowConfig) -> Result<usize> {
    let paper_gap = config
        .radius
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("context radius is too large"))?;
    match config.merge_rule {
        SpanMergeRule::PaperOverlap => Ok(paper_gap),
        SpanMergeRule::RepositoryContiguous => paper_gap
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("context radius is too large")),
    }
}

fn normalize_timelines(
    timelines: &[ScopeTimeline],
) -> Result<(Vec<ScopeTimeline>, HashMap<String, AnchorLocation>)> {
    let mut normalized = timelines.to_vec();
    normalized.sort_by(|left, right| left.scope.stable_key().cmp(&right.scope.stable_key()));

    let mut seen_scopes = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut seen_global_ordinals = HashSet::new();
    for timeline in &mut normalized {
        ensure!(
            seen_scopes.insert(timeline.scope.clone()),
            "duplicate scope timeline: {}/{:?}",
            timeline.scope.session_id,
            timeline.scope.repository
        );
        timeline.messages.sort_by(|left, right| {
            left.session_ordinal
                .cmp(&right.session_ordinal)
                .then_with(|| left.global_ordinal.cmp(&right.global_ordinal))
                .then_with(|| left.message_id.cmp(&right.message_id))
        });
        let mut session_ordinals = HashSet::new();
        for message in &timeline.messages {
            ensure!(
                !message.message_id.trim().is_empty(),
                "stored message has an empty id"
            );
            ensure!(
                !message.content.trim().is_empty(),
                "stored message {} has empty content",
                message.message_id
            );
            ensure!(
                message.scope() == timeline.scope,
                "stored message {} does not belong to its scope timeline",
                message.message_id
            );
            ensure!(
                seen_ids.insert(message.message_id.clone()),
                "duplicate stored message id: {}",
                message.message_id
            );
            ensure!(
                seen_global_ordinals.insert(message.global_ordinal),
                "duplicate global ordinal: {}",
                message.global_ordinal
            );
            ensure!(
                session_ordinals.insert(message.session_ordinal),
                "duplicate session ordinal {} in session {}",
                message.session_ordinal,
                timeline.scope.session_id
            );
        }
    }

    // Build after normalization so positions refer to the canonical order.
    let mut locations = HashMap::new();
    for (timeline_index, timeline) in normalized.iter().enumerate() {
        for (position, message) in timeline.messages.iter().enumerate() {
            locations.insert(
                message.message_id.clone(),
                AnchorLocation {
                    timeline_index,
                    position,
                },
            );
        }
    }
    Ok((normalized, locations))
}

fn validate_anchors(
    anchor_ids: &[String],
    locations: &HashMap<String, AnchorLocation>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for message_id in anchor_ids {
        ensure!(
            !message_id.trim().is_empty(),
            "anchor contains an empty message id"
        );
        ensure!(
            seen.insert(message_id.as_str()),
            "duplicate anchor id: {message_id}"
        );
        ensure!(
            locations.contains_key(message_id.as_str()),
            "anchor message is absent from scoped timelines: {message_id}"
        );
    }
    Ok(())
}

fn make_span(timeline: &ScopeTimeline, anchors: &[(usize, String)], radius: usize) -> EvidenceSpan {
    let start = anchors[0].0.saturating_sub(radius);
    let end = anchors[anchors.len() - 1]
        .0
        .saturating_add(radius)
        .saturating_add(1)
        .min(timeline.messages.len());
    let messages = timeline.messages[start..end].to_vec();
    EvidenceSpan {
        span_order: 0,
        scope: timeline.scope.clone(),
        start_session_ordinal: messages[0].session_ordinal,
        end_session_ordinal: messages[messages.len() - 1].session_ordinal,
        core_start_session_ordinal: timeline.messages[anchors[0].0].session_ordinal,
        core_end_session_ordinal: timeline.messages[anchors[anchors.len() - 1].0].session_ordinal,
        anchor_ids: anchors
            .iter()
            .map(|(_, message_id)| message_id.clone())
            .collect(),
        messages,
    }
}

fn first_global(messages: &[StoredMessage]) -> u64 {
    messages
        .first()
        .expect("evidence spans and windows are never empty")
        .global_ordinal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::lazymem::types::test_support::{message, message_in_scope};

    fn timeline(count: usize) -> ScopeTimeline {
        let messages = (0..count)
            .map(|index| message(&format!("m{index}"), index as u64, index as u64))
            .collect::<Vec<_>>();
        ScopeTimeline {
            scope: messages[0].scope(),
            messages,
        }
    }

    fn input(timeline: ScopeTimeline, anchors: &[&str]) -> HistoryWindowInput {
        HistoryWindowInput {
            timelines: vec![timeline],
            anchor_ids: anchors.iter().map(|id| (*id).to_owned()).collect(),
            config: HistoryWindowConfig::default(),
        }
    }

    fn message_ids(messages: &[StoredMessage]) -> Vec<&str> {
        messages
            .iter()
            .map(|message| message.message_id.as_str())
            .collect()
    }

    #[test]
    fn defaults_match_the_paper_benchmark() {
        assert_eq!(
            HistoryWindowConfig::default(),
            HistoryWindowConfig {
                radius: 2,
                merge_rule: SpanMergeRule::PaperOverlap,
                max_window_messages: 8,
                stride: 7,
            }
        );
    }

    #[test]
    fn expansion_clips_cleanly_at_both_timeline_edges() {
        let output = construct_history_windows(&input(timeline(6), &["m0", "m5"])).unwrap();

        assert_eq!(
            output.trace.anchors[0].expanded_message_ids,
            ["m0", "m1", "m2"]
        );
        assert_eq!(
            output.trace.anchors[1].expanded_message_ids,
            ["m3", "m4", "m5"]
        );
        assert_eq!(output.spans.len(), 2);
    }

    #[test]
    fn paper_and_repository_merge_rules_differ_only_for_touching_spans() {
        let mut paper = input(timeline(9), &["m1", "m6"]);
        paper.config.radius = 2;
        let paper = construct_history_windows(&paper).unwrap();
        assert_eq!(paper.spans.len(), 2); // anchor gap 5 > 2w

        let mut repository = input(timeline(9), &["m1", "m6"]);
        repository.config.merge_rule = SpanMergeRule::RepositoryContiguous;
        let repository = construct_history_windows(&repository).unwrap();
        assert_eq!(repository.spans.len(), 1); // anchor gap 5 == 2w + 1
        assert_eq!(
            message_ids(&repository.spans[0].messages),
            ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
        );
    }

    #[test]
    fn overlapping_anchor_spans_merge_transitively() {
        let output = construct_history_windows(&input(timeline(13), &["m1", "m5", "m9"])).unwrap();

        // m1 and m9 are too far apart directly, but each adjacent gap is 2w.
        assert_eq!(output.spans.len(), 1);
        assert_eq!(output.spans[0].anchor_ids, ["m1", "m5", "m9"]);
        assert_eq!(
            message_ids(&output.spans[0].messages),
            (0..=11)
                .map(|index| format!("m{index}"))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn long_spans_split_into_length_eight_stride_seven_windows() {
        let output =
            construct_history_windows(&input(timeline(18), &["m2", "m6", "m10", "m14"])).unwrap();

        assert_eq!(output.spans.len(), 1);
        assert_eq!(output.spans[0].messages.len(), 17);
        assert_eq!(
            output
                .windows
                .iter()
                .map(|window| window.messages.len())
                .collect::<Vec<_>>(),
            [8, 8, 3]
        );
        assert_eq!(output.windows[0].messages[7].message_id, "m7");
        assert_eq!(output.windows[1].messages[0].message_id, "m7");
        assert_eq!(output.windows[1].messages[7].message_id, "m14");
        assert_eq!(output.windows[2].messages[0].message_id, "m14");
        assert_eq!(
            output.windows[0].core_message_ids,
            ["m2", "m3", "m4", "m5", "m6", "m7"]
        );
        assert_eq!(output.windows[2].core_message_ids, ["m14"]);
        assert_eq!(output.windows[0].retrieval_ranks["m2"], 1);
        assert_eq!(output.windows[1].retrieval_ranks["m14"], 4);
    }

    #[test]
    fn input_timeline_order_is_normalized_by_stable_session_ordinal() {
        let mut reversed = timeline(5);
        reversed.messages.reverse();
        let output = construct_history_windows(&input(reversed, &["m2"])).unwrap();
        assert_eq!(
            message_ids(&output.spans[0].messages),
            ["m0", "m1", "m2", "m3", "m4"]
        );
    }

    #[test]
    fn scopes_never_merge_and_window_order_uses_global_ordinal() {
        let later = (0..3)
            .map(|index| {
                message_in_scope(
                    &format!("a{index}"),
                    10 + index,
                    index,
                    "session-a",
                    Some("/repo/a"),
                )
            })
            .collect::<Vec<_>>();
        let earlier = (0..3)
            .map(|index| {
                message_in_scope(
                    &format!("b{index}"),
                    1 + index,
                    index,
                    "session-b",
                    Some("/repo/b"),
                )
            })
            .collect::<Vec<_>>();
        let output = construct_history_windows(&HistoryWindowInput {
            timelines: vec![
                ScopeTimeline {
                    scope: later[0].scope(),
                    messages: later,
                },
                ScopeTimeline {
                    scope: earlier[0].scope(),
                    messages: earlier,
                },
            ],
            anchor_ids: vec!["a1".into(), "b1".into()],
            config: HistoryWindowConfig::default(),
        })
        .unwrap();

        assert_eq!(output.spans.len(), 2);
        assert_eq!(output.windows[0].messages[0].message_id, "b0");
        assert_eq!(output.windows[1].messages[0].message_id, "a0");
    }

    #[test]
    fn invalid_window_parameters_are_rejected() {
        let mut zero_stride = input(timeline(3), &["m1"]);
        zero_stride.config.stride = 0;
        assert!(
            construct_history_windows(&zero_stride)
                .unwrap_err()
                .to_string()
                .contains("stride must be positive")
        );

        let mut gap = input(timeline(3), &["m1"]);
        gap.config.max_window_messages = 4;
        gap.config.stride = 5;
        assert!(
            construct_history_windows(&gap)
                .unwrap_err()
                .to_string()
                .contains("must not exceed")
        );
    }

    #[test]
    fn missing_and_duplicate_anchors_are_rejected() {
        let missing = input(timeline(3), &["absent"]);
        assert!(
            construct_history_windows(&missing)
                .unwrap_err()
                .to_string()
                .contains("absent from scoped timelines")
        );

        let duplicate = input(timeline(3), &["m1", "m1"]);
        assert!(
            construct_history_windows(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate anchor")
        );
    }

    #[test]
    fn unstable_or_ambiguous_message_identity_is_rejected() {
        let mut duplicate_global = timeline(3);
        duplicate_global.messages[2].global_ordinal = duplicate_global.messages[1].global_ordinal;
        assert!(
            construct_history_windows(&input(duplicate_global, &["m1"]))
                .unwrap_err()
                .to_string()
                .contains("duplicate global ordinal")
        );

        let mut duplicate_session = timeline(3);
        duplicate_session.messages[2].session_ordinal =
            duplicate_session.messages[1].session_ordinal;
        assert!(
            construct_history_windows(&input(duplicate_session, &["m1"]))
                .unwrap_err()
                .to_string()
                .contains("duplicate session ordinal")
        );

        let mut wrong_scope = timeline(3);
        wrong_scope.messages[1].source.session_id = "different".into();
        assert!(
            construct_history_windows(&input(wrong_scope, &["m1"]))
                .unwrap_err()
                .to_string()
                .contains("does not belong")
        );
    }

    #[test]
    fn empty_message_content_is_not_sent_to_the_processing_model() {
        let mut history = timeline(3);
        history.messages[1].content = "  ".into();
        assert!(
            construct_history_windows(&input(history, &["m1"]))
                .unwrap_err()
                .to_string()
                .contains("empty content")
        );
    }
}
