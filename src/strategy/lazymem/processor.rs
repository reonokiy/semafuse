//! Strict model-output validation and deterministic cross-window resolution.

use super::{
    types::{ScopeKey, StoredMessage},
    window::EvidenceWindow,
};
use crate::inference::{MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS, MAX_PROCESSOR_REASON_CHARS};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Per-message operation emitted by the memory-processing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub(super) enum DecisionOp {
    Keep,
    Drop,
}

/// Exact three-field JSON schema required for every message in a window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelDecision {
    pub op: DecisionOp,
    pub compressed_content: String,
    pub reason: String,
}

/// A validated model decision bound positionally to its source message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WindowDecision {
    pub window_order: usize,
    pub message: StoredMessage,
    pub op: DecisionOp,
    pub compressed_content: String,
    pub reason: String,
}

/// Typed output for one independently processed evidence window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessedWindow {
    pub window_order: usize,
    pub scope: ScopeKey,
    pub decisions: Vec<WindowDecision>,
}

/// Raw response associated with a deterministic model-window order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WindowModelResponse {
    pub window_order: usize,
    pub json: String,
}

/// Inputs to parallel-window result resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessingInput {
    pub windows: Vec<EvidenceWindow>,
    pub responses: Vec<WindowModelResponse>,
}

/// One query-conditioned message that survives construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConstructedMemoryMessage {
    pub message: StoredMessage,
    pub compressed_content: String,
    pub reason: String,
    pub selected_window_order: usize,
}

/// A single window's vote for one source message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecisionCandidateTrace {
    pub window_order: usize,
    pub op: DecisionOp,
    pub compressed_content: String,
    pub reason: String,
}

/// All overlapping-window decisions and the deterministic winner for a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MessageResolutionTrace {
    pub message_id: String,
    pub candidates: Vec<DecisionCandidateTrace>,
    pub selected_window_order: usize,
    pub selected_op: DecisionOp,
}

/// Fully typed audit data for model-output validation and deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessingTrace {
    pub windows: Vec<ProcessedWindow>,
    pub message_resolutions: Vec<MessageResolutionTrace>,
    pub final_message_order: Vec<String>,
}

/// Typed output of query-conditioned memory construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ProcessingOutput {
    /// Kept messages in stable global chronological order.
    pub messages: Vec<ConstructedMemoryMessage>,
    pub trace: ProcessingTrace,
}

/// Parse and validate one model response against a specific evidence window.
///
/// The JSON array is positional by design, matching the paper schema: it must
/// contain exactly one three-field decision for every message in chronological
/// window order. A single leading Qwen thinking block is tolerated for the
/// public checkpoint profile; markdown fences, prose, nested/multiple thinking
/// blocks, unknown fields, invalid operations, or trailing JSON are rejected.
pub(super) fn process_window_response(
    window: &EvidenceWindow,
    response_json: &str,
) -> Result<ProcessedWindow> {
    validate_window(window)?;
    let payload = strip_optional_thinking_block(response_json)?;
    let model_decisions = serde_json::from_str::<Vec<ModelDecision>>(payload)
        .with_context(|| format!("parse model response for window {}", window.window_order))?;
    ensure!(
        model_decisions.len() == window.messages.len(),
        "window {} expected {} decisions but received {}",
        window.window_order,
        window.messages.len(),
        model_decisions.len()
    );

    let decisions = window
        .messages
        .iter()
        .cloned()
        .zip(model_decisions)
        .map(|(message, decision)| {
            ensure!(
                !decision.reason.trim().is_empty(),
                "window {} decision for {} has an empty reason",
                window.window_order,
                message.message_id
            );
            ensure!(
                decision.reason.chars().count() <= MAX_PROCESSOR_REASON_CHARS,
                "window {} decision for {} has a reason longer than {} characters",
                window.window_order,
                message.message_id,
                MAX_PROCESSOR_REASON_CHARS
            );
            match decision.op {
                DecisionOp::Keep => {
                    ensure!(
                        !decision.compressed_content.trim().is_empty(),
                        "window {} KEEP decision for {} has empty compressed_content",
                        window.window_order,
                        message.message_id
                    );
                    ensure!(
                        decision.compressed_content.chars().count()
                            <= MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS,
                        "window {} KEEP decision for {} has compressed_content longer than {} characters",
                        window.window_order,
                        message.message_id,
                        MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS
                    );
                }
                DecisionOp::Drop => ensure!(
                    decision.compressed_content.is_empty(),
                    "window {} DROP decision for {} must have empty compressed_content",
                    window.window_order,
                    message.message_id
                ),
            }
            Ok(WindowDecision {
                window_order: window.window_order,
                message,
                op: decision.op,
                compressed_content: decision.compressed_content,
                reason: decision.reason,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(ProcessedWindow {
        window_order: window.window_order,
        scope: window.scope.clone(),
        decisions,
    })
}

fn strip_optional_thinking_block(raw: &str) -> Result<&str> {
    let value = raw.trim();
    if !value.starts_with("<think>") {
        ensure!(
            !value.contains("<think>") && !value.contains("</think>"),
            "model response contains a misplaced thinking tag"
        );
        return Ok(value);
    }
    let after_open = &value["<think>".len()..];
    ensure!(
        !after_open.contains("<think>"),
        "model response contains nested or repeated thinking blocks"
    );
    let close = after_open
        .find("</think>")
        .context("model response has an unclosed thinking block")?;
    let payload = after_open[close + "</think>".len()..].trim();
    ensure!(
        !payload.is_empty(),
        "model response omitted JSON after the thinking block"
    );
    ensure!(
        !payload.contains("<think>") && !payload.contains("</think>"),
        "model response contains multiple or misplaced thinking blocks"
    );
    Ok(payload)
}

/// Resolve independently processed, possibly overlapping model windows.
///
/// For a repeated source message, `KEEP` always wins over `DROP`; when actions
/// tie, the lower (earlier) `window_order` wins. The surviving messages are then
/// sorted by `global_ordinal`, not model completion order or native timestamp.
pub(super) fn combine_window_responses(input: &ProcessingInput) -> Result<ProcessingOutput> {
    let mut window_orders = HashSet::new();
    let mut canonical_messages = HashMap::<&str, &StoredMessage>::new();
    let mut global_ordinals = HashMap::<u64, &str>::new();
    for window in &input.windows {
        ensure!(
            window_orders.insert(window.window_order),
            "duplicate evidence window order: {}",
            window.window_order
        );
        validate_window(window)?;
        for message in &window.messages {
            if let Some(existing) = canonical_messages.insert(&message.message_id, message) {
                ensure!(
                    existing == message,
                    "overlapping windows disagree on stored message {}",
                    message.message_id
                );
            }
            if let Some(existing_id) =
                global_ordinals.insert(message.global_ordinal, &message.message_id)
            {
                ensure!(
                    existing_id == message.message_id,
                    "global ordinal {} belongs to both {} and {}",
                    message.global_ordinal,
                    existing_id,
                    message.message_id
                );
            }
        }
    }

    let mut response_by_window = HashMap::new();
    for response in &input.responses {
        ensure!(
            window_orders.contains(&response.window_order),
            "model response refers to unknown window order: {}",
            response.window_order
        );
        ensure!(
            response_by_window
                .insert(response.window_order, response.json.as_str())
                .is_none(),
            "duplicate model response for window {}",
            response.window_order
        );
    }
    ensure!(
        response_by_window.len() == input.windows.len(),
        "expected one model response for every window (expected {}, got {})",
        input.windows.len(),
        response_by_window.len()
    );

    let mut ordered_windows = input.windows.iter().collect::<Vec<_>>();
    ordered_windows.sort_by_key(|window| window.window_order);
    let processed_windows = ordered_windows
        .into_iter()
        .map(|window| process_window_response(window, response_by_window[&window.window_order]))
        .collect::<Result<Vec<_>>>()?;

    let mut candidates = HashMap::<String, Vec<WindowDecision>>::new();
    for window in &processed_windows {
        for decision in &window.decisions {
            candidates
                .entry(decision.message.message_id.clone())
                .or_default()
                .push(decision.clone());
        }
    }

    let mut resolutions = candidates
        .into_iter()
        .map(|(message_id, mut message_candidates)| {
            message_candidates.sort_by_key(|decision| decision.window_order);
            let selected = message_candidates
                .iter()
                .min_by_key(|decision| {
                    let operation_priority = match decision.op {
                        DecisionOp::Keep => 0,
                        DecisionOp::Drop => 1,
                    };
                    (operation_priority, decision.window_order)
                })
                .expect("every candidate group is non-empty")
                .clone();
            let trace = MessageResolutionTrace {
                message_id,
                candidates: message_candidates
                    .iter()
                    .map(|decision| DecisionCandidateTrace {
                        window_order: decision.window_order,
                        op: decision.op,
                        compressed_content: decision.compressed_content.clone(),
                        reason: decision.reason.clone(),
                    })
                    .collect(),
                selected_window_order: selected.window_order,
                selected_op: selected.op,
            };
            (selected, trace)
        })
        .collect::<Vec<_>>();
    resolutions.sort_by(|(left, _), (right, _)| {
        left.message
            .global_ordinal
            .cmp(&right.message.global_ordinal)
            .then_with(|| left.message.message_id.cmp(&right.message.message_id))
    });

    let message_resolutions = resolutions
        .iter()
        .map(|(_, trace)| trace.clone())
        .collect::<Vec<_>>();
    let messages = resolutions
        .into_iter()
        .filter(|(decision, _)| decision.op == DecisionOp::Keep)
        .map(|(decision, _)| ConstructedMemoryMessage {
            message: decision.message,
            compressed_content: decision.compressed_content,
            reason: decision.reason,
            selected_window_order: decision.window_order,
        })
        .collect::<Vec<_>>();
    let final_message_order = messages
        .iter()
        .map(|message| message.message.message_id.clone())
        .collect();

    Ok(ProcessingOutput {
        messages,
        trace: ProcessingTrace {
            windows: processed_windows,
            message_resolutions,
            final_message_order,
        },
    })
}

fn validate_window(window: &EvidenceWindow) -> Result<()> {
    ensure!(
        !window.messages.is_empty(),
        "evidence window {} is empty",
        window.window_order
    );
    let mut ids = HashSet::new();
    let core_ids = window
        .core_message_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    ensure!(
        !core_ids.is_empty() && core_ids.len() == window.core_message_ids.len(),
        "evidence window {} has empty or duplicate core messages",
        window.window_order
    );
    let anchor_ids = window
        .anchor_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    ensure!(
        anchor_ids.len() == window.anchor_ids.len()
            && window.retrieval_ranks.len() == window.anchor_ids.len(),
        "evidence window {} has inconsistent anchors and retrieval ranks",
        window.window_order
    );
    let mut previous_session_ordinal = None;
    for message in &window.messages {
        ensure!(
            message.scope() == window.scope,
            "window {} contains out-of-scope message {}",
            window.window_order,
            message.message_id
        );
        ensure!(
            !message.message_id.trim().is_empty(),
            "window {} contains an empty message id",
            window.window_order
        );
        ensure!(
            !message.content.trim().is_empty(),
            "window {} contains empty message content for {}",
            window.window_order,
            message.message_id
        );
        ensure!(
            ids.insert(message.message_id.as_str()),
            "window {} repeats message {}",
            window.window_order,
            message.message_id
        );
        if let Some(previous) = previous_session_ordinal {
            ensure!(
                message.session_ordinal > previous,
                "window {} messages are not in session chronology",
                window.window_order
            );
        }
        previous_session_ordinal = Some(message.session_ordinal);
    }
    ensure!(
        core_ids.iter().all(|message_id| ids.contains(message_id)),
        "window {} core contains a message outside the window",
        window.window_order
    );
    ensure!(
        anchor_ids
            .iter()
            .all(|message_id| core_ids.contains(message_id)),
        "window {} anchor is outside the dense core",
        window.window_order
    );
    for message_id in &window.anchor_ids {
        ensure!(
            window.retrieval_ranks.get(message_id).copied().unwrap_or(0) > 0,
            "window {} has an invalid retrieval rank for {}",
            window.window_order,
            message_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::lazymem::types::test_support::message;

    fn window(order: usize, messages: Vec<StoredMessage>) -> EvidenceWindow {
        EvidenceWindow {
            window_order: order,
            span_order: 0,
            scope: messages[0].scope(),
            anchor_ids: vec![messages[0].message_id.clone()],
            core_message_ids: messages
                .iter()
                .map(|message| message.message_id.clone())
                .collect(),
            retrieval_ranks: std::iter::once((messages[0].message_id.clone(), 1)).collect(),
            messages,
        }
    }

    fn response(order: usize, json: &str) -> WindowModelResponse {
        WindowModelResponse {
            window_order: order,
            json: json.to_owned(),
        }
    }

    #[test]
    fn strict_array_is_mapped_positionally_to_every_message() {
        let evidence = window(3, vec![message("a", 10, 0), message("b", 11, 1)]);
        let processed = process_window_response(
            &evidence,
            r#"[
                {"op":"KEEP","compressed_content":"useful a","reason":"matches query"},
                {"op":"DROP","compressed_content":"","reason":"unrelated"}
            ]"#,
        )
        .unwrap();

        assert_eq!(processed.window_order, 3);
        assert_eq!(processed.decisions[0].message.message_id, "a");
        assert_eq!(processed.decisions[0].op, DecisionOp::Keep);
        assert_eq!(processed.decisions[1].message.message_id, "b");
        assert_eq!(processed.decisions[1].op, DecisionOp::Drop);
    }

    #[test]
    fn parser_rejects_prose_fences_unknown_fields_and_invalid_ops() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let invalid = [
            r#"```json [{"op":"DROP","compressed_content":"","reason":"no"}] ```"#,
            r#"[{"op":"DROP","compressed_content":"","reason":"no","message_id":"a"}]"#,
            r#"[{"op":"SKIP","compressed_content":"","reason":"no"}]"#,
            r#"[{"op":"DROP","compressed_content":"","reason":"no"}] trailing"#,
        ];
        for json in invalid {
            assert!(
                process_window_response(&evidence, json).is_err(),
                "accepted {json}"
            );
        }
    }

    #[test]
    fn parser_accepts_one_leading_thinking_block_and_rejects_tag_smuggling() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let processed = process_window_response(
            &evidence,
            "<think>check local relevance</think>\n[{\"op\":\"DROP\",\"compressed_content\":\"\",\"reason\":\"irrelevant\"}]",
        )
        .unwrap();
        assert_eq!(processed.decisions[0].op, DecisionOp::Drop);

        for invalid in [
            "prefix <think>x</think>[]",
            "<think><think>x</think></think>[]",
            "<think>x</think><think>y</think>[]",
            "<think>x[]",
        ] {
            assert!(process_window_response(&evidence, invalid).is_err());
        }
    }

    #[test]
    fn parser_requires_exactly_one_decision_per_message() {
        let evidence = window(0, vec![message("a", 0, 0), message("b", 1, 1)]);
        let error = process_window_response(
            &evidence,
            r#"[{"op":"DROP","compressed_content":"","reason":"no"}]"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("expected 2 decisions"));
    }

    #[test]
    fn keep_drop_and_reason_invariants_are_enforced() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let cases = [
            (
                r#"[{"op":"DROP","compressed_content":"not empty","reason":"no"}]"#,
                "DROP",
            ),
            (
                r#"[{"op":"KEEP","compressed_content":"  ","reason":"yes"}]"#,
                "KEEP",
            ),
            (
                r#"[{"op":"DROP","compressed_content":"","reason":"  "}]"#,
                "empty reason",
            ),
        ];
        for (json, expected) in cases {
            let error = process_window_response(&evidence, json).unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[test]
    fn reason_limit_counts_unicode_characters() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let accepted = serde_json::to_string(&vec![serde_json::json!({
            "op": "DROP",
            "compressed_content": "",
            "reason": "界".repeat(MAX_PROCESSOR_REASON_CHARS),
        })])
        .unwrap();
        process_window_response(&evidence, &accepted).unwrap();

        let rejected = serde_json::to_string(&vec![serde_json::json!({
            "op": "DROP",
            "compressed_content": "",
            "reason": "界".repeat(MAX_PROCESSOR_REASON_CHARS + 1),
        })])
        .unwrap();
        assert!(process_window_response(&evidence, &rejected).is_err());
    }

    #[test]
    fn compressed_content_limit_counts_unicode_characters() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let accepted = serde_json::to_string(&vec![serde_json::json!({
            "op": "KEEP",
            "compressed_content": "界".repeat(MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS),
            "reason": "relevant",
        })])
        .unwrap();
        process_window_response(&evidence, &accepted).unwrap();

        let rejected = serde_json::to_string(&vec![serde_json::json!({
            "op": "KEEP",
            "compressed_content": "界".repeat(MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS + 1),
            "reason": "relevant",
        })])
        .unwrap();
        assert!(process_window_response(&evidence, &rejected).is_err());
    }

    #[test]
    fn keep_wins_across_windows_even_when_it_arrives_later() {
        let first = window(0, vec![message("a", 0, 0), message("b", 1, 1)]);
        let second = window(1, vec![message("b", 1, 1), message("c", 2, 2)]);
        let output = combine_window_responses(&ProcessingInput {
            windows: vec![first, second],
            responses: vec![
                response(
                    0,
                    r#"[
                        {"op":"DROP","compressed_content":"","reason":"first drops a"},
                        {"op":"DROP","compressed_content":"","reason":"first drops b"}
                    ]"#,
                ),
                response(
                    1,
                    r#"[
                        {"op":"KEEP","compressed_content":"kept b","reason":"second sees context"},
                        {"op":"DROP","compressed_content":"","reason":"drops c"}
                    ]"#,
                ),
            ],
        })
        .unwrap();

        assert_eq!(output.trace.final_message_order, ["b"]);
        assert_eq!(output.messages[0].compressed_content, "kept b");
        assert_eq!(output.messages[0].selected_window_order, 1);
        let b = output
            .trace
            .message_resolutions
            .iter()
            .find(|trace| trace.message_id == "b")
            .unwrap();
        assert_eq!(b.candidates.len(), 2);
        assert_eq!(b.selected_op, DecisionOp::Keep);
    }

    #[test]
    fn equal_actions_choose_the_earliest_window() {
        let first = window(4, vec![message("a", 0, 0)]);
        let second = window(9, vec![message("a", 0, 0)]);
        let output = combine_window_responses(&ProcessingInput {
            // Deliberately reverse both inputs: window_order, not arrival order,
            // is the tie-break.
            windows: vec![second, first],
            responses: vec![
                response(
                    9,
                    r#"[{"op":"KEEP","compressed_content":"later","reason":"later reason"}]"#,
                ),
                response(
                    4,
                    r#"[{"op":"KEEP","compressed_content":"earlier","reason":"earlier reason"}]"#,
                ),
            ],
        })
        .unwrap();

        assert_eq!(output.messages[0].compressed_content, "earlier");
        assert_eq!(output.messages[0].selected_window_order, 4);
    }

    #[test]
    fn final_memory_uses_global_chronology_and_preserves_source_metadata() {
        let late = message("late", 20, 0);
        let mut early = message("early", 10, 1);
        // Session order and global order can differ after multi-source import;
        // the final memory contract explicitly uses the global ordinal.
        early.role = crate::model::MessageRole::Assistant;
        let evidence = window(0, vec![late, early.clone()]);
        let output = combine_window_responses(&ProcessingInput {
            windows: vec![evidence],
            responses: vec![response(
                0,
                r#"[
                    {"op":"KEEP","compressed_content":"late text","reason":"useful"},
                    {"op":"KEEP","compressed_content":"early text","reason":"useful"}
                ]"#,
            )],
        })
        .unwrap();

        assert_eq!(output.trace.final_message_order, ["early", "late"]);
        assert_eq!(
            output.messages[0].message.role,
            crate::model::MessageRole::Assistant
        );
        assert_eq!(output.messages[0].message.source, early.source);
    }

    #[test]
    fn response_set_must_match_window_set_exactly() {
        let evidence = window(2, vec![message("a", 0, 0)]);
        let missing = combine_window_responses(&ProcessingInput {
            windows: vec![evidence.clone()],
            responses: Vec::new(),
        })
        .unwrap_err();
        assert!(missing.to_string().contains("one model response"));

        let duplicate = combine_window_responses(&ProcessingInput {
            windows: vec![evidence.clone()],
            responses: vec![
                response(2, r#"[{"op":"DROP","compressed_content":"","reason":"x"}]"#),
                response(2, r#"[{"op":"DROP","compressed_content":"","reason":"x"}]"#),
            ],
        })
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate model response"));

        let unknown = combine_window_responses(&ProcessingInput {
            windows: vec![evidence],
            responses: vec![response(3, "[]")],
        })
        .unwrap_err();
        assert!(unknown.to_string().contains("unknown window"));
    }

    #[test]
    fn overlapping_windows_must_agree_on_the_stored_message() {
        let first = window(0, vec![message("a", 0, 0)]);
        let mut changed = message("a", 0, 0);
        changed.content = "different source content".into();
        let second = window(1, vec![changed]);
        let error = combine_window_responses(&ProcessingInput {
            windows: vec![first, second],
            responses: Vec::new(),
        })
        .unwrap_err();
        assert!(error.to_string().contains("disagree on stored message"));
    }

    #[test]
    fn trace_serialization_is_byte_stable() {
        let evidence = window(0, vec![message("a", 0, 0)]);
        let output = combine_window_responses(&ProcessingInput {
            windows: vec![evidence],
            responses: vec![response(
                0,
                r#"[{"op":"KEEP","compressed_content":"a","reason":"relevant"}]"#,
            )],
        })
        .unwrap();
        let expected = serde_json::to_vec(&output).unwrap();
        for _ in 0..100 {
            assert_eq!(serde_json::to_vec(&output).unwrap(), expected);
        }
    }
}
