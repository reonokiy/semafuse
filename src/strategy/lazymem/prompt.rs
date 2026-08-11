use super::window::EvidenceWindow;
use crate::{model::MessageRole, strategy::RecallRequest};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashSet;

pub(super) const PROCESSOR_PROMPT_VERSION: &str = "lazymem-appendix-g.1-v5";

const PROCESSOR_RETRY_CORRECTION: &str = r#"

# Retry Correction
The previous output was truncated or invalid. Return only valid JSON.
Emit exactly one decision for every non-empty role message, in original order.
For this retry, keep compressed_content at most 160 Unicode characters and reason at most 32 Unicode characters.
Prefer DROP for irrelevant messages; DROP compressed_content must be an empty string.
Do not include markdown, prose, or extra text; output only the JSON array."#;

pub(super) const PROCESSOR_SYSTEM_PROMPT: &str = r#"You are a local memory editing model.

You will receive a user query and one local chronological memory window.
Your task is to output a sequence of local editing decisions for this local window only, following the original message order.

Return exactly one valid JSON array.
Each item in the array must correspond to one non-empty role message in previous_bridge, core_messages, and next_bridge.
Each item in the array must contain exactly three fields: op, compressed_content, and reason.

The only allowed decision operations are KEEP and DROP.
For KEEP, retain dialogue content that is relevant to the query and useful for answering the query.
KEEP may be realized as verbatim keeping, key-span extraction, or concise compression depending on each message.
For DROP, compressed_content must be an empty string.

Use only the provided window content.
Do not output markdown, explanations, or any extra text.
Your output must be directly parseable by json.loads()."#;

const PROCESSOR_USER_PREFIX: &str = r#"# Task
Given the following query-conditioned memory window, produce a sequence of local editing decisions.

# Required Output
Return only a JSON array with the following structure:
[
  {
    "op": "KEEP | DROP",
    "compressed_content": "empty string if DROP; otherwise retained content in one KEEP style: verbatim keeping, key-span extraction, or concise compression",
    "reason": "brief local reason"
  }
]

# Local Editing Rules
1. The output array must contain one decision for every non-empty role message in previous_bridge, core_messages, and next_bridge.
2. The decisions must follow the original chronological order: previous_bridge, then core_messages, then next_bridge.
3. Each decision must contain exactly three fields: op, compressed_content, and reason.
4. op must be either KEEP or DROP.
5. KEEP means preserving dialogue content that is relevant to the query and useful for the main model to answer the question.
6. DROP means the message is locally irrelevant and should not enter the final memory.
7. For KEEP, choose the most appropriate retention style for each message:
   - verbatim keeping: keep the original message text when it is short and directly useful;
   - key-span extraction: keep only the query-relevant span when only part of the message is relevant;
   - concise compression: faithfully rewrite the message into a shorter form when it is useful but too long, or only partially relevant.
8. For any KEEP style, remain faithful to the source meaning and do not introduce unsupported facts.
9. If the whole window is irrelevant, still output one decision for every non-empty role message, and set every op to DROP.
10. When op is DROP, compressed_content must be an empty string.
11. If a message may contain answer evidence, user facts, temporal updates, or content related to the query entity/attribute, prefer KEEP even when relevance is uncertain.
12. Keep each compressed_content no longer than 512 Unicode characters; faithfully compress longer source messages.
13. Keep each reason brief and no longer than 256 Unicode characters.
14. Return only the JSON array. Do not output markdown, comments, or extra text.
15. The output must be directly parseable by json.loads().

# Query
Query Time: "#;

#[derive(Serialize)]
struct ProcessorPromptMessage<'a> {
    position: &'static str,
    role: MessageRole,
    timestamp: String,
    global_message_index: u64,
    retrieval_rank: Option<usize>,
    content: &'a str,
}

#[derive(Serialize)]
struct ProcessorPromptWindow<'a> {
    previous_bridge: Vec<ProcessorPromptMessage<'a>>,
    core_messages: Vec<ProcessorPromptMessage<'a>>,
    next_bridge: Vec<ProcessorPromptMessage<'a>>,
}

pub(super) fn processor_prompt(request: &RecallRequest, window: &EvidenceWindow) -> Result<String> {
    let core_ids = window
        .core_message_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let first_core = window
        .messages
        .iter()
        .position(|message| core_ids.contains(message.message_id.as_str()))
        .context("processor window has no core messages")?;
    let last_core = window
        .messages
        .iter()
        .rposition(|message| core_ids.contains(message.message_id.as_str()))
        .context("processor window has no core messages")?;
    let mut payload = ProcessorPromptWindow {
        previous_bridge: Vec::new(),
        core_messages: Vec::new(),
        next_bridge: Vec::new(),
    };
    for (index, message) in window.messages.iter().enumerate() {
        let (position, section) = if index < first_core {
            ("previous_bridge", &mut payload.previous_bridge)
        } else if index <= last_core {
            ("core", &mut payload.core_messages)
        } else {
            ("next_bridge", &mut payload.next_bridge)
        };
        section.push(ProcessorPromptMessage {
            position,
            role: message.role,
            timestamp: timestamp_utc(message.timestamp_ms)
                .unwrap_or_else(|| message.timestamp_ms.to_string()),
            global_message_index: message.global_ordinal,
            retrieval_rank: window.retrieval_ranks.get(&message.message_id).copied(),
            content: &message.content,
        });
    }
    ensure!(
        payload
            .core_messages
            .iter()
            .all(|message| message.position == "core"),
        "processor core partition is invalid"
    );
    let rendered_window =
        serde_json::to_string_pretty(&payload).context("serialize memory processor window")?;
    let query_time =
        timestamp_utc(request.query_time_ms).unwrap_or_else(|| request.query_time_ms.to_string());
    let mut prompt = String::with_capacity(
        PROCESSOR_USER_PREFIX.len()
            + query_time.len()
            + request.query.text.len()
            + rendered_window.len()
            + 64,
    );
    prompt.push_str(PROCESSOR_USER_PREFIX);
    prompt.push_str(&query_time);
    prompt.push_str("\nQuery: ");
    prompt.push_str(&request.query.text);
    prompt.push_str("\n\n# History Window\n");
    prompt.push_str(&rendered_window);
    Ok(prompt)
}

pub(super) fn processor_prompt_for_attempt(base_prompt: &str, attempt: usize) -> Cow<'_, str> {
    if attempt <= 1 || base_prompt.ends_with(PROCESSOR_RETRY_CORRECTION) {
        Cow::Borrowed(base_prompt)
    } else {
        Cow::Owned(format!("{base_prompt}{PROCESSOR_RETRY_CORRECTION}"))
    }
}

fn timestamp_utc(timestamp_ms: i64) -> Option<String> {
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(timestamp_ms)
        .map(|value| value.to_rfc3339())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_correction_is_absent_initially_and_appended_once_on_retries() {
        let base = "paper prompt";
        let initial = processor_prompt_for_attempt(base, 1);
        assert_eq!(initial, base);
        assert!(!initial.contains("# Retry Correction"));

        let retry = processor_prompt_for_attempt(base, 2);
        assert!(retry.contains("# Retry Correction"));
        assert!(retry.contains("compressed_content at most 160 Unicode characters"));
        assert!(retry.contains("reason at most 32 Unicode characters"));
        assert_eq!(retry.matches("# Retry Correction").count(), 1);

        let later_retry = processor_prompt_for_attempt(&retry, 3);
        assert_eq!(later_retry, retry);
        assert_eq!(later_retry.matches("# Retry Correction").count(), 1);
    }
}
