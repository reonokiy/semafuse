//! Native, label-isolated ingestion for the official LongMemEval JSON release.
//!
//! LongMemEval places evaluator annotations next to the messages that a memory
//! system consumes.  This module deliberately splits every question into a
//! serializable [`BenchmarkInput`](crate::longmemeval::BenchmarkInput) and a
//! non-serializable [`EvaluatorOnly`](crate::longmemeval::EvaluatorOnly)
//! view.  The former contains only opaque identifiers, message text, roles,
//! and the original timestamps.  Source identifiers and all answer labels stay
//! in the evaluator view.

use anyhow::{Context, Result, ensure};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{DeserializeSeed, Error as _, SeqAccess, Visitor},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::Path,
};

const DATASET_NAME: &str = "longmemeval";
const OPAQUE_ID_DOMAIN: &[u8] = b"semafuse/longmemeval/opaque-input-id/v1";

/// A set of official LongMemEval question IDs selected for one benchmark run.
///
/// Source question IDs are used only while parsing.  They are never copied into
/// [`BenchmarkInput`], including IDs whose `_abs` suffix would reveal an
/// abstention label.
#[derive(Clone, Debug, Default)]
pub struct SplitIdFilter {
    ids: HashSet<String>,
}

impl SplitIdFilter {
    /// Builds a filter from source question IDs, rejecting empty or duplicate
    /// values so a malformed split cannot silently change its denominator.
    pub fn from_ids<I, S>(ids: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut selected = HashSet::new();
        for id in ids {
            let id = id.into();
            ensure!(!id.trim().is_empty(), "split contains an empty question id");
            ensure!(
                selected.insert(id.clone()),
                "split contains duplicate question id {id}"
            );
        }
        Ok(Self { ids: selected })
    }

    /// Reads a split from a JSON manifest.
    ///
    /// Supported shapes are a direct string array, `{ "test": [...] }`,
    /// `{ "ids": { "test": [...] } }`, and
    /// `{ "splits": { "test": [...] } }`.
    pub fn from_manifest_reader<R: Read>(reader: R, split_name: &str) -> Result<Self> {
        ensure!(
            !split_name.trim().is_empty(),
            "split name must be non-empty"
        );
        let manifest: Value =
            serde_json::from_reader(reader).context("parse LongMemEval split manifest")?;
        let ids = if manifest.is_array() {
            manifest.as_array()
        } else {
            manifest
                .get("ids")
                .and_then(|value| value.get(split_name))
                .or_else(|| {
                    manifest
                        .get("splits")
                        .and_then(|value| value.get(split_name))
                })
                .or_else(|| manifest.get(split_name))
                .and_then(Value::as_array)
        }
        .with_context(|| format!("split manifest has no string-array split named {split_name}"))?;

        let mut parsed = Vec::with_capacity(ids.len());
        for (index, id) in ids.iter().enumerate() {
            let id = id.as_str().with_context(|| {
                format!("split {split_name} has a non-string id at index {index}")
            })?;
            parsed.push(id.to_owned());
        }
        Self::from_ids(parsed)
    }

    /// Returns the number of selected source questions.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns whether this filter selects no questions.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn contains(&self, source_question_id: &str) -> bool {
        self.ids.contains(source_question_id)
    }
}

/// The only question view that may be passed to a benchmarked memory system.
///
/// Its `id`, message IDs, and session IDs are deterministic SHA-256-based
/// opaque identifiers.  It intentionally has no answer, question type,
/// abstention flag, source identifier, or evidence label fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkInput {
    /// A stable opaque identifier for joining a prediction to evaluator state.
    pub id: String,
    /// The public dataset name, which contains no per-question information.
    pub dataset: String,
    /// The question presented to the memory system.
    pub query: String,
    /// The original LongMemEval `question_date`, preserved without rewriting.
    pub query_time: String,
    /// All history messages in source session/message order.
    pub messages: Vec<BenchmarkMessage>,
}

/// One label-free message supplied to a benchmarked memory system.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkMessage {
    /// A stable opaque message identifier.
    pub id: String,
    /// The zero-based position in the flattened chronological history.
    pub index: usize,
    /// The message role exactly as represented by the official dataset.
    pub role: String,
    /// The original message content.
    pub content: String,
    /// The original timestamp of this session, preserved without rewriting.
    pub timestamp: String,
    /// A stable opaque ID for this particular session occurrence.
    pub session_id: String,
}

/// Evaluator-owned annotations and opaque-to-source identifier mappings.
///
/// This type intentionally does not implement [`Serialize`].  A benchmark
/// runner must explicitly choose which aggregate fields to publish rather than
/// accidentally forwarding labels to a memory backend.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatorOnly {
    /// The original official question ID.
    pub source_question_id: String,
    /// The original LongMemEval question type.
    pub question_type: String,
    /// The reference answer in its original JSON representation.
    pub answer: Value,
    /// Whether the source question ID ends in the official `_abs` marker.
    pub is_abstention: bool,
    /// Source answer-session IDs, retained only for evaluator auditing.
    pub source_answer_session_ids: Vec<String>,
    /// Per-occurrence session ID mappings in chronological order.
    pub sessions: Vec<SessionIdMapping>,
    /// Message ID mappings in flattened chronological order.
    pub messages: Vec<MessageIdMapping>,
    /// Gold for the strict official user-turn retrieval protocol.
    pub official_user_turn: RetrievalProfileGold,
    /// Gold for all-role message retrieval over non-abstention questions.
    pub all_role_non_abstention: RetrievalProfileGold,
}

/// The evaluator mapping for one occurrence of a source session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionIdMapping {
    /// The opaque session-occurrence ID visible to the memory system.
    pub opaque_id: String,
    /// The original LongMemEval session ID, evaluator-only because it may
    /// contain label-signalling text such as `answer_`.
    pub source_id: String,
    /// The zero-based occurrence count among sessions with this source ID.
    pub source_occurrence: usize,
    /// The zero-based position in the question's session sequence.
    pub session_index: usize,
    /// The original session timestamp, preserved without rewriting.
    pub timestamp: String,
}

/// The evaluator mapping for one source message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageIdMapping {
    /// The opaque message ID visible to the memory system.
    pub opaque_id: String,
    /// The original LongMemEval session ID.
    pub source_session_id: String,
    /// The zero-based occurrence count of the source session ID.
    pub source_session_occurrence: usize,
    /// The zero-based position of the session in the question history.
    pub session_index: usize,
    /// The zero-based position of the message inside its session.
    pub message_index: usize,
}

/// Eligibility and gold opaque IDs for one retrieval evaluation profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetrievalProfileGold {
    /// Whether this question belongs in the profile denominator.
    pub eligible: bool,
    /// Gold opaque message IDs in chronological order.
    pub gold_message_ids: Vec<String>,
    /// Gold opaque session-occurrence IDs in chronological order.
    pub gold_session_ids: Vec<String>,
}

/// One parsed question with a hard boundary between model input and evaluator
/// annotations.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedQuestion {
    /// The label-free, serializable model input.
    pub input: BenchmarkInput,
    /// Evaluator-only source mappings and gold annotations.
    pub evaluator: EvaluatorOnly,
}

/// Streams the official top-level LongMemEval JSON array into prepared cases.
///
/// When `split_filter` is present, only selected questions are materialized,
/// but the parser still checks the entire source for duplicate question IDs and
/// fails if any requested split ID is absent.
pub fn parse_longmemeval<R: Read>(
    reader: R,
    split_filter: Option<&SplitIdFilter>,
) -> Result<Vec<PreparedQuestion>> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let prepared = DatasetSeed { split_filter }
        .deserialize(&mut deserializer)
        .context("parse official LongMemEval JSON array")?;
    deserializer
        .end()
        .context("reject trailing data after LongMemEval JSON array")?;
    Ok(prepared)
}

/// Opens and streams an official LongMemEval JSON file.
pub fn load_longmemeval(
    path: impl AsRef<Path>,
    split_filter: Option<&SplitIdFilter>,
) -> Result<Vec<PreparedQuestion>> {
    let path = path.as_ref();
    let file =
        File::open(path).with_context(|| format!("open LongMemEval dataset {}", path.display()))?;
    parse_longmemeval(file, split_filter)
        .with_context(|| format!("load LongMemEval dataset {}", path.display()))
}

#[derive(Debug, Deserialize)]
struct RawQuestion {
    question_id: String,
    question_type: String,
    question: String,
    answer: Value,
    question_date: String,
    haystack_session_ids: Vec<String>,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<RawMessage>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawMessage {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: Option<bool>,
}

struct DatasetSeed<'a> {
    split_filter: Option<&'a SplitIdFilter>,
}

impl<'de> DeserializeSeed<'de> for DatasetSeed<'_> {
    type Value = Vec<PreparedQuestion>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(DatasetVisitor {
            split_filter: self.split_filter,
        })
    }
}

struct DatasetVisitor<'a> {
    split_filter: Option<&'a SplitIdFilter>,
}

impl<'de> Visitor<'de> for DatasetVisitor<'_> {
    type Value = Vec<PreparedQuestion>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the official LongMemEval top-level JSON array")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut prepared = Vec::new();
        let mut seen_source_ids = HashSet::new();
        let mut matched_filter_ids = HashSet::new();

        while let Some(raw) = sequence.next_element::<RawQuestion>()? {
            if !seen_source_ids.insert(raw.question_id.clone()) {
                return Err(A::Error::custom(format!(
                    "LongMemEval contains duplicate question id {}",
                    raw.question_id
                )));
            }

            let selected = self
                .split_filter
                .is_none_or(|filter| filter.contains(&raw.question_id));
            if !selected {
                continue;
            }
            matched_filter_ids.insert(raw.question_id.clone());
            prepared.push(prepare_question(raw).map_err(A::Error::custom)?);
        }

        if let Some(filter) = self.split_filter {
            let mut missing = filter
                .ids
                .difference(&matched_filter_ids)
                .cloned()
                .collect::<Vec<_>>();
            missing.sort();
            if !missing.is_empty() {
                return Err(A::Error::custom(format!(
                    "split references question ids absent from LongMemEval: {}",
                    missing.join(", ")
                )));
            }
        }

        Ok(prepared)
    }
}

fn prepare_question(raw: RawQuestion) -> Result<PreparedQuestion> {
    ensure!(
        !raw.question_id.trim().is_empty(),
        "LongMemEval question has an empty question_id"
    );
    ensure!(
        !raw.question.trim().is_empty(),
        "LongMemEval question {} has empty question text",
        raw.question_id
    );
    ensure!(
        raw.haystack_session_ids.len() == raw.haystack_dates.len()
            && raw.haystack_session_ids.len() == raw.haystack_sessions.len(),
        "LongMemEval question {} has mismatched session id/date/content lengths ({}/{}/{})",
        raw.question_id,
        raw.haystack_session_ids.len(),
        raw.haystack_dates.len(),
        raw.haystack_sessions.len()
    );

    let answer_session_set = raw
        .answer_session_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let input_id = opaque_id("q", &[raw.question_id.as_str()]);
    let is_abstention = raw.question_id.ends_with("_abs");
    let mut source_occurrences = HashMap::<String, usize>::new();
    let mut input_messages = Vec::new();
    let mut session_mappings = Vec::with_capacity(raw.haystack_sessions.len());
    let mut message_mappings = Vec::new();
    let mut official_gold_messages = Vec::new();
    let mut official_gold_sessions = Vec::new();
    let mut all_role_gold_messages = Vec::new();
    let mut all_role_gold_sessions = Vec::new();

    for (session_index, ((source_session_id, timestamp), source_messages)) in raw
        .haystack_session_ids
        .iter()
        .zip(&raw.haystack_dates)
        .zip(&raw.haystack_sessions)
        .enumerate()
    {
        ensure!(
            !source_session_id.trim().is_empty(),
            "LongMemEval question {} has an empty session id at index {}",
            raw.question_id,
            session_index
        );
        let source_occurrence = source_occurrences
            .entry(source_session_id.clone())
            .and_modify(|occurrence| *occurrence += 1)
            .or_insert(0);
        let source_occurrence = *source_occurrence;
        let session_index_text = session_index.to_string();
        let opaque_session_id = opaque_id("s", &[input_id.as_str(), session_index_text.as_str()]);
        session_mappings.push(SessionIdMapping {
            opaque_id: opaque_session_id.clone(),
            source_id: source_session_id.clone(),
            source_occurrence,
            session_index,
            timestamp: timestamp.clone(),
        });

        let official_before = official_gold_messages.len();
        let all_role_before = all_role_gold_messages.len();
        let is_answer_session = answer_session_set.contains(source_session_id.as_str());

        for (message_index, source_message) in source_messages.iter().enumerate() {
            ensure!(
                !source_message.role.trim().is_empty(),
                "LongMemEval question {} has an empty role at session {}, message {}",
                raw.question_id,
                session_index,
                message_index
            );
            let message_index_text = message_index.to_string();
            let opaque_message_id = opaque_id(
                "m",
                &[
                    input_id.as_str(),
                    session_index_text.as_str(),
                    message_index_text.as_str(),
                ],
            );
            let has_answer = source_message.has_answer.unwrap_or(false);
            ensure!(
                !has_answer || is_answer_session,
                "LongMemEval question {} marks evidence outside answer_session_ids at session {}, message {}",
                raw.question_id,
                session_index,
                message_index
            );

            if has_answer {
                all_role_gold_messages.push(opaque_message_id.clone());
                if source_message.role == "user" {
                    official_gold_messages.push(opaque_message_id.clone());
                }
            }

            message_mappings.push(MessageIdMapping {
                opaque_id: opaque_message_id.clone(),
                source_session_id: source_session_id.clone(),
                source_session_occurrence: source_occurrence,
                session_index,
                message_index,
            });
            input_messages.push(BenchmarkMessage {
                id: opaque_message_id,
                index: input_messages.len(),
                role: source_message.role.clone(),
                content: source_message.content.clone(),
                timestamp: timestamp.clone(),
                session_id: opaque_session_id.clone(),
            });
        }

        if official_gold_messages.len() > official_before {
            official_gold_sessions.push(opaque_session_id.clone());
        }
        if all_role_gold_messages.len() > all_role_before {
            all_role_gold_sessions.push(opaque_session_id);
        }
    }

    let official_user_turn = profile_gold(
        is_abstention,
        official_gold_messages,
        official_gold_sessions,
    );
    let all_role_non_abstention = profile_gold(
        is_abstention,
        all_role_gold_messages,
        all_role_gold_sessions,
    );

    Ok(PreparedQuestion {
        input: BenchmarkInput {
            id: input_id,
            dataset: DATASET_NAME.to_owned(),
            query: raw.question,
            query_time: raw.question_date,
            messages: input_messages,
        },
        evaluator: EvaluatorOnly {
            source_question_id: raw.question_id,
            question_type: raw.question_type,
            answer: raw.answer,
            is_abstention,
            source_answer_session_ids: raw.answer_session_ids,
            sessions: session_mappings,
            messages: message_mappings,
            official_user_turn,
            all_role_non_abstention,
        },
    })
}

fn profile_gold(
    is_abstention: bool,
    gold_message_ids: Vec<String>,
    gold_session_ids: Vec<String>,
) -> RetrievalProfileGold {
    if is_abstention || gold_message_ids.is_empty() {
        RetrievalProfileGold {
            eligible: false,
            gold_message_ids: Vec::new(),
            gold_session_ids: Vec::new(),
        }
    } else {
        RetrievalProfileGold {
            eligible: true,
            gold_message_ids,
            gold_session_ids,
        }
    }
}

fn opaque_id(kind: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(OPAQUE_ID_DOMAIN);
    digest.update((kind.len() as u64).to_be_bytes());
    digest.update(kind.as_bytes());
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part.as_bytes());
    }
    format!("{kind}_{}", hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn question(
        question_id: &str,
        session_ids: Value,
        dates: Value,
        sessions: Value,
        answer_session_ids: Value,
    ) -> Value {
        json!({
            "question_id": question_id,
            "question_type": "single-session-user",
            "question": "What should be remembered?",
            "answer": "the evaluator-only answer",
            "question_date": "2025-03-04",
            "haystack_session_ids": session_ids,
            "haystack_dates": dates,
            "haystack_sessions": sessions,
            "answer_session_ids": answer_session_ids,
        })
    }

    fn parse_one(value: Value) -> PreparedQuestion {
        let bytes = serde_json::to_vec(&vec![value]).unwrap();
        parse_longmemeval(bytes.as_slice(), None)
            .unwrap()
            .pop()
            .unwrap()
    }

    #[test]
    fn duplicate_session_occurrences_receive_distinct_opaque_ids() {
        let prepared = parse_one(question(
            "source-question",
            json!(["answer_secret", "answer_secret"]),
            json!(["2024-01-02", "2024-02-03"]),
            json!([
                [{"role":"user", "content":"gold", "has_answer":true}],
                [{"role":"user", "content":"later duplicate", "has_answer":false}]
            ]),
            json!(["answer_secret"]),
        ));

        assert_ne!(
            prepared.input.messages[0].session_id,
            prepared.input.messages[1].session_id
        );
        assert_eq!(prepared.input.messages[0].timestamp, "2024-01-02");
        assert_eq!(prepared.input.messages[1].timestamp, "2024-02-03");
        assert_eq!(prepared.evaluator.sessions[0].source_occurrence, 0);
        assert_eq!(prepared.evaluator.sessions[1].source_occurrence, 1);
        assert_eq!(
            prepared.evaluator.official_user_turn.gold_session_ids,
            vec![prepared.input.messages[0].session_id.clone()]
        );

        let model_json = serde_json::to_string(&prepared.input).unwrap();
        assert!(!model_json.contains("source-question"));
        assert!(!model_json.contains("answer_secret"));
        assert!(!model_json.contains("has_answer"));
        assert!(!model_json.contains("question_type"));
    }

    #[test]
    fn assistant_only_gold_is_excluded_only_from_official_user_turn_profile() {
        let mut raw = question(
            "assistant-evidence",
            json!(["answer_assistant_secret"]),
            json!(["2024-05-06"]),
            json!([[
                {"role":"user", "content":"please remind me", "has_answer":false},
                {"role":"assistant", "content":"the answer is blue", "has_answer":true}
            ]]),
            json!(["answer_assistant_secret"]),
        );
        raw["question_type"] = json!("single-session-assistant");
        let prepared = parse_one(raw);

        assert!(!prepared.evaluator.official_user_turn.eligible);
        assert!(
            prepared
                .evaluator
                .official_user_turn
                .gold_message_ids
                .is_empty()
        );
        assert!(prepared.evaluator.all_role_non_abstention.eligible);
        assert_eq!(
            prepared.evaluator.all_role_non_abstention.gold_message_ids,
            vec![prepared.input.messages[1].id.clone()]
        );
    }

    #[test]
    fn abstention_is_ineligible_even_if_source_contains_an_evidence_label() {
        let prepared = parse_one(question(
            "private-id_abs",
            json!(["answer_anomalous"]),
            json!(["2024-07-08"]),
            json!([[{"role":"user", "content":"anomaly", "has_answer":true}]]),
            json!(["answer_anomalous"]),
        ));

        assert!(prepared.evaluator.is_abstention);
        assert!(!prepared.evaluator.official_user_turn.eligible);
        assert!(!prepared.evaluator.all_role_non_abstention.eligible);
        assert!(
            prepared
                .evaluator
                .all_role_non_abstention
                .gold_message_ids
                .is_empty()
        );
        let model_json = serde_json::to_string(&prepared.input).unwrap();
        assert!(!model_json.contains("private-id_abs"));
        assert!(!model_json.contains("answer_anomalous"));
    }

    #[test]
    fn split_filter_selects_ids_and_rejects_missing_ids() {
        let first = question(
            "first",
            json!(["session-a"]),
            json!(["2024-01-01"]),
            json!([[{"role":"user", "content":"a"}]]),
            json!([]),
        );
        let second = question(
            "second",
            json!(["session-b"]),
            json!(["2024-01-02"]),
            json!([[{"role":"user", "content":"b"}]]),
            json!([]),
        );
        let bytes = serde_json::to_vec(&vec![first, second]).unwrap();
        let filter = SplitIdFilter::from_ids(["second"]).unwrap();
        let parsed = parse_longmemeval(bytes.as_slice(), Some(&filter)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].evaluator.source_question_id, "second");

        let missing = SplitIdFilter::from_ids(["missing"]).unwrap();
        let error = parse_longmemeval(bytes.as_slice(), Some(&missing)).unwrap_err();
        assert!(format!("{error:#}").contains("absent from LongMemEval"));
    }

    #[test]
    fn split_manifest_supports_the_public_reproduction_shape() {
        let manifest = json!({"ids":{"train":["a"],"test":["b","c"]}});
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let filter = SplitIdFilter::from_manifest_reader(bytes.as_slice(), "test").unwrap();
        assert_eq!(filter.len(), 2);
        assert!(filter.contains("b"));
        assert!(filter.contains("c"));
    }
}
