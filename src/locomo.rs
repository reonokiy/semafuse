//! Native, label-isolated ingestion for the pinned LoCoMo release.
//!
//! The released LoCoMo JSON keeps answers, question categories, evidence
//! references, and source dialog identifiers next to the conversation text.
//! This module separates those annotations from the serializable
//! [`BenchmarkInput`] given to a memory system. Every question, session, and
//! message identifier in that input is a deterministic opaque digest.
//!
//! LoCoMo data is licensed under CC BY-NC 4.0. Parsing is fail-closed unless
//! the caller explicitly acknowledges the non-commercial restriction. This is
//! an execution guardrail, not legal advice or a license-compliance checker.

use crate::longmemeval::{BenchmarkInput, BenchmarkMessage, RetrievalProfileGold};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};

/// Public dataset identifier written to every label-free benchmark input.
pub const DATASET_NAME: &str = "locomo";
/// SPDX identifier for the released LoCoMo data license.
pub const DATA_LICENSE: &str = "CC-BY-NC-4.0";
/// Pinned upstream repository revision used to validate this adapter.
pub const UPSTREAM_COMMIT: &str = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376";
/// SHA-256 of `data/locomo10.json` at [`UPSTREAM_COMMIT`].
pub const UPSTREAM_DATA_SHA256: &str =
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4";
/// Number of category-1--4 QA items in the LazyMem paper-v2 LoCoMo subset.
pub const PAPER_V2_QA_COUNT: usize = 314;
/// Number of paper-v2 QA items with non-empty retrieval evidence.
pub const PAPER_V2_RETRIEVAL_COUNT: usize = 312;
/// Conversation IDs selected by the LazyMem paper-v2 protocol.
pub const PAPER_V2_CONVERSATION_IDS: [&str; 2] = ["conv-49", "conv-50"];

const OPAQUE_ID_DOMAIN: &[u8] = b"semafuse/locomo/opaque-input-id/v1";

/// A category retained by the LazyMem paper-v2 LoCoMo protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LocomoCategory {
    /// Questions that combine multiple facts from the conversation.
    MultiHop = 1,
    /// Questions requiring temporal reasoning.
    Temporal = 2,
    /// Questions requiring open-domain inference over the conversation.
    OpenDomain = 3,
    /// Questions answered by one conversation fact.
    SingleHop = 4,
}

impl LocomoCategory {
    /// Returns the numeric category stored by the official dataset.
    pub const fn number(self) -> u8 {
        self as u8
    }

    /// Returns the stable report label used by the benchmark harness.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MultiHop => "multi-hop",
            Self::Temporal => "temporal",
            Self::OpenDomain => "open-domain",
            Self::SingleHop => "single-hop",
        }
    }
}

impl TryFrom<u8> for LocomoCategory {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::MultiHop),
            2 => Ok(Self::Temporal),
            3 => Ok(Self::OpenDomain),
            4 => Ok(Self::SingleHop),
            _ => bail!("LoCoMo paper protocol supports only categories 1--4; got {value}"),
        }
    }
}

/// Selection and denominator policy for one LoCoMo load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocomoMode {
    /// Exact LazyMem paper-v2 subset: `conv-49`, `conv-50`, categories 1--4,
    /// 314 QA questions, and 312 retrieval-eligible questions.
    PaperV2,
    /// Explicit small-data configuration intended for fixtures and adapter
    /// tests. This mode is never presented as a paper-comparable run.
    Fixture(FixtureSelection),
}

impl LocomoMode {
    /// Builds a configurable fixture selection.
    ///
    /// At least one conversation and one paper category are required. Optional
    /// expected counts are checked after adaptation when supplied.
    pub fn fixture<I, S, J>(
        conversation_ids: I,
        categories: J,
        expected_qa_count: Option<usize>,
        expected_retrieval_count: Option<usize>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        J: IntoIterator<Item = LocomoCategory>,
    {
        let mut selected_conversations = BTreeSet::new();
        for conversation_id in conversation_ids {
            let conversation_id = conversation_id.into();
            ensure!(
                !conversation_id.trim().is_empty(),
                "LoCoMo fixture selection contains an empty conversation id"
            );
            ensure!(
                selected_conversations.insert(conversation_id.clone()),
                "LoCoMo fixture selection contains duplicate conversation id {conversation_id}"
            );
        }
        ensure!(
            !selected_conversations.is_empty(),
            "LoCoMo fixture selection requires at least one conversation id"
        );

        let selected_categories = categories.into_iter().collect::<BTreeSet<_>>();
        ensure!(
            !selected_categories.is_empty(),
            "LoCoMo fixture selection requires at least one category"
        );
        if let (Some(qa_count), Some(retrieval_count)) =
            (expected_qa_count, expected_retrieval_count)
        {
            ensure!(
                retrieval_count <= qa_count,
                "LoCoMo fixture retrieval count {retrieval_count} exceeds QA count {qa_count}"
            );
        }

        Ok(Self::Fixture(FixtureSelection {
            conversation_ids: selected_conversations,
            categories: selected_categories,
            expected_qa_count,
            expected_retrieval_count,
        }))
    }

    fn conversation_ids(&self) -> BTreeSet<String> {
        match self {
            Self::PaperV2 => PAPER_V2_CONVERSATION_IDS
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Self::Fixture(selection) => selection.conversation_ids.clone(),
        }
    }

    fn categories(&self) -> BTreeSet<LocomoCategory> {
        match self {
            Self::PaperV2 => [
                LocomoCategory::MultiHop,
                LocomoCategory::Temporal,
                LocomoCategory::OpenDomain,
                LocomoCategory::SingleHop,
            ]
            .into_iter()
            .collect(),
            Self::Fixture(selection) => selection.categories.clone(),
        }
    }

    fn expected_counts(&self) -> (Option<usize>, Option<usize>) {
        match self {
            Self::PaperV2 => (Some(PAPER_V2_QA_COUNT), Some(PAPER_V2_RETRIEVAL_COUNT)),
            Self::Fixture(selection) => (
                selection.expected_qa_count,
                selection.expected_retrieval_count,
            ),
        }
    }
}

/// Validated selection settings for [`LocomoMode::Fixture`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixtureSelection {
    conversation_ids: BTreeSet<String>,
    categories: BTreeSet<LocomoCategory>,
    expected_qa_count: Option<usize>,
    expected_retrieval_count: Option<usize>,
}

/// Aggregate denominators produced while adapting a LoCoMo selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetSummary {
    /// Number of selected source conversations.
    pub conversation_count: usize,
    /// Number of selected category QA items.
    pub qa_count: usize,
    /// Number of QA items with non-empty, valid evidence gold.
    pub retrieval_count: usize,
    /// QA counts keyed by the four paper category labels.
    pub category_counts: BTreeMap<LocomoCategory, usize>,
}

/// One adapted LoCoMo selection and its checked evaluation denominators.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDataset {
    /// Questions with a hard boundary between model input and labels.
    pub questions: Vec<PreparedQuestion>,
    /// Aggregate counts for reporting and denominator validation.
    pub summary: DatasetSummary,
}

/// One LoCoMo question split into model-visible and evaluator-only views.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedQuestion {
    /// Serializable input containing no answer, category, evidence, or source ID.
    pub input: BenchmarkInput,
    /// Non-serializable labels and source mappings retained by the evaluator.
    pub evaluator: EvaluatorOnly,
}

/// Evaluator-owned LoCoMo annotations and source identifier mappings.
///
/// This type intentionally does not implement [`serde::Serialize`].
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatorOnly {
    /// Synthetic source question ID matching the established Python adapter.
    pub source_question_id: String,
    /// Source conversation identifier (`conv-49`, for example).
    pub source_conversation_id: String,
    /// Zero-based QA position in the source conversation.
    pub source_qa_index: usize,
    /// Official category retained outside model input.
    pub category: LocomoCategory,
    /// Reference answer in its original JSON representation.
    pub answer: Value,
    /// Evidence entries exactly as represented in the release.
    pub raw_evidence: Vec<Value>,
    /// Normalized source evidence IDs, including unpacked and de-padded IDs.
    pub source_evidence_ids: Vec<String>,
    /// Opaque retrieval gold and eligibility for this question.
    pub retrieval: RetrievalProfileGold,
    /// Source-to-opaque session mappings in chronological order.
    pub sessions: Vec<SessionIdMapping>,
    /// Source-to-opaque message mappings in chronological order.
    pub messages: Vec<MessageIdMapping>,
    /// First named participant from the source conversation.
    pub speaker_a: String,
    /// Second named participant from the source conversation.
    pub speaker_b: String,
}

/// Evaluator mapping for one source LoCoMo session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionIdMapping {
    /// Opaque session ID supplied to the memory system.
    pub opaque_id: String,
    /// Native dynamic key such as `session_4`.
    pub source_id: String,
    /// Numeric day/session index parsed from the native key.
    pub source_index: usize,
    /// Timestamp copied from the companion `_date_time` field.
    pub timestamp: String,
}

/// Evaluator mapping for one source LoCoMo message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageIdMapping {
    /// Opaque message ID supplied to the memory system.
    pub opaque_id: String,
    /// Native dialog ID such as `D4:6`.
    pub source_id: String,
    /// Native session key containing this message.
    pub source_session_id: String,
    /// Zero-based position in the flattened conversation history.
    pub message_index: usize,
}

/// Parses and adapts the official LoCoMo top-level JSON array.
///
/// `acknowledge_noncommercial` must be `true` for every mode. [`LocomoMode::PaperV2`]
/// additionally checks the fixed 314-QA and 312-retrieval denominators.
pub fn parse_locomo<R: Read>(
    reader: R,
    acknowledge_noncommercial: bool,
    mode: &LocomoMode,
) -> Result<PreparedDataset> {
    ensure!(
        acknowledge_noncommercial,
        "LoCoMo is CC BY-NC 4.0; pass acknowledge_noncommercial=true only after confirming non-commercial use and attribution obligations"
    );

    let samples: Vec<RawSample> =
        serde_json::from_reader(reader).context("parse official LoCoMo JSON array")?;
    adapt_samples(samples, mode)
}

/// Opens and adapts an official LoCoMo JSON file.
pub fn load_locomo(
    path: impl AsRef<Path>,
    acknowledge_noncommercial: bool,
    mode: &LocomoMode,
) -> Result<PreparedDataset> {
    let path = path.as_ref();
    let file =
        File::open(path).with_context(|| format!("open LoCoMo dataset {}", path.display()))?;
    parse_locomo(file, acknowledge_noncommercial, mode)
        .with_context(|| format!("load LoCoMo dataset {}", path.display()))
}

#[derive(Debug, Deserialize)]
struct RawSample {
    sample_id: String,
    conversation: Value,
    qa: Value,
}

#[derive(Clone)]
struct PreparedConversation {
    input_messages: Vec<BenchmarkMessage>,
    sessions: Vec<SessionIdMapping>,
    messages: Vec<MessageIdMapping>,
    source_to_opaque_message: HashMap<String, String>,
    source_to_opaque_session: HashMap<String, String>,
    canonical_source_message_ids: HashMap<String, String>,
    query_time: String,
    speaker_a: String,
    speaker_b: String,
}

fn adapt_samples(samples: Vec<RawSample>, mode: &LocomoMode) -> Result<PreparedDataset> {
    let selected_conversations = mode.conversation_ids();
    let selected_categories = mode.categories();
    let mut seen_sample_ids = HashSet::new();
    let mut found_conversations = BTreeSet::new();
    let mut questions = Vec::new();
    let mut category_counts = BTreeMap::new();

    for sample in samples {
        ensure!(
            !sample.sample_id.trim().is_empty(),
            "LoCoMo contains an empty sample_id"
        );
        ensure!(
            seen_sample_ids.insert(sample.sample_id.clone()),
            "LoCoMo contains duplicate sample_id {}",
            sample.sample_id
        );
        if !selected_conversations.contains(&sample.sample_id) {
            continue;
        }
        found_conversations.insert(sample.sample_id.clone());

        let conversation = prepare_conversation(&sample.sample_id, &sample.conversation)?;
        let qa_items = sample
            .qa
            .as_array()
            .with_context(|| format!("LoCoMo {} qa must be an array", sample.sample_id))?;

        for (qa_index, raw_qa) in qa_items.iter().enumerate() {
            let qa = raw_qa.as_object().with_context(|| {
                format!(
                    "LoCoMo {} QA {qa_index} must be an object",
                    sample.sample_id
                )
            })?;
            let Some(category) = selected_category(qa, &sample.sample_id, qa_index)? else {
                continue;
            };
            if !selected_categories.contains(&category) {
                continue;
            }
            let prepared = prepare_qa(&sample.sample_id, qa_index, qa, category, &conversation)?;
            *category_counts.entry(category).or_insert(0) += 1;
            questions.push(prepared);
        }
    }

    let missing = selected_conversations
        .difference(&found_conversations)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "LoCoMo selection is missing conversations: {}",
        missing.join(", ")
    );

    let retrieval_count = questions
        .iter()
        .filter(|question| question.evaluator.retrieval.eligible)
        .count();
    let summary = DatasetSummary {
        conversation_count: found_conversations.len(),
        qa_count: questions.len(),
        retrieval_count,
        category_counts,
    };
    let (expected_qa_count, expected_retrieval_count) = mode.expected_counts();
    if let Some(expected) = expected_qa_count {
        ensure!(
            summary.qa_count == expected,
            "LoCoMo selection must contain exactly {expected} QA questions; got {}",
            summary.qa_count
        );
    }
    if let Some(expected) = expected_retrieval_count {
        ensure!(
            summary.retrieval_count == expected,
            "LoCoMo selection must contain exactly {expected} retrieval-eligible questions; got {}",
            summary.retrieval_count
        );
    }

    Ok(PreparedDataset { questions, summary })
}

fn selected_category(
    qa: &Map<String, Value>,
    sample_id: &str,
    qa_index: usize,
) -> Result<Option<LocomoCategory>> {
    let raw = qa
        .get("category")
        .and_then(Value::as_u64)
        .with_context(|| format!("LoCoMo {sample_id} QA {qa_index} category must be an integer"))?;
    if !(1..=4).contains(&raw) {
        return Ok(None);
    }
    let number = u8::try_from(raw).context("convert LoCoMo category to u8")?;
    Ok(Some(LocomoCategory::try_from(number)?))
}

fn prepare_conversation(sample_id: &str, raw: &Value) -> Result<PreparedConversation> {
    let conversation = raw
        .as_object()
        .with_context(|| format!("LoCoMo {sample_id} conversation must be an object"))?;
    let speaker_a = optional_string(conversation, "speaker_a", sample_id, "conversation")?;
    let speaker_b = optional_string(conversation, "speaker_b", sample_id, "conversation")?;
    let mut session_keys = conversation
        .iter()
        .filter_map(|(key, value)| {
            parse_session_index(key)
                .filter(|_| value.is_array())
                .map(|index| (index, key.clone()))
        })
        .collect::<Vec<_>>();
    session_keys.sort();

    let mut input_messages = Vec::new();
    let mut session_mappings = Vec::new();
    let mut message_mappings = Vec::new();
    let mut source_to_opaque_message = HashMap::new();
    let mut source_to_opaque_session = HashMap::new();
    let mut canonical_source_message_ids = HashMap::new();

    for (session_index, session_key) in &session_keys {
        let timestamp_key = format!("{session_key}_date_time");
        let timestamp = optional_string(conversation, &timestamp_key, sample_id, session_key)?;
        let opaque_session_id = opaque_id("s", &[sample_id, session_key]);
        source_to_opaque_session.insert(session_key.clone(), opaque_session_id.clone());
        session_mappings.push(SessionIdMapping {
            opaque_id: opaque_session_id.clone(),
            source_id: session_key.clone(),
            source_index: *session_index,
            timestamp: timestamp.clone(),
        });

        let turns = conversation[session_key]
            .as_array()
            .expect("session keys were filtered to arrays");
        for (turn_index, raw_turn) in turns.iter().enumerate() {
            let turn = raw_turn.as_object().with_context(|| {
                format!("LoCoMo {sample_id} {session_key} turn {turn_index} must be an object")
            })?;
            let source_message_id = match turn.get("dia_id") {
                Some(Value::String(value)) => value.clone(),
                Some(_) => bail!(
                    "LoCoMo {sample_id} {session_key} turn {turn_index} dia_id must be a string"
                ),
                None => format!("{session_key}:{turn_index}"),
            };
            ensure!(
                !source_message_id.trim().is_empty(),
                "LoCoMo {sample_id} {session_key} turn {turn_index} has an empty dia_id"
            );
            let opaque_message_id = opaque_id("m", &[sample_id, &source_message_id]);
            ensure!(
                source_to_opaque_message
                    .insert(source_message_id.clone(), opaque_message_id.clone())
                    .is_none(),
                "duplicate LoCoMo dialog id {source_message_id} in {sample_id}"
            );
            if let Some(canonical) = canonical_dialog_id(&source_message_id)
                && let Some(existing) = canonical_source_message_ids
                    .insert(canonical.clone(), source_message_id.clone())
            {
                ensure!(
                    existing == source_message_id,
                    "LoCoMo {sample_id} dialog ids {existing} and {source_message_id} normalize to the same {canonical}"
                );
            }

            let text = optional_string(turn, "text", sample_id, &source_message_id)?;
            let speaker = optional_string(turn, "speaker", sample_id, &source_message_id)?;
            let caption = optional_string(turn, "blip_caption", sample_id, &source_message_id)?;
            let image_query = optional_string(turn, "query", sample_id, &source_message_id)?;
            let content = projected_content(&speaker, &text, &caption, &image_query);
            let role = if !speaker.is_empty() && speaker == speaker_b {
                "assistant"
            } else {
                "user"
            };

            message_mappings.push(MessageIdMapping {
                opaque_id: opaque_message_id.clone(),
                source_id: source_message_id,
                source_session_id: session_key.clone(),
                message_index: input_messages.len(),
            });
            input_messages.push(BenchmarkMessage {
                id: opaque_message_id,
                index: input_messages.len(),
                role: role.to_owned(),
                content,
                timestamp: timestamp.clone(),
                session_id: opaque_session_id.clone(),
            });
        }
    }

    let query_time = session_mappings
        .last()
        .map_or_else(String::new, |session| session.timestamp.clone());
    Ok(PreparedConversation {
        input_messages,
        sessions: session_mappings,
        messages: message_mappings,
        source_to_opaque_message,
        source_to_opaque_session,
        canonical_source_message_ids,
        query_time,
        speaker_a,
        speaker_b,
    })
}

fn prepare_qa(
    sample_id: &str,
    qa_index: usize,
    qa: &Map<String, Value>,
    category: LocomoCategory,
    conversation: &PreparedConversation,
) -> Result<PreparedQuestion> {
    let query = qa
        .get("question")
        .and_then(Value::as_str)
        .with_context(|| format!("LoCoMo {sample_id} QA {qa_index} question must be a string"))?;
    ensure!(
        !query.trim().is_empty(),
        "LoCoMo {sample_id} QA {qa_index} has empty question text"
    );
    let answer = qa
        .get("answer")
        .cloned()
        .with_context(|| format!("LoCoMo {sample_id} QA {qa_index} is missing answer"))?;
    let raw_evidence = qa
        .get("evidence")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let raw_evidence = raw_evidence
        .as_array()
        .with_context(|| format!("LoCoMo {sample_id} QA {qa_index} evidence must be an array"))?;
    let source_evidence_ids = resolve_evidence(
        raw_evidence,
        &conversation.source_to_opaque_message,
        &conversation.canonical_source_message_ids,
    )
    .with_context(|| format!("normalize LoCoMo {sample_id} QA {qa_index} evidence"))?;

    let mut gold_message_ids = Vec::with_capacity(source_evidence_ids.len());
    let mut gold_session_ids = Vec::new();
    let mut seen_gold_sessions = HashSet::new();
    let source_message_to_session = conversation
        .messages
        .iter()
        .map(|mapping| {
            (
                mapping.source_id.as_str(),
                mapping.source_session_id.as_str(),
            )
        })
        .collect::<HashMap<_, _>>();
    for source_message_id in &source_evidence_ids {
        gold_message_ids.push(conversation.source_to_opaque_message[source_message_id].clone());
        let source_session_id = source_message_to_session[source_message_id.as_str()];
        let opaque_session_id = conversation.source_to_opaque_session[source_session_id].clone();
        if seen_gold_sessions.insert(opaque_session_id.clone()) {
            gold_session_ids.push(opaque_session_id);
        }
    }
    let eligible = !gold_message_ids.is_empty();
    let input_id = opaque_id("q", &[sample_id, &qa_index.to_string()]);

    Ok(PreparedQuestion {
        input: BenchmarkInput {
            id: input_id,
            dataset: DATASET_NAME.to_owned(),
            query: query.to_owned(),
            query_time: conversation.query_time.clone(),
            messages: conversation.input_messages.clone(),
        },
        evaluator: EvaluatorOnly {
            source_question_id: format!("{sample_id}:q{qa_index:04}"),
            source_conversation_id: sample_id.to_owned(),
            source_qa_index: qa_index,
            category,
            answer,
            raw_evidence: raw_evidence.clone(),
            source_evidence_ids,
            retrieval: RetrievalProfileGold {
                eligible,
                gold_message_ids,
                gold_session_ids,
            },
            sessions: conversation.sessions.clone(),
            messages: conversation.messages.clone(),
            speaker_a: conversation.speaker_a.clone(),
            speaker_b: conversation.speaker_b.clone(),
        },
    })
}

fn optional_string(
    object: &Map<String, Value>,
    field: &str,
    sample_id: &str,
    context: &str,
) -> Result<String> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("LoCoMo {sample_id} {context} {field} must be a string or null"),
    }
}

fn parse_session_index(key: &str) -> Option<usize> {
    let suffix = key.strip_prefix("session_")?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

fn projected_content(speaker: &str, text: &str, caption: &str, image_query: &str) -> String {
    let mut parts = Vec::new();
    if speaker.is_empty() {
        if !text.is_empty() {
            parts.push(text.to_owned());
        }
    } else {
        parts.push(format!("{speaker}: {text}"));
    }
    if !caption.trim().is_empty() {
        parts.push(format!("Image caption: {}", caption.trim()));
    }
    if !image_query.trim().is_empty() {
        parts.push(format!("Image query: {}", image_query.trim()));
    }
    parts.join("\n")
}

fn resolve_evidence(
    raw_evidence: &[Value],
    source_to_opaque_message: &HashMap<String, String>,
    canonical_source_message_ids: &HashMap<String, String>,
) -> Result<Vec<String>> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();
    for item in raw_evidence {
        let text = match item {
            Value::String(text) => text.clone(),
            value => value.to_string(),
        };
        let references = evidence_references(&text);
        ensure!(
            !references.is_empty() || text.trim().is_empty(),
            "unrecognized LoCoMo evidence reference {item}"
        );
        for reference in references {
            let resolved = if source_to_opaque_message.contains_key(&reference) {
                reference
            } else {
                let canonical = canonical_dialog_id(&reference).with_context(|| {
                    format!(
                        "evidence dialog id is outside the supported numeric range: {reference}"
                    )
                })?;
                canonical_source_message_ids
                    .get(&canonical)
                    .cloned()
                    .with_context(|| format!("evidence references missing dialog {reference}"))?
            };
            if seen.insert(resolved.clone()) {
                normalized.push(resolved);
            }
        }
    }
    Ok(normalized)
}

fn evidence_references(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut references = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        if bytes[position] != b'D' {
            position += 1;
            continue;
        }
        let start = position;
        position += 1;
        let day_start = position;
        while position < bytes.len() && bytes[position].is_ascii_digit() {
            position += 1;
        }
        if position == day_start || position >= bytes.len() || bytes[position] != b':' {
            position = start + 1;
            continue;
        }
        position += 1;
        let turn_start = position;
        while position < bytes.len() && bytes[position].is_ascii_digit() {
            position += 1;
        }
        if position == turn_start {
            position = start + 1;
            continue;
        }
        references.push(text[start..position].to_owned());
    }
    references
}

fn canonical_dialog_id(value: &str) -> Option<String> {
    let suffix = value.strip_prefix('D')?;
    let (day, turn) = suffix.split_once(':')?;
    if day.is_empty()
        || turn.is_empty()
        || !day.bytes().all(|byte| byte.is_ascii_digit())
        || !turn.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some(format!(
        "D{}:{}",
        day.parse::<u64>().ok()?,
        turn.parse::<u64>().ok()?
    ))
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

    fn fixture_mode(
        expected_qa_count: Option<usize>,
        expected_retrieval_count: Option<usize>,
    ) -> LocomoMode {
        LocomoMode::fixture(
            ["conv-fixture"],
            [
                LocomoCategory::MultiHop,
                LocomoCategory::Temporal,
                LocomoCategory::OpenDomain,
                LocomoCategory::SingleHop,
            ],
            expected_qa_count,
            expected_retrieval_count,
        )
        .unwrap()
    }

    fn fixture_json() -> Value {
        json!([{
            "sample_id": "conv-fixture",
            "conversation": {
                "speaker_a": "Ada",
                "speaker_b": "Ben",
                "session_4_date_time": "4 January 2026",
                "session_4": [
                    {"speaker":"Ada", "dia_id":"D4:4", "text":"first fact"},
                    {"speaker":"Ben", "dia_id":"D4:6", "text":"second fact"}
                ],
                "session_9_date_time": "9 January 2026",
                "session_9": [
                    {"speaker":"Ada", "dia_id":"D9:1", "text":"third fact"}
                ],
                "session_30_date_time": "30 January 2026",
                "session_30": [{
                    "speaker":"Ben",
                    "dia_id":"D30:5",
                    "text":"camera fact",
                    "blip_caption":" a vintage camera on a table ",
                    "query":" vintage camera "
                }]
            },
            "qa": [
                {"question":"Combine facts", "answer":"private multi-hop answer", "evidence":["D9:1 D4:4 D4:6"], "category":1},
                {"question":"When camera", "answer":"private temporal answer", "evidence":["D30:05"], "category":2},
                {"question":"Infer something", "answer":"private open answer", "evidence":[], "category":3},
                {"question":"One fact", "answer":"private single answer", "evidence":["D4:4"], "category":4},
                {"question":"Adversarial", "answer":"excluded", "evidence":["D4:4"], "category":5}
            ]
        }])
    }

    fn parse_fixture() -> PreparedDataset {
        let bytes = serde_json::to_vec(&fixture_json()).unwrap();
        parse_locomo(bytes.as_slice(), true, &fixture_mode(Some(4), Some(3))).unwrap()
    }

    #[test]
    fn license_acknowledgement_is_required_before_parsing() {
        let error = parse_locomo(
            b"not even JSON".as_slice(),
            false,
            &fixture_mode(None, None),
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("CC BY-NC 4.0"));
        assert!(message.contains("acknowledge_noncommercial=true"));
        assert!(!message.contains("parse official"));
    }

    #[test]
    fn fixture_mode_filters_and_reports_all_four_paper_categories() {
        let dataset = parse_fixture();
        assert_eq!(dataset.summary.conversation_count, 1);
        assert_eq!(dataset.summary.qa_count, 4);
        assert_eq!(dataset.summary.retrieval_count, 3);
        assert_eq!(
            dataset
                .questions
                .iter()
                .map(|question| question.evaluator.category)
                .collect::<Vec<_>>(),
            vec![
                LocomoCategory::MultiHop,
                LocomoCategory::Temporal,
                LocomoCategory::OpenDomain,
                LocomoCategory::SingleHop,
            ]
        );
        assert!(
            dataset
                .summary
                .category_counts
                .values()
                .all(|count| *count == 1)
        );
    }

    #[test]
    fn packed_and_zero_padded_evidence_resolve_to_opaque_gold() {
        let dataset = parse_fixture();
        let packed = &dataset.questions[0];
        assert_eq!(
            packed.evaluator.source_evidence_ids,
            ["D9:1", "D4:4", "D4:6"]
        );
        assert_eq!(packed.evaluator.retrieval.gold_message_ids.len(), 3);
        assert!(
            packed
                .evaluator
                .retrieval
                .gold_message_ids
                .iter()
                .all(|id| id.starts_with("m_") && !id.contains("D"))
        );

        let zero_padded = &dataset.questions[1];
        assert_eq!(zero_padded.evaluator.source_evidence_ids, ["D30:5"]);
        assert_eq!(zero_padded.evaluator.retrieval.gold_message_ids.len(), 1);
        assert!(!dataset.questions[2].evaluator.retrieval.eligible);
    }

    #[test]
    fn speaker_caption_and_image_query_projection_matches_python_adapter() {
        let dataset = parse_fixture();
        let messages = &dataset.questions[0].input.messages;
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(
            messages[3].content,
            "Ben: camera fact\nImage caption: a vintage camera on a table\nImage query: vintage camera"
        );
        assert_eq!(messages[3].timestamp, "30 January 2026");
        assert_eq!(dataset.questions[0].input.query_time, "30 January 2026");
    }

    #[test]
    fn benchmark_input_contains_no_labels_or_source_identifiers() {
        let dataset = parse_fixture();
        let prepared = &dataset.questions[1];
        let input = serde_json::to_value(&prepared.input).unwrap();
        let object = input.as_object().unwrap();
        assert_eq!(
            object.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from(["dataset", "id", "messages", "query", "query_time"])
        );
        for message in object["messages"].as_array().unwrap() {
            assert_eq!(
                message
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["content", "id", "index", "role", "session_id", "timestamp"])
            );
        }

        let serialized = serde_json::to_string(&prepared.input).unwrap();
        for forbidden in [
            "conv-fixture",
            "D30:5",
            "D30:05",
            "session_30",
            "private temporal answer",
            "category",
            "evidence",
            "source_id",
        ] {
            assert!(!serialized.contains(forbidden), "leaked {forbidden}");
        }
        assert!(prepared.input.id.starts_with("q_"));
        assert!(
            prepared
                .input
                .messages
                .iter()
                .all(|message| message.id.starts_with("m_") && message.session_id.starts_with("s_"))
        );
    }

    #[test]
    fn paper_mode_enforces_both_published_denominators() {
        let mut value = fixture_json();
        value[0]["sample_id"] = json!("conv-49");
        let mut second = value[0].clone();
        second["sample_id"] = json!("conv-50");
        value.as_array_mut().unwrap().push(second);
        let bytes = serde_json::to_vec(&value).unwrap();
        let error = parse_locomo(bytes.as_slice(), true, &LocomoMode::PaperV2).unwrap_err();
        assert!(format!("{error:#}").contains("exactly 314 QA questions"));

        assert_eq!(PAPER_V2_QA_COUNT, 314);
        assert_eq!(PAPER_V2_RETRIEVAL_COUNT, 312);
    }

    #[test]
    fn malformed_or_missing_evidence_fails_closed() {
        let mut malformed = fixture_json();
        malformed[0]["qa"][0]["evidence"] = json!(["not-a-dialog-id"]);
        let bytes = serde_json::to_vec(&malformed).unwrap();
        let error = parse_locomo(bytes.as_slice(), true, &fixture_mode(None, None)).unwrap_err();
        assert!(format!("{error:#}").contains("unrecognized LoCoMo evidence"));

        let mut missing = fixture_json();
        missing[0]["qa"][0]["evidence"] = json!(["D99:1"]);
        let bytes = serde_json::to_vec(&missing).unwrap();
        let error = parse_locomo(bytes.as_slice(), true, &fixture_mode(None, None)).unwrap_err();
        assert!(format!("{error:#}").contains("missing dialog D99:1"));
    }

    #[test]
    fn fixture_selection_rejects_ambiguous_denominators() {
        let error = LocomoMode::fixture(
            ["conv-fixture"],
            [LocomoCategory::SingleHop],
            Some(2),
            Some(3),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("exceeds QA count"));
    }

    #[test]
    fn pinned_official_release_matches_paper_counts_when_configured() {
        let Ok(path) = std::env::var("SEMAFUSE_TEST_LOCOMO_PATH") else {
            return;
        };
        let dataset = load_locomo(path, true, &LocomoMode::PaperV2).unwrap();
        assert_eq!(dataset.summary.qa_count, PAPER_V2_QA_COUNT);
        assert_eq!(dataset.summary.retrieval_count, PAPER_V2_RETRIEVAL_COUNT);
        assert_eq!(
            dataset.summary.category_counts,
            BTreeMap::from([
                (LocomoCategory::MultiHop, 69),
                (LocomoCategory::Temporal, 65),
                (LocomoCategory::OpenDomain, 20),
                (LocomoCategory::SingleHop, 160),
            ])
        );
    }
}
