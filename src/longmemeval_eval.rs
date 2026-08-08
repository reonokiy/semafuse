//! Pure LongMemEval answer-prompt and official LLM-judge evaluation helpers.
//!
//! This module deliberately performs no HTTP or model invocation.  The caller
//! generates one answer for every selected LongMemEval question, including
//! abstention questions, then passes the judge prompt to its pinned judge
//! backend and applies [`judge_response_is_correct`] to the raw response.

use serde::{Deserialize, Serialize};
use std::{error::Error, fmt, str::FromStr};

/// Schema shared by serializable prediction and judge-score records.
pub const LONGMEMEVAL_EVAL_SCHEMA_VERSION: u32 = 1;

/// Metric name used by the Python harness and result summaries.
pub const LONGMEMEVAL_OFFICIAL_LJ_METRIC: &str = "longmemeval_official_lj";

/// Exact upstream-compatible decision rule applied to a judge response.
pub const LONGMEMEVAL_JUDGE_DECISION_RULE: &str = "substring 'yes' after Unicode lowercase";

/// System prompt used by the common answer model for every memory method.
pub const ANSWER_SYSTEM_PROMPT: &str =
    "Answer the user question concisely and faithfully from the provided memory.";

/// General-answer judge template used for user, assistant, and multi-session questions.
pub const GENERAL_JUDGE_TEMPLATE: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.";

/// Temporal-reasoning judge template, including the official off-by-one rule.
pub const TEMPORAL_JUDGE_TEMPLATE: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.";

/// Knowledge-update judge template.
pub const KNOWLEDGE_UPDATE_JUDGE_TEMPLATE: &str = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.";

/// Preference judge template; the reference answer is treated as a rubric.
pub const PREFERENCE_JUDGE_TEMPLATE: &str = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.";

/// Abstention judge template.  Abstention questions remain in answer evaluation.
pub const ABSTENTION_JUDGE_TEMPLATE: &str = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.";

/// The six question-type values in the released LongMemEval dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LongMemEvalQuestionType {
    /// Fact supplied by the user within one session.
    #[serde(rename = "single-session-user")]
    SingleSessionUser,
    /// Fact supplied by the assistant within one session.
    #[serde(rename = "single-session-assistant")]
    SingleSessionAssistant,
    /// Personalized-response preference within one session.
    #[serde(rename = "single-session-preference")]
    SingleSessionPreference,
    /// Evidence that must be combined across sessions.
    #[serde(rename = "multi-session")]
    MultiSession,
    /// Question requiring temporal reasoning over the history.
    #[serde(rename = "temporal-reasoning")]
    TemporalReasoning,
    /// Question requiring the latest value after an update.
    #[serde(rename = "knowledge-update")]
    KnowledgeUpdate,
}

impl LongMemEvalQuestionType {
    /// Returns the released dataset string for this question type.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SingleSessionUser => "single-session-user",
            Self::SingleSessionAssistant => "single-session-assistant",
            Self::SingleSessionPreference => "single-session-preference",
            Self::MultiSession => "multi-session",
            Self::TemporalReasoning => "temporal-reasoning",
            Self::KnowledgeUpdate => "knowledge-update",
        }
    }
}

impl fmt::Display for LongMemEvalQuestionType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Unknown LongMemEval question type rejected before prompt construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedQuestionType {
    /// Unsupported raw type supplied by the caller.
    pub value: String,
}

impl fmt::Display for UnsupportedQuestionType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported LongMemEval question type {:?}",
            self.value
        )
    }
}

impl Error for UnsupportedQuestionType {}

impl FromStr for LongMemEvalQuestionType {
    type Err = UnsupportedQuestionType;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "single-session-user" => Ok(Self::SingleSessionUser),
            "single-session-assistant" => Ok(Self::SingleSessionAssistant),
            "single-session-preference" => Ok(Self::SingleSessionPreference),
            "multi-session" => Ok(Self::MultiSession),
            "temporal-reasoning" => Ok(Self::TemporalReasoning),
            "knowledge-update" => Ok(Self::KnowledgeUpdate),
            _ => Err(UnsupportedQuestionType {
                value: value.to_owned(),
            }),
        }
    }
}

/// Whether the answer model receives chronological source messages or compressed memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerMemoryKind {
    /// Uncompressed source messages rendered in chronological order.
    Chronological,
    /// Query-conditioned compressed chronological memory.
    Compressed,
}

/// Inputs visible to the common answer model.
///
/// There is intentionally no reference-answer or gold-annotation field here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerPromptInput {
    /// User question to answer.
    pub question: String,
    /// Optional time at which the question is asked.
    #[serde(default)]
    pub query_time: String,
    /// Method-produced memory visible to the answer model.
    pub memory: String,
    /// Interpretation used for the memory and empty-memory fallback.
    pub memory_kind: AnswerMemoryKind,
}

/// Frozen two-message prompt sent to the answer backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnswerPrompt {
    /// Frozen system instruction.
    pub system: String,
    /// Rendered query and memory payload.
    pub user: String,
}

/// Renders the same answer prompt for every compared memory method.
pub fn render_answer_prompt(input: &AnswerPromptInput) -> AnswerPrompt {
    let compressed = input.memory_kind == AnswerMemoryKind::Compressed;
    let mut lines = vec![
        if compressed {
            "You are answering a user question using compressed chronological conversation memory."
        } else {
            "You are answering a user question using chronological conversation memory."
        }
        .to_owned(),
        "Use the memory when it is relevant. If the memory is insufficient, say what is missing instead of inventing facts."
            .to_owned(),
        String::new(),
        "# Query".to_owned(),
    ];
    if !input.query_time.is_empty() {
        lines.push(format!("Query Time: {}", input.query_time));
    }
    lines.extend([
        format!("Query: {}", input.question),
        String::new(),
        "# Memory".to_owned(),
        match input.memory.trim() {
            "" if compressed => "(No kept memory was found.)".to_owned(),
            "" => "(No gold memory was found.)".to_owned(),
            memory => memory.to_owned(),
        },
        String::new(),
        "# Answer".to_owned(),
    ]);
    AnswerPrompt {
        system: ANSWER_SYSTEM_PROMPT.to_owned(),
        user: lines.join("\n"),
    }
}

/// One model prediction persisted before judge evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Stable LongMemEval question identifier.
    pub question_id: String,
    /// Released LongMemEval question category.
    pub question_type: LongMemEvalQuestionType,
    /// Whether dataset metadata marks the question as unanswerable.
    pub is_abstention: bool,
    /// Answer-model output to be judged.
    pub prediction: String,
}

impl PredictionRecord {
    /// Constructs a versioned prediction record.
    pub fn new(
        question_id: impl Into<String>,
        question_type: LongMemEvalQuestionType,
        is_abstention: bool,
        prediction: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: LONGMEMEVAL_EVAL_SCHEMA_VERSION,
            question_id: question_id.into(),
            question_type,
            is_abstention,
            prediction: prediction.into(),
        }
    }

    /// Mirrors the Python evaluator: metadata or the `_abs` ID suffix marks abstention.
    pub fn evaluates_as_abstention(&self) -> bool {
        self.is_abstention || self.question_id.ends_with("_abs")
    }
}

/// The five distinct official judge-template families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgePromptKind {
    /// Shared factual template for user, assistant, and multi-session questions.
    General,
    /// Temporal template with the official off-by-one tolerance.
    Temporal,
    /// Update template that accepts prior information beside the latest value.
    KnowledgeUpdate,
    /// Personalized-response rubric template.
    Preference,
    /// Unanswerable-question template.
    Abstention,
}

/// Rendered judge prompt plus its auditable template family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgePrompt {
    /// Template family selected for this question.
    pub kind: JudgePromptKind,
    /// Fully rendered official judge prompt.
    pub text: String,
}

/// Renders the official LongMemEval judge prompt for one prediction.
///
/// The reference answer is confined to this evaluator-side function and never
/// enters [`AnswerPromptInput`].  Abstention predictions use the explanation
/// template and remain part of the full 500-question dataset or 100-question
/// test evaluation.
pub fn render_judge_prompt(
    prediction: &PredictionRecord,
    question: &str,
    reference_answer: &str,
) -> JudgePrompt {
    let (kind, template) = if prediction.evaluates_as_abstention() {
        (JudgePromptKind::Abstention, ABSTENTION_JUDGE_TEMPLATE)
    } else {
        match prediction.question_type {
            LongMemEvalQuestionType::SingleSessionUser
            | LongMemEvalQuestionType::SingleSessionAssistant
            | LongMemEvalQuestionType::MultiSession => {
                (JudgePromptKind::General, GENERAL_JUDGE_TEMPLATE)
            }
            LongMemEvalQuestionType::TemporalReasoning => {
                (JudgePromptKind::Temporal, TEMPORAL_JUDGE_TEMPLATE)
            }
            LongMemEvalQuestionType::KnowledgeUpdate => (
                JudgePromptKind::KnowledgeUpdate,
                KNOWLEDGE_UPDATE_JUDGE_TEMPLATE,
            ),
            LongMemEvalQuestionType::SingleSessionPreference => {
                (JudgePromptKind::Preference, PREFERENCE_JUDGE_TEMPLATE)
            }
        }
    };
    JudgePrompt {
        kind,
        text: interpolate_judge_template(
            template,
            question,
            reference_answer,
            &prediction.prediction,
        ),
    }
}

fn interpolate_judge_template(
    template: &str,
    question: &str,
    reference_answer: &str,
    prediction: &str,
) -> String {
    let values = [question, reference_answer, prediction];
    let pieces = template.split("{}").collect::<Vec<_>>();
    debug_assert_eq!(pieces.len(), values.len() + 1);
    let capacity = pieces.iter().map(|piece| piece.len()).sum::<usize>()
        + values.iter().map(|value| value.len()).sum::<usize>();
    let mut rendered = String::with_capacity(capacity);
    for (piece, value) in pieces.iter().zip(values) {
        rendered.push_str(piece);
        rendered.push_str(value);
    }
    rendered.push_str(pieces.last().expect("judge template has a trailing piece"));
    rendered
}

/// Applies the upstream rule exactly: Unicode lowercase, then substring `yes`.
pub fn judge_response_is_correct(raw_response: &str) -> bool {
    raw_response.to_lowercase().contains("yes")
}

/// Serializable result of applying the official decision rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeScore {
    /// Record schema version.
    pub schema_version: u32,
    /// Stable LongMemEval question identifier.
    pub question_id: String,
    /// Metric identifier used by report aggregation.
    pub metric: String,
    /// Judge template family used for the question.
    pub prompt_kind: JudgePromptKind,
    /// Boolean result of the official decision rule.
    pub correct: bool,
    /// Numeric form of `correct`, either zero or one.
    pub score: f64,
    /// Unmodified text returned by the judge model.
    pub raw_response: String,
    /// Human-readable description of the applied decision rule.
    pub decision_rule: String,
}

/// Scores a raw judge response without performing any model or network call.
pub fn score_judge_response(
    question_id: impl Into<String>,
    prompt_kind: JudgePromptKind,
    raw_response: impl Into<String>,
) -> JudgeScore {
    let raw_response = raw_response.into();
    let correct = judge_response_is_correct(&raw_response);
    JudgeScore {
        schema_version: LONGMEMEVAL_EVAL_SCHEMA_VERSION,
        question_id: question_id.into(),
        metric: LONGMEMEVAL_OFFICIAL_LJ_METRIC.to_owned(),
        prompt_kind,
        correct,
        score: f64::from(correct),
        raw_response,
        decision_rule: LONGMEMEVAL_JUDGE_DECISION_RULE.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn prediction(question_type: LongMemEvalQuestionType, is_abstention: bool) -> PredictionRecord {
        PredictionRecord::new("question", question_type, is_abstention, "model response")
    }

    #[test]
    fn judge_templates_match_the_frozen_python_provenance_hash() {
        let templates = [
            GENERAL_JUDGE_TEMPLATE,
            TEMPORAL_JUDGE_TEMPLATE,
            KNOWLEDGE_UPDATE_JUDGE_TEMPLATE,
            PREFERENCE_JUDGE_TEMPLATE,
            ABSTENTION_JUDGE_TEMPLATE,
        ]
        .join("\n");
        assert_eq!(
            hex::encode(Sha256::digest(templates.as_bytes())),
            "84d4a15455723ba860545bb7be7f3ff44d9d8dbe894fcf1d2d4b7be7eca32b6f"
        );
    }

    #[test]
    fn renders_all_five_official_template_families() {
        let general = render_judge_prompt(
            &prediction(LongMemEvalQuestionType::SingleSessionUser, false),
            "question text",
            "reference",
        );
        assert_eq!(general.kind, JudgePromptKind::General);
        assert!(general.text.contains("Correct Answer: reference"));

        for question_type in [
            LongMemEvalQuestionType::SingleSessionAssistant,
            LongMemEvalQuestionType::MultiSession,
        ] {
            assert_eq!(
                render_judge_prompt(&prediction(question_type, false), "q", "r").kind,
                JudgePromptKind::General
            );
        }

        let temporal = render_judge_prompt(
            &prediction(LongMemEvalQuestionType::TemporalReasoning, false),
            "q",
            "r",
        );
        assert_eq!(temporal.kind, JudgePromptKind::Temporal);
        assert!(temporal.text.contains("do not penalize off-by-one errors"));

        let update = render_judge_prompt(
            &prediction(LongMemEvalQuestionType::KnowledgeUpdate, false),
            "q",
            "r",
        );
        assert_eq!(update.kind, JudgePromptKind::KnowledgeUpdate);
        assert!(update.text.contains("some previous information"));

        let preference = render_judge_prompt(
            &prediction(LongMemEvalQuestionType::SingleSessionPreference, false),
            "q",
            "rubric",
        );
        assert_eq!(preference.kind, JudgePromptKind::Preference);
        assert!(preference.text.contains("Rubric: rubric"));

        let mut abstention = prediction(LongMemEvalQuestionType::MultiSession, false);
        abstention.question_id = "question_abs".to_owned();
        let abstention = render_judge_prompt(&abstention, "q", "explanation");
        assert_eq!(abstention.kind, JudgePromptKind::Abstention);
        assert!(abstention.text.contains("Explanation: explanation"));
        assert!(abstention.text.contains("unanswerable"));
    }

    #[test]
    fn applies_the_official_yes_substring_rule() {
        assert!(judge_response_is_correct("YES"));
        assert!(judge_response_is_correct("The answer is yEs."));
        assert!(judge_response_is_correct("yesterday"));
        assert!(!judge_response_is_correct("no"));
        assert!(!judge_response_is_correct("correct"));
    }

    #[test]
    fn rejects_unknown_question_types() {
        let error = "future-question-type"
            .parse::<LongMemEvalQuestionType>()
            .unwrap_err();
        assert_eq!(error.value, "future-question-type");
        assert!(error.to_string().contains("unsupported LongMemEval"));

        assert!(
            serde_json::from_str::<PredictionRecord>(
                r#"{"schema_version":1,"question_id":"q","question_type":"future-question-type","is_abstention":false,"prediction":"p"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn reference_is_confined_to_the_judge_prompt() {
        let reference = "REFERENCE_MUST_NEVER_REACH_THE_ANSWER_MODEL";
        let answer = render_answer_prompt(&AnswerPromptInput {
            question: "What happened?".to_owned(),
            query_time: "2026/08/08".to_owned(),
            memory: "A chronological fact.".to_owned(),
            memory_kind: AnswerMemoryKind::Compressed,
        });
        assert!(!answer.system.contains(reference));
        assert!(!answer.user.contains(reference));

        let judge = render_judge_prompt(
            &prediction(LongMemEvalQuestionType::SingleSessionUser, false),
            "What happened?",
            reference,
        );
        assert!(judge.text.contains(reference));
    }

    #[test]
    fn answer_prompt_matches_the_frozen_python_layout() {
        let prompt = render_answer_prompt(&AnswerPromptInput {
            question: "Where?".to_owned(),
            query_time: "today".to_owned(),
            memory: "  useful memory  ".to_owned(),
            memory_kind: AnswerMemoryKind::Compressed,
        });
        assert_eq!(prompt.system, ANSWER_SYSTEM_PROMPT);
        assert_eq!(
            prompt.user,
            "You are answering a user question using compressed chronological conversation memory.\nUse the memory when it is relevant. If the memory is insufficient, say what is missing instead of inventing facts.\n\n# Query\nQuery Time: today\nQuery: Where?\n\n# Memory\nuseful memory\n\n# Answer"
        );

        let empty = render_answer_prompt(&AnswerPromptInput {
            question: "Where?".to_owned(),
            query_time: String::new(),
            memory: " \n ".to_owned(),
            memory_kind: AnswerMemoryKind::Chronological,
        });
        assert!(empty.user.contains("(No gold memory was found.)"));
        assert!(!empty.user.contains("Query Time:"));
    }

    #[test]
    fn prediction_and_score_records_round_trip_through_json() {
        let prediction = prediction(LongMemEvalQuestionType::KnowledgeUpdate, false);
        let prediction_json = serde_json::to_string(&prediction).unwrap();
        assert_eq!(
            serde_json::from_str::<PredictionRecord>(&prediction_json).unwrap(),
            prediction
        );

        let score = score_judge_response(
            prediction.question_id.clone(),
            JudgePromptKind::KnowledgeUpdate,
            "yes",
        );
        assert!(score.correct);
        assert_eq!(score.score, 1.0);
        let score_json = serde_json::to_string(&score).unwrap();
        assert_eq!(
            serde_json::from_str::<JudgeScore>(&score_json).unwrap(),
            score
        );
    }
}
