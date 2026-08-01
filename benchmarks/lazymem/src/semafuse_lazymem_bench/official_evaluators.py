"""Benchmark-native evaluators and their pinned public provenance."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .endpoints import ChatCompletionClient
from .types import Question


LONGMEMEVAL_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
LONGMEMEVAL_EVALUATOR_SHA256 = (
    "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
)
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SCORER_SHA256 = (
    "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
)


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    score: float
    metric: str
    raw: str | None
    latency_s: float
    metadata: Mapping[str, Any]


_GENERAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: "
    "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: "
    "{}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)
_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the "
    "model correctly identify the question as unanswerable? Answer yes or no only."
)
_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    "\n".join(
        (
            _GENERAL_TEMPLATE,
            _TEMPORAL_TEMPLATE,
            _UPDATE_TEMPLATE,
            _PREFERENCE_TEMPLATE,
            _ABSTENTION_TEMPLATE,
        )
    ).encode("utf-8")
).hexdigest()


def longmemeval_official_prompt(question: Question, prediction: str) -> str:
    """Render the MIT-licensed upstream LongMemEval answer-check prompt."""

    if question.metadata.get("is_abstention") or question.id.endswith("_abs"):
        template = _ABSTENTION_TEMPLATE
    elif question.question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = _GENERAL_TEMPLATE
    elif question.question_type == "temporal-reasoning":
        template = _TEMPORAL_TEMPLATE
    elif question.question_type == "knowledge-update":
        template = _UPDATE_TEMPLATE
    elif question.question_type == "single-session-preference":
        template = _PREFERENCE_TEMPLATE
    else:
        raise ValueError(
            f"unsupported LongMemEval question type {question.question_type!r}"
        )
    return template.format(question.query, question.answer, prediction)


class LongMemEvalOfficialLJ:
    """Official LongMemEval prompt/rule with a configurable paper judge model."""

    metric = "longmemeval_official_lj"

    def __init__(self, client: ChatCompletionClient) -> None:
        self.client = client

    def evaluate(self, question: Question, prediction: str) -> EvaluationScore:
        if question.dataset != "longmemeval":
            raise ValueError("LongMemEval evaluator received another dataset")
        prompt = longmemeval_official_prompt(question, prediction)
        response = self.client.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            extra_body={"n": 1},
        )
        # This deliberately preserves the upstream decision rule.
        correct = "yes" in response.text.lower()
        return EvaluationScore(
            score=float(correct),
            metric=self.metric,
            raw=response.text,
            latency_s=response.latency_s,
            metadata={
                "upstream": "xiaowu0162/LongMemEval",
                "upstream_commit": LONGMEMEVAL_COMMIT,
                "upstream_evaluator_sha256": LONGMEMEVAL_EVALUATOR_SHA256,
                "prompt_templates_sha256": _PROMPT_TEMPLATE_SHA256,
                "decision_rule": "substring 'yes' after Unicode lowercase",
                "temperature": 0.0,
                "max_tokens": 10,
                "response_model": response.metadata.get("model"),
            },
        )


def _normalize_locomo(text: str) -> str:
    value = text.replace(",", "").lower()
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(value.split())


def _locomo_f1_pair(
    prediction: str, ground_truth: str, stem: Callable[[str], str]
) -> float:
    predicted = [stem(token) for token in _normalize_locomo(prediction).split()]
    expected = [stem(token) for token in _normalize_locomo(ground_truth).split()]
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def locomo_official_f1(
    question: Question,
    prediction: str,
    *,
    stemmer: Callable[[str], str] | None = None,
) -> EvaluationScore:
    """Run LoCoMo's released category-aware token-F1 scorer.

    This is the original benchmark metric, not the undocumented binary LJ used
    for the LazyMem paper's LoCoMo table.
    """

    if question.dataset != "locomo":
        raise ValueError("LoCoMo evaluator received another dataset")
    if stemmer is None:
        try:
            from nltk.stem import PorterStemmer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'locomo-eval' extra for official F1") from error
        stemmer = PorterStemmer().stem
    category = int(question.metadata["category"])
    answer = question.answer
    if category == 3:
        answer = answer.split(";", 1)[0].strip()
    if category == 1:
        predicted_parts = [part.strip() for part in prediction.split(",")]
        answer_parts = [part.strip() for part in answer.split(",")]
        component_scores = [
            max(_locomo_f1_pair(part, expected, stemmer) for part in predicted_parts)
            for expected in answer_parts
        ]
        score = statistics.fmean(component_scores)
    elif category in {2, 3, 4}:
        score = _locomo_f1_pair(prediction, answer, stemmer)
    else:
        raise ValueError("paper LoCoMo subset supports only categories 1--4")
    return EvaluationScore(
        score=score,
        metric="locomo_official_f1",
        raw=None,
        latency_s=0.0,
        metadata={
            "upstream": "snap-research/locomo",
            "upstream_commit": LOCOMO_COMMIT,
            "upstream_scorer_sha256": LOCOMO_SCORER_SHA256,
            "license": "CC-BY-NC-4.0",
            "paper_comparable_lj": False,
        },
    )


def external_scores(path: str) -> dict[str, float]:
    """Load strict externally produced scores keyed by question ID."""

    scores: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != {"question_id", "score"}:
                raise ValueError(
                    f"external score line {line_number} must contain exactly question_id and score"
                )
            question_id = value["question_id"]
            score = value["score"]
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"external score line {line_number} has invalid question_id")
            if isinstance(score, bool):
                numeric_score = float(score)
            elif isinstance(score, (int, float)):
                numeric_score = float(score)
            else:
                raise TypeError(f"external score line {line_number} score must be numeric")
            if not 0.0 <= numeric_score <= 1.0:
                raise ValueError(f"external score line {line_number} is outside [0, 1]")
            if question_id in scores:
                raise ValueError(f"duplicate external score for {question_id}")
            scores[question_id] = numeric_score
    return scores
