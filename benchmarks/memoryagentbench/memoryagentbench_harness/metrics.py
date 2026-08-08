from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import re
import string

from .model import HarnessError


def normalize_answer(text: str) -> str:
    """Match the pinned reference implementation's DRQA normalization."""

    if not isinstance(text, str):
        raise HarnessError("prediction and ground truth must be strings")
    normalized = text.lower()
    normalized = "".join(char for char in normalized if char not in string.punctuation)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def parse_output(text: str, answer_prefix: str = "Answer:") -> str | None:
    for pattern in (
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ):
        match = pattern.search(text)
        if match:
            extracted = match[1].strip()
            return re.sub(
                f"^{re.escape(answer_prefix)}",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()
    return None


def substring_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    special = {"yes", "no", "noanswer"}
    if (
        normalize_answer(prediction) in special
        or normalize_answer(ground_truth) in special
    ) and normalize_answer(prediction) != normalize_answer(ground_truth):
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _max_over_answers(
    prediction: str,
    answers: Iterable[str],
    scorer,
) -> float:
    values = list(answers)
    if not values:
        raise HarnessError("answers cannot be empty")
    return float(max(scorer(prediction, answer) for answer in values))


def official_answer_metrics(
    prediction: str,
    answers: list[str],
    *,
    source: str | None = None,
) -> dict[str, float]:
    """Reproduce the pinned source-specific answer post-processing.

    EventQA evaluates only ``parse_output(prediction)``. Ruler and
    FactConsolidation use ``default_post_process``, which takes the maximum of
    raw and parsed predictions.
    """

    parsed = parse_output(prediction)
    if source is not None and source.startswith("eventqa_"):
        if parsed is None:
            raise HarnessError("EventQA prediction could not be parsed")
        variants = [parsed]
    else:
        variants = [prediction]
        if parsed is not None:
            variants.append(parsed)
    return {
        "substring_exact_match": max(
            _max_over_answers(value, answers, substring_exact_match) for value in variants
        ),
        "exact_match": max(
            _max_over_answers(value, answers, exact_match) for value in variants
        ),
        "f1": max(_max_over_answers(value, answers, token_f1) for value in variants),
    }


def answer_coverage(retrieved_text: str, answers: list[str]) -> float:
    """Non-official diagnostic: any gold answer appears in retrieved memory text."""

    return answer_coverage_against_normalized(normalize_answer(retrieved_text), answers)


def answer_coverage_against_normalized(
    normalized_retrieved_text: str, answers: list[str]
) -> float:
    """Answer coverage when the memory text has already been normalized."""

    return float(
        any(normalize_answer(answer) in normalized_retrieved_text for answer in answers)
    )
