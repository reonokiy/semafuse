from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from semafuse_lazymem_bench.endpoints import CompletionResult
from semafuse_lazymem_bench.official_evaluators import (
    LongMemEvalOfficialLJ,
    external_scores,
    locomo_official_f1,
    longmemeval_official_prompt,
)
from semafuse_lazymem_bench.types import Question


class _Client:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return CompletionResult("Yes.", 0.25, metadata={"model": "deepseek-v4-pro"})


class OfficialEvaluatorTests(unittest.TestCase):
    def test_longmemeval_uses_official_prompt_and_substring_rule(self) -> None:
        question = Question(
            "q",
            "longmemeval",
            "temporal-reasoning",
            "How many days?",
            "18 days",
            "",
            (),
        )
        prompt = longmemeval_official_prompt(question, "19 days")
        self.assertIn("predicting 19 days when the answer is 18", prompt)
        client = _Client()
        result = LongMemEvalOfficialLJ(client).evaluate(question, "19 days")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metric, "longmemeval_official_lj")
        messages, kwargs = client.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": prompt}])
        self.assertEqual(kwargs, {"temperature": 0.0, "max_tokens": 10, "extra_body": {"n": 1}})

    def test_locomo_category_aware_f1_is_not_lj(self) -> None:
        question = Question(
            "q",
            "locomo",
            "multi-hop",
            "q",
            "red apples, blue cars",
            "",
            (),
            metadata={"category": 1},
        )
        result = locomo_official_f1(
            question,
            "blue car, red apple",
            stemmer=lambda token: token.removesuffix("s"),
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metric, "locomo_official_f1")
        self.assertFalse(result.metadata["paper_comparable_lj"])

    def test_external_scores_require_exact_schema_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(
                json.dumps({"question_id": "q1", "score": True}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(external_scores(str(path)), {"q1": 1.0})
            path.write_text(
                json.dumps({"question_id": "q1", "score": 1, "extra": 0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly"):
                external_scores(str(path))


if __name__ == "__main__":
    unittest.main()
