from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from semafuse_lazymem_bench.cli import main
from semafuse_lazymem_bench.adapters.locomo import (
    LoCoMoLicenseWarning,
    _evidence_ids,
    load_locomo,
)
from semafuse_lazymem_bench.adapters.longmemeval import (
    adapt_longmemeval,
    load_longmemeval,
)
from semafuse_lazymem_bench.split import stratified_split
from semafuse_lazymem_bench.types import Question


FIXTURES = Path(__file__).parent / "fixtures"


class AdapterTests(unittest.TestCase):
    def test_adapt_cli_emits_canonical_compact_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "normalized.jsonl"
            self.assertEqual(
                main(
                    [
                        "adapt",
                        "--dataset",
                        "longmemeval",
                        "--input",
                        str(FIXTURES / "longmemeval.json"),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            for line in output.read_text(encoding="utf-8").splitlines():
                self.assertEqual(
                    line,
                    json.dumps(
                        json.loads(line),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )

    def test_longmemeval_normalizes_messages_and_gold(self) -> None:
        questions = load_longmemeval(FIXTURES / "longmemeval.json")
        self.assertEqual(len(questions), 2)
        first = questions[0]
        self.assertEqual(first.gold_message_ids, ("s1_01",))
        self.assertEqual(first.gold_session_ids, ("s1",))
        self.assertEqual([message.index for message in first.messages], [0, 1, 2, 3])
        self.assertTrue(first.messages[0].retrieval_text.startswith("User: "))
        self.assertTrue(questions[1].metadata["is_abstention"])

    def test_longmemeval_retains_empty_messages_for_exact_retrieval_corpus(self) -> None:
        question = adapt_longmemeval(
            [
                {
                    "question_id": "empty-message",
                    "question_type": "single-session-user",
                    "question": "q",
                    "answer": "a",
                    "answer_session_ids": ["native"],
                    "haystack_dates": ["2024/01/01"],
                    "haystack_session_ids": ["native"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": ""},
                            {"role": "assistant", "content": "answer", "has_answer": True},
                        ]
                    ],
                }
            ]
        )[0]
        self.assertEqual(
            [message.id for message in question.messages],
            ["native_01", "native_02"],
        )
        self.assertEqual([message.index for message in question.messages], [0, 1])
        self.assertEqual([message.global_message_index for message in question.messages], [0, 1])
        self.assertEqual(question.gold_message_ids, ("native_02",))
        self.assertEqual(question.metadata["empty_message_ids_present"], ["native_01"])

    def test_longmemeval_disambiguates_repeated_native_session_ids(self) -> None:
        question = adapt_longmemeval(
            [
                {
                    "question_id": "duplicate-session",
                    "question_type": "multi-session",
                    "question": "q",
                    "question_date": "2024/01/03",
                    "answer": "a",
                    "answer_session_ids": [],
                    "haystack_dates": ["2024/01/01", "2024/01/02"],
                    "haystack_session_ids": ["native", "native"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "first"}],
                        [{"role": "user", "content": "second"}],
                    ],
                }
            ]
        )[0]
        self.assertEqual(
            [message.id for message in question.messages],
            ["native#occurrence-0_01", "native#occurrence-1_01"],
        )
        self.assertEqual(
            [message.metadata["native_session_id"] for message in question.messages],
            ["native", "native"],
        )

    def test_locomo_requires_license_ack_and_exact_subset(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LoCoMoLicenseWarning)
            with self.assertRaises(PermissionError):
                load_locomo(FIXTURES / "locomo.json")
            questions = load_locomo(
                FIXTURES / "locomo.json",
                acknowledge_noncommercial=True,
                require_paper_count=False,
            )
        self.assertEqual(len(questions), 4)
        self.assertEqual(
            {question.metadata["conversation_id"] for question in questions},
            {"conv-49", "conv-50"},
        )
        self.assertEqual(
            {question.question_type for question in questions},
            {"single-hop", "temporal", "multi-hop", "open-domain"},
        )
        self.assertTrue(all(question.metadata["license"] == "CC-BY-NC-4.0" for question in questions))
        self.assertIn("Image caption: a basket of red apples", questions[0].messages[0].content)
        self.assertIn(questions[0].messages[0].role, {"user", "assistant"})
        self.assertEqual(
            questions[0].messages[0].retrieval_text,
            questions[0].messages[0].content,
        )

    def test_locomo_normalizes_packed_and_zero_padded_evidence(self) -> None:
        self.assertEqual(
            _evidence_ids(
                ["D9:1 D4:4 D4:6", "D30:05"],
                {"D9:1", "D4:4", "D4:6", "D30:5"},
            ),
            ("D9:1", "D4:4", "D4:6", "D30:5"),
        )


class SplitTests(unittest.TestCase):
    COUNTS = {
        "single-session-user": 70,
        "single-session-assistant": 56,
        "single-session-preference": 30,
        "multi-session": 133,
        "temporal-reasoning": 133,
        "knowledge-update": 78,
    }

    def _questions(self) -> list[Question]:
        return [
            Question(
                id=f"{question_type}:{index:03d}",
                dataset="longmemeval",
                question_type=question_type,
                query="q",
                answer="a",
                query_time="",
                messages=(),
            )
            for question_type, count in self.COUNTS.items()
            for index in range(count)
        ]

    def test_paper_allocation_hash_and_disjointness(self) -> None:
        first = stratified_split(self._questions(), source_sha256="abc")
        second = stratified_split(reversed(self._questions()), source_sha256="abc")
        self.assertEqual(first.ids_sha256, second.ids_sha256)
        self.assertEqual(
            first.counts_by_type["multi-session"],
            {"train": 96, "validation": 10, "test": 27},
        )
        self.assertEqual(
            first.counts_by_type["temporal-reasoning"],
            {"train": 96, "validation": 11, "test": 26},
        )
        self.assertEqual([len(first.ids[name]) for name in ("train", "validation", "test")], [360, 40, 100])
        sets = [set(first.ids[name]) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 500)

if __name__ == "__main__":
    unittest.main()
