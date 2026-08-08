from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memoryagentbench_harness.chunking import official_sentence_chunks
from memoryagentbench_harness.metrics import (
    answer_coverage,
    answer_coverage_against_normalized,
    normalize_answer,
    official_answer_metrics,
)
from memoryagentbench_harness.model import HarnessError, Manifest, load_manifest
from memoryagentbench_harness.prompts import official_bm25_query, query_message
from memoryagentbench_harness.protocol import prepare_source, read_jsonl
from memoryagentbench_harness.scoring import score_predictions
from memoryagentbench_harness.source_verification import download_and_verify_files


class MetricTests(unittest.TestCase):
    def test_official_normalization_and_substring_metric(self) -> None:
        self.assertEqual(normalize_answer("The Cobalt-Key!"), "cobaltkey")
        metrics = official_answer_metrics(
            "Reasoning first.\nAnswer: The cobalt-key!\nIgnored tail",
            ["cobalt key", "cobalt-key"],
        )
        self.assertEqual(metrics["substring_exact_match"], 1.0)

    def test_incorrect_prediction_scores_zero(self) -> None:
        metrics = official_answer_metrics("Answer: amber lock", ["cobalt key"])
        self.assertEqual(metrics["substring_exact_match"], 0.0)
        self.assertEqual(metrics["exact_match"], 0.0)

    def test_eventqa_uses_parsed_prediction_only(self) -> None:
        metrics = official_answer_metrics(
            "cobalt key appears in reasoning\nAnswer: amber lock",
            ["cobalt key"],
            source="eventqa_full",
        )
        self.assertEqual(metrics["substring_exact_match"], 0.0)

    def test_pre_normalized_coverage_is_equivalent(self) -> None:
        text = "The Cobalt-Key opens the archive."
        answers = ["cobalt key", "amber lock"]
        self.assertEqual(
            answer_coverage(text, answers),
            answer_coverage_against_normalized(normalize_answer(text), answers),
        )


class ChunkingTests(unittest.TestCase):
    def test_matches_upstream_sentence_packing(self) -> None:
        chunks = official_sentence_chunks(
            "one. two words. three.",
            3,
            sentence_splitter=lambda _: ["one.", "two words.", "three."],
            token_counter=lambda sentence: len(sentence.split()),
        )
        self.assertEqual(chunks, ["one. two words.", "three."])

    def test_preserves_upstream_oversized_first_sentence_behavior(self) -> None:
        chunks = official_sentence_chunks(
            "oversized sentence",
            1,
            sentence_splitter=lambda text: [text],
            token_counter=lambda _: 2,
        )
        self.assertEqual(chunks, ["", "oversized sentence"])


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest()
        self.fixture = ROOT / "fixtures" / "mini_rows.json"

    def _prepare(self, directory: Path, source: str = "eventqa_full") -> dict:
        return prepare_source(
            self.manifest,
            self.manifest.profiles[source],
            directory,
            fixture_path=self.fixture,
            chunk_size=8,
            sentence_splitter=lambda text: [
                sentence.strip() + "."
                for sentence in text.split(".")
                if sentence.strip()
            ],
            token_counter=lambda sentence: len(sentence.split()),
        )

    def test_prepare_is_label_isolated_and_deterministic(self) -> None:
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            first_run = self._prepare(Path(first))
            second_run = self._prepare(Path(second))
            self.assertEqual(first_run["input_sha256"], second_run["input_sha256"])
            self.assertEqual(first_run["labels_sha256"], second_run["labels_sha256"])
            self.assertFalse(first_run["paper_comparable_shape"])

            events = read_jsonl(Path(first) / "inputs.jsonl")
            labels = read_jsonl(Path(first) / "labels.jsonl")
            self.assertEqual(sum(event["op"] == "query" for event in events), 2)
            self.assertEqual(len(labels), 2)
            forbidden = {
                "answer",
                "answers",
                "label",
                "labels",
                "ground_truth",
                "qa_pair_id",
                "upstream_qa_pair_id",
            }
            for event in events:
                self.assertTrue(forbidden.isdisjoint(event))
            self.assertIn("answers", labels[0])

    def test_scoring_separates_official_and_retrieval_diagnostic(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._prepare(directory)
            events = read_jsonl(directory / "inputs.jsonl")
            labels = read_jsonl(directory / "labels.jsonl")
            memory_ids = [event["memory_id"] for event in events if event["op"] == "add"]
            predictions = directory / "predictions.jsonl"
            rows = [
                {
                    "case_id": label["case_id"],
                    "query_id": label["query_id"],
                    "prediction": label["answers"][0] if index == 0 else "wrong",
                    "retrieved_memory_ids": memory_ids,
                }
                for index, label in enumerate(labels)
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = score_predictions(
                directory / "inputs.jsonl",
                directory / "labels.jsonl",
                predictions,
            )
            self.assertEqual(report["metrics_percent"]["substring_exact_match"], 50.0)
            self.assertEqual(report["metrics_percent"]["answer_coverage_at_k"], 100.0)
            self.assertEqual(
                report["metrics_percent"]["answer_surface_oracle_coverage"], 100.0
            )
            self.assertEqual(
                report["metrics_percent"]["answer_coverage_given_surface_present"],
                100.0,
            )
            self.assertIn("non-official", report["comparability"]["answer_coverage_at_k"])

    def test_unknown_retrieval_id_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._prepare(directory)
            labels = read_jsonl(directory / "labels.jsonl")
            predictions = directory / "predictions.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps(
                        {
                            "case_id": label["case_id"],
                            "query_id": label["query_id"],
                            "prediction": "wrong",
                            "retrieved_memory_ids": ["memory-not-in-case"],
                        }
                    )
                    + "\n"
                    for label in labels
                ),
                encoding="utf-8",
            )
            with self.assertRaises(HarnessError):
                score_predictions(
                    directory / "inputs.jsonl",
                    directory / "labels.jsonl",
                    predictions,
                )

    def test_missing_and_duplicate_predictions_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._prepare(directory)
            labels = read_jsonl(directory / "labels.jsonl")
            predictions = directory / "predictions.jsonl"
            one = {
                "case_id": labels[0]["case_id"],
                "query_id": labels[0]["query_id"],
                "prediction": "wrong",
            }
            predictions.write_text(json.dumps(one) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "missing"):
                score_predictions(
                    directory / "inputs.jsonl",
                    directory / "labels.jsonl",
                    predictions,
                )

            predictions.write_text(
                json.dumps(one) + "\n" + json.dumps(one) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(HarnessError, "duplicate"):
                score_predictions(
                    directory / "inputs.jsonl",
                    directory / "labels.jsonl",
                    predictions,
                    allow_partial=True,
                )

    def test_manifest_only_exposes_supported_training_free_sources(self) -> None:
        self.assertEqual(
            set(self.manifest.profiles),
            {
                "ruler_qa1_197K",
                "ruler_qa2_421K",
                "eventqa_full",
                "factconsolidation_sh_262k",
                "factconsolidation_mh_262k",
            },
        )
        self.assertEqual(len(self.manifest.dataset_revision), 40)
        self.assertEqual(self.manifest.top_k, 10)

    def test_paper_query_contains_question_but_not_answer(self) -> None:
        prompt = query_message("eventqa", "What happened next?")
        self.assertIn("What happened next?", prompt)
        self.assertNotIn("cobalt key", prompt)

    def test_bm25_query_extraction_matches_reference_patterns(self) -> None:
        prompt = query_message("ruler_qa", "Where is the archive?")
        self.assertEqual(official_bm25_query(prompt), "Where is the archive?")
        event_prompt = query_message("eventqa", "What happened next?")
        self.assertEqual(official_bm25_query(event_prompt), event_prompt)


class SourceVerificationTests(unittest.TestCase):
    def _manifest_for(self, path: Path) -> Manifest:
        contents = path.read_bytes()
        return Manifest(
            path=path,
            profiles={},
            raw={
                "dataset": {
                    "repo_id": "fixture/repo",
                    "revision": "a" * 40,
                    "license": "mit",
                    "split_rows": {"Accurate_Retrieval": 1},
                    "files": {
                        "data/Accurate_Retrieval-00000-of-00001.parquet": {
                            "bytes": len(contents),
                            "lfs_sha256": hashlib.sha256(contents).hexdigest(),
                        }
                    },
                }
            },
        )

    def test_downloaded_file_hash_is_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.parquet"
            path.write_bytes(b"pinned fixture bytes")
            manifest = self._manifest_for(path)
            calls = []

            def downloader(**kwargs):
                calls.append(kwargs)
                return str(path)

            report = download_and_verify_files(
                manifest,
                Path(temporary) / "cache",
                splits=["Accurate_Retrieval"],
                downloader=downloader,
            )
            self.assertTrue(report["all_verified"])
            self.assertEqual(
                report["files"][0]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(calls[0]["revision"], "a" * 40)

    def test_hash_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.parquet"
            path.write_bytes(b"expected")
            manifest = self._manifest_for(path)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(HarnessError, "mismatch"):
                download_and_verify_files(
                    manifest,
                    Path(temporary) / "cache",
                    splits=["Accurate_Retrieval"],
                    downloader=lambda **_: str(path),
                )


if __name__ == "__main__":
    unittest.main()
