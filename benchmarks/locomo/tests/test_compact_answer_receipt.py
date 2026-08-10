from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


LOCOMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCOMO_DIR))

import compact_answer_receipt as receipt  # noqa: E402


def opaque_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()}"


def canonical_line(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


class CompleteAnswerFixture:
    methods = (
        "amem_source_rag",
        "bm25",
        "mem0_source_rag",
        "semafuse",
        "semafuse::retrieved",
    )
    all_methods = {"bm25", "semafuse", "semafuse::retrieved"}
    imported_methods = {"amem_source_rag", "mem0_source_rag"}

    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_run_key = hashlib.sha256(b"fixture-memory-run").hexdigest()
        self.run_key = hashlib.sha256(b"fixture-answer-run").hexdigest()
        self.questions = [opaque_id("q", f"question-{index}") for index in range(314)]
        self.retrieval_ids = set(self.questions[:312])
        self.memory_envelopes: list[dict[str, object]] = []
        self.answer_envelopes: list[dict[str, object]] = []
        self.compact_answers: dict[str, dict[str, dict[str, object]]] = {
            method: {} for method in self.methods
        }
        self.write_artifacts()

    @staticmethod
    def score(method: str, index: int) -> float:
        baseline = (index % 5) / 5.0
        offsets = {
            "bm25": 0.0,
            "semafuse": 0.1,
            "semafuse::retrieved": 0.05,
            "mem0_source_rag": -0.05,
            "amem_source_rag": 0.02,
        }
        return max(0.0, min(1.0, baseline + offsets[method]))

    def processor_trace(self) -> dict[str, object]:
        call_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "reasoning_tokens": 0,
            "reasoning_present": False,
            "cost": 0.001,
        }
        return {
            "strategy_trace": {
                "strategy": {
                    "generation": {
                        "requested_sampling": {"enable_thinking": False}
                    },
                    "processing": {
                        "attempts": 1,
                        "validation_retries": 0,
                        "provider": "DeepInfra",
                        "providers": ["DeepInfra"],
                        "model": "qwen/qwen3-32b",
                        "models": ["qwen/qwen3-32b"],
                        "processor_calls": [
                            {
                                "attempts": 1,
                                "window_order": 0,
                                "calls": [
                                    {
                                        "attempt": 1,
                                        "model": "qwen/qwen3-32b",
                                        "provider": "DeepInfra",
                                        "finish_reason": "stop",
                                        "usage": call_usage,
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "call_semantics": (
                                "successful_validation_level_responses; "
                                "transport_retries_without_usage_are_not_counted"
                            ),
                            "calls": 1,
                            "calls_missing_completion_tokens": 0,
                            "calls_missing_cost": 0,
                            "calls_missing_prompt_tokens": 0,
                            "calls_missing_reasoning_tokens": 0,
                            "completion_tokens": 20,
                            "cost": 0.001,
                            "prompt_tokens": 100,
                            "reasoning_tokens": 0,
                        },
                    }
                }
            }
        }

    def memory_payload(self, method: str, index: int) -> dict[str, object]:
        question_id = self.questions[index]
        eligible = question_id in self.retrieval_ids
        category_number = index % 4 + 1
        category = {
            1: "multi-hop",
            2: "temporal",
            3: "open-domain",
            4: "single-hop",
        }[category_number]
        context = f"private fixture context {method} {index}"
        context_raw = context.encode()
        paid_processor = method == "semafuse"
        usage = {
            "prompt_tokens": 100 if paid_processor else 0,
            "completion_tokens": 20 if paid_processor else 0,
            "cost": 0.001 if paid_processor else None,
            "calls": 1 if paid_processor else 0,
        }
        message_id = opaque_id("m", f"message-{index}")
        return {
            "schema_version": 1,
            "run_key": self.memory_run_key,
            "method": method,
            "question_id": question_id,
            "category": category,
            "category_number": category_number,
            "retrieval_eligible": eligible,
            "ranked_message_ids": [message_id],
            "covered_message_ids": [message_id],
            "retained_message_ids": [message_id],
            "context": context,
            "context_sha256": hashlib.sha256(context_raw).hexdigest(),
            "context_bytes": len(context_raw),
            "context_tokens": None,
            "retrieval_recall_all_at_10": 1.0 if eligible else None,
            "retrieval_ndcg_at_10": 1.0 if eligible else None,
            "memory_wall_s": 0.01,
            "retrieval_s": 0.005,
            "processing_s": 0.002 if paid_processor else None,
            "shared_construction_s": None,
            "processor_usage": usage,
            "metadata": self.memory_metadata(method, index),
        }

    def memory_metadata(self, method: str, index: int) -> dict[str, object]:
        if method == "bm25":
            return {
                "kind": "bm25",
                "context_order": "source_chronological",
                "shared_execution": False,
            }
        history_sha = hashlib.sha256(f"history-{index}".encode()).hexdigest()
        if method in {"semafuse", "semafuse::retrieved"}:
            common = {
                "config_sha256": "2" * 64,
                "resolved_config_sha256": "3" * 64,
                "construction_amortization_n": 1,
                "history_identity_sha256": history_sha,
                "shared_execution": True,
                "source_method": "semafuse",
            }
            if method == "semafuse":
                return {
                    **common,
                    "kind": "semafuse_lazymem_full",
                    "strategy": "lazymem",
                    **self.processor_trace(),
                }
            return {
                **common,
                "kind": "shared_semafuse_stage_context",
                "stage": "retrieved",
                "usage_shared_with": "semafuse",
            }
        source = "amem" if method == "amem_source_rag" else "mem0"
        pins = (
            ("4" * 64, "5" * 64, "6" * 64)
            if source == "amem"
            else ("7" * 64, "8" * 64, "9" * 64)
        )
        return {
            "kind": "imported_source_ranking_rag",
            "cohort": "retrieval_eligible_312",
            "source_method": source,
            "source_run_key": pins[0],
            "source_journal_sha256": pins[1],
            "source_summary_sha256": pins[2],
            "source_ranking_limit": 50,
            "native_memory_reused": False,
            "context_order": "source_chronological",
            "shared_execution": False,
        }

    def answer_payload(
        self, method: str, index: int, memory: dict[str, object]
    ) -> dict[str, object]:
        prediction = f"private fixture prediction {method} {index}"
        score = self.score(method, index)
        payload = {
            "schema_version": 1,
            "run_key": self.run_key,
            "memory_run_key": self.memory_run_key,
            "method": method,
            "question_id": self.questions[index],
            "category": memory["category"],
            "category_number": memory["category_number"],
            "context_sha256": memory["context_sha256"],
            "prediction": prediction,
            "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
            "official_f1": {
                "schema_version": 1,
                "metric": receipt.METRIC,
                "score": score,
                "category": memory["category_number"],
                "category_name": memory["category"],
                "provenance": {
                    "upstream": "snap-research/locomo",
                    "upstream_commit": receipt.OFFICIAL_UPSTREAM_COMMIT,
                    "upstream_scorer_sha256": receipt.OFFICIAL_SCORER_SHA256,
                    "data_license": "CC-BY-NC-4.0",
                    "stemmer": "nltk-porter/0.1.0 Mode::Nltk",
                    "paper_comparable_lj": False,
                },
            },
            "answer_wall_s": 0.02,
            "answer_usage": {
                "prompt_tokens": 30,
                "completion_tokens": 5,
                "cost": 0.0001,
                "calls": 1,
            },
            "metadata": {
                "prompt_version": "semafuse-locomo-common-answer/v1",
                "response_model": "qwen/qwen3-30b-a3b-instruct-2507",
                "response_provider": "StreamLake",
                "response_finish_reason": "stop",
                "response_reasoning_tokens": 0,
                "response_reasoning_present": False,
            },
        }
        self.compact_answers[method][self.questions[index]] = {
            "score": score,
            "category": memory["category"],
        }
        return payload

    @staticmethod
    def envelope(payload: dict[str, object], run_key: str, kind: str) -> dict[str, object]:
        return {
            "run_key": run_key,
            "method": payload["method"],
            "question_id": payload["question_id"],
            "record_kind": kind,
            "payload": payload,
        }

    def manifest(self) -> dict[str, object]:
        outputs = [
            {
                "id": method,
                "cohort": "all_qa" if method in self.all_methods else "retrieval_eligible",
            }
            for method in sorted(self.methods)
        ]
        return {
            "schema_version": 1,
            "run_key": self.run_key,
            "memory_run_key": self.memory_run_key,
            "suite": "fixture",
            "track": receipt.TRACK,
            "complete": True,
            "paper_comparable": False,
            "paper_comparability_reason": "fixture",
            "benchmark": {
                "name": "LoCoMo",
                "upstream_commit": receipt.OFFICIAL_UPSTREAM_COMMIT,
                "dataset_sha256": receipt.OFFICIAL_DATASET_SHA256,
                "selection": {
                    "conversation_ids": ["conv-49", "conv-50"],
                    "categories": [1, 2, 3, 4],
                    "qa_n": 314,
                    "retrieval_eval_n": 312,
                },
            },
            "license": {
                "spdx": "CC-BY-NC-4.0",
                "noncommercial_use_acknowledged": True,
                "dataset_redistributed": False,
                "dataset_path_serialized": False,
                "raw_context_artifacts_publishable": False,
            },
            "protocol": {
                "common_answer_model": "qwen/qwen3-30b-a3b-instruct-2507",
                "common_answer_expected_provider": "StreamLake",
                "common_answer_endpoint": {
                    "model": "qwen/qwen3-30b-a3b-instruct-2507",
                    "extra_body": {
                        "provider": {
                            "allow_fallbacks": False,
                            "data_collection": "deny",
                            "max_price": {"prompt": 0.11, "completion": 0.31},
                            "order": ["StreamLake"],
                            "require_parameters": True,
                        }
                    },
                },
                "answer_prompt_version": "semafuse-locomo-common-answer/v1",
                "answer_prompt_sha256": "1" * 64,
                "answer_generation_options": {"enable_thinking": False},
                "official_f1": {
                    "metric": receipt.METRIC,
                    "upstream_commit": receipt.OFFICIAL_UPSTREAM_COMMIT,
                    "upstream_scorer_sha256": receipt.OFFICIAL_SCORER_SHA256,
                },
            },
            "execution": {
                "validation_only": False,
                "scheduled_question_n": 314,
                "question_limit": None,
                "question_id_filter_n": 0,
                "question_id_filter_sha256": None,
                "source_method_filter": [],
                "partial_runs_cannot_set_complete": True,
            },
            "denominators": {"qa_n": 314, "retrieval_eval_n": 312},
            "baseline_method": "bm25",
            "methods": [
                {"id": "bm25", "kind": "bm25", "limit": 50},
                {
                    "id": "semafuse",
                    "kind": "semafuse_full_memory_and_retrieval",
                    "limit": 50,
                    "outputs": ["semafuse", "semafuse::retrieved"],
                    "main_output": "semafuse",
                    "stage_outputs": ["semafuse::retrieved"],
                    "config_sha256": "2" * 64,
                    "resolved_config_sha256": "3" * 64,
                    "embedding_model": "fixture/embed",
                    "reranker_model": "fixture/rerank",
                    "model_revisions_verified": False,
                },
                {
                    "id": "amem_source_rag",
                    "kind": "imported_source_ranking_rag",
                    "limit": 50,
                    "source_method": "amem",
                    "source_run_key": "4" * 64,
                    "source_journal_sha256": "5" * 64,
                    "source_summary_sha256": "6" * 64,
                    "native_memory_reused": False,
                },
                {
                    "id": "mem0_source_rag",
                    "kind": "imported_source_ranking_rag",
                    "limit": 50,
                    "source_method": "mem0",
                    "source_run_key": "7" * 64,
                    "source_journal_sha256": "8" * 64,
                    "source_summary_sha256": "9" * 64,
                    "native_memory_reused": False,
                },
            ],
            "answer_outputs": outputs,
            "pricing_guard": {
                "input_usd_per_million": 0.1,
                "output_usd_per_million": 0.3,
                "max_reported_usage_usd": 5.0,
                "embedding_reranker_costs_included": False,
                "live_external_workers_allowed": False,
            },
            "implementation": {
                "crate_version": "0.1.0",
                "cargo_lock_sha256": "a" * 64,
                "benchmark_code_sha256": "b" * 64,
            },
            "limitations": ["fixture"],
            "credential_handling": "fixture",
        }

    def write_artifacts(self) -> None:
        for method in self.methods:
            indexes = range(314) if method in self.all_methods else range(312)
            for index in indexes:
                memory = self.memory_payload(method, index)
                answer = self.answer_payload(method, index, memory)
                self.memory_envelopes.append(
                    self.envelope(memory, self.memory_run_key, receipt.MEMORY_RECORD_KIND)
                )
                self.answer_envelopes.append(
                    self.envelope(answer, self.run_key, receipt.ANSWER_RECORD_KIND)
                )
        (self.root / "memory-contexts.jsonl").write_bytes(
            b"".join(canonical_line(value) for value in self.memory_envelopes)
        )
        (self.root / "answers.jsonl").write_bytes(
            b"".join(canonical_line(value) for value in self.answer_envelopes)
        )

        all_report = receipt._build_scalar_report(
            self.compact_answers, self.all_methods, self.questions, "bm25"
        )
        eligible_answers = {
            method: {
                question_id: self.compact_answers[method][question_id]
                for question_id in self.retrieval_ids
            }
            for method in self.methods
        }
        eligible_report = receipt._build_scalar_report(
            eligible_answers, self.methods, self.retrieval_ids, "bm25"
        )
        processor_usage = receipt._aggregate_usage(
            envelope["payload"]["processor_usage"] for envelope in self.memory_envelopes
        )
        answer_usage = receipt._aggregate_usage(
            envelope["payload"]["answer_usage"] for envelope in self.answer_envelopes
        )
        progress = {}
        for method in self.methods:
            memories = [
                envelope["payload"]
                for envelope in self.memory_envelopes
                if envelope["method"] == method
            ]
            answers = [
                envelope["payload"]
                for envelope in self.answer_envelopes
                if envelope["method"] == method
            ]
            cohort = "all_qa" if method in self.all_methods else "retrieval_eligible"
            required = 314 if cohort == "all_qa" else 312
            progress[method] = {
                "memory_n": required,
                "answer_n": required,
                "required_n": required,
                "cohort": cohort,
                "partial_mean_f1": sum(answer["official_f1"]["score"] for answer in answers)
                / required,
                "mean_context_bytes": sum(memory["context_bytes"] for memory in memories)
                / required,
            }
        def estimate(usage: dict[str, object]) -> float:
            if usage["cost"] is not None:
                return float(usage["cost"])
            return (
                int(usage["prompt_tokens"]) * 0.1 / 1_000_000
                + int(usage["completion_tokens"]) * 0.3 / 1_000_000
            )

        estimated = sum(
            estimate(envelope["payload"]["processor_usage"])
            for envelope in self.memory_envelopes
        ) + sum(
            estimate(envelope["payload"]["answer_usage"])
            for envelope in self.answer_envelopes
        )
        report = {
            "metric": receipt.METRIC,
            "complete": True,
            "method_specific_denominators": True,
            "no_single_all_methods_denominator": True,
            "progress": progress,
            "official_f1_by_cohort": {
                "all_qa_314": {
                    "question_n": 314,
                    "methods": sorted(self.all_methods),
                    "official_f1": all_report,
                },
                "retrieval_eligible_312": {
                    "question_n": 312,
                    "methods": sorted(self.methods),
                    "official_f1": eligible_report,
                },
            },
            "usage": {
                "processor": processor_usage,
                "processor_completion_tokens_per_call": 20.0,
                "answerer": answer_usage,
                "answer_completion_tokens_per_call": 5.0,
                "estimated_or_reported_usd": estimated,
                "configured_fuse_usd": 5.0,
                "scope": "semafuse_processor_and_common_answerer_only",
                "untracked_costs": [
                    "embedding",
                    "reranker",
                    "failed or uncommitted completion calls whose usage was not journaled",
                ],
            },
        }
        summary = {
            "run_key": self.run_key,
            "summary": {"manifest": self.manifest(), "summary": report},
        }
        (self.root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class CompactAnswerReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = CompleteAnswerFixture(cls.root)
        baseline = [
            envelope["payload"]
            for envelope in cls.fixture.memory_envelopes
            if envelope["method"] == "bm25"
        ]
        all_ids = {row["question_id"] for row in baseline}
        eligible_ids = {
            row["question_id"] for row in baseline if row["retrieval_eligible"]
        }
        cls.fixture_pins = {
            "OFFICIAL_ALL_QUESTION_SET_SHA256": receipt._question_set_sha256(all_ids),
            "OFFICIAL_ELIGIBLE_QUESTION_SET_SHA256": receipt._question_set_sha256(
                eligible_ids
            ),
            "OFFICIAL_CATEGORY_COUNTS_ALL": dict(
                sorted(Counter(row["category"] for row in baseline).items())
            ),
            "OFFICIAL_CATEGORY_COUNTS_ELIGIBLE": dict(
                sorted(
                    Counter(
                        row["category"] for row in baseline if row["retrieval_eligible"]
                    ).items()
                )
            ),
            "OFFICIAL_EVALUATOR_METADATA_SHA256": receipt._binding_sha256(
                b"semafuse-locomo-answer-evaluator-metadata/v1",
                (
                    (
                        row["question_id"],
                        row["category"],
                        str(row["category_number"]),
                        "1" if row["retrieval_eligible"] else "0",
                    )
                    for row in baseline
                ),
            ),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def build(self, root: Path | None = None, *, pinned: bool = False) -> dict[str, object]:
        root = root or self.root
        kwargs: dict[str, object] = {}
        if pinned:
            for argument, filename in (
                ("expected_summary_sha256", "summary.json"),
                ("expected_memory_sha256", "memory-contexts.jsonl"),
                ("expected_answers_sha256", "answers.jsonl"),
            ):
                kwargs[argument] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        with mock.patch.multiple(receipt, **self.fixture_pins):
            return receipt.build_receipt(
                summary_path=root / "summary.json",
                memory_journal_path=root / "memory-contexts.jsonl",
                answers_journal_path=root / "answers.jsonl",
                **kwargs,
            )

    def copy_fixture(self) -> Path:
        root = Path(tempfile.mkdtemp(dir=self.root))
        for filename in ("summary.json", "memory-contexts.jsonl", "answers.jsonl"):
            shutil.copyfile(self.root / filename, root / filename)
        return root

    def processor_case(self) -> tuple[dict[str, object], dict[str, object]]:
        return self.fixture.processor_trace(), {
            "calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cost": 0.001,
        }

    def validate_first_answer(self, payload_updates: dict[str, object]) -> None:
        envelope = json.loads(json.dumps(self.fixture.answer_envelopes[0]))
        for key, value in payload_updates.items():
            if key.startswith("metadata."):
                envelope["payload"]["metadata"][key.removeprefix("metadata.")] = value
            else:
                envelope["payload"][key] = value
        receipt._validate_answer_record(
            envelope,
            run_key=self.fixture.run_key,
            memory_run_key=self.fixture.memory_run_key,
            expected_prompt_version="semafuse-locomo-common-answer/v1",
            expected_model="qwen/qwen3-30b-a3b-instruct-2507",
            expected_provider="StreamLake",
            telemetry=receipt._new_response_telemetry(),
            label="fixture answer",
        )

    def test_builds_text_free_receipt_and_recomputes_both_cohorts(self) -> None:
        result = self.build(pinned=True)
        self.assertTrue(result["validation"]["caller_hash_pins_supplied_and_matched"])
        self.assertEqual(result["artifacts"]["memory_journal"]["records"], 1566)
        self.assertEqual(result["artifacts"]["answers_journal"]["records"], 1566)
        self.assertEqual(
            result["results"]["official_f1_by_cohort"]["all_qa_314"]["question_n"], 314
        )
        self.assertEqual(
            result["results"]["official_f1_by_cohort"]["retrieval_eligible_312"][
                "question_n"
            ],
            312,
        )
        self.assertEqual(
            result["usage"]["processor_response_telemetry"]["logical_responses"], 314
        )
        serialized = json.dumps(result)
        self.assertNotIn("private fixture context", serialized)
        self.assertNotIn("private fixture prediction", serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_accepts_coreweave_provider_bound_end_to_end(self) -> None:
        root = self.copy_fixture()
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protocol = summary["summary"]["manifest"]["protocol"]
        protocol["common_answer_expected_provider"] = "CoreWeave"
        protocol["common_answer_endpoint"]["extra_body"]["provider"]["order"] = [
            "CoreWeave"
        ]
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        answers_path = root / "answers.jsonl"
        answers = [json.loads(line) for line in answers_path.read_bytes().splitlines()]
        for answer in answers:
            answer["payload"]["metadata"]["response_provider"] = "CoreWeave"
        answers_path.write_bytes(b"".join(canonical_line(answer) for answer in answers))

        result = self.build(root)

        self.assertEqual(result["classification"]["common_answer_provider"], "CoreWeave")

    def test_rejects_manifest_expected_provider_routing_mismatch(self) -> None:
        root = self.copy_fixture()
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["summary"]["manifest"]["protocol"][
            "common_answer_expected_provider"
        ] = "CoreWeave"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(receipt.VerificationError, "provider/routing order"):
            self.build(root)

    def test_rejects_answer_provider_not_matching_manifest(self) -> None:
        root = self.copy_fixture()
        path = root / "answers.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["payload"]["metadata"]["response_provider"] = "CoreWeave"
        lines[0] = canonical_line(first)
        path.write_bytes(b"".join(lines))

        with self.assertRaisesRegex(receipt.VerificationError, "response provider"):
            self.build(root)

    def test_rejects_prediction_hash_mismatch(self) -> None:
        root = self.copy_fixture()
        path = root / "answers.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["payload"]["prediction"] += " tampered"
        lines[0] = canonical_line(first)
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(receipt.VerificationError, "prediction SHA-256 mismatch"):
            self.build(root)

    def test_rejects_non_stop_common_answer(self) -> None:
        root = self.copy_fixture()
        path = root / "answers.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["payload"]["metadata"]["response_finish_reason"] = "length"
        lines[0] = canonical_line(first)
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(receipt.VerificationError, "did not finish with stop"):
            self.build(root)

    def test_rejects_final_length_truncated_processor_attempt(self) -> None:
        root = self.copy_fixture()
        path = root / "memory-contexts.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            value = json.loads(line)
            trace = value["payload"]["metadata"].get("strategy_trace")
            if trace is None:
                continue
            trace["strategy"]["processing"]["processor_calls"][0]["calls"][0][
                "finish_reason"
            ] = "length"
            lines[index] = canonical_line(value)
            break
        path.write_bytes(b"".join(lines))
        with self.assertRaisesRegex(receipt.VerificationError, "no later stop retry"):
            self.build(root)

    def test_allows_length_processor_attempt_followed_by_successful_retry(self) -> None:
        metadata = self.fixture.processor_trace()
        processing = metadata["strategy_trace"]["strategy"]["processing"]
        first_call = processing["processor_calls"][0]["calls"][0]
        first_call["finish_reason"] = "length"
        second_call = json.loads(json.dumps(first_call))
        second_call["attempt"] = 2
        second_call["finish_reason"] = "stop"
        processing["processor_calls"][0]["calls"].append(second_call)
        processing["processor_calls"][0]["attempts"] = 2
        processing["attempts"] = 2
        processing["validation_retries"] = 1
        processing["usage"].update(
            {
                "calls": 2,
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "reasoning_tokens": 0,
                "cost": 0.002,
            }
        )
        payload_usage = {
            "calls": 2,
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "cost": 0.002,
        }
        telemetry = receipt._new_response_telemetry()
        receipt._validate_processor_trace(metadata, payload_usage, telemetry, "fixture")
        self.assertEqual(telemetry["finish_reasons"], {"length": 1, "stop": 1})
        self.assertEqual(telemetry["validation_retry_responses"], 1)

    def test_rejects_unsupported_or_null_processor_finish_reasons(self) -> None:
        for finish_reason in ("content_filter", None):
            with self.subTest(finish_reason=finish_reason):
                metadata, usage = self.processor_case()
                call = metadata["strategy_trace"]["strategy"]["processing"][
                    "processor_calls"
                ][0]["calls"][0]
                call["finish_reason"] = finish_reason
                with self.assertRaisesRegex(receipt.VerificationError, "stop or retryable length"):
                    receipt._validate_processor_trace(
                        metadata,
                        usage,
                        receipt._new_response_telemetry(),
                        "fixture",
                    )

    def test_rejects_answer_reasoning_tokens_or_presence(self) -> None:
        mutations = (
            {"metadata.response_reasoning_tokens": 1},
            {"metadata.response_reasoning_present": True},
            {"metadata.response_reasoning_tokens": None},
            {"metadata.response_reasoning_tokens": False},
        )
        for updates in mutations:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(
                    receipt.VerificationError, "no-thinking common-answer|must be an integer"
                ):
                    self.validate_first_answer(updates)

    def test_rejects_processor_reasoning_above_delimiter_or_missing(self) -> None:
        for reasoning_tokens in (2, None):
            with self.subTest(reasoning_tokens=reasoning_tokens):
                metadata, usage = self.processor_case()
                processing = metadata["strategy_trace"]["strategy"]["processing"]
                call_usage = processing["processor_calls"][0]["calls"][0]["usage"]
                call_usage["reasoning_tokens"] = reasoning_tokens
                processing["usage"]["reasoning_tokens"] = reasoning_tokens or 0
                processing["usage"]["calls_missing_reasoning_tokens"] = int(
                    reasoning_tokens is None
                )
                with self.assertRaisesRegex(
                    receipt.VerificationError, "no-thinking processor telemetry"
                ):
                    receipt._validate_processor_trace(
                        metadata,
                        usage,
                        receipt._new_response_telemetry(),
                        "fixture",
                    )

    def test_answer_prompt_model_and_provider_are_manifest_bound(self) -> None:
        mutations = (
            ({"metadata.prompt_version": "wrong/v1"}, "prompt version"),
            ({"metadata.response_model": "qwen/wrong"}, "response model"),
            ({"metadata.response_provider": "DeepInfra"}, "response provider"),
        )
        for updates, message in mutations:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(receipt.VerificationError, message):
                    self.validate_first_answer(updates)

    def test_processor_usage_requires_cost_or_complete_tokens(self) -> None:
        metadata, usage = self.processor_case()
        processing = metadata["strategy_trace"]["strategy"]["processing"]
        call_usage = processing["processor_calls"][0]["calls"][0]["usage"]
        call_usage["prompt_tokens"] = None
        call_usage["cost"] = None
        processing["usage"].update(
            {
                "prompt_tokens": 0,
                "cost": None,
                "calls_missing_prompt_tokens": 1,
                "calls_missing_cost": 1,
            }
        )
        usage.update({"prompt_tokens": 0, "cost": None})
        with self.assertRaisesRegex(receipt.VerificationError, "neither cost nor complete token"):
            receipt._validate_processor_trace(
                metadata, usage, receipt._new_response_telemetry(), "fixture"
            )

    def test_processor_aggregate_cost_rejects_missing_per_call_cost(self) -> None:
        metadata, usage = self.processor_case()
        processing = metadata["strategy_trace"]["strategy"]["processing"]
        processing["processor_calls"][0]["calls"][0]["usage"]["cost"] = None
        processing["usage"]["calls_missing_cost"] = 1
        with self.assertRaisesRegex(receipt.VerificationError, "missing per-call cost"):
            receipt._validate_processor_trace(
                metadata, usage, receipt._new_response_telemetry(), "fixture"
            )

    def test_rejects_boolean_schema_version_and_category_number(self) -> None:
        bm25 = next(
            value for value in self.fixture.memory_envelopes if value["method"] == "bm25"
        )
        for field in ("schema_version", "category_number"):
            with self.subTest(field=field):
                envelope = json.loads(json.dumps(bm25))
                envelope["payload"][field] = True
                with self.assertRaisesRegex(receipt.VerificationError, field):
                    receipt._validate_memory_record(
                        envelope,
                        run_key=self.fixture.memory_run_key,
                        method_identities={
                            "bm25": {"id": "bm25", "cohort": "all_qa", "kind": "bm25"}
                        },
                        telemetry=receipt._new_response_telemetry(),
                        label="fixture memory",
                    )

        manifest = json.loads((self.root / "summary.json").read_text())["summary"][
            "manifest"
        ]
        manifest["schema_version"] = True
        with self.assertRaisesRegex(receipt.VerificationError, "manifest.schema_version"):
            receipt._validate_manifest(manifest, self.fixture.run_key)

    def test_rejects_mislabeled_bm25_and_mixed_import_identity(self) -> None:
        bm25 = next(
            value for value in self.fixture.memory_envelopes if value["method"] == "bm25"
        )
        mislabeled = json.loads(json.dumps(bm25))
        mislabeled["payload"]["metadata"]["kind"] = "imported_source_ranking_rag"
        with self.assertRaisesRegex(receipt.VerificationError, "BM25 metadata identity"):
            receipt._validate_memory_record(
                mislabeled,
                run_key=self.fixture.memory_run_key,
                method_identities={
                    "bm25": {"id": "bm25", "cohort": "all_qa", "kind": "bm25"}
                },
                telemetry=receipt._new_response_telemetry(),
                label="fixture memory",
            )

        imported = next(
            value
            for value in self.fixture.memory_envelopes
            if value["method"] == "mem0_source_rag"
        )
        mixed = json.loads(json.dumps(imported))
        mixed["payload"]["metadata"]["source_run_key"] = "4" * 64
        identity = {
            "id": "mem0_source_rag",
            "cohort": "retrieval_eligible",
            "kind": "imported_source_ranking_rag",
            "source_method": "mem0",
            "source_run_key": "7" * 64,
            "source_journal_sha256": "8" * 64,
            "source_summary_sha256": "9" * 64,
            "limit": 50,
        }
        with self.assertRaisesRegex(receipt.VerificationError, "imported method metadata"):
            receipt._validate_memory_record(
                mixed,
                run_key=self.fixture.memory_run_key,
                method_identities={"mem0_source_rag": identity},
                telemetry=receipt._new_response_telemetry(),
                label="fixture memory",
            )

    def test_recursive_safety_scan_rejects_embedded_paths_and_credential_uris(self) -> None:
        with self.assertRaisesRegex(receipt.VerificationError, "absolute path"):
            receipt._assert_receipt_safe({"note": "safe /home/alice/private"})
        with self.assertRaisesRegex(receipt.VerificationError, "credential-like"):
            receipt._assert_receipt_safe(
                {"note": "safe https://alice:hunter2@example.com"}
            )

    def test_rejects_question_set_and_category_population_mutations(self) -> None:
        original_question = self.fixture.questions[0]
        replacement_question = opaque_id("q", "not-an-official-question")
        root = self.copy_fixture()
        for filename in ("memory-contexts.jsonl", "answers.jsonl"):
            path = root / filename
            rewritten = []
            for line in path.read_bytes().splitlines(keepends=True):
                value = json.loads(line)
                if value["question_id"] == original_question:
                    value["question_id"] = replacement_question
                    value["payload"]["question_id"] = replacement_question
                    line = canonical_line(value)
                rewritten.append(line)
            path.write_bytes(b"".join(rewritten))
        with self.assertRaisesRegex(receipt.VerificationError, "pinned official set"):
            self.build(root)

        root = self.copy_fixture()
        for filename in ("memory-contexts.jsonl", "answers.jsonl"):
            path = root / filename
            rewritten = []
            for line in path.read_bytes().splitlines(keepends=True):
                value = json.loads(line)
                if value["question_id"] == original_question:
                    payload = value["payload"]
                    payload["category"] = "temporal"
                    payload["category_number"] = 2
                    if filename == "answers.jsonl":
                        payload["official_f1"]["category"] = 2
                        payload["official_f1"]["category_name"] = "temporal"
                    line = canonical_line(value)
                rewritten.append(line)
            path.write_bytes(b"".join(rewritten))
        with self.assertRaisesRegex(receipt.VerificationError, "category counts"):
            self.build(root)

    def test_rejects_incomplete_journals_with_clear_counts(self) -> None:
        root = self.copy_fixture()
        path = root / "answers.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        with self.assertRaisesRegex(receipt.VerificationError, "full run is incomplete"):
            self.build(root)

    def test_rejects_summary_metric_drift(self) -> None:
        root = self.copy_fixture()
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        summary["summary"]["summary"]["official_f1_by_cohort"]["all_qa_314"][
            "official_f1"
        ]["methods"]["bm25"]["overall"]["mean"] += 0.01
        path.write_text(json.dumps(summary) + "\n")
        with self.assertRaisesRegex(receipt.VerificationError, "differs from the recomputed value"):
            self.build(root)

    def test_failed_cli_validation_does_not_replace_existing_output(self) -> None:
        root = self.copy_fixture()
        summary_path = root / "summary.json"
        summary_path.write_text("{}\n")
        output = root / "receipt.json"
        output.write_text("sentinel\n")
        status = receipt.main(
            [
                "--summary",
                str(summary_path),
                "--memory-journal",
                str(root / "memory-contexts.jsonl"),
                "--answers-journal",
                str(root / "answers.jsonl"),
                "--output",
                str(output),
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(output.read_text(), "sentinel\n")


if __name__ == "__main__":
    unittest.main()
