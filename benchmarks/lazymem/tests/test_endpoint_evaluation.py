from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from semafuse_lazymem_bench.endpoints import CompletionResult, OpenAICompatibleClient
from semafuse_lazymem_bench.config import load_config, validate_reproduction_config
from semafuse_lazymem_bench.evaluation import (
    AnswerResult,
    ApproximateTokenCounter,
    EvaluationRunner,
    JudgeResult,
)
from semafuse_lazymem_bench.metrics import summarize
from semafuse_lazymem_bench.processing import (
    WINDOW_ANNOTATION_SYSTEM_PROMPT,
    WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE,
    ConcurrentMemoryProcessor,
)
from semafuse_lazymem_bench.resume import JsonlJournal, read_jsonl
from semafuse_lazymem_bench.types import MemoryContext, Message, Question
from semafuse_lazymem_bench.windows import EvidenceWindow


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


class EndpointTests(unittest.TestCase):
    def test_openai_compatible_client_with_offline_transport(self) -> None:
        requests = []

        def respond(request, timeout):
            requests.append((request, timeout))
            payload = json.loads(request.data)
            return _Response(
                json.dumps(
                    {
                        "id": "offline-test",
                        "model": payload["model"],
                        "choices": [{"message": {"content": "offline answer"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    }
                ).encode()
            )

        client = OpenAICompatibleClient(
            base_url="http://offline.invalid/v1", model="offline-model", retries=0
        )
        with patch("urllib.request.urlopen", side_effect=respond):
            result = client.complete(
                [{"role": "user", "content": "hello"}], top_p=0.95
            )
        self.assertEqual(result.text, "offline answer")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (3, 2))
        request_payload = json.loads(requests[0][0].data)
        self.assertEqual(request_payload["model"], "offline-model")
        self.assertEqual(request_payload["top_p"], 0.95)
        self.assertEqual(requests[0][0].full_url, "http://offline.invalid/v1/chat/completions")


class _DecisionClient:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        history_json = messages[-1]["content"].split("# History Window\n", 1)[1]
        payload = json.loads(history_json)
        source_messages = [
            item
            for section in ("previous_bridge", "core_messages", "next_bridge")
            for item in payload[section]
        ]
        output = [
            {
                "op": "KEEP" if "useful" in item["content"] else "DROP",
                "compressed_content": item["content"] if "useful" in item["content"] else "",
                "reason": "query evidence" if "useful" in item["content"] else "not evidence",
            }
            for item in source_messages
        ]
        return CompletionResult(json.dumps(output), 0.01, 10, 5)


class ProcessingTests(unittest.TestCase):
    def test_paper_mpm_defaults_are_explicit_in_config(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "paper_v2.public-repro.toml"
        with config_path.open("rb") as handle:
            mpm = tomllib.load(handle)["mpm"]
        self.assertEqual(mpm["temperature"], 0.1)
        self.assertEqual(mpm["top_p"], 0.95)
        self.assertIs(mpm["enable_thinking"], True)

    def test_named_profiles_fail_closed_on_merge_rule_drift(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "paper_v2.public-repro.toml"
        config = load_config(config_path)
        validate_reproduction_config(config)
        config["window"]["merge_rule"] = "paper"
        with self.assertRaisesRegex(ValueError, "window.merge_rule"):
            validate_reproduction_config(config)

    def test_appendix_g1_prompt_snapshots(self) -> None:
        self.assertEqual(
            hashlib.sha256(WINDOW_ANNOTATION_SYSTEM_PROMPT.encode()).hexdigest(),
            "28146db07a4ccf794f07a70b5678c496f279709a01974e4a5688146ef65d7b0d",
        )
        self.assertEqual(
            hashlib.sha256(WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE.encode()).hexdigest(),
            "e890b5348b3cedf05c34df850d652644a201e77085e579c2aa3dc7981c9a207e",
        )
        self.assertEqual(WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE.count("\n13."), 1)

    def test_parallel_processor_returns_chronological_memory(self) -> None:
        history = tuple(
            Message(f"m{i}", i, "user", "useful fact" if i == 2 else f"noise {i}")
            for i in range(4)
        )
        windows = [
            EvidenceWindow("w0", 0, 0, 3, history[:3], ("m1",)),
            EvidenceWindow("w1", 1, 2, 4, history[2:], ("m2",)),
        ]
        client = _DecisionClient()
        result = ConcurrentMemoryProcessor(client, max_workers=2).process(
            Question("q", "test", "type", "query", "answer", "", history), windows
        )
        self.assertEqual(result.memory_text, "[] user: useful fact")
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(len(result.decisions), 2)
        self.assertEqual(len(client.calls), 2)
        for prompt_messages, kwargs in client.calls:
            self.assertEqual(
                prompt_messages[0],
                {"role": "system", "content": WINDOW_ANNOTATION_SYSTEM_PROMPT},
            )
            self.assertIn("# Local Editing Rules\n1.", prompt_messages[1]["content"])
            self.assertEqual(kwargs["temperature"], 0.1)
            self.assertEqual(kwargs["top_p"], 0.95)
            self.assertEqual(kwargs["max_tokens"], 3072)
            self.assertEqual(
                kwargs["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": True}},
            )


class _Method:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def construct(self, question: Question) -> MemoryContext:
        self.calls += 1
        return MemoryContext(
            "useful evidence",
            retrieved_message_ids=("m0",),
            covered_message_ids=("m0",),
            retrieval_latency_s=0.1,
            processing_latency_s=0.2,
        )


class _Answerer:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: Question, memory: MemoryContext) -> AnswerResult:
        self.calls += 1
        return AnswerResult("yes", 0.3, 5, 1)


class _Judge:
    def judge(self, question: Question, prediction: str) -> JudgeResult:
        return JudgeResult(True, 0.4, '{"correct":true}')


class EvaluationTests(unittest.TestCase):
    def test_jsonl_resume_and_metrics(self) -> None:
        question = Question(
            "q1",
            "test",
            "single-hop",
            "question",
            "yes",
            "now",
            (Message("m0", 0, "user", "useful evidence", has_answer=True),),
            gold_message_ids=("m0",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            runner = EvaluationRunner(
                method=_Method(),
                answerer=_Answerer(),
                judge=_Judge(),
                journal=JsonlJournal(output),
                token_counter=ApproximateTokenCounter(),
                run_id="offline-test-run",
                run_metadata={"fixture": True},
            )
            self.assertEqual(runner.run([question]), {"completed": 1, "skipped": 0})
            self.assertEqual(runner.run([question]), {"completed": 0, "skipped": 1})
            rows = list(read_jsonl(output))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "offline-test-run")
            with self.assertRaisesRegex(ValueError, "different run_id"):
                EvaluationRunner(
                    method=_Method(),
                    answerer=_Answerer(),
                    journal=JsonlJournal(output),
                    token_counter=ApproximateTokenCounter(),
                    run_id="changed-run",
                )
            metrics = summarize(rows)
        self.assertIsNone(metrics["lj"])
        self.assertEqual(metrics["generic_judge_score"], 1.0)
        self.assertEqual(metrics["all_at_50"], 1.0)
        self.assertEqual(metrics["all_at_50_count"], 1)
        self.assertEqual(metrics["memory_tokens"]["count"], 1)
        self.assertEqual(metrics["latency_s"]["retrieval"]["mean"], 0.1)

    def test_all_at_50_excludes_questions_without_annotated_gold(self) -> None:
        common = {
            "question_type": "single-hop",
            "memory_tokens": 1,
            "tokenizer": "test",
            "latency_s": {},
        }
        metrics = summarize(
            [
                {
                    **common,
                    "gold_message_ids": [],
                    "covered_message_ids": [],
                },
                {
                    **common,
                    "gold_message_ids": ["gold"],
                    "covered_message_ids": [],
                },
            ]
        )
        self.assertEqual(metrics["all_at_50"], 0.0)
        self.assertEqual(metrics["all_at_50_count"], 1)

    def test_latency_protocol_discards_three_full_warmups_and_records_three_runs(self) -> None:
        question = Question(
            "q1",
            "test",
            "single-hop",
            "question",
            "yes",
            "now",
            (Message("m0", 0, "user", "evidence", has_answer=True),),
            gold_message_ids=("m0",),
        )
        method = _Method()
        answerer = _Answerer()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "latency.jsonl"
            result = EvaluationRunner(
                method=method,
                answerer=answerer,
                journal=JsonlJournal(output),
                token_counter=ApproximateTokenCounter(),
                run_id="latency-test",
            ).run([question], warmup_runs=3, measured_runs=3)
            rows = list(read_jsonl(output))
        self.assertEqual(result["warmup_executions"], 3)
        self.assertEqual(method.calls, 6)
        self.assertEqual(answerer.calls, 6)
        self.assertEqual([row["measurement_index"] for row in rows], [0, 1, 2])
        self.assertEqual(summarize(rows)["measured_runs"], 3)


if __name__ == "__main__":
    unittest.main()
