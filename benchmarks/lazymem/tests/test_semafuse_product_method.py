from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semafuse_lazymem_bench.cli import (
    _run_identity,
    _validate_method_config,
    build_parser,
)
from semafuse_lazymem_bench.config import load_config
from semafuse_lazymem_bench.semafuse import (
    SemafuseLazyMemMethod,
    SemafuseRunnerError,
)
from semafuse_lazymem_bench.types import Message, Question


def _question(question_id: str = "q1") -> Question:
    return Question(
        question_id,
        "longmemeval",
        "single-session-user",
        "What was the useful fact?",
        "The useful fact.",
        "2026-01-01T00:00:00Z",
        (
            Message(
                "m0",
                0,
                "user",
                "The useful fact.",
                "2025-12-31T00:00:00Z",
                "session-a",
            ),
        ),
        gold_message_ids=("m0",),
    )


class SemafuseProductMethodTests(unittest.TestCase):
    def _runner(self, directory: Path) -> tuple[Path, Path]:
        config = directory / "semafuse.toml"
        config.write_text("mode = 'valid'\n", encoding="utf-8")
        runner = directory / "fake_runner.py"
        runner.write_text(
            """#!{python}
import json
import pathlib
import sys

mode = pathlib.Path(sys.argv[2]).read_text(encoding=\"utf-8\").strip()
for line in sys.stdin:
    question = json.loads(line)
    response = {{
        \"question_id\": question[\"id\"],
        \"context\": \"actual Semafuse memory for \" + question[\"query\"],
        \"retrieved_message_ids\": [question[\"messages\"][0][\"id\"]],
        \"covered_message_ids\": [question[\"messages\"][0][\"id\"]],
        \"retrieval_latency_s\": 0.12,
        \"processing_latency_s\": 0.34,
        \"metadata\": {{\"input_query\": question[\"query\"]}},
    }}
    if mode == \"mode = 'wrong-id'\":
        response[\"question_id\"] = \"wrong\"
    print(json.dumps(response), flush=True)
""".format(python=sys.executable),
            encoding="utf-8",
        )
        runner.chmod(0o755)
        return runner, config

    def test_persistent_runner_receives_normalized_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner, config = self._runner(Path(temp_dir))
            method = SemafuseLazyMemMethod(binary=runner, config_path=config)
            with method:
                first = method.construct(_question("q1"))
                process_id = method._process.pid if method._process else None
                second = method.construct(_question("q2"))
                self.assertEqual(method._process.pid if method._process else None, process_id)
            self.assertEqual(first.text, "actual Semafuse memory for What was the useful fact?")
            self.assertEqual(first.retrieved_message_ids, ("m0",))
            self.assertEqual(first.covered_message_ids, ("m0",))
            self.assertEqual(first.retrieval_latency_s, 0.12)
            self.assertEqual(first.processing_latency_s, 0.34)
            self.assertEqual(first.metadata["backend"], "semafuse-product")
            self.assertIs(first.metadata["paper_comparable"], False)
            self.assertEqual(
                first.metadata["runner_metadata"],
                {"input_query": "What was the useful fact?"},
            )
            self.assertEqual(second.metadata["runner_metadata"]["input_query"], "What was the useful fact?")

    def test_runner_response_question_id_is_strictly_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner, config = self._runner(Path(temp_dir))
            config.write_text("mode = 'wrong-id'\n", encoding="utf-8")
            method = SemafuseLazyMemMethod(binary=runner, config_path=config)
            with self.assertRaisesRegex(SemafuseRunnerError, "does not match"):
                with method:
                    method.construct(_question())


class SemafuseProductCliTests(unittest.TestCase):
    def test_parser_exposes_product_runner_options(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--config",
                "bench.toml",
                "--dataset",
                "longmemeval",
                "--input",
                "data.json",
                "--method",
                "semafuse-lazymem",
                "--output",
                "out.jsonl",
                "--semafuse-config",
                "strategy.toml",
                "--semafuse-benchmark-binary",
                "target/release/semafuse-lazymem-benchmark",
            ]
        )
        self.assertEqual(args.method, "semafuse-lazymem")
        self.assertEqual(args.semafuse_config, "strategy.toml")
        self.assertEqual(
            args.semafuse_benchmark_binary,
            "target/release/semafuse-lazymem-benchmark",
        )

    def test_product_method_requires_custom_profile_and_runner_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "semafuse_lazymem.custom.toml"
        )
        config = load_config(config_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy_config = Path(temp_dir) / "strategy.toml"
            strategy_config.write_text("[strategies]\n", encoding="utf-8")
            args = argparse.Namespace(
                method="semafuse-lazymem", semafuse_config=str(strategy_config)
            )
            _validate_method_config(args, config)

            config["reproduction"]["status"] = "clean-room-public-reconstruction"
            with self.assertRaisesRegex(ValueError, "custom-ablation"):
                _validate_method_config(args, config)

            config["reproduction"]["status"] = "custom-ablation"
            args.semafuse_config = None
            with self.assertRaisesRegex(ValueError, "--semafuse-config"):
                _validate_method_config(args, config)

    def test_product_run_identity_fingerprints_strategy_overrides(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "semafuse_lazymem.custom.toml"
        )
        config = load_config(config_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "questions.json"
            input_path.write_text("[]", encoding="utf-8")
            strategy_config = Path(temp_dir) / "strategy.toml"
            strategy_config.write_text("[strategies]\n", encoding="utf-8")
            args = argparse.Namespace(
                command="run",
                dataset="longmemeval",
                input=str(input_path),
                config=str(config_path),
                split_manifest=None,
                split="test",
                method="semafuse-lazymem",
                limit=None,
                generic_judge=False,
                approximate_tokens=False,
                semafuse_config=str(strategy_config),
                semafuse_benchmark_binary="semafuse-lazymem-benchmark",
            )
            override = "SEMAFUSE__STRATEGIES__SETTINGS__LAZYMEM__EMBEDDING__MODEL"
            with patch.dict(os.environ, {}, clear=True):
                baseline_id, baseline_metadata = _run_identity(args, config)
                with patch.dict(os.environ, {override: "alternate-embedding"}):
                    overridden_id, overridden_metadata = _run_identity(args, config)
        self.assertNotEqual(baseline_id, overridden_id)
        self.assertEqual(
            baseline_metadata["product_backend"]["strategy_environment_override_keys"],
            [],
        )
        self.assertEqual(
            overridden_metadata["product_backend"]["strategy_environment_override_keys"],
            [override],
        )


if __name__ == "__main__":
    unittest.main()
