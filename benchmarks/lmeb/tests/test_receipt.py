from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lmeb_harness.receipt import ReceiptError, build_receipt


STAGES = ("dense", "bm25", "rrf", "retrieved")
METRICS = ("ndcg_at_10", "R_cap_at_10")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_result(
    task_id: str,
    query_ids: tuple[str, ...],
    values: dict[str, dict[str, tuple[float, ...]]],
    prediction_path: str,
    prediction_sha256: str,
) -> dict:
    stages = {}
    for stage in STAGES:
        per_query = {
            query_id: {
                metric: values[stage][metric][index]
                for metric in METRICS
            }
            for index, query_id in enumerate(query_ids)
        }
        stages[stage] = {
            "macro": {
                metric: round(
                    sum(values[stage][metric]) / len(values[stage][metric]), 5
                )
                for metric in METRICS
            },
            "per_query": per_query,
        }
    return {
        "task_id": task_id,
        "counts": {"queries": len(query_ids)},
        "metrics": {"stages": stages},
        "artifacts": {
            "predictions": prediction_path,
            "predictions_sha256": prediction_sha256,
        },
    }


class ReceiptTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, list[Path]]:
        result_dir = root / "actual" / "run"
        task_dirs = [result_dir / "tasks" / "task-a", result_dir / "tasks" / "task-b"]
        for directory in task_dirs:
            directory.mkdir(parents=True)
        query_sets = [("a-raw-query-1", "a-raw-query-2"), ("b-raw-query-1",)]
        prediction_paths = []
        for directory, task_id, query_ids in zip(
            task_dirs, ("Dataset/a", "Dataset/b"), query_sets, strict=True
        ):
            path = directory / "predictions.jsonl"
            rows = [
                {
                    "task_id": task_id,
                    "query_id": query_id,
                    "rankings": {stage: [] for stage in STAGES},
                }
                for query_id in query_ids
            ]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            prediction_paths.append(path)

        def stage_values(
            query_ids: tuple[str, ...], dense: float
        ) -> dict[str, dict[str, tuple[float, ...]]]:
            constants = {"dense": dense, "bm25": 0.25, "rrf": 0.5, "retrieved": 1.0}
            return {
                stage: {
                    metric: tuple(value for _ in query_ids)
                    for metric in METRICS
                }
                for stage, value in constants.items()
            }

        tasks = {
            "Dataset/a": task_result(
                "Dataset/a",
                query_sets[0],
                stage_values(query_sets[0], 0.0),
                "tasks/task-a/predictions.jsonl",
                sha256(prediction_paths[0]),
            ),
            "Dataset/b": task_result(
                "Dataset/b",
                query_sets[1],
                stage_values(query_sets[1], 1.0),
                "tasks/task-b/predictions.jsonl",
                sha256(prediction_paths[1]),
            ),
        }
        aggregate = {
            stage: {
                "mean_task": {
                    metric: sum(tasks[task]["metrics"]["stages"][stage]["macro"][metric] for task in tasks)
                    / len(tasks)
                    for metric in METRICS
                }
            }
            for stage in STAGES
        }
        input_descriptor = {"bytes": 1, "sha256": "0" * 64}
        provenance_inputs = {
            task_id: {
                name: copy.deepcopy(input_descriptor)
                for name in ("candidates", "corpus", "qrels", "queries")
            }
            for task_id in tasks
        }
        provenance_inputs["semafuse_artifacts"] = {
            str(root / "private" / "runner"): {
                "bytes": 2,
                "sha256": "1" * 64,
            }
        }
        result = {
            "schema_version": "lmeb-result-v1",
            "status": "complete",
            "protocol": "common-policy",
            "instruction_setting": "without",
            "cutoffs": [1, 10],
            "name": "synthetic",
            "run_id": "synthetic-run",
            "run_fingerprint": "2" * 64,
            "fidelity": {
                "candidate_filter": "common-policy deviation: candidates ingested cold per query"
            },
            "provenance": {
                "manifest_sha256": "3" * 64,
                "harness_source_sha256": "4" * 64,
                "fingerprint_payload": {"harness_source_sha256": "5" * 64},
                "inputs": provenance_inputs,
            },
            "tasks": tasks,
            "aggregate": aggregate,
        }
        result_path = result_dir / "result.json"
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

        spec = {
            "schema_version": "lmeb-actual-receipt-spec-v1",
            "bootstrap": {"seed": 7, "replicates": 10000},
            "models": {"embedding": {"model": "fixture"}},
            "runs": {
                "fixture": {
                    "dataset": "Dataset",
                    "aggregation": "equal task mean",
                    "result": "actual/run/result.json",
                    "result_sha256": sha256(result_path),
                    "name": "synthetic",
                    "run_id": "synthetic-run",
                    "run_fingerprint": "2" * 64,
                    "manifest_sha256": "3" * 64,
                    "final_harness_sha256": "4" * 64,
                    "fingerprint_harness_sha256": "5" * 64,
                    "cost_usd": "0.01",
                    "tasks": {
                        task_id: {
                            "queries": task["counts"]["queries"],
                            "predictions_sha256": task["artifacts"]["predictions_sha256"],
                        }
                        for task_id, task in tasks.items()
                    },
                    "dataset_inputs": {
                        task_id: provenance_inputs[task_id] for task_id in tasks
                    },
                    "runner_artifacts": {
                        "runner": {"bytes": 2, "sha256": "1" * 64}
                    },
                }
            },
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
        return spec_path, result_path, prediction_paths

    def test_deterministic_equal_task_paired_bootstrap_is_query_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path, _, _ = self.make_fixture(root)
            first = build_receipt(spec_path)
            second = build_receipt(spec_path)
            self.assertEqual(first, second)
            run = first["runs"]["fixture"]
            self.assertEqual(run["queries"], 3)
            # Task A is 0 and task B is 1, so equal task weight is .5 rather
            # than the query-pooled value of 1/3.
            self.assertEqual(run["metrics"]["dense"]["ndcg_at_10"]["estimate"], 0.5)
            self.assertEqual(run["metrics"]["retrieved"]["ndcg_at_10"]["estimate"], 1.0)
            delta = run["paired_deltas"]["retrieved_minus_dense"]["ndcg_at_10"]
            self.assertEqual(delta, {"estimate": 0.5, "ci95": [0.5, 0.5]})
            serialized = json.dumps(first)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("raw-query", serialized)

    def test_result_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec_path, result_path, _ = self.make_fixture(Path(temporary))
            result_path.write_text(result_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ReceiptError, "result sha256 mismatch"):
                build_receipt(spec_path)

    def test_prediction_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec_path, _, prediction_paths = self.make_fixture(Path(temporary))
            prediction_paths[0].write_text(
                prediction_paths[0].read_text() + "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ReceiptError, "predictions sha256 mismatch"):
                build_receipt(spec_path)

    def test_metric_inconsistency_fails_even_when_result_hash_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec_path, result_path, _ = self.make_fixture(Path(temporary))
            result = json.loads(result_path.read_text())
            result["tasks"]["Dataset/a"]["metrics"]["stages"]["dense"]["macro"][
                "ndcg_at_10"
            ] = 0.1
            result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            spec = json.loads(spec_path.read_text())
            spec["runs"]["fixture"]["result_sha256"] = sha256(result_path)
            spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ReceiptError, "metric inconsistency"):
                build_receipt(spec_path)

    def test_fewer_than_ten_thousand_replicates_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec_path, _, _ = self.make_fixture(Path(temporary))
            spec = json.loads(spec_path.read_text())
            spec["bootstrap"]["replicates"] = 9999
            spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ReceiptError, ">= 10000"):
                build_receipt(spec_path)


if __name__ == "__main__":
    unittest.main()
