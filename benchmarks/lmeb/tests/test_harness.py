from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

from lmeb_harness.dataset import load_task, resolve_candidates
from lmeb_harness.manifest import load_manifest
from lmeb_harness.metrics import evaluate_rankings, score_query
from lmeb_harness.model import HarnessError, RankingEntry
from lmeb_harness.protocol import export_ranker_input
from lmeb_harness.runner import run_manifest


FIXTURES = Path(__file__).parent / "fixtures"


def write_manifest(
    directory: Path,
    ranker: dict,
    protocol: str = "common-policy",
    instruction_setting: str = "without",
) -> Path:
    value = {
        "schema_version": 1,
        "run": {
            "name": "fixture",
            "protocol": protocol,
            "instruction_setting": instruction_setting,
            "output_dir": str(directory / "results"),
            "resume": True,
        },
        "ranker": ranker,
        "tasks": [
            {
                "id": "LoCoMo/single_hop",
                "dataset": "LoCoMo",
                "subset": "single_hop",
                "memory_type": "LMEB_Dialogue",
                "corpus": str(FIXTURES / "corpus.jsonl"),
                "queries": str(FIXTURES / "queries.jsonl"),
                "qrels": str(FIXTURES / "qrels.tsv"),
                "candidates": str(FIXTURES / "candidates.jsonl"),
                "candidate_match": "official",
                "license": "fixture",
                "instruction": "retrieve the relevant memory",
            }
        ],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_binary_metrics_and_macro(self) -> None:
        qrels = {
            "q1": {"d1": 1, "d2": 1},
            "q2": {"d4": 1},
        }
        rankings = {
            "q1": [RankingEntry("d1", 3), RankingEntry("d3", 2), RankingEntry("d2", 1)],
            "q2": [RankingEntry("d5", 2), RankingEntry("d4", 1)],
        }
        result = evaluate_rankings(qrels, rankings)
        macro = result["macro"]
        self.assertEqual(macro["ndcg_at_1"], 0.5)
        self.assertEqual(macro["map_at_1"], 0.25)
        self.assertEqual(macro["recall_at_1"], 0.25)
        self.assertEqual(macro["precision_at_1"], 0.5)
        self.assertEqual(macro["mrr_at_1"], 0.5)
        self.assertEqual(macro["R_cap_at_1"], 0.5)
        expected_ndcg_5 = ((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 2
        self.assertEqual(macro["ndcg_at_5"], round(expected_ndcg_5, 5))
        self.assertEqual(macro["map_at_5"], 0.66667)
        self.assertEqual(macro["recall_at_5"], 1.0)
        self.assertEqual(macro["precision_at_5"], 0.3)
        self.assertEqual(macro["mrr_at_5"], 0.75)
        self.assertEqual(macro["R_cap_at_5"], 1.0)

    def test_graded_ndcg_and_official_tie_difference(self) -> None:
        graded = score_query({"d1": 3, "d2": 1}, [RankingEntry("d2", 2), RankingEntry("d1", 1)])
        self.assertAlmostEqual(graded["ndcg_at_1"], 1 / 3)
        tied = score_query({"d1": 1}, [RankingEntry("d1", 0), RankingEntry("d2", 0)])
        self.assertEqual(tied["ndcg_at_1"], 0.0)  # trec_eval: reverse doc-id tie break
        self.assertEqual(tied["mrr_at_1"], 1.0)  # MTEB Python: stable insertion tie break
        self.assertEqual(tied["R_cap_at_1"], 1.0)


class DatasetAndRunnerTests(unittest.TestCase):
    def test_official_numeric_json_ids_match_string_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            qrels = root / "qrels.tsv"
            candidates = root / "candidates.jsonl"
            corpus.write_text('{"id": 7, "text": "relevant document"}\n', encoding="utf-8")
            queries.write_text('{"id": 3, "text": "relevant"}\n', encoding="utf-8")
            qrels.write_text("3\t7\t1\n", encoding="utf-8")
            candidates.write_text(
                '{"scene_id": 3, "candidate_doc_ids": [7]}\n', encoding="utf-8"
            )
            raw = {
                "schema_version": 1,
                "run": {
                    "name": "numeric-ids",
                    "protocol": "paper-reproduction",
                    "instruction_setting": "without",
                    "output_dir": str(root / "results"),
                    "resume": True,
                },
                "ranker": {"kind": "bm25"},
                "tasks": [
                    {
                        "id": "CovidQA/Covid-QA",
                        "dataset": "CovidQA",
                        "subset": "Covid-QA",
                        "memory_type": "LMEB_Semantic",
                        "corpus": str(corpus),
                        "queries": str(queries),
                        "qrels": str(qrels),
                        "candidates": str(candidates),
                        "candidate_match": "official",
                        "license": "fixture",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            task = load_task(load_manifest(manifest_path).tasks[0])
            self.assertEqual(tuple(task.queries), ("3",))
            self.assertEqual(tuple(task.corpus), ("7",))
            self.assertEqual(resolve_candidates(task, "3").doc_ids, ("7",))

    def test_official_prefix_candidate_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_manifest(write_manifest(Path(temporary), {"kind": "bm25"}))
            task = load_task(manifest.tasks[0])
            resolution = resolve_candidates(task, "scene_a_q1")
            self.assertEqual(resolution.scene_id, "scene_a")
            self.assertEqual(resolution.doc_ids, ("d1", "d2", "d3"))

    def test_bm25_run_is_complete_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_manifest(write_manifest(Path(temporary), {"kind": "bm25"}))
            result_path, resumed = run_manifest(manifest)
            self.assertFalse(resumed)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertIn("bm25", result["tasks"]["LoCoMo/single_hop"]["metrics"]["stages"])
            same_path, resumed = run_manifest(manifest)
            self.assertTrue(resumed)
            self.assertEqual(same_path, result_path)

    def test_export_omits_instruction_in_without_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_manifest(write_manifest(root, {"kind": "bm25"}))
            output = root / "input.jsonl"
            export_ranker_input(
                [load_task(manifest.tasks[0])], output, include_instruction=False
            )
            query_rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("type") == "query"
            ]
            self.assertTrue(query_rows)
            self.assertTrue(all("instruction" not in row for row in query_rows))

    def test_no_resume_rebuilds_partial_checkpoint_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_manifest(write_manifest(root, {"kind": "bm25"}))
            result_path, _ = run_manifest(manifest)
            task_dir = next((result_path.parent / "tasks").iterdir())
            partial = task_dir / "predictions.partial.jsonl"
            partial.write_text('{"broken":"stale"}\n', encoding="utf-8")
            (task_dir / "state.json").write_text('{"stale":true}\n', encoding="utf-8")
            rebuilt, resumed = run_manifest(manifest, resume_override=False)
            self.assertFalse(resumed)
            rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["query_id"] for row in rows}, {"scene_a_q1", "scene_b_q1"})
            self.assertEqual(json.loads(rebuilt.read_text())["status"], "complete")

    def test_semafuse_jsonl_scores_all_four_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_manifest(
                write_manifest(
                    Path(temporary),
                    {
                        "kind": "semafuse-jsonl",
                        "command": [sys.executable, str(FIXTURES / "fake_semafuse.py")],
                        "timeout_seconds": 10,
                    },
                )
            )
            result_path, _ = run_manifest(manifest)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            stages = result["tasks"]["LoCoMo/single_hop"]["metrics"]["stages"]
            self.assertEqual(set(stages), {"dense", "bm25", "rrf", "retrieved"})
            self.assertIn("common-policy deviation", result["fidelity"]["candidate_filter"])
            task_dir = next((result_path.parent / "tasks").iterdir())
            rows = [
                json.loads(line)
                for line in (task_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["metadata"]["runner_timing_s"]["ingest"], 0.02)
            self.assertEqual(
                rows[0]["metadata"]["runner_isolation"],
                "fresh_sqlite_memory_per_question",
            )
            latency = result["tasks"]["LoCoMo/single_hop"]["latency"]
            self.assertEqual(latency["runner_construction_s"]["mean"], 0.05)
            self.assertEqual(latency["runner_cold_pipeline_s"]["mean"], 0.06)

    def test_paper_label_rejects_cold_candidate_semafuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_manifest(
                write_manifest(
                    Path(temporary),
                    {
                        "kind": "semafuse-jsonl",
                        "command": [sys.executable, str(FIXTURES / "fake_semafuse.py")],
                    },
                    protocol="paper-reproduction",
                )
            )
            with self.assertRaisesRegex(HarnessError, "cannot run under the paper-reproduction"):
                run_manifest(manifest)

    def test_inline_secret_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_manifest(
                Path(temporary),
                {
                    "kind": "semafuse-jsonl",
                    "command": ["runner", "--api-key", "sk-inline-secret"],
                },
            )
            with self.assertRaisesRegex(HarnessError, "secret"):
                load_manifest(path)

    def test_partial_unterminated_response_times_out_and_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = load_manifest(
                write_manifest(
                    Path(temporary),
                    {
                        "kind": "semafuse-jsonl",
                        "command": [
                            sys.executable,
                            str(FIXTURES / "fake_partial_semafuse.py"),
                        ],
                        "timeout_seconds": 0.1,
                    },
                )
            )
            started = time.monotonic()
            with self.assertRaisesRegex(HarnessError, "timed out"):
                run_manifest(manifest)
            self.assertLess(time.monotonic() - started, 3)


if __name__ == "__main__":
    unittest.main()
