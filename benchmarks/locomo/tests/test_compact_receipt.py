from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


LOC0MO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOC0MO_DIR))

import compact_receipt as receipt  # noqa: E402


def opaque_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()}"


class SyntheticComparison:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.harness = root / "external-harness"
        (self.harness / "adapters").mkdir(parents=True)
        (self.harness / "__init__.py").write_text('"""fixture"""\n', encoding="utf-8")
        (self.harness / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.harness / "adapters" / "amem.py").write_text(
            "def fail_closed():\n    return True\n", encoding="utf-8"
        )
        harness_receipt = receipt.inspect_external_harness(self.harness)
        self.spec: dict[str, object] = {
            "schema_version": 1,
            "label": "development",
            "baseline": "BM25",
            "external_harness": {
                "directory": "external-harness",
                "tree_sha256": harness_receipt["tree_sha256"],
                "amem_adapter_sha256": harness_receipt["amem_adapter_sha256"],
            },
            "runs": [],
            "methods": [
                {"name": "BM25", "run": "semafuse-run", "source_method": "bm25"},
                {
                    "name": "Semafuse",
                    "run": "semafuse-run",
                    "source_method": "semafuse::retrieved",
                },
                {"name": "Mem0", "run": "mem0-run", "source_method": "mem0_oss_raw"},
                {"name": "A-MEM", "run": "amem-run", "source_method": "amem_openrouter"},
            ],
        }
        self.run_methods = {
            "semafuse-run": "semafuse::retrieved",
            "mem0-run": "mem0_oss_raw",
            "amem-run": "amem_openrouter",
        }
        self.patterns = {
            "bm25": ([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]),
            "semafuse::retrieved": ([1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
            "mem0_oss_raw": ([0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]),
            "amem_openrouter": ([1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]),
        }
        for run_id, candidate in self.run_methods.items():
            self.write_run(run_id, candidate)
            self.add_run_hashes(run_id)
        self.spec_path = root / "comparison.json"
        self.write_spec()

    def manifest(self, run_id: str, run_key: str, candidate: str) -> dict[str, object]:
        if candidate == "semafuse::retrieved":
            candidate_manifest: dict[str, object] = {
                "id": "semafuse",
                "kind": "semafuse_production_retrieval",
                "config_sha256": "1" * 64,
                "resolved_config_sha256": "2" * 64,
                "embedding_model": "fixture/embed",
                "reranker_model": "fixture/rerank",
                "model_revisions_verified": False,
                "outputs": [candidate],
            }
        else:
            candidate_manifest = {
                "id": candidate,
                "kind": "external_jsonl",
                "adapter_kind": "amem" if candidate.startswith("amem") else "mem0_oss",
                "track": "development",
                "upstream_revision": None,
                "model_revisions": {},
                "protected_identity_complete": False,
                "worker_config_sha256": "3" * 64,
                "command_sha256": "4" * 64,
            }
        return {
            "schema_version": 1,
            "run_key": run_key,
            "suite": run_id,
            "track": receipt.COMMON_TRACK,
            "paper_comparable": False,
            "benchmark": {
                "name": "LoCoMo",
                "upstream_commit": receipt.OFFICIAL_UPSTREAM_COMMIT,
                "dataset_sha256": receipt.OFFICIAL_DATASET_SHA256,
                "selection": {
                    "conversation_ids": receipt.OFFICIAL_CONVERSATIONS,
                    "categories": receipt.OFFICIAL_CATEGORIES,
                    "qa_n": receipt.OFFICIAL_QA_N,
                    "retrieval_eval_n": receipt.OFFICIAL_RETRIEVAL_N,
                },
            },
            "license": {
                "spdx": "CC-BY-NC-4.0",
                "noncommercial_use_acknowledged": True,
                "dataset_redistributed": False,
                "dataset_path_serialized": False,
            },
            "protocol": {
                "evidence_unit": "source_message",
                "cutoffs": receipt.EVIDENCE_CUTOFFS,
                "answer_generation": False,
            },
            "denominators": {
                "qa_n": receipt.OFFICIAL_QA_N,
                "retrieval_eval_n": receipt.OFFICIAL_RETRIEVAL_N,
            },
            "baseline_method": "bm25",
            "methods": [
                {"id": "bm25", "kind": "bm25", "limit": 50},
                candidate_manifest,
            ],
            "implementation": {
                "crate_version": "0.1.0",
                "cargo_lock_sha256": "5" * 64,
                "benchmark_code_sha256": "6" * 64,
            },
        }

    def evidence(self, recall10: float, recall50: float) -> list[dict[str, object]]:
        output = []
        for cutoff in receipt.EVIDENCE_CUTOFFS:
            hit = int(recall10 if cutoff <= 10 else recall50)
            output.append(
                {
                    "cutoff": cutoff,
                    "hits": hit,
                    "precision": hit / cutoff,
                    "recall_any": float(hit > 0),
                    "recall_all": float(hit == 1),
                    "mean_recall": float(hit),
                    "mrr": float(hit),
                    "ndcg": float(hit),
                }
            )
        return output

    def record(
        self, run_key: str, method: str, index: int, recall10: float, recall50: float
    ) -> dict[str, object]:
        question_id = opaque_id("q", f"question-{index}")
        category = list(receipt._CATEGORY_NUMBERS)[index % 4]
        ranked = [opaque_id("m", f"message-{index}-{rank}") for rank in range(50)]
        payload = {
            "schema_version": 1,
            "run_key": run_key,
            "method": method,
            "question_id": question_id,
            "category": category,
            "category_number": receipt._CATEGORY_NUMBERS[category],
            "gold_count": 1,
            "ranked_message_ids": ranked,
            "evidence": self.evidence(recall10, recall50),
            "efficiency": {
                "query_wall_s": 0.01,
                "retrieval_s": None,
                "shared_construction_s": None,
            },
            "metadata": {"fixture": True},
        }
        return {
            "run_key": run_key,
            "method": method,
            "question_id": question_id,
            "record_kind": receipt.RECORD_KIND,
            "payload": payload,
        }

    def write_run(self, run_id: str, candidate: str) -> None:
        directory = self.root / run_id
        directory.mkdir()
        run_key = hashlib.sha256(run_id.encode()).hexdigest()
        methods = ["bm25", candidate]
        journal: list[dict[str, object]] = []
        method_values: dict[str, dict[str, list[float]]] = {}
        for method in methods:
            recall10_pattern, recall50_pattern = self.patterns[method]
            method_values[method] = {
                "RecallAll@10": [],
                "nDCG@10": [],
                "RecallAll@50": [],
            }
            for index in range(receipt.OFFICIAL_RETRIEVAL_N):
                recall10 = recall10_pattern[index % len(recall10_pattern)]
                recall50 = recall50_pattern[index % len(recall50_pattern)]
                journal.append(self.record(run_key, method, index, recall10, recall50))
                method_values[method]["RecallAll@10"].append(recall10)
                method_values[method]["nDCG@10"].append(recall10)
                method_values[method]["RecallAll@50"].append(recall50)
        journal_bytes = b"".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for value in journal
        )
        (directory / "per-question.jsonl").write_bytes(journal_bytes)

        reports = {}
        for metric in receipt.METRICS:
            reports[metric.summary_name] = {
                "baseline_method": "bm25",
                "denominator": receipt.OFFICIAL_RETRIEVAL_N,
                "direction": "higher_is_better",
                "methods": {
                    method: {
                        "overall": {
                            "denominator": receipt.OFFICIAL_RETRIEVAL_N,
                            "mean": sum(method_values[method][metric.receipt_name])
                            / receipt.OFFICIAL_RETRIEVAL_N,
                        }
                    }
                    for method in methods
                },
            }
        manifest = self.manifest(run_id, run_key, candidate)
        report = {
            "schema_version": 1,
            "run_key": run_key,
            "benchmark": "locomo",
            "protocol": receipt.COMMON_PROTOCOL,
            "baseline_method": "bm25",
            "qa_n": receipt.OFFICIAL_QA_N,
            "retrieval_eval_n": receipt.OFFICIAL_RETRIEVAL_N,
            "cutoffs": receipt.EVIDENCE_CUTOFFS,
            "methods": sorted(methods),
            "reports": reports,
        }
        summary = {
            "run_key": run_key,
            "summary": {"manifest": manifest, "summary": report},
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def add_run_hashes(self, run_id: str) -> None:
        inspected = receipt.inspect_run(self.root / run_id)
        self.spec["runs"].append(
            {
                "id": run_id,
                "directory": run_id,
                "summary_sha256": inspected["summary_sha256"],
                "per_question_sha256": inspected["per_question_sha256"],
                "manifest_sha256": inspected["manifest_sha256"],
            }
        )

    def refresh_hash(self, run_id: str, artifact: str) -> None:
        run = next(value for value in self.spec["runs"] if value["id"] == run_id)
        filename = "summary.json" if artifact == "summary" else "per-question.jsonl"
        run[f"{artifact}_sha256"] = hashlib.sha256(
            (self.root / run_id / filename).read_bytes()
        ).hexdigest()

    def write_spec(self) -> None:
        self.spec_path.write_text(
            json.dumps(self.spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def build(self) -> dict[str, object]:
        return receipt.build_receipt(receipt.load_comparison_spec(self.spec_path))


class CompactReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticComparison(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_builds_deterministic_text_free_paired_receipt(self) -> None:
        result = self.fixture.build()
        self.assertEqual(result["population"]["question_count"], 312)
        self.assertEqual(result["bootstrap"]["seed"], 20260808)
        self.assertEqual(result["bootstrap"]["resamples"], 10_000)
        self.assertEqual(result["methods"]["BM25"]["metrics"]["RecallAll@10"]["mean"], 0.5)
        self.assertEqual(
            result["methods"]["Semafuse"]["metrics"]["RecallAll@10"]["mean"], 0.75
        )
        semafuse_delta = result["paired_deltas"]["relative_to_bm25"]["Semafuse"]
        self.assertEqual(semafuse_delta["metrics"]["RecallAll@10"]["mean_delta"], 0.25)
        self.assertEqual(len(result["paired_deltas"]["between_nonbaseline_methods"]), 3)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(self.temporary.name, serialized)
        self.assertNotIn(opaque_id("q", "question-0"), serialized)
        self.assertEqual(result, self.fixture.build())

    def test_artifact_hash_tampering_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "mem0-run" / "per-question.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(receipt.VerificationError, "SHA-256 mismatch"):
            self.fixture.build()

    def test_run_key_mismatch_fails_even_when_artifact_hash_is_refreshed(self) -> None:
        path = Path(self.temporary.name) / "mem0-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["payload"]["run_key"] = "f" * 64
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("mem0-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "identity mismatch"):
            self.fixture.build()

    def test_cross_run_question_set_mismatch_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "amem-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        old_id = opaque_id("q", "question-0")
        new_id = opaque_id("q", "different-question")
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["question_id"] == old_id:
                value["question_id"] = new_id
                value["payload"]["question_id"] = new_id
                lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("amem-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "different opaque question-id set"):
            self.fixture.build()

    def test_summary_mean_mismatch_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "semafuse-run" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["summary"]["summary"]["reports"]["evidence_ndcg@10"]["methods"]["bm25"][
            "overall"
        ]["mean"] = 0.123
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("semafuse-run", "summary")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "summary mean disagrees"):
            self.fixture.build()

    def test_paper_reproduction_label_is_fail_closed(self) -> None:
        self.fixture.spec["label"] = "paper-reproduction"
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "paper-reproduction claim rejected"):
            self.fixture.build()

    def test_external_harness_and_amem_patch_hash_are_required(self) -> None:
        path = self.fixture.harness / "adapters" / "amem.py"
        path.write_text("def fail_closed():\n    return False\n", encoding="utf-8")
        with self.assertRaisesRegex(receipt.VerificationError, "tree SHA-256 mismatch"):
            self.fixture.build()

    def test_contract_adapter_cannot_enter_a_development_receipt(self) -> None:
        path = Path(self.temporary.name) / "amem-run" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        method = value["summary"]["manifest"]["methods"][1]
        method["track"] = "contract-test"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run = next(value for value in self.fixture.spec["runs"] if value["id"] == "amem-run")
        inspected_manifest = json.loads(path.read_text(encoding="utf-8"))["summary"]["manifest"]
        run["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        run["manifest_sha256"] = hashlib.sha256(
            receipt._canonical_json(inspected_manifest)
        ).hexdigest()
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "contract-only adapter"):
            self.fixture.build()


if __name__ == "__main__":
    unittest.main()
