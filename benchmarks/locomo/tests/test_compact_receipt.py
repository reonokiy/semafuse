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
        (self.harness / "adapters" / "amem_paper.py").write_text(
            "def paper_backend():\n    return True\n", encoding="utf-8"
        )
        harness_receipt = receipt.inspect_external_harness(self.harness)
        self.spec: dict[str, object] = {
            "schema_version": receipt.SCHEMA_VERSION,
            "label": "development",
            "baseline": "BM25",
            "external_harness": {
                "directory": "external-harness",
                "tree_sha256": harness_receipt["tree_sha256"],
                "adapter_sha256s": harness_receipt["adapter_sha256s"],
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
                {
                    "name": "A-MEM package/dev",
                    "run": "amem-run",
                    "source_method": "amem_openrouter",
                },
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
        metadata: dict[str, object] = {"fixture": True}
        shared_construction_s = None
        query_wall_s = 0.01
        if method != "bm25":
            conversation_n = receipt.OFFICIAL_RETRIEVAL_N // len(
                receipt.OFFICIAL_CONVERSATIONS
            )
            conversation_index = index // conversation_n
            construction_values = {
                "semafuse::retrieved": (2.0, 4.0),
                "mem0_oss_raw": (6.0, 8.0),
                "amem_openrouter": (10.0, 12.0),
            }
            shared_construction_s = construction_values[method][conversation_index]
            query_wall_s = 0.1 * ((index % 4) + 1)
            metadata.update(
                {
                    "construction_scope": "source_conversation",
                    "construction_amortization_n": conversation_n,
                    "construction_value_repeated_per_query": True,
                }
            )
            if method == "semafuse::retrieved":
                metadata["history_identity_sha256"] = hashlib.sha256(
                    f"history-{conversation_index}".encode()
                ).hexdigest()
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
                "query_wall_s": query_wall_s,
                "retrieval_s": None,
                "shared_construction_s": shared_construction_s,
            },
            "metadata": metadata,
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

    def refresh_summary_and_manifest_hashes(self, run_id: str) -> None:
        path = self.root / run_id / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        run = next(entry for entry in self.spec["runs"] if entry["id"] == run_id)
        run["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        run["manifest_sha256"] = hashlib.sha256(
            receipt._canonical_json(value["summary"]["manifest"])
        ).hexdigest()

    def update_candidate_manifest(self, run_id: str, **updates: object) -> None:
        path = self.root / run_id / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        candidate = next(
            method
            for method in value["summary"]["manifest"]["methods"]
            if method["kind"] != "bm25"
        )
        candidate.update(updates)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.refresh_summary_and_manifest_hashes(run_id)
        self.write_spec()

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
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["population"]["question_count"], 312)
        self.assertEqual(result["bootstrap"]["seed"], 20260808)
        self.assertEqual(result["bootstrap"]["resamples"], 10_000)
        self.assertEqual(result["methods"]["BM25"]["metrics"]["RecallAll@10"]["mean"], 0.5)
        self.assertEqual(
            result["methods"]["Semafuse"]["metrics"]["RecallAll@10"]["mean"], 0.75
        )
        self.assertIn("A-MEM package/dev", result["methods"])
        self.assertFalse(
            result["methods"]["A-MEM package/dev"]["source"]["identity"][
                "run_key_bound"
            ]
        )
        self.assertFalse(
            result["methods"]["Mem0"]["source"]["identity"]["run_key_bound"]
        )
        self.assertNotIn(
            "dependency_pins_verified",
            result["methods"]["Mem0"]["source"]["identity"],
        )
        self.assertEqual(
            set(result["provenance"]["external_harness"]["required_adapter_paths"]),
            receipt._REQUIRED_ADAPTER_PATHS,
        )
        self.assertIn(
            "does not claim a clean environment",
            result["provenance"]["external_harness"]["run_key_binding_note"],
        )
        semafuse_latency = result["methods"]["Semafuse"]["latency"]
        self.assertAlmostEqual(semafuse_latency["query_wall_s"]["total"], 78.0)
        self.assertAlmostEqual(semafuse_latency["query_wall_s"]["mean"], 0.25)
        self.assertAlmostEqual(semafuse_latency["query_wall_s"]["p50"], 0.25)
        self.assertAlmostEqual(semafuse_latency["query_wall_s"]["p95"], 0.4)
        self.assertEqual(
            semafuse_latency["shared_construction_s"]["grouping"],
            "history_identity_sha256",
        )
        self.assertEqual(
            semafuse_latency["shared_construction_s"]["shared_attempt_count"], 2
        )
        self.assertEqual(
            semafuse_latency["shared_construction_s"]["amortization_denominators"],
            [156, 156],
        )
        self.assertEqual(semafuse_latency["shared_construction_s"]["total"], 6.0)
        self.assertAlmostEqual(
            semafuse_latency["shared_construction_s"]["amortized_per_question"],
            6.0 / receipt.OFFICIAL_RETRIEVAL_N,
        )
        self.assertEqual(semafuse_latency["accounted_wall_s"]["total"], 84.0)
        mem0_construction = result["methods"]["Mem0"]["latency"][
            "shared_construction_s"
        ]
        self.assertEqual(
            mem0_construction["grouping"],
            "exact_value_and_construction_amortization_n",
        )
        self.assertEqual(mem0_construction["shared_attempt_count"], 2)
        self.assertEqual(mem0_construction["total"], 14.0)
        bm25_latency = result["methods"]["BM25"]["latency"]
        self.assertEqual(bm25_latency["shared_construction_s"]["total"], 0.0)
        self.assertEqual(
            bm25_latency["accounted_wall_s"]["total"],
            bm25_latency["query_wall_s"]["total"],
        )
        semafuse_delta = result["paired_deltas"]["relative_to_bm25"]["Semafuse"]
        self.assertEqual(semafuse_delta["metrics"]["RecallAll@10"]["mean_delta"], 0.25)
        self.assertEqual(len(result["paired_deltas"]["between_nonbaseline_methods"]), 3)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(self.temporary.name, serialized)
        self.assertNotIn(opaque_id("q", "question-0"), serialized)
        self.assertNotIn(hashlib.sha256(b"history-0").hexdigest(), serialized)
        self.assertTrue(result["validation"]["recursive_safety_scan_passed"])
        self.assertFalse(result["validation"]["absolute_paths_serialized"])
        self.assertFalse(result["validation"]["credentials_serialized"])
        self.assertEqual(result, self.fixture.build())

    def test_display_name_allows_package_dev_without_relaxing_identifiers(self) -> None:
        self.assertEqual(
            receipt._require_display_name("A-MEM package/dev", "method"),
            "A-MEM package/dev",
        )
        for value in ("../A-MEM", " A-MEM", "A-MEM//paper", "A-MEM\npackage"):
            with self.subTest(display_name=value):
                with self.assertRaisesRegex(receipt.VerificationError, "safe display name"):
                    receipt._require_display_name(value, "method")
        with self.assertRaisesRegex(receipt.VerificationError, "safe opaque identifier"):
            receipt._require_name("amem run", "run_id")
        with self.assertRaisesRegex(receipt.VerificationError, "safe opaque identifier"):
            receipt._require_name("amem/paper", "source_method", method=True)

    def test_bound_paper_backend_identity_matches_live_harness(self) -> None:
        tree_sha256 = self.fixture.spec["external_harness"]["tree_sha256"]
        self.fixture.update_candidate_manifest(
            "amem-run",
            adapter_kind="amem_paper",
            executable_sha256="7" * 64,
            python_tree_sha256=tree_sha256,
            dependency_lock_sha256="8" * 64,
            python_environment_sha256="6" * 64,
            dependency_pins_verified=True,
        )
        amem_selection = next(
            method
            for method in self.fixture.spec["methods"]
            if method["source_method"] == "amem_openrouter"
        )
        amem_selection["name"] = "A-MEM paper"
        self.fixture.write_spec()

        result = self.fixture.build()
        identity = result["methods"]["A-MEM paper"]["source"]["identity"]
        self.assertEqual(identity["adapter_kind"], "amem_paper")
        self.assertEqual(identity["executable_sha256"], "7" * 64)
        self.assertEqual(identity["python_tree_sha256"], tree_sha256)
        self.assertEqual(identity["dependency_lock_sha256"], "8" * 64)
        self.assertEqual(identity["python_environment_sha256"], "6" * 64)
        self.assertTrue(identity["dependency_pins_verified"])
        self.assertTrue(identity["run_key_bound"])
        self.assertFalse(
            result["methods"]["Mem0"]["source"]["identity"]["run_key_bound"]
        )

    def test_bound_external_python_tree_must_match_live_harness(self) -> None:
        self.fixture.update_candidate_manifest(
            "amem-run",
            adapter_kind="amem_paper",
            executable_sha256="7" * 64,
            python_tree_sha256="9" * 64,
            dependency_lock_sha256="8" * 64,
            python_environment_sha256="6" * 64,
            dependency_pins_verified=True,
        )
        with self.assertRaisesRegex(receipt.VerificationError, "audited harness tree"):
            self.fixture.build()

    def test_bound_external_identity_keeps_pins_claim_separate(self) -> None:
        tree_sha256 = self.fixture.spec["external_harness"]["tree_sha256"]
        self.fixture.update_candidate_manifest(
            "amem-run",
            executable_sha256="7" * 64,
            python_tree_sha256=tree_sha256,
            dependency_lock_sha256="8" * 64,
            python_environment_sha256="6" * 64,
            dependency_pins_verified=False,
        )

        result = self.fixture.build()
        identity = result["methods"]["A-MEM package/dev"]["source"]["identity"]
        self.assertTrue(identity["run_key_bound"])
        self.assertFalse(identity["dependency_pins_verified"])

    def test_bound_external_identity_fields_are_all_or_none(self) -> None:
        self.fixture.update_candidate_manifest(
            "amem-run",
            executable_sha256="7" * 64,
        )
        with self.assertRaisesRegex(receipt.VerificationError, "provide executable_sha256"):
            self.fixture.build()

    def test_bound_external_identity_requires_complete_metadata_bundle(self) -> None:
        tree_sha256 = self.fixture.spec["external_harness"]["tree_sha256"]
        cases = (
            {
                "executable_sha256": "7" * 64,
                "python_tree_sha256": tree_sha256,
                "dependency_lock_sha256": "8" * 64,
                "python_environment_sha256": "6" * 64,
            },
            {"dependency_pins_verified": True},
        )
        for updates in cases:
            with self.subTest(updates=sorted(updates)):
                parent = {
                    "id": "amem_paper_original",
                    "kind": "external_jsonl",
                    **updates,
                }
                identity = receipt._method_identity(parent)
                with self.assertRaisesRegex(
                    receipt.VerificationError, "dependency_pins_verified"
                ):
                    receipt._set_external_run_key_binding(
                        parent, identity, tree_sha256
                    )

    def test_method_identity_rejects_invalid_bound_hash(self) -> None:
        with self.assertRaisesRegex(receipt.VerificationError, "lowercase SHA-256"):
            receipt._method_identity(
                {
                    "id": "amem_paper_original",
                    "kind": "external_jsonl",
                    "python_environment_sha256": "not-a-sha",
                }
            )

    def test_method_identity_rejects_non_boolean_pins_claim(self) -> None:
        with self.assertRaisesRegex(receipt.VerificationError, "must be boolean"):
            receipt._method_identity(
                {
                    "id": "amem_paper_original",
                    "kind": "external_jsonl",
                    "dependency_pins_verified": "true",
                }
            )

    def test_inconsistent_identified_construction_group_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "semafuse-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["method"] == "semafuse::retrieved":
                value["payload"]["efficiency"]["shared_construction_s"] = 3.0
                lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("semafuse-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "inconsistent timing"):
            self.fixture.build()

    def test_partial_unidentified_construction_group_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "mem0-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["method"] == "mem0_oss_raw":
                value["payload"]["metadata"]["construction_amortization_n"] = 155
                lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("mem0-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "amortization denominator"):
            self.fixture.build()

    def test_identified_construction_rejects_100_and_212_row_conversations(self) -> None:
        path = Path(self.temporary.name) / "semafuse-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        candidate_index = 0
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["method"] != "semafuse::retrieved":
                continue
            first_group = candidate_index < 100
            value["payload"]["metadata"]["construction_amortization_n"] = (
                100 if first_group else 212
            )
            value["payload"]["metadata"]["history_identity_sha256"] = hashlib.sha256(
                ("history-100" if first_group else "history-212").encode()
            ).hexdigest()
            value["payload"]["efficiency"]["shared_construction_s"] = (
                2.0 if first_group else 4.0
            )
            lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            candidate_index += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("semafuse-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "two 156-row"):
            self.fixture.build()

    def test_fallback_construction_rejects_100_and_212_row_conversations(self) -> None:
        path = Path(self.temporary.name) / "mem0-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        candidate_index = 0
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["method"] != "mem0_oss_raw":
                continue
            first_group = candidate_index < 100
            value["payload"]["metadata"]["construction_amortization_n"] = (
                100 if first_group else 212
            )
            value["payload"]["efficiency"]["shared_construction_s"] = (
                6.0 if first_group else 8.0
            )
            lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            candidate_index += 1
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("mem0-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "two 156-row"):
            self.fixture.build()

    def test_mixed_shared_construction_presence_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "amem-run" / "per-question.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            value = json.loads(line)
            if value["method"] == "amem_openrouter":
                value["payload"]["efficiency"]["shared_construction_s"] = None
                lines[index] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.fixture.refresh_hash("amem-run", "per_question")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "mixes present and absent"):
            self.fixture.build()

    def test_unknown_method_kind_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "amem-run" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["summary"]["manifest"]["methods"][1]["kind"] = "custom_backend"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.fixture.refresh_summary_and_manifest_hashes("amem-run")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "supported method kind"):
            self.fixture.build()

    def test_absolute_model_path_fails_closed(self) -> None:
        path = Path(self.temporary.name) / "semafuse-run" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["summary"]["manifest"]["methods"][1]["embedding_model"] = (
            "/tmp/private/model"
        )
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.fixture.refresh_summary_and_manifest_hashes("semafuse-run")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "safe model identifier"):
            self.fixture.build()

    def test_credential_shaped_model_id_fails_recursive_scan(self) -> None:
        path = Path(self.temporary.name) / "semafuse-run" / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["summary"]["manifest"]["methods"][1]["embedding_model"] = (
            "sk-or-v1-" + "a" * 64
        )
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.fixture.refresh_summary_and_manifest_hashes("semafuse-run")
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "credential-like value"):
            self.fixture.build()

    def test_recursive_safety_scan_rejects_nested_paths_keys_and_values(self) -> None:
        unsafe = (
            ({"nested": [{"value": "/tmp/private"}]}, "absolute path"),
            ({"nested": {"api_key": "opaque"}}, "credential-like key"),
            ({"nested": ["Bearer abcdefghijklmnop"]}, "credential-like value"),
        )
        for value, message in unsafe:
            with self.subTest(message=message):
                with self.assertRaisesRegex(receipt.VerificationError, message):
                    receipt._assert_receipt_safe(value)

    def test_duplicate_run_source_selection_fails_closed(self) -> None:
        self.fixture.spec["methods"].append(
            {
                "name": "SemafuseAlias",
                "run": "semafuse-run",
                "source_method": "semafuse::retrieved",
            }
        )
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "duplicate selected run"):
            receipt.load_comparison_spec(self.fixture.spec_path)

    def test_bm25_cannot_be_selected_under_a_candidate_alias(self) -> None:
        mem0 = next(
            method for method in self.fixture.spec["methods"] if method["name"] == "Mem0"
        )
        mem0["source_method"] = "bm25"
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "baseline display name"):
            receipt.load_comparison_spec(self.fixture.spec_path)

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

    def test_external_harness_tree_hash_is_required(self) -> None:
        path = self.fixture.harness / "adapters" / "amem.py"
        path.write_text("def fail_closed():\n    return False\n", encoding="utf-8")
        with self.assertRaisesRegex(receipt.VerificationError, "tree SHA-256 mismatch"):
            self.fixture.build()

    def test_external_harness_root_symlink_is_rejected(self) -> None:
        link = Path(self.temporary.name) / "harness-link"
        try:
            link.symlink_to(self.fixture.harness, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"directory symlinks are unavailable: {error}")

        with self.assertRaisesRegex(receipt.VerificationError, "root must be a real directory"):
            receipt.inspect_external_harness(link)

    def test_external_harness_nested_directory_symlink_is_rejected(self) -> None:
        outside = Path(self.temporary.name) / "outside-harness-source"
        outside.mkdir()
        (outside / "hidden.py").write_text("VALUE = 1\n", encoding="utf-8")
        link = self.fixture.harness / "linked-source"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"directory symlinks are unavailable: {error}")

        with self.assertRaisesRegex(receipt.VerificationError, "contains a symlink"):
            receipt.inspect_external_harness(self.fixture.harness)

    def test_committed_development_harness_snapshot_matches_pins(self) -> None:
        spec = receipt.load_comparison_spec(
            LOC0MO_DIR / "receipts" / "locomo314.development.inputs.json"
        )
        inspected = receipt.inspect_external_harness(spec.external_harness.directory)

        self.assertEqual(inspected["tree_sha256"], spec.external_harness.tree_sha256)
        for path, expected in spec.external_harness.adapter_sha256s.items():
            self.assertEqual(inspected["adapter_sha256s"][path], expected)

    def test_external_harness_requires_both_amem_adapter_pins(self) -> None:
        adapter_sha256s = self.fixture.spec["external_harness"]["adapter_sha256s"]
        del adapter_sha256s["adapters/amem_paper.py"]
        self.fixture.write_spec()
        with self.assertRaisesRegex(receipt.VerificationError, "lacks required adapters"):
            receipt.load_comparison_spec(self.fixture.spec_path)

    def test_external_harness_adapter_map_detects_stale_paper_adapter_pin(self) -> None:
        path = self.fixture.harness / "adapters" / "amem_paper.py"
        path.write_text("def paper_backend():\n    return False\n", encoding="utf-8")
        inspected = receipt.inspect_external_harness(self.fixture.harness)
        self.fixture.spec["external_harness"]["tree_sha256"] = inspected["tree_sha256"]
        self.fixture.write_spec()
        with self.assertRaisesRegex(
            receipt.VerificationError,
            "adapter SHA-256 mismatch for adapters/amem_paper.py",
        ):
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
