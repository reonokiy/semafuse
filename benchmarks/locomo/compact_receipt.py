#!/usr/bin/env python3
"""Build a traceable, text-free receipt from LoCoMo retrieval run artifacts.

The Rust runner deliberately keeps labels and text out of its persisted records.
This verifier adds an independent, fail-closed comparison layer without changing
the runner: it pins artifact hashes, validates run identity and population
coverage, recomputes selected metrics, and performs one paired bootstrap shared
by every method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
OFFICIAL_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
OFFICIAL_UPSTREAM_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
OFFICIAL_QA_N = 314
OFFICIAL_RETRIEVAL_N = 312
OFFICIAL_CONVERSATIONS = ["conv-49", "conv-50"]
OFFICIAL_CATEGORIES = [1, 2, 3, 4]
EVIDENCE_CUTOFFS = [1, 5, 10, 50]
COMMON_TRACK = "paper-v2-selection/retrieval-only-common-evaluator"
COMMON_PROTOCOL = "paper-v2-retrieval-only"
RECORD_KIND = "locomo_retrieval"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_OPAQUE_QUESTION_ID = re.compile(r"^q_[0-9a-f]{64}$")
_OPAQUE_MESSAGE_ID = re.compile(r"^m_[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SAFE_METHOD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CATEGORY_NUMBERS = {
    "multi-hop": 1,
    "temporal": 2,
    "open-domain": 3,
    "single-hop": 4,
}


class VerificationError(ValueError):
    """An input artifact cannot support a trustworthy receipt."""


@dataclass(frozen=True)
class MetricSpec:
    receipt_name: str
    cutoff: int
    record_name: str
    summary_name: str


METRICS = (
    MetricSpec("RecallAll@10", 10, "recall_all", "evidence_recall_all@10"),
    MetricSpec("nDCG@10", 10, "ndcg", "evidence_ndcg@10"),
    MetricSpec("RecallAll@50", 50, "recall_all", "evidence_recall_all@50"),
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    directory: Path
    summary_sha256: str
    per_question_sha256: str
    manifest_sha256: str
    manifest_file_sha256: str | None


@dataclass(frozen=True)
class MethodSelection:
    name: str
    run_id: str
    source_method: str


@dataclass(frozen=True)
class ExternalHarnessSpec:
    directory: Path
    tree_sha256: str
    amem_adapter_sha256: str


@dataclass
class RunData:
    spec: RunSpec
    run_key: str
    manifest: dict[str, Any]
    report: dict[str, Any]
    records: dict[str, dict[str, dict[str, Any]]]
    method_manifests: dict[str, dict[str, Any]]
    artifact_receipt: dict[str, Any]
    question_ids: tuple[str, ...]
    question_metadata: dict[str, tuple[str, int, int]]


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    baseline: str
    external_harness: ExternalHarnessSpec
    runs: tuple[RunSpec, ...]
    methods: tuple[MethodSelection, ...]


class SplitMix64:
    """Small specified PRNG so bootstrap samples do not depend on Python internals."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("stop must be positive")
        limit = (1 << 64) - ((1 << 64) % stop)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % stop


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, source: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{source} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid JSON in {source}: {error}") from error


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {label}: {error}") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, label: str) -> tuple[bytes, str]:
    raw = _read_bytes(path, label)
    return raw, _sha256_bytes(raw)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerificationError(f"value is not canonical JSON: {error}") from error


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerificationError(f"{label} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise VerificationError(
            f"{label} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise VerificationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_name(value: Any, label: str, *, method: bool = False) -> str:
    pattern = _SAFE_METHOD if method else _SAFE_NAME
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise VerificationError(f"{label} is not a safe opaque identifier")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerificationError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise VerificationError(f"{label} must be finite and >= {minimum}")
    return 0.0 if result == 0.0 else result


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def load_comparison_spec(path: Path) -> ComparisonSpec:
    raw = _read_bytes(path, "comparison spec")
    root = _require_object(_parse_json(raw, "comparison spec"), "comparison spec")
    _require_exact_keys(
        root,
        {
            "schema_version",
            "label",
            "baseline",
            "external_harness",
            "runs",
            "methods",
        },
        set(),
        "comparison spec",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise VerificationError("unsupported comparison spec schema_version")
    label = root["label"]
    if label not in {"development", "paper-reproduction"}:
        raise VerificationError("label must be development or paper-reproduction")
    baseline = _require_name(root["baseline"], "baseline")

    raw_harness = _require_object(root["external_harness"], "external_harness")
    _require_exact_keys(
        raw_harness,
        {"directory", "tree_sha256", "amem_adapter_sha256"},
        set(),
        "external_harness",
    )
    harness_directory_value = raw_harness["directory"]
    if not isinstance(harness_directory_value, str) or not harness_directory_value.strip():
        raise VerificationError("external_harness.directory must be a non-empty string")
    harness_directory = Path(harness_directory_value)
    if not harness_directory.is_absolute():
        harness_directory = path.parent / harness_directory
    external_harness = ExternalHarnessSpec(
        directory=harness_directory,
        tree_sha256=_require_sha256(
            raw_harness["tree_sha256"], "external_harness.tree_sha256"
        ),
        amem_adapter_sha256=_require_sha256(
            raw_harness["amem_adapter_sha256"],
            "external_harness.amem_adapter_sha256",
        ),
    )

    run_specs: list[RunSpec] = []
    seen_runs: set[str] = set()
    for index, raw_run in enumerate(_require_list(root["runs"], "runs")):
        run = _require_object(raw_run, f"runs[{index}]")
        _require_exact_keys(
            run,
            {
                "id",
                "directory",
                "summary_sha256",
                "per_question_sha256",
                "manifest_sha256",
            },
            {"manifest_file_sha256"},
            f"runs[{index}]",
        )
        run_id = _require_name(run["id"], f"runs[{index}].id")
        if run_id in seen_runs:
            raise VerificationError(f"duplicate run id {run_id!r}")
        seen_runs.add(run_id)
        directory_value = run["directory"]
        if not isinstance(directory_value, str) or not directory_value.strip():
            raise VerificationError(f"runs[{index}].directory must be a non-empty string")
        directory = Path(directory_value)
        if not directory.is_absolute():
            directory = path.parent / directory
        manifest_file_sha = run.get("manifest_file_sha256")
        if manifest_file_sha is not None:
            manifest_file_sha = _require_sha256(
                manifest_file_sha, f"runs[{index}].manifest_file_sha256"
            )
        run_specs.append(
            RunSpec(
                run_id=run_id,
                directory=directory,
                summary_sha256=_require_sha256(
                    run["summary_sha256"], f"runs[{index}].summary_sha256"
                ),
                per_question_sha256=_require_sha256(
                    run["per_question_sha256"], f"runs[{index}].per_question_sha256"
                ),
                manifest_sha256=_require_sha256(
                    run["manifest_sha256"], f"runs[{index}].manifest_sha256"
                ),
                manifest_file_sha256=manifest_file_sha,
            )
        )
    if len(run_specs) < 2:
        raise VerificationError("a comparison receipt requires at least two run outputs")

    selections: list[MethodSelection] = []
    seen_names: set[str] = set()
    used_runs: set[str] = set()
    for index, raw_method in enumerate(_require_list(root["methods"], "methods")):
        method_entry = _require_object(raw_method, f"methods[{index}]")
        _require_exact_keys(
            method_entry,
            {"name", "run", "source_method"},
            set(),
            f"methods[{index}]",
        )
        name = _require_name(method_entry["name"], f"methods[{index}].name")
        run_id = _require_name(method_entry["run"], f"methods[{index}].run")
        source_method = _require_name(
            method_entry["source_method"], f"methods[{index}].source_method", method=True
        )
        if name in seen_names:
            raise VerificationError(f"duplicate method display name {name!r}")
        if run_id not in seen_runs:
            raise VerificationError(f"method {name!r} references unknown run {run_id!r}")
        seen_names.add(name)
        used_runs.add(run_id)
        selections.append(MethodSelection(name, run_id, source_method))
    if len(selections) < 2:
        raise VerificationError("a comparison receipt requires at least two methods")
    if baseline not in seen_names:
        raise VerificationError("baseline does not name a selected method")
    if used_runs != seen_runs:
        raise VerificationError(
            f"every pinned run must be used; unused={sorted(seen_runs - used_runs)}"
        )
    return ComparisonSpec(
        label,
        baseline,
        external_harness,
        tuple(run_specs),
        tuple(selections),
    )


def _validate_manifest_and_report(
    outer_run_key: str, manifest: dict[str, Any], report: dict[str, Any], label: str
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    _require_sha256(outer_run_key, f"{label}.run_key")
    if manifest.get("schema_version") != 1 or report.get("schema_version") != 1:
        raise VerificationError(f"{label} uses an unsupported manifest/report schema")
    if manifest.get("run_key") != outer_run_key or report.get("run_key") != outer_run_key:
        raise VerificationError(f"{label} outer, manifest, and report run_key differ")
    if manifest.get("track") != COMMON_TRACK:
        raise VerificationError(f"{label} is not the common LoCoMo retrieval track")
    if report.get("benchmark") != "locomo" or report.get("protocol") != COMMON_PROTOCOL:
        raise VerificationError(f"{label} report protocol mismatch")

    benchmark = _require_object(manifest.get("benchmark"), f"{label}.benchmark")
    selection = _require_object(benchmark.get("selection"), f"{label}.benchmark.selection")
    if (
        benchmark.get("name") != "LoCoMo"
        or benchmark.get("dataset_sha256") != OFFICIAL_DATASET_SHA256
        or benchmark.get("upstream_commit") != OFFICIAL_UPSTREAM_COMMIT
        or selection.get("conversation_ids") != OFFICIAL_CONVERSATIONS
        or selection.get("categories") != OFFICIAL_CATEGORIES
        or selection.get("qa_n") != OFFICIAL_QA_N
        or selection.get("retrieval_eval_n") != OFFICIAL_RETRIEVAL_N
    ):
        raise VerificationError(f"{label} does not use the pinned LoCoMo paper-v2 selection")
    denominators = _require_object(manifest.get("denominators"), f"{label}.denominators")
    if (
        denominators.get("qa_n") != OFFICIAL_QA_N
        or denominators.get("retrieval_eval_n") != OFFICIAL_RETRIEVAL_N
        or report.get("qa_n") != OFFICIAL_QA_N
        or report.get("retrieval_eval_n") != OFFICIAL_RETRIEVAL_N
        or report.get("cutoffs") != EVIDENCE_CUTOFFS
    ):
        raise VerificationError(f"{label} denominator/cutoff mismatch")
    if manifest.get("baseline_method") != report.get("baseline_method"):
        raise VerificationError(f"{label} manifest/report baseline mismatch")

    license_info = _require_object(manifest.get("license"), f"{label}.license")
    if (
        license_info.get("spdx") != "CC-BY-NC-4.0"
        or license_info.get("noncommercial_use_acknowledged") is not True
        or license_info.get("dataset_redistributed") is not False
        or license_info.get("dataset_path_serialized") is not False
    ):
        raise VerificationError(f"{label} license guard or serialization policy mismatch")
    implementation = _require_object(
        manifest.get("implementation"), f"{label}.implementation"
    )
    _require_sha256(
        implementation.get("cargo_lock_sha256"), f"{label}.implementation.cargo_lock_sha256"
    )
    _require_sha256(
        implementation.get("benchmark_code_sha256"),
        f"{label}.implementation.benchmark_code_sha256",
    )

    method_entries = _require_list(manifest.get("methods"), f"{label}.methods")
    output_to_parent: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(method_entries):
        entry = _require_object(raw_entry, f"{label}.methods[{index}]")
        method_id = _require_name(entry.get("id"), f"{label}.methods[{index}].id", method=True)
        outputs = entry.get("outputs")
        if outputs is None:
            output_ids = [method_id]
        else:
            output_ids = [
                _require_name(value, f"{label}.methods[{index}].outputs", method=True)
                for value in _require_list(outputs, f"{label}.methods[{index}].outputs")
            ]
            if not output_ids or len(output_ids) != len(set(output_ids)):
                raise VerificationError(f"{label} method outputs are empty or duplicated")
        for output_id in output_ids:
            if output_id in output_to_parent:
                raise VerificationError(f"{label} method output {output_id!r} is duplicated")
            output_to_parent[output_id] = entry
        for key, value in entry.items():
            if key.endswith("sha256") and value is not None:
                _require_sha256(value, f"{label}.methods[{index}].{key}")

    raw_report_methods = _require_list(report.get("methods"), f"{label}.report.methods")
    report_methods = tuple(
        _require_name(value, f"{label}.report.methods", method=True)
        for value in raw_report_methods
    )
    if len(report_methods) != len(set(report_methods)):
        raise VerificationError(f"{label} report methods are duplicated")
    if set(report_methods) != set(output_to_parent):
        raise VerificationError(f"{label} manifest outputs and report methods differ")
    return output_to_parent, report_methods


def _validate_evidence(record: Mapping[str, Any], label: str) -> dict[str, float]:
    gold_count = _require_int(record.get("gold_count"), f"{label}.gold_count", minimum=1)
    ranked = _require_list(record.get("ranked_message_ids"), f"{label}.ranked_message_ids")
    if len(ranked) > 50:
        raise VerificationError(f"{label} contains more than 50 ranked messages")
    ranked_ids: list[str] = []
    for value in ranked:
        if not isinstance(value, str) or not _OPAQUE_MESSAGE_ID.fullmatch(value):
            raise VerificationError(f"{label} contains a non-opaque message id")
        ranked_ids.append(value)
    if len(ranked_ids) != len(set(ranked_ids)):
        raise VerificationError(f"{label} contains duplicate ranked message ids")

    evidence = _require_list(record.get("evidence"), f"{label}.evidence")
    actual_cutoffs = [
        item.get("cutoff") if isinstance(item, dict) else None for item in evidence
    ]
    if actual_cutoffs != EVIDENCE_CUTOFFS:
        raise VerificationError(f"{label} does not contain exact ordered cutoffs")
    selected: dict[str, float] = {}
    previous_hits = -1
    for index, raw_metric in enumerate(evidence):
        metric = _require_object(raw_metric, f"{label}.evidence[{index}]")
        _require_exact_keys(
            metric,
            {
                "cutoff",
                "hits",
                "precision",
                "recall_any",
                "recall_all",
                "mean_recall",
                "mrr",
                "ndcg",
            },
            set(),
            f"{label}.evidence[{index}]",
        )
        cutoff = metric["cutoff"]
        hits = _require_int(metric["hits"], f"{label}.evidence[{index}].hits")
        if hits < previous_hits or hits > min(gold_count, cutoff, len(ranked_ids)):
            raise VerificationError(f"{label} has impossible or decreasing evidence hits")
        previous_hits = hits
        values = {
            key: _finite_number(metric[key], f"{label}.evidence[{index}].{key}")
            for key in ("precision", "recall_any", "recall_all", "mean_recall", "mrr", "ndcg")
        }
        if any(value < 0.0 or value > 1.0 + 1e-12 for value in values.values()):
            raise VerificationError(f"{label} evidence metric is outside [0, 1]")
        expected = {
            "precision": hits / cutoff,
            "recall_any": float(hits > 0),
            "recall_all": float(hits == gold_count),
            "mean_recall": hits / gold_count,
        }
        if any(not _close(values[key], value) for key, value in expected.items()):
            raise VerificationError(f"{label} evidence metrics disagree with hits/gold_count")
        for metric_spec in METRICS:
            if metric_spec.cutoff == cutoff:
                selected[metric_spec.receipt_name] = values[metric_spec.record_name]
    if set(selected) != {metric.receipt_name for metric in METRICS}:
        raise VerificationError(f"{label} lacks a receipt metric")
    return selected


def _parse_journal(
    raw: bytes, run_key: str, label: str
) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{label} per-question journal is not UTF-8") from error
    if not text.endswith("\n"):
        raise VerificationError(f"{label} per-question journal lacks a final newline")
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise VerificationError(f"{label} journal contains an empty line")
        envelope = _require_object(
            _parse_json(line.encode("utf-8"), f"{label} journal line {line_number}"),
            f"{label} journal line {line_number}",
        )
        _require_exact_keys(
            envelope,
            {"run_key", "method", "question_id", "record_kind", "payload"},
            set(),
            f"{label} journal line {line_number}",
        )
        payload = _require_object(
            envelope["payload"], f"{label} journal line {line_number}.payload"
        )
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "run_key",
                "method",
                "question_id",
                "category",
                "category_number",
                "gold_count",
                "ranked_message_ids",
                "evidence",
                "efficiency",
                "metadata",
            },
            set(),
            f"{label} journal line {line_number}.payload",
        )
        method = _require_name(payload.get("method"), f"{label}.method", method=True)
        question_id = payload.get("question_id")
        if not isinstance(question_id, str) or not _OPAQUE_QUESTION_ID.fullmatch(question_id):
            raise VerificationError(f"{label} journal contains a non-opaque question id")
        if (
            envelope["run_key"] != run_key
            or payload.get("run_key") != run_key
            or envelope["method"] != method
            or envelope["question_id"] != question_id
            or envelope["record_kind"] != RECORD_KIND
            or payload.get("schema_version") != 1
        ):
            raise VerificationError(f"{label} journal envelope/payload identity mismatch")
        category = payload.get("category")
        category_number = payload.get("category_number")
        if category not in _CATEGORY_NUMBERS or category_number != _CATEGORY_NUMBERS[category]:
            raise VerificationError(f"{label} journal category mismatch")
        metrics = _validate_evidence(payload, f"{label}:{method}:{question_id}")
        efficiency = _require_object(payload.get("efficiency"), f"{label}.efficiency")
        _require_exact_keys(
            efficiency,
            {"query_wall_s", "retrieval_s", "shared_construction_s"},
            set(),
            f"{label}.efficiency",
        )
        _finite_number(efficiency["query_wall_s"], f"{label}.query_wall_s", minimum=0.0)
        for key in ("retrieval_s", "shared_construction_s"):
            if efficiency[key] is not None:
                _finite_number(efficiency[key], f"{label}.{key}", minimum=0.0)
        _require_object(payload.get("metadata"), f"{label}.metadata")

        method_records = records.setdefault(method, {})
        if question_id in method_records:
            raise VerificationError(f"{label} duplicates ({method}, {question_id})")
        method_records[question_id] = {
            "category": category,
            "category_number": category_number,
            "gold_count": payload["gold_count"],
            "metrics": metrics,
        }
    if not records:
        raise VerificationError(f"{label} journal is empty")
    return records


def _verify_summary_metrics(run: RunData) -> None:
    reports = _require_object(run.report.get("reports"), f"{run.spec.run_id}.reports")
    for metric in METRICS:
        metric_report = _require_object(
            reports.get(metric.summary_name),
            f"{run.spec.run_id}.reports.{metric.summary_name}",
        )
        if (
            metric_report.get("baseline_method") != run.manifest.get("baseline_method")
            or metric_report.get("denominator") != OFFICIAL_RETRIEVAL_N
            or metric_report.get("direction") != "higher_is_better"
        ):
            raise VerificationError(
                f"{run.spec.run_id} summary metadata mismatch for {metric.summary_name}"
            )
        methods = _require_object(
            metric_report.get("methods"),
            f"{run.spec.run_id}.reports.{metric.summary_name}.methods",
        )
        if set(methods) != set(run.records):
            raise VerificationError(
                f"{run.spec.run_id} summary/journal method sets differ for {metric.summary_name}"
            )
        for method, method_records in run.records.items():
            overall = _require_object(
                _require_object(methods[method], f"{run.spec.run_id}.{method}").get("overall"),
                f"{run.spec.run_id}.{method}.overall",
            )
            if overall.get("denominator") != OFFICIAL_RETRIEVAL_N:
                raise VerificationError(f"{run.spec.run_id}.{method} has a bad denominator")
            expected = math.fsum(
                record["metrics"][metric.receipt_name] for record in method_records.values()
            ) / OFFICIAL_RETRIEVAL_N
            actual = _finite_number(
                overall.get("mean"), f"{run.spec.run_id}.{method}.{metric.receipt_name}"
            )
            if not _close(actual, expected):
                raise VerificationError(
                    f"{run.spec.run_id}.{method} summary mean disagrees with per-question records"
                )


def load_run(spec: RunSpec, *, require_expected_hashes: bool = True) -> RunData:
    summary_path = spec.directory / "summary.json"
    journal_path = spec.directory / "per-question.jsonl"
    summary_raw, summary_sha = _sha256_file(summary_path, f"{spec.run_id} summary")
    journal_raw, journal_sha = _sha256_file(journal_path, f"{spec.run_id} per-question")
    if require_expected_hashes and summary_sha != spec.summary_sha256:
        raise VerificationError(f"{spec.run_id} summary SHA-256 mismatch")
    if require_expected_hashes and journal_sha != spec.per_question_sha256:
        raise VerificationError(f"{spec.run_id} per-question SHA-256 mismatch")

    envelope = _require_object(_parse_json(summary_raw, f"{spec.run_id} summary"), "summary")
    _require_exact_keys(envelope, {"run_key", "summary"}, set(), f"{spec.run_id} summary")
    run_key = envelope["run_key"]
    inner = _require_object(envelope["summary"], f"{spec.run_id}.summary")
    _require_exact_keys(inner, {"manifest", "summary"}, set(), f"{spec.run_id}.summary")
    manifest = _require_object(inner["manifest"], f"{spec.run_id}.manifest")
    report = _require_object(inner["summary"], f"{spec.run_id}.report")
    manifest_sha = _sha256_bytes(_canonical_json(manifest))
    if require_expected_hashes and manifest_sha != spec.manifest_sha256:
        raise VerificationError(f"{spec.run_id} canonical manifest SHA-256 mismatch")

    manifest_path = spec.directory / "manifest.json"
    manifest_file_receipt: dict[str, Any] | None = None
    if manifest_path.exists():
        if require_expected_hashes and spec.manifest_file_sha256 is None:
            raise VerificationError(
                f"{spec.run_id} has manifest.json but the spec does not pin its SHA-256"
            )
        manifest_raw, manifest_file_sha = _sha256_file(
            manifest_path, f"{spec.run_id} manifest file"
        )
        if require_expected_hashes and manifest_file_sha != spec.manifest_file_sha256:
            raise VerificationError(f"{spec.run_id} manifest file SHA-256 mismatch")
        file_manifest = _require_object(
            _parse_json(manifest_raw, f"{spec.run_id} manifest file"),
            f"{spec.run_id} manifest file",
        )
        if file_manifest != manifest:
            raise VerificationError(f"{spec.run_id} file and embedded manifests differ")
        manifest_file_receipt = {"bytes": len(manifest_raw), "sha256": manifest_file_sha}
    elif spec.manifest_file_sha256 is not None:
        raise VerificationError(f"{spec.run_id} pins a missing manifest.json")

    method_manifests, report_methods = _validate_manifest_and_report(
        run_key, manifest, report, spec.run_id
    )
    records = _parse_journal(journal_raw, run_key, spec.run_id)
    if set(records) != set(report_methods):
        raise VerificationError(f"{spec.run_id} report/journal method sets differ")
    question_sets = {method: set(values) for method, values in records.items()}
    first_questions = next(iter(question_sets.values()))
    if len(first_questions) != OFFICIAL_RETRIEVAL_N:
        raise VerificationError(f"{spec.run_id} does not contain 312 questions per method")
    for method, questions in question_sets.items():
        if questions != first_questions:
            raise VerificationError(f"{spec.run_id} method {method!r} has a different question set")
    question_ids = tuple(sorted(first_questions))
    first_method = next(iter(records))
    question_metadata = {
        question_id: (
            records[first_method][question_id]["category"],
            records[first_method][question_id]["category_number"],
            records[first_method][question_id]["gold_count"],
        )
        for question_id in question_ids
    }
    for method, method_records in records.items():
        for question_id in question_ids:
            metadata = method_records[question_id]
            if (
                metadata["category"],
                metadata["category_number"],
                metadata["gold_count"],
            ) != question_metadata[question_id]:
                raise VerificationError(
                    f"{spec.run_id} method {method!r} changes evaluator metadata"
                )

    run = RunData(
        spec=spec,
        run_key=run_key,
        manifest=manifest,
        report=report,
        records=records,
        method_manifests=method_manifests,
        artifact_receipt={
            "summary": {"bytes": len(summary_raw), "sha256": summary_sha},
            "per_question": {"bytes": len(journal_raw), "sha256": journal_sha},
            "embedded_manifest": {
                "canonicalization": "RFC-8259 JSON sorted keys, UTF-8, no whitespace",
                "sha256": manifest_sha,
            },
            "manifest_file": manifest_file_receipt,
        },
        question_ids=question_ids,
        question_metadata=question_metadata,
    )
    _verify_summary_metrics(run)
    return run


def _question_set_sha256(question_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(b"semafuse-locomo-question-set/v1")
    for question_id in sorted(question_ids):
        raw = question_id.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def inspect_external_harness(directory: Path) -> dict[str, Any]:
    try:
        candidates = sorted(
            directory.rglob("*.py"),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
    except OSError as error:
        raise VerificationError(f"cannot enumerate external harness: {error}") from error
    if not candidates:
        raise VerificationError("external harness contains no Python source files")
    digest = hashlib.sha256(b"semafuse-external-memory-python-tree/v1")
    files: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise VerificationError("external harness Python tree contains a non-regular file")
        relative = path.relative_to(directory).as_posix()
        raw = _read_bytes(path, f"external harness {relative}")
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        files[relative] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    amem_path = "adapters/amem.py"
    if amem_path not in files:
        raise VerificationError("external harness lacks adapters/amem.py")
    return {
        "tree_sha256": digest.hexdigest(),
        "algorithm": "domain-separated SHA-256 over sorted relative .py path+bytes",
        "file_count": len(files),
        "total_bytes": sum(value["bytes"] for value in files.values()),
        "amem_adapter_sha256": files[amem_path]["sha256"],
    }


def _verify_external_harness(spec: ExternalHarnessSpec) -> dict[str, Any]:
    inspected = inspect_external_harness(spec.directory)
    if inspected["tree_sha256"] != spec.tree_sha256:
        raise VerificationError("external harness deterministic tree SHA-256 mismatch")
    if inspected["amem_adapter_sha256"] != spec.amem_adapter_sha256:
        raise VerificationError("A-MEM adapter SHA-256 mismatch")
    return inspected


def _percentile_linear(values: Sequence[float], probability: float) -> float:
    if not values:
        raise VerificationError("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ci95(values: Sequence[float]) -> dict[str, float]:
    return {
        "lower": _percentile_linear(values, 0.025),
        "upper": _percentile_linear(values, 0.975),
    }


def _bootstrap_means(
    ordered_rows: Sequence[Sequence[float]], columns: int
) -> list[list[float]]:
    n = len(ordered_rows)
    if n == 0:
        raise VerificationError("cannot bootstrap an empty aligned population")
    random = SplitMix64(BOOTSTRAP_SEED)
    distributions = [[] for _ in range(columns)]
    for _ in range(BOOTSTRAP_RESAMPLES):
        sums = [0.0] * columns
        for _ in range(n):
            row = ordered_rows[random.randbelow(n)]
            for column, value in enumerate(row):
                sums[column] += value
        for column, value in enumerate(sums):
            distributions[column].append(value / n)
    return distributions


def _method_identity(parent: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "id": parent.get("id"),
        "kind": parent.get("kind"),
    }
    for key in (
        "track",
        "adapter_kind",
        "upstream_revision",
        "protected_identity_complete",
        "model_revisions_verified",
        "config_sha256",
        "resolved_config_sha256",
        "worker_config_sha256",
        "command_sha256",
        "embedding_model",
        "reranker_model",
    ):
        if key in parent:
            identity[key] = parent[key]
    if "model_revisions" in parent:
        identity["model_revisions_sha256"] = _sha256_bytes(
            _canonical_json(parent["model_revisions"])
        )
    return identity


def _paper_reproduction_blockers(
    runs: Mapping[str, RunData], selections: Sequence[MethodSelection]
) -> list[str]:
    blockers: set[str] = set()
    for run in runs.values():
        if run.manifest.get("paper_comparable") is not True:
            blockers.add(f"run {run.spec.run_id} declares paper_comparable=false")
    for selection in selections:
        parent = runs[selection.run_id].method_manifests[selection.source_method]
        kind = parent.get("kind")
        if kind == "external_jsonl":
            track = parent.get("track")
            if track != "paper-reproduction":
                blockers.add(f"method {selection.name} external track is not paper-reproduction")
            if parent.get("protected_identity_complete") is not True:
                blockers.add(f"method {selection.name} lacks a complete protected identity")
            revision = parent.get("upstream_revision")
            if not isinstance(revision, str) or not _GIT_SHA.fullmatch(revision):
                blockers.add(f"method {selection.name} lacks an immutable upstream revision")
            revisions = parent.get("model_revisions")
            if not isinstance(revisions, dict) or not revisions:
                blockers.add(f"method {selection.name} lacks immutable model revisions")
        elif kind == "semafuse_production_retrieval":
            if parent.get("model_revisions_verified") is not True:
                blockers.add(f"method {selection.name} model revisions are not verified")
        elif kind not in {"bm25", "recent"}:
            blockers.add(f"method {selection.name} has an unsupported paper identity kind")
    return sorted(blockers)


def _common_manifest_fingerprint(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    common = {
        "benchmark": manifest["benchmark"],
        "denominators": manifest["denominators"],
        "implementation": manifest["implementation"],
        "license": manifest["license"],
        "protocol": manifest["protocol"],
        "track": manifest["track"],
    }
    return common, _sha256_bytes(_canonical_json(common))


def build_receipt(spec: ComparisonSpec) -> dict[str, Any]:
    external_harness = _verify_external_harness(spec.external_harness)
    runs = {run_spec.run_id: load_run(run_spec) for run_spec in spec.runs}
    if len({run.run_key for run in runs.values()}) != len(runs):
        raise VerificationError("comparison spec repeats the same run identity")

    first_run = next(iter(runs.values()))
    common_manifest, common_manifest_sha = _common_manifest_fingerprint(first_run.manifest)
    for run in runs.values():
        other_common, _ = _common_manifest_fingerprint(run.manifest)
        if other_common != common_manifest:
            raise VerificationError(
                f"run {run.spec.run_id} uses a different dataset/protocol/implementation"
            )
        if run.question_ids != first_run.question_ids:
            raise VerificationError(
                f"run {run.spec.run_id} has a different opaque question-id set"
            )
        if run.question_metadata != first_run.question_metadata:
            raise VerificationError(
                f"run {run.spec.run_id} has different evaluator metadata"
            )

    baseline_method_id = first_run.manifest.get("baseline_method")
    if baseline_method_id != "bm25":
        raise VerificationError("the common evaluator baseline must be bm25")
    baseline_vectors: dict[str, tuple[float, ...]] | None = None
    for run in runs.values():
        if run.manifest.get("baseline_method") != baseline_method_id:
            raise VerificationError(f"run {run.spec.run_id} uses a different baseline")
        if baseline_method_id not in run.records:
            raise VerificationError(f"run {run.spec.run_id} does not contain its BM25 baseline")
        current = {
            metric.receipt_name: tuple(
                run.records[baseline_method_id][question_id]["metrics"][metric.receipt_name]
                for question_id in first_run.question_ids
            )
            for metric in METRICS
        }
        if baseline_vectors is None:
            baseline_vectors = current
        elif current != baseline_vectors:
            raise VerificationError(f"run {run.spec.run_id} has different BM25 records")

    selection_by_name = {selection.name: selection for selection in spec.methods}
    baseline_selection = selection_by_name[spec.baseline]
    if baseline_selection.source_method != baseline_method_id:
        raise VerificationError("selected baseline must point to the BM25 source method")
    for selection in spec.methods:
        run = runs[selection.run_id]
        if selection.source_method not in run.records:
            raise VerificationError(
                f"method {selection.name!r} is absent from run {selection.run_id!r}"
            )
        if selection.source_method not in run.method_manifests:
            raise VerificationError(
                f"method {selection.name!r} has no manifest identity"
            )
        parent = run.method_manifests[selection.source_method]
        if parent.get("kind") == "external_jsonl" and (
            parent.get("track") == "contract-test" or parent.get("adapter_kind") == "fake"
        ):
            raise VerificationError(f"method {selection.name!r} is a contract-only adapter")

    paper_blockers = _paper_reproduction_blockers(runs, spec.methods)
    if spec.label == "paper-reproduction" and paper_blockers:
        raise VerificationError(
            "paper-reproduction claim rejected: " + "; ".join(paper_blockers)
        )

    method_metric_values: dict[str, dict[str, tuple[float, ...]]] = {}
    for selection in spec.methods:
        records = runs[selection.run_id].records[selection.source_method]
        method_metric_values[selection.name] = {
            metric.receipt_name: tuple(
                records[question_id]["metrics"][metric.receipt_name]
                for question_id in first_run.question_ids
            )
            for metric in METRICS
        }

    columns: list[tuple[str, str]] = [
        (selection.name, metric.receipt_name)
        for selection in spec.methods
        for metric in METRICS
    ]
    rows = [
        [
            method_metric_values[method][metric][question_index]
            for method, metric in columns
        ]
        for question_index in range(OFFICIAL_RETRIEVAL_N)
    ]
    distributions = _bootstrap_means(rows, len(columns))
    distribution_by_column = {
        column: distributions[index] for index, column in enumerate(columns)
    }

    method_receipts: dict[str, Any] = {}
    for selection in spec.methods:
        metric_receipts: dict[str, Any] = {}
        for metric in METRICS:
            values = method_metric_values[selection.name][metric.receipt_name]
            metric_receipts[metric.receipt_name] = {
                "denominator": OFFICIAL_RETRIEVAL_N,
                "mean": math.fsum(values) / OFFICIAL_RETRIEVAL_N,
                "ci95": _ci95(
                    distribution_by_column[(selection.name, metric.receipt_name)]
                ),
            }
        method_receipts[selection.name] = {
            "source": {
                "run": selection.run_id,
                "method": selection.source_method,
                "identity": _method_identity(
                    runs[selection.run_id].method_manifests[selection.source_method]
                ),
            },
            "metrics": metric_receipts,
        }

    def paired_delta(candidate: str, reference: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            candidate_values = method_metric_values[candidate][metric.receipt_name]
            reference_values = method_metric_values[reference][metric.receipt_name]
            differences = [
                candidate_value - reference_value
                for candidate_value, reference_value in zip(
                    candidate_values, reference_values, strict=True
                )
            ]
            bootstrap_differences = [
                candidate_value - reference_value
                for candidate_value, reference_value in zip(
                    distribution_by_column[(candidate, metric.receipt_name)],
                    distribution_by_column[(reference, metric.receipt_name)],
                    strict=True,
                )
            ]
            metrics[metric.receipt_name] = {
                "denominator": OFFICIAL_RETRIEVAL_N,
                "mean_delta": math.fsum(differences) / OFFICIAL_RETRIEVAL_N,
                "ci95": _ci95(bootstrap_differences),
                "wins": sum(value > 0.0 for value in differences),
                "ties": sum(value == 0.0 for value in differences),
                "losses": sum(value < 0.0 for value in differences),
            }
        return {"candidate": candidate, "reference": reference, "metrics": metrics}

    relative_to_bm25 = {
        selection.name: paired_delta(selection.name, spec.baseline)
        for selection in spec.methods
        if selection.name != spec.baseline
    }
    nonbaseline = [
        selection.name for selection in spec.methods if selection.name != spec.baseline
    ]
    between_methods = [
        paired_delta(nonbaseline[right], nonbaseline[left])
        for left in range(len(nonbaseline))
        for right in range(left + 1, len(nonbaseline))
    ]

    script_sha = _sha256_file(Path(__file__), "compact receipt tool")[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "LoCoMo",
        "track": COMMON_TRACK,
        "classification": {
            "label": spec.label,
            "paper_reproduction_eligible": not paper_blockers,
            "paper_reproduction_blockers": paper_blockers,
        },
        "metric_notice": (
            "Source-message retrieval diagnostics in fraction units; these are not "
            "LoCoMo answer F1 or a published paper score."
        ),
        "population": {
            "question_count": OFFICIAL_RETRIEVAL_N,
            "alignment_key": "opaque_question_id",
            "question_ids_serialized": False,
            "question_set_sha256": _question_set_sha256(first_run.question_ids),
            "evaluator_metadata_equal_across_runs": True,
            "bm25_metric_vectors_equal_across_runs": True,
        },
        "bootstrap": {
            "unit": "opaque_question_id",
            "paired_across_all_methods": True,
            "sampling": "with_replacement",
            "prng": "SplitMix64 with unbiased rejection sampling",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile_linear",
        },
        "provenance": {
            "dataset_sha256": OFFICIAL_DATASET_SHA256,
            "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
            "common_manifest_fingerprint_sha256": common_manifest_sha,
            "tool": {
                "path": "benchmarks/locomo/compact_receipt.py",
                "sha256": script_sha,
            },
            "external_harness": {
                **external_harness,
                "scope": "benchmarks/external_memory/src/semafuse_external_memory/**/*.py",
                "amem_fail_closed_adapter_included": True,
                "audit_state": "final_tree_at_receipt_build",
                "run_key_bound": False,
                "run_key_binding_note": (
                    "The LoCoMo run manifest did not hash python_path source files. This "
                    "receipt verifies the final audited external harness tree, including "
                    "the A-MEM fail-closed adapter patch, but does not retroactively claim "
                    "that earlier run keys committed to these bytes."
                ),
            },
            "runs": {
                run_id: {
                    "run_key": run.run_key,
                    "artifacts": run.artifact_receipt,
                }
                for run_id, run in runs.items()
            },
        },
        "validation": {
            "expected_hashes_match": True,
            "run_key_consistent_within_each_run": True,
            "summary_recomputed_from_per_question": True,
            "question_sets_identical": True,
            "raw_query_or_message_serialized": False,
            "absolute_paths_serialized": False,
            "credentials_serialized": False,
            "run_key_preimage_recomputed": False,
            "run_key_preimage_note": (
                "The runner does not serialize suite bytes, so this tool verifies run_key "
                "equality across envelope, manifest, report, and every journal record but "
                "does not independently reconstruct its preimage."
            ),
        },
        "baseline": spec.baseline,
        "methods": method_receipts,
        "paired_deltas": {
            "unit": "candidate_minus_reference",
            "relative_to_bm25": relative_to_bm25,
            "between_nonbaseline_methods": between_methods,
        },
    }


def inspect_run(directory: Path) -> dict[str, Any]:
    placeholder = "0" * 64
    manifest_file = directory / "manifest.json"
    spec = RunSpec(
        run_id="inspected-run",
        directory=directory,
        summary_sha256=placeholder,
        per_question_sha256=placeholder,
        manifest_sha256=placeholder,
        manifest_file_sha256=placeholder if manifest_file.exists() else None,
    )
    run = load_run(spec, require_expected_hashes=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_key": run.run_key,
        "summary_sha256": run.artifact_receipt["summary"]["sha256"],
        "per_question_sha256": run.artifact_receipt["per_question"]["sha256"],
        "manifest_sha256": run.artifact_receipt["embedded_manifest"]["sha256"],
        "manifest_file_sha256": (
            run.artifact_receipt["manifest_file"]["sha256"]
            if run.artifact_receipt["manifest_file"] is not None
            else None
        ),
        "methods": sorted(run.records),
        "question_count": len(run.question_ids),
        "question_set_sha256": _question_set_sha256(run.question_ids),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
        raise VerificationError(f"cannot write receipt: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser(
        "inspect-run", help="validate one run and print secret-free hashes for a comparison spec"
    )
    inspect_parser.add_argument("--run-dir", type=Path, required=True)
    harness_parser = commands.add_parser(
        "inspect-external-harness",
        help="print deterministic Python-tree and A-MEM adapter hashes for a spec",
    )
    harness_parser.add_argument("--source-dir", type=Path, required=True)
    build_parser = commands.add_parser(
        "build", help="verify pinned runs and build a compact paired-comparison receipt"
    )
    build_parser.add_argument("--spec", type=Path, required=True)
    build_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-run":
            value = inspect_run(args.run_dir)
        elif args.command == "inspect-external-harness":
            value = inspect_external_harness(args.source_dir)
        else:
            value = build_receipt(load_comparison_spec(args.spec))
        if getattr(args, "output", None) is not None:
            _write_json_atomic(args.output, value)
        else:
            json.dump(
                value,
                sys.stdout,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            sys.stdout.write("\n")
        return 0
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
