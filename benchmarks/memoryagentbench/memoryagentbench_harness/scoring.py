from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

from .metrics import (
    answer_coverage,
    answer_coverage_against_normalized,
    normalize_answer,
    official_answer_metrics,
)
from .model import HarnessError
from .protocol import read_jsonl, sha256_file


def _index_inputs(records: list[dict[str, Any]]) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], str],
]:
    memories: dict[tuple[str, str], str] = {}
    queries: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        op = record.get("op")
        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            raise HarnessError("input event is missing case_id")
        if op == "add":
            memory_id = record.get("memory_id")
            text = record.get("paper_message")
            if not isinstance(memory_id, str) or not isinstance(text, str):
                raise HarnessError("invalid add event")
            key = (case_id, memory_id)
            if key in memories:
                raise HarnessError(f"duplicate memory id: {memory_id}")
            memories[key] = text
        elif op == "query":
            query_id = record.get("query_id")
            if not isinstance(query_id, str):
                raise HarnessError("invalid query event")
            key = (case_id, query_id)
            if key in queries:
                raise HarnessError(f"duplicate query id: {query_id}")
            queries[key] = record
        elif op != "reset":
            raise HarnessError(f"unknown input op: {op!r}")
    return queries, memories


def _index_unique(
    records: list[dict[str, Any]], name: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        query_id = record.get("query_id")
        if not isinstance(case_id, str) or not isinstance(query_id, str):
            raise HarnessError(f"{name} record is missing case_id/query_id")
        key = (case_id, query_id)
        if key in indexed:
            raise HarnessError(f"duplicate {name} record for {key}")
        indexed[key] = record
    return indexed


def score_predictions(
    inputs_path: str | Path,
    labels_path: str | Path,
    predictions_path: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    input_records = read_jsonl(inputs_path)
    label_records = read_jsonl(labels_path)
    prediction_records = read_jsonl(predictions_path)
    queries, memories = _index_inputs(input_records)
    full_memory_by_case: dict[str, list[str]] = defaultdict(list)
    for (case_id, _), text in memories.items():
        full_memory_by_case[case_id].append(text)
    normalized_full_memory_by_case = {
        case_id: normalize_answer("\n".join(texts))
        for case_id, texts in full_memory_by_case.items()
    }
    labels = _index_unique(label_records, "label")
    predictions = _index_unique(prediction_records, "prediction")

    if set(labels) != set(queries):
        raise HarnessError("label sidecar does not match the public query set")
    unknown = set(predictions) - set(labels)
    if unknown:
        raise HarnessError(f"predictions contain {len(unknown)} unknown query ids")
    missing = set(labels) - set(predictions)
    if missing and not allow_partial:
        raise HarnessError(f"predictions are missing {len(missing)} query ids")

    totals: dict[str, list[float]] = defaultdict(list)
    by_source_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    official_count = 0
    retrieval_count = 0

    for key in sorted(set(labels) & set(predictions)):
        label = labels[key]
        prediction = predictions[key]
        source = label.get("source")
        answers = label.get("answers")
        if not isinstance(source, str) or not isinstance(answers, list):
            raise HarnessError("invalid label record")
        if not all(isinstance(answer, str) for answer in answers):
            raise HarnessError("label answers must be strings")

        predicted_text = prediction.get("prediction")
        retrieved_ids = prediction.get("retrieved_memory_ids")
        if predicted_text is None and retrieved_ids is None:
            raise HarnessError("prediction record has neither an answer nor retrieval ids")
        if predicted_text is not None:
            if not isinstance(predicted_text, str):
                raise HarnessError("prediction must be a string or null")
            metrics = official_answer_metrics(predicted_text, answers, source=source)
            for name, value in metrics.items():
                totals[name].append(value)
                by_source_values[source][name].append(value)
            official_count += 1

        if retrieved_ids is not None:
            if not isinstance(retrieved_ids, list) or not all(
                isinstance(value, str) for value in retrieved_ids
            ):
                raise HarnessError("retrieved_memory_ids must be a list of strings")
            if len(retrieved_ids) != len(set(retrieved_ids)):
                raise HarnessError("retrieved_memory_ids contains duplicates")
            top_k = queries[key].get("top_k")
            if not isinstance(top_k, int) or len(retrieved_ids) > top_k:
                raise HarnessError("retrieved_memory_ids exceeds the declared top_k")
            retrieved_texts: list[str] = []
            for memory_id in retrieved_ids:
                memory_key = (key[0], memory_id)
                if memory_key not in memories:
                    raise HarnessError(
                        f"query {key[1]} retrieved an unknown/cross-case memory id"
                    )
                retrieved_texts.append(memories[memory_key])
            coverage = answer_coverage("\n".join(retrieved_texts), answers)
            oracle_coverage = answer_coverage_against_normalized(
                normalized_full_memory_by_case[key[0]], answers
            )
            totals["answer_coverage_at_k"].append(coverage)
            by_source_values[source]["answer_coverage_at_k"].append(coverage)
            totals["answer_surface_oracle_coverage"].append(oracle_coverage)
            by_source_values[source]["answer_surface_oracle_coverage"].append(
                oracle_coverage
            )
            if oracle_coverage:
                totals["answer_coverage_given_surface_present"].append(coverage)
                by_source_values[source][
                    "answer_coverage_given_surface_present"
                ].append(coverage)
            totals["retrieved_count"].append(float(len(retrieved_ids)))
            by_source_values[source]["retrieved_count"].append(
                float(len(retrieved_ids))
            )
            retrieval_count += 1

    def summarize_percent(values: Mapping[str, list[float]]) -> dict[str, float]:
        return {
            name: sum(items) / len(items) * 100.0
            for name, items in sorted(values.items())
            if items and name != "retrieved_count"
        }

    def mean_retrieved_count(values: Mapping[str, list[float]]) -> float | None:
        items = values.get("retrieved_count", [])
        return sum(items) / len(items) if items else None

    report = {
        "schema_version": 1,
        "query_count_expected": len(labels),
        "query_count_scored": len(set(labels) & set(predictions)),
        "missing_prediction_count": len(missing),
        "official_answer_count": official_count,
        "retrieval_diagnostic_count": retrieval_count,
        "complete_official_answer_set": (
            not missing and official_count == len(labels)
        ),
        "metrics_percent": summarize_percent(totals),
        "metric_counts": {
            name: len(items)
            for name, items in sorted(totals.items())
            if name != "retrieved_count"
        },
        "mean_retrieved_count": mean_retrieved_count(totals),
        "by_source": {
            source: {
                "metrics_percent": summarize_percent(values),
                "mean_retrieved_count": mean_retrieved_count(values),
            }
            for source, values in sorted(by_source_values.items())
        },
        "comparability": {
            "substring_exact_match": (
                "paper-comparable only when the same source, chunking, TopK, prompt, "
                "GPT-4o-mini reader settings, and complete query set are used"
            ),
            "answer_coverage_at_k": (
                "non-official retrieval diagnostic; never compare directly to the "
                "paper's end-to-end accuracy"
            ),
            "answer_surface_oracle_coverage": (
                "non-official ceiling: the gold answer string occurs anywhere in "
                "the complete memory"
            ),
            "answer_coverage_given_surface_present": (
                "non-official diagnostic restricted to questions whose gold answer "
                "string occurs somewhere in the complete memory"
            ),
        },
        "artifacts": {
            "inputs_sha256": sha256_file(Path(inputs_path)),
            "labels_sha256": sha256_file(Path(labels_path)),
            "predictions_sha256": sha256_file(Path(predictions_path)),
        },
    }
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
