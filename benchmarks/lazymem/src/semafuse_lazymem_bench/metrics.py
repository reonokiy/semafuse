"""LazyMem paper metrics over completed JSONL records."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution_stats(values: Iterable[float | int]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("metric values must be finite")
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p90": _percentile(numbers, 0.90),
        "p95": _percentile(numbers, 0.95),
        "min": min(numbers),
        "max": max(numbers),
    }


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    labels_by_type: defaultdict[str, list[bool]] = defaultdict(list)
    all_labels: list[bool] = []
    evaluation_scores: list[float] = []
    scores_by_type: defaultdict[str, list[float]] = defaultdict(list)
    evaluation_metrics: set[str] = set()
    coverage: list[bool] = []
    coverage_by_type: defaultdict[str, list[bool]] = defaultdict(list)
    tokenizers: set[str] = set()
    latency: defaultdict[str, list[float]] = defaultdict(list)
    memory_tokens: list[int] = []
    run_ids: set[str] = set()
    record_kinds: set[str] = set()
    measurement_indices: set[int] = set()

    for row in rows:
        if "run_id" in row:
            run_ids.add(str(row["run_id"]))
        record_kinds.add(str(row.get("record_kind", "evaluation")))
        if row.get("measurement_index") is not None:
            measurement_indices.add(int(row["measurement_index"]))
        question_type = str(row["question_type"])
        label = row.get("judge_label")
        if label is not None:
            if not isinstance(label, bool):
                raise TypeError("judge_label must be boolean or null")
            labels_by_type[question_type].append(label)
            all_labels.append(label)

        score = row.get("evaluation_score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("evaluation_score must be numeric or null")
            numeric_score = float(score)
            if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
                raise ValueError("evaluation_score must be finite and within [0, 1]")
            metric = row.get("evaluation_metric")
            if not isinstance(metric, str) or not metric:
                raise ValueError("evaluated rows require a non-empty evaluation_metric")
            evaluation_metrics.add(metric)
            evaluation_scores.append(numeric_score)
            scores_by_type[question_type].append(numeric_score)

        gold = set(str(value) for value in row.get("gold_message_ids", []))
        covered = set(str(value) for value in row.get("covered_message_ids", []))
        if gold:
            complete = gold.issubset(covered)
            coverage.append(complete)
            coverage_by_type[question_type].append(complete)

        memory_tokens.append(int(row["memory_tokens"]))
        tokenizers.add(str(row["tokenizer"]))
        for stage, value in row.get("latency_s", {}).items():
            latency[str(stage)].append(float(value))

    if len(run_ids) > 1:
        raise ValueError("metrics input mixes multiple run_id values")
    if len(record_kinds) > 1:
        raise ValueError("metrics input mixes latency and evaluation records")
    if len(evaluation_metrics) > 1:
        raise ValueError("metrics input mixes multiple evaluation_metric values")
    if evaluation_scores and len(evaluation_scores) != len(rows):
        raise ValueError("metrics input contains only a partial set of evaluation scores")

    def accuracy(values: Sequence[bool]) -> float | None:
        return statistics.fmean(values) if values else None

    evaluation_metric = next(iter(evaluation_metrics), None)
    evaluation_score = (
        statistics.fmean(evaluation_scores) if evaluation_scores else None
    )
    paper_lj = (
        evaluation_score
        if evaluation_metric == "longmemeval_official_lj"
        else None
    )
    return {
        "schema_version": 1,
        "run_id": next(iter(run_ids), None),
        "record_kind": next(iter(record_kinds), None),
        "measured_runs": len(measurement_indices) if measurement_indices else 1,
        "questions": len(rows),
        "evaluation_metric": evaluation_metric,
        "evaluation_score": evaluation_score,
        "evaluation_count": len(evaluation_scores),
        "evaluation_by_type": {
            question_type: {
                "score": statistics.fmean(values),
                "count": len(values),
            }
            for question_type, values in sorted(scores_by_type.items())
        },
        "lj": paper_lj,
        "lj_count": len(evaluation_scores) if paper_lj is not None else 0,
        "generic_judge_score": accuracy(all_labels),
        "generic_judge_count": len(all_labels),
        "lj_by_type": {
            question_type: {
                "score": statistics.fmean(values),
                "count": len(values),
            }
            for question_type, values in sorted(scores_by_type.items())
        }
        if paper_lj is not None
        else {},
        "generic_judge_by_type": {
            question_type: {
                "score": accuracy(values),
                "count": len(values),
            }
            for question_type, values in sorted(labels_by_type.items())
        },
        "all_at_50": accuracy(coverage),
        "all_at_50_count": len(coverage),
        "all_at_50_by_type": {
            question_type: {
                "score": accuracy(values),
                "count": len(values),
            }
            for question_type, values in sorted(coverage_by_type.items())
        },
        "memory_tokens": distribution_stats(memory_tokens),
        "tokenizers": sorted(tokenizers),
        "latency_s": {
            stage: distribution_stats(values) for stage, values in sorted(latency.items())
        },
        "definitions": {
            "all_at_50": (
                "complete gold-message coverage after retrieval plus configured "
                "neighbor expansion; questions without annotated gold are excluded"
            ),
            "percentile": "linear interpolation at (n-1)*p",
        },
    }
