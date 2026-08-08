"""Versioned JSONL input/output protocol for pluggable retrieval rankers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from .dataset import resolve_candidates
from .model import HarnessError, LoadedTask, RankingEntry

INPUT_SCHEMA = "lmeb-ranker-input-v1"
RANKING_SCHEMA = "lmeb-ranking-v1"


def normalize_ranking(value: Any, where: str) -> list[RankingEntry]:
    if not isinstance(value, list):
        raise HarnessError(f"{where} must be an array")
    entries: list[RankingEntry] = []
    total = len(value)
    for index, item in enumerate(value):
        if isinstance(item, str):
            doc_id, score = item, float(total - index)
        elif isinstance(item, dict):
            doc_id = item.get("doc_id")
            score = item.get("score", float(total - index))
            if not isinstance(doc_id, str) or not doc_id:
                raise HarnessError(f"{where}[{index}].doc_id must be a non-empty string")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise HarnessError(f"{where}[{index}].score must be numeric")
            score = float(score)
        else:
            raise HarnessError(f"{where}[{index}] must be a doc id or ranking object")
        if not math.isfinite(score):
            raise HarnessError(f"{where}[{index}].score must be finite")
        entries.append(RankingEntry(doc_id, score))
    if len({entry.doc_id for entry in entries}) != len(entries):
        raise HarnessError(f"{where} contains duplicate document ids")
    return entries


def parse_prediction_record(
    record: Mapping[str, Any], *, default_task_id: str | None = None
) -> tuple[str, str, dict[str, list[RankingEntry]], Mapping[str, Any]]:
    if record.get("schema_version") != RANKING_SCHEMA:
        raise HarnessError(f"prediction schema_version must be {RANKING_SCHEMA!r}")
    task_id = record.get("task_id", default_task_id)
    query_id = record.get("query_id")
    if not isinstance(task_id, str) or not task_id:
        raise HarnessError("prediction task_id must be a non-empty string")
    if not isinstance(query_id, str) or not query_id:
        raise HarnessError("prediction query_id must be a non-empty string")
    values = record.get("rankings")
    if values is None and "ranking" in record:
        values = {"default": record["ranking"]}
    if not isinstance(values, dict) or not values:
        raise HarnessError("prediction rankings must be a non-empty object")
    rankings: dict[str, list[RankingEntry]] = {}
    for stage, ranking in values.items():
        if not isinstance(stage, str) or not stage:
            raise HarnessError("prediction stage names must be non-empty strings")
        rankings[stage] = normalize_ranking(ranking, f"rankings.{stage}")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise HarnessError("prediction metadata must be an object")
    return task_id, query_id, rankings, metadata


def prediction_record(
    task_id: str,
    query_id: str,
    rankings: Mapping[str, Iterable[RankingEntry]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RANKING_SCHEMA,
        "task_id": task_id,
        "query_id": query_id,
        "rankings": {
            stage: [
                {"doc_id": entry.doc_id, "score": entry.score} for entry in entries
            ]
            for stage, entries in sorted(rankings.items())
        },
        "metadata": dict(metadata or {}),
    }


def load_predictions(
    path: Path, *, default_task_id: str | None = None
) -> dict[tuple[str, str], tuple[dict[str, list[RankingEntry]], Mapping[str, Any]]]:
    predictions: dict[
        tuple[str, str], tuple[dict[str, list[RankingEntry]], Mapping[str, Any]]
    ] = {}
    if not path.is_file():
        raise HarnessError(f"prediction JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid prediction JSONL {path}:{line_number}: {error}") from error
            if isinstance(record, dict) and record.get("type") == "metadata":
                continue
            if not isinstance(record, dict):
                raise HarnessError(f"prediction row must be an object: {path}:{line_number}")
            task_id, query_id, rankings, metadata = parse_prediction_record(
                record, default_task_id=default_task_id
            )
            key = (task_id, query_id)
            if key in predictions:
                raise HarnessError(f"duplicate prediction row {key}: {path}:{line_number}")
            predictions[key] = (rankings, metadata)
    return predictions


def write_jsonl_record(handle: TextIO, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def export_ranker_input(
    tasks: Iterable[LoadedTask], output: Path, *, include_instruction: bool
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        write_jsonl_record(handle, {"type": "metadata", "schema_version": INPUT_SCHEMA})
        for task in tasks:
            for document in task.corpus.values():
                write_jsonl_record(
                    handle,
                    {
                        "type": "document",
                        "schema_version": INPUT_SCHEMA,
                        "task_id": task.config.id,
                        "doc_id": document.id,
                        "title": document.title,
                        "text": document.text,
                    },
                )
            for query in task.queries.values():
                candidates = resolve_candidates(task, query.id)
                record = {
                        "type": "query",
                        "schema_version": INPUT_SCHEMA,
                        "task_id": task.config.id,
                        "query_id": query.id,
                        "text": query.text,
                        "candidate_doc_ids": list(candidates.doc_ids),
                        "candidate_scene_id": candidates.scene_id,
                    }
                if include_instruction:
                    record["instruction"] = task.config.instruction
                write_jsonl_record(handle, record)
