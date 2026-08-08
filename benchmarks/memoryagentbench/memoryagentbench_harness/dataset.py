from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import json

from .model import HarnessError, Manifest, SourceProfile


def _as_string_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise HarnessError(f"{field} must be a string or list")

    flattened: list[str] = []
    for item in values:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, (list, tuple)):
            flattened.extend(_as_string_list(item, field))
        else:
            raise HarnessError(f"{field} contains a non-string value")
    if not flattened or any(not item.strip() for item in flattened):
        raise HarnessError(f"{field} must contain non-empty strings")
    return flattened


def normalize_row(row: Mapping[str, Any], profile: SourceProfile) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise HarnessError("row.metadata must be an object")
    source = metadata.get("source")
    if source != profile.name:
        raise HarnessError(f"expected source {profile.name!r}, got {source!r}")

    context = row.get("context")
    if not isinstance(context, str) or not context.strip():
        raise HarnessError("row.context must be a non-empty string")
    questions = row.get("questions")
    answers = row.get("answers")
    if not isinstance(questions, (list, tuple)) or not isinstance(answers, (list, tuple)):
        raise HarnessError("row.questions and row.answers must be lists")
    if len(questions) != len(answers) or not questions:
        raise HarnessError("questions and answers must be non-empty and aligned")

    normalized_questions = _as_string_list(questions, "questions")
    # Questions are not allowed to be nested; flattening changing the length is a
    # strong signal that the source schema changed.
    if len(normalized_questions) != len(questions):
        raise HarnessError("questions must be a flat list of strings")
    normalized_answers = [_as_string_list(answer, "answers") for answer in answers]

    upstream_ids = metadata.get("qa_pair_ids") or []
    if upstream_ids and len(upstream_ids) != len(normalized_questions):
        raise HarnessError("qa_pair_ids are not aligned with questions")

    return {
        "context": context,
        "questions": normalized_questions,
        "answers": normalized_answers,
        "upstream_qa_pair_ids": [str(value) for value in upstream_ids],
    }


def load_huggingface_rows(
    manifest: Manifest,
    profile: SourceProfile,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised by real runs
        raise HarnessError(
            "loading the pinned Hub snapshot requires the datasets package"
        ) from error

    split = load_dataset(
        manifest.dataset_repo_id,
        split=profile.split,
        revision=manifest.dataset_revision,
    )
    matching = [row for row in split if row.get("metadata", {}).get("source") == profile.name]
    return [normalize_row(row, profile) for row in matching]


def load_fixture_rows(path: str | Path, profile: SourceProfile) -> list[dict[str, Any]]:
    fixture_path = Path(path)
    with fixture_path.open(encoding="utf-8") as handle:
        if fixture_path.suffix == ".jsonl":
            payload: Any = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.load(handle)

    # Dataset Viewer responses wrap samples under rows[].row. Supporting the
    # shape makes read-only API audits possible without treating it as a pinned
    # data source for actual scores.
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        payload = [item["row"] for item in payload["rows"]]
    if not isinstance(payload, list):
        raise HarnessError("fixture must be a JSON list, JSONL, or Dataset Viewer response")

    matching = [
        row
        for row in payload
        if isinstance(row, Mapping)
        and row.get("metadata", {}).get("source") == profile.name
    ]
    return [normalize_row(row, profile) for row in matching]
