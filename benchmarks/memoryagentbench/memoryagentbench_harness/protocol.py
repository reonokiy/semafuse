from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import hashlib
import json

from .chunking import SentenceSplitter, TokenCounter, official_sentence_chunks
from .dataset import load_fixture_rows, load_huggingface_rows
from .model import HarnessError, Manifest, SourceProfile
from .prompts import memory_message, query_message


def opaque_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:20]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _validate_public_event(event: Mapping[str, Any]) -> None:
    allowed = {
        "reset": {"schema_version", "op", "case_id", "source", "paper_name"},
        "add": {
            "schema_version",
            "op",
            "case_id",
            "memory_id",
            "ordinal",
            "text",
            "paper_message",
        },
        "query": {
            "schema_version",
            "op",
            "case_id",
            "query_id",
            "text",
            "paper_message",
            "top_k",
            "generation_max_tokens",
        },
    }
    op = event.get("op")
    if op not in allowed:
        raise HarnessError(f"unsupported public event op: {op!r}")
    extra = set(event) - allowed[str(op)]
    if extra:
        raise HarnessError(f"label isolation rejected fields: {sorted(extra)}")


def prepare_source(
    manifest: Manifest,
    profile: SourceProfile,
    output_dir: str | Path,
    *,
    fixture_path: str | Path | None = None,
    timestamp: str = "1970-01-01 00:00:00",
    max_contexts: int | None = None,
    chunk_size: int | None = None,
    sentence_splitter: SentenceSplitter | None = None,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """Create label-isolated input events and an evaluator-only label sidecar."""

    if max_contexts is not None and max_contexts <= 0:
        raise HarnessError("max_contexts must be positive")
    selected_chunk_size = chunk_size or profile.chunk_size_tokens
    rows = (
        load_fixture_rows(fixture_path, profile)
        if fixture_path is not None
        else load_huggingface_rows(manifest, profile)
    )
    if max_contexts is not None:
        rows = rows[:max_contexts]
    if not rows:
        raise HarnessError(f"no rows found for source {profile.name!r}")

    is_full_run = fixture_path is None and max_contexts is None
    if is_full_run and len(rows) != profile.expected_contexts:
        raise HarnessError(
            f"source {profile.name!r} expected {profile.expected_contexts} contexts, "
            f"found {len(rows)}"
        )

    public_events: list[dict[str, Any]] = []
    private_labels: list[dict[str, Any]] = []
    memory_count = 0
    query_count = 0

    for row_index, row in enumerate(rows):
        case_id = opaque_id("case", profile.name, row_index)
        reset = {
            "schema_version": 1,
            "op": "reset",
            "case_id": case_id,
            "source": profile.name,
            "paper_name": profile.paper_name,
        }
        _validate_public_event(reset)
        public_events.append(reset)

        chunks = official_sentence_chunks(
            row["context"],
            selected_chunk_size,
            sentence_splitter=sentence_splitter,
            token_counter=token_counter,
        )
        for ordinal, text in enumerate(chunks):
            memory_id = opaque_id("memory", profile.name, row_index, ordinal)
            add = {
                "schema_version": 1,
                "op": "add",
                "case_id": case_id,
                "memory_id": memory_id,
                "ordinal": ordinal,
                "text": text,
                "paper_message": memory_message(profile.prompt_family, text, timestamp),
            }
            _validate_public_event(add)
            public_events.append(add)
            memory_count += 1

        upstream_ids = row["upstream_qa_pair_ids"]
        for question_index, (question, answers) in enumerate(
            zip(row["questions"], row["answers"], strict=True)
        ):
            query_id = opaque_id("query", profile.name, row_index, question_index)
            query = {
                "schema_version": 1,
                "op": "query",
                "case_id": case_id,
                "query_id": query_id,
                "text": question,
                "paper_message": query_message(profile.prompt_family, question),
                "top_k": manifest.top_k,
                "generation_max_tokens": profile.generation_max_tokens,
            }
            _validate_public_event(query)
            public_events.append(query)

            label = {
                "schema_version": 1,
                "case_id": case_id,
                "query_id": query_id,
                "source": profile.name,
                "paper_name": profile.paper_name,
                "answers": answers,
                "primary_metric": profile.primary_metric,
            }
            if upstream_ids:
                label["upstream_qa_pair_id"] = upstream_ids[question_index]
            private_labels.append(label)
            query_count += 1

    if is_full_run and query_count != profile.expected_queries:
        raise HarnessError(
            f"source {profile.name!r} expected {profile.expected_queries} queries, "
            f"found {query_count}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inputs_path = destination / "inputs.jsonl"
    labels_path = destination / "labels.jsonl"
    _write_jsonl(inputs_path, public_events)
    _write_jsonl(labels_path, private_labels)

    run = {
        "schema_version": 1,
        "dataset": {
            "repo_id": manifest.dataset_repo_id,
            "revision": manifest.dataset_revision,
            "license": manifest.raw["dataset"]["license"],
        },
        "reference_implementation_revision": manifest.raw["reference_implementation"][
            "revision"
        ],
        "source_manifest_sha256": sha256_file(manifest.path),
        "source": profile.name,
        "paper_name": profile.paper_name,
        "context_count": len(rows),
        "memory_count": memory_count,
        "query_count": query_count,
        "chunk_size_tokens": selected_chunk_size,
        "paper_chunk_size_tokens": profile.chunk_size_tokens,
        "paper_comparable_shape": (
            fixture_path is None
            and max_contexts is None
            and selected_chunk_size == profile.chunk_size_tokens
        ),
        "top_k": manifest.top_k,
        "timestamp": timestamp,
        "input_sha256": sha256_file(inputs_path),
        "labels_sha256": sha256_file(labels_path),
        "label_isolation": {
            "runner_file": inputs_path.name,
            "evaluator_only_file": labels_path.name,
        },
    }
    run_path = destination / "run.json"
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return run


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise HarnessError(f"{path} contains a non-object JSONL record")
    return records
