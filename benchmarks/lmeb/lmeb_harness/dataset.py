"""LMEB corpus/query/qrels/candidate loading and candidate resolution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .model import (
    CandidateResolution,
    Document,
    HarnessError,
    LoadedTask,
    Query,
    TaskConfig,
)


# This is the task-name branch in LMEB's SubsetRetrieval.py at commit a02ae842.
_DIRECT_QUERY_ID_DATASETS = {
    "CovidQA",
    "ESGReports",
    "QASPER",
    "PeerQA",
    "MLDR",
    "NovelQA",
    "LooGLE",
    "DeepPlanning",
}


def _jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if not path.is_file():
        raise HarnessError(f"dataset file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(f"invalid JSONL {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise HarnessError(f"JSONL record must be an object: {path}:{line_number}")
            yield line_number, value


def _record_string(record: Mapping[str, Any], field: str, where: str, *, optional: bool = False) -> str:
    value = record.get(field, "" if optional else None)
    if not isinstance(value, str) or (not optional and not value):
        raise HarnessError(f"{where}.{field} must be a non-empty string")
    return value


def _record_id(record: Mapping[str, Any], field: str, where: str) -> str:
    """Canonicalize Hub JSON scalar IDs to the string form used by qrels TSV files."""
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise HarnessError(f"{where}.{field} must be a non-empty string or integer")
    canonical = str(value)
    if not canonical:
        raise HarnessError(f"{where}.{field} must be a non-empty string or integer")
    return canonical


def _load_corpus(path: Path) -> dict[str, Document]:
    corpus: dict[str, Document] = {}
    for line, record in _jsonl(path):
        where = f"{path}:{line}"
        doc_id = _record_id(record, "id", where)
        if doc_id in corpus:
            raise HarnessError(f"duplicate corpus id {doc_id!r} at {where}")
        document = Document(
            id=doc_id,
            text=_record_string(record, "text", where, optional=True),
            title=_record_string(record, "title", where, optional=True),
        )
        if not document.combined_text.strip():
            raise HarnessError(f"corpus document {doc_id!r} is empty at {where}")
        corpus[doc_id] = document
    if not corpus:
        raise HarnessError(f"empty corpus: {path}")
    return corpus


def _load_queries(path: Path) -> dict[str, Query]:
    queries: dict[str, Query] = {}
    for line, record in _jsonl(path):
        where = f"{path}:{line}"
        query_id = _record_id(record, "id", where)
        if query_id in queries:
            raise HarnessError(f"duplicate query id {query_id!r} at {where}")
        queries[query_id] = Query(query_id, _record_string(record, "text", where))
    if not queries:
        raise HarnessError(f"empty queries: {path}")
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise HarnessError(f"dataset file does not exist: {path}")
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row or all(not value.strip() for value in row):
                continue
            if line_number == 1 and [value.strip().lower() for value in row] == [
                "query-id",
                "corpus-id",
                "score",
            ]:
                continue
            if len(row) != 3:
                raise HarnessError(f"qrels row must have 3 TSV fields: {path}:{line_number}")
            query_id, doc_id, score_text = (value.strip() for value in row)
            if not query_id or not doc_id:
                raise HarnessError(f"empty qrels id: {path}:{line_number}")
            try:
                score = int(score_text)
            except ValueError as error:
                raise HarnessError(
                    f"qrels relevance must be an integer: {path}:{line_number}"
                ) from error
            if score < 0:
                raise HarnessError(f"negative qrels relevance is unsupported: {path}:{line_number}")
            docs = qrels.setdefault(query_id, {})
            if doc_id in docs:
                raise HarnessError(f"duplicate qrel ({query_id}, {doc_id}): {path}:{line_number}")
            docs[doc_id] = score
    return qrels


def _load_candidates(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    candidates: dict[str, tuple[str, ...]] = {}
    for line, record in _jsonl(path):
        where = f"{path}:{line}"
        scene_id = _record_id(record, "scene_id", where)
        values = record.get("candidate_doc_ids")
        if not isinstance(values, list) or not all(
            not isinstance(value, bool) and isinstance(value, (str, int)) for value in values
        ):
            raise HarnessError(f"{where}.candidate_doc_ids must be a string/integer array")
        canonical_values = tuple(str(value) for value in values)
        if any(not value for value in canonical_values):
            raise HarnessError(f"{where}.candidate_doc_ids must not contain empty IDs")
        if len(canonical_values) != len(set(canonical_values)):
            raise HarnessError(f"duplicate candidate doc id at {where}")
        if scene_id in candidates:
            raise HarnessError(f"duplicate candidate scene_id {scene_id!r} at {where}")
        candidates[scene_id] = canonical_values
    return candidates


def load_task(config: TaskConfig) -> LoadedTask:
    corpus = _load_corpus(config.corpus)
    queries = _load_queries(config.queries)
    qrels = _load_qrels(config.qrels)
    candidates = _load_candidates(config.candidates)
    corpus_ids = set(corpus)
    query_ids = set(queries)
    unknown_qrels_queries = set(qrels) - query_ids
    if unknown_qrels_queries:
        raise HarnessError(f"qrels contain unknown queries: {sorted(unknown_qrels_queries)[:5]}")
    missing_qrels = query_ids - set(qrels)
    if missing_qrels:
        raise HarnessError(f"queries have no qrels rows: {sorted(missing_qrels)[:5]}")
    unknown_qrel_docs = {
        doc_id for docs in qrels.values() for doc_id in docs if doc_id not in corpus_ids
    }
    if unknown_qrel_docs:
        raise HarnessError(f"qrels contain unknown corpus ids: {sorted(unknown_qrel_docs)[:5]}")
    unknown_candidates = {
        doc_id for docs in candidates.values() for doc_id in docs if doc_id not in corpus_ids
    }
    if unknown_candidates:
        raise HarnessError(
            f"candidates contain unknown corpus ids: {sorted(unknown_candidates)[:5]}"
        )
    return LoadedTask(config, corpus, queries, qrels, candidates)


def _candidate_key(task: LoadedTask, query_id: str) -> str | None:
    policy = task.config.candidate_match
    if not task.candidates:
        return None
    if policy == "official":
        policy = "query-id" if task.config.dataset in _DIRECT_QUERY_ID_DATASETS else "prefix-2"
    if policy == "query-id":
        return query_id if query_id in task.candidates else None
    if policy == "prefix-2":
        key = "_".join(query_id.split("_")[:2])
        return key if key in task.candidates else None
    matches = [key for key in task.candidates if query_id == key or query_id.startswith(key + "_")]
    return max(matches, key=len) if matches else None


def resolve_candidates(task: LoadedTask, query_id: str) -> CandidateResolution:
    key = _candidate_key(task, query_id)
    if key is None:
        return CandidateResolution(tuple(task.corpus), None, bool(task.candidates))
    selected = set(task.candidates[key])
    # SubsetRetrieval uses a set for filtering, so candidate-file order is not a rank signal.
    ordered = tuple(doc_id for doc_id in task.corpus if doc_id in selected)
    return CandidateResolution(ordered, key, False)
