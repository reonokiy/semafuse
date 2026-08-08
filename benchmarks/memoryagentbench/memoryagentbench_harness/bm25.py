from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .model import HarnessError
from .prompts import official_bm25_query
from .protocol import read_jsonl


def run_bm25(inputs_path: str | Path, output_path: str | Path) -> int:
    """Create retrieval-only predictions with the official rank_bm25 defaults."""

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as error:  # pragma: no cover - exercised by real runs
        raise HarnessError("the BM25 baseline requires rank-bm25==0.2.2") from error

    records = read_jsonl(inputs_path)
    current_case: str | None = None
    memory_ids: list[str] = []
    documents: list[str] = []
    ranker: Any = None
    result_count = 0

    with Path(output_path).open("w", encoding="utf-8") as handle:
        for record in records:
            op = record.get("op")
            case_id = record.get("case_id")
            if op == "reset":
                current_case = str(case_id)
                memory_ids = []
                documents = []
                ranker = None
            elif op == "add":
                if case_id != current_case:
                    raise HarnessError("add event occurred outside its active case")
                memory_ids.append(str(record["memory_id"]))
                documents.append(str(record["paper_message"]))
            elif op == "query":
                if case_id != current_case or not documents:
                    raise HarnessError("query event occurred before its case memories")
                if ranker is None:
                    ranker = BM25Okapi([document.split() for document in documents])
                paper_query = str(record["paper_message"])
                query_tokens = official_bm25_query(paper_query).split()
                scores = ranker.get_scores(query_tokens)
                top_k = min(int(record["top_k"]), len(memory_ids))
                # BM25Retriever delegates to rank_bm25.get_top_n(), whose ordering
                # is numpy.argsort(scores)[::-1]. Keep the same reverse-index tie
                # behavior instead of introducing a new document-id policy.
                order = scores.argsort()[::-1][:top_k].tolist()
                result = {
                    "schema_version": 1,
                    "case_id": current_case,
                    "query_id": record["query_id"],
                    "prediction": None,
                    "retrieved_memory_ids": [memory_ids[index] for index in order],
                    "method": "rank_bm25-0.2.2-retrieval-only",
                }
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                result_count += 1
            else:
                raise HarnessError(f"unsupported event op: {op!r}")
    return result_count
