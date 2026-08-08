"""Built-in lightweight BM25, external import, and Semafuse JSONL rankers."""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import CandidateResolution, HarnessError, LoadedTask, Query, RankingEntry
from .protocol import load_predictions, normalize_ranking


_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_ENGLISH_STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been before being "
    "below between both but by can did do does doing down during each few for from further had has "
    "have having he her here hers herself him himself his how i if in into is it its itself just me "
    "more most my myself no nor not now of off on once only or other our ours ourselves out over own "
    "same she should so some such than that the their theirs them themselves then there these they "
    "this those through to too under until up very was we were what when where which while who whom "
    "why will with you your yours yourself yourselves".split()
)


def _tokens(text: str, remove_stopwords: bool) -> list[str]:
    words = _TOKEN_PATTERN.findall(text.lower())
    return [word for word in words if not remove_stopwords or word not in _ENGLISH_STOPWORDS]


class BM25Ranker:
    """A stdlib Lucene-BM25 diagnostic baseline over global task statistics."""

    stages = ("bm25",)

    def __init__(self, task: LoadedTask, options: Mapping[str, Any], use_instruction: bool):
        self.task = task
        self.k1 = float(options.get("k1", 1.5))
        self.b = float(options.get("b", 0.75))
        self.remove_stopwords = bool(options.get("remove_stopwords", True))
        self.use_instruction = use_instruction
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise HarnessError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.doc_order = tuple(task.corpus)
        self.lengths: dict[str, int] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for document in task.corpus.values():
            counts = Counter(_tokens(document.combined_text, self.remove_stopwords))
            self.lengths[document.id] = sum(counts.values())
            for token, frequency in counts.items():
                self.postings[token].append((document.id, frequency))
        self.average_length = sum(self.lengths.values()) / len(self.lengths) or 1.0
        self.document_count = len(self.lengths)

    def rank(
        self, query: Query, candidates: CandidateResolution
    ) -> tuple[dict[str, list[RankingEntry]], dict[str, Any]]:
        text = query.text
        if self.use_instruction and self.task.config.instruction:
            text = f"{self.task.config.instruction}\n{text}"
        query_counts = Counter(_tokens(text, self.remove_stopwords))
        candidate_set = set(candidates.doc_ids)
        scores = {doc_id: 0.0 for doc_id in candidates.doc_ids}
        for token, query_frequency in query_counts.items():
            postings = self.postings.get(token, ())
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for doc_id, term_frequency in postings:
                if doc_id not in candidate_set:
                    continue
                norm = self.k1 * (
                    1 - self.b + self.b * self.lengths[doc_id] / self.average_length
                )
                # bm25s 0.3.6's default Lucene variant omits (k1 + 1).
                scores[doc_id] += (
                    query_frequency * inverse_document_frequency * term_frequency / (norm + term_frequency)
                )
        entries = [RankingEntry(doc_id, score) for doc_id, score in scores.items()]
        entries.sort(key=lambda entry: entry.doc_id, reverse=True)
        entries.sort(key=lambda entry: entry.score, reverse=True)
        return {"bm25": entries}, {"candidate_scene_id": candidates.scene_id}

    def close(self) -> None:
        return None


class ExternalRanker:
    def __init__(self, path: Path, task_ids: Sequence[str]):
        default = task_ids[0] if len(task_ids) == 1 else None
        self.predictions = load_predictions(path, default_task_id=default)

    def rank_for(
        self, task: LoadedTask, query: Query, candidates: CandidateResolution
    ) -> tuple[dict[str, list[RankingEntry]], dict[str, Any]]:
        key = (task.config.id, query.id)
        if key not in self.predictions:
            raise HarnessError(f"external predictions have no row for {key}")
        rankings, metadata = self.predictions[key]
        return rankings, dict(metadata)

    def close(self) -> None:
        return None


class SemafuseJsonlRanker:
    stages = ("dense", "bm25", "rrf", "retrieved")

    def __init__(self, options: Mapping[str, Any], use_instruction: bool):
        command = options.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise HarnessError("semafuse-jsonl ranker.command must be a non-empty string array")
        self.command = command
        self.timeout = float(options.get("timeout_seconds", 600))
        self.max_response_bytes = int(options.get("max_response_bytes", 64 * 1024 * 1024))
        if self.timeout <= 0 or self.max_response_bytes <= 0:
            raise HarnessError("semafuse-jsonl timeout_seconds and max_response_bytes must be positive")
        self.use_instruction = use_instruction
        self.instruction_template = str(
            options.get("instruction_template", "Instruct: {instruction}\nQuery: {query}")
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._stdout_buffer = bytearray()
        self._closed = False

    def _abort(self) -> None:
        if self._closed:
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._closed = True

    def _wait(self, file_object: Any, event: int, deadline: float, query_id: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._abort()
            raise HarnessError(f"semafuse-jsonl timed out after {self.timeout:g}s on {query_id}")
        selector = selectors.DefaultSelector()
        try:
            selector.register(file_object, event)
            if not selector.select(remaining):
                self._abort()
                raise HarnessError(
                    f"semafuse-jsonl timed out after {self.timeout:g}s on {query_id}"
                )
        finally:
            selector.close()

    def _exchange(self, payload: bytes, query_id: str) -> bytes:
        assert self.process.stdin is not None and self.process.stdout is not None
        deadline = time.monotonic() + self.timeout
        view = memoryview(payload)
        while view:
            self._wait(self.process.stdin, selectors.EVENT_WRITE, deadline, query_id)
            try:
                written = os.write(self.process.stdin.fileno(), view)
            except BlockingIOError:
                continue
            except BrokenPipeError as error:
                self._abort()
                raise HarnessError(f"semafuse-jsonl closed stdin on {query_id}") from error
            view = view[written:]

        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                return line
            self._wait(self.process.stdout, selectors.EVENT_READ, deadline, query_id)
            try:
                chunk = os.read(self.process.stdout.fileno(), 64 * 1024)
            except BlockingIOError:
                continue
            if not chunk:
                partial_bytes = len(self._stdout_buffer)
                self._abort()
                detail = f" with {partial_bytes} unterminated response bytes" if partial_bytes else ""
                raise HarnessError(
                    f"semafuse-jsonl closed stdout before a complete newline-framed response{detail} "
                    f"on {query_id}"
                )
            self._stdout_buffer.extend(chunk)
            if len(self._stdout_buffer) > self.max_response_bytes:
                self._abort()
                raise HarnessError(
                    f"semafuse-jsonl response exceeded {self.max_response_bytes} bytes on {query_id}"
                )

    def rank_for(
        self, task: LoadedTask, query: Query, candidates: CandidateResolution
    ) -> tuple[dict[str, list[RankingEntry]], dict[str, Any]]:
        if task.config.candidates is None or candidates.used_full_corpus_fallback:
            raise HarnessError(
                f"semafuse-jsonl refuses task/query without an explicit candidate pool: "
                f"{task.config.id}/{query.id}; cold per-query full-corpus ingestion is not comparable"
            )
        query_text = query.text
        if self.use_instruction:
            query_text = self.instruction_template.format(
                instruction=task.config.instruction, query=query.text
            )
        messages = []
        for index, doc_id in enumerate(candidates.doc_ids):
            document = task.corpus[doc_id]
            messages.append(
                {
                    "id": doc_id,
                    "index": index,
                    "role": "user",
                    "content": document.combined_text,
                    "timestamp": "",
                    "session_id": f"lmeb-document-{index:08d}",
                }
            )
        question = {
            "id": query.id,
            "dataset": task.config.dataset,
            "question_type": task.config.subset,
            "query": query_text,
            "query_time": "",
            "messages": messages,
        }
        if self.process.poll() is not None:
            raise HarnessError(f"semafuse-jsonl process exited with {self.process.returncode}")
        payload = (
            json.dumps(question, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        line = self._exchange(payload, query.id)
        try:
            response = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HarnessError(f"invalid semafuse JSONL response for {query.id}: {error}") from error
        if response.get("question_id") != query.id:
            raise HarnessError(
                f"semafuse response id mismatch: expected {query.id!r}, got {response.get('question_id')!r}"
            )
        values = response.get("stage_rankings")
        if not isinstance(values, dict):
            raise HarnessError("semafuse response has no stage_rankings object")
        rankings = {
            stage: normalize_ranking(values.get(stage), f"stage_rankings.{stage}")
            for stage in self.stages
        }
        metadata = {
            "candidate_scene_id": candidates.scene_id,
            "retrieval_latency_s": response.get("retrieval_latency_s"),
            "processing_latency_s": response.get("processing_latency_s"),
        }
        return rankings, metadata

    def close(self) -> None:
        if self._closed:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._closed = True


def resolve_external_path(value: Any, manifest_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise HarnessError("external ranker.predictions must be a non-empty path string")
    path = Path(value).expanduser()
    return (manifest_dir / path).resolve() if not path.is_absolute() else path.resolve()
