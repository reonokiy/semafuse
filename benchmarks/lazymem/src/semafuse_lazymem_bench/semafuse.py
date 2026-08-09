"""Bridge the benchmark protocol to Semafuse's production LazyMem runner.

The clean-room methods in :mod:`semafuse_lazymem_bench.methods` intentionally
reimplement the paper protocol in Python.  This module is separate: it sends
the harness's normalized ``Question`` objects to the Rust runner that uses the
actual Semafuse ingestion and ``lazymem`` strategy implementation.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import MemoryContext, Question


class SemafuseRunnerError(RuntimeError):
    """The Semafuse JSONL runner did not honor the benchmark protocol."""


def _json_line(question: Question) -> str:
    return json.dumps(
        question.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _string_ids(value: Any, *, field: str, question_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise SemafuseRunnerError(
            f"runner response for {question_id!r} has invalid {field}; "
            "expected a JSON array of non-empty strings"
        )
    if len(value) != len(set(value)):
        raise SemafuseRunnerError(
            f"runner response for {question_id!r} has duplicate {field}"
        )
    return tuple(value)


def _latency(value: Any, *, field: str, question_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemafuseRunnerError(
            f"runner response for {question_id!r} has non-numeric {field}"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SemafuseRunnerError(
            f"runner response for {question_id!r} has invalid {field}"
        )
    return result


class SemafuseLazyMemMethod:
    """A persistent JSONL client for Semafuse's production LazyMem benchmark.

    The process stays alive for one harness run.  That makes every
    ``construct`` call correspond to one actual Semafuse ingest/index/recall
    execution while avoiding a Rust process startup on each benchmark question.
    The runner must read one normalized question per stdin line and flush one
    response line before accepting the next request.
    """

    name = "semafuse-lazymem"

    def __init__(
        self,
        *,
        binary: str | Path,
        config_path: str | Path,
    ) -> None:
        self.binary = str(binary)
        self.config_path = Path(config_path)
        self._process: subprocess.Popen[str] | None = None

    def __enter__(self) -> SemafuseLazyMemMethod:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except BaseException:
            # Preserve the construct/evaluation failure, which is the useful
            # diagnosis. The runner has still been terminated or reaped by
            # ``close`` before this point.
            pass

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None:
            return self._process
        try:
            process = subprocess.Popen(
                [self.binary, "bench", "jsonl", "--config", str(self.config_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as error:
            raise SemafuseRunnerError(
                f"could not start Semafuse benchmark runner {self.binary!r}: {error}"
            ) from error
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            self._close_streams(process)
            raise SemafuseRunnerError("runner process did not expose text streams")
        self._process = process
        return process

    @staticmethod
    def _stderr(process: subprocess.Popen[str]) -> str:
        if process.poll() is None or process.stderr is None:
            return ""
        try:
            return process.stderr.read().strip()
        except OSError:
            return ""

    def _failed(self, process: subprocess.Popen[str], message: str) -> SemafuseRunnerError:
        detail = self._stderr(process)
        if detail:
            message = f"{message}; runner stderr: {detail}"
        return SemafuseRunnerError(message)

    @staticmethod
    def _close_streams(process: subprocess.Popen[str]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _response_to_context(
        self, response: Mapping[str, Any], question: Question
    ) -> MemoryContext:
        expected = {
            "question_id",
            "context",
            "retrieved_message_ids",
            "covered_message_ids",
            "retrieval_latency_s",
            "processing_latency_s",
            "metadata",
        }
        missing = expected.difference(response)
        if missing:
            raise SemafuseRunnerError(
                f"runner response for {question.id!r} is missing fields: {sorted(missing)}"
            )
        response_id = response["question_id"]
        if response_id != question.id:
            raise SemafuseRunnerError(
                f"runner response question_id {response_id!r} does not match "
                f"request {question.id!r}"
            )
        context = response["context"]
        if not isinstance(context, str):
            raise SemafuseRunnerError(
                f"runner response for {question.id!r} has non-string context"
            )
        runner_metadata = response["metadata"]
        if not isinstance(runner_metadata, Mapping):
            raise SemafuseRunnerError(
                f"runner response for {question.id!r} has non-object metadata"
            )
        return MemoryContext(
            text=context,
            retrieved_message_ids=_string_ids(
                response["retrieved_message_ids"],
                field="retrieved_message_ids",
                question_id=question.id,
            ),
            covered_message_ids=_string_ids(
                response["covered_message_ids"],
                field="covered_message_ids",
                question_id=question.id,
            ),
            retrieval_latency_s=_latency(
                response["retrieval_latency_s"],
                field="retrieval_latency_s",
                question_id=question.id,
            ),
            processing_latency_s=_latency(
                response["processing_latency_s"],
                field="processing_latency_s",
                question_id=question.id,
            ),
            metadata={
                "backend": "semafuse-product",
                "paper_comparable": False,
                "prompt_memory_kind": "compressed",
                "runner_metadata": dict(runner_metadata),
            },
        )

    def construct(self, question: Question) -> MemoryContext:
        process = self._start()
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(_json_line(question))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._failed(
                process,
                f"runner exited before accepting question {question.id!r}: {error}",
            ) from error
        line = process.stdout.readline()
        if not line:
            raise self._failed(
                process,
                f"runner exited before responding to question {question.id!r}",
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise SemafuseRunnerError(
                f"runner emitted invalid JSON for question {question.id!r}"
            ) from error
        if not isinstance(response, Mapping):
            raise SemafuseRunnerError(
                f"runner response for {question.id!r} must be a JSON object"
            )
        return self._response_to_context(response, question)

    def close(self) -> None:
        """Close stdin and require the persistent runner to exit successfully."""

        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self._close_streams(process)
            raise SemafuseRunnerError(
                "Semafuse benchmark runner did not exit after stdin closed"
            ) from error
        if return_code != 0:
            failure = self._failed(
                process,
                f"Semafuse benchmark runner exited with status {return_code}",
            )
            self._close_streams(process)
            raise failure
        self._close_streams(process)
