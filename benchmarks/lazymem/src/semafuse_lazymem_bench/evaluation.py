"""Answer-generation and judge orchestration with resumable outputs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .endpoints import ChatCompletionClient, CompletionResult
from .methods import MemoryMethod
from .resume import JsonlJournal
from .types import MemoryContext, Question


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Offline smoke-test counter; never label its output paper-comparable."""

    name = "approximate-whitespace-punctuation"

    def count(self, text: str) -> int:
        import re

        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


class TransformersTokenCounter:
    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install the 'tokenizer' extra for exact token counts") from error
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.name = model_name

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True, slots=True)
class AnswerResult:
    prediction: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    correct: bool
    latency_s: float
    raw: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Answerer(Protocol):
    def answer(self, question: Question, memory: MemoryContext) -> AnswerResult: ...


class Judge(Protocol):
    def judge(self, question: Question, prediction: str) -> JudgeResult: ...


class EndpointAnswerer:
    SYSTEM_PROMPT = "Answer the user question concisely and faithfully from the provided memory."

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        configured_extra_body = dict(extra_body or {})
        chat_template_kwargs = dict(
            configured_extra_body.get("chat_template_kwargs", {})
        )
        chat_template_kwargs.setdefault("enable_thinking", enable_thinking)
        configured_extra_body["chat_template_kwargs"] = chat_template_kwargs
        self.extra_body = configured_extra_body

    def answer(self, question: Question, memory: MemoryContext) -> AnswerResult:
        compressed = memory.metadata.get("prompt_memory_kind") == "compressed"
        lines = [
            (
                "You are answering a user question using compressed chronological "
                "conversation memory."
                if compressed
                else "You are answering a user question using chronological conversation memory."
            ),
            "Use the memory when it is relevant. If the memory is insufficient, say what is missing instead of inventing facts.",
            "",
            "# Query",
        ]
        if question.query_time:
            lines.append(f"Query Time: {question.query_time}")
        lines.extend(
            [
                f"Query: {question.query}",
                "",
                "# Memory",
                memory.text.strip()
                or ("(No kept memory was found.)" if compressed else "(No gold memory was found.)"),
                "",
                "# Answer",
            ]
        )
        prompt = "\n".join(lines)
        response = self.client.complete(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        return AnswerResult(
            prediction=response.text.strip(),
            latency_s=response.latency_s,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            metadata=response.metadata,
        )


class EndpointBinaryJudge:
    """Generic blinded binary judge.

    Paper-comparable runs should replace this prompt with each benchmark's
    official evaluator through the same ``Judge`` protocol.
    """

    SYSTEM_PROMPT = (
        "Judge whether a prediction correctly answers a question relative to the "
        "reference. Ignore stylistic differences. Return only JSON with exactly one "
        "boolean field named correct."
    )

    def __init__(self, client: ChatCompletionClient) -> None:
        self.client = client

    def judge(self, question: Question, prediction: str) -> JudgeResult:
        blinded = json.dumps(
            {
                "question": question.query,
                "reference_answer": question.answer,
                "prediction": prediction,
            },
            ensure_ascii=False,
        )
        response = self.client.complete(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": blinded},
            ],
            temperature=0.0,
            max_tokens=32,
        )
        try:
            value = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise ValueError("judge did not return JSON") from error
        if not isinstance(value, dict) or set(value) != {"correct"}:
            raise ValueError("judge output must contain exactly the 'correct' field")
        if not isinstance(value["correct"], bool):
            raise ValueError("judge 'correct' field must be boolean")
        return JudgeResult(
            correct=value["correct"],
            latency_s=response.latency_s,
            raw=response.text,
            metadata=response.metadata,
        )


class EvaluationRunner:
    def __init__(
        self,
        *,
        method: MemoryMethod,
        answerer: Answerer,
        journal: JsonlJournal,
        token_counter: TokenCounter,
        run_id: str,
        run_metadata: Mapping[str, Any] | None = None,
        judge: Judge | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        existing_run_ids = journal.field_values("run_id")
        if journal.completed_keys and existing_run_ids != {run_id}:
            raise ValueError(
                "output journal belongs to a different run_id; use a new output file"
            )
        self.method = method
        self.answerer = answerer
        self.journal = journal
        self.token_counter = token_counter
        self.judge = judge
        self.run_id = run_id
        self.run_metadata = dict(run_metadata or {})

    def run(
        self,
        questions: Sequence[Question],
        *,
        warmup_runs: int = 0,
        measured_runs: int = 1,
    ) -> dict[str, int]:
        if warmup_runs < 0 or measured_runs <= 0:
            raise ValueError("warmup_runs must be non-negative and measured_runs positive")
        latency_protocol = warmup_runs > 0 or measured_runs > 1
        if latency_protocol and self.judge is not None:
            raise ValueError("latency protocol excludes correctness-judge calls")

        def execute(question: Question) -> tuple[MemoryContext, int, AnswerResult, float]:
            started = time.perf_counter()
            memory = self.method.construct(question)
            memory_tokens = (
                memory.memory_token_count
                if memory.memory_token_count is not None
                else self.token_counter.count(memory.text)
            )
            answer = self.answerer.answer(question, memory)
            return memory, memory_tokens, answer, time.perf_counter() - started

        for _ in range(warmup_runs):
            for question in questions:
                execute(question)

        completed = 0
        skipped = 0
        for measurement_index in range(measured_runs):
            for question in questions:
                run_key = (
                    f"{self.run_id}:latency:{measurement_index}:{question.dataset}:"
                    f"{self.method.name}:{question.id}"
                    if latency_protocol
                    else f"{self.run_id}:{question.dataset}:{self.method.name}:{question.id}"
                )
                if run_key in self.journal.completed_keys:
                    skipped += 1
                    continue
                memory, memory_tokens, answer, online_latency = execute(question)
                judge_started = time.perf_counter()
                judge = self.judge.judge(question, answer.prediction) if self.judge else None
                evaluation_wall_latency = online_latency + (
                    time.perf_counter() - judge_started
                )
                self.journal.append(
                {
                    "schema_version": 1,
                    "record_kind": "latency" if latency_protocol else "evaluation",
                    "measurement_index": measurement_index if latency_protocol else None,
                    "run_id": self.run_id,
                    "run_metadata": self.run_metadata,
                    "run_key": run_key,
                    "dataset": question.dataset,
                    "method": self.method.name,
                    "question_id": question.id,
                    "question_type": question.question_type,
                    "question": question.query,
                    "prediction": answer.prediction,
                    "reference_answer": question.answer,
                    "dataset_metadata": {
                        key: question.metadata[key]
                        for key in ("category", "is_abstention")
                        if key in question.metadata
                    },
                    "judge_label": judge.correct if judge else None,
                    "judge_raw": judge.raw if judge else None,
                    "gold_message_ids": list(question.gold_message_ids),
                    "retrieved_message_ids": list(memory.retrieved_message_ids),
                    "covered_message_ids": list(memory.covered_message_ids),
                    "memory_tokens": memory_tokens,
                    "tokenizer": self.token_counter.name,
                    "latency_s": {
                        "retrieval": memory.retrieval_latency_s,
                        "processing": memory.processing_latency_s,
                        "answer": answer.latency_s,
                        "judge": judge.latency_s if judge else 0.0,
                        "total": online_latency,
                        "evaluation_wall": evaluation_wall_latency,
                    },
                    "answer_usage": {
                        "prompt_tokens": answer.prompt_tokens,
                        "completion_tokens": answer.completion_tokens,
                    },
                    "method_metadata": dict(memory.metadata),
                    "answer_metadata": dict(answer.metadata),
                    "judge_metadata": dict(judge.metadata) if judge else {},
                }
                )
                completed += 1
        summary = {"completed": completed, "skipped": skipped}
        if latency_protocol:
            summary.update(
                {
                    "warmup_executions": warmup_runs * len(questions),
                    "measured_runs": measured_runs,
                }
            )
        return summary
