"""LazyMem and the paper's directly comparable RAG/oracle baselines."""

from __future__ import annotations

from typing import Protocol

from .processing import ConcurrentMemoryProcessor
from .retrieval import HybridRetriever
from .types import MemoryContext, Message, Question
from .windows import MergeRule, build_windows, covered_message_ids


class MemoryMethod(Protocol):
    name: str

    def construct(self, question: Question) -> MemoryContext: ...


def _render_messages(messages: list[Message]) -> str:
    return "\n".join(
        f"[{message.timestamp}] {message.role}: {message.content}" for message in messages
    )


class RagBaseline:
    def __init__(self, retriever: HybridRetriever, *, k: int) -> None:
        if k not in (20, 50):
            raise ValueError("paper baseline supports only RAG Top20 or Top50")
        self.retriever = retriever
        self.k = k
        self.name = f"rag{self.k}"

    def construct(self, question: Question) -> MemoryContext:
        result = self.retriever.retrieve(question.query, question.messages, limit=self.k)
        ranked_ids = tuple(item.message.id for item in result.ranked)
        chronological = sorted(
            (item.message for item in result.ranked), key=lambda message: message.index
        )
        return MemoryContext(
            text=_render_messages(chronological),
            retrieved_message_ids=ranked_ids,
            covered_message_ids=ranked_ids,
            retrieval_latency_s=result.latency_s,
            metadata={"top_k": self.k, "prompt_memory_kind": "chronological"},
        )


class OracleTurnBaseline:
    name = "oracle-turn"

    def construct(self, question: Question) -> MemoryContext:
        gold = set(question.gold_message_ids)
        messages = [message for message in question.messages if message.id in gold]
        ids = tuple(message.id for message in messages)
        return MemoryContext(
            text=_render_messages(messages),
            retrieved_message_ids=ids,
            covered_message_ids=ids,
            metadata={"prompt_memory_kind": "chronological"},
        )


class OracleSessionBaseline:
    name = "oracle-session"

    def construct(self, question: Question) -> MemoryContext:
        if question.dataset == "locomo":
            raise ValueError("Oracle Session is undefined for the paper's LoCoMo evaluation")
        sessions = set(question.gold_session_ids)
        messages = [message for message in question.messages if message.session_id in sessions]
        ids = tuple(message.id for message in messages)
        return MemoryContext(
            text=_render_messages(messages),
            retrieved_message_ids=ids,
            covered_message_ids=ids,
            metadata={"prompt_memory_kind": "chronological"},
        )


class LazyMemMethod:
    name = "lazymem"

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        processor: ConcurrentMemoryProcessor,
        retrieval_limit: int = 50,
        radius: int = 2,
        max_messages: int = 8,
        stride: int = 7,
        merge_rule: MergeRule = "paper",
    ) -> None:
        self.retriever = retriever
        self.processor = processor
        self.retrieval_limit = retrieval_limit
        self.radius = radius
        self.max_messages = max_messages
        self.stride = stride
        self.merge_rule = merge_rule

    def construct(self, question: Question) -> MemoryContext:
        retrieval = self.retriever.retrieve(
            question.query, question.messages, limit=self.retrieval_limit
        )
        windows = build_windows(
            question.messages,
            retrieval.ranked,
            radius=self.radius,
            max_messages=self.max_messages,
            stride=self.stride,
            merge_rule=self.merge_rule,
        )
        processing = self.processor.process(question, windows)
        return MemoryContext(
            text=processing.memory_text,
            retrieved_message_ids=tuple(item.message.id for item in retrieval.ranked),
            covered_message_ids=covered_message_ids(windows),
            retrieval_latency_s=retrieval.latency_s,
            processing_latency_s=processing.latency_s,
            metadata={
                "window_count": len(windows),
                "prompt_memory_kind": "compressed",
                "merge_rule": self.merge_rule,
                "mpm_request_latency_s": list(processing.request_latency_s),
                "mpm_prompt_tokens": processing.prompt_tokens,
                "mpm_completion_tokens": processing.completion_tokens,
            },
        )
