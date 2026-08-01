"""Dataset-neutral immutable types used throughout the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


JsonObject = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    """One chronological role message from an interaction history."""

    id: str
    index: int
    role: str
    content: str
    timestamp: str = ""
    session_id: str = ""
    has_answer: bool = False
    metadata: JsonObject = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("message id must be non-empty")
        if self.index < 0:
            raise ValueError("message index must be non-negative")
        if not self.role:
            raise ValueError("message role must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    @property
    def global_message_index(self) -> int:
        """Original history index exposed to the memory-processing model."""

        value = self.metadata.get("global_message_index", self.index)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"message {self.id} has an invalid global_message_index {value!r}"
            )
        return value

    @property
    def retrieval_text(self) -> str:
        """Exact text indexed by the paper's retrieval artifacts."""

        value = self.metadata.get("retrieval_text", self.content)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"message {self.id} has invalid retrieval_text")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "has_answer": self.has_answer,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        return cls(
            id=str(value["id"]),
            index=int(value["index"]),
            role=str(value["role"]),
            content=str(value["content"]),
            timestamp=str(value.get("timestamp", "")),
            session_id=str(value.get("session_id", "")),
            has_answer=bool(value.get("has_answer", False)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Question:
    """A question plus the complete interaction history available to memory."""

    id: str
    dataset: str
    question_type: str
    query: str
    answer: str
    query_time: str
    messages: tuple[Message, ...]
    gold_message_ids: tuple[str, ...] = ()
    gold_session_ids: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.id or not self.dataset or not self.question_type:
            raise ValueError("question id, dataset, and question_type are required")
        if not isinstance(self.query, str) or not isinstance(self.answer, str):
            raise TypeError("query and answer must be strings")
        ids = [message.id for message in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate message id in question {self.id}")
        indices = [message.index for message in self.messages]
        if indices != list(range(len(self.messages))):
            raise ValueError(
                f"message indices for {self.id} must be contiguous and chronological"
            )
        unknown_gold = set(self.gold_message_ids).difference(ids)
        if unknown_gold:
            raise ValueError(f"gold message ids are absent from history: {unknown_gold}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question_type": self.question_type,
            "query": self.query,
            "answer": self.answer,
            "query_time": self.query_time,
            "messages": [message.to_dict() for message in self.messages],
            "gold_message_ids": list(self.gold_message_ids),
            "gold_session_ids": list(self.gold_session_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Question:
        return cls(
            id=str(value["id"]),
            dataset=str(value["dataset"]),
            question_type=str(value["question_type"]),
            query=str(value["query"]),
            answer=str(value["answer"]),
            query_time=str(value.get("query_time", "")),
            messages=tuple(Message.from_dict(item) for item in value["messages"]),
            gold_message_ids=tuple(str(item) for item in value.get("gold_message_ids", [])),
            gold_session_ids=tuple(str(item) for item in value.get("gold_session_ids", [])),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Method-produced evidence passed to the answer model."""

    text: str
    retrieved_message_ids: tuple[str, ...] = ()
    covered_message_ids: tuple[str, ...] = ()
    memory_token_count: int | None = None
    retrieval_latency_s: float = 0.0
    processing_latency_s: float = 0.0
    metadata: JsonObject = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.memory_token_count is not None and self.memory_token_count < 0:
            raise ValueError("memory_token_count must be non-negative")
        if self.retrieval_latency_s < 0 or self.processing_latency_s < 0:
            raise ValueError("latencies must be non-negative")
