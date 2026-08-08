"""Adapter interface and common normalization policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .protocol import Message


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    score: float | None = None
    source_ids: tuple[str, ...] = ()
    session: str | None = None
    timestamp: str | None = None


class ExternalMemoryAdapter(ABC):
    """Minimal translation layer; implementations remain in upstream packages."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state belonging to the current benchmark history."""

    @abstractmethod
    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        """Ingest a chronological message group and return its accepted message count."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        """Return at most top_k native-ranked memories."""

    def close(self) -> None:
        """Release resources when the upstream package exposes no close method."""


class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        """Return reader-tokenizer token count for text."""


class WhitespaceTokenCounter(TokenCounter):
    """Deterministic test-only counter; one non-whitespace span is one token."""

    def count(self, text: str) -> int:
        return len(text.split())


class HuggingFaceTokenCounter(TokenCounter):
    def __init__(self, model: str, revision: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            from .errors import AdapterUnavailableError

            raise AdapterUnavailableError(
                "transformers is required for tokenizer.kind=huggingface; install the method environment"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def build_token_counter(raw: Any) -> TokenCounter:
    if not isinstance(raw, dict):
        raise ValueError("tokenizer must be an object")
    kind = raw.get("kind")
    if kind == "whitespace" and set(raw) == {"kind"}:
        return WhitespaceTokenCounter()
    if kind == "huggingface" and set(raw) == {"kind", "model", "revision"}:
        model, revision = raw["model"], raw["revision"]
        if not isinstance(model, str) or not model or not isinstance(revision, str) or not revision:
            raise ValueError("huggingface tokenizer model and revision must be non-empty strings")
        return HuggingFaceTokenCounter(model, revision)
    raise ValueError("tokenizer must be whitespace or pinned huggingface {model, revision}")


def enforce_limits(
    items: Sequence[MemoryItem], top_k: int, max_context_tokens: int, counter: TokenCounter
) -> Mapping[str, Any]:
    selected: list[dict[str, Any]] = []
    parts: list[str] = []
    truncated = len(items) > top_k
    for item in items[:top_k]:
        text = item.text.strip()
        if not text:
            raise ValueError("adapter returned an empty memory")
        token_count = counter.count(text)
        if token_count < 1:
            raise ValueError("tokenizer returned zero tokens for a non-empty memory")
        candidate_context = "\n\n".join([*parts, text])
        if counter.count(candidate_context) > max_context_tokens:
            truncated = True
            break
        parts.append(text)
        selected.append(
            {
                "id": item.id,
                "text": text,
                "score": item.score,
                "source_ids": list(dict.fromkeys(item.source_ids)),
                "session": item.session,
                "timestamp": item.timestamp,
                "token_count": token_count,
            }
        )
    context = "\n\n".join(parts)
    return {
        "items": selected,
        "context": context,
        "context_tokens": counter.count(context),
        "truncated": truncated,
    }
