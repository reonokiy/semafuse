"""Thin adapter for the official agiresearch/A-mem system package."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..adapter import ExternalMemoryAdapter, MemoryItem
from ..config import resolve_env_references
from ..errors import AdapterUnavailableError, ConfigurationError
from ..protocol import Message


class AMemAdapter(ExternalMemoryAdapter):
    def __init__(self, config: Mapping[str, Any]):
        allowed = {"model_name", "llm_backend", "llm_model", "evo_threshold", "api_key_env"}
        if set(config) - allowed:
            raise ConfigurationError(f"A-MEM config has unknown fields: {sorted(set(config) - allowed)}")
        backend = config.get("llm_backend", "openai")
        if backend not in {"openai", "ollama"}:
            raise ConfigurationError("A-MEM llm_backend must be openai or ollama")
        if backend == "openai" and "api_key_env" not in config:
            raise ConfigurationError("A-MEM openai backend requires api_key_env")
        resolved, self.secrets = resolve_env_references(dict(config), "adapter.config")
        self._config = resolved
        self._memory: Any = None
        self._provenance: dict[str, tuple[tuple[str, ...], str, str | None]] = {}

    def reset(self) -> None:
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as exc:
            raise AdapterUnavailableError(
                "A-MEM is not installed; install methods/amem/requirements.txt in its own environment"
            ) from exc
        self._memory = AgenticMemorySystem(
            model_name=self._config.get("model_name", "all-MiniLM-L6-v2"),
            llm_backend=self._config.get("llm_backend", "openai"),
            llm_model=self._config.get("llm_model", "gpt-4o-mini"),
            evo_threshold=self._config.get("evo_threshold", 100),
            api_key=self._config.get("api_key"),
        )
        self._provenance.clear()

    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        if self._memory is None:
            raise RuntimeError("A-MEM has not been reset")
        for message in messages:
            memory_id = self._memory.add_note(
                f"[{message.role}] {message.content}", time=timestamp, category=session
            )
            self._provenance[str(memory_id)] = ((message.source_id,), session, timestamp)
        return len(messages)

    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        if self._memory is None:
            raise RuntimeError("A-MEM has not been reset")
        raw_results = self._memory.search_agentic(query, k=top_k)
        if not isinstance(raw_results, list):
            raise RuntimeError("A-MEM search returned an unexpected shape")
        normalized = []
        for raw in raw_results[:top_k]:
            if not isinstance(raw, dict) or not raw.get("id") or not raw.get("content"):
                raise RuntimeError("A-MEM search item is missing id or content")
            memory_id = str(raw["id"])
            score = raw.get("score")
            if score is not None:
                score = float(score)
                if not math.isfinite(score):
                    raise RuntimeError("A-MEM returned a non-finite score")
            source_ids, session, timestamp = self._provenance.get(
                memory_id, ((), raw.get("category"), raw.get("timestamp"))
            )
            normalized.append(
                MemoryItem(
                    id=memory_id,
                    text=str(raw["content"]),
                    score=score,
                    source_ids=source_ids,
                    session=session,
                    timestamp=timestamp,
                )
            )
        return normalized

    def close(self) -> None:
        self._memory = None
        self._provenance.clear()
