"""Thin adapter for the self-hosted Mem0 OSS Python package."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..adapter import ExternalMemoryAdapter, MemoryItem
from ..config import resolve_env_references
from ..errors import AdapterUnavailableError, ConfigurationError
from ..protocol import Message


class Mem0OssAdapter(ExternalMemoryAdapter):
    def __init__(self, config: Mapping[str, Any]):
        allowed = {"memory_config", "scope_id", "infer", "search_threshold", "rerank"}
        if set(config) != {"memory_config", "scope_id"} | (set(config) & {"infer", "search_threshold", "rerank"}):
            missing = {"memory_config", "scope_id"} - set(config)
            extra = set(config) - allowed
            raise ConfigurationError(f"Mem0 config fields invalid; missing={sorted(missing)}, extra={sorted(extra)}")
        if not isinstance(config["memory_config"], dict):
            raise ConfigurationError("Mem0 memory_config must be an object")
        if not isinstance(config["scope_id"], str) or not config["scope_id"]:
            raise ConfigurationError("Mem0 scope_id must be a non-empty string")
        if not isinstance(config.get("infer", True), bool) or not isinstance(config.get("rerank", False), bool):
            raise ConfigurationError("Mem0 infer and rerank must be booleans")
        threshold = config.get("search_threshold", 0.1)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ConfigurationError("Mem0 search_threshold must be in [0, 1]")
        resolved, self.secrets = resolve_env_references(config["memory_config"], "adapter.config.memory_config")
        self._memory_config = resolved
        self._scope_id = config["scope_id"]
        self._infer = config.get("infer", True)
        self._threshold = float(threshold)
        self._rerank = config.get("rerank", False)
        self._memory: Any = None

    def reset(self) -> None:
        try:
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Mem0 OSS is not installed; install methods/mem0_oss/requirements.txt in its own environment"
            ) from exc
        self._memory = Memory(MemoryConfig(**self._memory_config))
        self._memory.reset()

    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        if self._memory is None:
            raise RuntimeError("Mem0 has not been reset")
        # A batch-level metadata list would falsely credit every extracted memory
        # with every source in the batch. Per-message ingestion preserves exact
        # attribution; the ingestion-unit difference is recorded in the docs.
        for message in messages:
            self._memory.add(
                [{"role": message.role, "content": message.content}],
                user_id=self._scope_id,
                metadata={
                    "semafuse_session": session,
                    "semafuse_timestamp": timestamp or "",
                    "semafuse_source_ids": [message.source_id],
                },
                infer=self._infer,
            )
        return len(messages)

    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        if self._memory is None:
            raise RuntimeError("Mem0 has not been reset")
        raw = self._memory.search(
            query,
            top_k=top_k,
            filters={"user_id": self._scope_id},
            threshold=self._threshold,
            rerank=self._rerank,
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
            raise RuntimeError("Mem0 search returned an unexpected shape")
        normalized = []
        for item in raw["results"][:top_k]:
            if not isinstance(item, dict) or not item.get("id") or not item.get("memory"):
                raise RuntimeError("Mem0 search item is missing id or memory")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            source_ids = metadata.get("semafuse_source_ids", item.get("semafuse_source_ids", []))
            if not isinstance(source_ids, list) or any(not isinstance(source_id, str) for source_id in source_ids):
                source_ids = []
            score = item.get("score")
            if score is not None:
                score = float(score)
                if not math.isfinite(score):
                    raise RuntimeError("Mem0 returned a non-finite score")
            timestamp = metadata.get("semafuse_timestamp", item.get("semafuse_timestamp")) or None
            session = metadata.get("semafuse_session", item.get("semafuse_session")) or None
            normalized.append(
                MemoryItem(
                    id=str(item["id"]),
                    text=str(item["memory"]),
                    score=score,
                    source_ids=tuple(source_ids),
                    session=session,
                    timestamp=timestamp,
                )
            )
        return normalized

    def close(self) -> None:
        self._memory = None
