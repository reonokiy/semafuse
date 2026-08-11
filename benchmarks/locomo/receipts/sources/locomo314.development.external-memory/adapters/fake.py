"""Deterministic dependency-free adapter used only for contract tests."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..adapter import ExternalMemoryAdapter, MemoryItem
from ..errors import ConfigurationError
from ..protocol import Message

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _Stored:
    message: Message
    session: str
    timestamp: str | None
    ordinal: int


class FakeAdapter(ExternalMemoryAdapter):
    def __init__(self, config: Mapping[str, Any]):
        allowed = {"fail_on", "crash_on", "hang_on", "hang_seconds"}
        if set(config) - allowed:
            raise ConfigurationError(f"fake config has unknown fields: {sorted(set(config) - allowed)}")
        for key in ("fail_on", "crash_on", "hang_on"):
            if config.get(key) is not None and config[key] not in {"reset", "add", "search", "close"}:
                raise ConfigurationError(f"fake.{key} must be an operation name or null")
        hang_seconds = config.get("hang_seconds", 60.0)
        if isinstance(hang_seconds, bool) or not isinstance(hang_seconds, (int, float)) or hang_seconds <= 0:
            raise ConfigurationError("fake.hang_seconds must be positive")
        self._fail_on = config.get("fail_on")
        self._crash_on = config.get("crash_on")
        self._hang_on = config.get("hang_on")
        self._hang_seconds = float(hang_seconds)
        self._stored: list[_Stored] = []

    def _before(self, op: str) -> None:
        if self._crash_on == op:
            os._exit(70)
        if self._hang_on == op:
            time.sleep(self._hang_seconds)
        if self._fail_on == op:
            raise RuntimeError(f"intentional fake failure during {op}")

    def reset(self) -> None:
        self._before("reset")
        self._stored.clear()

    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        self._before("add")
        start = len(self._stored)
        self._stored.extend(
            _Stored(message=message, session=session, timestamp=timestamp, ordinal=start + index)
            for index, message in enumerate(messages)
        )
        return len(messages)

    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        self._before("search")
        query_words = set(_WORD.findall(query.casefold()))
        ranked = sorted(
            self._stored,
            key=lambda item: (
                -len(query_words & set(_WORD.findall(item.message.content.casefold()))),
                item.ordinal,
            ),
        )
        return [
            MemoryItem(
                id=f"fake-{item.ordinal}",
                text=item.message.content,
                score=float(len(query_words & set(_WORD.findall(item.message.content.casefold())))),
                source_ids=(item.message.source_id,),
                session=item.session,
                timestamp=item.timestamp,
            )
            for item in ranked[:top_k]
        ]

    def close(self) -> None:
        self._before("close")
        self._stored.clear()
