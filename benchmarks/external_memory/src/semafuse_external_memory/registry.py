"""Adapter registry with imports delayed until the selected method is built."""

from __future__ import annotations

from typing import Any, Mapping

from .adapter import ExternalMemoryAdapter
from .errors import ConfigurationError


def build_adapter(kind: str, config: Mapping[str, Any]) -> ExternalMemoryAdapter:
    if kind == "fake":
        from .adapters.fake import FakeAdapter

        return FakeAdapter(config)
    if kind == "amem":
        from .adapters.amem import AMemAdapter

        return AMemAdapter(config)
    if kind == "amem_paper":
        from .adapters.amem_paper import AMemPaperAdapter

        return AMemPaperAdapter(config)
    if kind == "mem0_oss":
        from .adapters.mem0_oss import Mem0OssAdapter

        return Mem0OssAdapter(config)
    raise ConfigurationError(f"unknown external memory adapter: {kind!r}")
