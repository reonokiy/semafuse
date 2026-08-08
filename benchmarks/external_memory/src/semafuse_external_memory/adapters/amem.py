"""Thin adapter for the official agiresearch/A-mem system package."""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from ..adapter import ExternalMemoryAdapter, MemoryItem
from ..config import resolve_env_references
from ..errors import AdapterUnavailableError, ConfigurationError
from ..protocol import Message


class _UpstreamErrorCounter(logging.Handler):
    """Count upstream errors without retaining potentially sensitive log text."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.count += 1


class AMemAdapter(ExternalMemoryAdapter):
    def __init__(self, config: Mapping[str, Any]):
        allowed = {
            "model_name",
            "model_revision",
            "model_cache_dir",
            "llm_backend",
            "llm_model",
            "evo_threshold",
            "api_key_env",
            "openai_base_url",
        }
        if set(config) - allowed:
            raise ConfigurationError(f"A-MEM config has unknown fields: {sorted(set(config) - allowed)}")
        if "llm_backend" not in config:
            raise ConfigurationError("A-MEM requires an explicit llm_backend")
        backend = config["llm_backend"]
        if backend not in {"openai", "ollama"}:
            raise ConfigurationError("A-MEM llm_backend must be openai or ollama")
        for field in ("model_name", "llm_model"):
            if not isinstance(config.get(field), str) or not str(config[field]).strip():
                raise ConfigurationError(f"A-MEM requires an explicit non-empty {field}")
        if str(config["model_name"]).count("/") != 1:
            raise ConfigurationError("A-MEM model_name must be an explicit Hugging Face owner/repository id")
        model_revision = config.get("model_revision")
        if not isinstance(model_revision, str) or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
            raise ConfigurationError("A-MEM model_revision must be an immutable 40-character lowercase SHA")
        model_cache_dir = config.get("model_cache_dir")
        if model_cache_dir is not None and (
            not isinstance(model_cache_dir, str) or not model_cache_dir.strip()
        ):
            raise ConfigurationError("A-MEM model_cache_dir must be a non-empty string when provided")
        threshold = config.get("evo_threshold", 100)
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ConfigurationError("A-MEM evo_threshold must be a positive integer")
        if backend == "openai":
            if "api_key_env" not in config:
                raise ConfigurationError("A-MEM openai backend requires api_key_env")
            base_url = config.get("openai_base_url")
            if not isinstance(base_url, str) or not base_url.strip():
                raise ConfigurationError("A-MEM openai backend requires an explicit openai_base_url")
            parsed = urlsplit(base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ConfigurationError(
                    "A-MEM openai_base_url must be an http(s) URL without credentials, query, or fragment"
                )
        elif "api_key_env" in config or "openai_base_url" in config:
            raise ConfigurationError("A-MEM ollama backend does not accept api_key_env or openai_base_url")
        resolved, self.secrets = resolve_env_references(dict(config), "adapter.config")
        self._config = resolved
        self._memory: Any = None
        self._provenance: dict[str, tuple[tuple[str, ...], str, str | None]] = {}
        self._upstream_logger: logging.Logger | None = None
        self._upstream_error_counter: _UpstreamErrorCounter | None = None

    def _detach_upstream_error_counter(self) -> None:
        logger = getattr(self, "_upstream_logger", None)
        counter = getattr(self, "_upstream_error_counter", None)
        if logger is not None and counter is not None:
            logger.removeHandler(counter)
            counter.close()
        self._upstream_logger = None
        self._upstream_error_counter = None

    def _upstream_error_count(self) -> int:
        counter = getattr(self, "_upstream_error_counter", None)
        return counter.count if counter is not None else 0

    def reset(self) -> None:
        self._detach_upstream_error_counter()
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AdapterUnavailableError(
                "A-MEM is not installed; install methods/amem/requirements.txt in its own environment"
            ) from exc
        upstream_logger = logging.getLogger("agentic_memory.memory_system")
        error_counter = _UpstreamErrorCounter()
        upstream_logger.addHandler(error_counter)
        self._upstream_logger = upstream_logger
        self._upstream_error_counter = error_counter
        model_path = snapshot_download(
            repo_id=self._config["model_name"],
            revision=self._config["model_revision"],
            cache_dir=self._config.get("model_cache_dir"),
        )
        previous_base_url = os.environ.get("OPENAI_BASE_URL")
        if self._config["llm_backend"] == "openai":
            # The pinned upstream constructor exposes an API key but no base
            # URL.  OpenAI's SDK resolves OPENAI_BASE_URL during construction,
            # so scope the explicit config value to that call and restore the
            # worker environment immediately afterwards.
            os.environ["OPENAI_BASE_URL"] = self._config["openai_base_url"]
        try:
            self._memory = AgenticMemorySystem(
                # The pinned upstream exposes only model_name. Resolve the
                # immutable Hub revision first and pass the local snapshot so
                # SentenceTransformer cannot follow a mutable branch.
                model_name=model_path,
                llm_backend=self._config["llm_backend"],
                llm_model=self._config["llm_model"],
                evo_threshold=self._config.get("evo_threshold", 100),
                api_key=self._config.get("api_key"),
            )
            if self._upstream_error_count():
                raise RuntimeError("A-MEM upstream reported an internal error during reset")
        except Exception:
            self._detach_upstream_error_counter()
            raise
        finally:
            if self._config["llm_backend"] == "openai":
                if previous_base_url is None:
                    os.environ.pop("OPENAI_BASE_URL", None)
                else:
                    os.environ["OPENAI_BASE_URL"] = previous_base_url
        self._provenance.clear()

    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        if self._memory is None:
            raise RuntimeError("A-MEM has not been reset")
        for message in messages:
            error_count = self._upstream_error_count()
            memory_id = self._memory.add_note(
                f"[{message.role}] {message.content}", time=timestamp, category=session
            )
            if self._upstream_error_count() != error_count:
                raise RuntimeError(
                    "A-MEM upstream reported an internal error while processing a memory"
                )
            memory_id = str(memory_id)
            provenance = ((message.source_id,), session, timestamp)
            previous = self._provenance.get(memory_id)
            if previous is not None and previous != provenance:
                raise RuntimeError(
                    "A-MEM associated one memory id with multiple sources; "
                    "refusing ambiguous evidence attribution"
                )
            self._provenance[memory_id] = provenance
        return len(messages)

    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        if self._memory is None:
            raise RuntimeError("A-MEM has not been reset")
        error_count = self._upstream_error_count()
        raw_results = self._memory.search_agentic(query, k=top_k)
        if self._upstream_error_count() != error_count:
            raise RuntimeError("A-MEM upstream reported an internal error during search")
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
            tracked = self._provenance.get(memory_id)
            if tracked is None:
                raise RuntimeError(
                    "A-MEM search returned an untracked memory after reset; "
                    "refusing unverifiable evidence attribution"
                )
            source_ids, session, timestamp = tracked
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
        self._detach_upstream_error_counter()
