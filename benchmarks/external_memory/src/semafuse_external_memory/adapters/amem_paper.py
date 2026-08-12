"""Adapter for the exact A-MEM paper implementation and retrieval pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapter import ExternalMemoryAdapter, MemoryItem
from ..config import resolve_env_references, validate_secret_bearing_base_url
from ..errors import AdapterUnavailableError, ConfigurationError
from ..protocol import Message
from ..upstreams.amem_paper import (
    UPSTREAM_BLOB_SHA256,
    UPSTREAM_REVISION,
    VENDORED_SOURCE_SHA256,
)

_QUERY_REWRITE_PROMPT = """Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a \"keywords\" field containing the selected text. 

                Example response format:
                {{\"keywords\": \"keyword1, keyword2, keyword3\"}}"""

_QUERY_REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"keywords": {"type": "string"}},
            "required": ["keywords"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _exact_object(value: Any, allowed: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must be an object")
    extra = set(value) - allowed
    if extra:
        raise ConfigurationError(f"{path} has unknown fields: {sorted(extra)}")
    return value


def _required_string(value: Mapping[str, Any], field: str, path: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise ConfigurationError(f"{path}.{field} must be a non-empty string")
    return result


def _nonempty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _validate_paper_semantics(prompt: str, payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("A-MEM paper LLM response must be an object")
    if "Generate a structured analysis of the following content" in prompt:
        if not _nonempty_strings(payload.get("keywords")):
            raise ValueError("A-MEM paper metadata has no keywords")
        context = payload.get("context")
        if not isinstance(context, str) or not context.strip() or context == "General":
            raise ValueError("A-MEM paper metadata has no specific context")
        if not _nonempty_strings(payload.get("tags")):
            raise ValueError("A-MEM paper metadata has no tags")
        return
    if "You are an AI memory evolution agent" in prompt:
        should_evolve = payload.get("should_evolve")
        if not should_evolve:
            # The paper implementation ignores every action payload when the
            # decision is false. Placeholder arrays therefore have no effect
            # and must not trigger paid retries.
            return
        actions = payload.get("actions")
        if not isinstance(actions, list) or any(
            action not in {"strengthen", "update_neighbor"} for action in actions
        ):
            raise ValueError("A-MEM paper evolution returned an unknown action")
        if not actions:
            raise ValueError("A-MEM paper evolution requested no action")
        candidate_indices = {
            int(value) for value in re.findall(r"memory index:(\d+)", prompt)
        }
        if "strengthen" in actions:
            connections = payload.get("suggested_connections")
            if (
                not isinstance(connections, list)
                or not connections
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index not in candidate_indices
                    for index in connections
                )
            ):
                raise ValueError("A-MEM paper evolution returned an invalid connection")
            if not _nonempty_strings(payload.get("tags_to_update")):
                raise ValueError("A-MEM paper evolution returned no strengthened tags")
        if "update_neighbor" in actions:
            count_match = re.search(r"The number of neighbors is (\d+)\.", prompt)
            if count_match is None:
                raise ValueError("A-MEM paper evolution prompt has no neighbor count")
            neighbor_count = int(count_match.group(1))
            contexts = payload.get("new_context_neighborhood")
            tags = payload.get("new_tags_neighborhood")
            if (
                not isinstance(contexts, list)
                or not isinstance(tags, list)
                or len(contexts) != neighbor_count
                or len(tags) != neighbor_count
            ):
                raise ValueError("A-MEM paper evolution returned incomplete neighbor updates")
        return
    if "generate several keywords, using 'cosmos' as the separator" in prompt:
        keywords = payload.get("keywords")
        if not isinstance(keywords, str) or not keywords.strip():
            raise ValueError("A-MEM paper query rewrite returned no keywords")


class AMemPaperAdapter(ExternalMemoryAdapter):
    """Paper A-MEM with rich notes, evolution, links, and query rewriting."""

    def __init__(self, config: Mapping[str, Any]):
        root = _exact_object(
            config,
            {"upstream_revision", "embedding", "llm", "memory", "retrieval"},
            "A-MEM paper config",
        )
        revision = _required_string(root, "upstream_revision", "A-MEM paper config")
        if revision != UPSTREAM_REVISION:
            raise ConfigurationError(
                "A-MEM paper upstream_revision does not match the bundled audited source"
            )

        embedding = _exact_object(
            root.get("embedding"), {"model", "revision", "cache_dir"}, "embedding"
        )
        embedding_model = _required_string(embedding, "model", "embedding")
        if embedding_model.count("/") != 1:
            raise ConfigurationError(
                "embedding.model must be an explicit Hugging Face owner/repository id"
            )
        embedding_revision = _required_string(embedding, "revision", "embedding")
        if re.fullmatch(r"[0-9a-f]{40}", embedding_revision) is None:
            raise ConfigurationError(
                "embedding.revision must be an immutable 40-character lowercase SHA"
            )
        cache_dir = embedding.get("cache_dir")
        if cache_dir is not None and (not isinstance(cache_dir, str) or not cache_dir.strip()):
            raise ConfigurationError("embedding.cache_dir must be a non-empty string when provided")

        llm = _exact_object(
            root.get("llm"),
            {
                "backend",
                "model",
                "api_key_env",
                "base_url",
                "max_attempts",
                "request_timeout_seconds",
                "require_parameters",
            },
            "llm",
        )
        if _required_string(llm, "backend", "llm") != "openai":
            raise ConfigurationError("A-MEM paper currently supports llm.backend='openai'")
        _required_string(llm, "model", "llm")
        _required_string(llm, "api_key_env", "llm")
        base_url = _required_string(llm, "base_url", "llm")
        validate_secret_bearing_base_url(base_url, "llm.base_url")
        max_attempts = llm.get("max_attempts", 1)
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 10
        ):
            raise ConfigurationError("llm.max_attempts must be an integer in [1, 10]")
        request_timeout_seconds = llm.get("request_timeout_seconds")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, int)
            or not 1 <= request_timeout_seconds <= 300
        ):
            raise ConfigurationError(
                "llm.request_timeout_seconds must be an integer in [1, 300]"
            )
        if not isinstance(llm.get("require_parameters"), bool):
            raise ConfigurationError("llm.require_parameters must be a boolean")

        memory = _exact_object(root.get("memory"), {"evo_threshold"}, "memory")
        evo_threshold = memory.get("evo_threshold", 100)
        if (
            isinstance(evo_threshold, bool)
            or not isinstance(evo_threshold, int)
            or evo_threshold < 1
        ):
            raise ConfigurationError("memory.evo_threshold must be a positive integer")

        retrieval = _exact_object(
            root.get("retrieval"), {"base_k", "query_rewrite", "link_expansion"}, "retrieval"
        )
        base_k = retrieval.get("base_k")
        if isinstance(base_k, bool) or not isinstance(base_k, int) or not 1 <= base_k <= 1000:
            raise ConfigurationError("retrieval.base_k must be an integer in [1, 1000]")
        query_rewrite = _required_string(retrieval, "query_rewrite", "retrieval")
        if query_rewrite not in {"paper_keywords", "none"}:
            raise ConfigurationError("retrieval.query_rewrite must be paper_keywords or none")
        link_expansion = _required_string(retrieval, "link_expansion", "retrieval")
        if link_expansion not in {"paper_interleaved", "none"}:
            raise ConfigurationError("retrieval.link_expansion must be paper_interleaved or none")

        resolved, self.secrets = resolve_env_references(dict(config), "adapter.config")
        self._config = resolved
        self._llm_max_attempts = max_attempts
        self._llm_retry_count = 0
        self._llm_failure_count = 0
        self._memory: Any = None
        self._retriever_type: Any = None
        self._model_path: str | None = None
        self._provenance: dict[str, tuple[tuple[str, ...], str, str | None]] = {}

    def _load_upstream(self) -> tuple[Any, Any]:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "upstreams"
            / "amem_paper"
            / "memory_layer.py"
        )
        try:
            source = source_path.read_bytes()
        except OSError as exc:
            raise AdapterUnavailableError("cannot verify the bundled A-MEM paper source") from exc
        if hashlib.sha256(source).hexdigest() != VENDORED_SOURCE_SHA256:
            raise RuntimeError("bundled A-MEM paper source hash does not match its audited lock")
        if (
            not source.endswith(b"\n")
            or hashlib.sha256(source[:-1]).hexdigest() != UPSTREAM_BLOB_SHA256
        ):
            raise RuntimeError("bundled A-MEM paper source differs from upstream beyond normalization")

        module_name = "semafuse_external_memory.upstreams.amem_paper.memory_layer"
        module = types.ModuleType(module_name)
        module.__file__ = str(source_path)
        module.__package__ = module_name.rpartition(".")[0]
        previous_module = sys.modules.get(module_name)
        previous = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        try:
            sys.modules[module_name] = module
            # Execute the already verified bytes, not a second filesystem read.
            exec(compile(source, str(source_path), "exec"), module.__dict__)
        except ImportError as exc:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            raise AdapterUnavailableError(
                "A-MEM paper dependencies are unavailable; install "
                "methods/amem_paper/requirements.txt"
            ) from exc
        except BaseException:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            raise
        finally:
            if previous is None:
                os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
            else:
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous
        # The exact paper file calls re.sub in MemoryNote.analyze_content but
        # omits `import re`. Inject only that standard-library module after the
        # byte check; otherwise every valid rich-note response silently falls
        # back to empty metadata.
        module.re = re
        return module.AgenticMemorySystem, module.SimpleEmbeddingRetriever

    def _install_llm_retry_wrapper(self) -> None:
        try:
            from jsonschema import validate
        except ImportError as exc:
            raise AdapterUnavailableError(
                "jsonschema is required by the A-MEM paper adapter"
            ) from exc
        controller = getattr(getattr(self._memory, "llm_controller", None), "llm", None)
        completion = getattr(controller, "get_completion", None)
        if not callable(completion):
            raise RuntimeError("A-MEM paper LLM controller cannot be wrapped for validation")

        def completion_with_retries(*args: Any, **kwargs: Any) -> str:
            response_format = kwargs.get("response_format")
            if response_format is None and len(args) >= 2:
                response_format = args[1]
            schema = None
            if isinstance(response_format, dict):
                json_schema = response_format.get("json_schema")
                if isinstance(json_schema, dict):
                    schema = json_schema.get("schema")
            for attempt in range(1, self._llm_max_attempts + 1):
                try:
                    response = completion(*args, **kwargs)
                    if not isinstance(response, str):
                        raise TypeError("LLM completion was not text")
                    payload = json.loads(response)
                    if isinstance(schema, dict):
                        validate(instance=payload, schema=schema)
                    prompt_value = args[0] if args else kwargs.get("prompt")
                    prompt = prompt_value if isinstance(prompt_value, str) else ""
                    _validate_paper_semantics(prompt, payload)
                    return response
                except Exception:
                    if attempt == self._llm_max_attempts:
                        self._llm_failure_count += 1
                        raise
                    self._llm_retry_count += 1
                    time.sleep(float(attempt))
            raise AssertionError("unreachable A-MEM paper retry state")

        controller.get_completion = completion_with_retries

    def _install_provider_routing(self) -> None:
        if not self._config["llm"]["require_parameters"]:
            return
        controller = getattr(getattr(self._memory, "llm_controller", None), "llm", None)
        chat = getattr(getattr(controller, "client", None), "chat", None)
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        if not callable(create):
            raise RuntimeError("A-MEM paper OpenAI client cannot enforce provider routing")

        def create_with_required_parameters(*args: Any, **kwargs: Any) -> Any:
            extra_body = kwargs.get("extra_body")
            if extra_body is None:
                extra_body = {}
            if not isinstance(extra_body, dict):
                raise RuntimeError("OpenAI extra_body must be an object")
            provider = extra_body.get("provider")
            if provider is None:
                provider = {}
            if not isinstance(provider, dict):
                raise RuntimeError("OpenRouter provider preferences must be an object")
            if provider.get("require_parameters") is False:
                raise RuntimeError("OpenRouter provider preferences disable required parameters")
            kwargs["extra_body"] = {
                **extra_body,
                "provider": {**provider, "require_parameters": True},
            }
            return create(*args, **kwargs)

        completions.create = create_with_required_parameters

    def _configure_openai_client(self) -> None:
        controller = getattr(getattr(self._memory, "llm_controller", None), "llm", None)
        client = getattr(controller, "client", None)
        with_options = getattr(client, "with_options", None)
        if not callable(with_options):
            raise RuntimeError("A-MEM paper OpenAI client cannot set retry and timeout bounds")
        controller.client = with_options(
            timeout=float(self._config["llm"]["request_timeout_seconds"]),
            max_retries=0,
        )

    def _pin_consolidation_retriever(self) -> None:
        memory = self._memory
        retriever_type = self._retriever_type
        model_path = self._model_path
        if memory is None or retriever_type is None or model_path is None:
            raise RuntimeError("A-MEM paper consolidation patch was installed before reset")

        def consolidate_memories() -> None:
            retriever = retriever_type(model_path)
            for note in memory.memories.values():
                metadata = f"{note.context} {' '.join(note.keywords)} {' '.join(note.tags)}"
                retriever.add_documents([note.content + " , " + metadata])
            memory.retriever = retriever

        memory.consolidate_memories = consolidate_memories

    def _guard_empty_memory_evolution(self) -> None:
        memory = self._memory
        process_memory = getattr(memory, "process_memory", None)
        if not callable(process_memory):
            raise RuntimeError("A-MEM paper memory evolution entry point is unavailable")

        def process_memory_with_empty_guard(note: Any) -> tuple[bool, Any]:
            # There is no possible neighbor action before the first note. The
            # upstream nevertheless asks the LLM to choose strengthen/update,
            # which can create an invalid index. Preserve the only valid paper
            # outcome deterministically and use the exact upstream path once a
            # neighbor exists.
            if not memory.memories:
                return False, note
            return process_memory(note)

        memory.process_memory = process_memory_with_empty_guard

    def reset(self) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AdapterUnavailableError("huggingface_hub is required by A-MEM paper") from exc
        system_type, self._retriever_type = self._load_upstream()
        embedding = self._config["embedding"]
        self._model_path = snapshot_download(
            repo_id=embedding["model"],
            revision=embedding["revision"],
            cache_dir=embedding.get("cache_dir"),
        )
        llm = self._config["llm"]
        previous_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_BASE_URL"] = llm["base_url"]
        try:
            self._memory = system_type(
                model_name=self._model_path,
                llm_backend="openai",
                llm_model=llm["model"],
                evo_threshold=self._config["memory"].get("evo_threshold", 100),
                api_key=llm["api_key"],
            )
        finally:
            if previous_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = previous_base_url
        self._llm_retry_count = 0
        self._llm_failure_count = 0
        self._configure_openai_client()
        self._install_provider_routing()
        self._install_llm_retry_wrapper()
        self._guard_empty_memory_evolution()
        self._pin_consolidation_retriever()
        self._provenance.clear()

    def _validate_links(self) -> None:
        notes = list(self._memory.memories.values())
        for note_index, note in enumerate(notes):
            if not isinstance(note.links, list):
                raise RuntimeError("A-MEM paper produced a non-list link collection")
            for link in note.links:
                if isinstance(link, bool) or not isinstance(link, (int,)):
                    raise RuntimeError("A-MEM paper produced a non-integer memory link")
                if not 0 <= int(link) < len(notes):
                    raise RuntimeError("A-MEM paper produced an out-of-range memory link")
                if int(link) == note_index:
                    raise RuntimeError("A-MEM paper produced a self-link")

    def add(self, messages: Sequence[Message], session: str, timestamp: str | None) -> int:
        if self._memory is None:
            raise RuntimeError("A-MEM paper has not been reset")
        for message in messages:
            failures = self._llm_failure_count
            memory_id = str(self._memory.add_note(message.content, time=timestamp))
            if self._llm_failure_count != failures:
                raise RuntimeError("A-MEM paper LLM failed while processing a memory")
            if not memory_id or memory_id not in self._memory.memories:
                raise RuntimeError("A-MEM paper returned an unknown memory id")
            note = self._memory.memories[memory_id]
            if (
                not isinstance(note.keywords, list)
                or not note.keywords
                or not isinstance(note.tags, list)
                or not note.tags
                or not isinstance(note.context, str)
                or not note.context.strip()
                or note.context == "General"
            ):
                raise RuntimeError("A-MEM paper silently degraded rich-note generation")
            provenance = ((message.source_id,), session, timestamp)
            previous = self._provenance.get(memory_id)
            if previous is not None and previous != provenance:
                raise RuntimeError(
                    "A-MEM paper associated one memory id with multiple sources"
                )
            self._provenance[memory_id] = provenance
            self._validate_links()
        return len(messages)

    def _rewrite_query(self, query: str) -> str:
        if self._config["retrieval"]["query_rewrite"] == "none":
            return query
        controller = self._memory.llm_controller.llm
        failures = self._llm_failure_count
        response = controller.get_completion(
            _QUERY_REWRITE_PROMPT.format(question=query), response_format=_QUERY_REWRITE_SCHEMA
        )
        if self._llm_failure_count != failures:
            raise RuntimeError("A-MEM paper LLM failed while rewriting a query")
        keywords = json.loads(response)["keywords"].strip()
        if not keywords:
            raise RuntimeError("A-MEM paper query rewrite returned no keywords")
        return keywords

    @staticmethod
    def _paper_text(note: Any) -> str:
        return (
            f"talk start time:{note.timestamp}"
            f"memory content: {note.content}"
            f"memory context: {note.context}"
            f"memory keywords: {note.keywords}"
            f"memory tags: {note.tags}"
        )

    def search(self, query: str, top_k: int) -> Sequence[MemoryItem]:
        if self._memory is None:
            raise RuntimeError("A-MEM paper has not been reset")
        rewritten = self._rewrite_query(query)
        base_k = self._config["retrieval"]["base_k"]
        raw_indices = self._memory.retriever.search(rewritten, base_k)
        notes = list(self._memory.memories.values())
        normalized: list[MemoryItem] = []
        for raw_index in raw_indices:
            index = int(raw_index)
            if not 0 <= index < len(notes):
                raise RuntimeError("A-MEM paper retriever returned an out-of-range index")
            related_indices = [index]
            if self._config["retrieval"]["link_expansion"] == "paper_interleaved":
                for neighbor_rank, raw_neighbor in enumerate(notes[index].links):
                    neighbor = int(raw_neighbor)
                    if not 0 <= neighbor < len(notes):
                        raise RuntimeError("A-MEM paper link expansion found an invalid index")
                    related_indices.append(neighbor)
                    if neighbor_rank >= base_k:
                        break

            source_ids: list[str] = []
            texts: list[str] = []
            for related_index in related_indices:
                related = notes[related_index]
                tracked = self._provenance.get(str(related.id))
                if tracked is None:
                    raise RuntimeError("A-MEM paper returned an untracked memory")
                source_ids.extend(tracked[0])
                texts.append(self._paper_text(related))
            seed = notes[index]
            seed_tracked = self._provenance[str(seed.id)]
            normalized.append(
                MemoryItem(
                    id=str(seed.id),
                    text="\n".join(texts),
                    score=None,
                    source_ids=tuple(dict.fromkeys(source_ids)),
                    session=seed_tracked[1],
                    timestamp=seed_tracked[2],
                )
            )
        return normalized

    def close(self) -> None:
        self._memory = None
        self._retriever_type = None
        self._model_path = None
        self._provenance.clear()
