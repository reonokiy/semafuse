"""Minimal OpenAI-compatible HTTP clients with no third-party dependency."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ChatCompletionClient(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> CompletionResult: ...


class OpenAICompatibleClient:
    """Synchronous client for ``/v1/chat/completions`` and ``/v1/embeddings``.

    Instances are safe to share between the MPM worker threads: every request
    creates its own urllib request and keeps no mutable request state.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 180.0,
        retries: int = 2,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not model:
            raise ValueError("model must be non-empty")
        if timeout_s <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries non-negative")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else (
            os.environ.get(api_key_env) if api_key_env else None
        )
        self.timeout_s = timeout_s
        self.retries = retries
        self.headers = dict(headers or {})

    def _post(self, path: str, payload: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers=headers, method="POST"
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read()
                latency = time.perf_counter() - started
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise RuntimeError("endpoint response root is not an object")
                return value, latency
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"endpoint request failed after {self.retries + 1} attempts") from last_error

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(extra_body or {})
        value, latency = self._post("/chat/completions", payload)
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("endpoint response has no assistant content") from error
        if not isinstance(content, str):
            raise RuntimeError("endpoint assistant content is not a string")
        usage = value.get("usage", {})
        return CompletionResult(
            text=content,
            latency_s=latency,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            metadata={"model": value.get("model", self.model), "id": value.get("id")},
        )

    def embeddings(self, inputs: Sequence[str]) -> tuple[list[list[float]], float]:
        value, latency = self._post(
            "/embeddings", {"model": self.model, "input": list(inputs)}
        )
        data = value.get("data")
        if not isinstance(data, list):
            raise RuntimeError("embedding response has no data array")
        ordered = sorted(data, key=lambda item: int(item["index"]))
        vectors = [[float(component) for component in item["embedding"]] for item in ordered]
        if len(vectors) != len(inputs):
            raise RuntimeError("embedding response count does not match input count")
        return vectors, latency


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
