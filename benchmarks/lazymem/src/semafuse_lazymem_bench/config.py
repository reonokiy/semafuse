"""TOML configuration loader for public-reproduction runs."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .endpoints import ChatCompletionClient, CodexCliClient, OpenAICompatibleClient


def validate_reproduction_config(config: Mapping[str, Any]) -> None:
    """Fail closed when a named paper/public profile drifts silently."""

    required = {
        "reproduction",
        "dataset",
        "retrieval",
        "window",
        "mpm",
        "answer",
        "judge",
        "metrics",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"config is missing tables: {sorted(missing)}")
    reproduction = config["reproduction"]
    status = str(reproduction.get("status", ""))
    expected_merge = {
        "clean-room-public-reconstruction": "code",
        "clean-room-paper-specification": "paper",
    }.get(status)
    if status == "custom-ablation":
        if config["window"].get("merge_rule") not in {"paper", "code"}:
            raise ValueError("custom-ablation window.merge_rule must be paper or code")
        return
    if expected_merge is None:
        raise ValueError(f"unknown reproduction status {status!r}")

    expected: tuple[tuple[str, str, object], ...] = (
        ("retrieval", "candidate_limit", 50),
        ("retrieval", "channel_limit", 100),
        ("retrieval", "fused_limit", 100),
        ("retrieval", "rrf_k", 60),
        ("retrieval", "dense_model", "Qwen/Qwen3-Embedding-8B"),
        ("retrieval", "reranker_model", "BAAI/bge-reranker-v2-m3"),
        ("retrieval", "sparse_backend", "rank-bm25"),
        (
            "retrieval",
            "sparse_tokenization",
            "ascii-alnum-underscore-hyphen-lower",
        ),
        ("retrieval", "bm25_k1", 1.5),
        ("retrieval", "bm25_b", 0.75),
        ("retrieval", "bm25_epsilon", 0.25),
        ("window", "radius", 2),
        ("window", "max_messages", 8),
        ("window", "stride", 7),
        ("window", "merge_rule", expected_merge),
        ("mpm", "concurrency", 64),
        ("mpm", "max_tokens", 3072),
        ("mpm", "temperature", 0.1),
        ("mpm", "top_p", 0.95),
        ("mpm", "enable_thinking", True),
        ("answer", "temperature", 0.0),
        ("answer", "max_tokens", 512),
        ("answer", "enable_thinking", True),
        ("judge", "temperature", 0.0),
        ("metrics", "latency_question_concurrency", 1),
        ("metrics", "latency_mpm_concurrency", 64),
        ("metrics", "latency_warmup_runs", 3),
        ("metrics", "latency_measured_runs", 3),
        ("metrics", "tokenizer_model", "Qwen/Qwen3-32B"),
    )
    for section, key, expected_value in expected:
        actual = config[section].get(key)
        if actual != expected_value:
            raise ValueError(
                f"{status} requires {section}.{key}={expected_value!r}; got {actual!r}"
            )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise TypeError("config root must be a TOML table")
    return value


def endpoint_client(config: Mapping[str, Any]) -> OpenAICompatibleClient:
    def required(name: str) -> str:
        direct = config.get(name)
        if isinstance(direct, str) and direct:
            return direct
        env_name = config.get(f"{name}_env")
        if isinstance(env_name, str) and os.environ.get(env_name):
            return os.environ[env_name]
        raise ValueError(f"endpoint requires {name!r} or {name}_env")

    return OpenAICompatibleClient(
        base_url=required("base_url"),
        model=required("model"),
        api_key_env=(
            str(config["api_key_env"]) if config.get("api_key_env") else None
        ),
        timeout_s=float(config.get("timeout_s", 180)),
        retries=int(config.get("retries", 2)),
    )


def completion_client(config: Mapping[str, Any]) -> ChatCompletionClient:
    """Build the configured completion provider without changing HTTP defaults.

    ``backend`` is optional for backward compatibility: absent means the
    original OpenAI-compatible HTTP client.  ``codex_cli`` keeps the benchmark
    local to an authenticated Codex CLI and has no API-key configuration.
    """

    backend = str(config.get("backend", "openai_compatible")).lower().replace("-", "_")
    if backend in {"openai_compatible", "openai"}:
        return endpoint_client(config)
    if backend != "codex_cli":
        raise ValueError(
            "completion backend must be 'openai_compatible' or 'codex_cli'"
        )

    def optional(name: str) -> str | None:
        direct = config.get(name)
        if isinstance(direct, str) and direct:
            return direct
        env_name = config.get(f"{name}_env")
        if isinstance(env_name, str) and os.environ.get(env_name):
            return os.environ[env_name]
        return None

    if config.get("profile") is not None or config.get("profile_env") is not None:
        raise ValueError(
            "codex_cli does not support profile because it ignores user configuration"
        )
    command = optional("command") or "codex"
    model = optional("model")
    if model is None:
        raise ValueError("codex_cli requires 'model' or 'model_env'")
    return CodexCliClient(
        command=command,
        model=model,
        timeout_s=float(config.get("timeout_s", 300)),
        retries=int(config.get("retries", 0)),
        max_output_bytes=int(config.get("max_output_bytes", 1_048_576)),
    )


def completion_client_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-secret provider coordinates for resumable run identities."""

    backend = str(config.get("backend", "openai_compatible")).lower().replace("-", "_")

    def resolved(name: str) -> str | None:
        direct = config.get(name)
        if isinstance(direct, str) and direct:
            return direct
        environment_name = config.get(f"{name}_env")
        if isinstance(environment_name, str):
            return os.environ.get(environment_name)
        return None

    if backend in {"openai_compatible", "openai"}:
        return {
            "backend": "openai_compatible",
            "base_url": resolved("base_url"),
            "model": resolved("model"),
            "api_key_env": (
                str(config["api_key_env"]) if config.get("api_key_env") else None
            ),
        }
    if backend == "codex_cli":
        if config.get("profile") is not None or config.get("profile_env") is not None:
            raise ValueError(
                "codex_cli does not support profile because it ignores user configuration"
            )
        model = resolved("model")
        if model is None:
            raise ValueError("codex_cli requires 'model' or 'model_env'")
        return {
            "backend": "codex_cli",
            "command": resolved("command") or "codex",
            "model": model,
            "sandbox": "read-only",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "web_search": "disabled",
            "output_schema": "semafuse-lazymem-codex-cli/v1",
            "timeout_s": float(config.get("timeout_s", 300)),
            "retries": int(config.get("retries", 0)),
            "max_output_bytes": int(config.get("max_output_bytes", 1_048_576)),
        }
    raise ValueError("completion backend must be 'openai_compatible' or 'codex_cli'")
