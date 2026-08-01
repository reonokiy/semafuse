"""TOML configuration loader for public-reproduction runs."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .endpoints import OpenAICompatibleClient


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
