"""Safe configuration loading and environment-secret resolution."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError

_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SENSITIVE_PARTS = ("api_key", "password", "secret", "token", "authorization", "credential", "private_key")
_UPSTREAM_REVISIONS = {
    "amem": "ceffb860f0712bbae97b184d440df62bc910ca8d",
    "mem0_oss": "4debc58a83377b18be81ae1e5969a300736b2fac",
}
_PROTECTED_TRACKS = {"common-policy-v1", "paper-reproduction", "paper-reproduction-substituted"}
_MODEL_ROLES = {"memory_llm", "embedding", "reader_tokenizer"}


@dataclass(frozen=True)
class WorkerConfig:
    adapter_kind: str
    adapter_config: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    run_policy: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "WorkerConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read worker configuration: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {"adapter", "tokenizer", "run_policy"}:
            raise ConfigurationError("worker config must contain exactly adapter, tokenizer, and run_policy")
        adapter = raw["adapter"]
        if not isinstance(adapter, dict) or set(adapter) != {"kind", "config"}:
            raise ConfigurationError("adapter must contain exactly kind and config")
        kind, adapter_config = adapter["kind"], adapter["config"]
        if not isinstance(kind, str) or not kind or not isinstance(adapter_config, dict):
            raise ConfigurationError("adapter kind must be a string and config must be an object")
        if not isinstance(raw["tokenizer"], dict):
            raise ConfigurationError("tokenizer must be an object")
        reject_inline_secrets(adapter_config)
        _validate_run_policy(kind, raw["run_policy"])
        return cls(
            adapter_kind=kind,
            adapter_config=adapter_config,
            tokenizer=raw["tokenizer"],
            run_policy=raw["run_policy"],
        )


def _validate_run_policy(adapter_kind: str, raw: Any) -> None:
    if not isinstance(raw, dict) or "track" not in raw or not isinstance(raw["track"], str):
        raise ConfigurationError("run_policy must contain a string track")
    track = raw["track"]
    if track in {"development", "contract-test"}:
        if set(raw) != {"track"}:
            raise ConfigurationError(f"{track} run_policy must contain only track")
        return
    if track not in _PROTECTED_TRACKS:
        raise ConfigurationError(f"unsupported run_policy track: {track!r}")
    required = {"track", "upstream", "models"}
    if track.startswith("paper-"):
        required.add("paper_protocol")
    if set(raw) != required:
        raise ConfigurationError(f"{track} run_policy must contain exactly {sorted(required)}")
    if adapter_kind not in _UPSTREAM_REVISIONS:
        raise ConfigurationError(f"adapter {adapter_kind!r} cannot be labeled {track}")
    if track == "paper-reproduction" and adapter_kind == "amem":
        raise ConfigurationError(
            "the agiresearch/A-mem system adapter is not the A-MEM paper repository; use a substituted track"
        )
    _validate_lock(raw["upstream"], "run_policy.upstream", expected=_UPSTREAM_REVISIONS[adapter_kind])
    models = raw["models"]
    if not isinstance(models, dict) or not _MODEL_ROLES <= models.keys():
        raise ConfigurationError(f"run_policy.models must lock at least {sorted(_MODEL_ROLES)}")
    for role, lock in models.items():
        if not isinstance(role, str) or not role:
            raise ConfigurationError("run_policy.models keys must be non-empty strings")
        _validate_lock(lock, f"run_policy.models.{role}")
    if track.startswith("paper-"):
        _validate_lock(raw["paper_protocol"], "run_policy.paper_protocol")


def _validate_lock(raw: Any, path: str, expected: str | None = None) -> None:
    if not isinstance(raw, dict) or set(raw) != {"revision", "lock_status"}:
        raise ConfigurationError(f"{path} must contain exactly revision and lock_status")
    revision = raw["revision"]
    if raw["lock_status"] != "verified":
        raise ConfigurationError(f"{path}.lock_status must be verified")
    if not isinstance(revision, str) or not _GIT_SHA.fullmatch(revision):
        raise ConfigurationError(f"{path}.revision must be an immutable 40-character lowercase SHA")
    if expected is not None and revision != expected:
        raise ConfigurationError(f"{path}.revision does not match this adapter's pinned upstream lock")


def reject_inline_secrets(value: Any, path: str = "adapter.config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"{path} keys must be strings")
            lowered = key.lower()
            sensitive = any(part in lowered for part in _SENSITIVE_PARTS)
            if sensitive and not lowered.endswith("_env"):
                raise ConfigurationError(f"{path}.{key} must be supplied as an *_env reference")
            if lowered.endswith("_env"):
                if not isinstance(child, str) or not _ENV_NAME.fullmatch(child):
                    raise ConfigurationError(f"{path}.{key} must name an uppercase environment variable")
            else:
                reject_inline_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_inline_secrets(child, f"{path}[{index}]")


def resolve_env_references(value: Any, path: str = "adapter.config") -> tuple[Any, tuple[str, ...]]:
    """Replace foo_env=ENV_NAME with foo=os.environ[ENV_NAME]."""
    secrets: list[str] = []

    def resolve(current: Any, current_path: str) -> Any:
        if isinstance(current, dict):
            result: dict[str, Any] = {}
            for key, child in current.items():
                if key.lower().endswith("_env"):
                    output_key = key[:-4]
                    if not output_key or output_key in current or output_key in result:
                        raise ConfigurationError(f"ambiguous environment reference at {current_path}.{key}")
                    env_name = child
                    secret = os.environ.get(env_name)
                    if secret is None or not secret:
                        raise ConfigurationError(f"required environment variable {env_name} is not set")
                    result[output_key] = secret
                    secrets.append(secret)
                else:
                    result[key] = resolve(child, f"{current_path}.{key}")
            return result
        if isinstance(current, list):
            return [resolve(child, f"{current_path}[]") for child in current]
        return current

    reject_inline_secrets(value, path)
    return resolve(value, path), tuple(secrets)


def environment_secrets(environment: Mapping[str, str] | None = None) -> tuple[str, ...]:
    source = os.environ if environment is None else environment
    return tuple(
        value
        for name, value in source.items()
        if value and any(part in name.lower() for part in _SENSITIVE_PARTS)
    )


def redact(message: str, secrets: tuple[str, ...]) -> str:
    result = message
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result
