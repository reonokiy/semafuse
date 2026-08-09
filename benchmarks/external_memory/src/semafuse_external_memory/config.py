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
    "amem_paper": "0c8039f28fdcc08189a23c07a3437d9d2482f9c2",
    "mem0_oss": "4debc58a83377b18be81ae1e5969a300736b2fac",
}
_PROTECTED_TRACKS = {"common-policy-v1", "paper-reproduction", "paper-reproduction-substituted"}
_MODEL_ROLES = {"memory_llm", "embedding", "reader_tokenizer"}

# The method requirement files currently pin the upstream repositories but not
# every transitive wheel (including platform-specific CPU/GPU variants).  A
# protected result must not imply a fully locked environment until a reviewed
# transitive lock is registered here.  Development runs remain available.
_TRANSITIVE_LOCK_SHA256: dict[str, str | None] = {
    "amem": None,
    "amem_paper": None,
    "mem0_oss": None,
}


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
        _validate_run_policy(kind, adapter_config, raw["tokenizer"], raw["run_policy"])
        return cls(
            adapter_kind=kind,
            adapter_config=adapter_config,
            tokenizer=raw["tokenizer"],
            run_policy=raw["run_policy"],
        )


def _validate_run_policy(
    adapter_kind: str,
    adapter_config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    raw: Any,
) -> None:
    if not isinstance(raw, dict) or "track" not in raw or not isinstance(raw["track"], str):
        raise ConfigurationError("run_policy must contain a string track")
    track = raw["track"]
    if track == "contract-test":
        if set(raw) != {"track"}:
            raise ConfigurationError("contract-test run_policy must contain only track")
        return
    if track == "development":
        if set(raw) == {"track"}:
            return
        if set(raw) != {"track", "upstream"}:
            raise ConfigurationError(
                "development run_policy may additionally contain only an audited upstream lock"
            )
        if adapter_kind not in _UPSTREAM_REVISIONS:
            raise ConfigurationError(
                f"adapter {adapter_kind!r} has no registered development upstream lock"
            )
        _validate_lock(
            raw["upstream"],
            "run_policy.upstream",
            expected=_UPSTREAM_REVISIONS[adapter_kind],
        )
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
        _validate_model_lock(lock, f"run_policy.models.{role}")
    _validate_model_identities(adapter_kind, adapter_config, tokenizer, models)
    if track.startswith("paper-"):
        _validate_lock(raw["paper_protocol"], "run_policy.paper_protocol")
    if _TRANSITIVE_LOCK_SHA256[adapter_kind] is None:
        raise ConfigurationError(
            f"{adapter_kind} has no reviewed complete transitive dependency lock; "
            "use run_policy.track='development'"
        )


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


def _validate_model_lock(raw: Any, path: str) -> None:
    if not isinstance(raw, dict) or set(raw) != {"provider", "model", "revision", "lock_status"}:
        raise ConfigurationError(
            f"{path} must contain exactly provider, model, revision, and lock_status"
        )
    for field in ("provider", "model"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{path}.{field} must be a non-empty string")
    revision = raw["revision"]
    if not isinstance(revision, str) or not _GIT_SHA.fullmatch(revision):
        raise ConfigurationError(f"{path}.revision must be an immutable 40-character lowercase SHA")
    if raw["lock_status"] != "verified":
        raise ConfigurationError(f"{path}.lock_status must be verified")


def _validate_model_identities(
    adapter_kind: str,
    adapter_config: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    models: Mapping[str, Any],
) -> None:
    """Match protected manifest identities to the actual adapter configuration."""

    expected: dict[str, tuple[str, str]] = {}
    if adapter_kind == "amem":
        backend = adapter_config.get("llm_backend")
        llm_model = adapter_config.get("llm_model")
        embedding_model = adapter_config.get("model_name")
        embedding_revision = adapter_config.get("model_revision")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (backend, llm_model, embedding_model, embedding_revision)
        ):
            raise ConfigurationError(
                "protected A-MEM tracks require explicit llm_backend, llm_model, "
                "model_name, and model_revision"
            )
        expected["memory_llm"] = (backend, llm_model)
        expected["embedding"] = ("sentence-transformers", embedding_model)
    elif adapter_kind == "amem_paper":
        llm = adapter_config.get("llm")
        embedding = adapter_config.get("embedding")
        if not isinstance(llm, dict) or not isinstance(embedding, dict):
            raise ConfigurationError(
                "protected A-MEM paper tracks require explicit llm and embedding objects"
            )
        backend = llm.get("backend")
        llm_model = llm.get("model")
        embedding_model = embedding.get("model")
        embedding_revision = embedding.get("revision")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (backend, llm_model, embedding_model, embedding_revision)
        ):
            raise ConfigurationError(
                "protected A-MEM paper tracks require explicit llm.backend, llm.model, "
                "embedding.model, and embedding.revision"
            )
        expected["memory_llm"] = (backend, llm_model)
        expected["embedding"] = ("sentence-transformers", embedding_model)
    elif adapter_kind == "mem0_oss":
        memory_config = adapter_config.get("memory_config")
        if not isinstance(memory_config, dict):
            raise ConfigurationError("protected Mem0 tracks require memory_config")
        for role, section_name in (("memory_llm", "llm"), ("embedding", "embedder")):
            section = memory_config.get(section_name)
            if not isinstance(section, dict) or not isinstance(section.get("config"), dict):
                raise ConfigurationError(
                    f"protected Mem0 tracks require memory_config.{section_name}.provider and .config.model"
                )
            provider = section.get("provider")
            model = section["config"].get("model")
            if (
                not isinstance(provider, str)
                or not provider.strip()
                or not isinstance(model, str)
                or not model.strip()
            ):
                raise ConfigurationError(
                    f"protected Mem0 tracks require memory_config.{section_name}.provider and .config.model"
                )
            expected[role] = (provider, model)

    tokenizer_kind = tokenizer.get("kind")
    tokenizer_model = tokenizer.get("model")
    if (
        tokenizer_kind != "huggingface"
        or not isinstance(tokenizer_model, str)
        or not tokenizer_model.strip()
    ):
        raise ConfigurationError("protected tracks require a pinned huggingface reader tokenizer")
    expected["reader_tokenizer"] = ("huggingface", tokenizer_model)

    for role, (provider, model) in expected.items():
        declared = models[role]
        if declared["provider"] != provider or declared["model"] != model:
            raise ConfigurationError(
                f"run_policy.models.{role} identity does not match the configured provider/model"
            )
    tokenizer_revision = tokenizer.get("revision")
    if models["reader_tokenizer"]["revision"] != tokenizer_revision:
        raise ConfigurationError(
            "run_policy.models.reader_tokenizer revision does not match the configured tokenizer"
        )
    if adapter_kind == "amem" and models["embedding"]["revision"] != adapter_config["model_revision"]:
        raise ConfigurationError(
            "run_policy.models.embedding revision does not match the configured A-MEM model_revision"
        )
    if adapter_kind == "amem_paper" and (
        models["embedding"]["revision"] != adapter_config["embedding"]["revision"]
    ):
        raise ConfigurationError(
            "run_policy.models.embedding revision does not match the configured A-MEM paper revision"
        )


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
