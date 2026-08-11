#!/usr/bin/env python3
"""Validate a complete LoCoMo answer run and emit a text-free receipt.

The full-answer runner intentionally keeps memory contexts and predictions in
gitignored journals.  This tool reads those artifacts without modifying them,
checks their cryptographic bindings and the two method-specific populations,
independently rebuilds the Rust scalar reports, and writes only compact hashes,
aggregates, and telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
OFFICIAL_DATASET_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
OFFICIAL_UPSTREAM_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
OFFICIAL_SCORER_SHA256 = (
    "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
)
OFFICIAL_QA_N = 314
OFFICIAL_RETRIEVAL_N = 312
OFFICIAL_ALL_QUESTION_SET_SHA256 = (
    "e17078090da0740cdcb279fa72e0cf2f43665463389fef490cd06975d91c74aa"
)
OFFICIAL_ELIGIBLE_QUESTION_SET_SHA256 = (
    "98dbfcf95d707909ac040aebcaec7fd0b9e5c3c12ca1ca2c2ce4e532d3dcf3dc"
)
OFFICIAL_EVALUATOR_METADATA_SHA256 = (
    "fc28c312f2e6730240d4c14b46c6d79223935a41384e0f6ff9dfe45bb1fd46c0"
)
OFFICIAL_CATEGORY_COUNTS_ALL = {
    "multi-hop": 69,
    "open-domain": 20,
    "single-hop": 160,
    "temporal": 65,
}
OFFICIAL_CATEGORY_COUNTS_ELIGIBLE = {
    "multi-hop": 69,
    "open-domain": 18,
    "single-hop": 160,
    "temporal": 65,
}
METRIC = "locomo_official_f1"
TRACK = "paper_v2-selection/full-answer-development"
SELECTION_TRACK = "paper_v2"
MEMORY_RECORD_KIND = "locomo_answer_memory"
ANSWER_RECORD_KIND = "locomo_answer_prediction"
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_CONFIDENCE = 0.95

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_QUESTION_ID = re.compile(r"^q_[0-9a-f]{64}$")
_OPAQUE_MESSAGE_ID = re.compile(r"^m_[0-9a-f]{64}$")
_SAFE_METHOD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./:+()@=-]{0,191}$")
_SAFE_MODEL_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127})?$"
)
_SAFE_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,63}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9._~+-])~?/[^\s]+")
_UNC_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])\\\\[^\\\s]+\\[^\s]+")
_CREDENTIAL_KEY = re.compile(
    r"(?:api[_-]?key|authorization|bearer|credential|password|passwd|private[_-]?key|"
    r"refresh[_-]?token|secret|access[_-]?token)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|"
    r"\bhf_[A-Za-z0-9]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}|\bxox[baprs]-[A-Za-z0-9-]{16,}|"
    r"\bAIza[0-9A-Za-z_-]{20,}|\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}|"
    r"\bAKIA[0-9A-Z]{16}\b|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_URI_WITH_CREDENTIALS = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@"
)
_RAW_TEXT_KEYS = {
    "answer",
    "content",
    "context",
    "history",
    "memory",
    "messages",
    "prediction",
    "prompt",
    "query",
    "question",
    "reference",
    "response",
    "text",
}
_CATEGORY_NUMBERS = {
    "multi-hop": 1,
    "temporal": 2,
    "open-domain": 3,
    "single-hop": 4,
}


class VerificationError(ValueError):
    """An artifact cannot support a trustworthy compact receipt."""


class SplitMix64:
    """The exact PRNG used by ``src/benchmark_report.rs``."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("stop must be positive")
        threshold = ((-stop) & self._MASK) % stop
        while True:
            value = self.next_u64()
            if value >= threshold:
                return value % stop


def _reject_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, source: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{source} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid JSON in {source}: {error}") from error


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerificationError(f"value is not canonical JSON: {error}") from error


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerificationError(f"{label} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise VerificationError(
            f"{label} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise VerificationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_method(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_METHOD.fullmatch(value):
        raise VerificationError(f"{label} must be a safe method identifier")
    return value


def _require_safe_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _SAFE_VALUE.fullmatch(value)
        or _looks_like_absolute_path(value)
        or _CREDENTIAL_VALUE.search(value)
    ):
        raise VerificationError(f"{label} must be a compact secret-free identifier")
    return value


def _require_model_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_MODEL_ID.fullmatch(value):
        raise VerificationError(f"{label} must be a strict model identifier")
    return value


def _require_provider_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_PROVIDER_ID.fullmatch(value):
        raise VerificationError(f"{label} must be a strict provider identifier")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VerificationError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise VerificationError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise VerificationError(f"{label} must be <= {maximum}")
    return 0.0 if result == 0.0 else result


def _optional_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite_number(value, label, minimum=minimum, maximum=maximum)


def _close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _read_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {label}: {error}") from error
    return raw, {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _check_expected_hash(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    expected = _require_sha256(expected, f"expected {label} SHA-256")
    if actual != expected:
        raise VerificationError(f"{label} SHA-256 does not match the caller pin")


def _question_set_sha256(question_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(b"semafuse-locomo-answer-question-set/v1")
    for question_id in sorted(question_ids):
        raw = question_id.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _selected_question_ids_sha256(question_ids: Iterable[str]) -> str:
    """Reproduce the selection digest emitted by the Rust answer runner."""
    ordered = sorted(question_ids)
    digest = hashlib.sha256(b"semafuse-locomo-selected-question-ids/v1")
    digest.update(len(ordered).to_bytes(8, "big"))
    for question_id in ordered:
        raw = question_id.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _binding_sha256(domain: bytes, rows: Iterable[tuple[str, ...]]) -> str:
    digest = hashlib.sha256(domain)
    for row in sorted(rows):
        for value in row:
            raw = value.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _validate_opaque_ids(value: Any, label: str) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(_require_list(value, label)):
        if not isinstance(item, str) or not _OPAQUE_MESSAGE_ID.fullmatch(item):
            raise VerificationError(f"{label}[{index}] is not an opaque message ID")
        if item in seen:
            raise VerificationError(f"{label} contains duplicate message ID {item}")
        seen.add(item)
        output.append(item)
    return tuple(output)


def _validate_usage(value: Any, label: str) -> dict[str, Any]:
    usage = _require_object(value, label)
    _require_exact_keys(
        usage,
        {"prompt_tokens", "completion_tokens", "cost", "calls"},
        set(),
        label,
    )
    return {
        "prompt_tokens": _require_int(usage["prompt_tokens"], f"{label}.prompt_tokens"),
        "completion_tokens": _require_int(
            usage["completion_tokens"], f"{label}.completion_tokens"
        ),
        "cost": _optional_number(usage["cost"], f"{label}.cost", minimum=0.0),
        "calls": _require_int(usage["calls"], f"{label}.calls"),
    }


def _require_zero_usage(usage: Mapping[str, Any], label: str) -> None:
    if usage != {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
        "calls": 0,
    }:
        raise VerificationError(f"{label} must have zero processor usage")


def _usage_add(aggregate: dict[str, Any], item: Mapping[str, Any]) -> None:
    was_empty = (
        aggregate["calls"] == 0
        and aggregate["prompt_tokens"] == 0
        and aggregate["completion_tokens"] == 0
        and aggregate["cost"] is None
    )
    left_cost = aggregate["cost"]
    right_cost = item["cost"]
    aggregate["prompt_tokens"] += item["prompt_tokens"]
    aggregate["completion_tokens"] += item["completion_tokens"]
    aggregate["calls"] += item["calls"]
    if left_cost is not None and right_cost is not None:
        aggregate["cost"] = left_cost + right_cost
    elif left_cost is not None and right_cost is None and item["calls"] == 0:
        aggregate["cost"] = left_cost
    elif left_cost is None and right_cost is not None and was_empty:
        aggregate["cost"] = right_cost
    elif left_cost is None and right_cost is None:
        aggregate["cost"] = None
    else:
        aggregate["cost"] = None


def _aggregate_usage(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
        "calls": 0,
    }
    for row in rows:
        _usage_add(aggregate, row)
    return aggregate


def _usage_equal(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key in ("prompt_tokens", "completion_tokens", "calls"):
        if actual.get(key) != expected.get(key):
            raise VerificationError(f"{label}.{key} disagrees with journal records")
    left = actual.get("cost")
    right = expected.get("cost")
    if (left is None) != (right is None) or (
        left is not None and not _close(_finite_number(left, f"{label}.cost"), right)
    ):
        raise VerificationError(f"{label}.cost disagrees with journal records")


def _new_response_telemetry() -> dict[str, Any]:
    return {
        "providers": Counter(),
        "models": Counter(),
        "finish_reasons": Counter(),
        "missing_provider": 0,
        "missing_model": 0,
        "missing_finish_reason": 0,
        "reasoning_tokens": 0,
        "missing_reasoning_tokens": 0,
        "reasoning_present_true": 0,
        "reasoning_present_false": 0,
        "responses": 0,
        "validation_retry_responses": 0,
    }


def _add_response_telemetry(
    aggregate: dict[str, Any],
    *,
    provider: Any,
    model: Any,
    finish_reason: Any,
    reasoning_tokens: Any,
    reasoning_present: Any,
    retry: bool,
    label: str,
) -> None:
    if provider is None:
        aggregate["missing_provider"] += 1
    else:
        aggregate["providers"][_require_provider_id(provider, f"{label}.provider")] += 1
    if model is None:
        aggregate["missing_model"] += 1
    else:
        aggregate["models"][_require_model_id(model, f"{label}.model")] += 1
    if finish_reason is None:
        aggregate["missing_finish_reason"] += 1
    else:
        aggregate["finish_reasons"][
            _require_safe_value(finish_reason, f"{label}.finish_reason")
        ] += 1
    if reasoning_tokens is None:
        aggregate["missing_reasoning_tokens"] += 1
    else:
        aggregate["reasoning_tokens"] += _require_int(
            reasoning_tokens, f"{label}.reasoning_tokens"
        )
    if not isinstance(reasoning_present, bool):
        raise VerificationError(f"{label}.reasoning_present must be boolean")
    aggregate[
        "reasoning_present_true" if reasoning_present else "reasoning_present_false"
    ] += 1
    aggregate["responses"] += 1
    aggregate["validation_retry_responses"] += int(retry)


def _finalize_response_telemetry(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "logical_responses": value["responses"],
        "provider_response_counts": dict(sorted(value["providers"].items())),
        "responses_missing_provider": value["missing_provider"],
        "model_response_counts": dict(sorted(value["models"].items())),
        "responses_missing_model": value["missing_model"],
        "finish_reason_response_counts": dict(sorted(value["finish_reasons"].items())),
        "responses_missing_finish_reason": value["missing_finish_reason"],
        "reasoning": {
            "reported_tokens": value["reasoning_tokens"],
            "responses_missing_token_count": value["missing_reasoning_tokens"],
            "responses_with_substantive_reasoning": value["reasoning_present_true"],
            "responses_without_substantive_reasoning": value["reasoning_present_false"],
        },
        "validation_retry_responses": value["validation_retry_responses"],
    }


def _validate_processor_trace(
    metadata: Mapping[str, Any],
    payload_usage: Mapping[str, Any],
    telemetry: dict[str, Any],
    label: str,
) -> None:
    strategy_trace = _require_object(metadata.get("strategy_trace"), f"{label}.strategy_trace")
    strategy = _require_object(strategy_trace.get("strategy"), f"{label}.strategy_trace.strategy")
    generation = _require_object(
        strategy.get("generation"), f"{label}.strategy_trace.strategy.generation"
    )
    requested_sampling = _require_object(
        generation.get("requested_sampling"),
        f"{label}.strategy_trace.strategy.generation.requested_sampling",
    )
    if requested_sampling.get("enable_thinking") is not False:
        raise VerificationError(f"{label} processor trace is not bound to no-thinking sampling")
    processing = _require_object(
        strategy.get("processing"), f"{label}.strategy_trace.strategy.processing"
    )
    trace_usage = _require_object(processing.get("usage"), f"{label}.processing.usage")
    expected_trace_usage_keys = {
        "call_semantics",
        "calls",
        "calls_missing_completion_tokens",
        "calls_missing_cost",
        "calls_missing_prompt_tokens",
        "calls_missing_reasoning_tokens",
        "completion_tokens",
        "cost",
        "prompt_tokens",
        "reasoning_tokens",
    }
    _require_exact_keys(
        trace_usage, expected_trace_usage_keys, set(), f"{label}.processing.usage"
    )
    trace_calls = _require_int(trace_usage["calls"], f"{label}.processing.usage.calls")
    if (
        trace_usage["call_semantics"]
        != "successful_validation_level_responses; transport_retries_without_usage_are_not_counted"
    ):
        raise VerificationError(f"{label} has unknown processor call semantics")
    trace_prompt = _require_int(
        trace_usage["prompt_tokens"], f"{label}.processing.usage.prompt_tokens"
    )
    trace_completion = _require_int(
        trace_usage["completion_tokens"], f"{label}.processing.usage.completion_tokens"
    )
    trace_reasoning = _require_int(
        trace_usage["reasoning_tokens"], f"{label}.processing.usage.reasoning_tokens"
    )
    trace_cost = _optional_number(
        trace_usage["cost"], f"{label}.processing.usage.cost", minimum=0.0
    )
    missing_prompt = _require_int(
        trace_usage["calls_missing_prompt_tokens"],
        f"{label}.processing.usage.calls_missing_prompt_tokens",
    )
    missing_completion = _require_int(
        trace_usage["calls_missing_completion_tokens"],
        f"{label}.processing.usage.calls_missing_completion_tokens",
    )
    missing_reasoning = _require_int(
        trace_usage["calls_missing_reasoning_tokens"],
        f"{label}.processing.usage.calls_missing_reasoning_tokens",
    )
    missing_cost = _require_int(
        trace_usage["calls_missing_cost"], f"{label}.processing.usage.calls_missing_cost"
    )
    if not (
        trace_calls == 0
        or trace_cost is not None
        or (missing_prompt == 0 and missing_completion == 0)
    ):
        raise VerificationError(
            f"{label} processor usage has neither cost nor complete token accounting"
        )
    if trace_cost is not None and missing_cost != 0:
        raise VerificationError(
            f"{label} processor aggregate cost is present despite a missing per-call cost"
        )
    if trace_calls != payload_usage["calls"]:
        raise VerificationError(f"{label} processor trace/payload call counts differ")
    if (
        trace_prompt != payload_usage["prompt_tokens"]
        or trace_completion != payload_usage["completion_tokens"]
    ):
        raise VerificationError(f"{label} processor trace/payload token counts differ")
    if (trace_cost is None) != (payload_usage["cost"] is None) or (
        trace_cost is not None and not _close(trace_cost, payload_usage["cost"])
    ):
        raise VerificationError(f"{label} processor trace/payload costs differ")

    groups = _require_list(processing.get("processor_calls"), f"{label}.processor_calls")
    detailed_calls = 0
    detail_prompt = 0
    detail_completion = 0
    detail_reasoning = 0
    detail_costs: list[float] = []
    detail_missing_prompt = 0
    detail_missing_completion = 0
    detail_missing_reasoning = 0
    detail_missing_cost = 0
    validation_retries = 0
    providers: set[str] = set()
    models: set[str] = set()
    for group_index, raw_group in enumerate(groups):
        group_label = f"{label}.processor_calls[{group_index}]"
        group = _require_object(raw_group, group_label)
        _require_exact_keys(group, {"attempts", "calls", "window_order"}, set(), group_label)
        _require_int(group["window_order"], f"{group_label}.window_order")
        calls = _require_list(group["calls"], f"{group_label}.calls")
        attempts = _require_int(group["attempts"], f"{group_label}.attempts", minimum=1)
        if attempts != len(calls):
            raise VerificationError(f"{group_label}.attempts differs from detailed calls")
        validation_retries += attempts - 1
        final_finish_reason: str | None = None
        for call_index, raw_call in enumerate(calls):
            call_label = f"{group_label}.calls[{call_index}]"
            call = _require_object(raw_call, call_label)
            _require_exact_keys(
                call,
                {"attempt", "model", "provider", "finish_reason", "usage"},
                set(),
                call_label,
            )
            attempt = _require_int(call["attempt"], f"{call_label}.attempt", minimum=1)
            if attempt != call_index + 1:
                raise VerificationError(f"{call_label}.attempt is not sequential")
            if call["model"] is None:
                raise VerificationError(f"{call_label}.model must be present")
            _require_model_id(call["model"], f"{call_label}.model")
            finish_reason = call["finish_reason"]
            if finish_reason not in {"stop", "length"}:
                raise VerificationError(
                    f"{call_label}.finish_reason must be stop or retryable length"
                )
            if finish_reason == "length" and not any(
                isinstance(later, dict) and later.get("finish_reason") == "stop"
                for later in calls[call_index + 1 :]
            ):
                raise VerificationError(f"{call_label} length result has no later stop retry")
            final_finish_reason = finish_reason
            call_usage = _require_object(call["usage"], f"{call_label}.usage")
            _require_exact_keys(
                call_usage,
                {
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "reasoning_present",
                    "cost",
                },
                set(),
                f"{call_label}.usage",
            )
            prompt = call_usage["prompt_tokens"]
            completion = call_usage["completion_tokens"]
            reasoning = call_usage["reasoning_tokens"]
            cost = call_usage["cost"]
            if prompt is None:
                detail_missing_prompt += 1
            else:
                detail_prompt += _require_int(prompt, f"{call_label}.usage.prompt_tokens")
            if completion is None:
                detail_missing_completion += 1
            else:
                detail_completion += _require_int(
                    completion, f"{call_label}.usage.completion_tokens"
                )
            if reasoning is None:
                detail_missing_reasoning += 1
            else:
                reasoning = _require_int(reasoning, f"{call_label}.usage.reasoning_tokens")
                detail_reasoning += reasoning
            if cost is None:
                detail_missing_cost += 1
            else:
                detail_costs.append(
                    _finite_number(cost, f"{call_label}.usage.cost", minimum=0.0)
                )
            if call_usage["reasoning_present"] is not False or reasoning is None or reasoning > 1:
                raise VerificationError(
                    f"{call_label} violates the no-thinking processor telemetry contract"
                )
            _add_response_telemetry(
                telemetry,
                provider=call["provider"],
                model=call["model"],
                finish_reason=call["finish_reason"],
                reasoning_tokens=reasoning,
                reasoning_present=call_usage["reasoning_present"],
                retry=attempt > 1,
                label=call_label,
            )
            if call["provider"] is not None:
                providers.add(_require_provider_id(call["provider"], f"{call_label}.provider"))
            if call["model"] is not None:
                models.add(_require_model_id(call["model"], f"{call_label}.model"))
            detailed_calls += 1
        if final_finish_reason != "stop":
            raise VerificationError(
                f"{group_label} final processor attempt did not finish with stop"
            )

    if detailed_calls != trace_calls:
        raise VerificationError(f"{label} processor detailed/aggregate call counts differ")
    expected_values = (
        (detail_prompt, trace_prompt, "prompt tokens"),
        (detail_completion, trace_completion, "completion tokens"),
        (detail_reasoning, trace_reasoning, "reasoning tokens"),
        (detail_missing_prompt, missing_prompt, "missing prompt-token counts"),
        (detail_missing_completion, missing_completion, "missing completion-token counts"),
        (detail_missing_reasoning, missing_reasoning, "missing reasoning-token counts"),
        (detail_missing_cost, missing_cost, "missing cost counts"),
    )
    for detailed, aggregate, name in expected_values:
        if detailed != aggregate:
            raise VerificationError(f"{label} processor detailed/aggregate {name} differ")
    if trace_cost is not None and not _close(math.fsum(detail_costs), trace_cost):
        raise VerificationError(f"{label} processor detailed/aggregate costs differ")
    if trace_calls > 0 and trace_cost is None and missing_cost == 0:
        raise VerificationError(f"{label} processor trace omits cost without a missing-cost call")

    if _require_int(processing.get("attempts"), f"{label}.processing.attempts") != detailed_calls:
        raise VerificationError(f"{label} processing attempt count differs")
    if _require_int(
        processing.get("validation_retries"), f"{label}.processing.validation_retries"
    ) != validation_retries:
        raise VerificationError(f"{label} processing validation retry count differs")
    raw_trace_providers = _require_list(
        processing.get("providers"), f"{label}.processing.providers"
    )
    trace_providers = {
        _require_provider_id(value, f"{label}.processing.providers")
        for value in raw_trace_providers
    }
    raw_trace_models = _require_list(
        processing.get("models"), f"{label}.processing.models"
    )
    trace_models = {
        _require_model_id(value, f"{label}.processing.models")
        for value in raw_trace_models
    }
    if (
        len(trace_providers) != len(raw_trace_providers)
        or len(trace_models) != len(raw_trace_models)
        or trace_providers != providers
        or trace_models != models
    ):
        raise VerificationError(f"{label} processing provider/model sets differ from calls")
    expected_provider = next(iter(providers)) if len(providers) == 1 else None
    expected_model = next(iter(models)) if len(models) == 1 else None
    if processing.get("provider") != expected_provider or processing.get("model") != expected_model:
        raise VerificationError(f"{label} processing singular provider/model summary differs")


def _validate_reconstructive_processor_trace(
    metadata: Mapping[str, Any],
    payload_usage: Mapping[str, Any],
    telemetry: dict[str, Any],
    label: str,
) -> None:
    strategy_trace = _require_object(metadata.get("strategy_trace"), f"{label}.strategy_trace")
    strategy = _require_object(strategy_trace.get("strategy"), f"{label}.strategy_trace.strategy")
    processing = _require_object(
        strategy.get("processing"), f"{label}.strategy_trace.strategy.processing"
    )
    trace_usage = _validate_usage(
        {
            "prompt_tokens": _require_object(
                processing.get("usage"), f"{label}.processing.usage"
            ).get("prompt_tokens"),
            "completion_tokens": processing["usage"].get("completion_tokens"),
            "cost": processing["usage"].get("cost"),
            "calls": processing["usage"].get("calls"),
        },
        f"{label}.processing.usage",
    )
    _usage_equal(trace_usage, payload_usage, f"{label}.processor_usage")

    usage_block = _require_object(processing["usage"], f"{label}.processing.usage")
    _require_exact_keys(
        usage_block,
        {
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost",
            "calls",
            "calls_missing_prompt_tokens",
            "calls_missing_completion_tokens",
            "calls_missing_reasoning_tokens",
            "calls_missing_cost",
        },
        set(),
        f"{label}.processing.usage",
    )
    expected_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "costs": [],
        "calls": 0,
        "calls_missing_prompt_tokens": 0,
        "calls_missing_completion_tokens": 0,
        "calls_missing_reasoning_tokens": 0,
        "calls_missing_cost": 0,
    }
    groups = _require_list(processing.get("processor_calls"), f"{label}.processor_calls")
    for group_index, raw_group in enumerate(groups):
        group_label = f"{label}.processor_calls[{group_index}]"
        group = _require_object(raw_group, group_label)
        _require_exact_keys(group, {"round", "calls"}, set(), group_label)
        _require_int(group["round"], f"{group_label}.round")
        calls = _require_list(group["calls"], f"{group_label}.calls")
        if not calls:
            raise VerificationError(f"{group_label} must not serialize an empty call group")
        for call_index, raw_call in enumerate(calls):
            call_label = f"{group_label}.calls[{call_index}]"
            call = _require_object(raw_call, call_label)
            _require_exact_keys(
                call,
                {
                    "attempt",
                    "model",
                    "provider",
                    "finish_reason",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "reasoning_present",
                    "cost",
                },
                set(),
                call_label,
            )
            if _require_int(call["attempt"], f"{call_label}.attempt", minimum=1) != call_index + 1:
                raise VerificationError(f"{call_label}.attempt is not sequential")
            model = _require_model_id(call["model"], f"{call_label}.model")
            provider = (
                None
                if call["provider"] is None
                else _require_provider_id(call["provider"], f"{call_label}.provider")
            )
            if call["finish_reason"] not in {"stop", "length"}:
                raise VerificationError(f"{call_label}.finish_reason is unsupported")
            if call_index == len(calls) - 1 and call["finish_reason"] != "stop":
                raise VerificationError(f"{group_label} final controller call did not stop")
            expected_totals["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                value = call[key]
                if value is None:
                    expected_totals[f"calls_missing_{key}"] += 1
                else:
                    expected_totals[key] += _require_int(value, f"{call_label}.{key}")
            if call["cost"] is None:
                expected_totals["calls_missing_cost"] += 1
            else:
                expected_totals["costs"].append(
                    _finite_number(call["cost"], f"{call_label}.cost", minimum=0.0)
                )
            reasoning_tokens = call["reasoning_tokens"]
            if call["reasoning_present"] is not False or reasoning_tokens not in {0, None}:
                raise VerificationError(
                    f"{call_label} violates the no-thinking controller contract"
                )
            _add_response_telemetry(
                telemetry,
                provider=provider,
                model=model,
                finish_reason=call["finish_reason"],
                reasoning_tokens=reasoning_tokens,
                reasoning_present=call["reasoning_present"],
                retry=call_index > 0,
                label=call_label,
            )

    for key in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "calls",
        "calls_missing_prompt_tokens",
        "calls_missing_completion_tokens",
        "calls_missing_reasoning_tokens",
        "calls_missing_cost",
    ):
        if (
            _require_int(usage_block[key], f"{label}.processing.usage.{key}")
            != expected_totals[key]
        ):
            raise VerificationError(f"{label} reconstructive processor {key} disagrees with calls")
    expected_cost = (
        math.fsum(expected_totals["costs"])
        if expected_totals["calls"] > 0
        and expected_totals["calls_missing_cost"] == 0
        else None
    )
    actual_cost = _optional_number(
        usage_block["cost"], f"{label}.processing.usage.cost", minimum=0.0
    )
    if (expected_cost is None) != (actual_cost is None) or (
        expected_cost is not None and not _close(expected_cost, actual_cost)
    ):
        raise VerificationError(f"{label} reconstructive processor cost disagrees with calls")


def _validate_envelope(
    value: Any,
    *,
    record_kind: str,
    run_key: str,
    label: str,
) -> tuple[dict[str, Any], str, str]:
    envelope = _require_object(value, label)
    _require_exact_keys(
        envelope,
        {"run_key", "method", "question_id", "record_kind", "payload"},
        set(),
        label,
    )
    payload = _require_object(envelope["payload"], f"{label}.payload")
    method = _require_method(payload.get("method"), f"{label}.payload.method")
    question_id = payload.get("question_id")
    if not isinstance(question_id, str) or not _OPAQUE_QUESTION_ID.fullmatch(question_id):
        raise VerificationError(f"{label} contains a non-opaque question ID")
    if (
        envelope["run_key"] != run_key
        or payload.get("run_key") != run_key
        or envelope["method"] != method
        or envelope["question_id"] != question_id
        or envelope["record_kind"] != record_kind
    ):
        raise VerificationError(f"{label} envelope/payload identity mismatch")
    return payload, method, question_id


def _validate_category(payload: Mapping[str, Any], label: str) -> tuple[str, int]:
    category = payload.get("category")
    category_number = _require_int(payload.get("category_number"), f"{label}.category_number")
    if category not in _CATEGORY_NUMBERS or category_number != _CATEGORY_NUMBERS[category]:
        raise VerificationError(f"{label} category name/number mismatch")
    return category, category_number


def _validate_memory_record(
    value: Any,
    *,
    run_key: str,
    method_identities: Mapping[str, Mapping[str, Any]],
    telemetry: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    payload, method, question_id = _validate_envelope(
        value, record_kind=MEMORY_RECORD_KIND, run_key=run_key, label=label
    )
    method_identity = method_identities.get(method)
    if method_identity is None:
        raise VerificationError(f"{label} contains unknown memory output {method}")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "run_key",
            "method",
            "question_id",
            "category",
            "category_number",
            "retrieval_eligible",
            "ranked_message_ids",
            "covered_message_ids",
            "retained_message_ids",
            "context",
            "context_sha256",
            "context_bytes",
            "context_tokens",
            "retrieval_recall_all_at_10",
            "retrieval_ndcg_at_10",
            "memory_wall_s",
            "retrieval_s",
            "processing_s",
            "shared_construction_s",
            "processor_usage",
            "metadata",
        },
        set(),
        f"{label}.payload",
    )
    if _require_int(payload["schema_version"], f"{label}.schema_version") != 1:
        raise VerificationError(f"{label} has unsupported memory schema_version")
    category, category_number = _validate_category(payload, label)
    eligible = payload["retrieval_eligible"]
    if not isinstance(eligible, bool):
        raise VerificationError(f"{label}.retrieval_eligible must be boolean")
    for key in ("ranked_message_ids", "covered_message_ids", "retained_message_ids"):
        _validate_opaque_ids(payload[key], f"{label}.{key}")
    context = payload["context"]
    if not isinstance(context, str) or "\0" in context:
        raise VerificationError(f"{label}.context must be valid text without NUL")
    context_raw = context.encode("utf-8")
    context_sha = _require_sha256(payload["context_sha256"], f"{label}.context_sha256")
    if hashlib.sha256(context_raw).hexdigest() != context_sha:
        raise VerificationError(f"{label} context SHA-256 mismatch")
    context_bytes = _require_int(payload["context_bytes"], f"{label}.context_bytes")
    if context_bytes != len(context_raw):
        raise VerificationError(f"{label} context byte count mismatch")
    context_tokens = payload["context_tokens"]
    if context_tokens is not None:
        context_tokens = _require_int(context_tokens, f"{label}.context_tokens")
    retrieval_scores = []
    for key in ("retrieval_recall_all_at_10", "retrieval_ndcg_at_10"):
        retrieval_scores.append(
            _optional_number(payload[key], f"{label}.{key}", minimum=0.0, maximum=1.0)
        )
    if eligible and any(value is None for value in retrieval_scores):
        raise VerificationError(f"{label} eligible row lacks retrieval diagnostics")
    if not eligible and any(value is not None for value in retrieval_scores):
        raise VerificationError(f"{label} ineligible row has retrieval diagnostics")
    for key in ("memory_wall_s", "retrieval_s", "processing_s", "shared_construction_s"):
        _optional_number(payload[key], f"{label}.{key}", minimum=0.0)
    if payload["memory_wall_s"] is None:
        raise VerificationError(f"{label}.memory_wall_s must be present")
    usage = _validate_usage(payload["processor_usage"], f"{label}.processor_usage")
    metadata = _require_object(payload["metadata"], f"{label}.metadata")
    _validate_memory_method_identity(
        method=method,
        method_identity=method_identity,
        metadata=metadata,
        usage=usage,
        telemetry=telemetry,
        label=label,
    )
    return {
        "pair": (method, question_id),
        "method": method,
        "question_id": question_id,
        "category": category,
        "category_number": category_number,
        "retrieval_eligible": eligible,
        "context_sha256": context_sha,
        "context_bytes": context_bytes,
        "context_tokens": context_tokens,
        "processor_usage": usage,
    }


def _validate_memory_method_identity(
    *,
    method: str,
    method_identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
    usage: Mapping[str, Any],
    telemetry: dict[str, Any],
    label: str,
) -> None:
    kind = method_identity["kind"]
    if kind == "bm25":
        _require_exact_keys(
            metadata,
            {"kind", "context_order", "shared_execution"},
            set(),
            f"{label}.metadata",
        )
        if (
            metadata["kind"] != "bm25"
            or metadata["context_order"] != "source_chronological"
            or metadata["shared_execution"] is not False
        ):
            raise VerificationError(f"{label} BM25 metadata identity mismatch")
        _require_zero_usage(usage, f"{label}.processor_usage")
        return

    if kind == "semafuse_full_memory_and_retrieval":
        expected_common = {
            "config_sha256": method_identity["config_sha256"],
            "resolved_config_sha256": method_identity["resolved_config_sha256"],
            "source_method": method_identity.get("parent_method", method),
            "shared_execution": True,
        }
        for key, expected in expected_common.items():
            if metadata.get(key) != expected:
                raise VerificationError(f"{label} Semafuse metadata {key} mismatch")
        _require_sha256(metadata.get("history_identity_sha256"), f"{label}.history_identity_sha256")
        _require_int(
            metadata.get("construction_amortization_n"),
            f"{label}.construction_amortization_n",
            minimum=1,
        )
        if method_identity.get("output_role") == "retrieval_stage":
            if (
                metadata.get("kind") != "shared_semafuse_stage_context"
                or metadata.get("stage") != "retrieved"
                or metadata.get("usage_shared_with") != method_identity["parent_method"]
            ):
                raise VerificationError(f"{label} Semafuse stage metadata identity mismatch")
            _require_zero_usage(usage, f"{label}.processor_usage")
            return
        if (
            method_identity.get("output_role") != "main"
            or metadata.get("kind") != "semafuse_strategy_full"
        ):
            raise VerificationError(f"{label} Semafuse main metadata identity mismatch")
        strategy = _require_safe_value(metadata.get("strategy"), f"{label}.strategy")
        strategy_trace = _require_object(
            metadata.get("strategy_trace"), f"{label}.strategy_trace"
        )
        benchmark_trace = _require_object(
            strategy_trace.get("benchmark_trace"),
            f"{label}.strategy_trace.benchmark_trace",
        )
        if (
            benchmark_trace.get("strategy") != strategy
            or benchmark_trace.get("execution_mode") != "full"
        ):
            raise VerificationError(f"{label} benchmark strategy identity mismatch")
        if strategy == "lazymem":
            _validate_processor_trace(metadata, usage, telemetry, label)
        elif strategy == "reconstructive":
            _validate_reconstructive_processor_trace(metadata, usage, telemetry, label)
        else:
            raise VerificationError(f"{label} has unsupported Semafuse strategy {strategy}")
        return

    if kind == "imported_source_ranking_rag":
        expected_strings = {
            "kind": "imported_source_ranking_rag",
            "cohort": "retrieval_eligible_312",
            "source_method": method_identity["source_method"],
            "source_run_key": method_identity["source_run_key"],
            "source_journal_sha256": method_identity["source_journal_sha256"],
            "source_summary_sha256": method_identity["source_summary_sha256"],
            "context_order": "source_chronological",
        }
        _require_exact_keys(
            metadata,
            set(expected_strings)
            | {"source_ranking_limit", "native_memory_reused", "shared_execution"},
            set(),
            f"{label}.metadata",
        )
        if (
            any(metadata[key] != value for key, value in expected_strings.items())
            or _require_int(
                metadata["source_ranking_limit"], f"{label}.source_ranking_limit", minimum=1
            )
            != method_identity["limit"]
            or metadata["native_memory_reused"] is not False
            or metadata["shared_execution"] is not False
        ):
            raise VerificationError(f"{label} imported method metadata identity mismatch")
        _require_zero_usage(usage, f"{label}.processor_usage")
        return
    raise VerificationError(f"{label} has unsupported manifest method kind {kind}")


def _validate_f1(value: Any, *, category: str, category_number: int, label: str) -> float:
    f1 = _require_object(value, label)
    _require_exact_keys(
        f1,
        {"schema_version", "metric", "score", "category", "category_name", "provenance"},
        set(),
        label,
    )
    if (
        _require_int(f1["schema_version"], f"{label}.schema_version") != 1
        or f1["metric"] != METRIC
        or _require_int(f1["category"], f"{label}.category") != category_number
        or f1["category_name"] != category
    ):
        raise VerificationError(f"{label} evaluator identity/category mismatch")
    provenance = _require_object(f1["provenance"], f"{label}.provenance")
    _require_exact_keys(
        provenance,
        {
            "upstream",
            "upstream_commit",
            "upstream_scorer_sha256",
            "data_license",
            "stemmer",
            "paper_comparable_lj",
        },
        set(),
        f"{label}.provenance",
    )
    if (
        provenance["upstream"] != "snap-research/locomo"
        or provenance["upstream_commit"] != OFFICIAL_UPSTREAM_COMMIT
        or provenance["upstream_scorer_sha256"] != OFFICIAL_SCORER_SHA256
        or provenance["data_license"] != "CC-BY-NC-4.0"
        or provenance["stemmer"] != "nltk-porter/0.1.0 Mode::Nltk"
        or provenance["paper_comparable_lj"] is not False
    ):
        raise VerificationError(f"{label} evaluator provenance mismatch")
    return _finite_number(f1["score"], f"{label}.score", minimum=0.0, maximum=1.0)


def _validate_answer_record(
    value: Any,
    *,
    run_key: str,
    memory_run_key: str,
    expected_prompt_version: str,
    expected_answer_protocol: Mapping[str, Any],
    expected_model: str,
    expected_provider: str,
    telemetry: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    payload, method, question_id = _validate_envelope(
        value, record_kind=ANSWER_RECORD_KIND, run_key=run_key, label=label
    )
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "run_key",
            "memory_run_key",
            "method",
            "question_id",
            "category",
            "category_number",
            "context_sha256",
            "prediction",
            "prediction_sha256",
            "official_f1",
            "answer_wall_s",
            "answer_usage",
            "metadata",
        },
        set(),
        f"{label}.payload",
    )
    if (
        _require_int(payload["schema_version"], f"{label}.schema_version") != 1
        or payload["memory_run_key"] != memory_run_key
    ):
        raise VerificationError(f"{label} answer/memory run identity mismatch")
    category, category_number = _validate_category(payload, label)
    context_sha = _require_sha256(payload["context_sha256"], f"{label}.context_sha256")
    prediction = payload["prediction"]
    if not isinstance(prediction, str) or not prediction.strip() or "\0" in prediction:
        raise VerificationError(f"{label}.prediction must be non-empty valid text")
    prediction_sha = _require_sha256(
        payload["prediction_sha256"], f"{label}.prediction_sha256"
    )
    if hashlib.sha256(prediction.encode("utf-8")).hexdigest() != prediction_sha:
        raise VerificationError(f"{label} prediction SHA-256 mismatch")
    score = _validate_f1(
        payload["official_f1"],
        category=category,
        category_number=category_number,
        label=f"{label}.official_f1",
    )
    _finite_number(payload["answer_wall_s"], f"{label}.answer_wall_s", minimum=0.0)
    usage = _validate_usage(payload["answer_usage"], f"{label}.answer_usage")
    if usage["calls"] != 1:
        raise VerificationError(f"{label} must represent exactly one common-answer response")
    metadata = _require_object(payload["metadata"], f"{label}.metadata")
    _require_exact_keys(
        metadata,
        {
            "prompt_version",
            "answer_protocol",
            "response_model",
            "response_provider",
            "response_finish_reason",
            "response_reasoning_tokens",
            "response_reasoning_present",
        },
        set(),
        f"{label}.metadata",
    )
    if metadata["prompt_version"] != expected_prompt_version:
        raise VerificationError(f"{label} answer prompt version differs from the manifest")
    answer_protocol = _require_object(
        metadata["answer_protocol"], f"{label}.metadata.answer_protocol"
    )
    if answer_protocol != expected_answer_protocol:
        raise VerificationError(f"{label} answer protocol identity differs from the manifest")
    if (
        _require_model_id(metadata["response_model"], f"{label}.metadata.response_model")
        != expected_model
    ):
        raise VerificationError(f"{label} response model differs from the manifest")
    if (
        _require_provider_id(
            metadata["response_provider"], f"{label}.metadata.response_provider"
        )
        != expected_provider
    ):
        raise VerificationError(f"{label} response provider is not {expected_provider}")
    if metadata["response_finish_reason"] != "stop":
        raise VerificationError(f"{label} common answer did not finish with stop")
    answer_reasoning_tokens = _require_int(
        metadata["response_reasoning_tokens"],
        f"{label}.metadata.response_reasoning_tokens",
    )
    if metadata["response_reasoning_present"] is not False or answer_reasoning_tokens != 0:
        raise VerificationError(
            f"{label} violates the no-thinking common-answer telemetry contract"
        )
    _add_response_telemetry(
        telemetry,
        provider=metadata["response_provider"],
        model=metadata["response_model"],
        finish_reason=metadata["response_finish_reason"],
        reasoning_tokens=answer_reasoning_tokens,
        reasoning_present=metadata["response_reasoning_present"],
        retry=False,
        label=f"{label}.metadata",
    )
    return {
        "pair": (method, question_id),
        "method": method,
        "question_id": question_id,
        "category": category,
        "category_number": category_number,
        "context_sha256": context_sha,
        "prediction_sha256": prediction_sha,
        "score": score,
        "answer_usage": usage,
    }


def _read_journal(
    path: Path,
    label: str,
    validator: Callable[[Any, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = hashlib.sha256()
    byte_count = 0
    records: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise VerificationError(f"{label} line {line_number} lacks a final newline")
                if raw_line == b"\n":
                    raise VerificationError(f"{label} contains an empty line")
                value = _parse_json(raw_line, f"{label} line {line_number}")
                records.append(validator(value, f"{label} line {line_number}"))
    except OSError as error:
        raise VerificationError(f"cannot read {label}: {error}") from error
    if not records:
        raise VerificationError(f"{label} is empty; the full run is incomplete")
    return records, {
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "records": len(records),
    }


def _stable_mean(values: Sequence[float]) -> float:
    if not values:
        raise VerificationError("cannot summarize an empty score vector")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    return scale * (sum(value / scale for value in values) / len(values))


def _percentile_linear(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise VerificationError("cannot take a percentile of an empty score vector")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_ci(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise VerificationError("cannot bootstrap an empty score vector")
    scale = max(abs(value) for value in values)
    random = SplitMix64(BOOTSTRAP_SEED)
    means: list[float] = []
    n = len(values)
    for _ in range(BOOTSTRAP_RESAMPLES):
        normalized_sum = 0.0
        if scale != 0.0:
            for _ in range(n):
                normalized_sum += values[random.randbelow(n)] / scale
            means.append(scale * normalized_sum / n)
        else:
            for _ in range(n):
                random.randbelow(n)
            means.append(0.0)
    means.sort()
    return {
        "lower": _percentile_linear(means, 0.025),
        "upper": _percentile_linear(means, 0.975),
    }


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": _stable_mean(values),
        "ci95": _bootstrap_ci(values),
        "denominator": len(values),
    }


def _build_scalar_report(
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    methods: Iterable[str],
    question_ids: Iterable[str],
    baseline: str,
) -> dict[str, Any]:
    ordered_methods = sorted(methods)
    ordered_questions = sorted(question_ids)
    if baseline not in ordered_methods:
        raise VerificationError("scalar report baseline is absent")
    if len(ordered_methods) < 2 or not ordered_questions:
        raise VerificationError("scalar report needs two methods and a non-empty population")

    method_reports: dict[str, Any] = {}
    for method in ordered_methods:
        rows = records[method]
        if set(rows) != set(ordered_questions):
            raise VerificationError(f"method {method} has an incomplete scalar population")
        values = [rows[question_id]["score"] for question_id in ordered_questions]
        by_type: dict[str, list[float]] = {}
        for question_id in ordered_questions:
            row = rows[question_id]
            by_type.setdefault(row["category"], []).append(row["score"])
        method_reports[method] = {
            "overall": _summarize(values),
            "by_question_type": {
                category: _summarize(category_values)
                for category, category_values in sorted(by_type.items())
            },
        }

    comparisons: dict[str, Any] = {}
    baseline_rows = records[baseline]
    for method in ordered_methods:
        if method == baseline:
            continue
        deltas = [
            records[method][question_id]["score"] - baseline_rows[question_id]["score"]
            for question_id in ordered_questions
        ]
        wins = sum(delta > 0.0 for delta in deltas)
        ties = sum(delta == 0.0 for delta in deltas)
        comparisons[method] = {
            "method": method,
            "baseline_method": baseline,
            "mean_delta": _stable_mean(deltas),
            "delta_ci95": _bootstrap_ci(deltas),
            "denominator": len(deltas),
            "wins": wins,
            "ties": ties,
            "losses": len(deltas) - wins - ties,
        }

    return {
        "schema_version": 1,
        "metric": METRIC,
        "baseline_method": baseline,
        "direction": "higher_is_better",
        "denominator": len(ordered_questions),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "interval_method": "percentile_linear",
        },
        "methods": method_reports,
        "comparisons": comparisons,
    }


def _assert_equivalent(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            actual_keys = sorted(actual) if isinstance(actual, dict) else type(actual).__name__
            raise VerificationError(
                f"{label} object keys differ; actual={actual_keys}, expected={sorted(expected)}"
            )
        for key in expected:
            _assert_equivalent(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise VerificationError(f"{label} array shape differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equivalent(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise VerificationError(f"{label} differs from the recomputed value")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise VerificationError(f"{label} differs from the recomputed value")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise VerificationError(f"{label} is not numeric")
        if not _close(_finite_number(actual, label), expected):
            raise VerificationError(f"{label} differs from the recomputed value")
        return
    raise VerificationError(f"{label} contains an unsupported comparison value")


_MANIFEST_KEYS = {
    "schema_version",
    "run_key",
    "memory_run_key",
    "suite",
    "track",
    "complete",
    "paper_comparable",
    "paper_comparability_reason",
    "benchmark",
    "license",
    "protocol",
    "execution",
    "denominators",
    "baseline_method",
    "methods",
    "answer_outputs",
    "pricing_guard",
    "implementation",
    "limitations",
    "credential_handling",
}


def _compact_method_identity(parent: Mapping[str, Any], output: str, cohort: str) -> dict[str, Any]:
    kind = _require_safe_value(parent.get("kind"), f"method {output}.kind")
    identity: dict[str, Any] = {"id": output, "cohort": cohort, "kind": kind}
    if kind == "semafuse_full_memory_and_retrieval" and parent.get("main_output") == output:
        identity["output_role"] = "main"
    if parent.get("id") != output:
        identity["parent_method"] = _require_method(parent.get("id"), f"method {output}.parent")
        if parent.get("main_output") == output:
            identity["output_role"] = "main"
        elif output in parent.get("stage_outputs", []):
            identity["output_role"] = "retrieval_stage"
        else:
            raise VerificationError(f"method {output} is not bound to its parent manifest")
    for key in (
        "config_sha256",
        "resolved_config_sha256",
        "source_journal_sha256",
        "source_summary_sha256",
    ):
        if key in parent:
            identity[key] = _require_sha256(parent[key], f"method {output}.{key}")
    if "source_run_key" in parent:
        identity["source_run_key"] = _require_sha256(
            parent["source_run_key"], f"method {output}.source_run_key"
        )
    if "source_method" in parent:
        identity["source_method"] = _require_method(
            parent["source_method"], f"method {output}.source_method"
        )
    for key in ("embedding_model", "reranker_model"):
        if key in parent:
            identity[key] = _require_model_id(parent[key], f"method {output}.{key}")
    for key in ("model_revisions_verified", "native_memory_reused"):
        if key in parent:
            if not isinstance(parent[key], bool):
                raise VerificationError(f"method {output}.{key} must be boolean")
            identity[key] = parent[key]
    if "limit" in parent:
        identity["limit"] = _require_int(parent["limit"], f"method {output}.limit", minimum=1)
    return identity


def _validate_answer_protocol(value: Any) -> dict[str, str]:
    protocol = _require_object(value, "manifest.protocol.answer_protocol")
    _require_exact_keys(
        protocol,
        {
            "protocol",
            "response_mode",
            "system_prompt_sha256",
            "user_template_sha256",
            "persisted_payload",
        },
        set(),
        "manifest.protocol.answer_protocol",
    )
    version = _require_safe_value(
        protocol["protocol"], "manifest.protocol.answer_protocol.protocol"
    )
    supported = {
        "semafuse-locomo-common-answer/v1": ("plain_text", "complete_response"),
        "semafuse-locomo-concise-json-answer/v2": ("json_schema", "answer_field_only"),
    }
    expected = supported.get(version)
    if expected is None or (
        protocol["response_mode"], protocol["persisted_payload"]
    ) != expected:
        raise VerificationError("manifest declares an unsupported answer protocol identity")
    return {
        "protocol": version,
        "response_mode": expected[0],
        "system_prompt_sha256": _require_sha256(
            protocol["system_prompt_sha256"],
            "manifest.protocol.answer_protocol.system_prompt_sha256",
        ),
        "user_template_sha256": _require_sha256(
            protocol["user_template_sha256"],
            "manifest.protocol.answer_protocol.user_template_sha256",
        ),
        "persisted_payload": expected[1],
    }


def _validate_manifest(manifest: Any, envelope_run_key: str) -> dict[str, Any]:
    manifest = _require_object(manifest, "summary.manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, set(), "summary.manifest")
    run_key = _require_sha256(manifest["run_key"], "manifest.run_key")
    memory_run_key = _require_sha256(manifest["memory_run_key"], "manifest.memory_run_key")
    if run_key != envelope_run_key:
        raise VerificationError("summary envelope/manifest run keys differ")
    if (
        _require_int(manifest["schema_version"], "manifest.schema_version") != 1
        or manifest["track"] != TRACK
        or manifest["complete"] is not True
        or manifest["paper_comparable"] is not False
    ):
        raise VerificationError("manifest is not a complete full-answer development run")

    benchmark = _require_object(manifest["benchmark"], "manifest.benchmark")
    _require_exact_keys(
        benchmark,
        {"name", "upstream_commit", "dataset_sha256", "selection"},
        set(),
        "manifest.benchmark",
    )
    selection = _require_object(benchmark["selection"], "manifest.benchmark.selection")
    _require_exact_keys(
        selection,
        {
            "track",
            "conversation_ids",
            "categories",
            "qa_n",
            "retrieval_eval_n",
            "selected_question_ids_sha256",
        },
        set(),
        "manifest.benchmark.selection",
    )
    categories = [
        _require_int(value, f"manifest.benchmark.selection.categories[{index}]")
        for index, value in enumerate(
            _require_list(selection["categories"], "manifest.benchmark.selection.categories")
        )
    ]
    if (
        benchmark["name"] != "LoCoMo"
        or benchmark["upstream_commit"] != OFFICIAL_UPSTREAM_COMMIT
        or benchmark["dataset_sha256"] != OFFICIAL_DATASET_SHA256
        or selection["track"] != SELECTION_TRACK
        or selection["conversation_ids"] != ["conv-49", "conv-50"]
        or categories != [1, 2, 3, 4]
    ):
        raise VerificationError("manifest benchmark identity/population mismatch")
    if (
        _require_int(selection["qa_n"], "manifest.benchmark.selection.qa_n")
        != OFFICIAL_QA_N
        or _require_int(
            selection["retrieval_eval_n"],
            "manifest.benchmark.selection.retrieval_eval_n",
        )
        != OFFICIAL_RETRIEVAL_N
    ):
        raise VerificationError("manifest benchmark population mismatch")
    selected_question_ids_sha256 = _require_sha256(
        selection["selected_question_ids_sha256"],
        "manifest.benchmark.selection.selected_question_ids_sha256",
    )
    denominators = _require_object(manifest["denominators"], "manifest.denominators")
    _require_exact_keys(
        denominators, {"qa_n", "retrieval_eval_n"}, set(), "manifest.denominators"
    )
    if (
        _require_int(denominators["qa_n"], "manifest.denominators.qa_n")
        != OFFICIAL_QA_N
        or _require_int(
            denominators["retrieval_eval_n"], "manifest.denominators.retrieval_eval_n"
        )
        != OFFICIAL_RETRIEVAL_N
    ):
        raise VerificationError("manifest denominators mismatch")

    license_block = _require_object(manifest["license"], "manifest.license")
    if (
        license_block.get("spdx") != "CC-BY-NC-4.0"
        or license_block.get("noncommercial_use_acknowledged") is not True
        or license_block.get("dataset_redistributed") is not False
        or license_block.get("dataset_path_serialized") is not False
        or license_block.get("raw_context_artifacts_publishable") is not False
    ):
        raise VerificationError("manifest license/safety declaration mismatch")

    execution = _require_object(manifest["execution"], "manifest.execution")
    if (
        execution.get("validation_only") is not False
        or _require_int(
            execution.get("scheduled_question_n"), "manifest.execution.scheduled_question_n"
        )
        != OFFICIAL_QA_N
        or execution.get("question_limit") is not None
        or _require_int(
            execution.get("question_id_filter_n"), "manifest.execution.question_id_filter_n"
        )
        != 0
        or execution.get("question_id_filter_sha256") is not None
        or execution.get("source_method_filter") != []
        or execution.get("partial_runs_cannot_set_complete") is not True
    ):
        raise VerificationError("manifest execution block describes a partial run")

    protocol = _require_object(manifest["protocol"], "manifest.protocol")
    official_f1 = _require_object(protocol.get("official_f1"), "manifest.protocol.official_f1")
    if official_f1 != {
        "metric": METRIC,
        "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        "upstream_scorer_sha256": OFFICIAL_SCORER_SHA256,
    }:
        raise VerificationError("manifest official F1 identity mismatch")
    common_answer_model = _require_model_id(
        protocol.get("common_answer_model"), "manifest.protocol.common_answer_model"
    )
    common_answer_provider = _require_provider_id(
        protocol.get("common_answer_expected_provider"),
        "manifest.protocol.common_answer_expected_provider",
    )
    answer_endpoint = _require_object(
        protocol.get("common_answer_endpoint"), "manifest.protocol.common_answer_endpoint"
    )
    if answer_endpoint.get("model") != common_answer_model:
        raise VerificationError("manifest common answer model/endpoint mismatch")
    answer_extra_body = _require_object(
        answer_endpoint.get("extra_body"),
        "manifest.protocol.common_answer_endpoint.extra_body",
    )
    _require_exact_keys(
        answer_extra_body,
        {"provider"},
        set(),
        "manifest.protocol.common_answer_endpoint.extra_body",
    )
    answer_routing = _require_object(
        answer_extra_body.get("provider"),
        "manifest.protocol.common_answer_endpoint.extra_body.provider",
    )
    _require_exact_keys(
        answer_routing,
        {
            "allow_fallbacks",
            "data_collection",
            "max_price",
            "order",
            "require_parameters",
        },
        set(),
        "manifest.protocol.common_answer_endpoint.extra_body.provider",
    )
    provider_order = [
        _require_provider_id(
            provider,
            f"manifest.protocol.common_answer_endpoint.extra_body.provider.order[{index}]",
        )
        for index, provider in enumerate(
            _require_list(
                answer_routing["order"],
                "manifest.protocol.common_answer_endpoint.extra_body.provider.order",
            )
        )
    ]
    if provider_order != [common_answer_provider]:
        raise VerificationError(
            "manifest common answer expected provider/routing order mismatch"
        )
    if (
        answer_routing["allow_fallbacks"] is not False
        or answer_routing["require_parameters"] is not True
        or answer_routing["data_collection"] != "deny"
    ):
        raise VerificationError(
            "manifest common answer provider routing is not exact and privacy-pinned"
        )
    answer_protocol = _validate_answer_protocol(protocol.get("answer_protocol"))
    answer_prompt_version = answer_protocol["protocol"]
    answer_generation = _require_object(
        protocol.get("answer_generation_options"),
        "manifest.protocol.answer_generation_options",
    )
    if answer_generation.get("enable_thinking") is not False:
        raise VerificationError("manifest common answer generation is not a no-thinking profile")

    output_entries = _require_list(manifest["answer_outputs"], "manifest.answer_outputs")
    outputs: dict[str, str] = {}
    for index, raw_output in enumerate(output_entries):
        output = _require_object(raw_output, f"manifest.answer_outputs[{index}]")
        _require_exact_keys(
            output,
            {"id", "cohort"},
            set(),
            f"manifest.answer_outputs[{index}]",
        )
        output_id = _require_method(output["id"], f"manifest.answer_outputs[{index}].id")
        cohort = output["cohort"]
        if cohort not in {"all_qa", "retrieval_eligible"}:
            raise VerificationError(f"manifest answer output {output_id} has unknown cohort")
        if output_id in outputs:
            raise VerificationError(f"manifest duplicates answer output {output_id}")
        outputs[output_id] = cohort
    all_outputs = {method for method, cohort in outputs.items() if cohort == "all_qa"}
    imported_outputs = {
        method for method, cohort in outputs.items() if cohort == "retrieval_eligible"
    }
    if len(all_outputs) < 2:
        raise VerificationError(
            "manifest must declare a baseline and at least one all-QA candidate"
        )

    method_parents: list[dict[str, Any]] = []
    parent_ids: set[str] = set()
    for index, raw_parent in enumerate(_require_list(manifest["methods"], "manifest.methods")):
        parent = _require_object(raw_parent, f"manifest.methods[{index}]")
        parent_id = _require_method(parent.get("id"), f"manifest.methods[{index}].id")
        if parent_id in parent_ids:
            raise VerificationError(f"manifest duplicates method parent {parent_id}")
        parent_ids.add(parent_id)
        method_parents.append(parent)
    bm25 = [parent for parent in method_parents if parent.get("kind") == "bm25"]
    semafuse = [
        parent
        for parent in method_parents
        if parent.get("kind") == "semafuse_full_memory_and_retrieval"
    ]
    imported = [
        parent for parent in method_parents if parent.get("kind") == "imported_source_ranking_rag"
    ]
    if len(bm25) != 1 or not semafuse:
        raise VerificationError(
            "manifest method structure requires one BM25 baseline and Semafuse candidates"
        )
    if len(method_parents) != len(bm25) + len(semafuse) + len(imported):
        raise VerificationError("manifest contains an unsupported full-answer method kind")
    baseline = _require_method(manifest["baseline_method"], "manifest.baseline_method")
    if baseline != bm25[0]["id"] or baseline not in all_outputs:
        raise VerificationError("manifest baseline is not the all-QA BM25 output")
    semafuse_outputs: set[str] = set()
    parent_by_output: dict[str, dict[str, Any]] = {baseline: bm25[0]}
    for parent in semafuse:
        label = f"Semafuse method {parent['id']}"
        raw_parent_outputs = _require_list(parent.get("outputs"), f"{label}.outputs")
        parent_outputs = {
            _require_method(output, f"{label}.outputs") for output in raw_parent_outputs
        }
        if len(parent_outputs) != len(raw_parent_outputs):
            raise VerificationError(f"{label} outputs contain a duplicate")
        main_output = _require_method(parent.get("main_output"), f"{label}.main_output")
        raw_stage_outputs = _require_list(
            parent.get("stage_outputs"), f"{label}.stage_outputs"
        )
        stage_outputs = {
            _require_method(output, f"{label}.stage_outputs")
            for output in raw_stage_outputs
        }
        if (
            main_output != parent["id"]
            or len(stage_outputs) != len(raw_stage_outputs)
            or parent_outputs != {main_output} | stage_outputs
            or main_output in stage_outputs
        ):
            raise VerificationError(f"{label} main/stage output topology mismatch")
        overlap = semafuse_outputs & parent_outputs
        if overlap:
            raise VerificationError(
                f"manifest binds Semafuse outputs more than once: {sorted(overlap)}"
            )
        semafuse_outputs.update(parent_outputs)
        parent_by_output.update({output: parent for output in parent_outputs})
    if semafuse_outputs != all_outputs - {baseline}:
        raise VerificationError("manifest Semafuse outputs do not match the all-QA output set")
    if {parent["id"] for parent in imported} != imported_outputs:
        raise VerificationError("manifest imported methods do not match retrieval-eligible outputs")

    parent_by_output.update({parent["id"]: parent for parent in imported})
    method_receipts = [
        _compact_method_identity(parent_by_output[output], output, outputs[output])
        for output in sorted(outputs)
    ]
    output_identities = {identity["id"]: identity for identity in method_receipts}

    implementation = _require_object(manifest["implementation"], "manifest.implementation")
    implementation_receipt = {
        "crate_version": _require_safe_value(
            implementation.get("crate_version"), "manifest.implementation.crate_version"
        ),
        "cargo_lock_sha256": _require_sha256(
            implementation.get("cargo_lock_sha256"), "manifest.implementation.cargo_lock_sha256"
        ),
        "benchmark_code_sha256": _require_sha256(
            implementation.get("benchmark_code_sha256"),
            "manifest.implementation.benchmark_code_sha256",
        ),
    }
    return {
        "run_key": run_key,
        "memory_run_key": memory_run_key,
        "baseline": baseline,
        "outputs": outputs,
        "all_outputs": all_outputs,
        "imported_outputs": imported_outputs,
        "expected_records": sum(
            OFFICIAL_QA_N if cohort == "all_qa" else OFFICIAL_RETRIEVAL_N
            for cohort in outputs.values()
        ),
        "selected_question_ids_sha256": selected_question_ids_sha256,
        "method_receipts": method_receipts,
        "output_identities": output_identities,
        "common_answer_model": common_answer_model,
        "common_answer_provider": common_answer_provider,
        "answer_protocol": answer_protocol,
        "answer_prompt_version": answer_prompt_version,
        "reasoning_profile": "no_thinking",
        "implementation": implementation_receipt,
    }


def _load_summary(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw, artifact = _read_file(path, "summary")
    envelope = _require_object(_parse_json(raw, "summary"), "summary")
    _require_exact_keys(envelope, {"run_key", "summary"}, set(), "summary")
    run_key = _require_sha256(envelope["run_key"], "summary.run_key")
    body = _require_object(envelope["summary"], "summary.summary")
    _require_exact_keys(body, {"manifest", "summary"}, set(), "summary.summary")
    manifest_info = _validate_manifest(body["manifest"], run_key)
    report = _require_object(body["summary"], "summary.summary.summary")
    _require_exact_keys(
        report,
        {
            "metric",
            "complete",
            "method_specific_denominators",
            "no_single_all_methods_denominator",
            "progress",
            "official_f1_by_cohort",
            "usage",
        },
        set(),
        "summary.summary.summary",
    )
    if (
        report["metric"] != METRIC
        or report["complete"] is not True
        or report["method_specific_denominators"] is not True
        or report["no_single_all_methods_denominator"] is not True
        or report["official_f1_by_cohort"] is None
    ):
        raise VerificationError("summary report is incomplete or uses the wrong metric semantics")
    artifact.update(
        {
            "embedded_manifest_sha256": hashlib.sha256(
                _canonical_json(body["manifest"])
            ).hexdigest(),
            "embedded_report_sha256": hashlib.sha256(_canonical_json(report)).hexdigest(),
        }
    )
    return body["manifest"], report, manifest_info, artifact


def _index_and_validate_populations(
    memories: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    manifest_info: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = manifest_info["outputs"]
    memory_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    memory_by_method: dict[str, dict[str, Mapping[str, Any]]] = {
        method: {} for method in outputs
    }
    for record in memories:
        pair = record["pair"]
        if record["method"] not in outputs:
            raise VerificationError(f"memory journal contains unknown method {record['method']}")
        if pair in memory_by_pair:
            raise VerificationError(f"memory journal duplicates pair {pair}")
        memory_by_pair[pair] = record
        memory_by_method[record["method"]][record["question_id"]] = record

    answer_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    answer_by_method: dict[str, dict[str, Mapping[str, Any]]] = {
        method: {} for method in outputs
    }
    for record in answers:
        pair = record["pair"]
        if record["method"] not in outputs:
            raise VerificationError(f"answer journal contains unknown method {record['method']}")
        if pair in answer_by_pair:
            raise VerificationError(f"answer journal duplicates pair {pair}")
        answer_by_pair[pair] = record
        answer_by_method[record["method"]][record["question_id"]] = record

    expected_records = manifest_info["expected_records"]
    if len(memories) != expected_records or len(answers) != expected_records:
        raise VerificationError(
            f"full run is incomplete: expected {expected_records} memory and answer rows; "
            f"found {len(memories)} memory and {len(answers)} answer rows"
        )
    if set(memory_by_pair) != set(answer_by_pair):
        missing_answers = len(set(memory_by_pair) - set(answer_by_pair))
        missing_memories = len(set(answer_by_pair) - set(memory_by_pair))
        raise VerificationError(
            "memory/answer pair sets differ: "
            f"missing_answers={missing_answers}, missing_memories={missing_memories}"
        )

    baseline = manifest_info["baseline"]
    all_outputs = manifest_info["all_outputs"]
    imported_outputs = manifest_info["imported_outputs"]
    all_question_ids = set(memory_by_method[baseline])
    if len(all_question_ids) != OFFICIAL_QA_N:
        raise VerificationError(
            "baseline population is incomplete: "
            f"expected {OFFICIAL_QA_N}, found {len(all_question_ids)}"
        )
    retrieval_ids = {
        question_id
        for question_id, record in memory_by_method[baseline].items()
        if record["retrieval_eligible"]
    }
    if len(retrieval_ids) != OFFICIAL_RETRIEVAL_N:
        raise VerificationError(
            "retrieval-eligible population mismatch: "
            f"expected {OFFICIAL_RETRIEVAL_N}, found {len(retrieval_ids)}"
        )
    for method in sorted(all_outputs):
        if set(memory_by_method[method]) != all_question_ids:
            raise VerificationError(f"all-QA method {method} does not contain the exact 314 IDs")
    for method in sorted(imported_outputs):
        if set(memory_by_method[method]) != retrieval_ids:
            raise VerificationError(
                f"retrieval-eligible method {method} does not contain the exact 312 IDs"
            )

    question_metadata: dict[str, tuple[str, int, bool]] = {}
    for question_id in sorted(all_question_ids):
        baseline_memory = memory_by_method[baseline][question_id]
        expected_metadata = (
            baseline_memory["category"],
            baseline_memory["category_number"],
            baseline_memory["retrieval_eligible"],
        )
        question_metadata[question_id] = expected_metadata
        for method in all_outputs:
            memory = memory_by_method[method][question_id]
            if (
                memory["category"],
                memory["category_number"],
                memory["retrieval_eligible"],
            ) != expected_metadata:
                raise VerificationError(f"evaluator metadata drifts for question {question_id}")
        if question_id in retrieval_ids:
            for method in imported_outputs:
                memory = memory_by_method[method][question_id]
                if (
                    memory["category"],
                    memory["category_number"],
                    memory["retrieval_eligible"],
                ) != expected_metadata:
                    raise VerificationError(f"imported evaluator metadata drifts for {question_id}")

    for pair, answer in answer_by_pair.items():
        memory = memory_by_pair[pair]
        if (
            answer["context_sha256"] != memory["context_sha256"]
            or answer["category"] != memory["category"]
            or answer["category_number"] != memory["category_number"]
        ):
            raise VerificationError(f"answer is not bound to its matching memory for pair {pair}")

    category_counts_all = Counter(
        question_metadata[question_id][0] for question_id in all_question_ids
    )
    category_counts_retrieval = Counter(
        question_metadata[question_id][0] for question_id in retrieval_ids
    )
    all_question_set_sha256 = _question_set_sha256(all_question_ids)
    selected_question_ids_sha256 = _selected_question_ids_sha256(all_question_ids)
    eligible_question_set_sha256 = _question_set_sha256(retrieval_ids)
    evaluator_metadata_sha256 = _binding_sha256(
        b"semafuse-locomo-answer-evaluator-metadata/v1",
        (
            (
                question_id,
                category,
                str(category_number),
                "1" if eligible else "0",
            )
            for question_id, (category, category_number, eligible) in question_metadata.items()
        ),
    )
    if all_question_set_sha256 != OFFICIAL_ALL_QUESTION_SET_SHA256:
        raise VerificationError("all-QA question ID set differs from the pinned official set")
    if selected_question_ids_sha256 != manifest_info["selected_question_ids_sha256"]:
        raise VerificationError("journal question IDs differ from the manifest selection digest")
    if eligible_question_set_sha256 != OFFICIAL_ELIGIBLE_QUESTION_SET_SHA256:
        raise VerificationError(
            "retrieval-eligible question ID set differs from the pinned official set"
        )
    if dict(sorted(category_counts_all.items())) != OFFICIAL_CATEGORY_COUNTS_ALL:
        raise VerificationError("all-QA category counts differ from the pinned official counts")
    if dict(sorted(category_counts_retrieval.items())) != OFFICIAL_CATEGORY_COUNTS_ELIGIBLE:
        raise VerificationError(
            "retrieval-eligible category counts differ from the pinned official counts"
        )
    if evaluator_metadata_sha256 != OFFICIAL_EVALUATOR_METADATA_SHA256:
        raise VerificationError("per-question evaluator metadata differs from the pinned digest")
    memory_binding = _binding_sha256(
        b"semafuse-locomo-answer-memory-bindings/v1",
        (
            (record["method"], record["question_id"], record["context_sha256"])
            for record in memories
        ),
    )
    answer_binding = _binding_sha256(
        b"semafuse-locomo-answer-prediction-bindings/v1",
        (
            (
                record["method"],
                record["question_id"],
                record["context_sha256"],
                record["prediction_sha256"],
                struct.pack(">d", record["score"]).hex(),
            )
            for record in answers
        ),
    )
    return {
        "memory_by_method": memory_by_method,
        "answer_by_method": answer_by_method,
        "all_question_ids": all_question_ids,
        "retrieval_ids": retrieval_ids,
        "question_metadata": question_metadata,
        "category_counts_all": dict(sorted(category_counts_all.items())),
        "category_counts_retrieval": dict(sorted(category_counts_retrieval.items())),
        "all_question_set_sha256": all_question_set_sha256,
        "selected_question_ids_sha256": selected_question_ids_sha256,
        "eligible_question_set_sha256": eligible_question_set_sha256,
        "evaluator_metadata_sha256": evaluator_metadata_sha256,
        "memory_binding_sha256": memory_binding,
        "answer_binding_sha256": answer_binding,
    }


def _validate_progress(
    report: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
    manifest_info: Mapping[str, Any],
) -> None:
    progress = _require_object(report["progress"], "summary.progress")
    if set(progress) != set(manifest_info["outputs"]):
        raise VerificationError("summary progress/output method sets differ")
    for method, cohort in sorted(manifest_info["outputs"].items()):
        method_memories = [record for record in memories if record["method"] == method]
        method_answers = [record for record in answers if record["method"] == method]
        expected_n = OFFICIAL_QA_N if cohort == "all_qa" else OFFICIAL_RETRIEVAL_N
        expected = {
            "memory_n": expected_n,
            "answer_n": expected_n,
            "required_n": expected_n,
            "cohort": cohort,
            "partial_mean_f1": sum(record["score"] for record in method_answers) / expected_n,
            "mean_context_bytes": sum(record["context_bytes"] for record in method_memories)
            / expected_n,
        }
        _assert_equivalent(progress[method], expected, f"summary.progress.{method}")


def _validate_usage_summary(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    memories: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    usage = _require_object(report["usage"], "summary.usage")
    _require_exact_keys(
        usage,
        {
            "processor",
            "processor_completion_tokens_per_call",
            "answerer",
            "answer_completion_tokens_per_call",
            "estimated_or_reported_usd",
            "configured_fuse_usd",
            "scope",
            "untracked_costs",
        },
        set(),
        "summary.usage",
    )
    processor = _aggregate_usage(record["processor_usage"] for record in memories)
    answerer = _aggregate_usage(record["answer_usage"] for record in answers)
    actual_processor = _validate_usage(usage["processor"], "summary.usage.processor")
    actual_answerer = _validate_usage(usage["answerer"], "summary.usage.answerer")
    _usage_equal(actual_processor, processor, "summary.usage.processor")
    _usage_equal(actual_answerer, answerer, "summary.usage.answerer")

    expected_processor_ratio = (
        processor["completion_tokens"] / processor["calls"] if processor["calls"] else None
    )
    expected_answer_ratio = (
        answerer["completion_tokens"] / answerer["calls"] if answerer["calls"] else None
    )
    _assert_equivalent(
        usage["processor_completion_tokens_per_call"],
        expected_processor_ratio,
        "summary.usage.processor_completion_tokens_per_call",
    )
    _assert_equivalent(
        usage["answer_completion_tokens_per_call"],
        expected_answer_ratio,
        "summary.usage.answer_completion_tokens_per_call",
    )

    pricing = _require_object(manifest["pricing_guard"], "manifest.pricing_guard")
    input_price = _finite_number(
        pricing.get("input_usd_per_million"),
        "manifest.pricing_guard.input_usd_per_million",
        minimum=0.0,
    )
    output_price = _finite_number(
        pricing.get("output_usd_per_million"),
        "manifest.pricing_guard.output_usd_per_million",
        minimum=0.0,
    )
    fuse = _finite_number(
        pricing.get("max_reported_usage_usd"),
        "manifest.pricing_guard.max_reported_usage_usd",
        minimum=0.0,
    )

    def estimate(record_usage: Mapping[str, Any]) -> float:
        if record_usage["cost"] is not None:
            return record_usage["cost"]
        return (
            record_usage["prompt_tokens"] * input_price / 1_000_000.0
            + record_usage["completion_tokens"] * output_price / 1_000_000.0
        )

    estimated = sum(estimate(record["processor_usage"]) for record in memories) + sum(
        estimate(record["answer_usage"]) for record in answers
    )
    _assert_equivalent(
        usage["estimated_or_reported_usd"], estimated, "summary.usage.estimated_or_reported_usd"
    )
    _assert_equivalent(usage["configured_fuse_usd"], fuse, "summary.usage.configured_fuse_usd")
    if usage["scope"] != "semafuse_processor_and_common_answerer_only":
        raise VerificationError("summary usage scope mismatch")
    expected_untracked = [
        "embedding",
        "reranker",
        "failed or uncommitted completion calls whose usage was not journaled",
    ]
    if usage["untracked_costs"] != expected_untracked:
        raise VerificationError("summary untracked-cost declaration mismatch")
    return {
        "processor": processor,
        "processor_completion_tokens_per_call": expected_processor_ratio,
        "answerer": answerer,
        "answer_completion_tokens_per_call": expected_answer_ratio,
        "estimated_or_reported_usd": estimated,
        "configured_fuse_usd": fuse,
        "scope": "semafuse_processor_and_common_answerer_only",
        "untracked_cost_categories": [
            "embedding",
            "reranker",
            "failed_uncommitted_completion_calls",
        ],
    }


def _recompute_and_validate_reports(
    report: Mapping[str, Any],
    population: Mapping[str, Any],
    manifest_info: Mapping[str, Any],
) -> dict[str, Any]:
    answer_by_method = population["answer_by_method"]
    all_ids = population["all_question_ids"]
    retrieval_ids = population["retrieval_ids"]
    all_methods = manifest_info["all_outputs"]
    all_report = _build_scalar_report(
        answer_by_method, all_methods, all_ids, manifest_info["baseline"]
    )

    eligible_records: dict[str, dict[str, Mapping[str, Any]]] = {}
    for method in manifest_info["outputs"]:
        eligible_records[method] = {
            question_id: answer_by_method[method][question_id]
            for question_id in retrieval_ids
        }
    eligible_report = _build_scalar_report(
        eligible_records,
        manifest_info["outputs"],
        retrieval_ids,
        manifest_info["baseline"],
    )
    expected = {
        "all_qa_314": {
            "question_n": OFFICIAL_QA_N,
            "methods": sorted(all_methods),
            "official_f1": all_report,
        },
        "retrieval_eligible_312": {
            "question_n": OFFICIAL_RETRIEVAL_N,
            "methods": sorted(manifest_info["outputs"]),
            "official_f1": eligible_report,
        },
    }
    _assert_equivalent(
        report["official_f1_by_cohort"], expected, "summary.official_f1_by_cohort"
    )
    return expected


def _looks_like_absolute_path(value: str) -> bool:
    candidate = value.strip()
    return (
        "file://" in candidate.lower()
        or _POSIX_ABSOLUTE_PATH.search(candidate) is not None
        or _WINDOWS_ABSOLUTE_PATH.search(candidate) is not None
        or _UNC_ABSOLUTE_PATH.search(candidate) is not None
    )


def _assert_receipt_safe(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise VerificationError("receipt contains a non-string object key")
            normalized = key.lower().replace("-", "_")
            if _looks_like_absolute_path(key):
                raise VerificationError("receipt contains an absolute path as an object key")
            if normalized == "credentials_serialized":
                if child is not False:
                    raise VerificationError("receipt credential-safety flag is not false")
            elif _CREDENTIAL_KEY.search(key):
                raise VerificationError("receipt contains a credential-like object key")
            if normalized == "raw_text_serialized":
                if child is not False:
                    raise VerificationError("receipt raw-text safety flag is not false")
            elif normalized in _RAW_TEXT_KEYS:
                raise VerificationError(f"receipt contains forbidden raw-text field {key!r}")
            _assert_receipt_safe(child)
        return
    if isinstance(value, list):
        for child in value:
            _assert_receipt_safe(child)
        return
    if isinstance(value, str):
        if len(value) > 1_024 or any(character in value for character in ("\n", "\r", "\0")):
            raise VerificationError("receipt contains an unbounded or multiline string")
        if _CREDENTIAL_VALUE.search(value) or _URI_WITH_CREDENTIALS.search(value):
            raise VerificationError("receipt contains a credential-like value")
        if _looks_like_absolute_path(value):
            raise VerificationError("receipt contains an absolute path")


def build_receipt(
    *,
    summary_path: Path,
    memory_journal_path: Path,
    answers_journal_path: Path,
    expected_summary_sha256: str | None = None,
    expected_memory_sha256: str | None = None,
    expected_answers_sha256: str | None = None,
) -> dict[str, Any]:
    pins = (expected_summary_sha256, expected_memory_sha256, expected_answers_sha256)
    if any(value is not None for value in pins) and not all(value is not None for value in pins):
        raise VerificationError("caller artifact SHA-256 pins must be supplied all together")

    manifest, report, manifest_info, summary_artifact = _load_summary(summary_path)
    processor_telemetry = _new_response_telemetry()
    answer_telemetry = _new_response_telemetry()
    memories, memory_artifact = _read_journal(
        memory_journal_path,
        "memory journal",
        lambda value, label: _validate_memory_record(
            value,
            run_key=manifest_info["memory_run_key"],
            method_identities=manifest_info["output_identities"],
            telemetry=processor_telemetry,
            label=label,
        ),
    )
    answers, answers_artifact = _read_journal(
        answers_journal_path,
        "answers journal",
        lambda value, label: _validate_answer_record(
            value,
            run_key=manifest_info["run_key"],
            memory_run_key=manifest_info["memory_run_key"],
            expected_prompt_version=manifest_info["answer_prompt_version"],
            expected_answer_protocol=manifest_info["answer_protocol"],
            expected_model=manifest_info["common_answer_model"],
            expected_provider=manifest_info["common_answer_provider"],
            telemetry=answer_telemetry,
            label=label,
        ),
    )
    _check_expected_hash(summary_artifact["sha256"], expected_summary_sha256, "summary")
    _check_expected_hash(memory_artifact["sha256"], expected_memory_sha256, "memory journal")
    _check_expected_hash(answers_artifact["sha256"], expected_answers_sha256, "answers journal")

    population = _index_and_validate_populations(memories, answers, manifest_info)
    _validate_progress(report, memories, answers, manifest_info)
    usage = _validate_usage_summary(manifest, report, memories, answers)
    recomputed_reports = _recompute_and_validate_reports(report, population, manifest_info)
    if processor_telemetry["responses"] != usage["processor"]["calls"]:
        raise VerificationError("processor response telemetry/call accounting differs")
    if answer_telemetry["responses"] != usage["answerer"]["calls"]:
        raise VerificationError("answer response telemetry/call accounting differs")

    tool_raw, tool_artifact = _read_file(Path(__file__), "compact answer receipt tool")
    del tool_raw
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "LoCoMo",
        "track": TRACK,
        "classification": {
            "paper_reproduction_eligible": False,
            "metric": METRIC,
            "metric_units": "fraction",
            "reasoning_profile": manifest_info["reasoning_profile"],
            "common_answer_provider": manifest_info["common_answer_provider"],
            "metric_notice": (
                "Released category-aware LoCoMo token F1; not the LazyMem binary LLM judge."
            ),
        },
        "population": {
            "artifact_output_cohorts": {
                "all_qa_314": {
                    "question_count": OFFICIAL_QA_N,
                    "output_method_count": len(manifest_info["all_outputs"]),
                    "pair_count": len(manifest_info["all_outputs"]) * OFFICIAL_QA_N,
                    "question_set_sha256": population["all_question_set_sha256"],
                    "category_counts": population["category_counts_all"],
                },
                "retrieval_eligible_312": {
                    "question_count": OFFICIAL_RETRIEVAL_N,
                    "output_method_count": len(manifest_info["imported_outputs"]),
                    "comparison_method_count": len(manifest_info["outputs"]),
                    "pair_count": len(manifest_info["imported_outputs"])
                    * OFFICIAL_RETRIEVAL_N,
                    "question_set_sha256": population["eligible_question_set_sha256"],
                    "category_counts": population["category_counts_retrieval"],
                },
            },
            "expected_memory_records": manifest_info["expected_records"],
            "expected_answer_records": manifest_info["expected_records"],
            "manifest_selected_question_ids_sha256": population[
                "selected_question_ids_sha256"
            ],
            "opaque_question_ids_serialized": False,
            "evaluator_metadata_sha256": population["evaluator_metadata_sha256"],
        },
        "provenance": {
            "dataset_sha256": OFFICIAL_DATASET_SHA256,
            "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
            "official_scorer_sha256": OFFICIAL_SCORER_SHA256,
            "run_key": manifest_info["run_key"],
            "memory_run_key": manifest_info["memory_run_key"],
            "common_answer_model": manifest_info["common_answer_model"],
            "answer_prompt_version": manifest_info["answer_prompt_version"],
            "answer_protocol": manifest_info["answer_protocol"],
            "implementation": manifest_info["implementation"],
            "tool": {
                "path": "benchmarks/locomo/compact_answer_receipt_v2.py",
                "sha256": tool_artifact["sha256"],
            },
        },
        "artifacts": {
            "summary": summary_artifact,
            "memory_journal": memory_artifact,
            "answers_journal": answers_artifact,
            "memory_binding_sha256": population["memory_binding_sha256"],
            "answer_binding_sha256": population["answer_binding_sha256"],
        },
        "validation": {
            "complete_manifest_and_summary": True,
            "caller_hash_pins_supplied_and_matched": all(value is not None for value in pins),
            "journal_envelopes_and_run_keys_validated": True,
            "unique_method_question_pairs_validated": True,
            "cohort_314_and_312_coverage_validated": True,
            "official_question_sets_categories_and_metadata_pinned": True,
            "manifest_selection_track_and_hash_validated": True,
            "memory_answer_pair_sets_identical": True,
            "memory_outputs_bound_to_manifest_method_identity": True,
            "context_hashes_and_byte_counts_validated": len(memories),
            "prediction_hashes_validated": len(answers),
            "summary_means_deltas_and_bootstrap_recomputed": True,
            "usage_and_response_telemetry_reconciled": True,
            "manifest_and_processor_no_thinking_profile_bound": True,
            "answer_reasoning_absent_and_zero_tokens": True,
            "processor_reasoning_absent_with_delimiter_allowance": True,
            "finish_reason_contract_validated": True,
            "answer_prompt_model_and_provider_bound": True,
            "official_f1_text_rescored": False,
            "raw_text_serialized": False,
            "absolute_paths_serialized": False,
            "credentials_serialized": False,
            "recursive_safety_scan_passed": True,
        },
        "bootstrap": {
            "unit": "opaque_question_id",
            "sampling": "with_replacement",
            "prng": "SplitMix64_unbiased_rejection_sampling",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "interval": "percentile_linear",
        },
        "baseline": manifest_info["baseline"],
        "methods": manifest_info["method_receipts"],
        "results": {"official_f1_by_cohort": recomputed_reports},
        "usage": {
            **usage,
            "processor_response_telemetry": _finalize_response_telemetry(processor_telemetry),
            "answerer_response_telemetry": _finalize_response_telemetry(answer_telemetry),
        },
        "limitations": [
            (
                "Reference labels are absent from the receipt inputs, so per-question F1 "
                "is not text-rescored."
            ),
            *(
                [
                    "Imported source-ranking outputs are pinned RAG contexts, not native "
                    "memory contexts."
                ]
                if manifest_info["imported_outputs"]
                else []
            ),
            (
                "Embedding, reranker, and failed uncommitted call costs are outside journal "
                "accounting."
            ),
            "Raw memory contexts and predictions remain only in the local input journals.",
        ],
    }
    serialized = _canonical_json(receipt)
    _assert_receipt_safe(_parse_json(serialized, "serialized receipt"))
    return receipt


def _write_json_atomic(path: Path, value: Any) -> None:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VerificationError(f"cannot create receipt directory: {error}") from error
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The receipt itself is already flushed and atomically installed;
            # some filesystems do not permit opening directories for fsync.
            pass
    except OSError as error:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
        raise VerificationError(f"cannot atomically write receipt: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--memory-journal", type=Path, required=True)
    parser.add_argument("--answers-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256")
    parser.add_argument("--expected-memory-sha256")
    parser.add_argument("--expected-answers-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {args.summary.resolve(), args.memory_journal.resolve(), args.answers_journal.resolve()}
    if args.output.resolve() in inputs:
        print("verification failed: output must not overwrite an input artifact", file=sys.stderr)
        return 2
    try:
        receipt = build_receipt(
            summary_path=args.summary,
            memory_journal_path=args.memory_journal,
            answers_journal_path=args.answers_journal,
            expected_summary_sha256=args.expected_summary_sha256,
            expected_memory_sha256=args.expected_memory_sha256,
            expected_answers_sha256=args.expected_answers_sha256,
        )
        _write_json_atomic(args.output, receipt)
        return 0
    except VerificationError as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
