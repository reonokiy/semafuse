"""Strict schema gate and deterministic cross-window decision reduction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .types import Message
from .windows import EvidenceWindow


class DecisionFormatError(ValueError):
    pass


def _reject_non_finite_number(value: str) -> None:
    raise DecisionFormatError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DecisionFormatError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def strip_optional_thinking_block(raw: str) -> str:
    """Accept JSON directly or after exactly one leading ``<think>`` block."""

    if not isinstance(raw, str):
        raise DecisionFormatError("decision output must be a string")
    stripped = raw.strip()
    if "```" in stripped:
        raise DecisionFormatError("markdown fences are not allowed")
    if not stripped.startswith("<think>"):
        if "<think>" in stripped or "</think>" in stripped:
            raise DecisionFormatError("thinking tags are only allowed as one leading block")
        return stripped

    closing = stripped.find("</think>", len("<think>"))
    if closing < 0:
        raise DecisionFormatError("leading thinking block is not closed")
    thinking = stripped[len("<think>") : closing]
    remainder = stripped[closing + len("</think>") :].strip()
    if "<think>" in thinking or "</think>" in thinking:
        raise DecisionFormatError("nested thinking blocks are not allowed")
    if "<think>" in remainder or "</think>" in remainder:
        raise DecisionFormatError("multiple thinking blocks are not allowed")
    if not remainder:
        raise DecisionFormatError("thinking block must be followed by a JSON array")
    return remainder


@dataclass(frozen=True, slots=True)
class EditDecision:
    message: Message
    op: str
    compressed_content: str
    reason: str
    window_id: str
    window_order: int


@dataclass(frozen=True, slots=True)
class WindowDecisions:
    window: EvidenceWindow
    decisions: tuple[EditDecision, ...]


@dataclass(frozen=True, slots=True)
class SelectedMemory:
    message: Message
    compressed_content: str
    source_window_id: str


def parse_edit_decisions(raw: str, window: EvidenceWindow) -> WindowDecisions:
    """Apply the paper's format gate: exactly one strict decision per message."""

    payload = strip_optional_thinking_block(raw)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite_number,
        )
    except json.JSONDecodeError as error:
        raise DecisionFormatError(f"output is not one JSON value: {error}") from error
    if not isinstance(value, list):
        raise DecisionFormatError("decision output must be a JSON array")
    decision_messages = window.decision_messages
    if len(value) != len(decision_messages):
        raise DecisionFormatError(
            f"expected {len(decision_messages)} decisions, received {len(value)}"
        )

    parsed: list[EditDecision] = []
    required_keys = {"op", "compressed_content", "reason"}
    for index, (item, message) in enumerate(
        zip(value, decision_messages, strict=True)
    ):
        if not isinstance(item, dict) or set(item) != required_keys:
            raise DecisionFormatError(
                f"decision {index} must contain exactly {sorted(required_keys)}"
            )
        op = item["op"]
        compressed = item["compressed_content"]
        reason = item["reason"]
        if op not in ("KEEP", "DROP"):
            raise DecisionFormatError(f"decision {index} has invalid op {op!r}")
        if not isinstance(compressed, str) or not isinstance(reason, str):
            raise DecisionFormatError(
                f"decision {index} compressed_content and reason must be strings"
            )
        if op == "KEEP" and not compressed.strip():
            raise DecisionFormatError(f"KEEP decision {index} has empty content")
        if op == "DROP" and compressed != "":
            raise DecisionFormatError(f"DROP decision {index} must have empty content")
        if not reason.strip():
            raise DecisionFormatError(f"decision {index} has empty reason")
        parsed.append(
            EditDecision(
                message=message,
                op=op,
                compressed_content=compressed,
                reason=reason,
                window_id=window.id,
                window_order=window.order,
            )
        )
    return WindowDecisions(window, tuple(parsed))


def deduplicate_decisions(results: Sequence[WindowDecisions]) -> list[SelectedMemory]:
    """Prefer KEEP for duplicate messages, then the earliest window order."""

    chosen: dict[str, EditDecision] = {}
    for result in sorted(results, key=lambda item: item.window.order):
        for decision in result.decisions:
            existing = chosen.get(decision.message.id)
            if existing is None:
                chosen[decision.message.id] = decision
            elif existing.op == "DROP" and decision.op == "KEEP":
                chosen[decision.message.id] = decision
            elif existing.op == decision.op and decision.window_order < existing.window_order:
                chosen[decision.message.id] = decision
    selected = [
        SelectedMemory(decision.message, decision.compressed_content, decision.window_id)
        for decision in chosen.values()
        if decision.op == "KEEP"
    ]
    return sorted(selected, key=lambda item: (item.message.index, item.message.id))


def render_selected_memory(items: Sequence[SelectedMemory]) -> str:
    return "\n".join(
        f"[{item.message.timestamp}] {item.message.role}: {item.compressed_content}"
        for item in items
    )
