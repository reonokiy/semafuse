"""Strict JSONL protocol shared by the benchmark driver and worker."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ProtocolError

PROTOCOL_VERSION = "semafuse.external-memory.v1"
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_MESSAGES_PER_ADD = 4096
MAX_TOP_K = 1000
MAX_CONTEXT_TOKENS = 10_000_000
ROLES = frozenset({"system", "user", "assistant", "tool"})
OPS = frozenset({"reset", "add", "search", "close"})


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{where} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise ProtocolError(f"{where} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ProtocolError(f"{where} has unknown fields: {', '.join(sorted(extra))}")


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ProtocolError(f"{where} must be a non-empty string")
    return value


def _positive_int(value: Any, where: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ProtocolError(f"{where} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    source_id: str

    @classmethod
    def parse(cls, raw: Any, index: int) -> "Message":
        value = _object(raw, f"messages[{index}]")
        _exact_keys(value, {"role", "content", "source_id"}, set(), f"messages[{index}]")
        role = _string(value["role"], f"messages[{index}].role")
        if role not in ROLES:
            raise ProtocolError(f"messages[{index}].role is not supported")
        return cls(
            role=role,
            content=_string(value["content"], f"messages[{index}].content"),
            source_id=_string(value["source_id"], f"messages[{index}].source_id"),
        )

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content, "source_id": self.source_id}


@dataclass(frozen=True)
class Request:
    request_id: str
    op: str
    messages: tuple[Message, ...] = ()
    session: str | None = None
    timestamp: str | None = None
    query: str | None = None
    top_k: int | None = None
    max_context_tokens: int | None = None

    @classmethod
    def parse(cls, raw: Any) -> "Request":
        value = _object(raw, "request")
        _exact_keys(value, {"protocol", "request_id", "op"}, {
            "messages", "session", "timestamp", "query", "top_k", "max_context_tokens"
        }, "request")
        if value["protocol"] != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol: {value['protocol']!r}")
        request_id = _string(value["request_id"], "request.request_id")
        op = _string(value["op"], "request.op")
        if op not in OPS:
            raise ProtocolError(f"unsupported operation: {op!r}")

        expected = {
            "reset": set(),
            "add": {"messages", "session", "timestamp"},
            "search": {"query", "top_k", "max_context_tokens"},
            "close": set(),
        }[op]
        supplied = value.keys() - {"protocol", "request_id", "op"}
        if supplied != expected:
            missing = expected - supplied
            extra = supplied - expected
            details = []
            if missing:
                details.append(f"missing {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unexpected {', '.join(sorted(extra))}")
            raise ProtocolError(f"invalid {op} request: {'; '.join(details)}")

        if op == "add":
            raw_messages = value["messages"]
            if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_MESSAGES_PER_ADD:
                raise ProtocolError(f"request.messages must contain 1..{MAX_MESSAGES_PER_ADD} messages")
            timestamp = value["timestamp"]
            if timestamp is not None:
                timestamp = _string(timestamp, "request.timestamp")
            return cls(
                request_id=request_id,
                op=op,
                messages=tuple(Message.parse(item, i) for i, item in enumerate(raw_messages)),
                session=_string(value["session"], "request.session"),
                timestamp=timestamp,
            )
        if op == "search":
            return cls(
                request_id=request_id,
                op=op,
                query=_string(value["query"], "request.query"),
                top_k=_positive_int(value["top_k"], "request.top_k", MAX_TOP_K),
                max_context_tokens=_positive_int(
                    value["max_context_tokens"], "request.max_context_tokens", MAX_CONTEXT_TOKENS
                ),
            )
        return cls(request_id=request_id, op=op)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "op": self.op,
        }
        if self.op == "add":
            result.update(
                messages=[message.to_json() for message in self.messages],
                session=self.session,
                timestamp=self.timestamp,
            )
        elif self.op == "search":
            result.update(query=self.query, top_k=self.top_k, max_context_tokens=self.max_context_tokens)
        return result


def decode_line(line: str) -> Any:
    if len(line.encode("utf-8")) > MAX_LINE_BYTES:
        raise ProtocolError(f"JSONL record exceeds {MAX_LINE_BYTES} bytes")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON at column {exc.colno}") from exc


def encode_line(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProtocolError("value is not strict JSON") from exc


def validate_response(raw: Any, request: Request) -> Mapping[str, Any]:
    value = _object(raw, "response")
    _exact_keys(value, {"protocol", "request_id", "op", "ok"}, {"result", "error"}, "response")
    if value["protocol"] != PROTOCOL_VERSION or value["request_id"] != request.request_id or value["op"] != request.op:
        raise ProtocolError("response correlation fields do not match the request")
    if not isinstance(value["ok"], bool):
        raise ProtocolError("response.ok must be boolean")
    if value["ok"]:
        if set(value.keys()) != {"protocol", "request_id", "op", "ok", "result"}:
            raise ProtocolError("successful response must contain result and no error")
        _validate_result(value["result"], request)
    else:
        if set(value.keys()) != {"protocol", "request_id", "op", "ok", "error"}:
            raise ProtocolError("failed response must contain error and no result")
        error = _object(value["error"], "response.error")
        _exact_keys(error, {"code", "message", "retryable"}, set(), "response.error")
        _string(error["code"], "response.error.code")
        _string(error["message"], "response.error.message")
        if not isinstance(error["retryable"], bool):
            raise ProtocolError("response.error.retryable must be boolean")
    return value


def _validate_result(raw: Any, request: Request) -> None:
    value = _object(raw, "response.result")
    if request.op == "reset":
        _exact_keys(value, {"reset"}, set(), "response.result")
        if value["reset"] is not True:
            raise ProtocolError("reset result must be true")
    elif request.op == "add":
        _exact_keys(value, {"accepted"}, set(), "response.result")
        if isinstance(value["accepted"], bool) or not isinstance(value["accepted"], int) or value["accepted"] < 0:
            raise ProtocolError("accepted must be a non-negative integer")
        if value["accepted"] > len(request.messages):
            raise ProtocolError("accepted exceeds the number of submitted messages")
    elif request.op == "search":
        _exact_keys(value, {"items", "context", "context_tokens", "truncated"}, set(), "response.result")
        if not isinstance(value["items"], list):
            raise ProtocolError("search items must be an array")
        if len(value["items"]) > (request.top_k or 0):
            raise ProtocolError("search response exceeds request.top_k")
        _string(value["context"], "response.result.context", allow_empty=True)
        if isinstance(value["context_tokens"], bool) or not isinstance(value["context_tokens"], int) or value["context_tokens"] < 0:
            raise ProtocolError("context_tokens must be a non-negative integer")
        if not isinstance(value["truncated"], bool):
            raise ProtocolError("truncated must be boolean")
        if value["context_tokens"] > (request.max_context_tokens or 0):
            raise ProtocolError("search response exceeds request.max_context_tokens")
        for index, item_raw in enumerate(value["items"]):
            item = _object(item_raw, f"response.result.items[{index}]")
            _exact_keys(
                item,
                {"id", "text", "score", "source_ids", "session", "timestamp", "token_count"},
                set(),
                f"response.result.items[{index}]",
            )
            _string(item["id"], f"items[{index}].id")
            _string(item["text"], f"items[{index}].text")
            if item["score"] is not None and (
                isinstance(item["score"], bool)
                or not isinstance(item["score"], (int, float))
                or not math.isfinite(item["score"])
            ):
                raise ProtocolError(f"items[{index}].score must be finite or null")
            if not isinstance(item["source_ids"], list) or any(
                not isinstance(source_id, str) or not source_id for source_id in item["source_ids"]
            ):
                raise ProtocolError(f"items[{index}].source_ids must be an array of non-empty strings")
            if item["session"] is not None:
                _string(item["session"], f"items[{index}].session")
            if item["timestamp"] is not None:
                _string(item["timestamp"], f"items[{index}].timestamp")
            if isinstance(item["token_count"], bool) or not isinstance(item["token_count"], int) or item["token_count"] < 1:
                raise ProtocolError(f"items[{index}].token_count must be positive")
        expected_context = "\n\n".join(item["text"] for item in value["items"])
        if value["context"] != expected_context:
            raise ProtocolError("search context does not match the emitted item texts")
        if bool(value["items"]) != bool(value["context_tokens"]):
            raise ProtocolError("search context_tokens is inconsistent with emitted items")
    elif request.op == "close":
        _exact_keys(value, {"closed"}, set(), "response.result")
        if value["closed"] is not True:
            raise ProtocolError("close result must be true")
