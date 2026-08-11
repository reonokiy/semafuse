"""JSONL worker process for one external-memory implementation."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from enum import Enum
from typing import Any

from .adapter import ExternalMemoryAdapter, TokenCounter, build_token_counter, enforce_limits
from .config import WorkerConfig, environment_secrets, redact
from .errors import AdapterUnavailableError, ConfigurationError, ProtocolError
from .protocol import PROTOCOL_VERSION, Request, decode_line, encode_line
from .registry import build_adapter


class _State(Enum):
    NEW = "new"
    READY = "ready"
    POISONED = "poisoned"
    CLOSED = "closed"


class Worker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.state = _State.NEW
        self.adapter: ExternalMemoryAdapter | None = None
        self.counter: TokenCounter | None = None
        self.secrets: tuple[str, ...] = environment_secrets()

    def handle(self, request: Request) -> dict[str, Any]:
        try:
            result = self._dispatch(request)
            return {
                "protocol": PROTOCOL_VERSION,
                "request_id": request.request_id,
                "op": request.op,
                "ok": True,
                "result": result,
            }
        except Exception as exc:  # Errors cross the process boundary as data.
            if isinstance(exc, AdapterUnavailableError):
                code = "ADAPTER_UNAVAILABLE"
            elif isinstance(exc, (ConfigurationError, ValueError)):
                code = "CONFIGURATION_ERROR"
            elif isinstance(exc, ProtocolError):
                code = "PROTOCOL_ERROR"
            else:
                code = "ADAPTER_ERROR"
            return {
                "protocol": PROTOCOL_VERSION,
                "request_id": request.request_id,
                "op": request.op,
                "ok": False,
                "error": {
                    "code": code,
                    "message": redact(str(exc) or type(exc).__name__, self.secrets),
                    "retryable": False,
                },
            }

    def _dispatch(self, request: Request) -> dict[str, Any]:
        if self.state is _State.CLOSED:
            raise ProtocolError("worker is already closed")
        if request.op == "reset":
            # Clear READY before touching upstream state. A failed reset leaves
            # stale memory unreachable until another reset succeeds.
            self.state = _State.POISONED
            if self.adapter is None:
                self.adapter = build_adapter(self.config.adapter_kind, self.config.adapter_config)
                self.secrets = tuple(dict.fromkeys((*self.secrets, *getattr(self.adapter, "secrets", ()))))
            if self.counter is None:
                self.counter = build_token_counter(self.config.tokenizer)
            self.adapter.reset()
            self.state = _State.READY
            return {"reset": True}
        if self.state is not _State.READY or self.adapter is None or self.counter is None:
            raise ProtocolError("reset must succeed before add, search, or close")
        if request.op == "add":
            self.state = _State.POISONED
            accepted = self.adapter.add(request.messages, request.session or "", request.timestamp)
            if isinstance(accepted, bool) or not isinstance(accepted, int) or not 0 <= accepted <= len(request.messages):
                raise RuntimeError("adapter returned an invalid accepted count")
            self.state = _State.READY
            return {"accepted": accepted}
        if request.op == "search":
            items = self.adapter.search(request.query or "", request.top_k or 1)
            return dict(enforce_limits(items, request.top_k or 1, request.max_context_tokens or 1, self.counter))
        if request.op == "close":
            self.state = _State.POISONED
            self.adapter.close()
            self.state = _State.CLOSED
            return {"closed": True}
        raise ProtocolError(f"unsupported operation {request.op}")


def serve(config: WorkerConfig) -> int:
    worker = Worker(config)
    with open(os.devnull, "w", encoding="utf-8") as upstream_output:
        for line in sys.stdin:
            try:
                request = Request.parse(decode_line(line))
            except ProtocolError as exc:
                print(f"fatal protocol error: {exc}", file=sys.stderr)
                return 2
            # Third-party packages sometimes print full requests or headers.
            # Neither stream is part of the protocol, so discard both rather
            # than risking credential disclosure or corrupting JSONL stdout.
            with contextlib.redirect_stdout(upstream_output), contextlib.redirect_stderr(upstream_output):
                response = worker.handle(request)
            sys.stdout.write(encode_line(response))
            sys.stdout.flush()
            if request.op == "close" and response["ok"]:
                return 0
    return 0 if worker.state is _State.CLOSED else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = WorkerConfig.load(args.config)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    return serve(config)


if __name__ == "__main__":
    raise SystemExit(main())
