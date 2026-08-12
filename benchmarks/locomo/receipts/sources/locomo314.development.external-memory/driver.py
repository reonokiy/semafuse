"""Timeout-safe subprocess driver for the external-memory JSONL protocol."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import environment_secrets, redact
from .errors import ProcessExitedError, ProtocolError, RemoteAdapterError, RequestTimeoutError
from .protocol import Message, Request, decode_line, encode_line, validate_response

_EOF = object()


class ExternalMemoryDriver:
    def __init__(
        self,
        config_path: str | Path,
        *,
        timeout_seconds: float = 120.0,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.config_path = str(config_path)
        self.timeout_seconds = timeout_seconds
        self.command = list(command) if command else [
            sys.executable,
            "-m",
            "semafuse_external_memory.worker",
            "--config",
            self.config_path,
        ]
        self.env = dict(env) if env is not None else None
        secret_environment = os.environ.copy()
        if self.env is not None:
            secret_environment.update(self.env)
        self._secrets = environment_secrets(secret_environment)
        self._process: subprocess.Popen[str] | None = None
        self._stdout: queue.Queue[object] = queue.Queue()
        self._ready = False
        self._closed = False
        self._terminal = False

    def __enter__(self) -> "ExternalMemoryDriver":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _start(self) -> None:
        if self._process is not None:
            return
        if self._terminal:
            raise ProtocolError("driver is terminal after a previous worker failure")
        child_env = os.environ.copy()
        if self.env is not None:
            child_env.update(self.env)
        self._stdout = queue.Queue()
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=child_env,
            )
        except OSError as exc:
            self._terminal = True
            raise ProcessExitedError("worker could not be started") from exc
        self._process = process

        def pump_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self._stdout.put(line)
            self._stdout.put(_EOF)

        def pump_stderr() -> None:
            # Drain to avoid deadlock, but never retain or surface upstream
            # stderr because SDKs may log credentials or request headers.
            assert process.stderr is not None
            for _ in process.stderr:
                pass

        threading.Thread(target=pump_stdout, daemon=True).start()
        threading.Thread(target=pump_stderr, daemon=True).start()

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _mark_terminal(self) -> None:
        self._terminal = True
        self._ready = False
        self._stop()

    def _rpc(self, request: Request) -> Mapping[str, Any]:
        if self._closed:
            raise ProtocolError("driver is closed")
        if self._terminal:
            raise ProtocolError("driver is terminal after a previous worker failure")
        request = Request.parse(request.to_json())
        self._start()
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(encode_line(request.to_json()))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            return_code = self._process.poll()
            self._mark_terminal()
            raise ProcessExitedError(f"worker pipe closed (exit={return_code})") from exc
        try:
            raw_line = self._stdout.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._mark_terminal()
            raise RequestTimeoutError(f"{request.op} exceeded {self.timeout_seconds:.3f}s") from exc
        if raw_line is _EOF:
            return_code = self._process.poll()
            self._mark_terminal()
            raise ProcessExitedError(f"worker exited before response (exit={return_code})")
        try:
            response = validate_response(decode_line(str(raw_line)), request)
        except ProtocolError:
            self._mark_terminal()
            raise
        if not response["ok"]:
            error = response["error"]
            if request.op in {"reset", "add", "close"}:
                self._mark_terminal()
            raise RemoteAdapterError(
                error["code"], redact(error["message"], self._secrets), error["retryable"]
            )
        return response["result"]

    def reset(self) -> None:
        self._ready = False
        self._rpc(Request(request_id=str(uuid.uuid4()), op="reset"))
        self._ready = True

    def add(self, messages: Sequence[Mapping[str, str]], session: str, timestamp: str | None) -> int:
        parsed = tuple(Message.parse(message, index) for index, message in enumerate(messages))
        request = Request.parse(
            Request(
                request_id=str(uuid.uuid4()),
                op="add",
                messages=parsed,
                session=session,
                timestamp=timestamp,
            ).to_json()
        )
        self._ready = False
        result = self._rpc(request)
        self._ready = True
        return int(result["accepted"])

    def search(self, query: str, top_k: int, max_context_tokens: int) -> Mapping[str, Any]:
        return self._rpc(
            Request(
                request_id=str(uuid.uuid4()),
                op="search",
                query=query,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process is not None and self._ready:
                self._rpc(Request(request_id=str(uuid.uuid4()), op="close"))
                if self._process is not None:
                    try:
                        self._process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            self._closed = True
            self._stop()
