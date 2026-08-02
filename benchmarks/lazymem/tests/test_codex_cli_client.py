from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semafuse_lazymem_bench.config import (
    completion_client,
    completion_client_identity,
)
from semafuse_lazymem_bench.endpoints import (
    CodexCliClient,
    CodexCliError,
    OpenAICompatibleClient,
)


class CodexCliClientTests(unittest.TestCase):
    def _fake_codex(self, directory: Path) -> Path:
        command = directory / "fake_codex.py"
        command.write_text(
            """#!{python}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
assert args[0] == "exec", args
schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text(encoding="utf-8"))
output = Path(args[args.index("--output-last-message") + 1])
prompt = sys.stdin.read()
log_path = os.environ.get("FAKE_CODEX_LOG")
if log_path:
    Path(log_path).write_text(json.dumps({{"args": args, "schema": schema, "prompt": prompt}}), encoding="utf-8")
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
if mode == "extra-field":
    output.write_text('{{"response":"bad", "extra":true}}', encoding="utf-8")
elif mode == "missing-output":
    pass
elif mode == "failure":
    print("fake failure", file=sys.stderr)
    raise SystemExit(7)
elif mode == "oversized":
    output.write_text(json.dumps({{"response": "x" * 128}}), encoding="utf-8")
else:
    output.write_text(json.dumps({{"response": "local Codex result"}}), encoding="utf-8")
""".format(python=sys.executable),
            encoding="utf-8",
        )
        command.chmod(0o755)
        return command

    def test_runs_ephemeral_read_only_codex_exec_with_structured_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            command = self._fake_codex(directory)
            log_path = directory / "codex-log.json"
            with patch.dict(os.environ, {"FAKE_CODEX_LOG": str(log_path)}, clear=False):
                result = CodexCliClient(
                    command=str(command),
                    model="test-codex-model",
                    timeout_s=5,
                ).complete(
                    [
                        {"role": "system", "content": "Answer briefly."},
                        {"role": "user", "content": "What is two plus two?"},
                    ],
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=32,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
            invocation = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(result.text, "local Codex result")
        self.assertGreaterEqual(result.latency_s, 0.0)
        self.assertEqual(result.metadata["provider"], "codex-cli")
        self.assertEqual(result.metadata["model"], "test-codex-model")
        self.assertEqual(result.metadata["sandbox"], "read-only")
        self.assertIs(result.metadata["ephemeral"], True)
        self.assertIs(result.metadata["ignore_user_config"], True)
        self.assertIs(result.metadata["ignore_rules"], True)
        self.assertEqual(result.metadata["web_search"], "disabled")
        self.assertIn("shell_tool", result.metadata["disabled_features"])
        self.assertEqual(
            result.metadata["unsupported_request_options"],
            ["temperature", "top_p", "max_tokens", "extra_body"],
        )
        self.assertEqual(
            result.metadata["requested_options"],
            {
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 32,
                "extra_body_keys": ["chat_template_kwargs"],
            },
        )
        self.assertIn("--ephemeral", invocation["args"])
        self.assertIn("--strict-config", invocation["args"])
        self.assertEqual(
            invocation["args"][invocation["args"].index("--sandbox") + 1], "read-only"
        )
        self.assertIn("--skip-git-repo-check", invocation["args"])
        self.assertIn("--ignore-rules", invocation["args"])
        self.assertIn("--ignore-user-config", invocation["args"])
        self.assertIn("--json", invocation["args"])
        self.assertIn("shell_tool", invocation["args"])
        self.assertIn('approval_policy="never"', invocation["args"])
        self.assertIn('web_search="disabled"', invocation["args"])
        self.assertEqual(
            invocation["schema"],
            {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
                "additionalProperties": False,
            },
        )
        self.assertIn('"role":"system"', invocation["prompt"])
        self.assertIn("What is two plus two?", invocation["prompt"])

    def test_rejects_invalid_or_missing_final_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            command = self._fake_codex(Path(temp_dir))
            client = CodexCliClient(
                command=str(command), model="test-codex-model", timeout_s=5
            )
            with patch.dict(os.environ, {"FAKE_CODEX_MODE": "extra-field"}, clear=False):
                with self.assertRaisesRegex(CodexCliError, "exactly"):
                    client.complete([{"role": "user", "content": "hello"}])
            with patch.dict(os.environ, {"FAKE_CODEX_MODE": "missing-output"}, clear=False):
                with self.assertRaisesRegex(CodexCliError, "empty"):
                    client.complete([{"role": "user", "content": "hello"}])
            with patch.dict(os.environ, {"FAKE_CODEX_MODE": "failure"}, clear=False):
                with self.assertRaisesRegex(CodexCliError, "status 7") as failure:
                    client.complete([{"role": "user", "content": "hello"}])
                self.assertNotIn("fake failure", str(failure.exception))
            with patch.dict(os.environ, {"FAKE_CODEX_MODE": "oversized"}, clear=False):
                limited_client = CodexCliClient(
                    command=str(command),
                    model="test-codex-model",
                    timeout_s=5,
                    max_output_bytes=16,
                )
                with self.assertRaisesRegex(CodexCliError, "output limit"):
                    limited_client.complete([{"role": "user", "content": "hello"}])

    def test_factory_preserves_http_default_and_identifies_codex_without_secrets(self) -> None:
        http = completion_client(
            {"base_url": "http://offline.invalid/v1", "model": "offline", "retries": 0}
        )
        self.assertIsInstance(http, OpenAICompatibleClient)
        identity = completion_client_identity(
            {
                "backend": "codex-cli",
                "command": "codex",
                "model": "local-model",
                "timeout_s": 300,
                "retries": 0,
                "api_key_env": "DO_NOT_RECORD_ME",
            }
        )
        self.assertEqual(
            identity,
            {
                "backend": "codex_cli",
                "command": "codex",
                "model": "local-model",
                "sandbox": "read-only",
                "ephemeral": True,
                "ignore_user_config": True,
                "ignore_rules": True,
                "web_search": "disabled",
                "output_schema": "semafuse-lazymem-codex-cli/v1",
                "timeout_s": 300.0,
                "retries": 0,
                "max_output_bytes": 1_048_576,
            },
        )

    def test_factory_rejects_unknown_completion_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "completion backend"):
            completion_client({"backend": "unsupported"})

    def test_factory_requires_explicit_model_and_rejects_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 'model'"):
            completion_client({"backend": "codex_cli"})
        with self.assertRaisesRegex(ValueError, "does not support profile"):
            completion_client(
                {
                    "backend": "codex_cli",
                    "model": "local-model",
                    "profile": "unsafe-profile",
                }
            )


if __name__ == "__main__":
    unittest.main()
