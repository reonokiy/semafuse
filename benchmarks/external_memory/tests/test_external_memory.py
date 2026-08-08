from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from semafuse_external_memory.adapter import MemoryItem, TokenCounter, enforce_limits
from semafuse_external_memory.adapters.amem import AMemAdapter
from semafuse_external_memory.adapters.mem0_oss import Mem0OssAdapter
from semafuse_external_memory.config import WorkerConfig, redact
from semafuse_external_memory.driver import ExternalMemoryDriver
from semafuse_external_memory.errors import (
    AdapterUnavailableError,
    ConfigurationError,
    ProcessExitedError,
    ProtocolError,
    RemoteAdapterError,
    RequestTimeoutError,
)
from semafuse_external_memory.protocol import PROTOCOL_VERSION, Message, Request, validate_response


class ExternalMemoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def config(self, adapter_config: dict | None = None) -> Path:
        path = Path(self.temp.name) / f"config-{len(list(Path(self.temp.name).iterdir()))}.json"
        path.write_text(
            json.dumps(
                {
                    "adapter": {"kind": "fake", "config": adapter_config or {}},
                    "tokenizer": {"kind": "whitespace"},
                    "run_policy": {"track": "contract-test"},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_fake_enforces_label_isolation_top_k_and_token_cap(self) -> None:
        with ExternalMemoryDriver(self.config(), timeout_seconds=2) as driver:
            driver.reset()
            driver.add(
                [
                    {"role": "user", "content": "alpha beta gamma", "source_id": "gold-looking-z"},
                    {"role": "assistant", "content": "alpha delta", "source_id": "ordinary-a"},
                    {"role": "user", "content": "unrelated", "source_id": "gold-looking-a"},
                ],
                "session-1",
                "2026-01-01T00:00:00Z",
            )
            top_one = driver.search("alpha", top_k=1, max_context_tokens=100)
            self.assertEqual([item["source_ids"] for item in top_one["items"]], [["gold-looking-z"]])
            capped = driver.search("alpha", top_k=3, max_context_tokens=4)
            self.assertEqual(len(capped["items"]), 1)
            self.assertEqual(capped["context_tokens"], 3)
            self.assertTrue(capped["truncated"])

            driver.reset()
            self.assertEqual(driver.search("alpha", 10, 100)["items"], [])

        leaked = {
            "protocol": PROTOCOL_VERSION,
            "request_id": "r",
            "op": "search",
            "query": "alpha",
            "top_k": 1,
            "max_context_tokens": 10,
            "gold_answer": "forbidden",
        }
        with self.assertRaises(ProtocolError):
            Request.parse(leaked)

    def test_remote_error_is_structured(self) -> None:
        with ExternalMemoryDriver(self.config({"fail_on": "search"}), timeout_seconds=2) as driver:
            driver.reset()
            with self.assertRaises(RemoteAdapterError) as caught:
                driver.search("x", 1, 10)
            self.assertEqual(caught.exception.code, "ADAPTER_ERROR")

    def test_failed_mutation_and_reset_make_driver_terminal(self) -> None:
        with ExternalMemoryDriver(self.config({"fail_on": "add"}), timeout_seconds=2) as driver:
            driver.reset()
            with self.assertRaises(RemoteAdapterError):
                driver.add(
                    [{"role": "user", "content": "partial write", "source_id": "s1"}],
                    "session",
                    None,
                )
            with self.assertRaises(ProtocolError):
                driver.search("partial", 1, 10)
            with self.assertRaises(ProtocolError):
                driver.reset()

        with ExternalMemoryDriver(self.config({"fail_on": "reset"}), timeout_seconds=2) as driver:
            with self.assertRaises(RemoteAdapterError):
                driver.reset()
            with self.assertRaises(ProtocolError):
                driver.reset()

    def test_process_crash_fails_closed(self) -> None:
        with ExternalMemoryDriver(self.config({"crash_on": "search"}), timeout_seconds=2) as driver:
            driver.reset()
            with self.assertRaises(ProcessExitedError):
                driver.search("x", 1, 10)
            with self.assertRaises(ProtocolError):
                driver.reset()

    def test_process_timeout_is_killed(self) -> None:
        with ExternalMemoryDriver(
            self.config({"hang_on": "search", "hang_seconds": 5}), timeout_seconds=0.1
        ) as driver:
            driver.reset()
            with self.assertRaises(RequestTimeoutError):
                driver.search("x", 1, 10)

    def test_unavailable_upstreams_do_not_fallback(self) -> None:
        real_import = __import__

        def reject(name: str, *args: object, **kwargs: object):
            if name == "agentic_memory.memory_system" or name == "mem0":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"TEST_API_KEY": "not-a-real-secret"}):
            amem = AMemAdapter(
                {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_revision": "1" * 40,
                    "llm_backend": "openai",
                    "llm_model": "test/model",
                    "api_key_env": "TEST_API_KEY",
                    "openai_base_url": "https://openrouter.ai/api/v1",
                }
            )
            mem0 = Mem0OssAdapter(
                {
                    "memory_config": {},
                    "scope_id": "test",
                    "infer": False,
                }
            )
        with patch("builtins.__import__", side_effect=reject):
            with self.assertRaises(AdapterUnavailableError):
                amem.reset()
            with self.assertRaises(AdapterUnavailableError):
                mem0.reset()

    def test_non_json_worker_output_is_protocol_failure(self) -> None:
        command = [sys.executable, "-c", "print('not json', flush=True); input()"]
        with ExternalMemoryDriver(self.config(), timeout_seconds=2, command=command) as driver:
            with self.assertRaises(ProtocolError):
                driver.reset()

    def test_stderr_and_api_secrets_are_never_surfaced(self) -> None:
        secret = "super-secret-api-value"
        command = [
            sys.executable,
            "-c",
            f"import sys; input(); print({secret!r}, file=sys.stderr, flush=True); raise SystemExit(9)",
        ]
        with ExternalMemoryDriver(self.config(), timeout_seconds=2, command=command) as driver:
            with self.assertRaises(ProcessExitedError) as caught:
                driver.reset()
            self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(redact(f"provider failed with {secret}", (secret,)), "provider failed with [REDACTED]")

        error_script = (
            "import json,sys; r=json.loads(input()); "
            "print(json.dumps({'protocol':r['protocol'],'request_id':r['request_id'],"
            "'op':r['op'],'ok':False,'error':{'code':'API_ERROR',"
            "'message':'provider exposed '+sys.argv[1],'retryable':False}}),flush=True)"
        )
        with ExternalMemoryDriver(
            self.config(),
            timeout_seconds=2,
            command=[sys.executable, "-c", error_script, secret],
            env={"OPENROUTER_API_KEY": secret},
        ) as driver:
            with self.assertRaises(RemoteAdapterError) as caught:
                driver.reset()
            self.assertNotIn(secret, str(caught.exception))
            self.assertIn("[REDACTED]", str(caught.exception))

    def test_mem0_ingests_per_message_with_exact_provenance(self) -> None:
        class MemoryStub:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def add(self, messages: list[dict], **kwargs: object) -> dict:
                self.calls.append({"messages": messages, **kwargs})
                return {"results": [{"id": f"memory-{len(self.calls)}", "event": "ADD"}]}

        adapter = Mem0OssAdapter({"memory_config": {}, "scope_id": "history", "infer": False})
        memory = MemoryStub()
        adapter._memory = memory
        accepted = adapter.add(
            [
                Message("user", "first", "source-1"),
                Message("assistant", "second", "source-2"),
            ],
            "session",
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(accepted, 2)
        self.assertEqual(len(memory.calls), 2)
        self.assertEqual(memory.calls[0]["metadata"]["semafuse_source_ids"], ["source-1"])
        self.assertEqual(memory.calls[1]["metadata"]["semafuse_source_ids"], ["source-2"])
        self.assertEqual(adapter._provenance["memory-1"][0], ("source-1",))
        self.assertEqual(adapter._provenance["memory-2"][0], ("source-2",))

    def test_mem0_inference_requires_explicit_conservative_provenance(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "origin_only"):
            Mem0OssAdapter({"memory_config": {}, "scope_id": "history", "infer": True})
        adapter = Mem0OssAdapter(
            {
                "memory_config": {},
                "scope_id": "history",
                "infer": True,
                "provenance_policy": "origin_only",
            }
        )
        self.assertEqual(adapter._provenance_policy, "origin_only")

    def test_mem0_search_fails_closed_on_untracked_or_conflicting_provenance(self) -> None:
        class MemoryStub:
            result: dict

            def search(self, *_: object, **__: object) -> dict:
                return {"results": [self.result]}

        adapter = Mem0OssAdapter({"memory_config": {}, "scope_id": "history", "infer": False})
        memory = MemoryStub()
        memory.result = {"id": "m1", "memory": "fact", "score": 0.5, "metadata": {}}
        adapter._memory = memory
        with self.assertRaisesRegex(RuntimeError, "untracked memory"):
            adapter.search("fact", 1)

        adapter._provenance["m1"] = (("source-1",), "session", None)
        memory.result["metadata"] = {
            "semafuse_source_ids": ["different-source"],
            "semafuse_session": "session",
            "semafuse_timestamp": "",
        }
        with self.assertRaisesRegex(RuntimeError, "disagrees with tracked source"):
            adapter.search("fact", 1)

    def test_amem_requires_and_scopes_explicit_openai_base_url(self) -> None:
        base = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": "1" * 40,
            "llm_backend": "openai",
            "llm_model": "test/model",
            "api_key_env": "TEST_API_KEY",
        }
        with self.assertRaisesRegex(ConfigurationError, "model_revision"):
            AMemAdapter({key: value for key, value in base.items() if key != "model_revision"})

        with patch.dict(os.environ, {"TEST_API_KEY": "not-a-real-secret"}):
            with self.assertRaisesRegex(ConfigurationError, "openai_base_url"):
                AMemAdapter(base)
            adapter = AMemAdapter(
                {**base, "openai_base_url": "https://openrouter.ai/api/v1"}
            )

        observed: list[tuple[str | None, str | None]] = []

        class SystemStub:
            def __init__(self, **kwargs: object) -> None:
                observed.append((os.environ.get("OPENAI_BASE_URL"), str(kwargs.get("model_name"))))

        module = types.ModuleType("agentic_memory.memory_system")
        module.AgenticMemorySystem = SystemStub
        hub_module = types.ModuleType("huggingface_hub")
        hub_module.snapshot_download = lambda **_: "/tmp/exact-minilm-snapshot"
        with patch.dict(
            sys.modules,
            {"agentic_memory.memory_system": module, "huggingface_hub": hub_module},
        ):
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://original.example/v1"}):
                adapter.reset()
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://original.example/v1")
        self.assertEqual(
            observed,
            [("https://openrouter.ai/api/v1", "/tmp/exact-minilm-snapshot")],
        )
        adapter.close()

    def test_amem_fails_closed_on_unknown_or_conflicting_provenance(self) -> None:
        class MemoryStub:
            def add_note(self, *_: object, **__: object) -> str:
                return "same-id"

            def search_agentic(self, *_: object, **__: object) -> list[dict[str, object]]:
                return [{"id": "unknown-id", "content": "fact", "score": 0.5}]

        adapter = object.__new__(AMemAdapter)
        adapter._memory = MemoryStub()
        adapter._provenance = {}
        with self.assertRaisesRegex(RuntimeError, "untracked memory"):
            adapter.search("fact", 1)

        first = Message(role="user", content="one", source_id="source-1")
        second = Message(role="user", content="two", source_id="source-2")
        adapter.add([first], "session", None)
        with self.assertRaisesRegex(RuntimeError, "multiple sources"):
            adapter.add([second], "session", None)

    def test_amem_fails_closed_when_upstream_swallows_internal_errors(self) -> None:
        class SystemStub:
            def __init__(self, **_: object) -> None:
                pass

            def add_note(self, *_: object, **__: object) -> str:
                logging.getLogger("agentic_memory.memory_system").error(
                    "provider error text that must not be retained"
                )
                return "stored-despite-error"

            def search_agentic(self, *_: object, **__: object) -> list[dict[str, object]]:
                logging.getLogger("agentic_memory.memory_system").error(
                    "search error text that must not be retained"
                )
                return []

        module = types.ModuleType("agentic_memory.memory_system")
        module.AgenticMemorySystem = SystemStub
        hub_module = types.ModuleType("huggingface_hub")
        hub_module.snapshot_download = lambda **_: "/tmp/exact-minilm-snapshot"
        with patch.dict(os.environ, {"TEST_API_KEY": "not-a-real-secret"}):
            adapter = AMemAdapter(
                {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_revision": "1" * 40,
                    "llm_backend": "openai",
                    "llm_model": "test/model",
                    "api_key_env": "TEST_API_KEY",
                    "openai_base_url": "https://openrouter.ai/api/v1",
                }
            )
        with patch.dict(
            sys.modules,
            {"agentic_memory.memory_system": module, "huggingface_hub": hub_module},
        ):
            adapter.reset()
        with self.assertRaisesRegex(RuntimeError, "internal error while processing"):
            adapter.add([Message("user", "fact", "source-1")], "session", None)

        adapter._memory = SystemStub()
        with self.assertRaisesRegex(RuntimeError, "internal error during search"):
            adapter.search("fact", 1)
        self.assertEqual(adapter._upstream_error_counter.count, 2)
        adapter.close()
        self.assertIsNone(adapter._upstream_error_counter)

    def test_amem_retries_and_validates_structured_llm_responses(self) -> None:
        class CompletionStub:
            def __init__(self) -> None:
                self.calls = 0

            def get_completion(self, *_: object, **__: object) -> str:
                self.calls += 1
                return "not-json" if self.calls == 1 else '{"value":"ok"}'

        class SystemStub:
            def __init__(self, **_: object) -> None:
                self.llm_controller = types.SimpleNamespace(llm=CompletionStub())

        module = types.ModuleType("agentic_memory.memory_system")
        module.AgenticMemorySystem = SystemStub
        hub_module = types.ModuleType("huggingface_hub")
        hub_module.snapshot_download = lambda **_: "/tmp/exact-minilm-snapshot"
        config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_revision": "1" * 40,
            "llm_backend": "openai",
            "llm_model": "test/model",
            "llm_max_attempts": 3,
            "api_key_env": "TEST_API_KEY",
            "openai_base_url": "https://openrouter.ai/api/v1",
        }
        with patch.dict(os.environ, {"TEST_API_KEY": "not-a-real-secret"}):
            adapter = AMemAdapter(config)
            with self.assertRaisesRegex(ConfigurationError, "llm_max_attempts"):
                AMemAdapter({**config, "llm_max_attempts": 0})
        with patch.dict(
            sys.modules,
            {"agentic_memory.memory_system": module, "huggingface_hub": hub_module},
        ):
            with patch("semafuse_external_memory.adapters.amem.time.sleep"):
                adapter.reset()
                response = adapter._memory.llm_controller.llm.get_completion(
                    "prompt",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "schema": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                                "additionalProperties": False,
                            }
                        },
                    },
                )
        self.assertEqual(json.loads(response), {"value": "ok"})
        self.assertEqual(adapter._memory.llm_controller.llm.calls, 2)
        self.assertEqual(adapter._llm_retry_count, 1)
        adapter.close()

    def test_joined_context_is_counted_exactly(self) -> None:
        class NonAdditiveCounter(TokenCounter):
            def count(self, text: str) -> int:
                return len(text.replace("\n\n", "#"))

        result = enforce_limits(
            [MemoryItem("1", "a"), MemoryItem("2", "b")],
            top_k=2,
            max_context_tokens=2,
            counter=NonAdditiveCounter(),
        )
        self.assertEqual(result["context"], "a")
        self.assertEqual(result["context_tokens"], 1)
        self.assertTrue(result["truncated"])

    def test_response_is_checked_against_requested_caps_and_context(self) -> None:
        request = Request(
            request_id="r",
            op="search",
            query="q",
            top_k=1,
            max_context_tokens=2,
        )

        def response(items: list[dict], context: str, context_tokens: int) -> dict:
            return {
                "protocol": PROTOCOL_VERSION,
                "request_id": "r",
                "op": "search",
                "ok": True,
                "result": {
                    "items": items,
                    "context": context,
                    "context_tokens": context_tokens,
                    "truncated": False,
                },
            }

        item = {
            "id": "1",
            "text": "one",
            "score": 1.0,
            "source_ids": ["s1"],
            "session": None,
            "timestamp": None,
            "token_count": 1,
        }
        with self.assertRaises(ProtocolError):
            validate_response(response([item, {**item, "id": "2"}], "one\n\none", 2), request)
        with self.assertRaises(ProtocolError):
            validate_response(response([item], "one", 3), request)
        with self.assertRaises(ProtocolError):
            validate_response(response([item], "different", 1), request)

    def test_protected_tracks_require_model_identity_and_complete_environment_lock(self) -> None:
        path = Path(self.temp.name) / "protected.json"
        raw = {
            "adapter": {
                "kind": "mem0_oss",
                "config": {
                    "memory_config": {
                        "llm": {"provider": "openai", "config": {"model": "test/llm"}},
                        "embedder": {
                            "provider": "openai",
                            "config": {"model": "test/embedding"},
                        },
                    },
                    "scope_id": "h",
                    "infer": False,
                },
            },
            "tokenizer": {
                "kind": "huggingface",
                "model": "test/tokenizer",
                "revision": "1" * 40,
            },
            "run_policy": {"track": "common-policy-v1"},
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            WorkerConfig.load(path)

        sha = "1" * 40
        raw["run_policy"] = {
            "track": "common-policy-v1",
            "upstream": {
                "revision": "4debc58a83377b18be81ae1e5969a300736b2fac",
                "lock_status": "verified",
            },
            "models": {
                "memory_llm": {
                    "provider": "openai",
                    "model": "test/llm",
                    "revision": sha,
                    "lock_status": "verified",
                },
                "embedding": {
                    "provider": "openai",
                    "model": "test/embedding",
                    "revision": sha,
                    "lock_status": "verified",
                },
                "reader_tokenizer": {
                    "provider": "huggingface",
                    "model": "test/tokenizer",
                    "revision": sha,
                    "lock_status": "verified",
                },
            },
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "transitive dependency lock"):
            WorkerConfig.load(path)

        raw["run_policy"]["models"]["reader_tokenizer"]["revision"] = "3" * 40
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "reader_tokenizer revision"):
            WorkerConfig.load(path)

        raw["run_policy"]["models"]["reader_tokenizer"]["revision"] = sha
        raw["run_policy"]["models"]["memory_llm"].pop("provider")
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "provider, model, revision"):
            WorkerConfig.load(path)


if __name__ == "__main__":
    unittest.main()
