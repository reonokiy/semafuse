"""Parallel query-conditioned memory construction."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

from .decisions import (
    WindowDecisions,
    deduplicate_decisions,
    parse_edit_decisions,
    render_selected_memory,
)
from .endpoints import ChatCompletionClient
from .types import Question
from .windows import EvidenceWindow


WINDOW_ANNOTATION_SYSTEM_PROMPT = """You are a local memory editing model.

You will receive a user query and one local chronological memory window.
Your task is to output a sequence of local editing decisions for this local window only, following the original message order.

Return exactly one valid JSON array.
Each item in the array must correspond to one non-empty role message in previous_bridge, core_messages, and next_bridge.
Each item in the array must contain exactly three fields: op, compressed_content, and reason.

The only allowed decision operations are KEEP and DROP.
For KEEP, retain dialogue content that is relevant to the query and useful for answering the query.
KEEP may be realized as verbatim keeping, key-span extraction, or concise compression depending on each message.
For DROP, compressed_content must be an empty string.

Use only the provided window content.
Do not output markdown, explanations, or any extra text.
Your output must be directly parseable by json.loads()."""


WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE = """# Task
Given the following query-conditioned memory window, produce a sequence of local editing decisions.

# Required Output
Return only a JSON array with the following structure:
[
  {
    "op": "KEEP | DROP",
    "compressed_content": "empty string if DROP; otherwise retained content in one KEEP style: verbatim keeping, key-span extraction, or concise compression",
    "reason": "brief local reason"
  }
]

# Local Editing Rules
1. The output array must contain one decision for every non-empty role message in previous_bridge, core_messages, and next_bridge.
2. The decisions must follow the original chronological order: previous_bridge, then core_messages, then next_bridge.
3. Each decision must contain exactly three fields: op, compressed_content, and reason.
4. op must be either KEEP or DROP.
5. KEEP means preserving dialogue content that is relevant to the query and useful for the main model to answer the question.
6. DROP means the message is locally irrelevant and should not enter the final memory.
7. For KEEP, choose the most appropriate retention style for each message:
   - verbatim keeping: keep the original message text when it is short and directly useful;
   - key-span extraction: keep only the query-relevant span when only part of the message is relevant;
   - concise compression: faithfully rewrite the message into a shorter form when it is useful but too long, or only partially relevant.
8. For any KEEP style, remain faithful to the source meaning and do not introduce unsupported facts.
9. If the whole window is irrelevant, still output one decision for every non-empty role message, and set every op to DROP.
10. When op is DROP, compressed_content must be an empty string.
11. If a message may contain answer evidence, user facts, temporal updates, or content related to the query entity/attribute, prefer KEEP even when relevance is uncertain.
12. Return only the JSON array. Do not output markdown, comments, or extra text.
13. The output must be directly parseable by json.loads().

# Query
Query Time: {query_time}
Query: {query}

# History Window
{window}"""


_USER_PROMPT_PLACEHOLDER = re.compile(r"\{(query_time|query|window)\}")


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    memory_text: str
    decisions: tuple[WindowDecisions, ...]
    latency_s: float
    request_latency_s: tuple[float, ...]
    prompt_tokens: int | None
    completion_tokens: int | None


def model_window_payload(window: EvidenceWindow) -> dict[str, list[dict[str, object]]]:
    """Render the three Appendix G.1 window sections and message metadata."""

    assert window.previous_bridge is not None
    assert window.core_messages is not None
    assert window.next_bridge is not None
    retrieval_ranks = window.retrieval_rank_by_message_id
    payload: dict[str, list[dict[str, object]]] = {}
    for section_name, position, messages in (
        ("previous_bridge", "previous_bridge", window.previous_bridge),
        ("core_messages", "core", window.core_messages),
        ("next_bridge", "next_bridge", window.next_bridge),
    ):
        payload[section_name] = [
            {
                "position": position,
                "role": message.role,
                "timestamp": message.timestamp,
                "global_message_index": message.global_message_index,
                "retrieval_rank": retrieval_ranks.get(message.id),
                "content": message.content,
            }
            for message in messages
            if message.role.strip() and message.content.strip()
        ]
    return payload


def render_window_annotation_user_prompt(
    question: Question, window: EvidenceWindow
) -> str:
    replacements = {
        "query_time": question.query_time,
        "query": question.query,
        "window": json.dumps(
            model_window_payload(window), ensure_ascii=False, indent=2
        ),
    }
    return _USER_PROMPT_PLACEHOLDER.sub(
        lambda match: replacements[match.group(1)],
        WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE,
    )


def window_annotation_messages(
    question: Question, window: EvidenceWindow
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WINDOW_ANNOTATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_window_annotation_user_prompt(question, window),
        },
    ]


class ConcurrentMemoryProcessor:
    """Issue independent window decisions concurrently, preserving window order."""

    SYSTEM_PROMPT = WINDOW_ANNOTATION_SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = WINDOW_ANNOTATION_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        max_workers: int = 64,
        max_tokens: int = 3072,
        temperature: float = 0.1,
        top_p: float = 0.95,
        enable_thinking: bool = True,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        if max_workers <= 0 or max_tokens <= 0:
            raise ValueError("MPM worker and token limits must be positive")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("MPM temperature must be between 0 and 2")
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1
        ):
            raise ValueError("MPM top_p must be in (0, 1]")
        if not isinstance(enable_thinking, bool):
            raise TypeError("MPM enable_thinking must be a boolean")
        self.client = client
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        configured_extra_body = dict(extra_body or {})
        raw_chat_template_kwargs = configured_extra_body.get(
            "chat_template_kwargs", {}
        )
        if not isinstance(raw_chat_template_kwargs, Mapping):
            raise TypeError("MPM chat_template_kwargs must be an object")
        chat_template_kwargs = dict(raw_chat_template_kwargs)
        chat_template_kwargs.setdefault("enable_thinking", enable_thinking)
        configured_extra_body["chat_template_kwargs"] = chat_template_kwargs
        self.extra_body = configured_extra_body

    def _process_one(
        self, question: Question, window: EvidenceWindow
    ) -> tuple[WindowDecisions, float, int | None, int | None]:
        response = self.client.complete(
            window_annotation_messages(question, window),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        return (
            parse_edit_decisions(response.text, window),
            response.latency_s,
            response.prompt_tokens,
            response.completion_tokens,
        )

    def process(
        self, question: Question, windows: Sequence[EvidenceWindow]
    ) -> ProcessingResult:
        if not windows:
            return ProcessingResult("", (), 0.0, (), 0, 0)
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(windows)),
            thread_name_prefix="lazymem-mpm",
        ) as pool:
            futures = [pool.submit(self._process_one, question, window) for window in windows]
            values = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        decisions = tuple(value[0] for value in values)
        selected = deduplicate_decisions(decisions)
        prompt_values = [value[2] for value in values]
        completion_values = [value[3] for value in values]
        return ProcessingResult(
            memory_text=render_selected_memory(selected),
            decisions=decisions,
            latency_s=elapsed,
            request_latency_s=tuple(value[1] for value in values),
            prompt_tokens=(
                sum(value for value in prompt_values if value is not None)
                if all(value is not None for value in prompt_values)
                else None
            ),
            completion_tokens=(
                sum(value for value in completion_values if value is not None)
                if all(value is not None for value in completion_values)
                else None
            ),
        )
