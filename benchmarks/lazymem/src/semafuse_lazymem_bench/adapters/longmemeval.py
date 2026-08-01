"""Adapter for the official LongMemEval JSON release."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..types import Message, Question


QUESTION_TYPES = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    }
)


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def adapt_longmemeval(items: Iterable[Mapping[str, Any]]) -> list[Question]:
    """Normalize LongMemEval records without mutating or downloading data."""

    questions: list[Question] = []
    seen_question_ids: set[str] = set()
    for raw in items:
        question_id = str(raw["question_id"])
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate LongMemEval question_id: {question_id}")
        seen_question_ids.add(question_id)

        question_type = str(raw["question_type"])
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"unknown LongMemEval question_type {question_type!r} for {question_id}"
            )

        session_ids = raw["haystack_session_ids"]
        dates = raw["haystack_dates"]
        sessions = raw["haystack_sessions"]
        if not isinstance(session_ids, list) or not isinstance(dates, list) or not isinstance(sessions, list):
            raise TypeError(f"history fields for {question_id} must be lists")
        if not (len(session_ids) == len(dates) == len(sessions)):
            raise ValueError(f"history list lengths differ for {question_id}")

        messages: list[Message] = []
        gold_message_ids: list[str] = []
        empty_message_ids: list[str] = []
        native_session_ids = [str(value) for value in session_ids]
        session_counts = Counter(native_session_ids)
        session_occurrences: defaultdict[str, int] = defaultdict(int)
        normalized_sessions_by_native: defaultdict[str, list[str]] = defaultdict(list)
        source_global_message_index = 0
        for session_position, (session_id_value, timestamp_value, turns) in enumerate(
            zip(session_ids, dates, sessions, strict=True)
        ):
            native_session_id = str(session_id_value)
            occurrence = session_occurrences[native_session_id]
            session_occurrences[native_session_id] += 1
            session_id = (
                native_session_id
                if session_counts[native_session_id] == 1
                else f"{native_session_id}#occurrence-{occurrence}"
            )
            normalized_sessions_by_native[native_session_id].append(session_id)
            timestamp = str(timestamp_value)
            if not isinstance(turns, list):
                raise TypeError(f"session {session_id} in {question_id} is not a list")
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, Mapping):
                    raise TypeError(f"turn {turn_index} in {session_id} is not an object")
                global_message_index = source_global_message_index
                source_global_message_index += 1
                message_id = f"{session_id}_{turn_index + 1:02d}"
                role = turn.get("role")
                content = turn.get("content")
                if not isinstance(role, str) or not role.strip():
                    raise TypeError(f"message {message_id} in {question_id} has no string role")
                if not isinstance(content, str):
                    raise TypeError(
                        f"message {message_id} in {question_id} content must be a string"
                    )
                has_answer = turn.get("has_answer") is True
                if not content.strip():
                    if has_answer:
                        raise ValueError(
                            f"gold message {message_id} in {question_id} is empty"
                        )
                    empty_message_ids.append(message_id)
                message = Message(
                    id=message_id,
                    index=len(messages),
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    session_id=session_id,
                    has_answer=has_answer,
                    metadata={
                        "global_message_index": global_message_index,
                        "retrieval_text": (
                            f"User: {content}"
                            if role == "user"
                            else f"Assistant: {content}"
                            if role == "assistant"
                            else f"{role.title()}: {content}"
                        ),
                        "native_session_id": native_session_id,
                        "session_position": session_position,
                    },
                )
                messages.append(message)
                if has_answer:
                    gold_message_ids.append(message_id)

        gold_session_ids = tuple(
            normalized_session_id
            for value in raw.get("answer_session_ids", [])
            for normalized_session_id in normalized_sessions_by_native[str(value)]
        )
        reasoning_chain = raw.get(
            "referenced_reasoning_chain", raw.get("reasoning_chain", "")
        )
        if not isinstance(reasoning_chain, str):
            reasoning_chain = json.dumps(
                reasoning_chain, ensure_ascii=False, sort_keys=True
            )
        questions.append(
            Question(
                id=question_id,
                dataset="longmemeval",
                question_type=question_type,
                query=str(raw["question"]),
                answer=_answer_text(raw["answer"]),
                query_time=str(raw.get("question_date", "")),
                messages=tuple(messages),
                gold_message_ids=tuple(gold_message_ids),
                gold_session_ids=gold_session_ids,
                metadata={
                    "is_abstention": question_id.endswith("_abs"),
                    "referenced_reasoning_chain": reasoning_chain,
                    "empty_message_ids_present": empty_message_ids,
                },
            )
        )
    return questions


def load_longmemeval(path: str | Path) -> list[Question]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise TypeError("LongMemEval root must be a JSON array")
    return adapt_longmemeval(value)
