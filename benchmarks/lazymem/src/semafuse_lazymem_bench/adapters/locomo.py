"""Adapter for the paper-v2 LoCoMo evaluation subset."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..types import Message, Question


PAPER_CONVERSATIONS = frozenset({"conv-49", "conv-50"})
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}
EXPECTED_PAPER_QUESTION_COUNT = 314
_SESSION_RE = re.compile(r"^session_(\d+)$")
_EVIDENCE_RE = re.compile(r"D\d+:\d+")


class LoCoMoLicenseWarning(UserWarning):
    """LoCoMo is CC BY-NC 4.0 and cannot be assumed commercial-safe."""


def _session_keys(conversation: Mapping[str, Any]) -> list[str]:
    keys: list[tuple[int, str]] = []
    for key, value in conversation.items():
        match = _SESSION_RE.match(key)
        if match and isinstance(value, list):
            keys.append((int(match.group(1)), key))
    return [key for _, key in sorted(keys)]


def _evidence_ids(value: Any, message_ids: set[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("LoCoMo evidence must be a list")
    evidence: list[str] = []
    seen: set[str] = set()
    for item in value:
        matches = _EVIDENCE_RE.findall(str(item))
        if not matches and str(item).strip():
            raise ValueError(f"unrecognized LoCoMo evidence reference: {item!r}")
        for reference in matches:
            resolved = reference
            if resolved not in message_ids:
                day, turn = reference[1:].split(":", 1)
                canonical = f"D{int(day)}:{int(turn)}"
                if canonical in message_ids:
                    resolved = canonical
            if resolved not in seen:
                seen.add(resolved)
                evidence.append(resolved)
    return tuple(evidence)


def adapt_locomo(
    samples: Iterable[Mapping[str, Any]],
    *,
    acknowledge_noncommercial: bool = False,
    require_paper_count: bool = True,
) -> list[Question]:
    """Normalize only conv-49/conv-50 and QA categories 1--4.

    LoCoMo's repository applies CC BY-NC 4.0 to the released material. The
    caller must explicitly acknowledge that restriction; this is a guardrail,
    not legal advice or a license checker.
    """

    warnings.warn(
        "LoCoMo is distributed under CC BY-NC 4.0; verify that this run and "
        "all retained artifacts are non-commercial and satisfy attribution.",
        LoCoMoLicenseWarning,
        stacklevel=2,
    )
    if not acknowledge_noncommercial:
        raise PermissionError(
            "LoCoMo adaptation requires acknowledge_noncommercial=True"
        )

    questions: list[Question] = []
    found_conversations: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if sample_id not in PAPER_CONVERSATIONS:
            continue
        found_conversations.add(sample_id)
        conversation = sample.get("conversation")
        if not isinstance(conversation, Mapping):
            raise TypeError(f"LoCoMo {sample_id} conversation must be an object")

        messages: list[Message] = []
        message_ids: set[str] = set()
        speaker_a = str(conversation.get("speaker_a", "") or "")
        speaker_b = str(conversation.get("speaker_b", "") or "")
        session_keys = _session_keys(conversation)
        for session_key in session_keys:
            timestamp = str(conversation.get(f"{session_key}_date_time", ""))
            for turn_index, turn in enumerate(conversation[session_key]):
                if not isinstance(turn, Mapping):
                    raise TypeError(f"LoCoMo {sample_id} {session_key} turn is not an object")
                message_id = str(turn.get("dia_id", f"{session_key}:{turn_index}"))
                if message_id in message_ids:
                    raise ValueError(f"duplicate LoCoMo dialog id {message_id} in {sample_id}")
                message_ids.add(message_id)
                text = turn.get("text", "")
                if not isinstance(text, str):
                    raise TypeError(f"LoCoMo message {message_id} text must be a string")
                caption = turn.get("blip_caption")
                if caption is not None and not isinstance(caption, str):
                    raise TypeError(
                        f"LoCoMo message {message_id} blip_caption must be a string"
                    )
                speaker = str(turn.get("speaker", "") or "")
                content_parts = [f"{speaker}: {text}" if speaker else text]
                if caption and caption.strip():
                    content_parts.append(f"Image caption: {caption.strip()}")
                image_query = turn.get("query")
                if image_query is not None and not isinstance(image_query, str):
                    raise TypeError(
                        f"LoCoMo message {message_id} image query must be a string"
                    )
                if image_query and image_query.strip():
                    content_parts.append(f"Image query: {image_query.strip()}")
                content = "\n".join(part for part in content_parts if part)
                messages.append(
                    Message(
                        id=message_id,
                        index=len(messages),
                        role="assistant" if speaker and speaker == speaker_b else "user",
                        content=content,
                        timestamp=timestamp,
                        session_id=session_key,
                        metadata={
                            "global_message_index": len(messages),
                            "retrieval_text": content,
                            "img_url": turn.get("img_url"),
                            "blip_caption": caption,
                            "image_query": image_query,
                            "speaker": speaker,
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                            "original_text": text,
                        },
                    )
                )

        qa_items = sample.get("qa")
        if not isinstance(qa_items, list):
            raise TypeError(f"LoCoMo {sample_id} qa must be a list")
        query_time = (
            str(conversation.get(f"{session_keys[-1]}_date_time", ""))
            if session_keys
            else ""
        )
        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, Mapping):
                raise TypeError(f"LoCoMo {sample_id} QA item must be an object")
            category = int(qa.get("category", 0))
            if category not in CATEGORY_NAMES:
                continue
            evidence = _evidence_ids(qa.get("evidence", []), message_ids)
            missing = set(evidence).difference(message_ids)
            if missing:
                raise ValueError(
                    f"LoCoMo {sample_id} QA {qa_index} references missing evidence {missing}"
                )
            questions.append(
                Question(
                    id=f"{sample_id}:q{qa_index:04d}",
                    dataset="locomo",
                    question_type=CATEGORY_NAMES[category],
                    query=str(qa["question"]),
                    answer=(
                        qa["answer"]
                        if isinstance(qa["answer"], str)
                        else json.dumps(qa["answer"], ensure_ascii=False, sort_keys=True)
                    ),
                    query_time=query_time,
                    messages=tuple(messages),
                    gold_message_ids=evidence,
                    gold_session_ids=tuple(
                        sorted(
                            {
                                message.session_id
                                for message in messages
                                if message.id in evidence
                            }
                        )
                    ),
                    metadata={
                        "conversation_id": sample_id,
                        "category": category,
                        "license": "CC-BY-NC-4.0",
                        "raw_evidence": list(qa.get("evidence", [])),
                    },
                )
            )

    if found_conversations != PAPER_CONVERSATIONS:
        missing = PAPER_CONVERSATIONS.difference(found_conversations)
        raise ValueError(f"LoCoMo paper subset is missing conversations: {sorted(missing)}")
    if require_paper_count and len(questions) != EXPECTED_PAPER_QUESTION_COUNT:
        raise ValueError(
            "LoCoMo paper subset must contain exactly "
            f"{EXPECTED_PAPER_QUESTION_COUNT} category-1--4 questions; got {len(questions)}"
        )
    return questions


def load_locomo(
    path: str | Path,
    *,
    acknowledge_noncommercial: bool = False,
    require_paper_count: bool = True,
) -> list[Question]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise TypeError("LoCoMo root must be a JSON array")
    return adapt_locomo(
        value,
        acknowledge_noncommercial=acknowledge_noncommercial,
        require_paper_count=require_paper_count,
    )
