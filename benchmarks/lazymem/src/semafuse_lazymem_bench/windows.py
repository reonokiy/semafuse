"""History expansion, merging, and overlapping sub-window construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .retrieval import RankedMessage
from .types import Message


MergeRule = Literal["paper", "code"]


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    id: str
    order: int
    start: int
    stop: int
    messages: tuple[Message, ...]
    retrieved_message_ids: tuple[str, ...]
    retrieval_ranks: tuple[tuple[str, int], ...] = ()
    previous_bridge: tuple[Message, ...] | None = None
    core_messages: tuple[Message, ...] | None = None
    next_bridge: tuple[Message, ...] | None = None

    def __post_init__(self) -> None:
        if not self.messages or self.stop - self.start != len(self.messages):
            raise ValueError("evidence window bounds do not match its messages")
        if len(self.retrieved_message_ids) != len(set(self.retrieved_message_ids)):
            raise ValueError("evidence window contains duplicate retrieved message ids")

        sections = (self.previous_bridge, self.core_messages, self.next_bridge)
        if all(section is None for section in sections):
            positions = {
                message.id: position for position, message in enumerate(self.messages)
            }
            missing = set(self.retrieved_message_ids).difference(positions)
            if missing:
                raise ValueError(
                    f"retrieved messages are absent from evidence window: {sorted(missing)}"
                )
            if self.retrieved_message_ids:
                retrieved_positions = [
                    positions[message_id] for message_id in self.retrieved_message_ids
                ]
                core_start = min(retrieved_positions)
                core_stop = max(retrieved_positions) + 1
            else:
                core_start = 0
                core_stop = len(self.messages)
            object.__setattr__(self, "previous_bridge", self.messages[:core_start])
            object.__setattr__(self, "core_messages", self.messages[core_start:core_stop])
            object.__setattr__(self, "next_bridge", self.messages[core_stop:])
        elif any(section is None for section in sections):
            raise ValueError("all three evidence-window sections must be provided together")

        assert self.previous_bridge is not None
        assert self.core_messages is not None
        assert self.next_bridge is not None
        if not self.core_messages:
            raise ValueError("evidence window core_messages must be non-empty")
        if (
            self.previous_bridge + self.core_messages + self.next_bridge
            != self.messages
        ):
            raise ValueError(
                "previous_bridge, core_messages, and next_bridge must partition messages"
            )

        if not self.retrieval_ranks:
            object.__setattr__(
                self,
                "retrieval_ranks",
                tuple(
                    (message_id, rank)
                    for rank, message_id in enumerate(
                        self.retrieved_message_ids, start=1
                    )
                ),
            )
        rank_ids: set[str] = set()
        ranks: set[int] = set()
        for message_id, rank in self.retrieval_ranks:
            if message_id in rank_ids:
                raise ValueError(f"duplicate retrieval rank entry for {message_id}")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                raise ValueError("retrieval ranks must be positive integers")
            if rank in ranks:
                raise ValueError(f"duplicate retrieval rank {rank}")
            rank_ids.add(message_id)
            ranks.add(rank)
        if rank_ids != set(self.retrieved_message_ids):
            raise ValueError(
                "retrieval_ranks must contain exactly the retrieved message ids"
            )

    @property
    def retrieval_rank_by_message_id(self) -> dict[str, int]:
        return dict(self.retrieval_ranks)

    @property
    def decision_messages(self) -> tuple[Message, ...]:
        """Non-empty role messages in the model-visible section order."""

        assert self.previous_bridge is not None
        assert self.core_messages is not None
        assert self.next_bridge is not None
        return tuple(
            message
            for section in (
                self.previous_bridge,
                self.core_messages,
                self.next_bridge,
            )
            for message in section
            if message.role.strip() and message.content.strip()
        )

    @property
    def position_by_message_id(self) -> dict[str, str]:
        assert self.previous_bridge is not None
        assert self.core_messages is not None
        assert self.next_bridge is not None
        return {
            message.id: position
            for position, section in (
                ("previous_bridge", self.previous_bridge),
                ("core", self.core_messages),
                ("next_bridge", self.next_bridge),
            )
            for message in section
        }


def build_windows(
    history: Sequence[Message],
    retrieved: Sequence[RankedMessage | Message],
    *,
    radius: int = 2,
    max_messages: int = 8,
    stride: int = 7,
    merge_rule: MergeRule = "paper",
) -> list[EvidenceWindow]:
    """Build chronological local windows around retrieved messages.

    ``paper`` implements arXiv v2 equation (1): merge consecutive retrieved
    indices when their distance is <= ``2 * radius``. ``code`` is the common
    inclusive-boundary implementation that also merges directly adjacent
    expanded spans (distance <= ``2 * radius + 1``). Both are exposed because
    that one-message boundary choice changes the public reconstruction.
    """

    if radius < 0:
        raise ValueError("window radius must be non-negative")
    if max_messages <= 0 or stride <= 0 or stride > max_messages:
        raise ValueError("require 0 < stride <= max_messages")
    if merge_rule not in ("paper", "code"):
        raise ValueError("merge_rule must be 'paper' or 'code'")
    if [message.index for message in history] != list(range(len(history))):
        raise ValueError("history indices must be contiguous")

    history_by_id = {message.id: message for message in history}
    session_order: list[str] = []
    messages_by_session: dict[str, list[Message]] = {}
    closed_sessions: set[str] = set()
    previous_session: str | None = None
    for message in history:
        if message.session_id != previous_session:
            if message.session_id in closed_sessions:
                raise ValueError("history interleaves messages from the same session")
            if previous_session is not None:
                closed_sessions.add(previous_session)
            previous_session = message.session_id
            session_order.append(message.session_id)
            messages_by_session[message.session_id] = []
        messages_by_session[message.session_id].append(message)

    positions_by_id = {
        message.id: position
        for session_messages in messages_by_session.values()
        for position, message in enumerate(session_messages)
    }
    retrieved_ids: set[str] = set()
    retrieval_rank_by_id: dict[str, int] = {}
    core_by_session: dict[str, list[int]] = {}
    for input_rank, item in enumerate(retrieved, start=1):
        message = item.message if isinstance(item, RankedMessage) else item
        if message.id not in history_by_id:
            raise ValueError(f"retrieved message {message.id} is absent from history")
        if message.id in retrieved_ids:
            continue
        rank = item.rank if isinstance(item, RankedMessage) else input_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("retrieved message ranks must be positive integers")
        retrieved_ids.add(message.id)
        retrieval_rank_by_id[message.id] = rank
        core_by_session.setdefault(message.session_id, []).append(
            positions_by_id[message.id]
        )
    if not core_by_session:
        return []

    threshold = 2 * radius + (1 if merge_rule == "code" else 0)
    windows: list[EvidenceWindow] = []
    for session_id in session_order:
        core_indices = sorted(core_by_session.get(session_id, []))
        if not core_indices:
            continue
        session_history = messages_by_session[session_id]
        groups: list[list[int]] = [[core_indices[0]]]
        for index in core_indices[1:]:
            if index - groups[-1][-1] <= threshold:
                groups[-1].append(index)
            else:
                groups.append([index])

        for group in groups:
            interval_start = max(0, group[0] - radius)
            interval_stop = min(len(session_history), group[-1] + radius + 1)
            sub_start = interval_start
            while sub_start < interval_stop:
                sub_stop = min(sub_start + max_messages, interval_stop)
                messages = tuple(session_history[sub_start:sub_stop])
                global_start = messages[0].index
                global_stop = messages[-1].index + 1
                if global_stop - global_start != len(messages):
                    raise ValueError("session messages must be contiguous in history")
                section_core_start = max(sub_start, group[0])
                section_core_stop = min(sub_stop, group[-1] + 1)
                if section_core_start >= section_core_stop:
                    if sub_stop == interval_stop:
                        break
                    sub_start += stride
                    continue
                previous_bridge = tuple(
                    session_history[sub_start:section_core_start]
                )
                core_messages = tuple(
                    session_history[section_core_start:section_core_stop]
                )
                next_bridge = tuple(session_history[section_core_stop:sub_stop])
                sub_core_ids_chronological = tuple(
                    session_history[index].id
                    for index in group
                    if sub_start <= index < sub_stop
                )
                sub_core_ids = tuple(
                    sorted(
                        sub_core_ids_chronological,
                        key=retrieval_rank_by_id.__getitem__,
                    )
                )
                windows.append(
                    EvidenceWindow(
                        id=f"pending:{global_start}-{global_stop}",
                        order=0,
                        start=global_start,
                        stop=global_stop,
                        messages=messages,
                        retrieved_message_ids=sub_core_ids,
                        retrieval_ranks=tuple(
                            (message_id, retrieval_rank_by_id[message_id])
                            for message_id in sub_core_ids
                        ),
                        previous_bridge=previous_bridge,
                        core_messages=core_messages,
                        next_bridge=next_bridge,
                    )
                )
                if sub_stop == interval_stop:
                    break
                sub_start += stride
    windows.sort(key=lambda window: (window.start, window.stop, window.messages[0].id))
    return [
        EvidenceWindow(
            id=f"w{order:04d}:{window.start}-{window.stop}",
            order=order,
            start=window.start,
            stop=window.stop,
            messages=window.messages,
            retrieved_message_ids=window.retrieved_message_ids,
            retrieval_ranks=window.retrieval_ranks,
            previous_bridge=window.previous_bridge,
            core_messages=window.core_messages,
            next_bridge=window.next_bridge,
        )
        for order, window in enumerate(windows)
    ]


def covered_message_ids(windows: Sequence[EvidenceWindow]) -> tuple[str, ...]:
    by_index = {
        message.index: message.id for window in windows for message in window.messages
    }
    return tuple(by_index[index] for index in sorted(by_index))
