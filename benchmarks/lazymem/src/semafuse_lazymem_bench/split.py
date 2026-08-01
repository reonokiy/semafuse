"""Deterministic public reconstruction of the paper's LongMemEval split."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import Question


SPLIT_ORDER = ("train", "validation", "test")
PAPER_SPLIT_SIZES = {"train": 360, "validation": 40, "test": 100}


@dataclass(frozen=True, slots=True)
class SplitManifest:
    seed: int
    source_sha256: str | None
    ids: dict[str, tuple[str, ...]]
    counts_by_type: dict[str, dict[str, int]]
    ids_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm": {
                "name": "stratified-largest-remainder",
                "seed": self.seed,
                "pre_shuffle_order": "question_id ascending",
                "stratum_order": "question_type ascending",
                "rng": "Python random.Random (MT19937), one stream",
                "allocation": "row-wise largest remainder, then column rebalance",
                "output_id_order": "ascending",
            },
            "source_sha256": self.source_sha256,
            "ids_sha256": self.ids_sha256,
            "counts": {name: len(self.ids[name]) for name in SPLIT_ORDER},
            "counts_by_type": self.counts_by_type,
            "ids": {name: list(self.ids[name]) for name in SPLIT_ORDER},
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_remainder_rows(
    type_counts: dict[str, int], split_sizes: dict[str, int]
) -> dict[str, dict[str, int]]:
    total = sum(type_counts.values())
    if total != sum(split_sizes.values()):
        raise ValueError("type counts and requested split sizes have different totals")

    quotas: dict[str, dict[str, float]] = {}
    allocations: dict[str, dict[str, int]] = {}
    for question_type in sorted(type_counts):
        count = type_counts[question_type]
        quotas[question_type] = {
            split: count * split_sizes[split] / total for split in SPLIT_ORDER
        }
        row = {split: math.floor(quotas[question_type][split]) for split in SPLIT_ORDER}
        remaining = count - sum(row.values())
        ranked = sorted(
            SPLIT_ORDER,
            key=lambda split: (-(quotas[question_type][split] - row[split]), SPLIT_ORDER.index(split)),
        )
        for split in ranked[:remaining]:
            row[split] += 1
        allocations[question_type] = row

    # Independent row rounding can miss an exact requested column total. Move
    # one unit at a time while minimizing the increase in absolute quota error.
    while True:
        column_totals = {
            split: sum(row[split] for row in allocations.values())
            for split in SPLIT_ORDER
        }
        surplus = [
            split for split in SPLIT_ORDER if column_totals[split] > split_sizes[split]
        ]
        deficit = [
            split for split in SPLIT_ORDER if column_totals[split] < split_sizes[split]
        ]
        if not surplus and not deficit:
            break
        if not surplus or not deficit:
            raise AssertionError("unbalanced largest-remainder allocation")

        candidates: list[tuple[float, str, str, str]] = []
        for source in surplus:
            for target in deficit:
                for question_type in sorted(type_counts):
                    if allocations[question_type][source] == 0:
                        continue
                    before = abs(
                        allocations[question_type][source] - quotas[question_type][source]
                    ) + abs(
                        allocations[question_type][target] - quotas[question_type][target]
                    )
                    after = abs(
                        allocations[question_type][source] - 1 - quotas[question_type][source]
                    ) + abs(
                        allocations[question_type][target] + 1 - quotas[question_type][target]
                    )
                    candidates.append((after - before, question_type, source, target))
        if not candidates:
            raise AssertionError("cannot rebalance largest-remainder allocation")
        _, question_type, source, target = min(candidates)
        allocations[question_type][source] -= 1
        allocations[question_type][target] += 1
    return allocations


def stratified_split(
    questions: Iterable[Question],
    *,
    seed: int = 42,
    split_sizes: dict[str, int] | None = None,
    source_sha256: str | None = None,
) -> SplitManifest:
    question_list = list(questions)
    requested = dict(split_sizes or PAPER_SPLIT_SIZES)
    if tuple(requested) != SPLIT_ORDER:
        requested = {name: requested[name] for name in SPLIT_ORDER}
    ids = [question.id for question in question_list]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique before splitting")
    if len(question_list) != sum(requested.values()):
        raise ValueError(
            f"expected {sum(requested.values())} questions, got {len(question_list)}"
        )

    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in question_list:
        grouped[question.question_type].append(question)
    allocations = _largest_remainder_rows(
        {question_type: len(group) for question_type, group in grouped.items()}, requested
    )

    rng = random.Random(seed)
    split_ids: dict[str, list[str]] = {name: [] for name in SPLIT_ORDER}
    for question_type in sorted(grouped):
        group = sorted(grouped[question_type], key=lambda question: question.id)
        rng.shuffle(group)
        cursor = 0
        for split in SPLIT_ORDER:
            next_cursor = cursor + allocations[question_type][split]
            split_ids[split].extend(question.id for question in group[cursor:next_cursor])
            cursor = next_cursor
        if cursor != len(group):
            raise AssertionError(f"failed to allocate complete stratum {question_type}")

    stable_ids = {name: tuple(sorted(split_ids[name])) for name in SPLIT_ORDER}
    canonical = json.dumps(
        {name: list(stable_ids[name]) for name in SPLIT_ORDER},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SplitManifest(
        seed=seed,
        source_sha256=source_sha256,
        ids=stable_ids,
        counts_by_type={
            question_type: dict(allocations[question_type])
            for question_type in sorted(allocations)
        },
        ids_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
