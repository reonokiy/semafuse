"""Append-only JSONL journal with deterministic resume keys."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


class JsonlJournal:
    def __init__(self, path: str | Path, *, key_field: str = "run_key") -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = set()
        if self.path.exists():
            for line_number, item in enumerate(read_jsonl(self.path), start=1):
                if key_field not in item:
                    raise ValueError(f"JSONL row {line_number} lacks {key_field!r}")
                key = str(item[key_field])
                if key in self._keys:
                    raise ValueError(f"duplicate resume key {key!r} in {self.path}")
                self._keys.add(key)

    @property
    def completed_keys(self) -> frozenset[str]:
        return frozenset(self._keys)

    def field_values(self, field: str) -> frozenset[str]:
        if not self.path.exists():
            return frozenset()
        return frozenset(
            str(item[field]) for item in read_jsonl(self.path) if field in item
        )

    def append(self, item: Mapping[str, Any]) -> None:
        if self.key_field not in item:
            raise ValueError(f"result lacks {self.key_field!r}")
        key = str(item[self.key_field])
        if key in self._keys:
            raise ValueError(f"resume key {key!r} already exists")
        encoded = json.dumps(
            dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._keys.add(key)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value
