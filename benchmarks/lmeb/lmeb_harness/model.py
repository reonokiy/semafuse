"""Typed values shared by LMEB loaders, rankers, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class HarnessError(ValueError):
    """An actionable manifest, dataset, protocol, or evaluation error."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""

    @property
    def combined_text(self) -> str:
        if self.title and self.text:
            return f"{self.title}\n{self.text}"
        return self.title or self.text


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class RankingEntry:
    doc_id: str
    score: float


@dataclass(frozen=True)
class CandidateResolution:
    doc_ids: tuple[str, ...]
    scene_id: str | None
    used_full_corpus_fallback: bool


@dataclass(frozen=True)
class TaskConfig:
    id: str
    dataset: str
    subset: str
    memory_type: str
    corpus: Path
    queries: Path
    qrels: Path
    candidates: Path | None
    candidate_match: str
    instruction: str
    license: str
    declared_paths: Mapping[str, str]


@dataclass(frozen=True)
class RunConfig:
    name: str
    protocol: str
    instruction_setting: str
    output_dir: Path
    resume: bool


@dataclass(frozen=True)
class RankerConfig:
    kind: str
    options: Mapping[str, Any]


@dataclass(frozen=True)
class Manifest:
    source_path: Path
    source_sha256: str
    run: RunConfig
    ranker: RankerConfig
    tasks: tuple[TaskConfig, ...]


@dataclass(frozen=True)
class LoadedTask:
    config: TaskConfig
    corpus: Mapping[str, Document]
    queries: Mapping[str, Query]
    qrels: Mapping[str, Mapping[str, int]]
    candidates: Mapping[str, tuple[str, ...]]
