from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


class HarnessError(ValueError):
    """Raised when benchmark inputs violate the pinned protocol."""


@dataclass(frozen=True)
class SourceProfile:
    name: str
    split: str
    competency: str
    paper_name: str
    expected_contexts: int
    expected_queries: int
    chunk_size_tokens: int
    generation_max_tokens: int
    prompt_family: str
    primary_metric: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    raw: dict[str, Any]
    profiles: dict[str, SourceProfile]

    @property
    def dataset_repo_id(self) -> str:
        return str(self.raw["dataset"]["repo_id"])

    @property
    def dataset_revision(self) -> str:
        return str(self.raw["dataset"]["revision"])

    @property
    def top_k(self) -> int:
        return int(self.raw["paper_track"]["top_k"])


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "manifests" / "official-main-v2.json"


def load_manifest(path: str | Path | None = None) -> Manifest:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    with manifest_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw.get("schema_version") != 1:
        raise HarnessError("unsupported manifest schema_version")

    revision = raw.get("dataset", {}).get("revision", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise HarnessError("dataset revision must be a full lowercase Git commit")
    if raw.get("dataset", {}).get("license") != "mit":
        raise HarnessError("unexpected dataset license in manifest")
    files = raw.get("dataset", {}).get("files", {})
    if not files:
        raise HarnessError("manifest has no pinned dataset files")
    for name, value in files.items():
        digest = value.get("lfs_sha256", "")
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise HarnessError(f"invalid LFS SHA-256 for {name}")
        if not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
            raise HarnessError(f"invalid byte size for {name}")

    profiles: dict[str, SourceProfile] = {}
    for name, value in raw.get("sources", {}).items():
        profile = SourceProfile(name=name, **value)
        if profile.primary_metric != "substring_exact_match":
            raise HarnessError(f"unsupported primary metric for {name}")
        if min(
            profile.expected_contexts,
            profile.expected_queries,
            profile.chunk_size_tokens,
            profile.generation_max_tokens,
        ) <= 0:
            raise HarnessError(f"non-positive profile setting for {name}")
        profiles[name] = profile
    if not profiles:
        raise HarnessError("manifest has no source profiles")

    return Manifest(path=manifest_path.resolve(), raw=raw, profiles=profiles)
