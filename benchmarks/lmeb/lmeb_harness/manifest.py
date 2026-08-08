"""Strict JSON/TOML manifest loading with path resolution."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .model import HarnessError, Manifest, RankerConfig, RunConfig, TaskConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only Python 3.10 and older
    tomllib = None  # type: ignore[assignment]


_PROTOCOLS = {"paper-reproduction", "common-policy"}
_INSTRUCTION_SETTINGS = {"with", "without"}
_CANDIDATE_POLICIES = {"official", "query-id", "prefix-2", "longest-prefix"}
_RANKERS = {"bm25", "external", "semafuse-jsonl"}
_SECRET_FLAG = re.compile(
    r"(?i)(api[-_]?key|access[-_]?key|auth(?:orization)?|bearer|password|secret|token)"
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:^|[?&;,\s])(?:[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)|authorization)\s*[:=]"
    r"|^(?:sk-|hf_|ghp_|github_pat_|Bearer\s)"
)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be an object/table")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise HarnessError(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise HarnessError(f"{label} must be a boolean")
    return value


def _resolve(base: Path, value: Any, label: str) -> tuple[Path, str]:
    declared = _string(value, label)
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(), declared


def _load_raw(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".json":
            try:
                value = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise HarnessError(f"invalid JSON manifest {path}: {error}") from error
        elif suffix in {".toml", ".tml"}:
            if tomllib is None:
                raise HarnessError("TOML manifests require Python 3.11+; JSON works on older Python")
            try:
                value = tomllib.load(handle)
            except tomllib.TOMLDecodeError as error:
                raise HarnessError(f"invalid TOML manifest {path}: {error}") from error
        else:
            raise HarnessError("manifest extension must be .json or .toml")
    return _mapping(value, "manifest")


def validate_semafuse_options(options: Mapping[str, Any]) -> None:
    command = options.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise HarnessError("semafuse-jsonl ranker.command must be a non-empty string array")
    for index, argument in enumerate(command):
        flag_name = argument.split("=", 1)[0] if argument.startswith("-") else ""
        if flag_name and _SECRET_FLAG.search(flag_name):
            raise HarnessError(
                "semafuse-jsonl command must not contain secret flags/values; pass secrets via environment"
            )
        if _SECRET_VALUE.search(argument):
            raise HarnessError(
                f"semafuse-jsonl command argument {index} looks like an inline secret; "
                "pass only its environment-variable name"
            )
    env_names = options.get("env_names", [])
    if not isinstance(env_names, list) or not all(
        isinstance(name, str) and _ENV_NAME.fullmatch(name) for name in env_names
    ):
        raise HarnessError("semafuse-jsonl ranker.env_names must be an array of environment names")


def load_manifest(path_value: str | Path) -> Manifest:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise HarnessError(f"manifest does not exist: {path}")
    raw = _load_raw(path)
    if raw.get("schema_version") != 1:
        raise HarnessError("manifest schema_version must be 1")
    base = path.parent

    run_raw = _mapping(raw.get("run"), "run")
    protocol = _string(run_raw.get("protocol", "common-policy"), "run.protocol")
    if protocol not in _PROTOCOLS:
        raise HarnessError(f"run.protocol must be one of {sorted(_PROTOCOLS)}")
    instruction_setting = _string(
        run_raw.get("instruction_setting", "without"), "run.instruction_setting"
    )
    if instruction_setting not in _INSTRUCTION_SETTINGS:
        raise HarnessError(
            f"run.instruction_setting must be one of {sorted(_INSTRUCTION_SETTINGS)}"
        )
    output_dir, _ = _resolve(base, run_raw.get("output_dir", "results"), "run.output_dir")
    run = RunConfig(
        name=_string(run_raw.get("name", "lmeb"), "run.name"),
        protocol=protocol,
        instruction_setting=instruction_setting,
        output_dir=output_dir,
        resume=_bool(run_raw.get("resume", True), "run.resume"),
    )

    ranker_raw = dict(_mapping(raw.get("ranker"), "ranker"))
    kind = _string(ranker_raw.pop("kind", None), "ranker.kind")
    if kind not in _RANKERS:
        raise HarnessError(f"ranker.kind must be one of {sorted(_RANKERS)}")
    if kind == "semafuse-jsonl":
        validate_semafuse_options(ranker_raw)
    ranker = RankerConfig(kind=kind, options=ranker_raw)

    task_values = raw.get("tasks")
    if not isinstance(task_values, list) or not task_values:
        raise HarnessError("tasks must be a non-empty array")
    tasks: list[TaskConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(task_values):
        task_raw = _mapping(item, f"tasks[{index}]")
        prefix = f"tasks[{index}]"
        task_id = _string(task_raw.get("id"), f"{prefix}.id")
        if task_id in seen:
            raise HarnessError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        corpus, corpus_declared = _resolve(base, task_raw.get("corpus"), f"{prefix}.corpus")
        queries, queries_declared = _resolve(base, task_raw.get("queries"), f"{prefix}.queries")
        qrels, qrels_declared = _resolve(base, task_raw.get("qrels"), f"{prefix}.qrels")
        candidates_value = task_raw.get("candidates")
        candidates: Path | None = None
        candidates_declared = ""
        if candidates_value is not None:
            candidates, candidates_declared = _resolve(
                base, candidates_value, f"{prefix}.candidates"
            )
        candidate_match = _string(
            task_raw.get("candidate_match", "official"), f"{prefix}.candidate_match"
        )
        if candidate_match not in _CANDIDATE_POLICIES:
            raise HarnessError(
                f"{prefix}.candidate_match must be one of {sorted(_CANDIDATE_POLICIES)}"
            )
        instruction = _string(
            task_raw.get("instruction", ""), f"{prefix}.instruction", allow_empty=True
        )
        if instruction_setting == "with" and not instruction.strip():
            raise HarnessError(
                f"{prefix}.instruction is required when run.instruction_setting='with'"
            )
        tasks.append(
            TaskConfig(
                id=task_id,
                dataset=_string(task_raw.get("dataset"), f"{prefix}.dataset"),
                subset=_string(task_raw.get("subset", "test"), f"{prefix}.subset"),
                memory_type=_string(
                    task_raw.get("memory_type", "LMEB_Unknown"), f"{prefix}.memory_type"
                ),
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                candidates=candidates,
                candidate_match=candidate_match,
                instruction=instruction,
                license=_string(
                    task_raw.get("license", "unknown"), f"{prefix}.license", allow_empty=True
                ),
                declared_paths={
                    "corpus": corpus_declared,
                    "queries": queries_declared,
                    "qrels": qrels_declared,
                    "candidates": candidates_declared,
                },
            )
        )
    return Manifest(
        source_path=path,
        source_sha256=sha256_file(path),
        run=run,
        ranker=ranker,
        tasks=tuple(tasks),
    )
