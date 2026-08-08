"""Reproducible multi-task execution, resume, aggregation, and provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .dataset import load_task, resolve_candidates
from .manifest import sha256_file, validate_semafuse_options
from .metrics import CUTOFFS, evaluate_rankings
from .model import HarnessError, LoadedTask, Manifest, RankerConfig, RankingEntry
from .protocol import load_predictions, prediction_record, write_jsonl_record
from .rankers import BM25Ranker, ExternalRanker, SemafuseJsonlRanker, resolve_external_path

_OFFICIAL = {
    "lmeb_repository": "https://github.com/KaLM-Embedding/LMEB",
    "lmeb_commit": "a02ae842598183ed162fc90c58ffaae5eec89f12",
    "mteb_version": "2.3.0",
    "mteb_commit": "072e6ef63e961128ea60a697cc077feb2fd638d0",
    "trec_eval_version": "9.0.8",
    "bm25s_version": "0.3.6",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _safe_config(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in ("key", "token", "secret", "password")):
        return "<redacted>"
    if isinstance(value, dict):
        return {name: _safe_config(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_safe_config(item) for item in value]
    return value


def _task_token(task_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-")[:60] or "task"
    return f"{slug}-{hashlib.sha256(task_id.encode()).hexdigest()[:8]}"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _input_hashes(tasks: tuple[LoadedTask, ...], ranker: RankerConfig, base: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for task in tasks:
        files = {
            "corpus": task.config.corpus,
            "queries": task.config.queries,
            "qrels": task.config.qrels,
        }
        if task.config.candidates is not None:
            files["candidates"] = task.config.candidates
        values[task.config.id] = {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        }
    if ranker.kind == "external":
        path = resolve_external_path(ranker.options.get("predictions"), base)
        values["external_predictions"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if ranker.kind == "semafuse-jsonl":
        command = ranker.options.get("command", [])
        inferred: dict[str, Any] = {}
        if isinstance(command, list):
            for index, value in enumerate(command):
                if not isinstance(value, str):
                    continue
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = (base / candidate).resolve()
                if candidate.is_file() and (index == 0 or command[index - 1] == "--config"):
                    inferred[str(candidate)] = {
                        "sha256": sha256_file(candidate),
                        "bytes": candidate.stat().st_size,
                    }
        values["semafuse_artifacts"] = inferred
    return values


def _fingerprint(
    manifest: Manifest, ranker: RankerConfig, tasks: tuple[LoadedTask, ...], hashes: Mapping[str, Any]
) -> tuple[str, Mapping[str, Any]]:
    semantic = {
        "schema": "lmeb-run-fingerprint-v1",
        "harness_version": __version__,
        "harness_source_sha256": _source_hash(),
        "protocol": manifest.run.protocol,
        "instruction_setting": manifest.run.instruction_setting,
        "cutoffs": list(CUTOFFS),
        "ranker_kind": ranker.kind,
        "ranker_config_sha256": hashlib.sha256(_canonical(ranker.options)).hexdigest(),
        "tasks": [
            {
                "id": task.config.id,
                "dataset": task.config.dataset,
                "subset": task.config.subset,
                "memory_type": task.config.memory_type,
                "candidate_match": task.config.candidate_match,
                "instruction": task.config.instruction,
            }
            for task in tasks
        ],
        "inputs": hashes,
    }
    digest = hashlib.sha256(_canonical(semantic)).hexdigest()
    return digest, semantic


def _filter_rankings(
    task: LoadedTask,
    query_id: str,
    rankings: Mapping[str, list[RankingEntry]],
    candidate_ids: tuple[str, ...],
    *,
    require_coverage: bool,
) -> dict[str, list[RankingEntry]]:
    corpus_ids = set(task.corpus)
    candidates = set(candidate_ids)
    filtered: dict[str, list[RankingEntry]] = {}
    for stage, entries in rankings.items():
        unknown = {entry.doc_id for entry in entries} - corpus_ids
        if unknown:
            raise HarnessError(
                f"{task.config.id}/{query_id}/{stage} contains unknown docs: {sorted(unknown)[:5]}"
            )
        values = [entry for entry in entries if entry.doc_id in candidates]
        if require_coverage:
            missing = candidates - {entry.doc_id for entry in values}
            if missing:
                raise HarnessError(
                    f"paper reproduction requires scores for every admissible doc; "
                    f"{task.config.id}/{query_id}/{stage} misses {len(missing)}"
                )
        filtered[stage] = values
    if not filtered:
        raise HarnessError(f"ranker emitted no stages for {task.config.id}/{query_id}")
    return filtered


def _load_checkpoint(path: Path, task_id: str) -> dict[str, tuple[dict[str, list[RankingEntry]], Mapping[str, Any]]]:
    if not path.exists():
        return {}
    return {
        query_id: value
        for (loaded_task_id, query_id), value in load_predictions(path, default_task_id=task_id).items()
        if loaded_task_id == task_id
    }


def _average_metrics(values: list[Mapping[str, float | None]]) -> dict[str, float | None]:
    if not values:
        return {}
    output: dict[str, float | None] = {}
    for metric in values[0]:
        present = [value[metric] for value in values if value[metric] is not None]
        output[metric] = sum(present) / len(present) if present else None
    return output


def _aggregate(task_results: Mapping[str, Any]) -> dict[str, Any]:
    stages = sorted(
        {stage for result in task_results.values() for stage in result["metrics"]["stages"]}
    )
    aggregate: dict[str, Any] = {}
    for stage in stages:
        by_dataset_raw: dict[str, list[Mapping[str, float | None]]] = {}
        dataset_type: dict[str, str] = {}
        task_macros = []
        for result in task_results.values():
            stage_result = result["metrics"]["stages"].get(stage)
            if stage_result is None:
                continue
            macro = stage_result["macro"]
            task_macros.append(macro)
            by_dataset_raw.setdefault(result["dataset"], []).append(macro)
            dataset_type[result["dataset"]] = result["memory_type"]
        by_dataset = {name: _average_metrics(items) for name, items in sorted(by_dataset_raw.items())}
        by_type_raw: dict[str, list[Mapping[str, float | None]]] = {}
        for dataset, metrics in by_dataset.items():
            by_type_raw.setdefault(dataset_type[dataset], []).append(metrics)
        by_type = {name: _average_metrics(items) for name, items in sorted(by_type_raw.items())}
        aggregate[stage] = {
            "mean_task": _average_metrics(task_macros),
            "by_dataset": by_dataset,
            "mean_dataset": _average_metrics(list(by_dataset.values())),
            "by_memory_type": by_type,
            "mean_memory_type": _average_metrics(list(by_type.values())),
        }
    return aggregate


def run_manifest(
    manifest: Manifest,
    *,
    ranker_override: str | None = None,
    output_override: Path | None = None,
    resume_override: bool | None = None,
) -> tuple[Path, bool]:
    ranker = RankerConfig(ranker_override or manifest.ranker.kind, manifest.ranker.options)
    if ranker.kind not in {"bm25", "external", "semafuse-jsonl"}:
        raise HarnessError(f"unknown ranker: {ranker.kind}")
    if ranker.kind == "semafuse-jsonl":
        validate_semafuse_options(ranker.options)
    if ranker.kind == "semafuse-jsonl" and manifest.run.protocol == "paper-reproduction":
        raise HarnessError(
            "semafuse-jsonl uses a cold candidate corpus and cannot run under the "
            "paper-reproduction label; select run.protocol='common-policy'"
        )
    tasks = tuple(load_task(config) for config in manifest.tasks)
    base = manifest.source_path.parent
    hashes = _input_hashes(tasks, ranker, base)
    fingerprint, fingerprint_payload = _fingerprint(manifest, ranker, tasks, hashes)
    output_root = (output_override or manifest.run.output_dir).resolve()
    run_dir = output_root / f"{manifest.run.name}-{fingerprint[:16]}"
    result_path = run_dir / "result.json"
    resume = manifest.run.resume if resume_override is None else resume_override
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != fingerprint:
            raise HarnessError(f"result fingerprint mismatch: {result_path}")
        if resume:
            return result_path, True
        # An explicit no-resume is a request to rebuild this exact
        # content-addressed run, including its per-task checkpoints.
        result_path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    use_instruction = manifest.run.instruction_setting == "with"
    external = None
    semafuse = None
    if ranker.kind == "external":
        external = ExternalRanker(
            resolve_external_path(ranker.options.get("predictions"), base),
            [task.config.id for task in tasks],
        )
    elif ranker.kind == "semafuse-jsonl":
        semafuse_options = dict(ranker.options)
        command = semafuse_options.get("command")
        if isinstance(command, list):
            resolved = []
            for index, value in enumerate(command):
                if isinstance(value, str) and (index == 0 or command[index - 1] == "--config"):
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute() and (base / candidate).exists():
                        value = str((base / candidate).resolve())
                resolved.append(value)
            semafuse_options["command"] = resolved
        semafuse = SemafuseJsonlRanker(semafuse_options, use_instruction)

    task_results: dict[str, Any] = {}
    try:
        for task in tasks:
            task_dir = run_dir / "tasks" / _task_token(task.config.id)
            task_dir.mkdir(parents=True, exist_ok=True)
            partial = task_dir / "predictions.partial.jsonl"
            completed = _load_checkpoint(partial, task.config.id) if resume else {}
            if not resume:
                # These are exact, content-addressed run-local files. Starting without
                # resume must rebuild them instead of appending duplicate query rows.
                (task_dir / "state.json").unlink(missing_ok=True)
            bm25 = BM25Ranker(task, ranker.options, use_instruction) if ranker.kind == "bm25" else None
            mode = "candidate-cold-corpus" if ranker.kind == "semafuse-jsonl" else "global-score-then-filter"
            with partial.open("a" if resume else "w", encoding="utf-8") as handle:
                for query in task.queries.values():
                    if query.id in completed:
                        continue
                    candidates = resolve_candidates(task, query.id)
                    if bm25 is not None:
                        rankings, metadata = bm25.rank(query, candidates)
                    elif external is not None:
                        rankings, metadata = external.rank_for(task, query, candidates)
                    else:
                        assert semafuse is not None
                        rankings, metadata = semafuse.rank_for(task, query, candidates)
                    rankings = _filter_rankings(
                        task,
                        query.id,
                        rankings,
                        candidates.doc_ids,
                        require_coverage=ranker.kind in {"bm25", "external"}
                        and manifest.run.protocol == "paper-reproduction",
                    )
                    record = prediction_record(
                        task.config.id,
                        query.id,
                        rankings,
                        {**metadata, "candidate_policy": mode},
                    )
                    write_jsonl_record(handle, record)
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed[query.id] = (rankings, record["metadata"])
                    _atomic_json(
                        task_dir / "state.json",
                        {
                            "schema_version": "lmeb-resume-state-v1",
                            "run_fingerprint": fingerprint,
                            "task_id": task.config.id,
                            "completed_queries": len(completed),
                            "total_queries": len(task.queries),
                        },
                    )
            if set(completed) != set(task.queries):
                raise HarnessError(f"checkpoint is incomplete for {task.config.id}")
            final_predictions = task_dir / "predictions.jsonl"
            shutil.copyfile(partial, final_predictions)
            stages = sorted({stage for rankings, _ in completed.values() for stage in rankings})
            stage_metrics = {}
            for stage in stages:
                stage_rankings = {
                    query_id: rankings[stage]
                    for query_id, (rankings, _) in completed.items()
                    if stage in rankings
                }
                stage_metrics[stage] = evaluate_rankings(task.qrels, stage_rankings)
            task_result = {
                "schema_version": "lmeb-task-result-v1",
                "task_id": task.config.id,
                "dataset": task.config.dataset,
                "subset": task.config.subset,
                "memory_type": task.config.memory_type,
                "license": task.config.license,
                "counts": {
                    "corpus": len(task.corpus),
                    "queries": len(task.queries),
                    "candidate_scenes": len(task.candidates),
                },
                "metrics": {"stages": stage_metrics},
                "artifacts": {
                    "predictions": str(final_predictions.relative_to(run_dir)),
                    "predictions_sha256": sha256_file(final_predictions),
                },
            }
            _atomic_json(task_dir / "result.json", task_result)
            task_results[task.config.id] = task_result
    finally:
        if external is not None:
            external.close()
        if semafuse is not None:
            semafuse.close()

    fidelity = {
        "metrics": "exact LMEB/MTEB-2.3.0 formulas",
        "candidate_filter": "official global-score-then-filter"
        if ranker.kind != "semafuse-jsonl"
        else "common-policy deviation: candidates ingested cold per query",
        "ranker": (
            "diagnostic approximation: stdlib BM25 lacks PyStemmer/bm25s float32"
            if ranker.kind == "bm25"
            else "user-supplied predictions"
            if ranker.kind == "external"
            else "Semafuse production retrieval-only pipeline"
        ),
    }
    result = {
        "schema_version": "lmeb-result-v1",
        "status": "complete",
        "run_id": fingerprint[:16],
        "run_fingerprint": fingerprint,
        "name": manifest.run.name,
        "protocol": manifest.run.protocol,
        "instruction_setting": manifest.run.instruction_setting,
        "cutoffs": list(CUTOFFS),
        "ranker": {"kind": ranker.kind, "config": _safe_config(dict(ranker.options))},
        "fidelity": fidelity,
        "tasks": task_results,
        "aggregate": _aggregate(task_results),
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "harness_version": __version__,
            "harness_source_sha256": _source_hash(),
            "manifest": str(manifest.source_path),
            "manifest_sha256": manifest.source_sha256,
            "fingerprint_payload": fingerprint_payload,
            "inputs": hashes,
            "official_compatibility_targets": _OFFICIAL,
            "runtime": {"python": sys.version, "platform": platform.platform()},
        },
    }
    _atomic_json(result_path, result)
    return result_path, False
