from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import platform
import sys
import time

from .bm25 import run_bm25
from .model import HarnessError, Manifest
from .protocol import prepare_source, sha256_file
from .scoring import score_predictions, write_report
from .source_verification import download_and_verify_files


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_versions() -> dict[str, str]:
    packages = (
        "semafuse-memoryagentbench",
        "datasets",
        "nltk",
        "rank-bm25",
        "tiktoken",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError as error:
            raise HarnessError(f"required package is not installed: {package}") from error
    return versions


def _installed_packages() -> dict[str, str]:
    found: dict[str, str] = {}
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name:
            found[str(name)] = distribution.version
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def _harness_tree_sha256(root: Path) -> str:
    files = sorted((root / "memoryagentbench_harness").glob("*.py"))
    files.extend(
        [
            root / "pyproject.toml",
            root / "manifests" / "official-main-v2.json",
        ]
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _configure_cache(cache_dir: Path) -> None:
    resolved = cache_dir.resolve()
    os.environ["HF_HOME"] = str(resolved)
    os.environ["HF_HUB_CACHE"] = str(resolved / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(resolved / "datasets")
    os.environ["NLTK_DATA"] = str(resolved / "nltk_data")


def _write_sha256sums(output: Path) -> Path:
    checksum_path = output / "SHA256SUMS"
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n")
    return checksum_path


def run_retrieval_suite(
    manifest: Manifest,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    selected_sources = sources or list(manifest.profiles)
    unknown = set(selected_sources) - set(manifest.profiles)
    if unknown:
        raise HarnessError(f"unknown sources: {sorted(unknown)}")
    if not selected_sources:
        raise HarnessError("at least one source must be selected")

    output = Path(output_dir).resolve()
    cache = Path(cache_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    _configure_cache(cache)

    suite_started_at = _utc_now()
    suite_started = time.perf_counter()
    required_splits = sorted(
        {manifest.profiles[source].split for source in selected_sources}
    )
    verification = download_and_verify_files(
        manifest,
        cache / "hub",
        splits=required_splits,
    )

    source_reports: dict[str, Any] = {}
    for source in selected_sources:
        profile = manifest.profiles[source]
        source_dir = output / source
        prepared_dir = source_dir / "prepared"
        source_dir.mkdir(parents=True, exist_ok=True)

        prepare_started = time.perf_counter()
        run = prepare_source(manifest, profile, prepared_dir)
        prepare_seconds = time.perf_counter() - prepare_started

        predictions_path = source_dir / "bm25.predictions.jsonl"
        retrieval_started = time.perf_counter()
        prediction_count = run_bm25(
            prepared_dir / "inputs.jsonl", predictions_path
        )
        retrieval_seconds = time.perf_counter() - retrieval_started

        scoring_started = time.perf_counter()
        report = score_predictions(
            prepared_dir / "inputs.jsonl",
            prepared_dir / "labels.jsonl",
            predictions_path,
        )
        scoring_seconds = time.perf_counter() - scoring_started
        report_path = source_dir / "bm25.report.json"
        write_report(report, report_path)

        if prediction_count != profile.expected_queries:
            raise HarnessError(
                f"{source} produced {prediction_count} predictions, expected "
                f"{profile.expected_queries}"
            )
        if report["query_count_scored"] != profile.expected_queries:
            raise HarnessError(f"{source} did not score its full query set")
        source_reports[source] = {
            "paper_name": profile.paper_name,
            "context_count": run["context_count"],
            "memory_count": run["memory_count"],
            "query_count": run["query_count"],
            "chunk_size_tokens": run["chunk_size_tokens"],
            "top_k": run["top_k"],
            "paper_comparable_shape": run["paper_comparable_shape"],
            "metrics_percent": report["metrics_percent"],
            "metric_counts": report["metric_counts"],
            "mean_retrieved_count": report["mean_retrieved_count"],
            "official_answer_count": report["official_answer_count"],
            "timing_seconds": {
                "prepare": prepare_seconds,
                "bm25_retrieval": retrieval_seconds,
                "score": scoring_seconds,
                "total": prepare_seconds + retrieval_seconds + scoring_seconds,
            },
            "artifacts": {
                "run": {
                    "path": str((prepared_dir / "run.json").relative_to(output)),
                    "sha256": sha256_file(prepared_dir / "run.json"),
                },
                "inputs": {
                    "path": str((prepared_dir / "inputs.jsonl").relative_to(output)),
                    "sha256": run["input_sha256"],
                },
                "labels": {
                    "path": str((prepared_dir / "labels.jsonl").relative_to(output)),
                    "sha256": run["labels_sha256"],
                },
                "predictions": {
                    "path": str(predictions_path.relative_to(output)),
                    "sha256": report["artifacts"]["predictions_sha256"],
                },
                "report": {
                    "path": str(report_path.relative_to(output)),
                    "sha256": sha256_file(report_path),
                },
            },
        }

    total_queries = sum(item["query_count"] for item in source_reports.values())
    metric_names = sorted(
        {
            name
            for item in source_reports.values()
            for name in item["metrics_percent"]
        }
    )
    aggregate_metrics: dict[str, float] = {}
    aggregate_counts: dict[str, int] = {}
    for name in metric_names:
        count = sum(item["metric_counts"].get(name, 0) for item in source_reports.values())
        if count:
            aggregate_metrics[name] = sum(
                item["metrics_percent"].get(name, 0.0)
                * item["metric_counts"].get(name, 0)
                for item in source_reports.values()
            ) / count
            aggregate_counts[name] = count
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "suite": "MemoryAgentBench-main-retrieval-only",
        "started_at_utc": suite_started_at,
        "finished_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - suite_started,
        "paid_api_calls": 0,
        "reader": None,
        "primary_output": "non_official_answer_coverage_at_k",
        "aggregate_is_official_benchmark_score": False,
        "comparability": (
            "retrieval-only diagnostic; not comparable to published end-to-end "
            "Substring Exact Match accuracy"
        ),
        "dataset_verification": verification,
        "source_manifest": {
            "path": str(manifest.path.relative_to(root)),
            "sha256": sha256_file(manifest.path),
        },
        "harness_tree_sha256": _harness_tree_sha256(root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "installed_packages": _installed_packages(),
            "cache_dir": str(cache),
        },
        "reproduction": {
            "command": (
                "benchmarks/memoryagentbench/.venv/bin/python -m "
                "memoryagentbench_harness run-suite "
                f"--output-dir {output} --cache-dir {cache}"
            ),
            "sources": selected_sources,
        },
        "total_contexts": sum(
            item["context_count"] for item in source_reports.values()
        ),
        "total_memories": sum(
            item["memory_count"] for item in source_reports.values()
        ),
        "total_queries": total_queries,
        "aggregate_metrics_percent": aggregate_metrics,
        "aggregate_metric_counts": aggregate_counts,
        "metric_limitations": {
            "EventQA": (
                "Gold next-event text is generally not a verbatim span in the novel; "
                "use the oracle ceiling before interpreting answer coverage."
            ),
            "FC-MH": (
                "Answer coverage does not test whether all multi-hop supporting facts "
                "were retrieved or correctly composed."
            ),
            "all": (
                "The dataset provides no common gold chunk labels for these sources; "
                "all retrieval metrics here are diagnostics, not paper metrics."
            ),
        },
        "sources": source_reports,
    }
    suite_path = output / "suite-report.json"
    with suite_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    checksums_path = _write_sha256sums(output)
    report["suite_report_sha256"] = sha256_file(suite_path)
    report["sha256sums"] = {
        "path": str(checksums_path.relative_to(output)),
        "sha256": sha256_file(checksums_path),
    }
    return report
