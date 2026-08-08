"""Build and verify a compact receipt for pinned LMEB result artifacts.

The large per-query result and prediction files are intentionally ignored by
git.  This module verifies their content hashes and internal metric summaries,
then emits a small, query-free receipt with paired-bootstrap uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path, PureWindowsPath
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "lmeb-actual-receipt-v1"
SPEC_SCHEMA_VERSION = "lmeb-actual-receipt-spec-v1"
STAGES = ("dense", "bm25", "rrf", "retrieved")
METRICS = ("ndcg_at_10", "R_cap_at_10")
MASK_64 = (1 << 64) - 1


class ReceiptError(ValueError):
    """Raised when an artifact does not match the pinned receipt contract."""


class SplitMix64:
    """Small, specified PRNG used to make bootstrap receipts portable."""

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ReceiptError("bootstrap seed must be a non-negative integer")
        self.state = seed & MASK_64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK_64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_64
        return (value ^ (value >> 31)) & MASK_64

    def randbelow(self, upper: int) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise ReceiptError("randbelow upper bound must be a positive integer")
        # Reject the short high tail instead of introducing modulo bias.
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReceiptError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _expect_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ReceiptError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _expect_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReceiptError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ReceiptError(f"{field} is not hexadecimal") from error
    return value.lower()


def _expect_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReceiptError(f"{field} must be finite")
    return result


def _safe_prediction_path(result_path: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReceiptError("prediction artifact path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReceiptError("prediction artifact path must stay inside the result directory")
    root = result_path.parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ReceiptError("prediction artifact path escaped the result directory") from error
    return resolved


def _prediction_query_ids(path: Path, task_id: str) -> set[str]:
    query_ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReceiptError(
                        f"invalid prediction JSON at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ReceiptError(f"prediction line {line_number} is not an object")
                _expect_equal(row.get("task_id"), task_id, "prediction task_id")
                query_id = row.get("query_id")
                if not isinstance(query_id, str) or not query_id:
                    raise ReceiptError(f"prediction line {line_number} has invalid query_id")
                if query_id in query_ids:
                    raise ReceiptError(f"duplicate prediction query_id {query_id!r}")
                rankings = row.get("rankings")
                if not isinstance(rankings, dict) or set(rankings) != set(STAGES):
                    raise ReceiptError(
                        f"prediction {query_id!r} must contain exactly {STAGES!r} stages"
                    )
                query_ids.add(query_id)
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(f"cannot read predictions {path}: {error}") from error
    return query_ids


def _dataset_inputs(result: Mapping[str, Any], task_ids: set[str]) -> dict[str, Any]:
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ReceiptError("result provenance must be an object")
    raw_inputs = provenance.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise ReceiptError("result provenance.inputs must be an object")
    if set(raw_inputs) != task_ids | {"semafuse_artifacts"}:
        raise ReceiptError("provenance.inputs task set does not match result tasks")

    inputs: dict[str, Any] = {}
    for task_id in sorted(task_ids):
        files = raw_inputs.get(task_id)
        if not isinstance(files, dict) or set(files) != {
            "candidates",
            "corpus",
            "qrels",
            "queries",
        }:
            raise ReceiptError(f"unexpected input file set for {task_id}")
        clean_files: dict[str, Any] = {}
        for name in sorted(files):
            descriptor = files[name]
            if not isinstance(descriptor, dict) or set(descriptor) != {"bytes", "sha256"}:
                raise ReceiptError(f"invalid input descriptor for {task_id}/{name}")
            size = descriptor["bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ReceiptError(f"invalid byte count for {task_id}/{name}")
            clean_files[name] = {
                "bytes": size,
                "sha256": _expect_sha(
                    descriptor["sha256"], f"input {task_id}/{name} sha256"
                ),
            }
        inputs[task_id] = clean_files
    return inputs


def _runner_artifacts(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result["provenance"]["inputs"]["semafuse_artifacts"]
    if not isinstance(raw, dict) or not raw:
        raise ReceiptError("semafuse_artifacts must be a non-empty object")
    artifacts: dict[str, Any] = {}
    for source_name, descriptor in raw.items():
        if not isinstance(source_name, str) or not isinstance(descriptor, dict):
            raise ReceiptError("invalid Semafuse artifact descriptor")
        name = Path(source_name).name
        if not name or name in artifacts:
            raise ReceiptError("Semafuse artifact basenames must be unique")
        if set(descriptor) != {"bytes", "sha256"}:
            raise ReceiptError(f"invalid Semafuse artifact descriptor for {name}")
        size = descriptor["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReceiptError(f"invalid byte count for Semafuse artifact {name}")
        artifacts[name] = {
            "bytes": size,
            "sha256": _expect_sha(descriptor["sha256"], f"artifact {name} sha256"),
        }
    return dict(sorted(artifacts.items()))


def _validate_metrics(
    result: Mapping[str, Any], expected_tasks: Mapping[str, Any]
) -> tuple[list[list[tuple[float, ...]]], dict[str, dict[str, float]]]:
    tasks = result.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(expected_tasks):
        raise ReceiptError("result task set does not match the pinned task set")
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != set(STAGES):
        raise ReceiptError(f"result aggregate must contain exactly {STAGES!r}")

    task_matrices: list[list[tuple[float, ...]]] = []
    task_macro: dict[str, dict[str, list[float]]] = {
        stage: {metric: [] for metric in METRICS} for stage in STAGES
    }

    for task_id in sorted(tasks):
        task = tasks[task_id]
        expected = expected_tasks[task_id]
        if not isinstance(task, dict) or not isinstance(expected, dict):
            raise ReceiptError(f"invalid task descriptor for {task_id}")
        expected_query_count = expected.get("queries")
        if (
            isinstance(expected_query_count, bool)
            or not isinstance(expected_query_count, int)
            or expected_query_count <= 0
        ):
            raise ReceiptError(f"invalid expected query count for {task_id}")
        _expect_equal(task.get("task_id"), task_id, f"{task_id} task_id")
        counts = task.get("counts")
        if not isinstance(counts, dict):
            raise ReceiptError(f"missing counts for {task_id}")
        _expect_equal(counts.get("queries"), expected_query_count, f"{task_id} query count")
        stages = task.get("metrics", {}).get("stages")
        if not isinstance(stages, dict) or set(stages) != set(STAGES):
            raise ReceiptError(f"{task_id} must contain exactly {STAGES!r} metric stages")

        query_ids: set[str] | None = None
        per_stage: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            stage_value = stages[stage]
            if not isinstance(stage_value, dict):
                raise ReceiptError(f"invalid stage {stage} for {task_id}")
            per_query = stage_value.get("per_query")
            macro = stage_value.get("macro")
            if not isinstance(per_query, dict) or not isinstance(macro, dict):
                raise ReceiptError(f"missing per_query/macro for {task_id}/{stage}")
            current_ids = set(per_query)
            if len(current_ids) != expected_query_count:
                raise ReceiptError(f"wrong per-query metric count for {task_id}/{stage}")
            if query_ids is None:
                query_ids = current_ids
            elif current_ids != query_ids:
                raise ReceiptError(f"unpaired query sets across stages for {task_id}")
            per_stage[stage] = per_query
            for metric in METRICS:
                values = [
                    _expect_number(row.get(metric), f"{task_id}/{stage}/{query_id}/{metric}")
                    for query_id, row in per_query.items()
                    if isinstance(row, dict)
                ]
                if len(values) != expected_query_count:
                    raise ReceiptError(f"invalid per-query rows for {task_id}/{stage}")
                recomputed = round(fmean(values), 5)
                reported = _expect_number(
                    macro.get(metric), f"{task_id}/{stage}/macro/{metric}"
                )
                if not math.isclose(recomputed, reported, rel_tol=0.0, abs_tol=1e-12):
                    raise ReceiptError(
                        f"metric inconsistency for {task_id}/{stage}/{metric}: "
                        f"recomputed {recomputed}, reported {reported}"
                    )
                task_macro[stage][metric].append(reported)

        assert query_ids is not None
        matrix: list[tuple[float, ...]] = []
        for query_id in sorted(query_ids):
            matrix.append(
                tuple(
                    _expect_number(
                        per_stage[stage][query_id][metric],
                        f"{task_id}/{stage}/{query_id}/{metric}",
                    )
                    for stage in STAGES
                    for metric in METRICS
                )
            )
        task_matrices.append(matrix)

    estimates: dict[str, dict[str, float]] = {stage: {} for stage in STAGES}
    for stage in STAGES:
        aggregate_stage = aggregate[stage]
        if not isinstance(aggregate_stage, dict):
            raise ReceiptError(f"invalid aggregate stage {stage}")
        mean_task = aggregate_stage.get("mean_task")
        if not isinstance(mean_task, dict):
            raise ReceiptError(f"missing aggregate mean_task for {stage}")
        for metric in METRICS:
            recomputed = fmean(task_macro[stage][metric])
            reported = _expect_number(mean_task.get(metric), f"aggregate/{stage}/{metric}")
            if not math.isclose(recomputed, reported, rel_tol=0.0, abs_tol=1e-12):
                raise ReceiptError(
                    f"aggregate inconsistency for {stage}/{metric}: "
                    f"recomputed {recomputed}, reported {reported}"
                )
            estimates[stage][metric] = reported
    return task_matrices, estimates


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ReceiptError("cannot compute a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    task_matrices: Sequence[Sequence[tuple[float, ...]]],
    *,
    seed: int,
    replicates: int,
) -> list[list[float]]:
    """Return bootstrap samples for 4 stages x 2 metrics.

    Queries are paired across every stage.  Each task/subset is resampled
    independently and task means are then equally weighted.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 10_000:
        raise ReceiptError("bootstrap replicates must be an integer >= 10000")
    if not task_matrices or any(not matrix for matrix in task_matrices):
        raise ReceiptError("bootstrap requires at least one query in every task")
    width = len(STAGES) * len(METRICS)
    if any(any(len(row) != width for row in matrix) for matrix in task_matrices):
        raise ReceiptError(f"bootstrap rows must contain exactly {width} values")

    generator = SplitMix64(seed)
    distributions = [[] for _ in range(width)]
    task_count = len(task_matrices)
    for _ in range(replicates):
        across_tasks = [0.0] * width
        for matrix in task_matrices:
            count = len(matrix)
            sums = [0.0] * width
            for _ in range(count):
                row = matrix[generator.randbelow(count)]
                for index, value in enumerate(row):
                    sums[index] += value
            for index in range(width):
                across_tasks[index] += sums[index] / count
        for index in range(width):
            distributions[index].append(across_tasks[index] / task_count)
    return distributions


def _rounded(value: float) -> float:
    return round(value, 8)


def _validate_receipt_safety(value: Any, field: str = "receipt") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ReceiptError(f"{field} contains a non-string key")
            _validate_receipt_safety(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_receipt_safety(child, f"{field}[{index}]")
    elif isinstance(value, str):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ReceiptError(f"{field} contains an absolute path")
        lowered = value.lower()
        if "sk-or-v1-" in lowered or "bearer sk-" in lowered:
            raise ReceiptError(f"{field} contains a credential-like value")


def _summarize_bootstrap(
    estimates: Mapping[str, Mapping[str, float]], distributions: list[list[float]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    sorted_distributions = [sorted(values) for values in distributions]
    for stage_index, stage in enumerate(STAGES):
        metrics[stage] = {}
        for metric_index, metric in enumerate(METRICS):
            index = stage_index * len(METRICS) + metric_index
            metrics[stage][metric] = {
                "estimate": _rounded(estimates[stage][metric]),
                "ci95": [
                    _rounded(_quantile(sorted_distributions[index], 0.025)),
                    _rounded(_quantile(sorted_distributions[index], 0.975)),
                ],
            }

    deltas: dict[str, Any] = {}
    retrieved_offset = STAGES.index("retrieved") * len(METRICS)
    for baseline in ("dense", "bm25", "rrf"):
        baseline_offset = STAGES.index(baseline) * len(METRICS)
        delta_name = f"retrieved_minus_{baseline}"
        deltas[delta_name] = {}
        for metric_index, metric in enumerate(METRICS):
            samples = sorted(
                retrieved - base
                for retrieved, base in zip(
                    distributions[retrieved_offset + metric_index],
                    distributions[baseline_offset + metric_index],
                    strict=True,
                )
            )
            deltas[delta_name][metric] = {
                "estimate": _rounded(
                    estimates["retrieved"][metric] - estimates[baseline][metric]
                ),
                "ci95": [
                    _rounded(_quantile(samples, 0.025)),
                    _rounded(_quantile(samples, 0.975)),
                ],
            }
    return metrics, deltas


def _verify_run(
    spec_path: Path,
    run_name: str,
    run_spec: Mapping[str, Any],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    source = run_spec.get("result")
    if not isinstance(source, str) or not source:
        raise ReceiptError(f"{run_name}.result must be a non-empty relative path")
    source_path = Path(source)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ReceiptError(f"{run_name}.result must not be absolute or escape its spec")
    result_path = (spec_path.parent / source_path).resolve()
    expected_result_sha = _expect_sha(
        run_spec.get("result_sha256"), f"{run_name}.result_sha256"
    )
    actual_result_sha = _sha256(result_path)
    _expect_equal(actual_result_sha, expected_result_sha, f"{run_name} result sha256")
    result = _read_json(result_path)

    _expect_equal(result.get("schema_version"), "lmeb-result-v1", "result schema_version")
    _expect_equal(result.get("status"), "complete", "result status")
    _expect_equal(result.get("protocol"), "common-policy", "result protocol")
    _expect_equal(result.get("instruction_setting"), "without", "instruction setting")
    if 10 not in result.get("cutoffs", []):
        raise ReceiptError("result cutoffs do not include 10")
    fidelity = result.get("fidelity")
    if not isinstance(fidelity, dict) or "cold per query" not in str(
        fidelity.get("candidate_filter", "")
    ):
        raise ReceiptError("result is not labeled as a cold-per-query policy deviation")
    for field in ("name", "run_id", "run_fingerprint"):
        _expect_equal(result.get(field), run_spec.get(field), f"{run_name}.{field}")

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ReceiptError("result provenance must be an object")
    fingerprint_payload = provenance.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ReceiptError("result fingerprint_payload must be an object")
    provenance_fields = {
        "manifest_sha256": provenance.get("manifest_sha256"),
        "final_harness_sha256": provenance.get("harness_source_sha256"),
        "fingerprint_harness_sha256": fingerprint_payload.get("harness_source_sha256"),
    }
    for field, value in provenance_fields.items():
        expected = _expect_sha(run_spec.get(field), f"{run_name}.{field}")
        _expect_equal(value, expected, f"{run_name}.{field}")

    expected_tasks = run_spec.get("tasks")
    if not isinstance(expected_tasks, dict) or not expected_tasks:
        raise ReceiptError(f"{run_name}.tasks must be a non-empty object")
    task_matrices, estimates = _validate_metrics(result, expected_tasks)

    prediction_hashes: dict[str, str] = {}
    for task_id in sorted(expected_tasks):
        task = result["tasks"][task_id]
        artifacts = task.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ReceiptError(f"missing artifacts for {task_id}")
        expected_prediction_sha = _expect_sha(
            expected_tasks[task_id].get("predictions_sha256"),
            f"{task_id}.predictions_sha256",
        )
        _expect_equal(
            artifacts.get("predictions_sha256"),
            expected_prediction_sha,
            f"{task_id} embedded predictions sha256",
        )
        prediction_path = _safe_prediction_path(result_path, artifacts.get("predictions"))
        actual_prediction_sha = _sha256(prediction_path)
        _expect_equal(
            actual_prediction_sha,
            expected_prediction_sha,
            f"{task_id} predictions sha256",
        )
        prediction_ids = _prediction_query_ids(prediction_path, task_id)
        metric_ids = set(task["metrics"]["stages"][STAGES[0]]["per_query"])
        _expect_equal(prediction_ids, metric_ids, f"{task_id} prediction/metric query set")
        prediction_hashes[task_id] = actual_prediction_sha

    dataset_inputs = _dataset_inputs(result, set(expected_tasks))
    runner_artifacts = _runner_artifacts(result)
    expected_inputs = run_spec.get("dataset_inputs")
    expected_runner_artifacts = run_spec.get("runner_artifacts")
    _expect_equal(dataset_inputs, expected_inputs, f"{run_name} dataset input hashes")
    _expect_equal(runner_artifacts, expected_runner_artifacts, f"{run_name} runner hashes")

    distributions = paired_bootstrap(task_matrices, seed=seed, replicates=replicates)
    metrics, deltas = _summarize_bootstrap(estimates, distributions)
    total_queries = sum(int(task["queries"]) for task in expected_tasks.values())
    cost = run_spec.get("cost_usd")
    if not isinstance(cost, str):
        raise ReceiptError(f"{run_name}.cost_usd must be an exact decimal string")
    try:
        parsed_cost = float(cost)
    except ValueError as error:
        raise ReceiptError(f"{run_name}.cost_usd is not numeric") from error
    if not math.isfinite(parsed_cost) or parsed_cost < 0:
        raise ReceiptError(f"{run_name}.cost_usd must be finite and non-negative")

    receipt_run: dict[str, Any] = {
        "dataset": run_spec.get("dataset"),
        "aggregation": run_spec.get("aggregation"),
        "queries": total_queries,
        "metrics": metrics,
        "paired_deltas": deltas,
        "cost": {
            "usd": cost,
            "measurement": "OpenRouter account usage difference",
        },
        "integrity": {
            "result_sha256": actual_result_sha,
            "prediction_sha256": prediction_hashes,
            "dataset_inputs": dataset_inputs,
            "runner_artifacts": runner_artifacts,
            "manifest_sha256": provenance_fields["manifest_sha256"],
            "run_id": result["run_id"],
            "run_fingerprint": result["run_fingerprint"],
            "fingerprint_harness_sha256": provenance_fields[
                "fingerprint_harness_sha256"
            ],
            "final_harness_sha256": provenance_fields["final_harness_sha256"],
        },
    }
    notes = run_spec.get("notes")
    if notes is not None:
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            raise ReceiptError(f"{run_name}.notes must contain only strings")
        receipt_run["notes"] = notes
    return receipt_run


def build_receipt(spec_path: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    _expect_equal(spec.get("schema_version"), SPEC_SCHEMA_VERSION, "spec schema_version")
    bootstrap = spec.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ReceiptError("spec bootstrap must be an object")
    seed = bootstrap.get("seed")
    replicates = bootstrap.get("replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReceiptError("bootstrap seed must be an integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ReceiptError("bootstrap replicates must be an integer")
    runs = spec.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ReceiptError("spec runs must be a non-empty object")
    receipt_runs = {
        name: _verify_run(
            spec_path,
            name,
            run_spec,
            seed=seed,
            replicates=replicates,
        )
        for name, run_spec in sorted(runs.items())
        if isinstance(run_spec, dict)
    }
    if len(receipt_runs) != len(runs):
        raise ReceiptError("every run spec must be an object")

    models = spec.get("models")
    if not isinstance(models, dict):
        raise ReceiptError("spec models must be an object")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "label": "candidate-cold per query common-policy deviation",
            "harness_protocol": "common-policy",
            "instruction_setting": "without",
            "paper_reproduction": False,
        },
        "models": models,
        "bootstrap": {
            "method": (
                "query-paired percentile bootstrap; independently resample queries "
                "within each task/subset, then equally weight task/subset means"
            ),
            "prng": "SplitMix64 with unbiased rejection sampling",
            "seed": seed,
            "replicates": replicates,
            "ci": 0.95,
            "quantile": "linear interpolation at 2.5% and 97.5%",
        },
        "runs": receipt_runs,
        "caveats": [
            (
                "These legacy artifacts record retrieval and processing latency only; "
                "their latency excludes cold per-query ingest, embedding/index construction, "
                "and total pipeline wall time. Scores and hashes remain valid."
            ),
            (
                "OpenRouter model IDs are recorded, but provider model revisions are not "
                "exposed or pinned by these runs."
            ),
            (
                "Hybrid/reranked candidate-cold scores are diagnostics and must not be "
                "presented as LMEB paper-reproduction embedding scores."
            ),
        ],
    }
    _validate_receipt_safety(receipt)
    return receipt


def _write_receipt(receipt: Mapping[str, Any], output: Path) -> None:
    serialized = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    # The output is deliberately whitelist-built, but reject accidental local-path leakage.
    if os.path.isabs(str(output)) and str(output.parent) in serialized:
        raise ReceiptError("receipt unexpectedly contains its absolute output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    except OSError as error:
        raise ReceiptError(f"cannot write receipt {output}: {error}") from error


def _verify_existing(expected: Mapping[str, Any], receipt_path: Path) -> None:
    actual = _read_json(receipt_path)
    if actual != expected:
        raise ReceiptError(
            f"{receipt_path} is stale; regenerate it from the pinned source artifacts"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="pinned run specification")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="write a canonical receipt")
    action.add_argument("--verify", type=Path, help="verify an existing canonical receipt")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(args.spec.resolve())
        if args.output is not None:
            _write_receipt(receipt, args.output)
            print(args.output)
        else:
            _verify_existing(receipt, args.verify)
            print(f"verified {args.verify}")
    except ReceiptError as error:
        print(f"receipt error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
