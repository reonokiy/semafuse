"""Command-line entry point for adaptation, splitting, running, and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .adapters import load_locomo, load_longmemeval
from .config import endpoint_client, load_config, validate_reproduction_config
from .evaluation import (
    ApproximateTokenCounter,
    EndpointAnswerer,
    EndpointBinaryJudge,
    EvaluationRunner,
    TransformersTokenCounter,
)
from .methods import (
    LazyMemMethod,
    OracleSessionBaseline,
    OracleTurnBaseline,
    RagBaseline,
)
from .metrics import summarize
from .official_evaluators import (
    EvaluationScore,
    LongMemEvalOfficialLJ,
    external_scores,
    locomo_official_f1,
)
from .processing import ConcurrentMemoryProcessor
from .resume import JsonlJournal, read_jsonl
from .retrieval import (
    BM25Ranker,
    CrossEncoderReranker,
    HybridRetriever,
    RankBM25Ranker,
    SentenceTransformerRanker,
)
from .split import sha256_file, stratified_split, write_split_manifest
from .types import Question


def _write_json(path: str | Path | None, value: Any) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


def _load_questions(args: argparse.Namespace) -> list[Question]:
    if args.dataset == "longmemeval":
        return load_longmemeval(args.input)
    return load_locomo(
        args.input,
        acknowledge_noncommercial=args.acknowledge_locomo_noncommercial,
        require_paper_count=not args.allow_nonpaper_count,
    )


def _filter_split(
    questions: list[Question],
    manifest_path: str | None,
    split_name: str,
    *,
    source_path: str | Path | None = None,
    expected_counts: dict[str, int] | None = None,
) -> list[Question]:
    if not manifest_path:
        if expected_counts is not None:
            raise ValueError(
                "LongMemEval paper runs require --split-manifest; refusing to "
                f"label all questions as {split_name!r}"
            )
        return questions
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("split manifest must use schema_version 1")
    if source_path is not None:
        source_sha256 = manifest.get("source_sha256")
        actual_source_sha256 = sha256_file(source_path)
        if source_sha256 != actual_source_sha256:
            raise ValueError(
                "split manifest source_sha256 does not match the benchmark input"
            )
    split_ids = manifest.get("ids")
    split_order = ("train", "validation", "test")
    if not isinstance(split_ids, dict) or set(split_ids) != set(split_order):
        raise ValueError("split manifest ids must contain train, validation, and test")
    normalized: dict[str, list[str]] = {}
    for name in split_order:
        values = split_ids[name]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"split manifest ids.{name} must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"split manifest ids.{name} contains duplicate ids")
        normalized[name] = values
    id_sets = {name: set(values) for name, values in normalized.items()}
    if any(
        id_sets[left] & id_sets[right]
        for index, left in enumerate(split_order)
        for right in split_order[index + 1 :]
    ):
        raise ValueError("split manifest partitions must be disjoint")
    question_ids = {question.id for question in questions}
    if len(question_ids) != len(questions):
        raise ValueError("benchmark question ids must be unique")
    if set.union(*id_sets.values()) != question_ids:
        raise ValueError("split manifest ids must exactly cover the benchmark input")
    if expected_counts is not None:
        actual_counts = {name: len(normalized[name]) for name in split_order}
        if actual_counts != expected_counts:
            raise ValueError(
                f"split manifest counts {actual_counts} do not match {expected_counts}"
            )
    canonical_ids = json.dumps(
        {name: sorted(normalized[name]) for name in split_order},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if manifest.get("ids_sha256") != hashlib.sha256(canonical_ids).hexdigest():
        raise ValueError("split manifest ids_sha256 does not match its ids")
    selected = id_sets[split_name]
    filtered = [question for question in questions if question.id in selected]
    return filtered


def _longmemeval_split_counts(
    dataset: str, config: dict[str, Any]
) -> dict[str, int] | None:
    if dataset != "longmemeval":
        return None
    settings = config["dataset"]["longmemeval"]
    return {
        name: int(settings[name]) for name in ("train", "validation", "test")
    }


def _build_retriever(config: dict[str, Any]) -> HybridRetriever:
    retrieval = config["retrieval"]
    sparse_backend = str(retrieval.get("sparse_backend", "rank-bm25"))
    if sparse_backend == "rank-bm25":
        sparse = RankBM25Ranker()
    elif sparse_backend == "stdlib-bm25":
        sparse = BM25Ranker(
            k1=float(retrieval.get("bm25_k1", 1.5)),
            b=float(retrieval.get("bm25_b", 0.75)),
        )
    else:
        raise ValueError("sparse_backend must be 'rank-bm25' or 'stdlib-bm25'")
    return HybridRetriever(
        dense=SentenceTransformerRanker(
            str(retrieval["dense_model"]),
            query_prompt=(
                str(retrieval["dense_query_prompt"])
                if retrieval.get("dense_query_prompt")
                else None
            ),
        ),
        sparse=sparse,
        reranker=CrossEncoderReranker(str(retrieval["reranker_model"])),
        channel_limit=int(retrieval.get("channel_limit", 100)),
        fused_limit=int(retrieval.get("fused_limit", 100)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
    )


def _run_identity(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    def endpoint_identity(section: str) -> dict[str, str | None]:
        value = config[section]

        def resolved(name: str) -> str | None:
            direct = value.get(name)
            if isinstance(direct, str) and direct:
                return direct
            environment_name = value.get(f"{name}_env")
            if isinstance(environment_name, str):
                return os.environ.get(environment_name)
            return None

        return {
            "base_url": resolved("base_url"),
            "model": resolved("model"),
            "api_key_env": (
                str(value["api_key_env"]) if value.get("api_key_env") else None
            ),
        }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "dataset_sha256": sha256_file(args.input),
        "config_sha256": sha256_file(args.config),
        "split_manifest_sha256": (
            sha256_file(args.split_manifest) if args.split_manifest else None
        ),
        "split": args.split,
        "method": args.method,
        "limit": args.limit,
        "run_kind": "latency" if args.command == "run-latency" else "evaluation",
        "generic_judge": getattr(args, "generic_judge", False),
        "approximate_tokens": args.approximate_tokens,
        "retrieval": config.get("retrieval", {}),
        "window": config.get("window", {}),
        "tokenizer_model": config.get("metrics", {}).get("tokenizer_model"),
        "endpoints": {
            section: endpoint_identity(section)
            for section in ("mpm", "answer", "judge")
        },
    }
    encoded = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), metadata


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_reproduction_config(config)
    questions = _filter_split(
        _load_questions(args),
        args.split_manifest,
        args.split,
        source_path=args.input,
        expected_counts=_longmemeval_split_counts(args.dataset, config),
    )
    run_id, run_metadata = _run_identity(args, config)
    if args.limit is not None:
        questions = questions[: args.limit]

    retriever = None
    if args.method in ("lazymem", "rag20", "rag50"):
        retriever = _build_retriever(config)
    if args.method == "lazymem":
        window = config["window"]
        mpm_config = config["mpm"]
        processor = ConcurrentMemoryProcessor(
            endpoint_client(mpm_config),
            max_workers=int(mpm_config.get("concurrency", 64)),
            max_tokens=int(mpm_config.get("max_tokens", 3072)),
            temperature=float(mpm_config.get("temperature", 0.1)),
            top_p=float(mpm_config.get("top_p", 0.95)),
            enable_thinking=mpm_config.get("enable_thinking", True),
            extra_body=dict(mpm_config.get("extra_body", {})),
        )
        method = LazyMemMethod(
            retriever=retriever,
            processor=processor,
            retrieval_limit=int(config["retrieval"].get("candidate_limit", 50)),
            radius=int(window.get("radius", 2)),
            max_messages=int(window.get("max_messages", 8)),
            stride=int(window.get("stride", 7)),
            merge_rule=str(window.get("merge_rule", "paper")),
        )
    elif args.method in ("rag20", "rag50"):
        method = RagBaseline(retriever, k=int(args.method[3:]))
    elif args.method == "oracle-turn":
        method = OracleTurnBaseline()
    else:
        method = OracleSessionBaseline()

    token_counter = (
        ApproximateTokenCounter()
        if args.approximate_tokens
        else TransformersTokenCounter(str(config["metrics"]["tokenizer_model"]))
    )
    answer_config = config["answer"]
    answerer = EndpointAnswerer(
        endpoint_client(answer_config),
        max_tokens=int(answer_config.get("max_tokens", 512)),
        temperature=float(answer_config.get("temperature", 0.0)),
        enable_thinking=answer_config.get("enable_thinking", True),
        extra_body=dict(answer_config.get("extra_body", {})),
    )
    latency_protocol = args.command == "run-latency"
    judge = (
        EndpointBinaryJudge(endpoint_client(config["judge"]))
        if getattr(args, "generic_judge", False)
        else None
    )
    result = EvaluationRunner(
        method=method,
        answerer=answerer,
        judge=judge,
        journal=JsonlJournal(args.output),
        token_counter=token_counter,
        run_id=run_id,
        run_metadata=run_metadata,
    ).run(
        questions,
        warmup_runs=(
            int(config["metrics"].get("latency_warmup_runs", 3))
            if latency_protocol
            else 0
        ),
        measured_runs=(
            int(config["metrics"].get("latency_measured_runs", 3))
            if latency_protocol
            else 1
        ),
    )
    _write_json(None, result)


def _evaluate(args: argparse.Namespace) -> None:
    questions = _load_questions(args)
    question_by_id = {question.id: question for question in questions}
    rows = list(read_jsonl(args.results))
    if not rows:
        raise ValueError("results JSONL is empty")
    row_ids = [str(row["question_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("results JSONL contains duplicate question_id values")
    missing_questions = set(row_ids).difference(question_by_id)
    if missing_questions:
        raise ValueError(
            f"results contain IDs absent from the dataset: {sorted(missing_questions)}"
        )

    evaluator = None
    imported: dict[str, float] | None = None
    imported_metadata: dict[str, Any] = {}
    if args.evaluator == "longmemeval-official-lj":
        if args.dataset != "longmemeval":
            raise ValueError("longmemeval-official-lj requires --dataset longmemeval")
        if not args.config:
            raise ValueError("longmemeval-official-lj requires --config")
        config = load_config(args.config)
        evaluator = LongMemEvalOfficialLJ(endpoint_client(config["judge"]))
    elif args.evaluator == "locomo-official-f1":
        if args.dataset != "locomo":
            raise ValueError("locomo-official-f1 requires --dataset locomo")
    else:
        if not args.scores or not args.metric or not args.evaluator_id:
            raise ValueError(
                "external-scores requires --scores, --metric, and --evaluator-id"
            )
        imported = external_scores(args.scores)
        expected = set(row_ids)
        if set(imported) != expected:
            raise ValueError(
                "external scores must contain exactly the result question IDs; "
                f"missing={sorted(expected.difference(imported))}, "
                f"extra={sorted(set(imported).difference(expected))}"
            )
        imported_metadata = {
            "external_evaluator_id": args.evaluator_id,
            "external_scores_sha256": sha256_file(args.scores),
            "paper_comparable": False,
        }

    output = Path(args.output)
    if output.resolve() == Path(args.results).resolve():
        raise ValueError("evaluation output must differ from the answer-results input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                if "evaluation_score" in row:
                    raise ValueError(
                        "results already contain evaluation_score; evaluate the original answers"
                    )
                question_id = str(row["question_id"])
                question = question_by_id[question_id]
                prediction = str(row["prediction"])
                if evaluator is not None:
                    result = evaluator.evaluate(question, prediction)
                elif args.evaluator == "locomo-official-f1":
                    result = locomo_official_f1(question, prediction)
                else:
                    assert imported is not None
                    result = EvaluationScore(
                        score=imported[question_id],
                        metric=args.metric,
                        raw=None,
                        latency_s=0.0,
                        metadata=imported_metadata,
                    )
                evaluated = dict(row)
                evaluated.update(
                    {
                        "evaluation_score": result.score,
                        "evaluation_metric": result.metric,
                        "evaluation_raw": result.raw,
                        "evaluation_latency_s": result.latency_s,
                        "evaluator_metadata": dict(result.metadata),
                    }
                )
                handle.write(
                    json.dumps(
                        evaluated,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, output)
    _write_json(
        None,
        {
            "evaluated": len(rows),
            "evaluator": args.evaluator,
            "output": str(output),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semafuse-lazymem-bench",
        description="Clean-room LazyMem paper-v2 benchmark harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split-longmemeval")
    split.add_argument("--input", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)

    adapt = subparsers.add_parser("adapt")
    adapt.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    adapt.add_argument("--input", required=True)
    adapt.add_argument("--output", required=True)
    adapt.add_argument("--acknowledge-locomo-noncommercial", action="store_true")
    adapt.add_argument("--allow-nonpaper-count", action="store_true", help=argparse.SUPPRESS)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--input", required=True)
    metrics.add_argument("--output")

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--evaluator",
        choices=(
            "longmemeval-official-lj",
            "locomo-official-f1",
            "external-scores",
        ),
        required=True,
    )
    evaluate.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    evaluate.add_argument("--input", required=True, help="raw benchmark dataset")
    evaluate.add_argument("--results", required=True, help="answer-results JSONL from run")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--scores", help="strict question_id/score JSONL")
    evaluate.add_argument("--metric", help="metric identifier for external scores")
    evaluate.add_argument("--evaluator-id", help="immutable external evaluator identifier")
    evaluate.add_argument("--acknowledge-locomo-noncommercial", action="store_true")
    evaluate.add_argument("--allow-nonpaper-count", action="store_true", help=argparse.SUPPRESS)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    run.add_argument("--input", required=True)
    run.add_argument(
        "--method",
        choices=("lazymem", "rag20", "rag50", "oracle-turn", "oracle-session"),
        required=True,
    )
    run.add_argument("--output", required=True)
    run.add_argument("--split-manifest")
    run.add_argument("--split", choices=("train", "validation", "test"), default="test")
    run.add_argument("--limit", type=int)
    run.add_argument("--generic-judge", action="store_true")
    run.add_argument("--approximate-tokens", action="store_true")
    run.add_argument("--acknowledge-locomo-noncommercial", action="store_true")
    run.add_argument("--allow-nonpaper-count", action="store_true", help=argparse.SUPPRESS)

    latency_run = subparsers.add_parser(
        "run-latency",
        help="discard configured full-run warmups and record configured measured repeats",
    )
    latency_run.add_argument("--config", required=True)
    latency_run.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    latency_run.add_argument("--input", required=True)
    latency_run.add_argument(
        "--method",
        choices=("lazymem", "rag20", "rag50", "oracle-turn", "oracle-session"),
        required=True,
    )
    latency_run.add_argument("--output", required=True)
    latency_run.add_argument("--split-manifest")
    latency_run.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    latency_run.add_argument("--limit", type=int)
    latency_run.add_argument("--approximate-tokens", action="store_true")
    latency_run.add_argument("--acknowledge-locomo-noncommercial", action="store_true")
    latency_run.add_argument(
        "--allow-nonpaper-count", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "split-longmemeval":
        questions = load_longmemeval(args.input)
        manifest = stratified_split(
            questions, seed=args.seed, source_sha256=sha256_file(args.input)
        )
        write_split_manifest(args.output, manifest)
    elif args.command == "adapt":
        questions = _load_questions(args)
        with Path(args.output).open("w", encoding="utf-8") as handle:
            for question in questions:
                handle.write(
                    json.dumps(
                        question.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
    elif args.command == "metrics":
        _write_json(args.output, summarize(read_jsonl(args.input)))
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "validate-config":
        config = load_config(args.config)
        validate_reproduction_config(config)
        _write_json(None, {"valid": True, "config": str(Path(args.config).resolve())})
    else:
        _run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
