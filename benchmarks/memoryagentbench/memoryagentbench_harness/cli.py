from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

from .bm25 import run_bm25
from .model import HarnessError, load_manifest
from .protocol import prepare_source
from .scoring import score_predictions, write_report
from .source_verification import download_and_verify_files
from .suite import run_retrieval_suite


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="semafuse-memoryagentbench")
    parser.add_argument("--manifest", help="pinned source manifest JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="prepare public input events and evaluator-only labels"
    )
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument(
        "--fixture",
        help="local JSON/JSONL rows for tests only; disables paper-comparable status",
    )
    prepare.add_argument("--max-contexts", type=int)
    prepare.add_argument("--chunk-size", type=int)
    prepare.add_argument(
        "--timestamp",
        default="1970-01-01 00:00:00",
        help="fixed timestamp inserted into upstream memory prompts",
    )

    bm25 = subparsers.add_parser(
        "bm25", help="run the retrieval-only rank_bm25 diagnostic"
    )
    bm25.add_argument("--inputs", required=True)
    bm25.add_argument("--output", required=True)

    score = subparsers.add_parser(
        "score", help="score answer predictions and/or retrieval diagnostics"
    )
    score.add_argument("--inputs", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--allow-partial", action="store_true")

    verify = subparsers.add_parser(
        "verify-data", help="download pinned Hub files and verify size/SHA-256"
    )
    verify.add_argument("--cache-dir", required=True)
    verify.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="split to verify; repeat as needed (default: all)",
    )
    verify.add_argument("--output", help="optional JSON verification report")

    suite = subparsers.add_parser(
        "run-suite", help="run the complete pinned retrieval-only BM25 suite"
    )
    suite.add_argument("--output-dir", required=True)
    suite.add_argument("--cache-dir", required=True)
    suite.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="source to run; repeat as needed (default: all five)",
    )

    inspect = subparsers.add_parser(
        "inspect-manifest", help="print pinned profiles without downloading data"
    )
    inspect.add_argument("--json", action="store_true")
    return parser


def _prepare(args: Namespace) -> int:
    manifest = load_manifest(args.manifest)
    try:
        profile = manifest.profiles[args.source]
    except KeyError as error:
        choices = ", ".join(sorted(manifest.profiles))
        raise HarnessError(f"unknown source {args.source!r}; choose one of: {choices}") from error
    run = prepare_source(
        manifest,
        profile,
        args.output_dir,
        fixture_path=args.fixture,
        timestamp=args.timestamp,
        max_contexts=args.max_contexts,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(run, ensure_ascii=False, sort_keys=True))
    return 0


def _inspect(args: Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if args.json:
        print(json.dumps(manifest.raw, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"dataset: {manifest.dataset_repo_id}@{manifest.dataset_revision}")
    for name, profile in sorted(manifest.profiles.items()):
        print(
            f"{name}: split={profile.split} paper={profile.paper_name} "
            f"contexts={profile.expected_contexts} queries={profile.expected_queries} "
            f"chunk={profile.chunk_size_tokens} top_k={manifest.top_k}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "bm25":
            count = run_bm25(args.inputs, args.output)
            print(f"wrote {count} predictions to {Path(args.output)}")
            return 0
        if args.command == "score":
            report = score_predictions(
                args.inputs,
                args.labels,
                args.predictions,
                allow_partial=args.allow_partial,
            )
            write_report(report, args.output)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "verify-data":
            manifest = load_manifest(args.manifest)
            report = download_and_verify_files(
                manifest, args.cache_dir, splits=args.splits
            )
            if args.output:
                with Path(args.output).open("w", encoding="utf-8") as handle:
                    json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "run-suite":
            manifest = load_manifest(args.manifest)
            report = run_retrieval_suite(
                manifest,
                args.output_dir,
                args.cache_dir,
                sources=args.sources,
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "inspect-manifest":
            return _inspect(args)
        raise HarnessError(f"unsupported command: {args.command}")
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
