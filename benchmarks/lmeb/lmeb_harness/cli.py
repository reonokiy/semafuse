"""Command-line interface for the dependency-free LMEB harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .dataset import load_task
from .manifest import load_manifest
from .model import HarnessError
from .protocol import export_ranker_input
from .runner import run_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmeb-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate manifest and all local data")
    validate.add_argument("--manifest", required=True)
    export = subparsers.add_parser("export", help="export the ranker input JSONL protocol")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    run = subparsers.add_parser("run", help="rank, score, aggregate, and write provenance")
    run.add_argument("--manifest", required=True)
    run.add_argument("--ranker", choices=("bm25", "external", "semafuse-jsonl"))
    run.add_argument("--output-dir")
    resume = run.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", dest="resume")
    resume.add_argument("--no-resume", action="store_false", dest="resume")
    run.set_defaults(resume=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            tasks = [load_task(task) for task in manifest.tasks]
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "tasks": len(tasks),
                        "queries": sum(len(task.queries) for task in tasks),
                        "corpus_records": sum(len(task.corpus) for task in tasks),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "export":
            export_ranker_input(
                (load_task(task) for task in manifest.tasks),
                Path(args.output),
                include_instruction=manifest.run.instruction_setting == "with",
            )
            print(str(Path(args.output).resolve()))
            return 0
        result, resumed = run_manifest(
            manifest,
            ranker_override=args.ranker,
            output_override=Path(args.output_dir) if args.output_dir else None,
            resume_override=args.resume,
        )
        print(json.dumps({"result": str(result), "resumed": resumed}, sort_keys=True))
        return 0
    except (HarnessError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
