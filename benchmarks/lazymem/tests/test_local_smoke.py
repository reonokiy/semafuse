from __future__ import annotations

import unittest
from pathlib import Path

from semafuse_lazymem_bench.adapters.longmemeval import load_longmemeval
from semafuse_lazymem_bench.cli import _filter_split, _run, build_parser


FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS = Path(__file__).parents[1] / "configs"


class LocalSmokeModeTests(unittest.TestCase):
    def test_fixture_input_requires_an_explicit_smoke_gate(self) -> None:
        questions = load_longmemeval(FIXTURES / "longmemeval.json")
        paper_counts = {"train": 360, "validation": 40, "test": 100}

        with self.assertRaisesRegex(ValueError, "require --split-manifest"):
            _filter_split(
                questions,
                None,
                "test",
                expected_counts=paper_counts,
            )

        selected = _filter_split(
            questions,
            None,
            "test",
            expected_counts=paper_counts,
            allow_nonpaper_count=True,
        )
        self.assertEqual([question.id for question in selected], ["q-user-1", "q-update-1_abs"])

    def test_smoke_gate_is_visible_and_rejects_paper_profiles(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--config",
                str(CONFIGS / "paper_v2.public-repro.toml"),
                "--dataset",
                "longmemeval",
                "--input",
                str(FIXTURES / "longmemeval.json"),
                "--method",
                "oracle-turn",
                "--output",
                "unused.jsonl",
                "--allow-nonpaper-count",
            ]
        )
        self.assertTrue(args.allow_nonpaper_count)
        with self.assertRaisesRegex(ValueError, "limited to custom-ablation"):
            _run(args)


if __name__ == "__main__":
    unittest.main()
