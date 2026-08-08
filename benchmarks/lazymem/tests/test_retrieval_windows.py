from __future__ import annotations

import json
import unittest

from semafuse_lazymem_bench.decisions import (
    DecisionFormatError,
    deduplicate_decisions,
    parse_edit_decisions,
)
from semafuse_lazymem_bench.processing import model_window_payload
from semafuse_lazymem_bench.retrieval import (
    BM25Ranker,
    RankedMessage,
    reciprocal_rank_fusion,
)
from semafuse_lazymem_bench.types import Message
from semafuse_lazymem_bench.windows import build_windows


def messages(count: int) -> tuple[Message, ...]:
    return tuple(
        Message(
            id=f"m{index}",
            index=index,
            role="user" if index % 2 == 0 else "assistant",
            content=f"message {index}",
            timestamp=f"t{index}",
            session_id="s",
        )
        for index in range(count)
    )


class RetrievalTests(unittest.TestCase):
    def test_rrf_deduplicates_each_channel_and_is_deterministic(self) -> None:
        fused = reciprocal_rank_fusion(
            [["b", "a", "a"], ["a", "c"]], k=60
        )
        self.assertEqual([item[0] for item in fused], ["a", "b", "c"])
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61)

    def test_rrf_ties_use_corpus_order_not_label_bearing_ids(self) -> None:
        fused = reciprocal_rank_fusion(
            [["z"], ["answer_evidence"]],
            tie_break_order={"z": 0, "answer_evidence": 1},
        )
        self.assertEqual([item[0] for item in fused], ["z", "answer_evidence"])

        with self.assertRaises(ValueError):
            reciprocal_rank_fusion(
                [["z"], ["answer_evidence"]], tie_break_order={"z": 0}
            )

    def test_dependency_free_bm25(self) -> None:
        history = list(messages(3))
        history[1] = Message("m1", 1, "assistant", "rare apple orchard")
        ranked = BM25Ranker().rank("apple", history, 2)
        self.assertEqual(ranked[0].message.id, "m1")


class WindowAndDecisionTests(unittest.TestCase):
    def test_paper_and_code_merge_boundary_are_explicit(self) -> None:
        history = messages(10)
        retrieved = [
            RankedMessage(history[0], 1.0, 1),
            RankedMessage(history[5], 0.9, 2),
        ]
        paper = build_windows(history, retrieved, radius=2, merge_rule="paper")
        code = build_windows(history, retrieved, radius=2, merge_rule="code")
        self.assertEqual(len(paper), 2)
        self.assertEqual(len(code), 1)
        self.assertEqual((code[0].start, code[0].stop), (0, 8))

    def test_long_merged_span_splits_eight_stride_seven(self) -> None:
        history = messages(13)
        retrieved = [
            RankedMessage(history[index], 1.0, rank)
            for rank, index in enumerate((2, 6, 10), start=1)
        ]
        windows = build_windows(history, retrieved, radius=2, max_messages=8, stride=7)
        self.assertEqual([(window.start, window.stop) for window in windows], [(0, 8), (7, 13)])
        self.assertEqual(windows[0].messages[-1].id, windows[1].messages[0].id)

    def test_window_sections_and_retrieval_metadata_match_paper_payload(self) -> None:
        history = tuple(
            Message(
                id=f"m{index}",
                index=index,
                role="user",
                content=f"message {index}",
                timestamp=f"t{index}",
                session_id="s",
                metadata={"global_message_index": index + 10},
            )
            for index in range(7)
        )
        windows = build_windows(
            history,
            [
                RankedMessage(history[4], 0.8, 2),
                RankedMessage(history[2], 0.9, 1),
            ],
            radius=1,
        )
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual([item.id for item in window.previous_bridge], ["m1"])
        self.assertEqual(
            [item.id for item in window.core_messages], ["m2", "m3", "m4"]
        )
        self.assertEqual([item.id for item in window.next_bridge], ["m5"])
        self.assertEqual(window.retrieval_ranks, (("m2", 1), ("m4", 2)))

        payload = model_window_payload(window)
        self.assertEqual(
            list(payload), ["previous_bridge", "core_messages", "next_bridge"]
        )
        self.assertEqual(payload["previous_bridge"][0]["position"], "previous_bridge")
        self.assertEqual(payload["core_messages"][1]["position"], "core")
        self.assertEqual(payload["next_bridge"][0]["position"], "next_bridge")
        self.assertEqual(
            [item["global_message_index"] for item in payload["core_messages"]],
            [12, 13, 14],
        )
        self.assertEqual(
            [item["retrieval_rank"] for item in payload["core_messages"]],
            [1, None, 2],
        )
        self.assertEqual(
            list(payload["core_messages"][0]),
            [
                "position",
                "role",
                "timestamp",
                "global_message_index",
                "retrieval_rank",
                "content",
            ],
        )

    def test_windows_never_restore_neighbors_across_session_boundaries(self) -> None:
        history = [
            Message(f"a{index}", index, "user", f"a{index}", session_id="a")
            for index in range(3)
        ] + [
            Message(f"b{index}", index + 3, "user", f"b{index}", session_id="b")
            for index in range(3)
        ]
        windows = build_windows(history, [history[2], history[3]], radius=2)
        self.assertEqual(
            [[message.id for message in window.messages] for window in windows],
            [["a0", "a1", "a2"], ["b0", "b1", "b2"]],
        )

    def test_strict_schema_and_keep_wins_cross_window(self) -> None:
        history = messages(13)
        retrieved = [
            RankedMessage(history[index], 1.0, rank)
            for rank, index in enumerate((2, 6, 10), start=1)
        ]
        windows = build_windows(history, retrieved)

        def output(window, keep_first: bool) -> str:
            return json.dumps(
                [
                    {
                        "op": "KEEP" if keep_first and index == 0 else "DROP",
                        "compressed_content": "kept overlap" if keep_first and index == 0 else "",
                        "reason": "relevant" if keep_first and index == 0 else "irrelevant",
                    }
                    for index, _ in enumerate(window.messages)
                ]
            )

        first = parse_edit_decisions(output(windows[0], False), windows[0])
        second = parse_edit_decisions(output(windows[1], True), windows[1])
        selected = deduplicate_decisions([second, first])
        self.assertEqual([(item.message.id, item.compressed_content) for item in selected], [("m7", "kept overlap")])

        invalid = json.dumps(
            [{"op": "DROP", "compressed_content": "", "reason": "x", "extra": 1}]
            * len(windows[0].messages)
        )
        with self.assertRaises(DecisionFormatError):
            parse_edit_decisions(invalid, windows[0])

        prefixed = parse_edit_decisions(
            f"<think>private reasoning</think>\n{output(windows[0], False)}",
            windows[0],
        )
        self.assertEqual(len(prefixed.decisions), len(windows[0].decision_messages))
        for malformed in (
            f"<think>one</think><think>two</think>{output(windows[0], False)}",
            f"prefix<think>one</think>{output(windows[0], False)}",
            f"<think>unfinished{output(windows[0], False)}",
        ):
            with self.assertRaises(DecisionFormatError):
                parse_edit_decisions(malformed, windows[0])

    def test_empty_role_message_is_not_model_visible_or_counted(self) -> None:
        history = (
            Message("m0", 0, "user", "", session_id="s"),
            Message("m1", 1, "assistant", "evidence", session_id="s"),
        )
        window = build_windows(history, [RankedMessage(history[1], 1.0, 1)])[0]
        payload = model_window_payload(window)
        self.assertEqual(payload["previous_bridge"], [])
        self.assertEqual([item["content"] for item in payload["core_messages"]], ["evidence"])
        output = '[{"op":"KEEP","compressed_content":"evidence","reason":"useful"}]'
        self.assertEqual(len(parse_edit_decisions(output, window).decisions), 1)

if __name__ == "__main__":
    unittest.main()
