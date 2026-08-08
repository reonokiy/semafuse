import json
import sys

for line in sys.stdin:
    question = json.loads(line)
    ids = [message["id"] for message in question["messages"]]
    response = {
        "question_id": question["id"],
        "stage_rankings": {
            "dense": ids,
            "bm25": list(reversed(ids)),
            "rrf": ids[1:] + ids[:1],
            "retrieved": ids[:2],
        },
        "retrieval_latency_s": 0.01,
        "processing_latency_s": 0.0,
        "metadata": {
            "benchmark_trace": {
                "isolation": "fresh_sqlite_memory_per_question",
                "timing_source": "fixture",
                "execution_mode": "retrieval-only",
                "timing_s": {
                    "ingest": 0.02,
                    "index": 0.03,
                    "recall_total": 0.01,
                    "strategy_recall_total": 0.01,
                    "retrieval": 0.01,
                    "window_and_processing": 0.0,
                },
            }
        },
    }
    print(json.dumps(response), flush=True)
