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
    }
    print(json.dumps(response), flush=True)
