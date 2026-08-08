# LMEB retrieval harness

This directory is an independent, Python-standard-library harness for the
[Long-horizon Memory Embedding Benchmark (LMEB)](https://github.com/KaLM-Embedding/LMEB).
It loads the official `corpus.jsonl`, `queries.jsonl`, headerless `qrels.tsv`,
and optional `candidates.jsonl` shapes without downloading data. Python 3.11+
is required for standard-library TOML support; JSON manifests use the same
schema.

The evaluator targets LMEB commit
`a02ae842598183ed162fc90c58ffaae5eec89f12`, MTEB `2.3.0`, and trec_eval
`9.0.8`. At cutoffs `[1, 5, 10, 25, 50]` it emits per-query and macro
`nDCG`, `MAP`, `Recall`, `Precision`, `MRR`, and `R_cap`. The implementation
preserves a subtle official detail: trec_eval breaks equal scores by reverse
document ID for nDCG/MAP/Recall/Precision, while MTEB's MRR and LMEB's patched
`R_cap` preserve input order for equal scores. Macro rounding also follows the
official code (five decimals except MRR).

## Run it

From this directory:

```bash
python -m lmeb_harness validate --manifest example.paper.toml
python -m lmeb_harness run --manifest my-run.toml --ranker bm25
python -m lmeb_harness run --manifest my-run.json --ranker semafuse-jsonl
python -m lmeb_harness export --manifest my-run.toml --output ranker-input.jsonl
python -m unittest discover -s tests -v
```

## Verify the pinned OpenRouter runs

The large actual-run artifacts stay under the ignored `results/actual/`
directory. `actual_runs.json` pins their result, prediction, dataset-input,
manifest, harness, config, and runner-binary hashes. Rebuild or verify the
query-free receipt with:

```bash
python -m lmeb_harness.receipt \
  --spec actual_runs.json \
  --output receipts/semafuse-openrouter-actual.json
python -m lmeb_harness.receipt \
  --spec actual_runs.json \
  --verify receipts/semafuse-openrouter-actual.json
```

Receipt generation fails closed on any content-hash, query-set, task-set, or
reported-metric inconsistency. Its fixed 10,000-replicate paired bootstrap
resamples queries independently inside each task/subset and then gives each
task/subset equal weight. The same sampled query indices are used for all four
stages, so `retrieved` deltas retain their query pairing.

`--ranker` overrides only the manifest's `ranker.kind`; the remaining ranker
options still come from the manifest. Runs are content-addressed. Each task
gets an fsynced `predictions.partial.jsonl` checkpoint and `state.json`, so a
rerun with `resume = true` continues incomplete queries or returns the already
verified result. `result.json` contains hashes for every input, the manifest,
the harness source, inferred local Semafuse binary/config artifacts, runtime
provenance, task/subset scores, and the LMEB paper aggregation levels:
mean-task, mean-dataset, memory-type, and mean-memory-type.

## Paper reproduction versus common policy

These modes must not be mixed in a leaderboard:

| Property | `paper-reproduction` | `common-policy` / Semafuse |
|---|---|---|
| Candidate semantics | Score with global corpus statistics, then apply LMEB `SubsetRetrieval` scene filtering | Treat each query's candidates as its admissible cold corpus |
| Candidate lookup | Exact query ID for the official direct-ID task list; otherwise the first two underscore-separated query-ID components | Same lookup by default; `longest-prefix` is available explicitly |
| Metrics | Exact formulas and aggregation targeted above | The same metrics, but the retrieval population differs |
| Ranker | Use external scores from the same model/instruction setup as the paper | Built-in diagnostic BM25 or Semafuse production retrieval |

The built-in `bm25` uses global-corpus Lucene BM25 statistics (`k1=1.5`,
`b=0.75`) and deterministic full candidate coverage. It is deliberately named
a diagnostic approximation: unlike LMEB's official `bm25s==0.3.6` baseline it
uses a small bundled English stop-word list, no PyStemmer, Python doubles
instead of float32, and deterministic tie ordering. Do not cite it as the
paper's BM25 number. For paper-comparable model results, import scores through
the external JSONL protocol and keep the paper's with/without-instruction
setting identical.

`semafuse-jsonl` launches the configured command once and streams a
`BenchmarkQuestion` per query. Each LMEB candidate document becomes one
isolated `user` message/session, with no qrels or gold fields in the request.
The adapter reads `response.stage_rankings` and independently stores and scores
`dense`, `bm25`, `rrf`, and `retrieved`. It refuses a query without an explicit
candidate scene: repeatedly ingesting an unrestricted full corpus into a fresh
store would be both prohibitively expensive and a misleading comparison.
The runner rejects `paper-reproduction + semafuse-jsonl` outright; a cold
candidate run cannot carry a paper-policy label. Subprocess responses must be
complete newline-framed JSON within `timeout_seconds`; timeout, EOF with a
partial line, or an oversized response kills the process and fails the run.
Inline API keys, tokens, authorization headers, passwords, and secret-looking
command arguments are rejected. Supply secrets through the parent environment
and list only optional variable names in `ranker.env_names`; values are never
serialized.

When the backend returns Semafuse's `metadata.benchmark_trace`, the harness
also preserves its validated ingest, index, recall, retrieval, and processing
timings. Task results summarize both retrieval-only latency and the cold
per-query construction/pipeline latency at mean, p50, and p95. Older result
artifacts that predate this field remain score-valid, but their reported
latency covers only retrieval and processing and must be labeled accordingly.
Use the Rust runner in retrieval-only mode, for example:

```json
{
  "kind": "semafuse-jsonl",
  "command": [
    "../../target/release/semafuse-lazymem-benchmark",
    "--config", "../../semafuse.toml",
    "--retrieval-only", "--limit", "50"
  ]
}
```

That cold-candidate policy is useful for comparing Semafuse stages under one
admissible corpus, but it is recorded as a paper-policy deviation because the
official `SubsetRetrieval` code scores globally before filtering.

## Ranker JSONL protocol

`export` writes `lmeb-ranker-input-v1`: a metadata row, document rows keyed by
`task_id`, then query rows with text, optional instruction, candidate IDs, and
the resolved scene. An external ranker returns one `lmeb-ranking-v1` row per
query:

When `instruction_setting = "without"`, exported query rows omit the
`instruction` key entirely rather than exposing an unused task instruction.

```json
{"schema_version":"lmeb-ranking-v1","task_id":"LoCoMo/single_hop","query_id":"scene_a_q1","rankings":{"dense":[{"doc_id":"d1","score":0.91},{"doc_id":"d2","score":0.73}],"reranked":["d2","d1"]},"metadata":{"model":"example"}}
```

A ranking item may be `{doc_id, score}` or a bare document ID; bare IDs receive
strictly descending rank scores. Multiple named stages are evaluated in one
run. In `paper-reproduction`, each imported stage must cover every admissible
candidate so missing tail scores cannot silently change Precision/MAP at the
larger cutoffs. Unknown or duplicate IDs, missing queries, non-finite scores,
and qrels/candidate integrity errors fail closed.

LMEB combines datasets with different upstream licenses. This harness ships
only synthetic fixtures; users remain responsible for the license recorded in
each task manifest.
