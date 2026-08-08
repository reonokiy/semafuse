# LoCoMo retrieval-only benchmark

This directory configures the native `semafuse-locomo-benchmark` runner. It
uses the exact LazyMem paper-v2 LoCoMo selection:

- `conv-49` and `conv-50`;
- official categories 1–4;
- 314 QA items, of which exactly 312 contain retrieval evidence;
- source-message evidence metrics at 1, 5, 10, and 50, aggregated overall and
  by category with paired 95% bootstrap intervals.

The runner accepts only the pinned upstream `locomo10.json` SHA-256 and never
copies question, answer, conversation, or evidence text into result artifacts.
Do **not** add the official dataset to this repository.

LoCoMo data is licensed CC BY-NC 4.0. `--acknowledge-noncommercial` is a
fail-closed execution guard: pass it only after confirming the intended run is
permitted non-commercial use and satisfying attribution/change-notice duties.
It is not legal advice.

## Model-free baseline run

After obtaining `data/locomo10.json` from the pinned upstream revision, run:

```bash
cargo run --release --bin semafuse-locomo-benchmark -- \
  --suite benchmarks/locomo/configs/paper_v2.baselines.toml \
  --dataset /absolute/path/to/locomo10.json \
  --acknowledge-noncommercial \
  --output-dir benchmarks/locomo/results/paper-v2-baselines
```

The JSONL journal is flushed and synced after every method/question record. A
rerun with identical data, suite, code, Cargo lock, and resolved method configs
resumes missing keys; a changed identity requires a new output directory.

## Production Semafuse stages

`paper_v2.local-retrieval.toml` invokes the same production LazyMem ingestion,
indexing, dense retrieval, BM25, RRF, reranker, and evidence expansion paths as
Semafuse. Start the local embedding and reranker services described in
`benchmarks/lazymem/compose.local-retrieval.yml` first. The prompt processor is
not called in retrieval-only mode.

One production execution emits four separately scored rankings:

- `semafuse::dense`
- `semafuse::bm25`
- `semafuse::rrf`
- `semafuse::retrieved` (final reranked anchors)

These rows share model calls and wall time. The current production benchmark
runner creates one isolated persistent store per source conversation, performs
ingestion and indexing once, then searches that store for every missing query
sequentially. This is the same lifecycle enforced for external methods.

`shared_construction_s` is the once-per-conversation construction attempt and
is repeated on that conversation's query records for join-safe analysis. Never
sum it across records; divide it by `metadata.construction_amortization_n` for
an amortized per-query value. Query wall time and strategy retrieval time remain
separate. A resumed run reconstructs only conversations with missing records.

`paper_v2.openrouter.toml` is an optional paid development/common-evaluator
configuration using the OpenRouter embedding and reranker endpoints referenced
by the Semafuse config. It reads `OPENROUTER_API_KEY` only from the environment.
It substitutes the paper models and evaluates retrieval rather than answer F1,
so its output is **not** a published-paper score and must remain in a separate
panel.

## Mem0/A-MEM external worker bridge

`kind = "external_jsonl"` reuses the process-isolated protocol under
`benchmarks/external_memory`. The runner resets and constructs an external
memory once for each source conversation, searches all missing questions
sequentially, then closes it. Inputs contain only chronological role/text,
opaque source message ID, opaque session ID, and timestamp. Answers, evidence,
categories, and source dialog IDs never cross the worker boundary.

Example method block:

```toml
[[methods]]
kind = "external_jsonl"
id = "mem0_oss"
worker_config = "../../external_memory/configs/mem0_oss.locomo314.openrouter-raw.json"
command = ["/absolute/path/to/mem0-venv/bin/python", "-m", "semafuse_external_memory.worker"]
working_directory = "../../external_memory"
python_path = "../../external_memory/src"
top_k = 50
max_context_tokens = 16384
timeout_seconds = 600
```

The runner appends `--config <resolved-worker-config>` to `command`. External
items may cite multiple source messages; the common evaluator flattens source
IDs in native item order, keeps the first occurrence, and caps the message
ranking at 50. Unknown provenance fails the run. Use each upstream's locked
worker config and isolated environment; the bridge does not reimplement Mem0
or A-MEM.

## Score interpretation

The output contains retrieval evidence metrics only: precision, recall-any,
recall-all, mean recall, MRR, and conventional nDCG. Each metric report includes
aggregate and `multi-hop`, `temporal`, `open-domain`, and `single-hop` means.

The official category-aware LoCoMo token F1 requires answer predictions. It is
therefore deliberately `null` here and remains a separate evaluator in
`src/locomo_eval.rs`. Do not compare these retrieval scores with published
LoCoMo F1 or LazyMem binary-judge answer scores.
