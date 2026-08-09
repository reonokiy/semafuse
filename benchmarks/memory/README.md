# Native memory benchmark

`semafuse bench longmemeval` is the Rust comparison runner for LongMemEval. It
loads the official cleaned JSON directly, hides all answers and evidence labels
behind opaque IDs, runs each configured method on an isolated case, writes a
crash-safe JSONL journal, and builds paired bootstrap comparisons.

The latest complete 470-question retrieval run and its reproducibility hashes
are recorded in [`RESULTS.md`](RESULTS.md).

## Protocols

The runner keeps three result populations separate:

- `official_user_turn`: the upstream user-turn retrieval projection. On the
  full cleaned release it evaluates 419 questions: abstention questions and 51
  assistant-only evidence questions are excluded.
- `all_role_message`: the Semafuse/LazyMem message projection. It evaluates all
  470 non-abstention questions on the full release.
- full QA/LJ: all 500 questions, including 30 abstention questions, using one
  frozen answer model and the official type-specific LongMemEval judge prompt.

The generated 360/40/100 reconstruction remains useful for development, but it
is not the paper's private split and must not replace the full-500 result.

## Run retrieval locally

Build once in release mode, then run either retrieval protocol:

```bash
cargo build --release --bin semafuse --features benchmarks

target/release/semafuse bench longmemeval \
  --suite benchmarks/memory/configs/longmemeval-all-role.retrieval.toml \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output-dir benchmarks/memory/results/all-role-full
```

Add `--split-manifest PATH --split test` only for a frozen development split.
Re-running with the same source and inputs resumes completed `(method,
question)` records. A different dataset, suite, method config, split, mode, or
benchmark execution/scoring source produces another `run_key` and is rejected
if pointed at the old output directory.

## Run Semafuse

The Semafuse suite calls the actual configured production strategy once per
question and records its dense, BM25, RRF, reranked, expanded-context, and
retained-memory views without repeating embedding/reranker calls:

```bash
target/release/semafuse bench longmemeval \
  --suite benchmarks/memory/configs/longmemeval-semafuse.retrieval.toml \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output-dir benchmarks/memory/results/semafuse-full
```

Edit the referenced Semafuse strategy config to point at the frozen embedding,
reranker, and prompt-processor services. Stage IDs such as `semafuse::dense`
share one physical strategy call, which keeps model budgets and latency
attribution auditable.

For a retrieval-complete run that does not spend tokens on memory compression,
use the checked-in OpenRouter profile:

```bash
target/release/semafuse bench longmemeval \
  --suite benchmarks/memory/configs/longmemeval-semafuse-openrouter.retrieval.toml \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output-dir benchmarks/memory/results/semafuse-openrouter-current
```

This profile executes the real dense index, role-aware BM25, RRF, Voyage
reranker, and neighbor/window expansion. `execution = "retrieval_only"` is a
hard boundary: it skips the prompt processor and is rejected with `--mode
full`. The main method is the reranked output; `::dense`, `::bm25`, `::rrf`,
and `::retrieved` are shared-call stage ablations rather than extra API runs.
Their quality scores are valid ablations, but their copied pipeline timing is
not reported as independent stage latency. The OpenRouter key is read only
from `OPENROUTER_API_KEY`.

## Rebuild reports without model calls

The append-only journal is the source of truth. A role-aware comparison can be
rebuilt from an existing run without loading `.env`, retrieval models, or API
clients:

```bash
cargo build --release --bin semafuse --features benchmarks

target/release/semafuse bench report \
  --run-dir benchmarks/memory/results/semafuse-openrouter-full
```

This writes `comparison.json`. Deployable baselines, the candidate, and
shared-call ablations appear in paired reports; full-history/no-memory controls
and label-dependent oracles appear only as unpaired diagnostics. The neutral
`context_*` metrics mean evidence coverage of the actual downstream context;
post-compression `retained_*` is intentionally not compared across methods.

## Full QA/LJ

Copy `longmemeval-full.example.toml`, replace both model placeholders, then run
with `--mode full`. The optional workspace `.env` is loaded automatically; use
`--no-dotenv` to require an already-populated environment.

Artifacts:

- `per-question.jsonl`: fsynced after every completed record; rankings,
  expansion/retention, context, predictions, judge output, timing, and usage.
- `summary.json`: run manifest plus fail-closed aggregate tables, 2,000-sample
  paired bootstrap 95% intervals, and win/tie/loss against the configured
  baseline. Version-one run summaries remain immutable provenance.
- `comparison.json`: regenerable role-aware report over the same journal,
  separating paired methods, shared-call ablations, controls, and oracles.

The official benchmark and evaluator implementations are documented in the
[LongMemEval repository](https://github.com/xiaowu0162/LongMemEval). The memory
pipeline reconstructed here follows the [LazyMem paper](https://arxiv.org/abs/2607.22690).
