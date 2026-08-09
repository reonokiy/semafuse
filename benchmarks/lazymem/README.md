# LazyMem paper-v2 public reconstruction harness

This directory is an independent, clean-room benchmark harness for the
retrieve-broadly/construct-selectively pipeline described in
[LazyMem arXiv v2](https://arxiv.org/abs/2607.22690v2). It was written from the
paper, the public benchmark schemas, and independently observed public artifact
contracts. No source from the unlicensed upstream LazyMem repository is copied,
imported, or vendored. The repository was inspected only to establish artifact
availability and public experiment metadata; all implementation here is
independently authored.

The harness makes a careful distinction between:

- **paper-reported parameters**, recorded in
  [`manifests/paper_v2.public-repro.json`](manifests/paper_v2.public-repro.json);
- **parameters recovered from the public artifact**, including the exact sparse
  tokenizer, BM25 defaults, RRF `k = 60`, retrieval cutoffs, and the released
  window-builder boundary; and
- **artifacts that the paper text does not make reproducible**, such as the
  exact split IDs. The paper's exact LoCoMo binary-LLM-judge prompt is also
  absent from the paper and public artifact.

Consequently, this is a runnable evaluation protocol, not a claim that an
arbitrary run reproduces the paper's scores. A paper-comparable report must pass
every comparability gate in the manifest and must identify any remaining
differences.

## Reproduction profiles

Two named configurations are strict and fail closed if a locked field drifts:

- [`configs/paper_v2.paper-spec.toml`](configs/paper_v2.paper-spec.toml) follows
  Equation (1) and merges adjacent retrieval anchors only when their gap is
  `<= 2 * radius` (`4` at radius 2).
- [`configs/paper_v2.public-repro.toml`](configs/paper_v2.public-repro.toml)
  follows the released window builder's inclusive boundary
  `<= 2 * radius + 1` (`5` at radius 2).

Both profiles target the paper's prompt-only Qwen3-32B MPM and lock
dense/sparse/fused limits `100/100/100`, RRF `k = 60`, rerank Top-50, radius 2,
window length/stride `8/7`, MPM generation, answer generation, latency protocol,
and tokenizer identity. To run a deliberate ablation, copy a configuration, set
`reproduction.status = "custom-ablation"`, and record every changed field. A
custom-ablation run is not paper-comparable by default. `validate-config`
checks these rules before endpoint work begins:

```bash
semafuse-lazymem-bench validate-config \
  --config configs/paper_v2.public-repro.toml
```

## What is implemented

- dataset-neutral immutable `Message`, `Question`, and `MemoryContext` types;
- a strict [LongMemEval](https://github.com/xiaowu0162/LongMemEval) adapter;
- deterministic 360/40/100 question-type-stratified splitting with seed 42,
  row-wise largest remainder plus exact column rebalancing, emitted IDs, source
  SHA-256, and split-ID SHA-256;
- a LoCoMo adapter restricted to `conv-49` and `conv-50`, categories 1--4,
  with the paper's expected 314-question assertion; released BLIP image
  captions are appended to message text and retained separately in metadata;
- dependency-free Okapi BM25 for smoke tests plus the exact public
  `rank_bm25.BM25Okapi` path (`k1 = 1.5`, `b = 0.75`, `epsilon = 0.25`) using
  lowercase ASCII alphanumeric/underscore/hyphen tokens; optional local Qwen3
  dense retrieval and BGE cross-encoder reranking; stable RRF `k = 60`; and
  public cutoffs of 100 per channel, 100 fused, and 50 reranked;
- radius-2 history expansion, transitive merging, max-8 sub-windows, stride 7,
  and both documented merge boundary interpretations;
- the complete Appendix G.1 MPM system/user prompt and its structured
  `previous_bridge`, `core_messages`, and `next_bridge` payload;
- strict one-decision-per-message JSON parsing, optional single leading Qwen
  thinking-block parsing, and cross-window deduplication (`KEEP` wins, then
  earlier window order);
- OpenAI-compatible HTTP endpoints using the Python standard library and up to
  64 concurrent MPM window requests while questions remain sequential;
- the paper answer prompt with Qwen thinking enabled, a generic smoke-test
  judge, the pinned official LongMemEval LLM-judge prompt/rule, the released
  LoCoMo category-aware F1 scorer, and strict external-score import;
- RAG Top20, RAG Top50, Oracle Turn, and LongMemEval Oracle Session baselines;
- a `semafuse-lazymem` product backend that streams normalized questions to the
  Rust benchmark runner and therefore exercises Semafuse's actual ingest,
  indexing, retrieval, windowing, and memory-processing path;
- crash-safe append-only JSONL resumption;
- LJ, per-type LJ, expanded-neighbor All@50, memory-token, and latency
  mean/median/P90/P95 summaries.

The runtime core has no required Python dependency. The `retrieval` and
`tokenizer` extras are loaded only for local model inference and exact Qwen
token counting.

## Data and license guardrails

No dataset is downloaded by this package.

- LongMemEval is obtained from its official release. The adapter accepts the
  documented 500-record JSON schema and carries `has_answer` turn labels into
  `gold_message_ids`. Empty non-gold messages remain in the retrieval corpus to
  match the public implementation but are omitted from MPM decisions, which are
  defined only over non-empty role messages.
- **LoCoMo is CC BY-NC 4.0.** Its use is limited to non-commercial purposes and
  requires attribution. The adapter emits a warning and refuses to load the
  data unless `--acknowledge-locomo-noncommercial` is passed. This guard is not
  legal advice and does not determine whether a particular use is lawful. See
  the [official LoCoMo license](https://github.com/snap-research/locomo/blob/main/LICENSE.txt).

Normalized data and model outputs may remain derived or licensed material. Do
not publish or use them outside the applicable dataset and model terms.

## Install and offline checks

From this directory:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

An editable install is optional:

```bash
python -m pip install -e .
python -m pip install -e '.[retrieval,tokenizer]'
python -m pip install -e '.[locomo-eval]'
```

The second command is needed only for local Qwen3 embeddings, BGE reranking,
and exact Qwen3-32B tokenizer counts. The third installs NLTK's Porter stemmer
for the released LoCoMo F1 scorer. Model downloads are never initiated by the
tests.

## Reconstruct the LongMemEval split

Use the exact LongMemEval file intended for the experiment and preserve its
hash in the generated split manifest:

```bash
semafuse-lazymem-bench split-longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --output runs/longmemeval.paper-v2.split.json
```

The output contains all train/validation/test IDs, counts by type, the source
file SHA-256, and a canonical SHA-256 over the three sorted ID lists. The
expected allocation is:

| Type | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| single-session user | 50 | 6 | 14 |
| single-session assistant | 40 | 5 | 11 |
| single-session preference | 22 | 2 | 6 |
| multi-session | 96 | 10 | 27 |
| temporal reasoning | 96 | 11 | 26 |
| knowledge update | 56 | 6 | 16 |

The paper does not publish its exact split IDs or an ID hash. Matching the
reported allocation is necessary but does not prove ID-level identity with the
authors' private run; report the generated hash.

The checked real-data validation record for the pinned public releases is in
[`results/public_data_validation.json`](results/public_data_validation.json).
It records data and generated split hashes but deliberately contains no model
score because no configured inference endpoints were available during
validation.

## Native memory benchmark

The maintained Semafuse and baseline comparison runner is now the native Rust
[memory benchmark](../memory/README.md). Its latest complete LongMemEval
retrieval scores and reproducibility hashes are recorded in
[the results report](../memory/RESULTS.md).

## Inspect normalized inputs

```bash
semafuse-lazymem-bench adapt \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --output runs/longmemeval.normalized.jsonl

semafuse-lazymem-bench adapt \
  --dataset locomo \
  --input /data/locomo.json \
  --output runs/locomo-49-50.normalized.jsonl \
  --acknowledge-locomo-noncommercial
```

LoCoMo filtering is deliberately not configurable in a paper run: only
`conv-49`, `conv-50`, and categories 1--4 are retained, and the adapter asserts
that the result contains 314 questions. Packed evidence references and the one
zero-padded released dialog ID are normalized against the actual conversation,
while `raw_evidence`, original text, image URL, and caption remain auditable in
metadata.

## Window merge semantics

`merge_rule = "paper"` implements arXiv v2 Equation (1): merge consecutive
retrieved indices when their distance is at most `2 * radius`. With radius 2,
the threshold is 4. This is locked by the `paper-spec` profile.

`merge_rule = "code"` reproduces the released LazyMem window builder, which also
merges directly touching expanded spans using `2 * radius + 1` (threshold 5).
This is locked by the `public-repro` profile. Only a `custom-ablation` profile
may change the named profiles' fields; always record the boundary explicitly.

After transitive merging, any span longer than eight messages is split with
stride seven; full adjacent sub-windows therefore share one message and the
final sub-window may be shorter.

## Configure endpoints and run

Start from
[`configs/paper_v2.public-repro.toml`](configs/paper_v2.public-repro.toml).
Endpoint coordinates are read from environment variables so credentials never
need to enter TOML or JSONL outputs:

```bash
export LAZYMEM_MPM_BASE_URL=http://localhost:8001/v1
export LAZYMEM_MPM_MODEL=Qwen/Qwen3-32B
export LAZYMEM_ANSWER_BASE_URL=http://localhost:8002/v1
export LAZYMEM_ANSWER_MODEL=Qwen/Qwen3-32B
export LAZYMEM_JUDGE_BASE_URL=http://localhost:8003/v1
export LAZYMEM_JUDGE_MODEL=deepseek-v4-pro
```

The harness never starts a server, downloads weights, or calls an endpoint
until the operator explicitly invokes `run`:

```bash
semafuse-lazymem-bench run \
  --config configs/paper_v2.public-repro.toml \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --split-manifest runs/longmemeval.paper-v2.split.json \
  --split test \
  --method lazymem \
  --output runs/lazymem.longmemeval.test.jsonl
```

The same command accepts `rag20`, `rag50`, `oracle-turn`, or
`oracle-session`. Oracle Session is intentionally rejected for LoCoMo because
the paper does not define it there.

## Benchmark Semafuse's production LazyMem strategy

`lazymem` above is the independent Python reconstruction. To measure the
strategy that this repository actually ships, use `semafuse-lazymem`. The
harness keeps its dataset adapters, answer generation, official evaluators,
metrics, and JSONL resume journal, while a persistent Rust JSONL process
performs each question's Semafuse ingestion and recall.

Build the runner from the repository root and copy the supplied Semafuse
strategy template before replacing its local endpoint URLs. Its recall strategy
is already `lazymem` and its strategy settings match the public-reproduction
profile:

```bash
cargo build --release --bin semafuse-lazymem-benchmark
cd benchmarks/lazymem
cp configs/semafuse_lazymem.product.toml runs/semafuse-lazymem.toml

semafuse-lazymem-bench run \
  --config configs/semafuse_lazymem.custom.toml \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --split-manifest runs/longmemeval.paper-v2.split.json \
  --split test \
  --method semafuse-lazymem \
  --semafuse-config runs/semafuse-lazymem.toml \
  --semafuse-benchmark-binary ../../target/release/semafuse-lazymem-benchmark \
  --output runs/semafuse-lazymem.longmemeval.test.jsonl
```

The bridge writes each normalized `Question` as one JSONL request and requires
one flushed response with the same `question_id`, the constructed memory
`context`, retrieval/coverage IDs, stage latencies, and runner metadata. The
backend is deliberately restricted to
`configs/semafuse_lazymem.custom.toml` (or another
`reproduction.status = "custom-ablation"` configuration). It is a product
benchmark, not a paper-comparable result: Semafuse's production strategy,
storage lifecycle, and endpoint configuration are the system under test rather
than the clean-room reconstruction. Each result row records this distinction
and hashes the supplied Semafuse strategy configuration. Relevant
`SEMAFUSE__STRATEGIES__...` overrides are also fingerprinted without retaining
their values, so a resumed output cannot silently mix endpoint or model
overrides.

Every case uses a fresh in-memory SQLite store and a synthetic repository
scope, so one question cannot leak history into another. Consequently the
end-to-end latency is a cold replay measure; the versioned
`method_metadata.runner_metadata.benchmark_trace.timing_s` separates ingest,
index, and strategy-recall time for reporting.

### Use the installed Codex CLI as Semafuse's MPM

[`configs/semafuse_lazymem.codex.toml`](configs/semafuse_lazymem.codex.toml)
is a ready-to-copy custom-ablation strategy config. It replaces only Semafuse's
query-time memory processor with the locally installed and authenticated
`codex` CLI; it does not turn Codex into an embedding, reranking, answer, or
judge service. Check authentication and the model catalog first:

```bash
codex login status
codex debug models | jq -r '.models[].slug'
```

Then copy the template and use it as the `--semafuse-config` in the product
benchmark command above:

```bash
cp configs/semafuse_lazymem.codex.toml runs/semafuse-lazymem-codex.toml

semafuse-lazymem-bench run \
  --config configs/semafuse_lazymem.custom.toml \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --split-manifest runs/longmemeval.paper-v2.split.json \
  --split test \
  --method semafuse-lazymem \
  --semafuse-config runs/semafuse-lazymem-codex.toml \
  --semafuse-benchmark-binary ../../target/release/semafuse-lazymem-benchmark \
  --output runs/semafuse-lazymem.codex.longmemeval.test.jsonl
```

For every MPM window Semafuse starts a fresh `codex exec` process in a private
temporary directory. It passes the prompt on stdin, uses read-only sandboxing,
disables tools, hooks, web, plugins, memories, and multi-agent features, and
requires Codex structured output to be a single object with a `decisions`
array. The array must contain exactly one positional object per window message:
`{"op":"KEEP|DROP","compressed_content":"...","reason":"..."}`. A
`DROP` must use an empty `compressed_content`; a `KEEP` must use a non-empty
one. Invalid output fails the question before a JSONL result is committed.

The `codex_cli` backend deliberately ignores user configuration and requests an
ephemeral session, so it uses Codex authentication but does not inherit project
instructions, hooks, model defaults, or a local `--oss` provider. It is a
local CLI transport to the Codex service, not local-on-GPU inference. The
embedding and reranker endpoints in the Semafuse config are still mandatory;
the companion harness configuration below sends `[answer]` and `[judge]`
through the same local Codex CLI. Preserve `codex --version`, the selected
model, and the copied TOML with the output journal.

### Use the local semafused models service instead of HTTP

The embedding and reranker can be moved off the HTTP path without moving them
into each short-lived Semafuse command. Install and start
[`semafused`](../../docs/semafused.md) separately, using absolute local
directories for `Qwen/Qwen3-Embedding-0.6B` and
`BAAI/bge-reranker-v2-m3`. After it reports ready, inspect its semantic
fingerprints:

```bash
cargo run --bin semafuse -- daemon status \
  --socket "$XDG_RUNTIME_DIR/semafuse/semafused.sock"
```

Copy the dedicated strategy template, replace its literal socket path and both
fingerprints with the status output, and pass the result to the same product
benchmark command:

```bash
cp configs/semafuse_lazymem.semafused.toml.example \
  runs/semafuse-lazymem-semafused.toml
# Edit runs/semafuse-lazymem-semafused.toml before running.
```

The persistent benchmark runner is only a socket client. It does not start,
restart, or stop the daemon, and a missing socket or fingerprint mismatch fails
the question without falling back to HTTP. The Codex MPM remains a fresh
one-shot process for every memory window.

### Local CPU smoke run

For a single-question, end-to-end functional check, the repository includes a
[TEI](https://huggingface.co/docs/text-embeddings-inference/quick_tour) Compose
file and two custom-ablation configurations:

- [`compose.local-retrieval.yml`](compose.local-retrieval.yml) runs local CPU
  embedding and reranking services on ports 8001 and 8002. It caches
  `Qwen/Qwen3-Embedding-0.6B` and `BAAI/bge-reranker-v2-m3` in named Docker
  volumes.
- [`configs/semafuse_lazymem.local-smoke.toml`](configs/semafuse_lazymem.local-smoke.toml)
  uses those services and invokes the installed Codex CLI for Semafuse's MPM.
- [`configs/semafuse_lazymem.codex-harness.toml`](configs/semafuse_lazymem.codex-harness.toml)
  invokes the same local Codex CLI for the harness's answer and optional judge
  calls.

The first container start downloads public model weights, and CPU inference is
intentionally appropriate only for a small smoke run. If `docker info` cannot
reach its socket, start a new login shell (or run `newgrp docker`) before
continuing. The minimal run omits `--generic-judge`, so it makes a Codex MPM
call and a Codex answer call, but no judge call.

```bash
cd /path/to/semafuse
cargo build --release --bin semafuse-lazymem-benchmark

cd benchmarks/lazymem
codex login status
docker compose -f compose.local-retrieval.yml up -d

until curl --fail --silent http://127.0.0.1:8001/health >/dev/null \
  && curl --fail --silent http://127.0.0.1:8002/health >/dev/null; do
  sleep 2
done

mkdir -p runs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m semafuse_lazymem_bench.cli run \
  --config configs/semafuse_lazymem.codex-harness.toml \
  --dataset longmemeval \
  --input tests/fixtures/longmemeval.json \
  --method semafuse-lazymem \
  --semafuse-config configs/semafuse_lazymem.local-smoke.toml \
  --semafuse-benchmark-binary ../../target/release/semafuse-lazymem-benchmark \
  --allow-nonpaper-count \
  --limit 1 \
  --approximate-tokens \
  --output runs/semafuse-lazymem.local-smoke.jsonl

docker compose -f compose.local-retrieval.yml down
```

`--allow-nonpaper-count` is deliberately limited to `custom-ablation` runs: it
permits the two-record test fixture without fabricating a 500-question split
manifest. The resulting JSONL verifies the actual Rust ingestion, embedding,
reranking, windowing, Codex MPM, and Codex answer path, but it is neither a
paper-comparable result nor a useful CPU latency/quality measurement. Use the
real LongMemEval release, a generated split manifest, and the product-sized
model configuration for a benchmark report.

### Use the installed Codex CLI for answers and judges

[`configs/semafuse_lazymem.codex-harness.toml`](configs/semafuse_lazymem.codex-harness.toml)
selects `backend = "codex_cli"` for the harness's `[answer]`, `[judge]`, and
(for the independent Python method) `[mpm]` clients. Pair it with the
Semafuse Codex-MPM strategy config above to make every LLM call in a
`semafuse-lazymem` run use the locally authenticated Codex CLI.

The harness launches a fresh `codex exec` process for every answer or judge.
It uses an ephemeral temporary workspace, a `read-only` sandbox,
`approval_policy = "never"`, `--strict-config`, `--ignore-user-config`,
`--ignore-rules`, JSON output, disabled web search, and disabled tool, hook,
plugin, memory, and multi-agent features. Final replies must satisfy a
one-field structured-output envelope before the benchmark accepts them. No API
key is configured or written to a journal.

The template pins `gpt-5.6-sol`; change `model` only deliberately, because it
is part of the resumable run identity. Codex CLI does not expose the
OpenAI-compatible `temperature`, `top_p`, `max_tokens`, or endpoint-specific
extra-body controls. The client records them as requested-but-unsupported in
`answer_metadata` or `judge_metadata` rather than claiming they were applied.
This makes the resulting row a Codex-backed custom ablation, not a
paper-comparable result.

For the optional generic smoke judge, add `--generic-judge` to `run`. For the
LongMemEval official judge, use the same harness configuration during
post-processing:

```bash
semafuse-lazymem-bench evaluate \
  --evaluator longmemeval-official-lj \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --results runs/semafuse-lazymem.codex.longmemeval.test.jsonl \
  --output runs/semafuse-lazymem.codex.longmemeval.test.evaluated.jsonl \
  --config configs/semafuse_lazymem.codex-harness.toml
```

Each successfully completed question is appended and `fsync`ed as one JSONL
record. Re-running the exact command skips existing `run_key` values. Malformed
MPM output fails before a partial row is committed.

### MPM prompt and output contract

The request uses the complete Appendix G.1 system and user templates. Every
window is rendered as three ordered lists: `previous_bridge`, `core_messages`,
and `next_bridge`. Entries carry their position, role, timestamp, global message
index, retrieval rank when directly retrieved, and content. The model emits one
positional object per non-empty role message, in section and chronological
order, with exactly `op`, `compressed_content`, and `reason`.

The parser accepts either the JSON array itself or exactly one closed leading
`<think>...</think>` block followed by that array. Markdown fences, surrounding
prose, unclosed/nested/multiple thinking blocks, unknown fields, invalid
operations, trailing JSON, and decision-count mismatches are rejected. Named
profiles send `temperature = 0.1`, `top_p = 0.95`, `max_tokens = 3072`, and
`chat_template_kwargs.enable_thinking = true`, with at most 64 concurrent
windows. The final answer request uses temperature 0, 512 output tokens, and
thinking enabled.

### Judges

`run --generic-judge` exercises a blinded binary-judge interface for smoke tests;
it is not an official benchmark metric. For LongMemEval, post-process answer
rows with the built-in, pinned official prompt and decision rule:

```bash
semafuse-lazymem-bench evaluate \
  --evaluator longmemeval-official-lj \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --results runs/lazymem.longmemeval.test.jsonl \
  --output runs/lazymem.longmemeval.test.evaluated.jsonl \
  --config configs/paper_v2.public-repro.toml
```

This evaluator uses the exact released LongMemEval type-specific prompts,
temperature 0, at most 10 output tokens, and the released lowercase-`"yes"`
substring decision rule. Configure the paper's declared judge model through the
`[judge]` endpoint before calling it.

The released LoCoMo repository does not provide a binary LLM judge. Its official
scorer is category-aware Porter-stem F1 and is available separately:

```bash
semafuse-lazymem-bench evaluate \
  --evaluator locomo-official-f1 \
  --dataset locomo \
  --input /data/locomo.json \
  --results runs/lazymem.locomo.jsonl \
  --output runs/lazymem.locomo.f1.jsonl \
  --acknowledge-locomo-noncommercial
```

The LazyMem paper reports a binary LLM-judge score for LoCoMo, but its exact
LoCoMo prompt is not published in the paper or public artifact. Do not invent
one. If independently obtained immutable scores are available, import an exact
question-ID match:

```bash
semafuse-lazymem-bench evaluate \
  --evaluator external-scores \
  --dataset locomo \
  --input /data/locomo.json \
  --results runs/lazymem.locomo.jsonl \
  --output runs/lazymem.locomo.external.jsonl \
  --scores scores.jsonl \
  --metric binary-lj \
  --evaluator-id IMMUTABLE_EVALUATOR_ID \
  --acknowledge-locomo-noncommercial
```

The score file must contain exactly `question_id` and `score` per JSONL row and
must match the result IDs exactly. Imported rows are explicitly marked
`paper_comparable = false`. Never mix generic, official F1, and binary LJ
results under one metric name.

Questions are processed sequentially. LazyMem windows for the current question
are processed concurrently up to 64. Use the dedicated latency command for the
paper protocol; it executes and discards three complete warm-up runs, then
records three measured repeats without correctness-judge calls:

```bash
semafuse-lazymem-bench run-latency \
  --config configs/paper_v2.public-repro.toml \
  --dataset longmemeval \
  --input /data/longmemeval_s_cleaned.json \
  --split-manifest runs/longmemeval.paper-v2.split.json \
  --split test \
  --method lazymem \
  --output runs/lazymem.longmemeval.latency.jsonl
```

Latency rows are tagged `record_kind = "latency"` and include a measurement
index. Keep them separate from one-row-per-question correctness output.

## Metrics

```bash
semafuse-lazymem-bench metrics \
  --input runs/lazymem.longmemeval.test.evaluated.jsonl \
  --output runs/lazymem.longmemeval.test.metrics.json
```

`All@50` is complete gold-message coverage after adding all messages in the
configured neighbor-expanded windows. Questions without annotated gold are
excluded, matching the released ripple evaluator. Memory
tokens count only the method-produced memory block, excluding the query and
fixed answer prompt. Omit `--approximate-tokens` for reported experiments so
the configured Qwen3-32B tokenizer is used; approximate-token runs are smoke
tests and are not comparable to the paper.

The summary records tokenizer identities, per-type denominators, and latency
distributions. `latency_s.total` is the online construction-plus-answer path.
`latency_s.evaluation_wall` additionally includes only a generic judge invoked
inside `run`; post-run official evaluator time is stored separately as
`evaluation_latency_s` and is never folded into the online latency. Keep the raw
JSONL, split manifest, resolved config, server versions, model revisions,
hardware, and dataset hashes with every report.

## Prompt-only and remaining comparability limits

This harness intentionally implements the paper's training-free
`LazyMem + Qwen3-32B` route. The trained LazyMem-4B SFT/GRPO variant is out of
scope and no training stack is included.

The exact LongMemEval split IDs and LoCoMo binary-judge prompt are unavailable.
Therefore the checked public-data record deliberately contains no claimed paper
score. Record immutable model revisions, resolved endpoint configuration,
serving software, hardware, dataset/split hashes, and evaluator identity with
every result. A Codex-backed processor or a model other than the declared
Qwen3-32B prompt-only MPM is a useful implementation profile, but it is an
ablation rather than a reproduction of the paper's reported row.
