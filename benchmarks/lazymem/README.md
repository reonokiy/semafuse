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
