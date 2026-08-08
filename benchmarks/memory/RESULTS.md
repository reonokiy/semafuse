# LongMemEval retrieval result

Run date: 2026-08-08

This is an actual native Rust run over the complete public
`longmemeval_s_cleaned.json` release. The loader validated all 500 questions;
the `all_role_message` retrieval protocol scores the 470 non-abstention
questions. It is not the paper's private split and is deliberately kept
separate from the official 419-question user-turn projection.

Configuration:

- pipeline: Semafuse LazyMem production retrieval path
- embedding: `perplexity/pplx-embed-v1-0.6b`, provider pinned without fallback
- reranker: `voyageai/rerank-2.5` through OpenRouter `/rerank`
- retrieval: dense 100 + BM25 100, RRF 100, rerank top 50
- expansion: radius 2, repository-contiguous windows, length 8 / stride 7
- execution: `retrieval_only`; the prompt processor was not invoked
- dataset SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- run key: `eaecff14a5cb904f8e4b8d6149449b41f26392ab02668c2ee437bfd55b99ab01`

## Scores

All values below use the same 470 questions. `Context coverage all` is
unordered gold coverage of the context set: top-50 for the standalone stages,
and post-window-expansion evidence for the final Semafuse row. The other
columns score ranked message IDs.

| Method | RecallAll@5 | RecallAll@10 | RecallAll@20 | RecallAll@50 | nDCG@10 | MRR@50 | Context coverage all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Role-aware BM25 | 0.5553 | 0.6830 | 0.7468 | 0.8234 | 0.6325 | 0.6479 | 0.8234 |
| Dense | 0.6596 | 0.8340 | 0.9298 | 0.9723 | 0.7326 | 0.7303 | 0.9723 |
| RRF | 0.6468 | 0.7957 | 0.8298 | 0.9532 | 0.7178 | 0.7235 | 0.9532 |
| Semafuse reranked | **0.7638** | **0.8851** | **0.9426** | **0.9851** | **0.8091** | **0.8043** | **0.9957** |
| Recent 50 | 0.0021 | 0.0064 | 0.0234 | 0.0553 | 0.0055 | 0.0108 | 0.0553 |

Against the role-aware BM25 baseline, the paired 2,000-resample bootstrap
report gives:

| Metric | Mean delta | 95% CI | Wins / ties / losses |
| --- | ---: | ---: | ---: |
| RecallAll@10 | +0.2021 | [+0.1638, +0.2426] | 100 / 365 / 5 |
| RecallAll@50 | +0.1617 | [+0.1298, +0.1979] | 77 / 392 / 1 |
| nDCG@10 | +0.1766 | [+0.1500, +0.2045] | 256 / 152 / 62 |
| Context RecallAll | +0.1723 | [+0.1404, +0.2106] | 81 / 389 / 0 |

RecallAll@10 by LongMemEval question type:

| Type | BM25 | Dense | RRF | Semafuse reranked |
| --- | ---: | ---: | ---: | ---: |
| knowledge-update | 0.8472 | 0.8889 | 0.9028 | **0.9861** |
| multi-session | 0.4463 | 0.7107 | 0.6364 | **0.7769** |
| single-session-assistant | 0.8750 | 0.9821 | 0.9464 | **1.0000** |
| single-session-preference | 0.3000 | **0.7333** | 0.6000 | **0.7333** |
| single-session-user | 0.9062 | 0.9844 | 0.9844 | **1.0000** |
| temporal-reasoning | 0.7087 | 0.8031 | 0.7717 | **0.8583** |

Mean cold replay time for the Semafuse call was 2.153 seconds per question in
this environment. This includes rebuilding an isolated store and embedding the
history, so it is not an online warm-query latency. OpenRouter account usage
increased by approximately USD 0.743 during the complete 470-question run.

## Artifacts and validation

Local artifacts (ignored by Git because they are large):

- `benchmarks/memory/results/semafuse-openrouter-full/per-question.jsonl`
  (4,700 records, SHA-256
  `c64fabf22563618701024f8e14e9fd6f8b16e3907867e8858c67525af69bb9f4`)
- `benchmarks/memory/results/semafuse-openrouter-full/summary.json`
  (SHA-256
  `693e4ab2130ee45578ba1dbcbd9b2f91d42c4bb9b4577c37ffbb8de632f509c1`)
- `benchmarks/memory/results/semafuse-openrouter-full/comparison.json`
  (role-aware report regenerated from the immutable journal, SHA-256
  `d77dae263b0d70c9aad620a16d352a659e820f88a16897d48753ad047f63cede`)

Replaying the completed journal without network access rebuilt both report
files byte-for-byte with the same SHA-256 values. In `comparison.json`, BM25
and Recent are deployable baselines, Semafuse is the candidate, its shared-call
stages are ablations, full-history is a control, and both oracles are unpaired
diagnostics. Shared-stage pipeline latency and cross-method retained-memory
coverage are deliberately excluded from paired reports.

The 100-question reconstruction first reproduced the previous independent
Python dense and RRF scores exactly, then the full run completed with 470
unique questions and 10 method/stage outputs per question. Rust tests pass for
all targets, Clippy passes with warnings denied, and the Python reference
harness passes all 39 maintained tests.

The upstream corpus construction and evaluator are in the
[LongMemEval repository](https://github.com/xiaowu0162/LongMemEval). OpenRouter's
[rerank API contract](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank)
is implemented as an explicit backend rather than emulated through chat.
