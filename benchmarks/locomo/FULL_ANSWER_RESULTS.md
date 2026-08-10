# LoCoMo full-answer development result

This is the completed end-to-end comparison for Semafuse's prompt-only
LazyMem strategy. It is a custom no-thinking development run, not a direct
reproduction of the LazyMem paper's reported judge score.

## Result

The primary 314-question comparison uses the same common reader for BM25 and
both Semafuse outputs. Scores are released category-aware LoCoMo token-F1;
intervals are paired 95% bootstrap intervals over questions (2,000 resamples,
seed 42).

| Method | n | Mean F1 | 95% CI | Delta vs BM25 | Delta 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Top-50 | 314 | 0.1760 | [0.1563, 0.1959] | — | — | — |
| Semafuse full prompt-only LazyMem | 314 | 0.2011 | [0.1813, 0.2219] | +0.0251 | [+0.0100, +0.0420] | 116/110/88 |
| Semafuse final retrieved context | 314 | **0.2082** | [0.1876, 0.2292] | **+0.0321** | **[+0.0170, +0.0475]** | 106/124/84 |

The imported A-MEM and Mem0 rankings only cover the 312 questions with valid
retrieval evidence, so their fair comparison is a separate exactly paired
cohort.

| Method | n | Mean F1 | 95% CI | Delta vs BM25 | Delta 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| A-MEM paper-original source-ranking RAG | 312 | 0.1758 | [0.1550, 0.1981] | -0.0013 | [-0.0196, +0.0165] | 94/111/107 |
| BM25 Top-50 | 312 | 0.1771 | [0.1583, 0.1987] | — | — | — |
| Semafuse full prompt-only LazyMem | 312 | 0.2023 | [0.1824, 0.2240] | +0.0252 | [+0.0102, +0.0418] | 116/109/87 |
| Mem0 OSS source-ranking RAG | 312 | 0.2044 | [0.1844, 0.2266] | +0.0273 | [+0.0123, +0.0429] | 98/120/94 |
| Semafuse final retrieved context | 312 | **0.2094** | [0.1895, 0.2319] | **+0.0324** | **[+0.0170, +0.0483]** | 106/123/83 |

Semafuse's complete memory output improves over BM25 by 0.0251 F1 on all 314
questions, and the paired confidence interval excludes zero. The final
retrieved context is another 0.0071 above the processed memory output, although
this run does not compute a direct confidence interval between those two
Semafuse rows. Under this no-thinking profile, the LazyMem processing stage
therefore did not demonstrate an advantage over Semafuse's own hybrid
retrieval frontend. Most of the full-memory gain over BM25 comes from
multi-hop and single-hop questions; open-domain and temporal results are more
mixed.

The Mem0 and Semafuse full-memory means are close, and no direct paired interval
between them was computed. A-MEM's delta from BM25 crosses zero, so this run
does not establish a difference between those two rows.

## Protocol

- Dataset: pinned LoCoMo `locomo10.json`, upstream commit
  `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, dataset SHA-256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Population: `conv-49` and `conv-50`, categories 1–4; 314 answer questions and
  the exactly paired 312-question retrieval-eligible cohort.
- Common reader: `qwen/qwen3-30b-a3b-instruct-2507`, 512-token cap,
  temperature 0, no thinking, pinned to CoreWeave with fallback disabled.
- Memory processor: `qwen/qwen3-32b`, DeepInfra, no-thinking prompt-only
  LazyMem. All 29 length-ended attempts were retained in usage accounting and
  followed by a successful bounded retry.
- All 1,566 persisted answer responses came from CoreWeave, used the exact
  reader model, ended with `stop`, and reported zero reasoning tokens.
- Configuration:
  [`paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml`](configs/paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml).
- Metric: the released category-aware LoCoMo token-F1 implementation. This is
  not the LazyMem paper's binary DeepSeek-V4-Pro LLM judge.

## Usage and provenance

Tracked OpenRouter usage was **$3.27027846**:

- memory processor: 9,072 calls, 12,780,234 prompt tokens, 3,262,118 completion
  tokens, $1.93581176;
- common answerer: 1,566 calls, 13,127,317 prompt tokens, 72,450 completion
  tokens, $1.33446670.

Embedding, reranker, and failed calls that did not commit usage are outside
this total. The final configured journal-accounting fuse was $3.50.

The publishable, text-free receipt is
[`locomo314.full-answer.openrouter-no-thinking.coreweave-reader512.receipt.json`](receipts/locomo314.full-answer.openrouter-no-thinking.coreweave-reader512.receipt.json)
(SHA-256
`ebc6f4fcb4dac3b8b00196b50a16426b4aba2bb0d2dd83bb94fcd998554c9704`).
It independently checks both populations, every journal envelope and binding,
usage/response telemetry, the no-thinking contract, and the manifest-to-routing
to-response provider binding. It recomputes aggregate means, paired deltas, and
bootstrap intervals from the persisted per-question scores.

The private input artifact hashes are:

- `summary.json`: `821865d58f7d2797df79023dff22f7a218c263cfe8cedd161ed1f9bc07fb7e98`;
- `memory-contexts.jsonl`: `ac1b92a8f4331db8e9765f2145b9b388e47709fedd0a31add6e29fa5c334a90b`;
- `answers.jsonl`: `e5c5a568fbf46e9dc930002a19453329333b981ef91e74daf4da194ddc5aa72e`.

The raw journals contain dataset-derived context and predictions, are ignored
by Git, and must not be published without a separate license/privacy review.
The receipt does not contain reference labels, so it verifies the per-question
scores and recomputes all aggregates but cannot independently re-tokenize and
re-score answer text. The runner itself performed a final complete no-network
replay against the pinned dataset and official scorer before the artifacts were
sealed.

Finally, the imported Mem0 and A-MEM rows reuse cryptographically pinned raw
source-message rankings and render them through the common RAG reader. They are
not native Mem0 inferred-memory text or A-MEM rich-note contexts, so those
labels must remain `source-ranking RAG` in downstream comparisons.
