# LoCoMo retrieval benchmark results

This development result compares five retrieval mechanisms on the same 312
evaluable questions from LoCoMo conversations `conv-49` and `conv-50`. The
evidence unit is a source message, retrieval is capped at Top-50, the table
reports Top-10 `RecallAll` and `nDCG`, and no answer reader or generation model
is evaluated. It is therefore not a LoCoMo answer-F1 result or a paper SOTA
claim.

## Frozen-runner result

| Method | RecallAll@10 | nDCG@10 | RecallAll@50 | Accounted seconds/question |
|---|---:|---:|---:|---:|
| Semafuse | 0.746795 | 0.739121 | 0.794872 | 0.4816 |
| Mem0 OSS raw | 0.605769 | 0.542536 | 0.759615 | 0.8025 |
| BM25 | 0.458333 | 0.377997 | 0.628205 | 0.0074 |
| A-MEM paper-original/common-policy | 0.349359 | 0.279275 | 0.592949 | 16.7212 |
| A-MEM reusable package/dev | 0.326923 | 0.238635 | 0.506410 | 11.0452 |

The paired 10,000-resample bootstrap puts Semafuse above BM25 by
`+0.288462` RecallAll@10 (95% CI `[0.237179, 0.339744]`) and `+0.361124`
nDCG@10 (`[0.319991, 0.402578]`). The paper-original A-MEM backend remains
below BM25 by `-0.108974` RecallAll@10 (`[-0.169872, -0.048077]`) and
`-0.098723` nDCG@10 (`[-0.149247, -0.050408]`).

Paper-original A-MEM improves over the reusable package by `+0.040640`
nDCG@10 (95% CI `[0.006887, 0.074335]`) and `+0.086538` RecallAll@50
(`[0.041667, 0.134615]`). Its `+0.022436` RecallAll@10 difference has a 95%
CI that crosses zero (`[-0.028846, 0.073718]`). This is evidence that the old
package adapter understated the paper structure, but not evidence that the
paper-original backend beats the lexical baseline on this retrieval-only
population.

## Scope and provenance

- The compact [schema-2 receipt](receipts/locomo314.development.receipt.json)
  pins four physical run directories, recomputes all metrics, verifies equal
  opaque question sets and BM25 vectors, and records every paired comparison.
- All four physical runs bind the same dataset, protocol, Cargo lock, and
  benchmark implementation hashes. External runs additionally bind the worker
  executable, Python source tree, dependency file, and complete installed
  distribution-map hash into their run keys.
- The receipt input points to a frozen copy of that exact historical Python
  tree under `receipts/sources`; the live adapters have since received security
  hardening and therefore intentionally have a different tree hash. Rebuilding
  the receipt never attributes the old runs to the newer code.
- `run_key_bound=true` does not mean dependencies are hermetic. The paper
  backend verified every one of its 72 required exact pins, while Mem0 and the
  reusable package use non-strict requirement files. The actual version maps
  are preserved under [receipts/environments](receipts/environments/README.md).
- A-MEM paper-original vendors the exact paper repository source at commit
  `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`; it still uses a hosted LLM alias
  without an immutable served-model revision, so the receipt correctly keeps
  the run in the `development` class.
- The sequential OpenRouter [usage receipt](actual_runs.json) observed a
  stable combined account window of `$0.892184194`. Per-method windows are not
  request-level invoices and can contain asynchronous settlement from an
  adjacent run.

Rebuild the compact receipt from the ignored raw result directories with:

```bash
python benchmarks/locomo/compact_receipt.py build \
  --spec benchmarks/locomo/receipts/locomo314.development.inputs.json \
  --output benchmarks/locomo/receipts/locomo314.development.receipt.json
```
