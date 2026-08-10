# Reconstructive memory benchmark

This directory contains the modular configurations for Semafuse's
training-free, source-preserving memory strategy. The design combines ideas
from [MemR3](https://arxiv.org/abs/2512.20237) and
[MemMachine](https://arxiv.org/abs/2604.04853): hybrid retrieval, a true
per-channel seen mask, evidence-gap query refinement, and local source-context
expansion. It is not an exact reproduction of either paper.

The controller never writes the answer and never rewrites stored evidence. It
can stop retrieval or emit a bounded next query. The returned memory contains
only original source events, with session, role, timestamp, anchor rank, and
source IDs retained for audit.

## Strategy profiles

All three profiles use the same embedding model, reranker, 25-anchor ceiling,
neighbor policy, 100-event ceiling, and 32,768-character final-context budget:

| Profile | Retrieval behavior | Controller calls |
| --- | --- | --- |
| `one-shot` | One query, 25 anchors | 0 |
| `masked` | Five static-query rounds, five unseen anchors per round | 0 |
| `evidence-gap` | Five-anchor rounds with sufficiency checks and query refinement | At most 4; the final round cannot affect another retrieval and is not called |

These profiles match anchor and context budgets, not compute. Masked and
evidence-gap retrieval can issue more embedding and reranker calls and can
reach deeper unseen pools. BM25 uses Top-50, so the complete comparison is an
end-to-end memory comparison rather than an equal-context experiment.

Configuration files:

- `configs/semafuse_reconstructive.one-shot.openrouter.toml`
- `configs/semafuse_reconstructive.masked.openrouter.toml`
- `configs/semafuse_reconstructive.evidence-gap.openrouter.toml`

Secrets are referenced only through `OPENROUTER_API_KEY`; no key is stored in
these files.

## LoCoMo tracks

The main full-answer suites deliberately set `stages = []`: only the memory
that a method actually returns is passed to the frozen common reader, avoiding
an extra paid answer for a raw retrieval-stage ablation.

- `../locomo/configs/paper_v2.reconstructive-ablation.openrouter.toml`: 314
  development questions from the two-conversation LazyMem selection.
- `../locomo/configs/community_full.reconstructive.openrouter.toml`: the common
  ten-conversation track with 1,540 questions, of which 1,536 have retrieval
  evidence.

Both use the released category-aware LoCoMo token-F1 implementation. That
metric is useful for fixed-reader comparisons inside this repository, but it
must not be compared directly with papers reporting binary or LLM-judge
accuracy.

Validate without model calls:

```bash
cargo run --release --locked --bin semafuse-locomo-benchmark -- \
  --mode full \
  --suite benchmarks/locomo/configs/paper_v2.reconstructive-ablation.openrouter.toml \
  --dataset /path/to/pinned/locomo10.json \
  --acknowledge-noncommercial \
  --validate-only \
  --output-dir /tmp/semafuse-reconstructive-validation
```

Run a deterministic stratified gate, then resume the same journal with a
larger or omitted limit:

```bash
target/release/semafuse-locomo-benchmark \
  --mode full \
  --suite benchmarks/locomo/configs/paper_v2.reconstructive-ablation.openrouter.toml \
  --dataset /path/to/pinned/locomo10.json \
  --acknowledge-noncommercial \
  --question-limit 8 \
  --output-dir benchmarks/locomo/results/paper-v2-reconstructive-v1
```

`--question-limit` and `--method` do not change the run identity, so methods
can be completed on different days in the same output directory. Code, method
configuration, reader configuration, and `Cargo.lock` do change the identity;
freeze them after the first paid call.

## Cost and audit scope

Every recall records per-round candidate counts, query-embedding calls,
reranker calls, controller queries, cited evidence IDs, token usage, reported
controller cost, retained event count, and context character count. Controller
traces keep the model's proposed query separate from the query actually
applied, and classify repeated-query and validation-exhaustion fallbacks. The
local fuse and summary cover controller/LazyMem processor and common-reader calls.
Embedding, reranker, transport failures, and history-index construction are not
priced by those records and must be reconciled against the provider account.
