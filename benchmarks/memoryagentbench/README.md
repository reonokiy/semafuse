# MemoryAgentBench harness

This directory provides a small, reproducible bridge from the official
MemoryAgentBench data to Semafuse or any other memory backend. It deliberately
supports only the training-free, retrieval-compatible main-paper tasks:

| Source | Paper task | Contexts | Questions | Paper chunk size | Primary metric |
| --- | --- | ---: | ---: | ---: | --- |
| `ruler_qa1_197K` | SH-QA | 1 | 100 | 512 | SubEM accuracy |
| `ruler_qa2_421K` | MH-QA | 1 | 100 | 512 | SubEM accuracy |
| `eventqa_full` | EventQA | 5 | 500 | 4096 | SubEM accuracy |
| `factconsolidation_sh_262k` | FC-SH | 1 | 100 | 512 | SubEM accuracy |
| `factconsolidation_mh_262k` | FC-MH | 1 | 100 | 512 | SubEM accuracy |

The paper calls the fourth competency *Selective Forgetting*. The current Hub
dataset and repository README call its split `Conflict_Resolution`; this harness
keeps both names visible rather than silently treating the naming change as a
new task.

## What is and is not comparable

The paper's main-table numbers are end-to-end answer scores. For RAG and memory
systems, the protocol is:

1. Split each context by NLTK sentences and GPT-4o-mini token counts.
2. Inject chunks sequentially into a fresh memory for that context.
3. Retrieve Top-10 memories for each question.
4. Give the retrieved memories and official prompt to GPT-4o-mini.
5. Average official normalized substring exact match over every question.

Consequently, a Semafuse result is directly comparable to the paper only when
it uses the same source, full question set, paper chunk size, Top-10 setting,
RAG prompt, GPT-4o-mini reader, 0.7 temperature, and output limit. The upstream
model name is an undated API alias, so exact historical model reproducibility is
not possible from the released artifacts alone.

The scorer also reports `answer_coverage_at_k`: whether any normalized gold
answer string occurs in the retrieved chunks. This is useful for debugging the
retriever without paying for answer generation, but it is **not an official
MemoryAgentBench metric** and must not be compared with the paper's answer
accuracy.

Two companion diagnostics prevent this proxy from being over-interpreted:
`answer_surface_oracle_coverage` measures whether the answer string appears
anywhere in the full memory, and `answer_coverage_given_surface_present`
restricts the denominator to those surface-answer-present questions. This is
essential for EventQA, whose generated next-event answers are usually not
verbatim spans in the novel. FC-MH has a different limitation: finding the
answer string does not prove that all multi-hop evidence was retrieved.

Published main-table context (percent), copied into the pinned manifest for
machine-readable reporting:

| Method | SH-QA | MH-QA | EventQA | FC-SH | FC-MH |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 66.0 | 56.0 | 74.6 | 48.0 | 3.0 |
| Text-Embed-3-Small | 60.0 | 44.0 | 63.0 | 28.0 | 3.0 |
| HippoRAG-v2 | 76.0 | 66.0 | 67.6 | 54.0 | 5.0 |
| Mem0 | 25.0 | 32.0 | 37.5 | 18.0 | 2.0 |
| Cognee | 31.0 | 26.0 | 26.8 | 28.0 | 3.0 |

These values remain reference context, not locally reproduced results. The
official loader used `revision="main"`; it did not record the immutable Hub
commit used for the table. This harness pins the current official snapshot and
therefore does not claim bit-identical historical data.

There is also a method-policy mismatch inside the released artifacts: the paper
states TopK=10 and 512-token chunks for synthetic AR/SF tasks, but expensive
memory methods use 4096-token chunks, and the current official Mem0 YAML sets
`retrieve_num: 100`. Thus the published Mem0 row is useful context, but it is
not a controlled retriever-only comparison against Semafuse. A fair Semafuse
versus Mem0 experiment should rerun both systems with one shared chunk size,
Top-K, reader, and prompt and label that result as a separate common-policy
track.

## Pinned sources

- Dataset: [`ai-hyz/MemoryAgentBench`](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
  at `7ea066982b140a19337e17e60d45d4076e042faf`, Hub license `mit`.
- Reference code:
  [`HUST-AI-HYZ/MemoryAgentBench`](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
  at `455306dcabc3842526eb83cd4e225e5d486c5c5d`, MIT license.
- Paper: [arXiv:2507.05257](https://arxiv.org/abs/2507.05257), published at
  [ICLR 2026](https://openreview.net/forum?id=DT7JyQC3MR).
- Exact parquet sizes and LFS SHA-256 values are in
  `manifests/official-main-v2.json`.

The Hub snapshot has 22 Accurate Retrieval rows, 6 Test-Time Learning rows,
110 Long-Range Understanding rows, and 8 Conflict Resolution rows. The harness
validates the expected context and question counts for every supported full
run.

## Install and prepare

Use a dedicated environment because the official chunker needs `datasets`,
`nltk`, and `tiktoken`:

```bash
cd benchmarks/memoryagentbench
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m nltk.downloader punkt punkt_tab
```

Inspect the immutable configuration without downloading data:

```bash
.venv/bin/python -m memoryagentbench_harness inspect-manifest
```

Explicitly download and hash-check the two Parquet files needed by the five
supported tasks:

```bash
.venv/bin/python -m memoryagentbench_harness verify-data \
  --cache-dir prepared/hf-cache/hub \
  --split Accurate_Retrieval \
  --split Conflict_Resolution \
  --output results/data-verification.json
```

Prepare a complete source:

```bash
.venv/bin/python -m memoryagentbench_harness prepare \
  --source ruler_qa1_197K \
  --output-dir prepared/ruler-sh
```

This writes:

- `inputs.jsonl`: public `reset`, `add`, and `query` events for the system under
  test.
- `labels.jsonl`: evaluator-only answers and upstream audit IDs.
- `run.json`: source revisions, settings, counts, comparability flag, and
  manifest/input/label SHA-256 values.

The public input schema has an allowlist. Answers, ground-truth fields, question
types, and upstream QA IDs cannot enter `inputs.jsonl`. Opaque case/query IDs are
derived only from source position. Gold answers may naturally occur in the
memory text; that is the evidence the system is expected to retrieve, not label
leakage.

`--fixture`, `--max-contexts`, or a non-paper `--chunk-size` is permitted for
development, but `run.json` then records `paper_comparable_shape: false`.

## Backend and prediction contract

Process events in file order. `reset` starts an isolated context, each `add`
injects one memory, and every `query` asks against the completed context. For a
paper-track RAG comparison, ingest and read `paper_message`; `text` is the raw
chunk or question for systems that keep prompts outside the memory store.

Write one JSON object per query:

```json
{
  "schema_version": 1,
  "case_id": "case-...",
  "query_id": "query-...",
  "prediction": "the reader's final answer",
  "retrieved_memory_ids": ["memory-...", "memory-..."]
}
```

`prediction` may be `null` for retrieval-only diagnostics. Retrieved IDs must
belong to the same case, be unique, and not exceed the query's Top-K. Missing,
duplicate, cross-case, and unknown IDs fail closed.

Score a completed run:

```bash
.venv/bin/python -m memoryagentbench_harness score \
  --inputs prepared/ruler-sh/inputs.jsonl \
  --labels prepared/ruler-sh/labels.jsonl \
  --predictions results/semafuse-ruler-sh.jsonl \
  --output results/semafuse-ruler-sh.report.json
```

The official primary value is
`metrics_percent.substring_exact_match`. It is absent when all predictions are
retrieval-only. Artifact hashes in the report prevent accidentally scoring a
different input/label pair.

## Complete retrieval-only suite

The full local diagnostic prepares all five main tasks (900 questions), verifies
the pinned source files, runs BM25, scores answer coverage, records per-stage
wall time, and hashes every artifact:

```bash
benchmarks/memoryagentbench/.venv/bin/python -m \
  memoryagentbench_harness run-suite \
  --output-dir benchmarks/memoryagentbench/results/bm25-full \
  --cache-dir benchmarks/memoryagentbench/prepared/hf-cache
```

The primary suite artifact is `results/bm25-full/suite-report.json`. It records
the exact command, Python/platform metadata, direct and transitive package
versions, verified Hub object hashes, the harness-tree hash, source-level
settings and timings, and SHA-256 for inputs, labels, predictions, and reports.
The adjacent `SHA256SUMS` also includes the suite report itself and can be
checked with `sha256sum --check SHA256SUMS` from the result directory.
Both `results/` and `prepared/` are ignored; neither raw data nor generated
labels should be committed.

A compact, committable receipt for the verified 900-question BM25 run is in
[`reports/bm25-full-2026-08-08.json`](reports/bm25-full-2026-08-08.json). It
contains exact task metrics, counts, timings, source and harness hashes, and the
relative reproduction command, but no raw data, absolute paths, package dump,
reader output, or credentials.

## Deterministic BM25 diagnostic

The optional local diagnostic uses the mature `rank-bm25==0.2.2` package and
reproduces the official code's whitespace tokenization, default BM25Okapi
parameters, query extraction, and Top-10 shape:

```bash
.venv/bin/python -m memoryagentbench_harness bm25 \
  --inputs prepared/ruler-sh/inputs.jsonl \
  --output results/bm25-ruler-sh.retrieval.jsonl
```

Score that file to obtain only `answer_coverage_at_k`. It does not generate an
answer and therefore does not reproduce the paper's BM25 accuracy.

## Unsupported benchmark competencies

The other splits are intentionally not exposed as Semafuse retrieval scores:

- Test-Time Learning evaluates learning a new label mapping and movie
  recommendation behavior from demonstrations. It is not a static memory
  retrieval benchmark, even though RAG systems can participate end to end.
- Long-Range Understanding requires global novel summarization or whole-book
  reasoning. A Top-K retriever alone is insufficient, and the summarization
  task uses an LLM judge.
- LongMemEval (S*) inside Accurate Retrieval uses answer generation and, for the
  released evaluation path, an LLM judge. Semafuse already has a native pinned
  LongMemEval harness, so duplicating it here would make protocol drift more
  likely.
- Shorter EventQA and FactConsolidation sources are context-length ablations,
  not the main-paper comparison rows.

No task in this harness requires training. FC-MH still requires multi-hop
reasoning after retrieval, so retrieval coverage alone cannot measure selective
forgetting or conflict resolution.

## Tests

The unit suite is offline and does not install packages, download Hub data, or
call a paid API:

```bash
python -m unittest discover -s benchmarks/memoryagentbench/tests -v
```

It covers official normalization/SubEM, exact sentence packing behavior, stable
artifact IDs and hashes, label isolation, count/top-k validation, cross-case
retrieval rejection, and separation of official versus diagnostic metrics.
