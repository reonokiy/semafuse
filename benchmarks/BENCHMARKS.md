# Semafuse memory benchmark matrix and comparison policy

This document defines which public benchmarks Semafuse should run and when two
results may be compared. It is a protocol specification, not a leaderboard.
Published numbers must not be copied into a Semafuse result table unless the
dataset revision, split, input policy, model stack, prompts, evaluator, and
resource limits are identical.

## Recommended benchmark portfolio

| Priority | Benchmark | What it measures | Public scale | Native primary metrics | License and redistribution notes | Cost and Semafuse fit |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) ([paper](https://arxiv.org/abs/2410.10813)) | Information extraction, multi-session reasoning, knowledge updates, temporal reasoning, preference use, and abstention over timestamped assistant histories | 500 questions; cleaned S histories are about 115K tokens and M histories extend to about 1.5M tokens | Official QA judge accuracy; official retrieval `recall_all@{5,10}` and `ndcg_any@{5,10}` at session level, plus `@50` at turn level. Retrieval excludes 30 abstention cases | Code and cleaned dataset are MIT. Pin the cleaned dataset revision because the ground truth was revised | Medium. Best first end-to-end and evidence-retrieval regression benchmark; already close to Semafuse's current interface |
| P0 | [LoCoMo](https://github.com/snap-research/locomo) ([paper](https://arxiv.org/abs/2402.17753)) | Personal conversational memory: single-hop, multi-hop, temporal, open-domain, and adversarial QA; optional event summarization and multimodal dialogue | Original paper: 50 conversations and 7,512 QA; current official repository release: a selected 10-conversation subset. The LazyMem reproduction subset is only 314 category 1–4 questions from two conversations | Official category-aware partial-match F1 and evidence recall for QA; the paper separately defines event-summary factual F1 and dialogue-generation metrics | Dataset is CC BY-NC 4.0. Do not redistribute the dataset or per-example text artifacts without attribution, change notices, and a non-commercial-use check; aggregate benchmark scores can still be published | Medium. High query reuse makes it affordable; strong test of temporal and multi-hop behavior, but version ambiguity must be eliminated |
| P0 | [LMEB](https://github.com/KaLM-Embedding/LMEB) ([paper](https://arxiv.org/abs/2603.12572)) | Zero-shot long-horizon retrieval across episodic, dialogue, semantic, and procedural memory | 22 datasets, 193 tasks, about 1.16 GB in the current Hub release | `nDCG@10` is the default summary metric; official code also reports capped recall `R_cap_at_10`, aggregated as Mean(Dataset) and Mean(Type), with and without task instructions | Framework code is Apache-2.0. Dataset licenses are mixed: MIT, Apache-2.0, several CC variants, copyright archives, and two sources with no declared license. Preserve per-source licensing and permit an allow-list run | Low-to-medium locally, potentially expensive through an embedding API. Directly diagnoses Semafuse ranking/embedding quality, but does **not** measure memory formation or answer generation |
| P1 | [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) ([paper](https://arxiv.org/abs/2507.05257)) | Incremental accurate retrieval, test-time learning, long-range understanding, and selective forgetting/conflict resolution | Paper v4: 2,071 questions, 103K–1.44M-token histories across document QA, LongMemEval S\*, EventQA, ICL, recommendation, summarization, detective QA, and FactConsolidation | Dataset-native accuracy/exact or substring match; recommendation Recall@5; LongMemEval LLM judge; summarization F1/LLM judge | Repository is MIT; it incorporates transformed upstream datasets, so their transitive dataset terms still apply | High for the full suite. Start with Accurate Retrieval and FactConsolidation; TTL/LRU measure abilities beyond a retrieval-only memory layer |
| P1 | [MemBench](https://github.com/import-myself/Membench) ([paper](https://arxiv.org/abs/2506.21605)) | Online participation vs observation, factual vs reflective memory, updates, capacity, and operational efficiency | Full data reports 3.5K PS-reflective, 39K PS-factual, 2K OS-reflective, and 8.5K OS-factual questions; paper test subsets cover normal and noise-extended 100K-token settings | Multiple-choice answer accuracy, evidence Recall@10 for retrieval methods, read/write time per operation, and the accuracy-vs-memory-length capacity curve | **The upstream repository currently declares no license.** Do not vendor or redistribute its data/code until the authors clarify terms. LMEB's converted task metadata is not a substitute for an upstream grant | Medium-to-high. Very useful native online-write/capacity test after licensing is resolved; better than static QA for write/update behavior |
| P1 | [AMA-Bench](https://github.com/AMA-Bench/AMA-Bench) ([paper](https://arxiv.org/abs/2602.22769), [dataset](https://huggingface.co/datasets/AMA-bench/AMA-bench)) | Memory construction and retrieval over real agent trajectories: recall, causal inference, state updating, and state abstraction | Current public test set: 208 trajectories, each with 12 QA pairs (2,496 questions), across six agent domains; trajectories range up to roughly one million tokens | Official end-to-end LLM-as-judge accuracy, reported overall and by domain/question type | Repository and Hub dataset are MIT | High because every trajectory is constructed once and judged on 12 questions. Strongest near-term agent-trajectory complement to conversational benchmarks |
| P2 | [BEAM](https://github.com/mohammadtavakoli78/BEAM) ([paper](https://arxiv.org/abs/2510.27246)) | Ten abilities including abstention, contradiction resolution, ordering, updates, instruction/preference following, multi-session and temporal reasoning, and summarization | 100 conversations and 2,000 validated questions: 20×128K, 35×500K, 35×1M, and 10×10M-token conversations | Official rubric-based LLM judge gives each criterion 0, 0.5, or 1; report mean by ability and length bucket, retaining event-ordering diagnostics | Code is MIT; BEAM data is CC BY-SA 4.0 | Very high at 1M/10M. Run 128K and 500K first as a scale curve, then 1M/10M only after the harness is stable |
| P2 | [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) ([paper](https://arxiv.org/abs/2605.12493), [dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)) | Experienced-colleague memory over multimodal web-agent trajectories: static state recall, dynamic state tracking, workflow knowledge, environment gotchas, and premise awareness | 451 manually curated questions, two domains, up to 500 trajectories and 115M tokens per haystack; public Small and Medium tiers | Answer accuracy and query latency; leaderboard uses LAFS gain over a fixed accuracy-latency frontier | Code and dataset are Apache-2.0 | Extreme storage/compute cost and multimodal engineering. Most realistic frontier test, but not the next implementation target |

Priority means implementation order, not benchmark quality. The minimum useful
portfolio is LongMemEval + LoCoMo + LMEB. Adding MemoryAgentBench AR/SF and
AMA-Bench broadens it from conversational retrieval to updating and agent
trajectories. BEAM and LongMemEval-V2 are scale/frontier gates.

### Version hazards that must be recorded

- LoCoMo's paper statistics (50 conversations), current repository subset (10),
  and the LazyMem two-conversation/314-question subset are three different test
  sets.
- `LongMemEval-S` (500 independent questions), LazyMem's stratified 100-question
  test split, and MemoryAgentBench's `LongMemEval_S*` (five histories reused for
  300 questions) are different workloads.
- MemoryAgentBench paper v4 calls the fourth competency **Selective
  Forgetting**; parts of the repository/earlier descriptions call the same
  FactConsolidation family **Conflict Resolution**. Pin a commit and preserve
  its task IDs instead of joining by display name.
- LongMemEval's official retrieval evaluator omits abstention questions, while
  its QA evaluator includes them with a dedicated unanswerable-question prompt.

## Two result tracks

Every run must declare exactly one of these tracks. Results from different
tracks may be shown next to each other for context, but never ranked together.

### Track A: `paper-reproduction`

Use this track only when reproducing a named paper/table. Freeze all of the
following to the paper's specification:

1. Dataset artifact, revision, split-generation code and seed.
2. History roles, record/session granularity, chronology and image-caption
   policy.
3. Memory-construction model, embedding model, reranker, vector store, native
   method parameters, retrieval depth, and context assembly.
4. Reader model, thinking mode, prompt, tokenizer, context budget, decoding
   parameters, and maximum output length.
5. Judge/evaluator model, prompt, decoding parameters and score parser.
6. Warm-up, repetition, concurrency, cache, hardware and latency boundary.

If any item changes, name the run `paper-reproduction-substituted` and compare
only methods rerun under that same substituted stack. It is not a reproduction
of the paper's published values.

#### LazyMem anchor protocol

For the current Semafuse/LazyMem comparison, reproduce the setup documented in
the [LazyMem paper](https://arxiv.org/abs/2607.22690), rather than mixing it with
the native LoCoMo or LongMemEval protocols:

- LongMemEval: sort IDs, stratify by category with seed 42 into 360/40/100, and
  evaluate all methods on the same 100-question test split.
- LoCoMo: use the paper's 314 questions from its two pinned conversations and
  categories 1–4. This is not the current 10-conversation LoCoMo release.
- Use Qwen3-32B with thinking enabled as both the answer model and prompt-only
  memory processor where applicable.
- Reproduce the retrieval stack when evaluating the paper method: hybrid dense
  + Okapi BM25 candidate depth 50, Qwen3-Embedding-8B,
  BGE-reranker-v2-M3, neighboring radius 2, at most 8 windows, stride 7.
- Use the paper's binary DeepSeek-V4-Pro judge at temperature 0 and the exact
  benchmark prompts/decision rules. A different judge creates a substituted
  track, even if the answer model is unchanged.
- Measure answer-context tokens with the Qwen3-32B tokenizer, excluding the
  question and fixed prompt. Run questions sequentially; discard 3 warm-ups and
  measure the following 3 repetitions for latency. Record any construction
  concurrency separately.

Paper-reproduction results may allow different memory-internal models when the
paper explicitly defines them per method. That tests the published systems as
configured; it does not isolate memory architecture.

### Track B: `common-policy-v1`

This is the primary Semafuse-vs-baseline engineering comparison. All methods
are rerun locally with the same external budgets. Published paper values are
not inserted into this table.

#### Dataset and ingestion

- Pin repository and dataset commit SHA plus file SHA-256. Use the full public
  test set unless the manifest names a pre-registered subset.
- Stream records in chronological order. Preserve native session/turn
  boundaries and timestamps. Never expose answers, evidence IDs, category
  labels, future turns, or judge rubrics to memory construction or retrieval.
- Store both user and assistant messages when the dataset contains both. A
  user-only ablation is a separate run.
- Reuse one constructed memory for all queries that share a history. Clear all
  state between independent users/histories. Disable cross-example caches.
- Methods may choose their native memory units, but the returned answer context
  is normalized to plain UTF-8 text plus source IDs before the common reader.

#### Retrieval and context budget

Use these defaults unless a benchmark's official protocol mandates another
value:

| Setting | Common value | Reporting rule |
| --- | --- | --- |
| First-stage candidate depth | `candidate_k = 50` | Applies to methods exposing candidates; record `not_applicable` for fully agentic retrieval |
| Final retrieval depth | `final_k = 10` | Return fewer only when the method abstains or the token cap is reached |
| Answer-context cap | `16,384` tokens under the reader tokenizer | The token cap is authoritative; `k` alone is not comparable when entries have different sizes |
| Evidence metrics | Recall-all and nDCG-any at 5 and 10; also at 50 when labels and candidates exist | Report at native evidence unit (turn/session/document) and do not merge units into one score |
| Context order | Chronological after retrieval/reranking | Record a relevance-order ablation separately |

Report two non-mergeable common-policy panels when possible:

- **Native system:** each method uses its intended extractor, embedder, reranker,
  and storage backend. This measures deployable system quality.
- **Backbone-matched:** all compatible methods use the same embedding and
  reranking models. This better isolates memory organization. Methods whose
  algorithm cannot accept the matched backbone remain in the native panel.

#### Reader and judge

- Default reader: the same pinned Qwen3-32B deployment for every method,
  thinking enabled, temperature 0 where the serving API honors it, and a fixed
  maximum output length. Save the exact request payload and model/provider
  revision; a model alias is insufficient.
- Use deterministic official metrics whenever available. For open-ended QA,
  use the benchmark's native evaluator first.
- For the cross-method binary-judge panel, use one pinned DeepSeek-V4-Pro
  deployment at temperature 0 with the exact same prompt and parser for all
  methods. If unavailable, select one replacement judge, rerun **all** methods,
  and label the complete panel `judge-substituted`; never compare it with
  published LazyMem values.
- Keep native F1/exact-match, native LLM-judge, retrieval, and common binary
  judge results in separate columns. No averaging across metric families.
- Bootstrap 95% confidence intervals over question IDs. For histories with many
  questions, additionally cluster-bootstrap by history so query reuse does not
  create falsely narrow intervals.

#### Latency and cost

- Measure construction, retrieval, answer generation, and judging as separate
  phases. Retrieval latency ends when final context text/source IDs are
  returned; generation latency starts after that boundary.
- Report cold construction once, warm sequential query p50/p95, and end-to-end
  p50/p95. Run with query concurrency 1, discard 3 warm-ups, then measure 3
  repetitions. Record construction concurrency and batch size.
- Synchronize GPU work before timestamps. Record CPU/GPU model, RAM/VRAM,
  storage backend, index size, software commit, and whether caches are warm.
- Report input/output tokens and actual billed USD separately for construction,
  retrieval-side LLMs, reader, and judge. Also report amortized construction
  cost per question; never hide judge spend inside generation cost.
- Record peak RSS/VRAM and on-disk index bytes. A quality score without resource
  data is not sufficient for a system comparison.

## External comparison methods

Run external methods from pinned upstream commits. Do not reimplement their
core algorithm inside Semafuse for headline comparisons; an adapter should only
translate the benchmark's chronological `add` and `search` operations.

| Method | Upstream implementation | License | Required comparison treatment |
| --- | --- | --- | --- |
| Mem0 OSS | [mem0ai/mem0](https://github.com/mem0ai/mem0) | Apache-2.0 | Use the self-hosted OSS library, not hosted Mem0 unless explicitly labeled. Pin extraction LLM, embedder, vector store, history role policy, `top_k`, and graph-memory setting. Do not mix vendor-published scores with OSS reruns |
| A-MEM | [agiresearch/A-mem](https://github.com/agiresearch/A-mem) for the system; [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem) for paper reproduction/evaluation | MIT | Paper and production repositories are distinct. Pin which one is used, preserve note construction/linking/evolution calls, and include its write-side LLM/token cost |
| LightMem | [zjunlp/LightMem](https://github.com/zjunlp/LightMem) | MIT | Prefer the official LoCoMo/LongMemEval reproduction and unified baseline harness. Pin offline-update, compression, topic segmentation, message-role, embedding and retrieval settings; measure offline work as construction cost |

The LightMem repository can serve as a reference adapter for running Mem0,
A-MEM, and LightMem on the same datasets, but its wrappers still need auditing
against each upstream method. A wrapper result is not automatically identical
to a method paper's result.

Always include non-learned controls in both tracks when the protocol allows it:
chronological recent memory, BM25, dense retrieval, hybrid/RRF, oracle evidence,
and full context when it fits. Oracle separates reader/judge errors from memory
errors; full context exposes benchmarks that no longer discriminate retrieval.

## Scores that cannot be compared directly

The following pairs must not share a ranking or a delta column:

- LongMemEval retrieval Recall/nDCG versus downstream QA judge accuracy.
- LoCoMo category-aware token F1 versus a binary LLM-judge score.
- Any LoCoMo 50-, 10-, or 2-conversation result against another release.
- LongMemEval full-500, LazyMem test-100, and MemoryAgentBench S\*-300 results.
- Turn-level retrieval against session-, document-, note-, or window-level
  retrieval without reporting each evidence unit separately.
- User-only history against all-role history.
- Native-system results against backbone-matched results.
- Different answer models, thinking modes, prompts, token caps, judges, or
  judge prompts—even when the dataset is identical.
- Hosted product results against an OSS implementation with different memory
  extraction, embedding, vector store, or graph features.
- Warm-cache latency against cold construction or a run with different query
  concurrency/hardware.
- LMEB retrieval-only scores against any end-to-end answer score.
- LongMemEval-V2 leaderboard LAFS against raw accuracy or latency alone.

## Execution order

1. **Regression set:** LongMemEval cleaned full 500 (470 retrieval-eligible),
   LoCoMo current 10-conversation native QA, and LMEB dialogue/episodic allow-list.
2. **Paper table:** LazyMem LongMemEval-100 and LoCoMo-314, with Semafuse,
   BM25/dense/hybrid controls, Mem0 OSS, A-MEM, and LightMem under the exact or
   explicitly substituted stack.
3. **Update and agent coverage:** MemoryAgentBench Accurate Retrieval +
   FactConsolidation, then AMA-Bench full public test.
4. **Scale gates:** BEAM 128K, then 500K/1M/10M; LongMemEval-V2 Small before
   Medium.
5. **Native online capacity:** MemBench only after upstream licensing is
   clarified; do not commit or redistribute its artifacts before then.

## Required result identity

Each result directory must contain a machine-readable manifest with at least:

```text
track, benchmark_name, benchmark_version, dataset_revision, dataset_sha256,
split_name, split_seed, question_ids_sha256, method_name, method_commit,
method_config_sha256, history_roles, memory_unit, candidate_k, final_k,
context_token_cap, reader_model_revision, reader_request_sha256,
embedding_model_revision, reranker_model_revision, judge_model_revision,
answer_prompt_sha256, judge_prompt_sha256, evaluator_commit,
hardware, construction_concurrency, query_concurrency, cache_state,
started_at, code_commit
```

Result tables should link the manifest and raw per-question journal. A row
without this identity is exploratory evidence, not a benchmark result.
