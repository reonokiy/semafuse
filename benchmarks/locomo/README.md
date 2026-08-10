# LoCoMo retrieval and end-to-end answer benchmarks

This directory configures the native `semafuse-locomo-benchmark` runner. It
uses the pinned paper-v2 LoCoMo selection:

- `conv-49` and `conv-50`;
- official categories 1–4;
- 314 QA items, of which exactly 312 contain retrieval evidence;
- source-message evidence metrics at 1, 5, 10, and 50, aggregated overall and
  by category with paired 95% bootstrap intervals.

The runner accepts only the pinned upstream `locomo10.json` SHA-256. Retrieval
mode persists only opaque IDs and metrics; full mode necessarily keeps rendered
memory contexts and answer predictions in private, gitignored journals. Do
**not** add the official dataset or full-mode journals to this repository.

See [RESULTS.md](RESULTS.md) for the frozen-runner retrieval comparison and
[FULL_ANSWER_RESULTS.md](FULL_ANSWER_RESULTS.md) for the completed end-to-end
answer comparison and its reproducibility limits.

The runner has two separate protocols:

- `--mode retrieval` scores source-message evidence on the 312 questions that
  contain valid evidence annotations. It never calls the LazyMem processor.
- `--mode full` runs real Semafuse memory construction, a method-neutral common
  answer prompt/model, and the released category-aware LoCoMo token-F1 scorer.
  Native/Semafuse outputs cover all 314 questions; imported source-ranking RAG
  comparisons use the exactly paired 312-question retrieval-eligible cohort.
  This scorer is official LoCoMo code, but it is **not** the binary
  DeepSeek-V4-Pro LLM judge reported in the LazyMem paper.

Full mode writes two independent, private, append-and-sync journals:
`memory-contexts.jsonl` and `answers.jsonl`. An interrupted answer run reuses
completed query-conditioned memories after their main record has committed.
Calls from a question that fails before that commit, and shared history
embedding/index construction after a process restart, can repeat. These files
contain dataset-derived text, stay under the ignored
`results/` directory, and must not be published without a separate license and
privacy review.

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

## End-to-end full-memory comparison

`paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml` is the completed
development profile. It configures one common Qwen3-30B-A3B-Instruct-2507
answerer and five answer outputs:

- BM25 Top-50;
- complete prompt-only LazyMem (retrieve, expand, process overlapping windows,
  KEEP/DROP/compress, merge);
- the raw final retrieval frontend from that exact same LazyMem call;
- Mem0 OSS raw retrieval's pinned source ranking, rendered as raw-message RAG;
- the audited original A-MEM run's pinned source ranking, rendered as
  raw-message RAG.

The Semafuse pair shares retrieval and processor execution. Only the complete
row owns processor usage; the `::retrieved` row is a shared-call ablation. The
Mem0 and A-MEM rows do not rerun either external memory system: the suite
validates the prior retrieval journals and summaries by SHA-256, run key,
dataset identity, source method, unique question set, and source IDs before any
model call. They therefore measure `source_ranking_rag`, not Mem0's inferred
memory text or A-MEM's native rich-note context. The full report has two paired
panels: 314 questions for BM25/Semafuse and 312 for all five outputs.

Credentials remain only in `OPENROUTER_API_KEY`; no key value belongs in a
TOML, JSON, journal, receipt, or committed file. Validate all identities without
calling a model first:

```bash
target/release/semafuse-locomo-benchmark \
  --mode full \
  --suite benchmarks/locomo/configs/paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml \
  --dataset /absolute/path/to/locomo10.json \
  --acknowledge-noncommercial \
  --validate-only \
  --output-dir benchmarks/locomo/results/full-answer-validation
```

This writes `validation.json`, never `summary.json` or either paid journal.

First run a stratified 16-question LazyMem-only pilot. The scheduling flags do
not change run identity, so the unrestricted command resumes committed records
from the same journal. Start with the cost-controlled no-thinking profile when
the account has limited credit:

```bash
cargo build --release --locked --bin semafuse-locomo-benchmark

target/release/semafuse-locomo-benchmark \
  --mode full \
  --suite benchmarks/locomo/configs/paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml \
  --dataset /absolute/path/to/locomo10.json \
  --acknowledge-noncommercial \
  --question-limit 16 \
  --method lazymem_prompt_only_no_thinking \
  --output-dir benchmarks/locomo/results/full-answer-openrouter-instruct2507-reader512-coreweave-v1
```

Inspect `summary.json`, the processor completion-tokens-per-call field, the
reported/estimated usage, response provider, and reasoning-token telemetry.
Also compare the provider account balance before and after the pilot because
embedding, reranking, and failed or uncommitted completion calls whose usage was
not journaled are outside the local fuse. Then resume every declared 314/312
cohort and output:

```bash
target/release/semafuse-locomo-benchmark \
  --mode full \
  --suite benchmarks/locomo/configs/paper_v2.full-no-thinking.openrouter-reader512-coreweave.toml \
  --dataset /absolute/path/to/locomo10.json \
  --acknowledge-noncommercial \
  --output-dir benchmarks/locomo/results/full-answer-openrouter-instruct2507-reader512-coreweave-v1
```

The suite checks its reported/estimated processor-plus-answer usage before each
question-level Semafuse call and each answer call, and stops with resumable
journals after the configured fuse is reached. Loaded journals are fully
validated before any paid call; the retrieval-score check tolerates only a
one-ULP serialization drift. A single question may contain
multiple concurrent window calls, so the fuse is not a hard provider-account
limit. Embedding, reranker, and failed or uncommitted calls missing from the
journals are excluded; the bundled `$3.50` fuse was calibrated from the
stratified 16-question pilot and intentionally leaves separate account headroom.

For a thinking-enabled public-reproduction profile,
`paper_v2.full.openrouter.toml` enables the processor's standard reasoning
control and the paper reconstruction's decoding cap. Its independent fuse may
stop before the no-thinking profile; use a different output directory because
suite/model options are part of run identity. In
the no-thinking profile, the memory processor pins Qwen3-32B on DeepInfra and
combines Qwen3's `/no_think` soft switch with OpenRouter's
`reasoning.enabled=false`. Live validation showed that either control alone can
be ignored for some requests, while the dual control preserved ordinary and
structured-output content. The transport still rejects substantive processor
reasoning in this mode (one delimiter token is allowed). Structured KEEP
content and audit reasons are capped at 512 and 256 Unicode characters,
respectively; null or non-`stop` structured completions retain their usage and
enter the bounded validation-retry path. The first attempt always uses the
paper-style prompt unchanged; a retry adds one idempotent correction that asks
for 160-character content and 32-character reasons so repeating a truncated
request is not merely another identical draw. All methods then share
Qwen3-30B-A3B-Instruct-2507 on CoreWeave as an inherently non-thinking common
reader after StreamLake repeatedly returned shared-pool 429s. Fallback is
disabled, no model-specific thinking parameter is sent, and returned model,
provider, finish reason, and reasoning telemetry are checked before persistence.
This lowers the equal downstream reader cost without changing pairwise
comparability. The profile
preserves the memory architecture but changes processor decoding, so it is a
`custom/no-thinking` ablation, not a public paper-reproduction score.

After a complete run, `compact_answer_receipt.py` can validate the private
journals and emit a publishable text-free receipt. Pin all three input hashes so
an accidental artifact replacement fails closed. The completed run was sealed
with:

```bash
python3 benchmarks/locomo/compact_answer_receipt.py \
  --summary benchmarks/locomo/results/full-answer-openrouter-instruct2507-reader512-coreweave-v1/summary.json \
  --memory-journal benchmarks/locomo/results/full-answer-openrouter-instruct2507-reader512-coreweave-v1/memory-contexts.jsonl \
  --answers-journal benchmarks/locomo/results/full-answer-openrouter-instruct2507-reader512-coreweave-v1/answers.jsonl \
  --expected-summary-sha256 821865d58f7d2797df79023dff22f7a218c263cfe8cedd161ed1f9bc07fb7e98 \
  --expected-memory-sha256 ac1b92a8f4331db8e9765f2145b9b388e47709fedd0a31add6e29fa5c334a90b \
  --expected-answers-sha256 e5c5a568fbf46e9dc930002a19453329333b981ef91e74daf4da194ddc5aa72e \
  --output benchmarks/locomo/receipts/locomo314.full-answer.openrouter-no-thinking.coreweave-reader512.receipt.json
```

The validator binds the manifest's expected provider to the exact OpenRouter
routing order and then to every persisted answer response. It also recomputes
the paired scalar reports and bootstrap intervals without serializing raw
context, prediction, credential, or absolute-path data. See
[FULL_ANSWER_RESULTS.md](FULL_ANSWER_RESULTS.md) for the verified scores and
limitations.

## Mem0/A-MEM external worker bridge

Retrieval mode's `kind = "external_jsonl"` reuses the process-isolated protocol under
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
source_root = "../../external_memory/src/semafuse_external_memory"
dependency_lock = "../../external_memory/methods/mem0_oss/requirements.txt"
top_k = 50
max_context_tokens = 16384
timeout_seconds = 600
```

Live external workers are deliberately rejected by full-answer mode because
their request-level cost and construction checkpoint cannot be bounded. Full
mode instead consumes the pinned, read-only retrieval journals described above.

The retrieval runner binds the selected executable bytes, the complete configured Python
source tree, the dependency-lock bytes, and an inventory of every distribution
installed in the actual invocation interpreter into the manifest and run key.
Files named `requirements.lock.txt` are additionally parsed fail-closed and all
exact pins must match that environment before execution. The runner then
appends `--config <resolved-worker-config>` to `command`. External
items may cite multiple source messages; the common evaluator flattens source
IDs in native item order, keeps the first occurrence, and caps the message
ranking at 50. Unknown provenance fails the run. Use each upstream's locked
worker config and isolated environment; the bridge does not reimplement Mem0
or A-MEM.

## Compact multi-run receipt

`compact_receipt.py` independently verifies and joins completed Semafuse,
Mem0, and A-MEM run directories without changing the Rust evaluator. It uses
only the Python standard library and never reads `.env` or makes a network
request.

First validate each run and obtain the hashes needed by a comparison spec:

```bash
python benchmarks/locomo/compact_receipt.py inspect-run \
  --run-dir benchmarks/locomo/results/semafuse-openrouter-v2

python benchmarks/locomo/compact_receipt.py inspect-external-harness \
  --source-dir benchmarks/external_memory/src/semafuse_external_memory
```

`inspect-run` checks the complete journal, not only its first row. Its output is
secret-free and contains the byte hashes for `summary.json` and
`per-question.jsonl`, the canonical embedded-manifest hash, the run key, method
IDs, and a hash of the 312 opaque question IDs. If a run also has a standalone
`manifest.json`, its bytes must equal the embedded manifest and its hash must be
pinned separately.

Create a JSON spec such as:

```json
{
  "schema_version": 2,
  "label": "development",
  "baseline": "BM25",
  "external_harness": {
    "directory": "../../external_memory/src/semafuse_external_memory",
    "tree_sha256": "<inspect-external-harness tree_sha256>",
    "adapter_sha256s": {
      "adapters/amem.py": "<inspect-external-harness adapter sha256>",
      "adapters/amem_paper.py": "<inspect-external-harness adapter sha256>"
    }
  },
  "runs": [
    {
      "id": "semafuse-run",
      "directory": "../results/semafuse-openrouter-v2",
      "summary_sha256": "<sha256>",
      "per_question_sha256": "<sha256>",
      "manifest_sha256": "<sha256>"
    },
    {
      "id": "mem0-run",
      "directory": "../results/mem0-openrouter-v2",
      "summary_sha256": "<sha256>",
      "per_question_sha256": "<sha256>",
      "manifest_sha256": "<sha256>"
    },
    {
      "id": "amem-run",
      "directory": "../results/amem-package-openrouter-v2",
      "summary_sha256": "<sha256>",
      "per_question_sha256": "<sha256>",
      "manifest_sha256": "<sha256>"
    },
    {
      "id": "amem-paper-original-run",
      "directory": "../results/amem-paper-original-openrouter",
      "summary_sha256": "<sha256>",
      "per_question_sha256": "<sha256>",
      "manifest_sha256": "<sha256>"
    }
  ],
  "methods": [
    {"name": "BM25", "run": "semafuse-run", "source_method": "bm25"},
    {"name": "Semafuse", "run": "semafuse-run", "source_method": "semafuse::retrieved"},
    {"name": "Mem0", "run": "mem0-run", "source_method": "mem0_oss_raw"},
    {"name": "A-MEM package/dev", "run": "amem-run", "source_method": "amem_openrouter"},
    {
      "name": "A-MEM paper-original/common-policy",
      "run": "amem-paper-original-run",
      "source_method": "amem_paper_original"
    }
  ]
}
```

Paths are resolved relative to the spec and are never copied into the receipt.
Build the final artifact with:

```bash
python benchmarks/locomo/compact_receipt.py build \
  --spec benchmarks/locomo/receipts/locomo314.development.inputs.json \
  --output benchmarks/locomo/receipts/locomo314.development.receipt.json
```

The repository includes that verified development spec and receipt for five
methods assembled from four runs: BM25 and Semafuse, Mem0 OSS raw, the reusable
A-MEM package, and the A-MEM paper-original backend under the common policy.
The large run directories remain ignored; the spec pins their summary, journal,
and embedded manifest hashes, while the compact receipt contains only aggregate
metrics, latency accounting, immutable identities, and validation claims.
Rebuilding the receipt requires the corresponding local result directories and
fails if any pinned byte changes.

`actual_runs.json` records the sequential OpenRouter account-usage observations
for those runs. Its combined settled window is stronger evidence than the
individual method windows because gateway usage may settle asynchronously. The
per-method values are therefore labeled observed account-window deltas, not
request-level invoices.

The builder fails closed on byte-hash changes; duplicate JSON keys; malformed
or non-opaque IDs; mixed envelope, manifest, report, or payload run keys;
missing or duplicate method/question rows; changed evaluator metadata; unequal
question sets; unequal BM25 metric vectors; cross-run dataset, protocol, Cargo
lock, or evaluator-code drift; and summary means that cannot be reproduced from
the journal.

Every receipt reports `RecallAll@10`, `nDCG@10`, and `RecallAll@50` in fraction
units. It uses one shared question-level paired bootstrap for all methods and
deltas: seed `20260808`, 10,000 resamples, a specified SplitMix64 stream, and
linear percentile 95% intervals. It emits candidate-minus-BM25 deltas plus all
non-baseline method pairs, including wins, ties, and losses.

Each selected method also receives a deterministic `latency` receipt derived
from its 312 hash-pinned journal rows. `query_wall_s` contains total, mean,
linear p50, and linear p95. `shared_construction_s` never sums the repeated
per-query field: Semafuse rows are grouped by the validated anonymous
`history_identity_sha256`; methods without that identity are accepted only when
each exact `(shared_construction_s, construction_amortization_n)` bucket covers
an integral number of full shared attempts. The completed paper-v2 selection
must prove the exact amortization-denominator multiset `[156, 156]`: one complete
construction attempt for each source conversation. Merely partitioning 312 rows
into another pair such as 100 and 212 is rejected. Mixed, partial, inconsistent,
or unscoped accounting fails closed. Group IDs are not serialized. The receipt
reports construction total and amortized time per question, plus
`accounted_wall_s` as query-wall total plus construction counted once per
validated attempt. Stateless baselines report zero shared construction.

Selected method identities accept only the four evaluator method kinds, validate
hashes and immutable revisions by type, and preserve model IDs only when they
match the bounded `organization/model` identifier grammar. Before the receipt
declares that credentials and absolute paths were not serialized, a recursive
scan rejects credential-shaped keys and values, POSIX/Windows/home-relative
absolute paths, `file://` paths, and credential-bearing URIs. Comparison specs
also reject duplicate `(run, source_method)` selections, and `bm25` may only be
selected under the declared baseline display name.

Only `development` and `paper-reproduction` labels are accepted. A paper claim
is rejected unless every run declares itself paper-comparable and all selected
model-backed methods provide the required protected immutable identities. The
current retrieval-only and development manifests therefore cannot silently
become paper-reproduction results.

The spec also pins a deterministic hash over every `.py` relative path and file
body in `external_memory/src/semafuse_external_memory`, plus the selected A-MEM
adapter files separately. For v2 external runs, `run_key_bound=true` means the
run key contains the executable, Python-tree, dependency-lock, and complete
installed-distribution inventory hashes and the tree matches this audit. It
does not mean the declared dependency pins or a VCS commit were verified in the
invocation environment. That separate fact is reported by
`dependency_pins_verified`; the paper backend's 72 exact pins verify, while the
development package and Mem0 Git requirements remain unverified by that check.
The secret-free version maps from the actual invocation interpreters are kept
under [`receipts/environments`](receipts/environments/README.md).

## Score interpretation

The output contains retrieval evidence metrics only: precision, recall-any,
recall-all, mean recall, MRR, and conventional nDCG. Each metric report includes
aggregate and `multi-hop`, `temporal`, `open-domain`, and `single-hop` means.

The official category-aware LoCoMo token F1 requires answer predictions. It is
therefore deliberately `null` here and remains a separate evaluator in
`src/locomo_eval.rs`. Do not compare these retrieval scores with published
LoCoMo F1 or LazyMem binary-judge answer scores.
