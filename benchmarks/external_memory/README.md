# External memory method adapters

This package runs third-party memory systems behind a strict, process-isolated
JSONL contract. It translates benchmark messages into upstream `add` and
`search` calls. Installable upstreams remain thin package adapters. Because the
A-MEM paper repository has no Python package metadata, its exact original
`memory_layer.py` is vendored with its MIT license and byte identity; all
behavioral bridge patches remain outside that source snapshot.
Each method should run in its own virtual environment so incompatible upstream
dependencies cannot change Semafuse's benchmark environment.

The core package has no runtime dependencies. Missing upstream packages are a
structured `ADAPTER_UNAVAILABLE` error on `reset`, never a silent fallback to a
fake or simpler method. Secrets are inherited through environment variables;
configuration files may contain only names such as `OPENROUTER_API_KEY` under
an `*_env` field. Literal `api_key`, `password`, `token`, `secret`, credential,
or authorization fields are rejected before an upstream package is imported.
Any base URL that carries those credentials must use HTTPS; plain HTTP is
accepted only for an explicit `localhost` or loopback-IP development endpoint.
Worker stderr is drained but never returned by the driver because upstream SDKs
can log request headers or credentials.

## Lifecycle and isolation

One worker owns one method instance. The legal lifecycle is:

```text
reset -> add(messages, session, timestamp)* -> search(query, top_k,
max_context_tokens)* -> close
```

`reset` must succeed before every independent benchmark history. Inputs expose
only role, text, a non-semantic source ID, session, and timestamp. Answer text,
gold evidence sets, question category, judge rubric, and other labels are not
fields in the protocol. Search limits are re-enforced after the upstream call,
so an adapter cannot exceed `top_k` or the context-token cap. Use the exact
reader tokenizer (`tokenizer.kind = "huggingface"` with a pinned model
revision) for results; the whitespace counter is test-only.

Every config also declares `run_policy.track`. `development` and
`contract-test` cannot be presented as benchmark-standard results. The
protected tracks (`common-policy-v1`, `paper-reproduction`, and
`paper-reproduction-substituted`) require the adapter's exact locked upstream
SHA plus explicit provider/model identities and verified 40-character
immutable revisions for the memory LLM, embedding model, and reader tokenizer.
The identities must match the actual adapter and tokenizer configuration.
Paper tracks additionally require a verified paper-protocol revision. A
mutable API alias or a provider that does not disclose an immutable model
revision cannot produce a protected track with this harness; use `development`
and report that limitation instead.

The method environments do not yet lock every platform-specific wheel by hash.
The loader therefore rejects protected labels until a reviewed complete
dependency lock is registered. A version graph attached after a run is useful
development evidence but does not retroactively turn that run into a protected
track.

Run the offline contract tests without either method installed:

```bash
cd benchmarks/external_memory
PYTHONPATH=src python -m unittest discover -s tests -v
```

Install a method in an isolated environment, then start a worker through
`ExternalMemoryDriver(config_path=...)`. See [PROTOCOL.md](PROTOCOL.md), the
example configs, and the method-specific installation notes:

- [Mem0 OSS](methods/mem0_oss/README.md)
- [A-MEM reusable package baseline](methods/amem/README.md)
- [A-MEM paper-original backend](methods/amem_paper/README.md)

## Comparison tracks

### `common-policy-v1`

Rerun every method over the same chronological input and enforce the shared
benchmark policy: state reset per independent history, identical roles,
`final_k=10`, the same reader-tokenizer context cap, one reader and judge
deployment, sequential query concurrency, and separate construction/retrieval
latency and cost. Use a dedicated storage path/collection for each worker.
Mem0's `reset()` clears its configured store and A-MEM's constructor resets its
fixed Chroma collection, so sharing either store with non-benchmark data is
unsafe.

For `infer=true`, Mem0 must declare `provenance_policy = "origin_only"`.
Every newly returned memory ID is bound to the one message that created it and
the adapter checks search metadata against that binding. If a later message is
deduplicated into an existing fact, it is deliberately **not** added as another
evidence source: guessing that relationship would over-credit retrieval. This
is conservative and must be stated beside evidence-recall results. With
`infer=false`, use `provenance_policy = "exact_message"`; one raw message is one
memory and attribution is exact.

The native-system panel may use each method's intended LLM/embedder/vector
store. A backbone-matched panel must explicitly configure matching models.
Never combine hosted Mem0 scores with this `mem0_oss` adapter, and never treat
the production A-MEM package as the paper evaluation repository.

### Official-paper reproduction

Paper values are comparable only with the exact dataset revision/split,
upstream implementation revision, ingestion unit, prompts, models, thinking
mode, retrieval depth, tokenizer/context cap, evaluator, latency boundary, and
hardware policy. The adapter is merely transport:

- `amem` targets the separately published `agiresearch/A-mem` reusable package
  and is only a package/development baseline. `amem_paper` runs the original
  system from `WujiangXu/A-mem`, including rich notes, evolution, query rewrite,
  and graph expansion. The latter still is not a paper-table score under the
  two-conversation common retrieval evaluator; full reproduction also requires
  ten conversations, categories 1-5, category-specific k, answer generation,
  and the paper's answer metrics.
- Mem0 runs use the self-hosted Apache-2.0 `mem0.Memory` API. A paper table that
  used hosted Mem0, graph memory, another package version, or another extraction
  stack requires its own pinned config and cannot reuse published numbers.

Record upstream commit, config hash (with environment variable *names*, never
values), model revisions, graph/inference flags, and write-side LLM cost in the
result manifest.

Protected-track lock shape:

```json
{
  "run_policy": {
    "track": "common-policy-v1",
    "upstream": {"revision": "40-character-git-sha", "lock_status": "verified"},
    "models": {
      "memory_llm": {
        "provider": "provider-key",
        "model": "model-id",
        "revision": "40-character-immutable-sha",
        "lock_status": "verified"
      },
      "embedding": {
        "provider": "provider-key",
        "model": "model-id",
        "revision": "40-character-immutable-sha",
        "lock_status": "verified"
      },
      "reader_tokenizer": {
        "provider": "huggingface",
        "model": "tokenizer-id",
        "revision": "40-character-immutable-sha",
        "lock_status": "verified"
      }
    }
  }
}
```

## LoCoMo-314 zero-LLM Mem0 run

[`configs/mem0_oss.locomo314.openrouter-raw.json`](configs/mem0_oss.locomo314.openrouter-raw.json)
is the cheapest real bridge check. It runs the pinned Mem0 implementation with
`infer=false`, exact per-message provenance, local Qdrant, the same
`perplexity/pplx-embed-v1-0.6b` OpenRouter embedding used by Semafuse, and the
pinned Qwen3-32B reader tokenizer. It is a raw-vector ablation, **not** Mem0's
LLM extraction result and not a published-paper reproduction. The complete
LoCoMo-314 input has 1,077 messages and 314 category 1--4 questions across
`conv-49` and `conv-50`; reset once per conversation so the histories remain
independent.

Create the isolated environment, export the existing key without printing it,
and start the worker with:

```bash
python3 -m venv /tmp/semafuse-mem0-venv
/tmp/semafuse-mem0-venv/bin/pip install -r methods/mem0_oss/requirements.txt
export MEM0_DIR=/tmp/semafuse-mem0-runtime
export MEM0_TELEMETRY=false
export OPENROUTER_API_KEY
PYTHONPATH=src /tmp/semafuse-mem0-venv/bin/python -m semafuse_external_memory.worker \
  --config configs/mem0_oss.locomo314.openrouter-raw.json
```

The LoCoMo suite owns the worker through `ExternalMemoryDriver`; do not type
JSONL manually for a scored run. Add this method entry to the suite TOML (make
the virtual-environment path absolute for the machine running it):

```toml
[[methods]]
kind = "external_jsonl"
id = "mem0_oss_raw"
worker_config = "../../external_memory/configs/mem0_oss.locomo314.openrouter-raw.json"
command = ["/tmp/semafuse-mem0-venv/bin/python", "-m", "semafuse_external_memory.worker"]
working_directory = "../../external_memory"
python_path = "../../external_memory/src"
source_root = "../../external_memory/src/semafuse_external_memory"
dependency_lock = "../../external_memory/methods/mem0_oss/requirements.txt"
top_k = 50
max_context_tokens = 16384
timeout_seconds = 600
```

The suite appends the resolved `--config` argument, resets independently for
`conv-49` and `conv-50`, and derives @10 metrics from the same @50 retrieval.
LoCoMo is CC BY-NC 4.0, so the suite's non-commercial acknowledgement and
attribution gate still applies.

### A-MEM reusable-package OpenRouter development baseline

[`configs/amem.locomo314.openrouter.development.json`](configs/amem.locomo314.openrouter.development.json)
uses GPT-4o mini through an explicit OpenRouter base URL, validates and retries
up to three structured-output attempts, and resolves
`sentence-transformers/all-MiniLM-L6-v2` at immutable Hub revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` before giving its local snapshot
to the upstream A-MEM system. It uses the same pinned Qwen3 reader tokenizer as
the Mem0 run. No secret value is stored in the config.

This remains a `development` run: `agiresearch/A-mem` is not the separate paper
evaluation repository, GPT-4o mini's OpenRouter alias does not expose an immutable
provider revision, and the transitive Python environment is not fully locked.
A-MEM invokes its LLM during ingestion, so unlike the raw Mem0 ablation this
configuration incurs paid calls when the suite actually adds messages.

```toml
[[methods]]
kind = "external_jsonl"
id = "amem_openrouter"
worker_config = "../../external_memory/configs/amem.locomo314.openrouter.development.json"
command = ["/tmp/semafuse-amem-venv/bin/python", "-m", "semafuse_external_memory.worker"]
working_directory = "../../external_memory"
python_path = "../../external_memory/src"
source_root = "../../external_memory/src/semafuse_external_memory"
dependency_lock = "../../external_memory/methods/amem/requirements.txt"
top_k = 50
max_context_tokens = 16384
timeout_seconds = 600
```

Install the CPU-only PyTorch wheel and the pinned A-MEM requirements as
described in [the method notes](methods/amem/README.md), then export only
`OPENROUTER_API_KEY` into the worker environment. The LoCoMo suite supplies the
resolved `--config` argument and applies the same reset and retrieval policy as
the Mem0 method above.

### A-MEM paper-original OpenRouter development run

[`configs/amem_paper_original.locomo314.openrouter.development.json`](configs/amem_paper_original.locomo314.openrouter.development.json)
selects the exact paper repository commit, pinned MiniLM snapshot, paper
metadata/evolution prompts, paper query-keyword rewrite, and per-seed link
expansion. `retrieval.base_k` is independent from the JSONL protocol's final
native-item `top_k`, so links are represented in each dense seed's ordered
multi-source provenance instead of being appended after a full top-k list.

The configuration requires OpenRouter endpoints to honor all requested
parameters, validates every JSON-schema response locally, disables implicit
SDK retries, and gives each of the adapter's three attempts a 90-second bound.
It remains a common-policy development result because the hosted LLM revision
and wheel artifacts are not immutable, and because this evaluator reports evidence
Recall/nDCG rather than LoCoMo answer F1.

Use the tracked suite after installing the pinned environment and placing the
API key only in `OPENROUTER_API_KEY`:

```bash
export PATH=/tmp/semafuse-amem-venv/bin:$PATH
export PYTHONPATH=benchmarks/external_memory/src
export HF_HOME=/tmp/semafuse-amem-hf-cache
cargo run --release --locked --features benchmarks --bin semafuse -- bench locomo \
  --suite benchmarks/locomo/configs/paper_v2.amem-paper-original.openrouter.toml \
  --dataset /absolute/path/to/locomo10.json \
  --acknowledge-noncommercial \
  --output-dir benchmarks/locomo/results/amem-paper-original-openrouter
```

The suite hashes the worker source tree and
`methods/amem_paper/requirements.lock.txt` into the run identity. Before any
paid call it uses the exact invocation interpreter to inventory every installed
distribution, verifies all 72 strict pins in that lock, and binds the complete
runtime inventory hash into the run key. The lock records the resolved graph
reachable from the paper backend's dependency roots, while the runtime hash
also binds unrelated packages present in the isolated environment; neither
claims wheel-hash completeness.
