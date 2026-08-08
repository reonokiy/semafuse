# External memory method adapters

This package runs third-party memory systems behind a strict, process-isolated
JSONL contract. It translates benchmark messages into upstream `add` and
`search` calls; it does **not** copy or reimplement Mem0 or A-MEM algorithms.
Each method should run in its own virtual environment so incompatible upstream
dependencies cannot change Semafuse's benchmark environment.

The core package has no runtime dependencies. Missing upstream packages are a
structured `ADAPTER_UNAVAILABLE` error on `reset`, never a silent fallback to a
fake or simpler method. Secrets are inherited through environment variables;
configuration files may contain only names such as `OPENROUTER_API_KEY` under
an `*_env` field. Literal `api_key`, `password`, `token`, `secret`, credential,
or authorization fields are rejected before an upstream package is imported.
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
SHA plus verified 40-character immutable revisions for the memory LLM,
embedding model, and reader tokenizer. Paper tracks additionally require a
verified paper-protocol revision. A mutable API alias or a provider that does
not disclose an immutable model revision cannot produce a protected track with
this harness; use `development` and report that limitation instead.

Run the offline contract tests without either method installed:

```bash
cd benchmarks/external_memory
PYTHONPATH=src python -m unittest discover -s tests -v
```

Install a method in an isolated environment, then start a worker through
`ExternalMemoryDriver(config_path=...)`. See [PROTOCOL.md](PROTOCOL.md), the
example configs, and the method-specific installation notes:

- [Mem0 OSS](methods/mem0_oss/README.md)
- [A-MEM system](methods/amem/README.md)

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

The native-system panel may use each method's intended LLM/embedder/vector
store. A backbone-matched panel must explicitly configure matching models.
Never combine hosted Mem0 scores with this `mem0_oss` adapter, and never treat
the production A-MEM package as the paper evaluation repository.

### Official-paper reproduction

Paper values are comparable only with the exact dataset revision/split,
upstream implementation revision, ingestion unit, prompts, models, thinking
mode, retrieval depth, tokenizer/context cap, evaluator, latency boundary, and
hardware policy. The adapter is merely transport:

- A-MEM's paper reproduction code is `WujiangXu/A-mem`; this first adapter
  targets the separately published `agiresearch/A-mem` system package. Runs
  through it must be labeled native or paper-substituted, not exact A-MEM paper
  reproduction.
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
      "memory_llm": {"revision": "40-character-immutable-sha", "lock_status": "verified"},
      "embedding": {"revision": "40-character-immutable-sha", "lock_status": "verified"},
      "reader_tokenizer": {"revision": "40-character-immutable-sha", "lock_status": "verified"}
    }
  }
}
```
