# Mem0 OSS dependency

Create an isolated Python 3.11 environment and install `requirements.txt`.
This pins the official self-hosted library at commit
`4debc58a83377b18be81ae1e5969a300736b2fac` (Apache-2.0). Optional providers
needed by a chosen LLM, embedder, reranker, or vector store must be installed in
this same environment and recorded in the run manifest.

The adapter calls `mem0.Memory(MemoryConfig(...))`, `add`, `search`, and
`reset`; it never uses `MemoryClient` or the hosted service. Configuration
requires `memory_config` and `scope_id`, and permits `infer`,
`search_threshold`, and `rerank`. Put provider secrets in the environment and
write only `api_key_env: "ENV_NAME"` in JSON. Use a dedicated vector-store path
and history database because reset is destructive for the configured store.

`requirements.txt` also pins the Transformers version used by the reader token
counter, but Mem0's remaining transitive dependencies are not a complete lock.
Protected result labels therefore fail closed; retain `run_policy.track =
"development"` and save `pip freeze` with the result until a reviewed lock is
added. Bare Mem0 does not install its optional spaCy/fastembed extras: with
local Qdrant this disables entity and BM25 boosts and leaves dense retrieval
working. Install and pin those extras only for a declared native-hybrid panel;
do not silently compare that panel with a dense-only run.

Messages are passed to `Memory.add` one at a time so every returned memory can
carry exactly one benchmark source ID. Passing a conversation batch and then
attaching all batch IDs to every extracted fact would inflate evidence recall.
This message-level ingestion unit must be recorded and matched across methods;
if a named paper used conversation-level batches, this adapter run is a
substitution rather than an exact reproduction.

Inference runs additionally require `provenance_policy: "origin_only"`. New
memory IDs are checked against the exact creating source message. When Mem0
deduplicates a later message and returns no new ID, the later source is not
guessed or merged into the original memory. This conservative rule prevents
inflated evidence recall but can under-credit repeated evidence, so report it.
Raw `infer=false` runs use `provenance_policy: "exact_message"` and do not have
that ambiguity.
