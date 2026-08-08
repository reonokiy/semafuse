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

Messages are passed to `Memory.add` one at a time so every returned memory can
carry exactly one benchmark source ID. Passing a conversation batch and then
attaching all batch IDs to every extracted fact would inflate evidence recall.
This message-level ingestion unit must be recorded and matched across methods;
if a named paper used conversation-level batches, this adapter run is a
substitution rather than an exact reproduction.
