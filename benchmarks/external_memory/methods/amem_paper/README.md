# A-MEM paper-original backend

This backend runs the original `AgenticMemorySystem` from the A-MEM paper
repository at immutable commit
`0c8039f28fdcc08189a23c07a3437d9d2482f9c2`. The repository is not an
installable Python package, so its `memory_layer.py` is vendored under the
external worker package and checked against `upstream.lock.json` at reset.
The upstream and vendored SHA-256 values differ only because the local source
snapshot has one final newline; that normalization is recorded explicitly.

The adapter uses the paper pipeline rather than the reusable system package:

- every message becomes an LLM-generated rich note with keywords, context,
  and tags;
- content plus rich metadata is embedded with a revision-pinned local
  SentenceTransformer snapshot;
- evolution operates on real corpus indices and may update links and neighbor
  metadata;
- the paper query-keyword prompt runs before dense retrieval;
- each dense seed is returned with its linked notes in paper order, and the
  worker exposes their ordered source-message provenance as one native item.

Three narrow runtime patches are required. The exact file calls `re.sub` without
importing `re`, so the adapter injects only that standard-library module after
the source hash check. Its consolidation path also falls back to a mutable
`all-MiniLM-L6-v2` name with current SentenceTransformers, so the adapter
rebuilds the same composite retriever with the already resolved immutable
snapshot. Finally, the upstream asks the evolution model for a neighbor action
even before the first note has any possible neighbor; the adapter returns the
only valid no-evolution outcome for that empty graph. All three patches live in
the hashed adapter, not the vendored source.

The OpenAI controller drops its `api_base` constructor argument. For an
OpenAI-compatible provider, the adapter therefore scopes the explicit
non-secret base URL through `OPENAI_BASE_URL` while constructing the SDK
client, then restores the worker environment. Credentials are accepted only
through `api_key_env`; remote base URLs must use HTTPS, while plain HTTP is
limited to explicit localhost or loopback-IP endpoints.
`LITELLM_LOCAL_MODEL_COST_MAP=True` is scoped around execution of the already
hash-verified upstream bytes to prevent its unused LiteLLM import from fetching
a mutable price map. The OpenRouter configuration also sends
`provider.require_parameters=true`, so routing excludes endpoints that ignore
the requested JSON-schema parameter; the adapter still validates every
response locally. The provider SDK's implicit retries are disabled and each
request has a configured 90-second bound; the adapter owns the single explicit
three-attempt retry budget. The suite allows 1,800 seconds for a whole session
RPC because one LoCoMo session can contain 43 messages and therefore many
bounded provider requests.

Install `requirements.lock.txt` in an isolated Python 3.12 environment. It is a
pinned dependency closure with 72 required distributions, not a claim that the
worker environment is clean or contains no additional packages. The runner
verifies those pins and binds a separate hash over every installed distribution
(117 in the recorded paid environment), so extras affect the run identity even
though the 72-pin lock cannot reconstruct them. The benchmark remains a
development/common-policy result because wheel artifacts and the
provider-served LLM revision are not independently immutable. It is not the
paper's answer-F1 table: the common evaluator uses two LoCoMo conversations,
categories 1-4, source-evidence Recall/nDCG, one fixed base `k`, and a shared
context cap.
