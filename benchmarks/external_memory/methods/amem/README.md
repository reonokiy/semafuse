# A-MEM dependency

Create an isolated Python 3.11 or 3.12 environment and install
`requirements.txt`.
This pins the official reusable system package at commit
`ceffb860f0712bbae97b184d440df62bc910ca8d` (MIT). It is distinct from the
authors' paper-evaluation repository, recorded separately in
`upstream.lock.json`; this adapter therefore does not by itself reproduce the
paper table.

The adapter calls the upstream `AgenticMemorySystem`, `add_note`, and
`search_agentic` APIs. Supported upstream backends are `openai` and `ollama`.
The pinned upstream catches some LLM, JSON, and retrieval failures internally
and otherwise returns a degraded result. The adapter therefore attaches an
error-level counter to the upstream logger and fails the worker on any such
event. It retains only the count, never the log record or message, so provider
headers and credentials cannot enter benchmark artifacts.
The adapter requires explicit `model_name`, its immutable 40-character
`model_revision`, `llm_backend`, and `llm_model`. It resolves that exact Hub
snapshot before constructing the pinned upstream system and passes the local
path to Chroma's SentenceTransformer wrapper, which prevents a mutable branch
from being loaded despite upstream exposing only `model_name`.
For the OpenAI-compatible backend it also requires `api_key_env` and an
explicit non-secret `openai_base_url`; the adapter scopes that URL through
`OPENAI_BASE_URL` only while constructing the upstream SDK client. For
OpenRouter, use `api_key_env: "OPENROUTER_API_KEY"` and
`openai_base_url: "https://openrouter.ai/api/v1"`. The pinned system uses a
fixed Chroma collection name and resets that collection during initialization,
so run it in a dedicated process/storage environment.

On Linux, an unconstrained installation may resolve a CUDA-enabled PyTorch and
download more than a gigabyte of GPU libraries even for a CPU run. Install an
official CPU-only PyTorch wheel first, then install `requirements.txt`. The
configuration tested on Python 3.12 used `torch==2.13.0+cpu`. Record the exact
wheel index and the complete resolved environment; upstream transitive
requirements are not locked by this repository yet.

The reusable package accepts model names but does not expose immutable model
revision parameters for every backend. Unless the execution environment locks
downloaded artifacts independently and records verified SHAs in `run_policy`,
the run must remain `development`; a mutable model name is not sufficient for a
reproducible or paper-standard label.
