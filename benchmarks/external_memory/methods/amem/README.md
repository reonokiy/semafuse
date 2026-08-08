# A-MEM dependency

Create an isolated Python 3.11 environment and install `requirements.txt`.
This pins the official reusable system package at commit
`ceffb860f0712bbae97b184d440df62bc910ca8d` (MIT). It is distinct from the
authors' paper-evaluation repository, recorded separately in
`upstream.lock.json`; this adapter therefore does not by itself reproduce the
paper table.

The adapter calls the upstream `AgenticMemorySystem`, `add_note`, and
`search_agentic` APIs. Supported upstream backends are `openai` and `ollama`.
For OpenAI, configure `api_key_env: "OPENAI_API_KEY"`; never put the key in the
file. The pinned system uses a fixed Chroma collection name and resets that
collection during initialization, so run it in a dedicated process/storage
environment.

The reusable package accepts model names but does not expose immutable model
revision parameters for every backend. Unless the execution environment locks
downloaded artifacts independently and records verified SHAs in `run_policy`,
the run must remain `development`; a mutable model name is not sufficient for a
reproducible or paper-standard label.
