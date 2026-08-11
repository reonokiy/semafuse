# Changelog

## 0.1.0-beta.1

First test release of Semafuse.

### Included

- Local SQLite-backed import, hook capture, search, raw provenance, export,
  retention, and MCP workflows for Amp, Codex, Claude Code, and OpenCode.
- Modular `history`, `lazymem`, and experimental `reconstructive` memory
  strategies behind one recall interface.
- Separate Unix `semafused` model service with Candle-backed embedding and
  reranking support.
- Optional, source-built benchmark harnesses for JSONL, LongMemEval, LoCoMo,
  and report generation.
- Five-platform `semafuse` release archives; Unix archives also contain the
  models-enabled `semafused` binary.

### Beta boundaries

- `reconstructive` is experimental. Its configuration and recall metadata may
  change before the stable release.
- Benchmark tooling is not included in prebuilt release archives; build it with
  `--features benchmarks` from a source checkout.
- The reconstructive-v4 LoCoMo suite is reproducible and validation-tested but
  has no complete paid run or publishable score. No quality-improvement claim
  is made for this beta.
- LoCoMo model-call fuses exclude embedding and reranking costs. Raw benchmark
  journals may contain dataset-derived text and must remain private.
- Stored histories and attachments are plaintext unless the host filesystem or
  volume provides encryption.
- `semafused` and its client transport are supported on Linux and macOS only.
