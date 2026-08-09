# semafused

`semafused` is Semafuse's separately managed, long-running local service
process. It runs in the foreground and serves a versioned binary-framed JSON
protocol over a Unix-domain socket. The one-shot `semafuse` CLI can inspect it,
but never starts, stops, daemonizes, or supervises it.

The models service is entirely local: model artifacts are read from absolute
filesystem paths, inference uses Rust Candle implementations, and requests go
over the Unix socket rather than HTTP. `semafused` never downloads model files
at startup or during inference.

## Build

The light build contains the daemon, protocol router, and injectable service
boundary but no model runtime:

```sh
cargo build --release --bin semafused --no-default-features --features daemon
```

For the production local Qwen embedding and BGE reranker service, enable the
model feature explicitly:

```sh
cargo build --release --bin semafused --no-default-features --features models-candle
# or
cargo install --path . --locked --bin semafused --no-default-features --features models-candle
```

`models-mkl` additionally enables Intel MKL where the host supports it:

```sh
cargo build --release --bin semafused --no-default-features --features models-mkl
```

The current model loader runs on CPU; `models-mkl` can accelerate supported
Intel hosts, but CUDA and Metal devices are not wired yet. Both models remain
resident in one process, so budget several GiB of RAM. The example keeps one
inference active at a time, embedding length/batch at `2048/1`, and reranker
length/batch at `1024/1`; raise those values only after measuring the host.

## Provision model artifacts

Provisioning is a one-shot operation outside `semafused`. Each configured model
directory must already exist and contain the official files below:

```text
config.json
tokenizer.json
model.safetensors
```

The current local loader intentionally requires a single `model.safetensors`
file; it does not silently select an ONNX export or a network fallback.

- Embedding: an official Qwen3 embedding checkpoint with model type `qwen3`
  and architecture `Qwen3ForCausalLM` (the supported default is
  `Qwen/Qwen3-Embedding-0.6B`). FastEmbed's Candle Qwen implementation performs
  left-padded last-token pooling and L2 normalization.
- Reranker: the official `BAAI/bge-reranker-v2-m3` XLM-R sequence classifier.
  Candle returns its raw classifier logit; the service applies a stable sigmoid
  so the public score is in `[0, 1]`.

Startup validates the model architecture, hashes the exact config, tokenizer,
weights, maximum length, and implementation version into a `sha256:` identity,
then runs golden inference checks. A mismatched or incompatible checkpoint
causes startup to fail instead of serving silently different vectors.

Copy [semafused.example.toml](../semafused.example.toml), replace all paths with
absolute paths, and keep the socket inside a private directory. TOML does not
expand `$HOME` or `$XDG_RUNTIME_DIR`.

## Run in the foreground

```sh
target/release/semafused --config /absolute/path/to/semafused.toml
```

With no explicit socket path the default is
`$XDG_RUNTIME_DIR/semafuse/semafused.sock`; when `XDG_RUNTIME_DIR` is absent it
uses `/tmp/semafuse-$UID/semafused.sock`. The parent directory must be owned by
the current UID with mode `0700`; the socket is mode `0600`, and peers are
checked by UID.

Probe the running process with the separate one-shot CLI:

```sh
semafuse daemon status \
  --socket "$XDG_RUNTIME_DIR/semafuse/semafused.sock"
```

The status output includes the embedding and reranker fingerprints. Copy those
values into the application's `backend = "semafused"` endpoint configuration
and into
`benchmarks/lazymem/configs/semafuse_lazymem.semafused.toml.example`. Changing
an artifact, maximum length, or implementation changes the fingerprint and
therefore the vector namespace; rebuild the affected LazyMem strategy before
recalling old data.

## systemd user service

Install the binary and config first, then create
`~/.config/systemd/user/semafused.service`:

```ini
[Unit]
Description=Semafuse local service daemon
After=default.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/semafused --config %h/.config/semafuse/semafused.toml
Restart=on-failure
RestartSec=2
RuntimeDirectory=semafuse
RuntimeDirectoryMode=0700
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
```

Then let systemd own the lifecycle:

```sh
systemctl --user daemon-reload
systemctl --user enable --now semafused.service
systemctl --user status semafused.service
journalctl --user -u semafused.service -f
```

Stopping or restarting this unit is deliberately not a `semafuse` one-shot CLI
operation.
