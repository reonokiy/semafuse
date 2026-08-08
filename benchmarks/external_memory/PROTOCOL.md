# JSONL protocol v1

The driver writes one UTF-8 JSON object per line and receives exactly one
object per request. Unknown/missing fields, non-finite numbers, invalid state,
non-JSON stdout, mismatched request IDs, timeout, or early process exit are hard
failures. The line limit is 8 MiB.

`reset` clears READY before calling the upstream method. A failed `reset`,
`add`, or `close` poisons the worker; the driver terminates that process and is
itself terminal, so a later request cannot observe partially reset or partially
written state. Construct a new driver and begin with `reset`. Any transport or
response-validation failure is terminal as well.

All requests contain:

```json
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"reset"}
```

Operation-specific requests are:

```json
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"add","messages":[{"role":"user","content":"text","source_id":"turn-1"}],"session":"session-1","timestamp":"2026-01-01T00:00:00Z"}
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"search","query":"question","top_k":10,"max_context_tokens":16384}
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"close"}
```

`timestamp` may be JSON `null`. Roles are `system`, `user`, `assistant`, or
`tool`. Add accepts 1–4096 messages, `top_k` is 1–1000, and the context limit is
1–10,000,000 tokens.

Success and error envelopes are mutually exclusive:

```json
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"add","ok":true,"result":{"accepted":1}}
{"protocol":"semafuse.external-memory.v1","request_id":"uuid","op":"search","ok":false,"error":{"code":"ADAPTER_ERROR","message":"sanitized message","retryable":false}}
```

Search returns `items`, normalized `context`, `context_tokens`, and `truncated`.
Each item has exactly `id`, `text`, finite-or-null `score`, `source_ids`,
`session`, `timestamp`, and positive `token_count`. Whole items are appended in
native rank order until the next item would exceed the cap; items are never
partially cut. `context` joins selected texts with two newlines.
The worker counts each candidate's fully joined context with the configured
reader tokenizer; token counts are not assumed to be additive across item
boundaries. The driver checks item count, declared token cap, and exact context
text against the originating request before returning the response.
