# Frozen external-memory provenance tree

This directory is a byte-for-byte snapshot of the Python harness used by the
four physical runs in `locomo314.development.receipt.json`. Its deterministic
Python-tree SHA-256 is
`cd2bc3ba106e0c9c448c8a2c03a28686a78102c2eb1bfdfe1214487f364728eb`,
and the same bytes are present at repository commit
`a1e43ab74625cee174a3c8915dc8f5a0064f24e1`.

It exists only so the ignored raw artifacts can still be audited after the live
adapters receive security fixes. Do not import, execute, or edit this snapshot;
current runs must use `benchmarks/external_memory/src/semafuse_external_memory`.
In particular, the current adapters reject remote plain-HTTP provider URLs and
verify the vendored A-MEM paper bytes before executing them. Historical runs
remain truthfully bound to the code they actually used rather than being
retroactively attributed to those fixes.

The vendored A-MEM paper file remains covered by
`benchmarks/external_memory/methods/amem_paper/LICENSE.upstream` and the
repository's third-party notices.
