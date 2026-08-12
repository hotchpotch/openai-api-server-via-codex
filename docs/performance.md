# Runtime performance

The repository includes a reproducible local benchmark for comparing the
Python reference server and the Go server. It uses the same deterministic SSE
backend, request payloads, concurrency, client, and host interface for both
runtimes.

This benchmark measures proxy overhead. It deliberately removes real model
latency, network variance, local response storage, and the default Codex
concurrency limit. Real Codex requests usually spend most of their wall time
waiting for model inference, so these throughput ratios must not be presented
as end-user generation-speed improvements.

## Recorded result

Measured on 2026-08-12 with Linux x86-64, an AMD Ryzen 9 7950X, Python 3.10.18,
and Go 1.23.5. Each scenario used 500 requests at concurrency 32 with
`max_stored_items=0` and `max_concurrent_requests=0`.

| Scenario | Runtime | Startup | Idle RSS | Peak RSS | Server CPU | Throughput | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Responses non-stream | Python | 337.09 ms | 58.52 MiB | 303.99 MiB | 6.52 s | 77.03 req/s | 385.85 ms | 647.83 ms |
| Responses non-stream | Go | 9.82 ms | 8.33 MiB | 19.74 MiB | 0.20 s | 831.89 req/s | 42.91 ms | 53.83 ms |
| Chat stream | Python | 325.20 ms | 58.75 MiB | 261.18 MiB | 6.51 s | 77.32 req/s | 380.98 ms | 587.11 ms |
| Chat stream | Go | 10.17 ms | 8.34 MiB | 17.89 MiB | 0.23 s | 644.54 req/s | 45.43 ms | 87.70 ms |

Under this proxy-bound workload, Go used about one-seventh of the idle RSS,
one-fifteenth of the peak RSS, and roughly one-thirtieth of the measured server
CPU time. It handled 8.3–10.8 times as many requests per second. The Python
peak includes its current per-request `AsyncOpenAI` client construction and
connection lifecycle; that is part of the reference implementation being
replaced, not an intrinsic limit of every possible Python proxy.

The complete machine metadata and unrounded workload configuration are kept in
[the raw benchmark result](benchmarks/go-vs-python-linux-amd64.json).

## Reproduce

The benchmark currently reads Linux `/proc` for process RSS, high-water RSS,
and user-plus-system CPU ticks:

```console
$ uv run python scripts/benchmark-runtimes.py \
    --requests 500 \
    --concurrency 32 \
    --output docs/benchmarks/go-vs-python-linux-amd64.json
```

When binding outside localhost, pass the exact safe host explicitly. The four
runtime processes use `--server-port` through `--server-port + 3`; the fake
backend uses `--upstream-port`.

Metrics are defined as follows:

- Startup: process creation until `/healthz` first returns HTTP 200.
- Idle RSS: `VmRSS` immediately after startup.
- Peak RSS: Linux `VmHWM` after warm-up and the measured request set.
- Server CPU: delta of `/proc/<pid>/stat` user and system CPU time during the
  measured request set. Client and fake-backend CPU are excluded.
- Latency: wall time observed by the shared HTTP client for each complete
  request or consumed stream.
- CPU percent: server CPU seconds divided by scenario wall time, normalized so
  100% represents one fully occupied logical CPU.
