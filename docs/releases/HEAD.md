# HEAD

- Added a Go HTTP server with the same Responses, Chat
  Completions, Models, Images, Audio, stored-object, and fallback proxy API
  surface as the Python server.
- Added process-level OpenAI SDK contract tests that run unchanged against
  both the Python and Go servers, backed by the same deterministic fake Codex
  HTTP service. This is the migration gate for eventually removing the Python
  runtime.
- Made the existing live Codex compatibility and image-generation E2E suites
  selectable between Python and Go runtimes. Go now passes the full live SDK
  matrix and the generated-image round trip.
- Added a reproducible Linux `/proc` benchmark and recorded Go/Python startup,
  RSS, CPU, throughput, and latency results for non-streaming Responses and
  streaming Chat requests.
- Published-platform wheels now bundle the standalone Go server for Linux,
  macOS, and Windows on x86_64 and ARM64. `uvx` transparently execs that binary
  without installing the Python web stack; source installs retain the Python
  fallback during migration.
- Added Go implementations of `start`, `stop`, and `status`, including shared
  config-backed PID/log discovery, backoff-controlled automatic server restart,
  and graceful HTTP shutdown on Linux and macOS. Windows daemon shutdown is
  best-effort and may interrupt in-flight streams.
- Added Go-owned unit, deterministic HTTP/SSE contract, fuzz, spawned-binary
  E2E, and opt-in real Codex live suites. The live suite exercises Responses,
  Chat, streaming, tools, structured outputs, stored-object lifecycle, vision,
  image generation, Audio reachability, and fallback proxying without using a
  Python test runner.
- Added dynamic-port foreground startup with a stable, machine-readable listen
  log, stronger auth refresh/cache coverage, and Linux race-detector CI.
