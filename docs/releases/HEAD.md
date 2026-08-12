# HEAD

- Added a Go HTTP server for Responses, Chat Completions, Models, Images,
  Audio, stored objects, and fallback proxying.
- Added process-level OpenAI SDK contract tests that start Go with a
  deterministic fake Codex HTTP service and exercise it through
  `openai-python`.
- Made the existing live Codex compatibility and image-generation E2E suites
  always start Go. Go passes the full live SDK matrix and generated-image round
  trip.
- Added a reproducible Linux `/proc` benchmark and recorded Go/Python startup,
  RSS, CPU, throughput, and latency results for non-streaming Responses and
  streaming Chat requests.
- Published-platform wheels now bundle the standalone Go server for Linux,
  macOS, and Windows on x86_64 and ARM64. `uvx` transparently execs that binary
  without installing a Python web stack.
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
- Removed the Python HTTP server, its FastAPI/Uvicorn runtime dependencies, and
  its implementation-only tests. Python remains only as the dependency-free
  `uvx` launcher and as an `openai-python` compatibility test client.
- Changed Docker images to build and run the static Go executable directly,
  and changed releases to publish only binary-bearing platform wheels.
