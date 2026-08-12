# Go migration test policy

The published and source server runtime is Go. The Python package remains only
as the `uvx` installation/launch shim and as an `openai-python` test consumer.
There is no Python HTTP implementation or fallback.

## Test ownership

Go owns the behavior closest to the production runtime:

- unit tests for auth, configuration, stores, redaction, request translation,
  SSE parsing, daemon behavior, and proxy path validation;
- deterministic handler contracts in `internal/app/contract_test.go`, using a
  fake Codex HTTP upstream and real HTTP/SSE requests;
- a thin E2E in `test/e2e`, which builds and starts the real executable,
  discovers its OS-assigned port from the stable listen log, makes requests,
  and verifies preflight and shutdown behavior;
- an opt-in real-backend matrix in `test/live`, authored entirely in Go.

The primary test seam is a fake Codex HTTP service. A narrow internal backend
interface is used only for failures that cannot be represented faithfully over
HTTP. Fake payloads use synthetic IDs and valid future-expiry test JWTs; tests
must never copy production tokens or authentication files.

Python has two deliberate roles:

- the package entry point selects and execs the bundled platform-specific Go
  executable for `uvx` installations;
- `openai-python` remains a consumer-level compatibility gate that starts the
  Go binary and exercises it through the public SDK.

Go wire-level tests are canonical for the server itself. An `openai-go` SDK
dependency is not required: direct `net/http` assertions make JSON, headers,
status codes, pagination, and SSE event sequences explicit, while
`openai-python` checks the client API promised by this project.

## Validation levels

Use the smallest useful level while iterating, then run the full gate before a
behavioral commit:

```console
$ go test ./internal/app
$ go test ./test/e2e -v
$ go test ./...
$ go vet ./...
$ go test -race ./...
$ uv run tox
```

The real suite is opt-in because it uses the machine's Codex login, network,
models, and image quota:

```console
$ RUN_CODEX_LIVE_TESTS=1 go test ./test/live -v -count=1 -timeout=20m
```

It checks semantic outputs and exact marker preservation rather than model
wording. A real backend capability can be unavailable independently of the
proxy: for example, the sibling Audio endpoint may return a Cloudflare browser
challenge. In that case the live test verifies transparent reachability and
the deterministic fake-upstream contract remains responsible for the 2xx
response shape.

CI runs native Go tests and vet on Linux, macOS, and Windows, the race detector
on Linux, and builds the packaged x86_64 and ARM64 binaries through the platform
wheel job. Live tests stay manual. ARM hardware is a release sign-off rather
than an emulation-only claim.

## Post-removal invariants

The Python HTTP server and its implementation-only tests have been retired.
Keep these conditions true:

1. Every public route and lifecycle operation has a deterministic Go contract.
2. The `openai-python` process contract and both Python-authored live suites
   always start Go; they must not grow a second server implementation.
3. The launcher either execs a bundled platform binary or exits clearly. It
   must never silently fall back to Python.
4. Releases publish only the six supported platform wheels. A generic wheel is
   an intermediate build input and must not be uploaded to PyPI.
5. Linux race detection, native Linux/macOS/Windows CI, x86_64/ARM64 package
   builds, and periodic real ARM64 sign-off remain green.

## ARM64 sign-off record

On 2026-08-12, a statically linked Linux ARM64 Go binary was copied to a
Raspberry Pi 5 over Tailscale and run from
`/home/hotchpotch/tmp/openai-via-codex-go-e2e.lyPw8VMz`. Using that machine's
existing `~/.codex/auth.json`, the live smoke exercised health, models,
Responses create/retrieve/input-items/input-tokens/previous/stream, and Chat
`n=2`. All checks passed and the server was stopped after the run.
