# Go migration test policy

The published server runtime is moving to Go, while the Python package remains
the `uvx` installation and launch shim. Removing the Python HTTP implementation
is intentionally a later release decision, not part of the initial Go port.

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

Python has two deliberate roles during the migration:

- the package entry point selects and execs the bundled platform-specific Go
  executable for `uvx` installations;
- `openai-python` remains a consumer-level compatibility gate. The same
  process-level suite runs against both implementations while the Python server
  is retained as a behavioral oracle.

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

## Python HTTP server removal gate

Delete the Python HTTP implementation only in a separate change after all of
the following are true:

1. Every public route and lifecycle operation has a deterministic Go contract,
   including non-streaming and streaming Responses and Chat, Images, Audio,
   Models, stored objects, auth, fallback proxying, and error redaction.
2. The unchanged `openai-python` cross-runtime suite is green against Go.
3. The Go-authored real live matrix and the existing broad Python-authored live
   matrix are green against Go for at least one released version.
4. Linux race detection, native Linux/macOS/Windows CI, x86_64/ARM64 package
   builds, and a real ARM64 smoke/live run are green.
5. `uvx` install, configuration, daemon commands, upgrades, and platform-binary
   selection are covered independently of the Python server implementation.
6. Release notes announce the fallback removal and identify unsupported source
   installation platforms before the code is deleted.

The `openai-python` consumer tests and lightweight `uvx` launcher remain after
that removal. Only the Python HTTP server and its implementation-only tests are
retired.

## ARM64 sign-off record

On 2026-08-12, a statically linked Linux ARM64 Go binary was copied to a
Raspberry Pi 5 over Tailscale and run from
`/home/hotchpotch/tmp/openai-via-codex-go-e2e.lyPw8VMz`. Using that machine's
existing `~/.codex/auth.json`, the live smoke exercised health, models,
Responses create/retrieve/input-items/input-tokens/previous/stream, and Chat
`n=2`. All checks passed and the server was stopped after the run.
