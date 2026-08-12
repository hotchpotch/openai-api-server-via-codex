# Repository Guidelines

## Project

This repository implements an OpenAI-compatible Go HTTP server that forwards
`/v1/responses` and `/v1/chat/completions` requests through the local Codex
HTTP backend credentials. Python is only the `uvx` launcher and an
`openai-python` test client; do not add a Python server fallback.

The stable `0.2.0` release is the documented breaking-change boundary for the
Go-only server runtime. The `0.1.6b1` prerelease is its evaluation build. Do not
describe Python as a server runtime in current documentation or examples;
versioned historical release notes and the historical performance comparison
may retain accurate references to the removed implementation.

Keep compatibility behavior aligned with the official `openai-python` client.
When changing request or response shapes, add or update tests that exercise the
client API rather than only raw HTTP payloads.

## Environment

- Use Go 1.23 or newer for the server, CLI, and Go tests.
- Use Python 3.10 or newer only for the `uvx` launcher, OpenAI SDK consumer
  tests, and packaging/release tooling.
- Use `uv` for Python development dependencies and validation commands.
- The default foreground server binds to `127.0.0.1:18080`.
- Do not commit local virtualenv, cache, tox, pytest, or editor artifacts.
- The live integration tests use the machine's existing Codex authentication.
  Do not add real tokens or copied auth files to the repository.

## Development Commands

Run the full local validation suite before committing behavior changes:

```bash
uv run tox
```

GitHub Actions runs the same validation suite for pushes to `main` and for pull
requests through `.github/workflows/ci.yml`.

Run focused compatibility tests while iterating on request/response behavior:

```bash
uv run python -m pytest tests/test_openai_client_contract.py -q
go test ./internal/app
uv run ruff check .
uv run ty check
```

Run real Codex backend integration tests only when live network/auth testing is
intended. These tests use the machine's Codex authentication and make real model
requests:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q -s
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_codex_http_compatibility.py -q -s
RUN_CODEX_LIVE_TESTS=1 go test ./test/live -v -count=1 -timeout=20m
```

Run Go-owned deterministic contracts and the spawned-binary E2E while changing
the packaged runtime:

```bash
go test ./internal/app
go test ./test/e2e -v
go test -race ./...
```

CI additionally installs each of the six native platform wheels and reruns the
spawned-server E2E through the packaged Python launcher. It uses only synthetic
auth and the local fake Codex backend.

Run the broad Codex HTTP OpenAI client compatibility matrix by itself when
investigating API surface regressions:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_codex_http_compatibility.py::test_live_codex_http_handles_openai_client_compatibility_matrix -q -s
```

Run the server locally from a checkout:

```bash
go run ./cmd/openai-api-server-via-codex serve
go run ./cmd/openai-api-server-via-codex serve --port 18080
go run ./cmd/openai-api-server-via-codex serve --verbose
go run ./cmd/openai-api-server-via-codex serve --config ~/.config/openai-api-server-via-codex/config.toml
```

Build a standalone development executable without Python:

```bash
go build -trimpath -o ./bin/openai-api-server-via-codex ./cmd/openai-api-server-via-codex
./bin/openai-api-server-via-codex --version
```

Keep [docs/build-from-source.md](docs/build-from-source.md) aligned with Go
version requirements, build flags, supported targets, and CLI behavior.

Build and run the Go-only Docker server:

```bash
docker compose up --build -d
docker compose ps
docker compose down
```

Release container images use
`ghcr.io/hotchpotch/openai-api-server-via-codex`. Stable release tags publish
both `vX.Y.Z` and `latest`; prerelease tags publish only `vX.Y.Z...`. Keep the
manifest on `linux/amd64` and `linux/arm64`, and preserve the OCI source label
that links the public GHCR package to this repository. Published images must be
pullable anonymously.

Generate a config template:

```bash
go run ./cmd/openai-api-server-via-codex config-generate
go run ./cmd/openai-api-server-via-codex config-generate --stdout
```

Validate package artifacts before a PyPI release:

```bash
uv run tox
rm -rf build dist
uv build --wheel --no-sources
uv run python scripts/build-platform-wheels.py --wheel "$(ls dist/*-py3-none-any.whl)" --output-dir dist --version "$(uv run python -c 'import openai_api_server_via_codex as p; print(p.__version__)')"
uv run twine check --strict dist/*
uv run --with "$(ls dist/*manylinux_2_17_x86_64.whl)" --no-project openai-api-server-via-codex --help
```

Generate release note text from the draft or finalized release notes:

```bash
python scripts/release-notes.py vX.Y.Z
```

## Implementation Notes

- Keep the public API OpenAI-compatible for both sync-style and async-style
  `openai-python` usage.
- Support both non-streaming and `stream=true` flows for Responses and Chat
  Completions.
- Preserve `previous_response_id` behavior by updating the in-memory response
  store when a response completes, including streaming responses.
- Keep stored Chat Completions compatible with the `openai-python`
  `client.chat.completions.list/retrieve/update/delete` and
  `client.chat.completions.messages.list` APIs. Chat `metadata` is stored by
  this compatibility server and should not be forwarded to Codex backends unless
  the backend explicitly supports it.
- Keep Responses helper APIs compatible with the `openai-python`
  `client.responses.retrieve(..., stream=True)`,
  `client.responses.input_tokens.count`, `client.responses.delete`, and
  `client.responses.cancel` call shapes.
- The only backend is `codex-http`. The previous native Codex app-server
  backend was removed because it was unstable; do not keep compatibility paths
  for it unless it is deliberately reimplemented later.
- Normalize Codex HTTP backend requests at the backend boundary: force the
  downstream Codex call to `stream=true` and `store=false`, default text
  verbosity to `low`, default `tool_choice`/`parallel_tool_calls`, include
  `reasoning.encrypted_content`, and add Codex-compatible stream headers.
  Public `store=true` compatibility is handled by local in-memory stores, not
  by forwarding `store=true` to ChatGPT Codex.
- Config is loaded from `--config`, `OPENAI_VIA_CODEX_CONFIG`, or the XDG path
  `$XDG_CONFIG_HOME/openai-api-server-via-codex/config.toml`, falling back to
  `~/.config/openai-api-server-via-codex/config.toml`. Setting precedence is
  CLI flag, environment variable, config file, default.
- Daemon PID and log files default under the config directory's `run/`
  subdirectory. `start`, `stop`, and `status` should all resolve the same
  config-backed daemon paths. `stop` and `status` should accept `--verbose`.
  When host is omitted and the exact default PID file is absent, they may
  discover a single PID file for the selected port; if multiple PID files
  match, they must refuse to guess and ask for `--host` or `--pid-file`.
- In-memory compatibility stores are intentionally bounded. Keep
  `max_stored_items` defaulting to 1000 and apply it consistently to
  `ResponseStore` and `ChatCompletionStore`. Evict oldest entries first; `0`
  means no in-memory storage.
- Codex backend request concurrency is intentionally bounded. Keep
  `max_concurrent_requests` defaulting to 10, expose it through CLI/env/config,
  and hold a slot for the full duration of streaming responses. `0` means no
  local concurrency cap.
- Codex backend timeout defaults to 300 seconds. Keep CLI/env/config/default
  fallback paths, config templates, README examples, and tests aligned when
  changing that value.
- README examples should use the current preferred documented model. Keep the
  examples on `gpt-5.6-luna` unless there is a deliberate model guidance change.
  Do not confuse README example models with the server's compatibility default;
  changing `DEFAULT_MODEL` requires tests and config-template updates.
- Keep the package version in `pyproject.toml` and
  `openai_api_server_via_codex/__init__.py` aligned. The GitHub Actions release
  workflow expects release tags like `v0.0.1` to match that version exactly.
- Keep user-visible release notes under `docs/releases/`. Add draft entries to
  `docs/releases/HEAD.md` while developing, move them to
  `docs/releases/vX.Y.Z.md` for a release, and use
  `python scripts/release-notes.py vX.Y.Z` to generate the GitHub Release body.
  Keep `tests/test_release_notes.py` aligned with the release-note file
  selection rules.
- Keep README and active documents Go-first. Explain the remaining Python code
  only as launcher, SDK-test, or release tooling. Preserve versioned historical
  release notes rather than rewriting project history.
- For PyPI releases, prefer Trusted Publishing through
  `.github/workflows/release.yml` and the `pypi` GitHub environment. Do not add
  PyPI tokens to repository secrets unless a deliberate fallback release path is
  being used.
- Publish the GHCR image from the same tag-driven release workflow only after
  PyPI succeeds. Use `GITHUB_TOKEN` with job-scoped `packages: write`; do not add
  a registry password or personal access token. A prerelease must never update
  the `latest` container tag.
- Existing releases may be backfilled through the release workflow's
  `container_release_tag` dispatch input. The build must use the immutable
  existing tag, verify its version, skip PyPI, and never update `latest`.
- Keep `.github/workflows/ci.yml` aligned with the local required validation
  command. It should run on pushes to `main`, pull requests, and manual
  dispatch.
- Incoming API key authentication is optional and disabled by default. When
  `--api-key`, `OPENAI_VIA_CODEX_API_KEY`, or `[server].api_key` is configured,
  require `Authorization: Bearer <api_key>` for `/v1/...` routes only; keep
  `/healthz` unauthenticated. Do not forward the incoming API key to Codex.
  When `start` launches `serve`, propagate the API key through the child
  environment, not the child command-line arguments.
- `serve` and `start` perform a Codex auth preflight before binding the HTTP
  port or spawning the daemon. Missing auth files, invalid JSON, wrong
  `auth_mode`, missing access tokens, expired tokens without refresh tokens, or
  refresh failures should fail the command with a redacted stderr message and
  must not bind the HTTP listener or start the background daemon.
- When Codex returns `401` before a response stream begins, invalidate the
  in-process auth cache, reload `auth.json`, and retry the authenticated request
  at most once. Keep request bodies replayable for this retry and never retry
  after response events have been forwarded to the client.
- `--verbose`, `OPENAI_VIA_CODEX_VERBOSE`, and `[server].verbose` should map to
  Go application debug logs and be preserved when `start` launches the
  foreground `serve` command in the background. Verbose mode should also emit
  application diagnostics for resolved config/settings, request lifecycle,
  endpoint summaries, model-list fallbacks, and Codex HTTP stream/auth behavior.
  Never log raw auth tokens. Use the shared redaction helpers when logging
  upstream errors, request query strings, or auth-related values.
- For Chat Completions, translate Responses stream events into
  `chat.completion.chunk` events.
- Prefer structured Go parsing and the official OpenAI SDK consumer tests over
  ad hoc string handling.

## Testing Expectations

- Add focused Go unit or contract tests under `internal/app/`. Add public SDK
  compatibility cases under `tests/test_openai_client_contract.py`.
- Add or update redaction tests when changing auth, logging, upstream error
  handling, or request logging code. Raw `access_token`, `refresh_token`,
  `id_token`, bearer tokens, JWTs, and client `api_key` values must not appear
  in logs or compatibility error responses.
- Use fake backends for deterministic tests. Prefer tests that call through
  `AsyncOpenAI` or `OpenAI` instead of raw HTTP so SDK parsing, pagination, and
  stream handling are covered.
- Keep live tests skipped unless `RUN_CODEX_LIVE_TESTS=1` is set.
- Before committing behavior changes, run `uv run tox`.
- For changes that affect real Codex request/response handling, also run the
  live integration test when credentials and network access are available.
- For broad compatibility work, run the Codex HTTP live matrix. It covers
  Responses, Chat Completions, streaming, stored Chat lifecycle, JSON mode,
  structured outputs, tool calling, images, multi-turn context, long plain-text
  conversations, and sync/async OpenAI clients.
- After every live run, manually inspect the `-s` output. Confirm that marker
  strings are preserved, JSON/structured outputs parse to the expected objects,
  tool calls use the expected names and arguments, streaming event sequences end
  in completion events, stored Chat retrieve/list/messages/delete behavior is
  coherent, and outputs are semantically appropriate even when wording differs.
- When a live test failure reveals a real backend difference, update the
  contract to match official `openai-python` behavior and add a deterministic
  fake-backend regression test before relying on the live test alone.

## Git

- Inspect the working tree before staging.
- Do not revert unrelated user changes.
- Do not commit secrets, credentials, auth files, or large generated artifacts.
- Use concise English commit messages that describe the behavior change.
