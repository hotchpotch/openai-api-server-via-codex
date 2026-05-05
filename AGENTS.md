# Repository Guidelines

## Project

This repository implements an OpenAI-compatible FastAPI server that forwards
`/v1/responses` and `/v1/chat/completions` requests through the local Codex
HTTP backend credentials.

Keep compatibility behavior aligned with the official `openai-python` client.
When changing request or response shapes, add or update tests that exercise the
client API rather than only raw HTTP payloads.

## Environment

- Use Python 3.11.
- Use `uv` for dependency management and command execution.
- Do not commit local virtualenv, cache, tox, pytest, or editor artifacts.
- The live integration tests use the machine's existing Codex authentication.
  Do not add real tokens or copied auth files to the repository.

## Development Commands

Run the full local validation suite before committing behavior changes:

```bash
uv run tox
```

Run focused compatibility tests while iterating on request/response behavior:

```bash
uv run python -m pytest tests/test_openai_compat_server.py -q
uv run ruff check .
uv run ty check
```

Run real Codex backend integration tests only when live network/auth testing is
intended. These tests use the machine's Codex authentication and make real model
requests:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q -s
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_dual_backend_integration.py -q -s
```

Run the broad dual-backend OpenAI client compatibility matrix by itself when
investigating API surface regressions:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_dual_backend_integration.py::test_live_dual_backends_handle_openai_client_compatibility_matrix -q -s
```

Run the server locally:

```bash
uv run openai-api-server-via-codex
uv run openai-api-server-via-codex --backend codex-http --port 8000
uv run openai-api-server-via-codex --backend codex-app-server --port 8001
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
- The default backend is `codex-http`. Keep `codex-app-server` explicitly
  selectable because Codex app-server JSON-RPC is experimental.
- Treat `codex-http` and `codex-app-server` as separate adapter contracts:
  `codex-http` preserves normal Responses API function-calling semantics,
  while `codex-app-server` maps OpenAI function schemas to Codex
  `dynamicTools` and projects app-server dynamic tool notifications back into
  OpenAI-compatible `function_call` / `tool_calls` items.
- The native app-server backend owns Codex thread bindings internally, so the
  FastAPI layer should forward `previous_response_id` to backends that declare
  native session support instead of replaying local context into the request.
- For Chat Completions, translate Responses stream events into
  `chat.completion.chunk` events.
- Prefer structured parsing and Pydantic/FastAPI/OpenAI SDK models over ad hoc
  string handling.

## Testing Expectations

- Add focused unit or contract tests under `tests/` for compatibility behavior.
- Use fake backends for deterministic tests. Prefer tests that call through
  `AsyncOpenAI` or `OpenAI` instead of raw HTTP so SDK parsing, pagination, and
  stream handling are covered.
- Keep live tests skipped unless `RUN_CODEX_LIVE_TESTS=1` is set.
- Before committing behavior changes, run `uv run tox`.
- For changes that affect real Codex request/response handling, also run the
  live integration test when credentials and network access are available.
- For broad compatibility work, run the dual-backend live matrix against both
  `codex-http` and `codex-app-server`. It covers Responses, Chat Completions,
  streaming, stored Chat lifecycle, JSON mode, structured outputs, tool calling,
  images, multi-turn context, long plain-text conversations, and sync/async
  OpenAI clients.
- After every live run, manually inspect the `-s` output. Confirm that marker
  strings are preserved, JSON/structured outputs parse to the expected objects,
  tool calls use the expected names and arguments, streaming event sequences end
  in completion events, stored Chat retrieve/list/messages/delete behavior is
  coherent, and both backends produce semantically equivalent results even when
  wording differs.
- When a live test failure reveals a real backend difference, update the
  contract to match official `openai-python` behavior and add a deterministic
  fake-backend regression test before relying on the live test alone.

## Git

- Inspect the working tree before staging.
- Do not revert unrelated user changes.
- Do not commit secrets, credentials, auth files, or large generated artifacts.
- Use concise English commit messages that describe the behavior change.
