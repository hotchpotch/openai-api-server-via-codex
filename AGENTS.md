# Repository Guidelines

## Project

This repository implements an OpenAI-compatible FastAPI server that forwards
`/v1/responses` and `/v1/chat/completions` requests through the local Codex
ChatGPT backend credentials.

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

Run the full local validation suite:

```bash
uv run tox
```

Run the real Codex backend integration test only when live network/auth testing
is intended:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q -s
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_dual_backend_integration.py -q -s
```

Run the server locally:

```bash
uv run openai-api-server-via-codex
uv run openai-api-server-via-codex --backend codex-app-server --port 8001
```

## Implementation Notes

- Keep the public API OpenAI-compatible for both sync-style and async-style
  `openai-python` usage.
- Support both non-streaming and `stream=true` flows for Responses and Chat
  Completions.
- Preserve `previous_response_id` behavior by updating the in-memory response
  store when a response completes, including streaming responses.
- The default backend is `chatgpt-http`. Keep `codex-app-server` explicitly
  selectable because Codex app-server JSON-RPC is experimental.
- Treat `chatgpt-http` and `codex-app-server` as separate adapter contracts:
  `chatgpt-http` preserves normal Responses API function-calling semantics,
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
- Use fake backends for deterministic tests.
- Keep live tests skipped unless `RUN_CODEX_LIVE_TESTS=1` is set.
- Before committing behavior changes, run `uv run tox`.
- For changes that affect real Codex request/response handling, also run the
  live integration test when credentials and network access are available.

## Git

- Inspect the working tree before staging.
- Do not revert unrelated user changes.
- Do not commit secrets, credentials, auth files, or large generated artifacts.
- Use concise English commit messages that describe the behavior change.
