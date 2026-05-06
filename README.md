# OpenAI API Server via Codex

OpenAI-compatible local API server backed by your logged-in Codex credentials.

This server exposes the common OpenAI API surface used by `openai-python` and
forwards generation work to the Codex HTTP backend using the local
`codex login` auth file.

> [!NOTE]
> This is a compatibility server for local or trusted environments. By default
> it does not authenticate incoming requests. Set `--api-key` when binding to
> anything other than localhost.

## Usage

Run the server in the foreground:

```console
$ uv run openai-api-server-via-codex
```

Call it with `openai-python`:

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:18080/v1")

response = client.responses.create(
    model="gpt-5.5",
    input="Reply in one sentence.",
    reasoning={"effort": "low"},
)
print(response.output_text)
```

Use Chat Completions:

```python
chat = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="low",
)
print(chat.choices[0].message.content)
```

Stream a response:

```python
stream = client.responses.create(
    model="gpt-5.5",
    input="Stream a short reply.",
    stream=True,
    reasoning={"effort": "low"},
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

Run as a background daemon:

```console
$ uv run openai-api-server-via-codex start --host 127.0.0.1 --port 18080
$ uv run openai-api-server-via-codex status --port 18080
$ uv run openai-api-server-via-codex stop --port 18080
```

`start` prints the PID file and log file paths.

## Install

From this checkout:

```console
$ uv sync --dev
$ uv run openai-api-server-via-codex --help
```

Run directly with `uvx` after the package is installed or published:

```console
$ uvx openai-api-server-via-codex
```

## Requirements

- Python 3.11+
- `uv`
- A working Codex login, usually at `~/.codex/auth.json`

Use an explicit Codex auth file when needed:

```console
$ uv run openai-api-server-via-codex --auth-json ~/.codex/auth.json
$ OPENAI_VIA_CODEX_AUTH_JSON=~/.codex/auth.json uv run openai-api-server-via-codex
```

> [!NOTE]
> The incoming OpenAI-compatible API key and the Codex auth file are separate.
> `--api-key` protects this local server. `--auth-json` selects the Codex
> credentials used by the server when it calls the Codex backend.

## Disclaimer

Use this project at your own risk. It is not the official OpenAI Platform API
and is not endorsed or supported by OpenAI. It forwards requests to the Codex
HTTP backend used by the Codex CLI and ChatGPT subscription flow instead of
`api.openai.com`.

For reference, Simon Willison describes this route as a
[semi-official OpenAI Codex backdoor API](https://simonwillison.net/2026/Apr/23/gpt-5-5/).
That matches this project's practical model: it uses the ChatGPT/Codex backend
available through your own logged-in Codex credentials, and that backend may
change without notice.

Use this server only with accounts and subscriptions you are allowed to use. Do
not expose it to untrusted networks without `--api-key` or another access
control layer, and follow OpenAI's
[Terms of Use](https://openai.com/policies/terms-of-use/) and
[Usage Policies](https://openai.com/policies/usage-policies/).

## Endpoints

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `POST /v1/responses/{response_id}/cancel`
- `POST /v1/responses/input_tokens`
- `POST /v1/chat/completions`
- `GET /v1/chat/completions`
- `GET /v1/chat/completions/{completion_id}`
- `POST /v1/chat/completions/{completion_id}`
- `DELETE /v1/chat/completions/{completion_id}`
- `GET /v1/chat/completions/{completion_id}/messages`

## Compatibility

The server supports both sync and async `openai-python` clients for the main
OpenAI APIs:

- `client.responses.create(...)`
- `client.chat.completions.create(...)`

Supported behavior includes:

- `stream=True` for Responses and Chat Completions
- `previous_response_id` for Responses, backed by local in-memory context
- standard Chat Completions multi-turn through the `messages` list
- function and tool calling, including streaming tool-call arguments
- JSON mode and structured outputs
- URL and data URL image parts
- reasoning effort fields where the selected model accepts them
- stored Chat Completions compatibility APIs backed by local in-memory storage

For Codex compatibility, backend requests are normalized to streaming
Responses calls with `store=false`, low text verbosity by default, automatic
tool choice defaults, and `reasoning.encrypted_content` included for reasoning
context. Public `store=true` behavior is implemented locally.

> [!NOTE]
> Model listing is best-effort because the upstream Codex HTTP model catalog can
> differ from the models that a subscription can actually run. As of
> 2026-05-06, with a ChatGPT Pro subscription, `gpt-5.3-codex-spark` did not
> appear in `GET /v1/models` in our live test, but direct requests using
> `model="gpt-5.3-codex-spark"` succeeded. OpenAI also describes
> GPT-5.3-Codex-Spark as a research preview for ChatGPT Pro users.

## Configuration

Generate a default config file:

```console
$ uv run openai-api-server-via-codex config-generate
$ uv run openai-api-server-via-codex config-generate --stdout
```

The default config path is:

```text
$XDG_CONFIG_HOME/openai-api-server-via-codex/config.toml
```

If `XDG_CONFIG_HOME` is unset, this becomes:

```text
~/.config/openai-api-server-via-codex/config.toml
```

You can also set `OPENAI_VIA_CODEX_CONFIG` or pass `--config` to `serve`,
`start`, `stop`, and `status`.

Resolution order is:

```text
CLI flag -> environment variable -> config file -> default
```

Example config:

```toml
[server]
host = "127.0.0.1"
port = 18080
default_model = "gpt-5.5"
timeout = 300.0
verbose = false
max_stored_items = 1000
max_concurrent_requests = 10
# api_key = "change-me"

[codex]
auth_json = "~/.codex/auth.json"
backend_base_url = "https://chatgpt.com/backend-api/codex"
client_version = "1.0.0"

[daemon]
state_dir = "~/.config/openai-api-server-via-codex/run"
# pid_file = "/path/to/openai-api-server-via-codex.pid"
# log_file = "/path/to/openai-api-server-via-codex.log"
stop_timeout = 10.0
```

### `server.host`

Default: `127.0.0.1`

```console
$ uv run openai-api-server-via-codex --host 0.0.0.0
```

> [!IMPORTANT]
> If you bind to `0.0.0.0`, set `--api-key` or put the server behind another
> trusted access-control layer. Otherwise anyone who can reach the port can use
> your Codex credentials through this server.

### `server.port`

Default: `18080`

```console
$ uv run openai-api-server-via-codex --port 18080
```

### `server.api_key`

Default: unset

When unset, incoming `Authorization` headers are accepted and ignored.

When set, `/v1/...` routes require:

```http
Authorization: Bearer <api_key>
```

`/healthz` remains unauthenticated.

```console
$ uv run openai-api-server-via-codex --api-key local-secret
$ OPENAI_VIA_CODEX_API_KEY=local-secret uv run openai-api-server-via-codex
```

`start` passes the API key to the background `serve` process through the child
environment, not through the child command-line arguments.

### `server.max_stored_items`

Default: `1000`

This bounds the in-memory stores used for Responses context and stored Chat
Completions compatibility. Older entries are evicted first.

Set `0` to disable these stores. That also disables local
`previous_response_id` chaining and stored-object retrieval.

### `server.max_concurrent_requests`

Default: `10`

This bounds concurrent Codex backend calls. Streaming responses hold a slot
until the stream ends.

Set `0` to remove the local concurrency cap.

### `server.timeout`

Default: `300.0`

Timeout in seconds for Codex backend calls.

### `server.verbose`

Default: `false`

Verbose mode enables debug-level uvicorn logs and application diagnostics:

- resolved settings
- request start/end status and latency
- endpoint-level summaries
- model-list fallback reasons
- Codex HTTP stream/auth activity

Raw auth tokens are not logged. Token-like values in upstream errors or query
strings are redacted to a short prefix plus `******`.

```console
$ uv run openai-api-server-via-codex --verbose
$ uv run openai-api-server-via-codex status --verbose
$ uv run openai-api-server-via-codex stop --verbose
```

### `codex.auth_json`

Default: `~/.codex/auth.json`

Selects the Codex ChatGPT OAuth credentials that the server borrows when it
calls the Codex backend.

### `daemon.state_dir`

Default:

```text
~/.config/openai-api-server-via-codex/run
```

`start`, `stop`, and `status` resolve PID and log paths from this directory by
default. The default PID/log stem is derived from `host` and `port`.

If `stop` or `status` is run without `--host` and the exact default PID file is
missing, the command looks for a single PID file matching the selected port. If
multiple matches exist, it refuses to guess and asks for `--host` or
`--pid-file`.

## Recipes

### Require an API key

```console
$ uv run openai-api-server-via-codex --api-key local-secret
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="local-secret",
    base_url="http://127.0.0.1:18080/v1",
)
```

### Start on all interfaces

```console
$ uv run openai-api-server-via-codex start \
  --host 0.0.0.0 \
  --port 18080 \
  --api-key local-secret \
  --verbose
```

### Use a custom config

```console
$ uv run openai-api-server-via-codex config-generate --config ./config.toml
$ uv run openai-api-server-via-codex --config ./config.toml
```

### Use Chat Completions streaming

```python
stream = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Stream a short reply."}],
    stream=True,
    reasoning_effort="low",
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Send image input

```python
response = client.responses.create(
    model="gpt-5.5",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,...",
                },
            ],
        }
    ],
)
```

### Use tool calling

```python
response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
)
```

## Development

Run the full local validation suite:

```console
$ uv run tox
```

Run focused tests while changing request/response compatibility:

```console
$ uv run python -m pytest tests/test_openai_compat_server.py -q
$ uv run ruff check .
$ uv run ty check
```

Run live Codex integration tests only when real network/auth testing is
intended:

```console
$ RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q
$ RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_codex_http_compatibility.py -q -s
```

The live tests use the machine's existing Codex credentials and make real model
requests.
