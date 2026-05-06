# OpenAI API Server via Codex

A local server that exposes the Codex backend from your ChatGPT subscription as
an OpenAI-compatible API, so OpenAI-compatible client libraries such as
`openai-python` work without code changes.

```console
$ uvx openai-api-server-via-codex
```

Point your client's `OPENAI_BASE_URL` at `http://127.0.0.1:18080/v1`. Both
Responses and Chat Completions are supported, including streaming.

## Use cases

- Run existing code or agents written with any OpenAI-compatible client
  library (e.g. `openai-python`) through your ChatGPT subscription's Codex
  instead of `api.openai.com`
- Prototype locally or develop agents without rewriting any client code
- Use your ChatGPT plan's Codex access in personal or trusted dev workflows

This is **not** the official OpenAI Platform API or a replacement for it — it
is a compatibility layer that forwards requests to the Codex backend used by
your ChatGPT subscription. Use it only with accounts and subscriptions you are
allowed to use, and follow OpenAI's terms and usage policies. It does not
bypass Codex or ChatGPT plan limits. Do not share your Codex credentials,
resell access, power third-party services, or expose a public API backed by
your ChatGPT account.

## Usage

### Start with `uvx`

If Codex is already logged in on the machine, start the server with one command:

```console
$ uvx openai-api-server-via-codex
Codex auth preflight OK: /home/you/.codex/auth.json (account_id_present=True)
INFO:     Uvicorn running on http://127.0.0.1:18080 (Press CTRL+C to quit)
```

The default server URL is `http://127.0.0.1:18080`. OpenAI-compatible API
endpoints are served under `/v1`, for example
`http://127.0.0.1:18080/v1/responses`.

> [!TIP]
> `uvx` is uv's tool-run command. If you do not have uv installed yet, follow
> the official uv documentation: <https://docs.astral.sh/uv/>.
>
> To force `uvx` to use the latest published package instead of a cached copy,
> run `uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex`.

> [!NOTE]
> This is a compatibility server for local or trusted environments. By default,
> it accepts any incoming OpenAI API key value because `openai-python` requires
> one even when this server does not. Set `--api-key` if you want the server to
> authenticate incoming requests, especially when binding to anything other than
> localhost.

### Call the Responses API

Point `openai-python` at the local server with the standard OpenAI client
environment variables:

```console
$ export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
$ export OPENAI_API_KEY=dummy
```

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.5",
    input="Reply in one sentence.",
    reasoning={"effort": "low"},
)
print(response.output_text)
```

`OPENAI_API_KEY=dummy` is only a placeholder required by the OpenAI SDK. Unless
you configure `--api-key`, the local server accepts any incoming API key value.

### Use chat completions

```python
chat = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="low",
)
print(chat.choices[0].message.content)
```

### Stream a response

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

### Run as a background daemon

```console
$ uvx openai-api-server-via-codex start
Codex auth preflight OK: /home/you/.codex/auth.json (account_id_present=True)
Started openai-api-server-via-codex on 127.0.0.1:18080
PID: 12345
PID file: /home/you/.config/openai-api-server-via-codex/run/server-127.0.0.1-18080.pid
Log file: /home/you/.config/openai-api-server-via-codex/run/server-127.0.0.1-18080.log

$ uvx openai-api-server-via-codex status
$ uvx openai-api-server-via-codex stop
```

Expose the server to other machines only with access control:

```console
$ uvx openai-api-server-via-codex start \
  --host 0.0.0.0 \
  --api-key local-secret
```

Then connect clients to `http://<server-host>:18080/v1` and pass
`api_key="local-secret"` to the OpenAI client.

## Installation options

Run without installing:

```console
$ uvx openai-api-server-via-codex
```

Install the command onto your standard user tool path:

```console
$ uv tool install openai-api-server-via-codex
$ openai-api-server-via-codex --help
```

Upgrade an installed tool:

```console
$ uv tool upgrade openai-api-server-via-codex
$ openai-api-server-via-codex --version
```

For development from this checkout:

```console
$ uv sync --dev
$ uv run openai-api-server-via-codex --help
```

## Requirements

- Python 3.10+
- `uv`
- A working Codex login, usually at `~/.codex/auth.json`

Use an explicit Codex auth file when needed:

```console
$ uvx openai-api-server-via-codex --auth-json ~/.codex/auth.json
$ OPENAI_VIA_CODEX_AUTH_JSON=~/.codex/auth.json uvx openai-api-server-via-codex
```

`serve` and `start` validate the Codex auth file before starting. If the file is
missing, not valid JSON, not a ChatGPT Codex auth file, missing tokens, expired
without a refresh token, or fails token refresh, the server exits before it
binds the HTTP port.

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
not use it to evade limits, share account access, resell access, or power
third-party services. Do not expose it to untrusted networks without `--api-key`
or another access control layer, and follow OpenAI's
[Terms of Use](https://openai.com/policies/terms-of-use/) and
[Usage Policies](https://openai.com/policies/usage-policies/).

## API endpoints

The endpoints below are implemented locally for OpenAI-compatible behavior.
They normalize Codex HTTP requests, translate streaming events, and maintain
the in-memory compatibility stores used by Responses and stored Chat
Completions.

| Method | Path |
| --- | --- |
| `GET` | `/healthz` |
| `GET` | `/v1/models` |
| `POST` | `/v1/responses` |
| `GET` | `/v1/responses/{response_id}` |
| `DELETE` | `/v1/responses/{response_id}` |
| `POST` | `/v1/responses/{response_id}/cancel` |
| `POST` | `/v1/responses/input_tokens` |
| `POST` | `/v1/chat/completions` |
| `GET` | `/v1/chat/completions` |
| `GET` | `/v1/chat/completions/{completion_id}` |
| `POST` | `/v1/chat/completions/{completion_id}` |
| `DELETE` | `/v1/chat/completions/{completion_id}` |
| `GET` | `/v1/chat/completions/{completion_id}/messages` |

For any other `/v1/...` request, the server falls back to a best-effort proxy:
it forwards the method, path, query string, safe OpenAI-style request headers,
and raw request body to the Codex HTTP backend, then returns the upstream status,
body, and safe response headers. This allows endpoints that are not implemented
locally, including Codex-specific or newly added OpenAI-style paths, to be tried
without adding a compatibility shim for each endpoint.

The fallback proxy uses the local Codex credentials selected by this server. It
does not forward the incoming `Authorization` header, local `--api-key`, or
cookies to Codex HTTP. Successful behavior still depends on what the upstream
Codex HTTP backend accepts for that path; unsupported upstream paths may return
Codex HTTP errors such as `400`, `403`, or `404`.

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
$ uvx openai-api-server-via-codex config-generate
$ uvx openai-api-server-via-codex config-generate --stdout
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
$ uvx openai-api-server-via-codex --host 0.0.0.0
```

> [!IMPORTANT]
> If you bind to `0.0.0.0`, set `--api-key` or put the server behind another
> trusted access-control layer. Otherwise anyone who can reach the port can use
> your Codex credentials through this server.

### `server.port`

Default: `18080`

```console
$ uvx openai-api-server-via-codex --port 18080
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
$ uvx openai-api-server-via-codex --api-key local-secret
$ OPENAI_VIA_CODEX_API_KEY=local-secret uvx openai-api-server-via-codex
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
$ uvx openai-api-server-via-codex --verbose
$ uvx openai-api-server-via-codex status --verbose
$ uvx openai-api-server-via-codex stop --verbose
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
$ uvx openai-api-server-via-codex --api-key local-secret
```

```python
from openai import OpenAI

client = OpenAI()
```

Run the client with `OPENAI_BASE_URL=http://127.0.0.1:18080/v1` and
`OPENAI_API_KEY=local-secret`.

### Start on all interfaces

```console
$ uvx openai-api-server-via-codex start \
  --host 0.0.0.0 \
  --port 18080 \
  --api-key local-secret \
  --verbose
```

### Use a custom config

```console
$ uvx openai-api-server-via-codex config-generate --config ./config.toml
$ uvx openai-api-server-via-codex --config ./config.toml
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

## Release

The package is released to PyPI through GitHub Actions Trusted Publishing. Use
the release checklist in [docs/release.md](docs/release.md).

The recommended production path is PyPI Trusted Publishing from GitHub Actions
with the `pypi` environment. Local release work should build, inspect, and smoke
test the artifacts before the tag is pushed.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

- Simon Willison's article,
  [A pelican for GPT-5.5 via the semi-official Codex backdoor API](https://simonwillison.net/2026/Apr/23/gpt-5-5/),
  and the implementation described there were the key references for this
  project. Without that article, this approach likely would not have been
  implemented here. Thank you to Simon for documenting the route clearly.
- [OpenClaw](https://github.com/openclaw/openclaw) was a useful reference for
  understanding Codex backend integration patterns.
- [Pi Monorepo](https://github.com/badlogic/pi-mono) was a useful reference for
  Codex backend API behavior and compatibility details.

## Author

- Yuichi Tateno ([@hotchpotch](https://github.com/hotchpotch))

<img src="https://secon.dev/images/profile_usa.png" width="64" height="64" alt="Yuichi Tateno" />
