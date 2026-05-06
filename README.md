OpenAI API Server via Codex
===========================

FastAPI-based OpenAI-compatible API facade that forwards generation requests to
the Codex HTTP backend using the local `codex login` credentials.

Supported endpoints:

- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`

Run locally:

```bash
uv run openai-api-server-via-codex
```

Use an explicit Codex auth file:

```bash
uv run openai-api-server-via-codex --auth-json ~/.codex/auth.json
# or
OPENAI_VIA_CODEX_AUTH_JSON=~/.codex/auth.json uv run openai-api-server-via-codex
```

Generate a default config file:

```bash
uv run openai-api-server-via-codex config-generate
uv run openai-api-server-via-codex config-generate --stdout
uv run openai-api-server-via-codex --config ~/.config/openai-api-server-via-codex/config.toml
```

By default the config file path is
`$XDG_CONFIG_HOME/openai-api-server-via-codex/config.toml`, or
`~/.config/openai-api-server-via-codex/config.toml` when `XDG_CONFIG_HOME` is
unset. You can also set `OPENAI_VIA_CODEX_CONFIG` or pass `--config` to
`serve`, `start`, `stop`, and `status`.

The TOML config mirrors the CLI options:

```toml
[server]
host = "127.0.0.1"
port = 18080
default_model = "gpt-5.4"
timeout = 180.0
verbose = false
max_stored_items = 1000

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

Resolution order is CLI flag, environment variable, config file, then default.

Run as a background server:

```bash
uv run openai-api-server-via-codex start --host 127.0.0.1 --port 18080
uv run openai-api-server-via-codex status --host 127.0.0.1 --port 18080
uv run openai-api-server-via-codex stop --host 127.0.0.1 --port 18080
```

`start` prints the PID file and log file paths. By default they live under the
config directory's `run/` subdirectory, for example
`~/.config/openai-api-server-via-codex/run`.
Override with `--state-dir`, `--pid-file`, `--log-file`, or the matching
`OPENAI_VIA_CODEX_STATE_DIR`, `OPENAI_VIA_CODEX_PID_FILE`,
`OPENAI_VIA_CODEX_LOG_FILE` environment variables, or with the `[daemon]`
config table. Foreground mode remains the default, so
`uvx openai-api-server-via-codex` starts the server in the current terminal.

Codex HTTP backend:

```bash
uv run openai-api-server-via-codex --port 18080
uv run openai-api-server-via-codex --verbose
```

`--verbose`, `OPENAI_VIA_CODEX_VERBOSE=1`, or `[server].verbose = true` enables
debug-level uvicorn logging plus application diagnostics. The server logs the
resolved config/settings, request start/end status and latency, endpoint-level
summaries for Responses and Chat Completions, model listing fallback reasons,
and Codex HTTP stream/auth activity. Auth tokens are not logged; only auth file
paths and whether a ChatGPT account id was present are reported.

When using `start`, pass `--verbose` or set it in config/env to preserve the
same diagnostics in the background server log file printed by `start`.

The in-memory compatibility stores are bounded by `max_stored_items`, default
`1000`. This cap applies to stored Responses and stored Chat Completions. Older
entries are evicted first; set `--max-stored-items`,
`OPENAI_VIA_CODEX_MAX_STORED_ITEMS`, or `[server].max_stored_items` to tune the
limit. Setting it to `0` disables these in-memory stores, which also disables
local `previous_response_id` chaining and stored-object retrieval.

The server forwards OpenAI Responses-compatible payloads to the Codex HTTP
backend using the borrowed Codex OAuth token. This route preserves normal
OpenAI function-calling semantics where the API returns `function_call` /
`tool_calls` and the client sends tool results in a later request.
For Codex compatibility, backend requests are normalized to streaming
Responses calls with `store = false`, low text verbosity by default, automatic
tool choice defaults, and `reasoning.encrypted_content` included for reasoning
context. Public `store=true` behavior remains local to this compatibility
server's in-memory stores.

Use with `openai-python`:

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:18080/v1")

response = client.responses.create(
    model="gpt-5.4",
    input="Reply in one sentence.",
    reasoning={"effort": "low"},
)
print(response.output_text)

chat = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="low",
)
print(chat.choices[0].message.content)

stream = client.responses.create(
    model="gpt-5.4",
    input="Stream a short reply.",
    stream=True,
    reasoning={"effort": "low"},
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")

chat_stream = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Stream a short reply."}],
    stream=True,
    reasoning_effort="low",
)
for chunk in chat_stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

`previous_response_id` is supported locally by storing response context in
memory and replaying it into the next Codex request. Chat Completions multi-turn
works through the standard `messages` list.

Image input is supported for URL and data URL image parts, for example
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.

Streaming is supported for both Responses and Chat Completions. Chat
Completions streaming converts Codex Responses events into
`chat.completion.chunk` deltas, including text, function tool-call arguments,
finish reasons, and `stream_options={"include_usage": true}` usage chunks.

Function calling is supported for both streaming and non-streaming Chat
Completions.

Live integration test:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_codex_http_compatibility.py -q -s
```
