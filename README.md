OpenAI API Server via Codex
===========================

FastAPI-based OpenAI-compatible API facade that forwards generation requests to
the Codex ChatGPT backend using the local `codex login` credentials.

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

Run as a background server:

```bash
uv run openai-api-server-via-codex start --host 127.0.0.1 --port 8000
uv run openai-api-server-via-codex status --host 127.0.0.1 --port 8000
uv run openai-api-server-via-codex stop --host 127.0.0.1 --port 8000
```

`start` prints the PID file and log file paths. By default they live under
`$XDG_STATE_HOME/openai-api-server-via-codex`, or
`~/.local/state/openai-api-server-via-codex` when `XDG_STATE_HOME` is unset.
Override with `--state-dir`, `--pid-file`, `--log-file`, or the matching
`OPENAI_VIA_CODEX_STATE_DIR`, `OPENAI_VIA_CODEX_PID_FILE`,
`OPENAI_VIA_CODEX_LOG_FILE` environment variables. Foreground mode remains the
default, so `uvx openai-api-server-via-codex` starts the server in the current
terminal.

Backend selection:

```bash
# Default ChatGPT Codex HTTP backend
uv run openai-api-server-via-codex --backend chatgpt-http --port 8000

# Experimental native Codex app-server backend
uv run openai-api-server-via-codex --backend codex-app-server --port 8001
```

The app-server backend starts `codex app-server --listen stdio://`, logs in with
the configured Codex `auth.json` tokens, and keeps native Codex threads in
memory so `previous_response_id` can continue a thread without replaying the
full local response context. Use `--codex-bin` or `OPENAI_VIA_CODEX_CODEX_BIN`
to select a Codex binary, and `--app-server-cwd` or
`OPENAI_VIA_CODEX_APP_SERVER_CWD` to set the Codex thread working directory.
The default backend remains `chatgpt-http`; select `codex-app-server`
explicitly because the app-server protocol is still experimental.

Use with `openai-python`:

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:8000/v1")

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

Live integration test:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_dual_backend_integration.py -q -s
```
