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
```

`previous_response_id` is supported locally by storing response context in
memory and replaying it into the next Codex request. Chat Completions multi-turn
works through the standard `messages` list.

Image input is supported for URL and data URL image parts, for example
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.

Streaming is not implemented yet; `stream=true` currently returns a structured
OpenAI-style error response.

Live integration test:

```bash
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q
```
