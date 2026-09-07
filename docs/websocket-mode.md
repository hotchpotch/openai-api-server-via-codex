# Responses WebSocket mode with Codex

Planned for 0.2.1; currently available from this development branch.

`GET /v1/responses` accepts a WebSocket upgrade and opens a matching connection
at the configured Codex backend. The Go server substitutes Codex authentication
and relays messages in both directions. Conversation state stays on the upstream
connection; the server does not create background jobs, expand
`previous_response_id`, or store WebSocket responses in its HTTP compatibility
store.

The following flows have live SDK E2E coverage with `gpt-6-astra` through
`https://chatgpt.com/backend-api/codex/responses`:

- Sequential responses using the same connection and `previous_response_id`.
- Async function and custom tools, including a conversation turn before a tool
  result arrives, followed by the result using the original `call_id`.
- Mid-turn steering through `response.steer`, including acceptance and the
  successor response containing the updated instructions.
- Both `OpenAI` and `AsyncOpenAI` WebSocket clients.

This is Codex backend compatibility, not a promise that every feature of the
public OpenAI API is available. Model/account access and upstream support still
apply.

## Model support

Live probes through this Go server and the Codex backend on 2026-09-07 showed:

| Capability | `gpt-6-astra` | `gpt-5.6-luna` | `gpt-5.6-terra` |
| --- | --- | --- | --- |
| WebSocket generation | Supported | Supported | Supported |
| Ordinary function tool calls | Supported | Supported | Supported |
| Async function/custom tools | Supported | Rejected | Rejected |
| `response.steer` | Supported | Rejected | Rejected |

For Luna and Terra, async tool requests produce a WebSocket `error` event with
`code: unsupported_value`, `param: tools`, and a model-specific message such as
`Async tools are not supported with gpt-5.6-luna.` Steering produces
`response.steer.failed` with `code: steering_not_supported`. HTTP async tool
requests were also tested on Luna and return HTTP 400 (`BadRequestError` in the
Python SDK). Ordinary Luna WebSocket continuation with `previous_response_id`
was verified across two turns.

The relay does not select a different model or emulate unsupported features.
Background rejection is local and applies to every model. Upstream named-stream
rejection was observed on Astra and Luna. These observations describe the tested
backend/account, rather than a guarantee for every model or future backend.

## Connect with the official Python SDK

Install `openai[realtime]>=3.8.0,<4` in the consumer application. Python remains
only the SDK consumer; the server runtime is Go.

```python
import asyncio
from openai import AsyncOpenAI


async def main():
    async with (
        AsyncOpenAI(
            api_key="local",  # Use the configured server API key when enabled.
            base_url="http://127.0.0.1:18080/v1",
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        previous_id = None
        for prompt in ["Reply exactly WS_OK", "Repeat the token you just said."]:
            args = {
                "model": "gpt-6-astra",
                "reasoning": {"effort": "low"},
                "input": prompt,
            }
            if previous_id:
                args["previous_response_id"] = previous_id
            await connection.response.create(**args)
            while True:
                event = await asyncio.wait_for(connection.recv(), timeout=120)
                data = event.model_dump(by_alias=True, exclude_none=True)
                if data["type"] == "response.output_text.delta":
                    print(data["delta"], end="", flush=True)
                elif data["type"] == "response.completed":
                    previous_id = data["response"]["id"]
                    print()
                    break
                elif data["type"] in {"error", "response.failed", "response.incomplete"}:
                    raise RuntimeError(data)


asyncio.run(main())
```

Use one reader per connection. A Codex terminal event may have an empty
`response.output` even when output was emitted earlier. Collect
`response.output_item.done` items (and text deltas as needed); the WebSocket
relay intentionally does not synthesize a different terminal response.

The example disables SDK reconnection retries. The Go relay never reconnects
or replays events after upgrading. If a connection is lost, retain client-side
history and pending tool results and explicitly start a new conversation with
that history. Do not assume an old response ID is available on a new connection,
or execute an already-started tool again just because the connection closed.

## Async tools

Define a function or custom tool with `async: True` and pass it in `tools`.
The application executes the tool and tracks its pending work. After an async
call, independent conversation turns can proceed before its output is ready.
Return a `function_call_output` or `custom_tool_call_output` with the original
`call_id`, using the latest response ID on that connection.

The relay preserves `async`, custom tool items, encrypted reasoning, unknown
fields, and upstream events. When serializing SDK objects for input history,
use `model_dump(by_alias=True, exclude_none=True)` so the Python attribute
`async_` is serialized as the wire field `async`.

The HTTP Responses path also preserves async call attributes, custom tool calls,
reasoning, and assistant phases when normalizing explicit input or constructing
its local `previous_response_id` history. HTTP still uses the existing in-memory
compatibility store for ID-based continuation; this is separate from WebSocket
connection-local state.

## Steering

After receiving `response.created`, send:

```python
await connection.response.steer(
    previous_response_id=running_response_id,
    input="Change the task: reply exactly STEER_OK.",
)
```

Continue reading. `response.steer.accepted` acknowledges queued input, not a
finished response. The original response may finish normally or end with
`response.incomplete` and reason `steered`. Wait for the successor response ID
and its completion to determine whether the new instructions were applied.
Steering does not undo emitted text or tools that already ran. Tool-result or
approval requirements remain the application's responsibility; the relay also
preserves `response.steer.pending` and `response.steer.failed`.

## Upstream limitations

- **Background mode:** `background=true` is rejected with
  `unsupported_parameter` for `background`, on both HTTP and WebSocket create.
  `background=false` is removed before forwarding. No local job runner emulates
  this unsupported upstream capability.
- **Named streams/multiplexing:** Codex currently rejects `stream_id` with
  `Unsupported parameter: stream_id`. The relay preserves the field and forwards
  the upstream error rather than emulating named lanes or forks. Omit
  `stream_id`; use separate connections for parallel conversations.
- **HTTP response IDs:** Codex HTTP rejects `previous_response_id`; the existing
  HTTP compatibility handler resolves these IDs locally. WebSocket continuation
  uses the actual upstream IDs on the same connection.
- **HTTP helper endpoints:** WebSocket response IDs are not registered with
  `responses.retrieve`, `input_items`, or `cancel` on the local server. Closing
  a WebSocket closes its upstream transport; there is no local background
  response to poll or cancel.

## Transport and operations

The relay defaults omitted model/instructions and the usual Codex tool/text
settings. It forces `store=false`, removes `stream`, `background=false`, and
`max_output_tokens`, and applies configured `drop_params` to `response.create`.
String input becomes a user input item. Existing input items, IDs, `response.steer`,
and other events otherwise pass through. Error payloads are redacted; ordinary
model text and encrypted reasoning are not rewritten.

`max_concurrent_requests` counts each open WebSocket as one backend request for
its entire lifetime, including idle time, and shares the limit with HTTP requests.
Requests arriving while the cap is full receive HTTP 503 with `Retry-After: 1`,
including WebSocket handshakes. Close idle connections to release slots. `0`
disables this local cap. The timeout setting bounds the upstream handshake and each relay write, not the lifetime of
an established idle connection. The relay sends upstream Ping frames every 20
seconds and bounds each Pong wait by the timeout setting. This detects dead peers
and provides transport keepalive; upstream session expiration still applies.
Apply response deadlines in the consumer.

The incoming API key is checked before connecting upstream and is never used as
Codex authentication. An upstream handshake `401` causes one auth-file reload and
one retry, before the client is upgraded. Errors after upgrade are not retried.
Cross-origin browser upgrades are rejected; SDK clients without an Origin header
are supported. Server shutdown sends close code 1001 and releases active relay
slots. HTTP and WebSocket shutdown share the configured stop timeout; the HTTP listener closes
while relays drain, and remaining sockets are forcibly closed at the deadline.
Valid peer close codes are preserved with empty reasons; transport failures use 1011.
Normal access logs record connection status/duration; verbose mode adds connection
lifecycle diagnostics without logging message bodies or credentials.

## Validation

```console
uv run python -m pytest tests/test_openai_client_contract.py -q
go test -race ./...
uv run tox
RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_astra_websocket.py -q -s
```

The live suite builds and launches the Go binary, uses the existing Codex login,
and runs on `gpt-6-astra`. It is skipped by default. Logs print text markers,
parsed tool arguments, and event sequences, without encrypted reasoning or auth
material. Review the output for semantic correctness as well as test success.
The named-stream live case verifies the current upstream rejection explicitly.

Official references: [WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode),
[async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling),
[mid-turn steering](https://developers.openai.com/api/docs/guides/steering), and
[background mode](https://developers.openai.com/api/docs/guides/background).

To check a longer pending-tool idle period against the real backend, set
`CODEX_WS_IDLE_SECONDS` (minimum 25 seconds). The test keeps the original socket
and call ID, disables reconnection, waits without application events, then submits
the tool result and checks the completed answer:

```console
RUN_CODEX_LIVE_TESTS=1 CODEX_WS_IDLE_SECONDS=600 uv run python -m pytest tests/test_live_astra_websocket.py::test_live_astra_websocket_async_tools -q -s
```

The deterministic SDK load regression opens 10 idle connections, sends 50
concurrent requests expecting 503, closes those connections, and verifies a new
conversation completes. It repeats this cycle 10 times without real model calls:

```console
uv run python -m pytest tests/test_openai_client_contract.py::test_contract_websocket_saturation_recovers_under_load -q -s
```
