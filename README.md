# OpenAI API Server via Codex

Run an OpenAI-compatible local API server using the Codex access from your
ChatGPT login. The server itself is a standalone Go executable; clients can use
the standard Responses and Chat Completions APIs by changing their base URL.

> [!IMPORTANT]
> **Breaking change in v0.2.0:** the server is now implemented in Go. The former
> Python/FastAPI HTTP server and Python fallback have been removed. The Python
> package remains only as a small `uvx` launcher that selects the bundled Go
> executable.

![Start the Go server with uvx, then call the OpenAI-compatible Responses API](https://raw.githubusercontent.com/hotchpotch/openai-api-server-via-codex/main/docs/assets/quick-start.png)

## Quick start

Make sure Codex is logged in and `~/.codex/auth.json` exists, then start the Go
server with either `uvx` or Docker.

### `uvx`

```console
$ uvx openai-api-server-via-codex
2026/08/12 12:34:56 openai-api-server-via-codex 0.2.0 (Go) listening on http://127.0.0.1:18080
```

### Docker

The public image needs no registry login. On amd64, the registry manifest
contains about 7 MB of compressed layers and the local Docker image is about
21 MB. Idle memory use is a few MiB (roughly 4–7 MiB in local measurements);
exact values depend on the release, architecture, host, and container runtime.

```console
$ docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:latest
$ docker run --rm -p 127.0.0.1:18080:18080 \
    -v ~/.codex:/home/app/.codex \
    ghcr.io/hotchpotch/openai-api-server-via-codex:latest
```

Point an OpenAI client at the local endpoint:

```console
$ export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
$ export OPENAI_API_KEY=dummy
```

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5.6-luna",
    input="Reply in one sentence.",
    reasoning={"effort": "low"},
)
print(response.output_text)
```

`OPENAI_API_KEY=dummy` only satisfies the OpenAI SDK's client-side validation.
The server accepts any incoming value unless you configure its `--api-key`.

## Why use it

- **OpenAI-compatible:** use existing `openai-python`, LangChain, LiteLLM, and
  other clients with a configurable base URL.
- **Codex-backed:** requests use the Codex access associated with your ChatGPT
  login instead of OpenAI Platform API credentials.
- **Go runtime:** fast startup, low resident memory, a single server executable,
  and no Python web stack.
- **Broad API coverage:** Responses, Chat Completions, streaming, tools,
  structured outputs, image input, image generation, audio transcription, and
  locally stored compatibility objects.
- **Local by default:** the server binds to `127.0.0.1`, validates Codex auth
  before listening, and supports an incoming API key when remote access is
  required.
- **Portable:** binary wheels cover Linux, macOS, and Windows on x86_64 and
  ARM64; Docker and direct Go builds are also supported.

This project does not raise or bypass Codex or ChatGPT plan limits. It is not
the official OpenAI Platform API. Use it only with accounts you are authorized
to use, and do not expose it publicly or resell access.

## Installation and execution

Choose one of these paths:

| Method | Host requirements | Server runtime |
| --- | --- | --- |
| `uvx openai-api-server-via-codex` | `uv`, Codex login | Bundled Go executable |
| `docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:latest` | Docker, Codex login | Public Go/Alpine image (~7 MB compressed/~21 MB local on amd64; a few MiB idle memory) |
| `docker compose up --build -d` | Docker, Codex login | Go on Alpine Linux |
| Build from source | Go 1.23+, Codex login | Locally built Go executable |

### Run with `uvx`

Published wheels contain the Go executable for:

- Linux x86_64 and ARM64
- macOS Intel and Apple silicon
- Windows x86_64 and ARM64

Run without a permanent installation:

```console
$ uvx openai-api-server-via-codex
```

Or install the launcher and bundled executable on your user tool path:

```console
$ uv tool install openai-api-server-via-codex
$ openai-api-server-via-codex --version
```

There is no Python server fallback and no generic source distribution. On an
unsupported platform, build the Go executable directly.

### Run with Docker

The production image builds the server from source and copies only the static
Go executable and CA certificates into a small Alpine runtime. Python and the
Go toolchain are absent from the final server image.

Stable Linux x86_64 and ARM64 images are published at
`ghcr.io/hotchpotch/openai-api-server-via-codex`. `latest` tracks the newest
stable release, while exact tags such as `v0.2.0` provide reproducible
deployments. Prereleases publish only their exact version tag and do not move
`latest`. The package is public, so pulls do not require a registry login.

```console
$ docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:latest
$ docker run --rm -p 127.0.0.1:18080:18080 \
    -v ~/.codex:/home/app/.codex \
    ghcr.io/hotchpotch/openai-api-server-via-codex:latest
```

Or build the same runtime image from this checkout:

```console
$ docker compose run --rm --service-ports codex-login  # only if auth.json is missing
$ docker compose up --build -d
$ curl http://127.0.0.1:18080/healthz
```

The server runs as a non-root user and mounts `~/.codex` read-write so refreshed
tokens can be saved. See [the Docker guide](docs/docker.md) for login methods,
permissions, configuration, and plain `docker run` usage.

### Build your own Go binary

Go can download, build, and install the command directly from its GitHub module
path:

```console
$ go install github.com/hotchpotch/openai-api-server-via-codex/cmd/openai-api-server-via-codex@latest
$ "$(go env GOPATH)/bin/openai-api-server-via-codex" --version
```

The executable is installed under `GOBIN`, or under `$(go env GOPATH)/bin`
when `GOBIN` is unset.

To build from a checkout instead:

```console
$ go build -trimpath -o ./bin/openai-api-server-via-codex ./cmd/openai-api-server-via-codex
$ ./bin/openai-api-server-via-codex --version
$ ./bin/openai-api-server-via-codex serve
```

This path does not require Python or `uv`. See
[Building the Go binary from source](docs/build-from-source.md) for version
stamping, installation, Windows commands, static builds, and cross-compilation.

## Authentication and security

The server borrows a Codex ChatGPT login, normally from
`~/.codex/auth.json`. `serve` and `start` validate the file before binding the
HTTP port. Invalid, missing, expired, or unrefreshable credentials fail startup
with a redacted error.

The server notices external changes to `auth.json` without a restart. If an
upstream request receives `401 Unauthorized` before any streaming response has
started, it discards its credential cache, reloads the file, and retries that
request once. This covers Codex CLI or another process rotating the token
between requests; a second `401` is returned without another retry.

Select another auth file when needed:

```console
$ openai-api-server-via-codex --auth-json /path/to/auth.json
$ OPENAI_VIA_CODEX_AUTH_JSON=/path/to/auth.json openai-api-server-via-codex
```

The incoming OpenAI-compatible API key is separate from Codex authentication.
Protect `/v1/...` routes when clients can reach the server over a network:

```console
$ openai-api-server-via-codex \
    --host 0.0.0.0 \
    --api-key local-secret
```

`/healthz` remains unauthenticated. Incoming API keys, cookies, and
`Authorization` headers are never forwarded to the Codex backend.

<details>
<summary><strong>More OpenAI client examples</strong></summary>

### Chat Completions

```python
chat = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="low",
)
print(chat.choices[0].message.content)
```

### Streaming Responses

```python
stream = client.responses.create(
    model="gpt-5.6-luna",
    input="Stream a short reply.",
    stream=True,
    reasoning={"effort": "low"},
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

### Streaming Chat Completions

```python
stream = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "Stream a short reply."}],
    stream=True,
    reasoning_effort="low",
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Image input

```python
response = client.responses.create(
    model="gpt-5.6-luna",
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

### Image generation

```python
import base64

image = client.images.generate(
    model="gpt-image-2",
    prompt="A cozy pixel art bowl of ramen, no text.",
    size="1024x1024",
    quality="medium",
    output_format="png",
)

with open("ramen.png", "wb") as file:
    file.write(base64.b64decode(image.data[0].b64_json))
```

Image generation returns `data[].b64_json`; URL results, image editing, and
streamed partial images are not implemented.

### Tool calling

```python
response = client.responses.create(
    model="gpt-5.6-luna",
    input="What is the weather in Tokyo?",
    tools=[
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ],
)
```

</details>

<details>
<summary><strong>Background daemon commands</strong></summary>

The Go executable implements foreground and daemon lifecycle commands:

```console
$ openai-api-server-via-codex start
$ openai-api-server-via-codex status
$ openai-api-server-via-codex stop
```

PID and log files default to:

```text
~/.config/openai-api-server-via-codex/run/
```

On Linux and macOS, `stop` drains in-flight HTTP requests up to
`--stop-timeout`. Windows terminates the daemon process tree on a best-effort
basis, so an active stream may be interrupted.

Docker should run `serve` in the foreground and let the container runtime manage
restarts; do not use `start` inside a container.

</details>

<details>
<summary><strong>API surface and compatibility details</strong></summary>

### Implemented endpoints

| Method | Path |
| --- | --- |
| `GET` | `/healthz` |
| `GET` | `/v1/models` |
| `POST` | `/v1/responses` |
| `GET` | `/v1/responses/{response_id}` |
| `DELETE` | `/v1/responses/{response_id}` |
| `POST` | `/v1/responses/{response_id}/cancel` |
| `POST` | `/v1/responses/input_tokens` |
| `POST` | `/v1/audio/transcriptions` |
| `POST` | `/v1/images/generations` |
| `POST` | `/v1/chat/completions` |
| `GET` | `/v1/chat/completions` |
| `GET` | `/v1/chat/completions/{completion_id}` |
| `POST` | `/v1/chat/completions/{completion_id}` |
| `DELETE` | `/v1/chat/completions/{completion_id}` |
| `GET` | `/v1/chat/completions/{completion_id}/messages` |

Unknown `/v1/...` requests use a best-effort fallback proxy. The Go server
forwards the method, path, query, safe OpenAI-style headers, and body with its
own Codex credentials. Upstream support still determines whether such a request
returns `2xx`, `400`, `403`, or `404`.

### Compatibility behavior

- sync and async `openai-python` clients
- non-streaming and streaming Responses and Chat Completions
- `previous_response_id` backed by bounded local context
- stored Chat list/retrieve/update/delete/messages APIs
- Responses retrieve streaming, delete, cancel, and input-token count
- function/tool calling and streamed tool arguments
- JSON mode and structured outputs
- URL and data-URL image input
- hosted image generation translated through a Codex Responses tool
- optional incoming API-key authentication
- bounded request concurrency and in-memory stores

At the Codex boundary, requests are normalized to `stream=true`, `store=false`,
low text verbosity by default, Codex-compatible tool defaults, and encrypted
reasoning content. Public storage compatibility is implemented in the Go
server's bounded in-memory stores.

Model listing is best-effort. A model can sometimes accept direct requests even
when it is absent from the upstream catalog returned by `/v1/models`.

</details>

<details>
<summary><strong>Configuration reference</strong></summary>

Generate a configuration file:

```console
$ openai-api-server-via-codex config-generate
$ openai-api-server-via-codex config-generate --stdout
```

The default path is
`$XDG_CONFIG_HOME/openai-api-server-via-codex/config.toml`, falling back to
`~/.config/openai-api-server-via-codex/config.toml`.

Settings resolve in this order:

```text
CLI flag -> environment variable -> config file -> default
```

```toml
[server]
host = "127.0.0.1"
port = 18080
default_model = "gpt-5.6-luna"
timeout = 300.0
verbose = false
max_stored_items = 1000
max_concurrent_requests = 10
# api_key = "change-me"

[codex]
auth_json = "~/.codex/auth.json"
backend_base_url = "https://chatgpt.com/backend-api/codex"
client_version = "1.0.0"

[compat]
drop_params = []

[daemon]
state_dir = "~/.config/openai-api-server-via-codex/run"
stop_timeout = 10.0
```

### Important server settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `server.host` | `127.0.0.1` | HTTP bind address |
| `server.port` | `18080` | HTTP port |
| `server.default_model` | `gpt-5.6-luna` | Model used when a request omits one |
| `server.api_key` | unset | Protect incoming `/v1/...` requests |
| `server.max_stored_items` | `1000` | Bound local Responses/Chat stores; `0` disables |
| `server.max_concurrent_requests` | `10` | Bound full Codex requests/streams; `0` disables |
| `server.timeout` | `300.0` | Codex backend timeout in seconds |
| `server.verbose` | `false` | Enable redacted Go application diagnostics |
| `codex.auth_json` | `~/.codex/auth.json` | Codex OAuth file |
| `compat.drop_params` | `[]` | Top-level request fields removed before forwarding |

CLI, environment, and config examples:

```console
$ openai-api-server-via-codex --port 19090 --verbose
$ OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS=20 openai-api-server-via-codex
$ openai-api-server-via-codex --config ./config.toml
```

Use `drop_params` only for parameters known to be rejected by the Codex
backend:

```toml
[compat]
drop_params = ["temperature", "top_p"]
```

Normal operation logs one completion line per API request with its method,
redacted path, status, response size, and duration. Routine `/healthz` probes
stay quiet. Verbose logs additionally include request starts and redacted query
strings, resolved settings, endpoint summaries, and Codex stream/auth activity.
Raw credentials and token-like values are redacted.

</details>

<details>
<summary><strong>Development and test guide</strong></summary>

The HTTP server, backend integration, auth, configuration, stores, daemon, and
redaction logic are all implemented in Go under `cmd/` and `internal/`.
Python is not a server implementation: it is used only for the `uvx` launcher,
the official `openai-python` consumer contract, and release tooling.

Requirements for full repository development:

- Go 1.23 or newer
- Python 3.10 or newer
- `uv`

Run the complete deterministic validation suite:

```console
$ uv run tox
```

Focused Go validation:

```console
$ go test ./internal/app
$ go test ./test/e2e -v
$ go test ./...
$ go vet ./...
$ go test -race ./...
```

OpenAI SDK consumer compatibility:

```console
$ uv run python -m pytest tests/test_openai_client_contract.py -q
```

Real Codex tests are opt-in because they use the current login, network, model
allowance, and image quota:

```console
$ RUN_CODEX_LIVE_TESTS=1 go test ./test/live -v -count=1 -timeout=20m
$ RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_integration.py -q -s
$ RUN_CODEX_LIVE_TESTS=1 uv run python -m pytest tests/test_live_codex_http_compatibility.py -q -s
```

See also:

- [Building the Go binary from source](docs/build-from-source.md)
- [Go runtime test policy](docs/go-migration.md)
- [Historical Go/Python performance comparison](docs/performance.md)
- [Release process](docs/release.md)

</details>

## Disclaimer

Use this project at your own risk. It is not the official OpenAI Platform API
and is not endorsed or supported by OpenAI. It forwards requests to the Codex
HTTP backend used by the Codex CLI and ChatGPT subscription flow instead of
`api.openai.com`; that backend may change without notice.

Use the server only with accounts and subscriptions you are authorized to use.
Do not evade limits, share account access, resell access, or expose the service
to untrusted networks without authentication. Follow OpenAI's
[Terms of Use](https://openai.com/policies/terms-of-use/) and
[Usage Policies](https://openai.com/policies/usage-policies/).

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

<img height="160" src="https://storage.googleapis.com/secons-site-images/other/blog_images/secon_icon_nendo.webp" alt="Yuichi Tateno" />
