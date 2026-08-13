# OpenAI API Server via Codex

[![CI](https://github.com/hotchpotch/openai-api-server-via-codex/actions/workflows/ci.yml/badge.svg)](https://github.com/hotchpotch/openai-api-server-via-codex/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openai-api-server-via-codex.svg)](https://pypi.org/project/openai-api-server-via-codex/)
[![GitHub Release](https://img.shields.io/github/v/release/hotchpotch/openai-api-server-via-codex?include_prereleases)](https://github.com/hotchpotch/openai-api-server-via-codex/releases)
[![License](https://img.shields.io/github/license/hotchpotch/openai-api-server-via-codex)](LICENSE)

Run a local OpenAI-compatible API server using the Codex access from your
ChatGPT login. Existing OpenAI clients can use Responses and Chat Completions
by changing their base URL.

The server is a small standalone Go executable. The Python package is only a
cross-platform `uvx` launcher for the bundled executable.

> [!TIP]
> **Already have Codex in your ChatGPT plan?** Within your included Codex usage
> limits, this server lets OpenAI-compatible clients use that access without a
> real OpenAI Platform API key or separate per-token Platform API charges. In
> other words, there is no additional API charge within the included allowance.
> Plan limits still apply, and extending usage with ChatGPT credits may cost
> extra. See the [official Codex pricing](https://learn.chatgpt.com/docs/pricing).

> [!IMPORTANT]
> **Upgrading to v0.2.0:** the HTTP server is now implemented in Go. The former
> Python/FastAPI server and Python fallback have been removed. See
> [Building from source](docs/build-from-source.md) if your platform is not
> covered by a bundled binary.

![Start the Go server with uvx, then call the OpenAI-compatible Responses API](https://raw.githubusercontent.com/hotchpotch/openai-api-server-via-codex/main/docs/assets/quick-start.png)

## Quick start

### 1. Sign in with Codex

Install the [Codex CLI](https://github.com/openai/codex) if needed, then create
`~/.codex/auth.json`:

```console
$ codex login
```

### 2. Start the server

The recommended installation requires only [`uv`](https://docs.astral.sh/uv/):

```console
$ uvx openai-api-server-via-codex
```

The server listens on `http://127.0.0.1:18080` by default. In another terminal,
confirm that it is healthy:

```console
$ curl http://127.0.0.1:18080/healthz
{"status":"ok"}
```

<details>
<summary><strong>Use Docker instead</strong></summary>

The public image does not require a registry login:

```console
$ docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:latest
$ docker run --rm -p 127.0.0.1:18080:18080 \
    -v ~/.codex:/home/app/.codex \
    ghcr.io/hotchpotch/openai-api-server-via-codex:latest
```

The auth directory is mounted read-write so refreshed tokens can be saved. See
the [Docker guide](docs/docker.md) for Windows, permissions, Compose, and login
instructions.

</details>

### 3. Call it with the OpenAI SDK

Point the SDK at the local server:

```console
$ export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
$ export OPENAI_API_KEY=any-dummy-string-no-openai-api-key
```

> [!NOTE]
> This is **not a real OpenAI Platform API key**. Do not paste your real key
> here. With the server's default settings, `OPENAI_API_KEY` may be any non-empty
> dummy string: it only satisfies the OpenAI SDK's client-side requirement and
> is not checked by the server. Incoming `/v1/...` requests are authenticated
> only when the server is started with `--api-key`; in that case, set this value
> to the configured server key. Upstream Codex authentication always uses
> `~/.codex/auth.json`.

Create `example.py`:

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

Run it without permanently installing the SDK:

```console
$ uv run --with openai python example.py
```

<details>
<summary><strong>PowerShell environment variables</strong></summary>

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:18080/v1"
$env:OPENAI_API_KEY = "any-dummy-string-no-openai-api-key"
uv run --with openai python example.py
```

</details>

## Why use it?

This server is useful when you want to:

- connect an OpenAI-compatible application to your authorized Codex access;
- reuse code written for `openai-python`, LangChain, LiteLLM, or another client
  that supports a custom base URL;
- run a lightweight local proxy without a Python web-server runtime; or
- expose one consistent local API to tools that cannot call the Codex backend
  directly.

| Capability | What you get |
| --- | --- |
| Subscription-backed access | Use the Codex allowance included in your ChatGPT plan without separate Platform API token charges |
| OpenAI compatibility | Responses, Chat Completions, streaming, tools, structured output, images, and audio |
| Small Go runtime | Fast startup, a single executable, and a few MiB of idle memory in local measurements |
| Portable distribution | Wheels and archives for Linux, macOS, and Windows on x86_64 and ARM64 |
| Local-first defaults | Loopback binding, auth preflight, redacted logs, and optional incoming API-key protection |
| Multiple installation paths | `uvx`, standalone archives, Docker/GHCR, `go install`, or a local source build |

> [!WARNING]
> This is an unofficial compatibility server, not the OpenAI Platform API. It
> uses the Codex HTTP backend associated with your ChatGPT login, which may
> change without notice. It does not raise or bypass plan limits. Do not expose
> the server publicly, resell access, or use an account you are not authorized
> to use.

## Installation options

| Method | Requirement | Best for |
| --- | --- | --- |
| `uvx openai-api-server-via-codex` | `uv` and a Codex login | Most users |
| GitHub Release archive | A Codex login | A standalone executable without Python or `uv` |
| GHCR image | Docker and a Codex login | Containers and reproducible deployment |
| `go install ...@latest` | Go 1.23+ and a Codex login | Go users |
| Build from a checkout | Go 1.23+ and a Codex login | Development and customization |

<details>
<summary><strong>Install with uvx or uv tool</strong></summary>

Published wheels contain the Go executable for:

- Linux x86_64 and ARM64
- macOS Intel and Apple silicon
- Windows x86_64 and ARM64

Run without a permanent installation:

```console
$ uvx openai-api-server-via-codex
```

Or install it on your user tool path:

```console
$ uv tool install openai-api-server-via-codex
$ openai-api-server-via-codex --version
```

There is no Python server fallback and no generic source distribution. Build
the Go executable directly on an unsupported platform.

</details>

<details>
<summary><strong>Download a standalone release archive</strong></summary>

Each GitHub Release includes versioned archives for Linux, macOS, and Windows
on x86_64 and ARM64, together with `checksums.txt`. Unix archives use `.tar.gz`;
Windows archives use `.zip`.

For example, on an Apple silicon Mac:

```console
$ curl -LO https://github.com/hotchpotch/openai-api-server-via-codex/releases/download/v0.2.0/openai-api-server-via-codex_0.2.0_darwin_arm64.tar.gz
$ curl -LO https://github.com/hotchpotch/openai-api-server-via-codex/releases/download/v0.2.0/checksums.txt
$ grep 'darwin_arm64.tar.gz' checksums.txt | shasum -a 256 --check
$ tar -xzf openai-api-server-via-codex_0.2.0_darwin_arm64.tar.gz
$ ./openai-api-server-via-codex --version
```

The stable archive URLs and SHA-256 checksums are suitable for Homebrew
Formulae. Installing a historical version through Homebrew requires a tap to
retain a versioned Formula such as `openai-api-server-via-codex@0.2.0`.

</details>

<details>
<summary><strong>Run the published Docker image or build it locally</strong></summary>

Stable Linux x86_64 and ARM64 images are published at
`ghcr.io/hotchpotch/openai-api-server-via-codex`. `latest` tracks the newest
stable release. Exact tags such as `v0.2.0` provide reproducible deployments;
prereleases publish only their exact version tag.

The final Alpine image contains the Go server and CA certificates, but no
Python runtime or Go toolchain. On amd64, local measurements show about 7 MB of
compressed registry layers, a 21 MB local image, and roughly 4–7 MiB of idle
memory. Exact values depend on the release, architecture, and container runtime.

```console
$ docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:latest
$ docker run --rm -p 127.0.0.1:18080:18080 \
    -v ~/.codex:/home/app/.codex \
    ghcr.io/hotchpotch/openai-api-server-via-codex:latest
```

Or build from this checkout:

```console
$ docker compose run --rm --service-ports codex-login  # only if auth.json is missing
$ docker compose up --build -d
$ curl http://127.0.0.1:18080/healthz
```

The container runs as a non-root user. See the
[Docker guide](docs/docker.md) for complete usage and troubleshooting.

</details>

<details>
<summary><strong>Install with Go or build from a checkout</strong></summary>

Go can download and install the command from its public module path:

```console
$ go install github.com/hotchpotch/openai-api-server-via-codex/cmd/openai-api-server-via-codex@latest
$ "$(go env GOPATH)/bin/openai-api-server-via-codex" --version
```

The executable is installed under `GOBIN`, or under `$(go env GOPATH)/bin`
when `GOBIN` is unset.

To build from a checkout:

```console
$ go build -trimpath -o ./bin/openai-api-server-via-codex ./cmd/openai-api-server-via-codex
$ ./bin/openai-api-server-via-codex --version
$ ./bin/openai-api-server-via-codex serve
```

This path does not require Python or `uv`. See
[Building the Go binary from source](docs/build-from-source.md) for version
stamping, installation, Windows commands, static builds, and cross-compilation.

</details>

## Authentication and security

There are two separate authentication layers:

| Layer | Default | Purpose |
| --- | --- | --- |
| Server → Codex | Required | Uses the ChatGPT login in `~/.codex/auth.json` |
| Client → local server | Disabled | Optionally protects incoming `/v1/...` requests with `--api-key` |

### Codex authentication

`serve` and `start` validate the Codex auth file before binding the HTTP port.
Missing, invalid, expired, or unrefreshable credentials fail startup with a
redacted error, a stable reason code, and a suggested action.

The server notices external changes to `auth.json` without a restart. If Codex
returns `401 Unauthorized` before a streaming response begins, the server
clears its credential cache, reloads the file, and retries once. A second `401`
is returned without another retry.

Select another auth file when needed:

```console
$ openai-api-server-via-codex --auth-json /path/to/auth.json
$ OPENAI_VIA_CODEX_AUTH_JSON=/path/to/auth.json openai-api-server-via-codex
```

### Protecting the local API

Configure an incoming API key when other machines or untrusted processes can
reach the server:

```console
$ openai-api-server-via-codex \
    --host 0.0.0.0 \
    --api-key local-secret
```

Clients must then send `Authorization: Bearer local-secret`. `/healthz` remains
unauthenticated.

> [!CAUTION]
> Do not bind to `0.0.0.0` without an incoming API key unless the surrounding
> network provides equivalent access control. The default `127.0.0.1` binding
> is the safest choice for local use.

Incoming API keys, cookies, and `Authorization` headers are never forwarded to
Codex. Normal logs include authentication failure reason codes and upstream
`401` retry decisions, but never raw credentials, tokens, or upstream response
bodies. `--verbose` adds deeper redacted diagnostics.

## Troubleshooting

| Symptom or reason code | What to do |
| --- | --- |
| `auth_file_not_found` | Run `codex login`, or pass the correct path with `--auth-json` |
| `invalid_auth_json` or `unsupported_auth_mode` | Run `codex login` again and confirm the login uses ChatGPT mode |
| `expired_without_refresh_token` | Run `codex login` again to create refreshable credentials |
| `token_refresh_failed` | Check network access and retry `codex login` if the failure persists |
| `auth_file_write_failed` in Docker | Mount `/home/app/.codex` read-write and check host UID/GID permissions |
| Repeated upstream `401` | Check the auth reason logs, refresh the Codex login, and avoid multiple servers sharing rotating credentials |
| Address already in use | Stop the existing server or choose another port with `--port` |
| OpenAI SDK rejects an empty key | Set `OPENAI_API_KEY` to a non-secret placeholder such as `any-dummy-string-no-openai-api-key` |

> [!TIP]
> Start with `--verbose` when diagnosing configuration, request routing, or
> Codex stream behavior. Sensitive token-like values remain redacted.

For container-specific problems, see the [Docker guide](docs/docker.md).

## Usage examples

<details>
<summary><strong>Chat Completions</strong></summary>

```python
from openai import OpenAI

client = OpenAI()
chat = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "Hello"}],
    reasoning_effort="low",
)
print(chat.choices[0].message.content)
```

</details>

<details>
<summary><strong>Streaming Responses and Chat Completions</strong></summary>

Responses:

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

Chat Completions:

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

</details>

<details>
<summary><strong>Image input and image generation</strong></summary>

Image input:

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

Image generation:

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

Image generation returns `data[].b64_json`. URL results, image editing, and
streamed partial images are not implemented.

</details>

<details>
<summary><strong>Function tool calling</strong></summary>

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

## API compatibility

<details>
<summary><strong>Supported behavior</strong></summary>

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
reasoning content. Public storage compatibility is implemented in the server's
bounded in-memory stores.

Model listing is best-effort. A model can sometimes accept direct requests even
when it is absent from the upstream catalog returned by `/v1/models`.

</details>

<details>
<summary><strong>Implemented endpoints</strong></summary>

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

Unknown `/v1/...` requests use a best-effort fallback proxy. The server
forwards the method, path, query, safe OpenAI-style headers, and body with its
own Codex credentials. Upstream support determines whether an unknown endpoint
returns `2xx`, `400`, `403`, or `404`.

</details>

## Operations and configuration

<details>
<summary><strong>Configuration file and important settings</strong></summary>

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

| Setting | Default | Purpose |
| --- | --- | --- |
| `server.host` | `127.0.0.1` | HTTP bind address |
| `server.port` | `18080` | HTTP port |
| `server.default_model` | `gpt-5.6-luna` | Model used when a request omits one |
| `server.api_key` | unset | Protect incoming `/v1/...` requests |
| `server.max_stored_items` | `1000` | Bound local stores; `0` disables storage |
| `server.max_concurrent_requests` | `10` | Bound complete requests/streams; `0` disables the cap |
| `server.timeout` | `300.0` | Codex backend timeout in seconds |
| `server.verbose` | `false` | Enable redacted application diagnostics |
| `codex.auth_json` | `~/.codex/auth.json` | Codex OAuth file |
| `compat.drop_params` | `[]` | Top-level request fields removed before forwarding |

Examples:

```console
$ openai-api-server-via-codex --port 19090 --verbose
$ OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS=20 openai-api-server-via-codex
$ openai-api-server-via-codex --config ./config.toml
```

Use `drop_params` only for parameters known to be rejected by Codex:

```toml
[compat]
drop_params = ["temperature", "top_p"]
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
<summary><strong>Logging behavior</strong></summary>

Normal operation logs one completion line per API request with its method,
redacted path, status, response size, and duration. Routine `/healthz` probes
stay quiet.

Verbose logs additionally include request starts and redacted query strings,
resolved settings, endpoint summaries, and Codex stream/auth activity. Raw
credentials and token-like values are redacted.

</details>

## Development

<details>
<summary><strong>Build, test, and release documentation</strong></summary>

The HTTP server, backend integration, auth, configuration, stores, daemon, and
redaction logic are implemented in Go under `cmd/` and `internal/`. Python is
used only for the `uvx` launcher, the official `openai-python` consumer
contract, and release tooling.

Requirements:

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

Further documentation:

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
