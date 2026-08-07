# Run with Docker

The repository ships a self-contained Docker setup. The only host requirement
is Docker (with Compose) — no local Python, `uv`, or even the Codex CLI is
needed. The server does need a Codex ChatGPT login at `~/.codex/auth.json`;
see [Getting a Codex login](#getting-a-codex-login) for the ways to create
one, including a containerized `codex login`.

## Quick start

Make sure a Codex login exists at `~/.codex/auth.json`. If the Codex CLI is
installed on this machine, `codex login` does it; otherwise use the
containerized login below.

Then build and start the server:

```console
$ docker compose up --build -d
```

The server listens on `http://127.0.0.1:18080`. Verify it:

```console
$ curl http://127.0.0.1:18080/healthz
$ curl http://127.0.0.1:18080/v1/models
```

Use it with any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="unused")
response = client.responses.create(model="gpt-5.5", input="Hello!")
print(response.output_text)
```

Follow logs and stop the server:

```console
$ docker compose logs -f
$ docker compose down
```

## Getting a Codex login

The server borrows a Codex ChatGPT login from `auth.json`. Any one of these
paths produces it:

### Log in without installing Codex (containerized login)

The Compose file includes a one-shot `codex-login` helper service that bundles
the official Codex CLI. It is profile-gated, so `docker compose up` never
starts it. Run it on demand:

```console
$ docker compose run --rm --service-ports codex-login
```

Open the printed `https://auth.openai.com/...` URL in a browser **on the same
machine**. The browser's callback to `http://localhost:1455` is forwarded into
the container, and Codex writes `auth.json` into the mounted `~/.codex`
directory. After that, `docker compose up -d` works as usual.

If your ChatGPT workspace has device-code authentication enabled, you can skip
the port forwarding entirely and approve the login from any device:

```console
$ docker compose run --rm codex-login login --device-auth
```

### Copy an existing login from another machine

`auth.json` is not tied to a machine. If you have Codex logged in anywhere
else (laptop, another server), copy the file over and the server takes care of
keeping it fresh from then on:

```console
$ scp other-machine:~/.codex/auth.json ~/.codex/auth.json
```

This is the easiest path for remote or headless Docker hosts, where the
browser-callback flow cannot reach the container. The alternatives there are
device-code auth (above) or an SSH tunnel
(`ssh -L 1455:localhost:1455 user@host`) while running the containerized
login on the remote host.

### Log in with the Codex CLI on the host

If Codex is installed on the host, plain `codex login` writes
`~/.codex/auth.json` where the Compose mount picks it up.

## How authentication works

The container does not store credentials. `docker-compose.yml` bind-mounts the
host's `~/.codex` directory (or `$CODEX_HOME` when set) into the container,
where the server reads `auth.json`. When the access token expires, the server
refreshes it and writes the new tokens back through the mount, so the mount
must remain read-write. Once a valid login exists, no re-login is needed as
long as the refresh token stays valid.

The `serve` command validates the Codex auth file before binding the port. If
the file is missing or invalid, the container exits with a redacted error —
check `docker compose logs` and create a login using one of the paths above.

## Configuration

The image sets `OPENAI_VIA_CODEX_HOST=0.0.0.0` so the server is reachable
through Docker's port mapping; the Compose file publishes it on loopback only,
matching the non-Docker default. All other settings keep their defaults and can
be changed with `OPENAI_VIA_CODEX_*` environment variables in the `environment`
block of `docker-compose.yml`, for example:

- `OPENAI_VIA_CODEX_API_KEY` — require `Authorization: Bearer <key>` for
  `/v1/...` routes (`/healthz` stays open, so the container healthcheck keeps
  working).
- `OPENAI_VIA_CODEX_VERBOSE=1` — debug-level logs.
- `OPENAI_VIA_CODEX_DEFAULT_MODEL` — default model when a request omits one.
- `OPENAI_VIA_CODEX_TIMEOUT`, `OPENAI_VIA_CODEX_MAX_STORED_ITEMS`,
  `OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS` — backend timeout and bounds.

Alternatively, mount a `config.toml` and point the server at it:

```yaml
    volumes:
      - ${CODEX_HOME:-$HOME/.codex}:/home/app/.codex
      - ./config.toml:/home/app/.config/openai-api-server-via-codex/config.toml:ro
```

Setting precedence is unchanged: CLI flag, environment variable, config file,
default.

## Plain `docker` (without Compose)

```console
$ docker build -t openai-api-server-via-codex:local .
$ docker run --rm -p 127.0.0.1:18080:18080 \
    -v ~/.codex:/home/app/.codex \
    openai-api-server-via-codex:local
```

## Notes

- Both the server and the `codex-login` helper run as non-root users with
  UID/GID 1000. On Linux hosts where your user is not 1000:1000, set
  `user: "<uid>:<gid>"` in `docker-compose.yml` (or `--user` for `docker run`)
  so the containers can read and update the mounted Codex login. Docker
  Desktop on macOS and Windows handles this automatically.
- The daemon subcommands (`start`, `stop`, `status`) are for host installs;
  in Docker the container itself is the daemon, so the image runs `serve` in
  the foreground and Compose manages restarts.
- The container healthcheck polls `/healthz`, so `docker ps` shows the
  service as `healthy` once the server is up.
