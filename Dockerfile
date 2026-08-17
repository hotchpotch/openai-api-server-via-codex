# syntax=docker/dockerfile:1

# Build the production server as a static Go executable.
FROM --platform=$BUILDPLATFORM golang:1.23-bookworm AS server-builder

WORKDIR /src

COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY cmd ./cmd
COPY internal ./internal

ARG TARGETOS=linux
ARG TARGETARCH
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/openai-api-server-via-codex ./cmd/openai-api-server-via-codex

# Login helper stage: bundles the official Codex CLI so `codex login` can run
# in a container without Codex installed on the host. Not part of the default
# build; used by the `codex-login` compose service (or --target login).
FROM node:22-slim AS login

ARG CODEX_VERSION=latest
# ca-certificates is required, not optional: the Codex CLI is a Rust binary with
# its own TLS stack, and node:22-slim ships no system trust store. Without it
# `codex login --device-auth` fails with "error sending request for url
# (https://auth.openai.com/api/accounts/deviceauth/usercode)" — which reads like
# an outage rather than a missing dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}"

COPY --chmod=755 docker/codex-login-entrypoint.sh /usr/local/bin/codex-login-entrypoint

ENV CODEX_HOME=/home/node/.codex
USER node
EXPOSE 1456
ENTRYPOINT ["codex-login-entrypoint"]
CMD ["login"]

# Console stage: the server plus the Codex CLI, so the operator console at /ui
# can run the device-code sign-in itself. Build with `--target runtime-ui`.
#
# Debian rather than Alpine on purpose: the Codex CLI ships a glibc-linked Rust
# binary, so it does not run on musl. The Go server is static (CGO_ENABLED=0) and
# runs on either, so the base is chosen by the CLI's constraint alone.
FROM node:22-slim AS runtime-ui

ARG VERSION=dev
ARG REVISION=unknown
ARG CODEX_VERSION=latest
LABEL org.opencontainers.image.title="openai-api-server-via-codex-ui" \
    org.opencontainers.image.description="OpenAI-compatible proxy backed by Codex credentials, with the operator console" \
    org.opencontainers.image.source="https://github.com/hotchpotch/openai-api-server-via-codex" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${REVISION}" \
    org.opencontainers.image.licenses="Apache-2.0"

# ca-certificates: see the note on the login stage — the CLI's TLS needs it.
# curl: only so HEALTHCHECK has something to call; node:*-slim has no wget.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

COPY --from=server-builder /out/openai-api-server-via-codex /usr/local/bin/openai-api-server-via-codex

ENV CODEX_HOME=/home/node/.codex \
    OPENAI_VIA_CODEX_HOST=0.0.0.0 \
    OPENAI_VIA_CODEX_PORT=18080 \
    OPENAI_VIA_CODEX_AUTH_JSON=/home/node/.codex/auth.json

USER node
EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS -m 4 -o /dev/null \
    "http://127.0.0.1:${OPENAI_VIA_CODEX_PORT}/healthz" || exit 1

ENTRYPOINT ["openai-api-server-via-codex"]
CMD ["serve"]

# Runtime stage: only the static Go server, CA roots, and Alpine's BusyBox tools.
# Keep this stage last so a plain `docker build` produces the server image.
FROM alpine:3.22 AS runtime

ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="openai-api-server-via-codex" \
    org.opencontainers.image.description="OpenAI-compatible proxy server backed by Codex HTTP credentials" \
    org.opencontainers.image.source="https://github.com/hotchpotch/openai-api-server-via-codex" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${REVISION}" \
    org.opencontainers.image.licenses="Apache-2.0"

RUN apk add --no-cache ca-certificates \
    && addgroup -g 1000 app \
    && adduser -D -u 1000 -G app app

COPY --from=server-builder /out/openai-api-server-via-codex /usr/local/bin/openai-api-server-via-codex

# The Codex login is expected as a bind mount at /home/app/.codex; the server
# reads auth.json from there and writes refreshed tokens back to it.
ENV CODEX_HOME=/home/app/.codex \
    OPENAI_VIA_CODEX_HOST=0.0.0.0 \
    OPENAI_VIA_CODEX_PORT=18080

USER app
EXPOSE 18080

# /healthz stays unauthenticated even when an API key is configured.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -T 4 -O /dev/null \
    "http://127.0.0.1:${OPENAI_VIA_CODEX_PORT}/healthz" || exit 1

ENTRYPOINT ["openai-api-server-via-codex"]
CMD ["serve"]
