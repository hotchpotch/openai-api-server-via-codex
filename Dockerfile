# syntax=docker/dockerfile:1

# Build the production server as a static Go executable.
FROM golang:1.23-bookworm AS server-builder

WORKDIR /src

COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY cmd ./cmd
COPY internal ./internal

ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/openai-api-server-via-codex ./cmd/openai-api-server-via-codex

# Login helper stage: bundles the official Codex CLI so `codex login` can run
# in a container without Codex installed on the host. Not part of the default
# build; used by the `codex-login` compose service (or --target login).
FROM node:22-slim AS login

ARG CODEX_VERSION=latest
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}"

COPY --chmod=755 docker/codex-login-entrypoint.sh /usr/local/bin/codex-login-entrypoint

ENV CODEX_HOME=/home/node/.codex
USER node
EXPOSE 1456
ENTRYPOINT ["codex-login-entrypoint"]
CMD ["login"]

# Runtime stage: only the static Go server, CA roots, and Alpine's BusyBox tools.
# Keep this stage last so a plain `docker build` produces the server image.
FROM alpine:3.22 AS runtime

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
