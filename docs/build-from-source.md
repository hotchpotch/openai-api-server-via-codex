# Build the Go server from source

The HTTP server is implemented entirely in Go. A source build produces one
executable and does not require Python, `uv`, FastAPI, or Uvicorn at build time
or runtime.

## Requirements

- Go 1.23 or newer
- Git, when cloning the repository or deriving a version string
- A Codex ChatGPT login for running the completed server

The default authentication file is `~/.codex/auth.json`. Never copy that file
into a binary, container image, source archive, or repository.

## Download a release binary

GitHub Releases provide standalone executables for Linux, macOS, and Windows on
x86_64 and ARM64. Linux and macOS use `.tar.gz`; Windows uses `.zip`. Each
archive contains the executable, `LICENSE`, and `README.md`, and
`checksums.txt` contains the SHA-256 digest for every archive.

Archive names are stable and include the release version, OS, and architecture:

```text
openai-api-server-via-codex_VERSION_GOOS_GOARCH.tar.gz
openai-api-server-via-codex_VERSION_windows_GOARCH.zip
```

The release workflow extracts these executables from the same platform wheels
published to PyPI, so the standalone archive and matching wheel contain the
same Go binary. Other package managers may consume the stable naming and
checksums. The official Homebrew path deliberately builds from tagged source;
Homebrew's CI then packages that build as a bottle. See the
[Homebrew packaging guide](homebrew.md).

## Install directly from GitHub

Go can download the module, build the command, and install it without cloning
the repository:

```console
$ go install github.com/hotchpotch/openai-api-server-via-codex/cmd/openai-api-server-via-codex@latest
$ "$(go env GOPATH)/bin/openai-api-server-via-codex" --version
```

`go install ...@latest` works outside a checkout and does not modify the
current project's `go.mod`. It writes the executable to `GOBIN` when that
environment variable is configured, otherwise to the `bin` directory under
`go env GOPATH` (normally `~/go/bin`). Add that directory to `PATH` if you want
to invoke `openai-api-server-via-codex` without an absolute path.

## Build for the current machine

```console
$ git clone https://github.com/hotchpotch/openai-api-server-via-codex.git
$ cd openai-api-server-via-codex
$ go mod download
$ mkdir -p bin
$ go build -trimpath \
    -o ./bin/openai-api-server-via-codex \
    ./cmd/openai-api-server-via-codex
$ ./bin/openai-api-server-via-codex --version
```

An unstamped development build reports `dev`. To embed a Git-derived version:

```console
$ version="$(git describe --tags --always --dirty)"
$ go build -trimpath \
    -ldflags="-s -w -X main.version=${version}" \
    -o ./bin/openai-api-server-via-codex \
    ./cmd/openai-api-server-via-codex
```

`-s -w` removes symbol and DWARF tables to reduce the executable size. Omit
those flags when producing a binary for debugging.

## Windows PowerShell

```powershell
git clone https://github.com/hotchpotch/openai-api-server-via-codex.git
Set-Location openai-api-server-via-codex
go mod download
New-Item -ItemType Directory -Force bin | Out-Null
go build -trimpath -o .\bin\openai-api-server-via-codex.exe .\cmd\openai-api-server-via-codex
.\bin\openai-api-server-via-codex.exe --version
```

For a stamped Windows build:

```powershell
$Version = git describe --tags --always --dirty
go build -trimpath `
  -ldflags "-s -w -X main.version=$Version" `
  -o .\bin\openai-api-server-via-codex.exe `
  .\cmd\openai-api-server-via-codex
```

## Run the binary

Foreground operation is the recommended starting point:

```console
$ ./bin/openai-api-server-via-codex serve
```

The default listener is `http://127.0.0.1:18080`. Confirm it from another
terminal:

```console
$ curl http://127.0.0.1:18080/healthz
$ curl http://127.0.0.1:18080/v1/models
```

Select another auth file or configuration when needed:

```console
$ ./bin/openai-api-server-via-codex serve \
    --auth-json /path/to/auth.json \
    --config /path/to/config.toml
```

Generate a configuration template directly from the binary:

```console
$ ./bin/openai-api-server-via-codex config-generate --stdout
```

## Install the executable

On Linux or macOS, copy it to a directory on `PATH`:

```console
$ install -d "$HOME/.local/bin"
$ install -m 0755 ./bin/openai-api-server-via-codex "$HOME/.local/bin/"
$ openai-api-server-via-codex --version
```

On Windows, copy `openai-api-server-via-codex.exe` to a directory included in
the user `PATH`.

## Static and cross-platform builds

The server does not require CGO. Setting `CGO_ENABLED=0` produces a static Go
executable that is suitable for minimal Linux containers and straightforward
cross-compilation.

The following examples use POSIX shell environment syntax:

```console
$ mkdir -p dist

$ CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -o dist/openai-api-server-via-codex-linux-amd64 \
    ./cmd/openai-api-server-via-codex

$ CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
    go build -trimpath -o dist/openai-api-server-via-codex-linux-arm64 \
    ./cmd/openai-api-server-via-codex

$ CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 \
    go build -trimpath -o dist/openai-api-server-via-codex-darwin-amd64 \
    ./cmd/openai-api-server-via-codex

$ CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 \
    go build -trimpath -o dist/openai-api-server-via-codex-darwin-arm64 \
    ./cmd/openai-api-server-via-codex

$ CGO_ENABLED=0 GOOS=windows GOARCH=amd64 \
    go build -trimpath -o dist/openai-api-server-via-codex-windows-amd64.exe \
    ./cmd/openai-api-server-via-codex

$ CGO_ENABLED=0 GOOS=windows GOARCH=arm64 \
    go build -trimpath -o dist/openai-api-server-via-codex-windows-arm64.exe \
    ./cmd/openai-api-server-via-codex
```

The repository's PyPI release process builds the same six OS/architecture
combinations and embeds each executable in its matching platform wheel.
Cross-compilation proves that the artifact can be produced; execute or smoke
test it on the target OS and architecture before distributing it independently.

## Verify a source build

Run deterministic Go validation before distributing a binary:

```console
$ go test ./...
$ go vet ./...
$ go test -race ./...  # supported on native race-detector platforms
```

Inspect embedded Go module and build information:

```console
$ go version -m ./bin/openai-api-server-via-codex
```

For a full repository check, including `openai-python` consumer compatibility
and wheel tooling, install `uv` and run:

```console
$ uv run tox
```

## Build with Docker instead

Docker also builds the Go executable from the checked-out source. No external
binary is downloaded:

```console
$ docker compose up --build -d
```

The builder stage uses the Go toolchain; the final Alpine image contains only
the server executable, CA certificates, and base BusyBox utilities. See the
[Docker guide](docker.md) for authentication and runtime configuration.
