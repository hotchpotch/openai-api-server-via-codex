# HEAD

- Added a self-contained Docker setup (`Dockerfile`, `docker-compose.yml`,
  `.dockerignore`) that runs the server with only Docker installed, borrowing
  the host Codex login via a `~/.codex` bind mount. See `docs/docker.md`.
- Added a one-shot `codex-login` Compose helper service that bundles the
  official Codex CLI so `codex login` (including `--device-auth`) can run in a
  container, writing `auth.json` into the mounted `~/.codex` without Codex
  installed on the host. Documented the portable-`auth.json` copy path for
  remote and headless Docker hosts.
