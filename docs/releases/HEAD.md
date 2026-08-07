# HEAD

- Changed the compatibility default model to `gpt-5.6-luna` and added the
  GPT-5.6 model family to the fallback model catalog.
- Pinned Ruff 0.16.1 for local development and CI, and updated the codebase for
  its lint rules.
- Added a self-contained Docker setup (`Dockerfile`, `docker-compose.yml`,
  `.dockerignore`) that runs the server with only Docker installed, borrowing
  the host Codex login via a `~/.codex` bind mount. See `docs/docker.md`.
- Added a one-shot `codex-login` Compose helper service that bundles the
  official Codex CLI so `codex login` (including `--device-auth`) can run in a
  container, writing `auth.json` into the mounted `~/.codex` without Codex
  installed on the host. Documented the portable-`auth.json` copy path for
  remote and headless Docker hosts.
