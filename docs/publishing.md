# Publishing

This project is intended to be installable as:

```console
$ uvx openai-api-server-via-codex --help
```

The current Trusted Publishing release target is `0.0.2`.

## Secure Publishing Model

Prefer GitHub Actions Trusted Publishing for real PyPI releases. Trusted
Publishing uses OpenID Connect and short-lived credentials, so no long-lived
PyPI token needs to be stored in GitHub secrets.

Local publishing with a PyPI API token is supported only as a fallback. If you
use it, use a freshly created token, keep it out of shell history, do not write
it to `.pypirc`, and revoke it after use.

## Local Release Check

Run this before creating a tag:

```console
$ uv sync --locked --dev
$ uv run tox
$ rm -rf dist
$ uv build --no-sources
$ uv run twine check --strict dist/*
$ uv run --with "$(ls dist/*.whl)" --no-project openai-api-server-via-codex --help
```

Inspect the distribution contents before upload:

```console
$ tar -tzf dist/openai_api_server_via_codex-0.0.2.tar.gz
$ python -m zipfile -l dist/openai_api_server_via_codex-0.0.2-py3-none-any.whl
```

The package should contain the `openai_api_server_via_codex` package,
`README.md`, `LICENSE`, and metadata only. It must not contain `.codex`,
`auth.json`, `.env`, `.venv`, `.tox`, caches, logs, or generated reports.

## Recommended PyPI Release

Use PyPI Trusted Publishing so the release workflow can publish without a
long-lived local upload token.

Configure PyPI:

- Publisher: GitHub Actions
- PyPI project name: `openai-api-server-via-codex`
- Owner: `hotchpotch`
- Repository name: `openai-api-server-via-codex`
- Workflow name: `release.yml`
- Environment name: `pypi`

Configure GitHub:

- Create the repository environment `pypi`.
- Add required reviewers for the environment before publishing.
- Do not add `PYPI_TOKEN` or any PyPI password secret.

Create and push the release tag:

```console
$ git status -sb
$ git tag -a v0.0.2 -m "Release v0.0.2"
$ git push origin main
$ git push origin v0.0.2
```

The release workflow checks that the git tag matches the version in
`pyproject.toml` and `openai_api_server_via_codex/__init__.py`, runs `tox`,
builds with `uv build --no-sources`, validates metadata with `twine`, smoke
tests the packaged console command, and publishes only from the `pypi`
environment.

## Fallback Local Upload

Prefer TestPyPI first:

```console
$ read -rs UV_PUBLISH_TOKEN
$ export UV_PUBLISH_TOKEN
$ uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --check-url https://test.pypi.org/simple/
$ unset UV_PUBLISH_TOKEN
```

If a real local PyPI upload is unavoidable:

```console
$ read -rs UV_PUBLISH_TOKEN
$ export UV_PUBLISH_TOKEN
$ uv publish
$ unset UV_PUBLISH_TOKEN
```

Do not paste tokens directly into commands that are saved in shell history. Do
not commit `.pypirc`, auth files, copied tokens, or built `dist/` artifacts.

## After Publishing

Verify installation from PyPI in a clean environment:

```console
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex --help
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex --version
```

Start the server:

```console
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex
```
