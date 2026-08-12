# Release

This project releases to PyPI through GitHub Actions Trusted Publishing. The
PyPI project is already configured with a Trusted Publisher, and the repository
uses the `pypi` GitHub environment for release approval. Do not add PyPI API
tokens to GitHub secrets for the normal release path.

## Release Notes

Keep release notes under `docs/releases/`.

- `docs/releases/HEAD.md` is the draft changelog for the next release.
- `docs/releases/vX.Y.Z.md` is the finalized changelog for a released version.
- `python scripts/release-notes.py vX.Y.Z` prints the release note body. It uses
  `HEAD.md` first, then falls back to `docs/releases/vX.Y.Z.md`.

When changing user-visible behavior, add a concise entry to
`docs/releases/HEAD.md` in the same change. Keep entries focused on what changed
for users and operators.

## Local Verification

Run these checks before creating a release tag:

```console
$ uv sync --locked --dev
$ uv run tox
$ rm -rf build dist
$ uv build --wheel --no-sources
$ uv run python scripts/build-platform-wheels.py \
    --wheel "$(ls dist/*-py3-none-any.whl)" \
    --output-dir dist \
    --version "$(uv run python -c 'import openai_api_server_via_codex as p; print(p.__version__)')"
$ uv run twine check --strict dist/*
$ uv run --with "$(ls dist/*manylinux_2_17_x86_64.whl)" --no-project openai-api-server-via-codex --version
```

Inspect the distribution contents:

```console
$ python -m zipfile -l dist/openai_api_server_via_codex-X.Y.Z-py3-none-manylinux_2_17_x86_64.whl
```

The release should contain exactly six platform wheels: Linux x86_64/ARM64,
macOS Intel/Apple silicon, and Windows x86_64/ARM64. Each wheel must contain
`openai_api_server_via_codex/bin/` with its Go executable. The generic pure
wheel is only an intermediate input and must be removed before publishing; no
source distribution is published because it cannot contain a prebuilt binary
for the install target. No artifact may contain `.codex`, `auth.json`, `.env`,
`.venv`, `.tox`, caches, logs, generated reports, or nested `dist/` artifacts.

## Version Bump

Update all package version locations together:

```console
$ uv version X.Y.Z
```

Then update:

- `openai_api_server_via_codex/__init__.py`
- `tests/test_package_metadata.py`
- `README.md`, if it mentions the current release version

Move the completed changelog from `docs/releases/HEAD.md` to
`docs/releases/vX.Y.Z.md`, then reset `HEAD.md` to:

```markdown
# HEAD
```

## Tag And Publish

Commit the release changes, push `main`, create an annotated tag, and push the
tag:

```console
$ git status -sb
$ git add pyproject.toml uv.lock openai_api_server_via_codex/__init__.py tests/test_package_metadata.py README.md docs/releases
$ git commit -m "Release version X.Y.Z"
$ git push origin main
$ git tag -a vX.Y.Z -m "Release vX.Y.Z"
$ git push origin vX.Y.Z
```

The release workflow checks that the tag matches the package version, runs
`tox`, builds all six Go platform wheels, validates
metadata with `twine`, smoke tests the bundled Go console command, publishes to
PyPI only from the `pypi` environment, and creates a GitHub Release from
`docs/releases`.

If the `pypi` environment has required reviewers, approve the deployment in the
GitHub Actions run. The job uses OpenID Connect short-lived credentials through
Trusted Publishing.

## GitHub Release Text

Use the release notes script to preview the GitHub Release body:

```console
$ python scripts/release-notes.py vX.Y.Z
```

If `docs/releases/HEAD.md` has content beyond the heading, the script prints
that draft. Otherwise it prints `docs/releases/vX.Y.Z.md`. This matches the
release-note workflow used in `sqlite-vaporetto`.

## After Publishing

Verify installation from PyPI:

```console
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex --version
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex --help
```

Optionally start the published package:

```console
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex
```
