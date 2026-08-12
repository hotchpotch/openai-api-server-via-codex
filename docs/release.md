# Release

This project releases platform wheels to PyPI and a multi-platform Linux image
to GitHub Container Registry through GitHub Actions. PyPI uses Trusted
Publishing and the repository's `pypi` environment; do not add PyPI API tokens
to GitHub secrets for the normal release path. GHCR uses the workflow's
short-lived `GITHUB_TOKEN` with `packages: write`.

Stable `0.2.0` is the documented breaking-change release for the Go-only server
runtime. Release notes must state that the Python HTTP server and fallback no
longer exist, while the dependency-free Python package remains as the `uvx`
launcher for bundled platform executables.

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
The platform-wheel builder rejects generic wheels, source archives, unexpected
wheel tags, stale package files, and missing or empty Go binaries before the
release workflow can upload the directory.

## Version Bump

Update all package version locations together:

```console
$ uv version X.Y.Z
```

Then update:

- `openai_api_server_via_codex/__init__.py`
- `tests/test_package_metadata.py`
- `README.md`, if it mentions the current release version

For a beta, use a PEP 440 version such as `0.1.6b1` and a matching tag such as
`v0.1.6b1`. The release workflow publishes the wheels normally to PyPI and
marks the corresponding GitHub Release as a pre-release. Publishing a beta does
not imply that the same implementation will be promoted to a stable release.

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
PyPI only from the `pypi` environment, publishes the runtime container to GHCR,
and creates a GitHub Release from `docs/releases` only after both package
publishes succeed.

The container job publishes one multi-platform manifest for `linux/amd64` and
`linux/arm64` under both the exact Git tag (`vX.Y.Z`) and, for stable versions,
`latest`. A prerelease such as `v0.2.0b1` receives only its exact tag, so it
cannot replace the stable image. The Dockerfile's OCI source label links the
package to this repository before the first push.

To backfill GHCR for an existing release without moving its Git tag or
republishing PyPI, dispatch the release workflow from a branch containing the
current workflow and provide the existing tag:

```console
$ gh workflow run release.yml \
    --ref <branch-with-current-workflow> \
    -f container_release_tag=vX.Y.Z
```

The job checks that the branch package version and tagged source version match,
then checks out the existing tag for the container build. A manual backfill
publishes only the exact `vX.Y.Z` tag and never moves `latest`.

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

The OCI source label links the package to this public repository before its
first push, so the workflow-created GHCR package inherits public visibility and
supports anonymous pulls. If the namespace was already occupied by a separately
configured package, confirm its Package settings before publishing; changing a
private package to public cannot be reversed. Verify anonymous access, both
architectures, and the versioned aliases:

```console
$ docker buildx imagetools inspect ghcr.io/hotchpotch/openai-api-server-via-codex:vX.Y.Z
$ docker pull ghcr.io/hotchpotch/openai-api-server-via-codex:vX.Y.Z
$ docker run --rm ghcr.io/hotchpotch/openai-api-server-via-codex:vX.Y.Z --version
$ docker buildx imagetools inspect ghcr.io/hotchpotch/openai-api-server-via-codex:latest
```

Run the inspect and pull checks from a Docker configuration without GHCR
credentials when explicitly validating anonymous access.

For a prerelease, confirm that the exact tag exists and that the digest behind
`latest` has not changed.

Optionally start the published package:

```console
$ uvx --refresh-package openai-api-server-via-codex openai-api-server-via-codex
```
