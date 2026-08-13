# Homebrew packaging

This project is prepared for a source Formula in the official
`Homebrew/homebrew-core` repository. Homebrew's official process, rather than
this repository's release workflow, creates and hosts the bottles:

1. the Formula downloads an immutable `vX.Y.Z` source tag and verifies its
   SHA-256 digest;
2. the Formula builds the Go command with `go` as a build-only dependency;
3. BrewTestBot builds the Formula on the supported macOS and Linux runners;
4. after the Formula PR is accepted, Homebrew publishes those bottles to its
   package registry; and
5. `brew install openai-api-server-via-codex` automatically downloads a
   matching bottle when one is available.

Do not publish the standalone GitHub Release archives as though they were
Homebrew bottles. A bottle is a Homebrew keg produced and described by the
Homebrew bottle workflow, not merely an upstream platform binary archive.

## Generate a Formula from a tag

Only stable `vX.Y.Z` tags are eligible. The generator rejects prerelease tags,
downloads GitHub's immutable source archive, calculates its checksum, and
renders a Formula using Homebrew's `std_go_args` helper:

```console
$ uv run python scripts/render-homebrew-formula.py vX.Y.Z \
    --output openai-api-server-via-codex.rb
```

Stable GitHub Releases attach the generated
`openai-api-server-via-codex.rb` as a release asset. The checked-in
[`packaging/homebrew/openai-api-server-via-codex.rb`](../packaging/homebrew/openai-api-server-via-codex.rb)
tracks the latest stable Formula prepared in this repository.

## Validate for homebrew/core

Before proposing the Formula, confirm the current
[Homebrew package acceptance policy](https://docs.brew.sh/Package-Acceptance-Policy)
and [acceptable Formula requirements](https://docs.brew.sh/Acceptable-Formulae).
In particular, an official Formula must meet Homebrew's current notability and
maintenance requirements; having a tagged stable release is necessary but is
not sufficient by itself.

From an up-to-date `homebrew/core` checkout, copy the generated Formula to the
normal Formula path and run Homebrew's required source-build checks:

```console
$ brew tap --force homebrew/core
$ core="$(brew --repository homebrew/core)"
$ cp openai-api-server-via-codex.rb \
    "${core}/Formula/o/openai-api-server-via-codex.rb"
$ HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source \
    openai-api-server-via-codex
$ brew test openai-api-server-via-codex
$ brew audit --strict --new --online openai-api-server-via-codex
$ brew style --formula openai-api-server-via-codex
```

Do not add a `bottle do` block to the new-Formula PR. BrewTestBot generates and
merges that block after successful platform builds. Once the initial Formula
has been accepted, future stable tags can be proposed with
`brew bump-formula-pr`; prerelease tags must not replace the stable Formula.

The Homebrew project owns approval and official bottle publication. Keeping a
Formula or generated asset in this repository does not by itself make this an
official Homebrew package.
