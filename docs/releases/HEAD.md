# HEAD

- Reworked the README and active documentation around the Go-only server,
  documented the planned `0.2.0` breaking change, and added a complete guide
  for installing from the GitHub Go module or building standalone binaries from
  source.
- Kept the Go-only Alpine Docker runtime small by using its built-in
  healthcheck client, and made the Compose host bind configurable while
  retaining its loopback-only default.
- Added tag-driven GHCR publishing for Linux x86_64 and ARM64 images. Stable
  releases publish matching `vX.Y.Z` and `latest` tags, while prereleases leave
  `latest` unchanged. Existing release tags can be backfilled through a guarded
  manual Actions dispatch without republishing PyPI.
