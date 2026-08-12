# HEAD

- Reworked the README and active documentation around the Go-only server,
  documented the planned `0.2.0` breaking change, and added a complete guide
  for installing from the GitHub Go module or building standalone binaries from
  source.
- Kept the Go-only Alpine Docker runtime small by using its built-in
  healthcheck client, and made the Compose host bind configurable while
  retaining its loopback-only default.
