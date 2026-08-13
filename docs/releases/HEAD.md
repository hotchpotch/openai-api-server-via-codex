# HEAD

- Aligned the active installation examples and quick-start artwork with the
  stable `0.2.0` release, clarified container size measurements, and documented
  safe handling of refreshable Docker credentials.
- Added standalone Go archives for all six supported OS/architecture targets,
  reproducible SHA-256 checksums suitable for Homebrew Formulae, and the PyPI
  platform wheels to each GitHub Release.
- Classified Codex authentication preflight failures with actionable reason
  codes, and added non-verbose logs for local authentication failures, upstream
  `401` reload/retry outcomes, and rejected incoming API keys without exposing
  credentials.
