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
- Reorganized the README around a complete quick start, clearer authentication
  guidance and troubleshooting, GitHub callouts, and collapsible installation,
  API, operations, and development references. Clarified that the SDK placeholder
  is not an OpenAI Platform API key and highlighted subscription-backed usage
  without separate Platform API token charges within included Codex limits.
  Made the opening flow immediately show the value proposition, environment-only
  client migration, unchanged SDK code, the `v0.2.0` break, and measured Go
  memory improvements.
