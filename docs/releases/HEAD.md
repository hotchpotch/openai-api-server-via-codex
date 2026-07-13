# HEAD

- Add GPT-5.6 Sol, Terra, and Luna to the Codex HTTP fallback model catalog.
- Default Codex backend `parallel_tool_calls` to `false` when requests omit it, while preserving explicit request values.
- Send Codex requests with Codex CLI client identity headers required by Luna-gated backends.
