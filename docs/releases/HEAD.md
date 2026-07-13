# HEAD

- Add Codex Responses Lite request handling for GPT-5.6 Sol, Terra, and Luna models.
- Default Codex backend `parallel_tool_calls` to `false` when requests omit it, while preserving explicit request values.
- Send Codex requests with Codex CLI client identity headers required by Luna-gated backends.
