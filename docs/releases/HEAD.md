# HEAD

- Added a fallback `/v1/{path}` proxy for endpoints that are not implemented
  locally, allowing Codex HTTP endpoints such as tokenizer-style APIs or newer
  OpenAI paths to be forwarded without per-endpoint compatibility shims.
