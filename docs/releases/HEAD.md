# HEAD

- Added a supervised background `start` runner that restarts the foreground
  server process after unexpected exits while preserving normal `stop` behavior.
- Hardened unexpected backend failures so non-streaming requests return
  OpenAI-compatible API errors and streaming responses emit terminal error
  events instead of leaking connection-level failures.
