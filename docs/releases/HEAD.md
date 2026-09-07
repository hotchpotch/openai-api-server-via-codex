# HEAD

Planned for **0.2.1**. These changes are unreleased; the version bump and
release tag will be prepared separately after the implementation PR is merged.

- Add a Go Responses WebSocket relay with Codex authentication, upstream
  conversation state, async function/custom tools, and mid-turn steering.
- Preserve async tool attributes, custom calls, reasoning, and assistant phases
  in HTTP Responses continuation history.
- Reject unsupported background generation explicitly; keep upstream named-stream
  errors visible instead of emulating background jobs or multiplexing.
- Add official OpenAI SDK WebSocket contracts and opt-in GPT-6 Astra live E2E
  coverage, including pending tools, steering, and upstream limitations.
- Reject saturated backend request/connection capacity with HTTP 503 and
  `Retry-After`, send upstream WebSocket keepalives, preserve valid close codes,
  and send a going-away close on shutdown. Ignore malformed non-object response
  output when reconstructing HTTP conversation history.
- Apply the stop timeout to HTTP and WebSocket shutdown together, closing the
  listener immediately and aborting unresponsive relays at the shared deadline.

- Document model support: GPT-6 Astra supports async tools and steering; live
  probes of GPT-5.6 Luna and Terra accepted ordinary WebSocket tool calls but
  rejected async tools and steering.
