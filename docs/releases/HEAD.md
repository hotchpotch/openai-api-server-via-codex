# HEAD

- Stream tool call arguments in Chat Completions when Codex sends them only on
  `response.output_item.done`. Codex does this for parallel tool calls, so
  streaming clients previously received every parallel call with empty
  `arguments`.
- Number streamed Chat Completions `tool_calls` contiguously from 0. Calls that
  followed text or reasoning previously started at a higher index, which made
  the `openai-python` `chat.completions.stream()` helper raise `IndexError`.
- Keep legacy `functions` streams to a single call. Requests with `functions` or
  `function_call` now default `parallel_tool_calls` to `false`, and streaming no
  longer merges further calls into one invalid `function_call`.
- Return the official `function_call` finish reason for legacy `functions`
  responses instead of `tool_calls`.
- Keep assistant text that precedes tool calls in non-streaming and stored Chat
  Completions. When replaying history, send that text before its calls so each
  call stays adjacent to its result.
- Give replayed legacy `function_call` messages unique call ids, so calling the
  same function more than once in a conversation no longer sends duplicate ids,
  and stop adding an empty assistant message next to them.
- Treat requests with `tools` as tool requests even when they also send
  `functions` or `function_call`, and ignore empty `functions`, so their calls
  keep ids instead of becoming a legacy `function_call`.
- Return an error when Codex reports a failure during a Chat Completion, instead
  of a successful completion that could carry truncated text or tool arguments.
  Streams end with an error chunk and no `finish_reason`; non-streaming requests
  and image generation return an HTTP error with the upstream `code`. Failed
  completions are not stored. `/v1/responses` still returns the failed response
  object, and now returns an HTTP error instead of an empty completed response
  when Codex sends only an `error` event.
- Log `chat.stream.tool_arguments_mismatch`, with sizes only, when streamed tool
  arguments differ from the final item in a way later chunks cannot correct.
