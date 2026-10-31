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
- Log `chat.stream.tool_arguments_mismatch`, with sizes only, when streamed tool
  arguments differ from the final item in a way later chunks cannot correct.
