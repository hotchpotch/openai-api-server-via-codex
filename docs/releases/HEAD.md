# HEAD

- Stream tool call arguments in Chat Completions when Codex sends them only on
  `response.output_item.done`. Codex does this for parallel tool calls, so
  streaming clients previously received every parallel call with empty
  `arguments`.
