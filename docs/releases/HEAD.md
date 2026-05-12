# HEAD

- Added `POST /v1/images/generations` support for non-streaming
  `client.images.generate(..., response_format="b64_json")` calls by
  translating requests into Codex Responses API hosted `image_generation` tool
  calls and returning `data[].b64_json` image bytes. The README now documents
  the compatibility scope, including base64-only responses, best-effort
  size/quality guidance, and the live image validation flow.
