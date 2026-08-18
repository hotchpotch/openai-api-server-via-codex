# HEAD

- Added `POST /v1/images/edits` support, compatible with
  `client.images.edit(...)`. Multipart uploads are inlined as data URLs and sent
  to Codex as the hosted `image_generation` tool with `action="edit"`, including
  multiple reference images and an optional `mask` forwarded as
  `input_image_mask`.
- Added streamed image generation and editing. `stream=true` with
  `partial_images` now translates the Codex
  `response.image_generation_call.partial_image` events into the public
  `image_generation.partial_image`/`image_generation.completed` and
  `image_edit.partial_image`/`image_edit.completed` events. Streaming requires
  `n=1`, and `partial_images` accepts `0`-`3` as a hint; Codex decides how many
  previews it actually sends, so only the completed event is guaranteed.
- Added `input_fidelity` passthrough for image generations and edits.
