from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from .backend import CodexBackend, CodexBackendError, OpenAICodexBackend
from .compat import (
    DEFAULT_MODEL,
    ResponseStore,
    chat_request_to_response_payload,
    ensure_response_defaults,
    extract_response_text,
    prepare_response_payload,
    response_to_chat_completion,
)


class OpenAICompatRequest(BaseModel):
    model: str | None = None
    stream: bool | None = False

    model_config = ConfigDict(extra="allow")


def create_app(
    *,
    backend: CodexBackend | None = None,
    default_model: str | None = None,
) -> FastAPI:
    app = FastAPI(title="OpenAI API Server via Codex")
    app.state.backend = backend or OpenAICodexBackend(
        base_url=os.environ.get(
            "OPENAI_VIA_CODEX_BACKEND_BASE_URL",
            "https://chatgpt.com/backend-api/codex",
        ),
        client_version=os.environ.get("OPENAI_VIA_CODEX_CLIENT_VERSION", "1.0.0"),
        timeout=float(os.environ.get("OPENAI_VIA_CODEX_TIMEOUT", "180")),
    )
    app.state.response_store = ResponseStore()
    app.state.default_model = default_model or os.environ.get(
        "OPENAI_VIA_CODEX_DEFAULT_MODEL", DEFAULT_MODEL
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        backend = _get_backend(request)
        model_ids = await backend.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "codex",
                }
                for model_id in model_ids
            ],
        }

    @app.post("/v1/responses", response_model=None)
    async def create_response(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse | StreamingResponse:
        compat_request = OpenAICompatRequest.model_validate(body)
        payload = compat_request.model_dump(exclude_none=True)
        prepared = prepare_response_payload(
            payload, default_model=request.app.state.default_model
        )
        store = _get_response_store(request)
        previous_response_id = prepared.get("previous_response_id")
        if previous_response_id:
            stored = store.get(str(previous_response_id))
            if stored is None:
                return _openai_error(
                    404,
                    f"Unknown previous_response_id: {previous_response_id}",
                    param="previous_response_id",
                )
            prepared["input"] = stored.context_items + prepared["input"]

        if payload.get("stream"):
            backend_payload = {
                key: value
                for key, value in prepared.items()
                if key != "previous_response_id"
            }
            return _sse_response(
                _responses_event_stream(
                    backend=_get_backend(request),
                    store=store,
                    prepared=prepared,
                    backend_payload=backend_payload,
                    previous_response_id=previous_response_id,
                )
            )

        backend_payload = {
            key: value
            for key, value in prepared.items()
            if key != "previous_response_id"
        }
        try:
            response = await _get_backend(request).create_response(backend_payload)
        except CodexBackendError as exc:
            return _openai_error(exc.status_code, str(exc), error_type="api_error")
        response = ensure_response_defaults(response, request_payload=prepared)
        if previous_response_id:
            response["previous_response_id"] = previous_response_id

        store.remember(
            str(response["id"]),
            effective_input=prepared["input"],
            response=response,
        )
        return JSONResponse(response)

    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse | StreamingResponse:
        compat_request = OpenAICompatRequest.model_validate(body)
        payload = compat_request.model_dump(exclude_none=True)
        response_payload = chat_request_to_response_payload(
            payload, default_model=request.app.state.default_model
        )
        if payload.get("stream"):
            return _sse_response(
                _chat_completion_event_stream(
                    backend=_get_backend(request),
                    response_payload=response_payload,
                    chat_payload=payload,
                )
            )

        try:
            response = await _get_backend(request).create_response(response_payload)
        except CodexBackendError as exc:
            return _openai_error(exc.status_code, str(exc), error_type="api_error")
        response = ensure_response_defaults(response, request_payload=response_payload)
        chat_completion = response_to_chat_completion(
            response, fallback_model=response_payload["model"]
        )
        return JSONResponse(chat_completion)

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "openai_api_server_via_codex.server:app",
        host=os.environ.get("OPENAI_VIA_CODEX_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENAI_VIA_CODEX_PORT", "8000")),
        reload=False,
    )


def _get_backend(request: Request) -> CodexBackend:
    return request.app.state.backend


def _get_response_store(request: Request) -> ResponseStore:
    return request.app.state.response_store


def _openai_error(
    status_code: int,
    message: str,
    *,
    param: str | None = None,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": None,
            }
        },
    )


@dataclass
class _ChatStreamState:
    chat_id: str
    created: int
    model: str
    role_sent: bool = False
    saw_text_delta: bool = False


def _sse_response(content: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _responses_event_stream(
    *,
    backend: CodexBackend,
    store: ResponseStore,
    prepared: dict[str, Any],
    backend_payload: dict[str, Any],
    previous_response_id: Any,
) -> AsyncIterator[bytes]:
    output_items: list[dict[str, Any]] = []
    final_response_stored = False
    sequence_number = 0

    try:
        async for raw_event in backend.stream_response(backend_payload):
            event = _normalize_stream_event(raw_event)
            sequence_number = int(event.get("sequence_number") or sequence_number)
            event_type = event.get("type")

            if event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    output_items.append(item)
            elif event_type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                response = event.get("response")
                if isinstance(response, dict):
                    if output_items and not response.get("output"):
                        response["output"] = output_items
                    response = ensure_response_defaults(response, request_payload=prepared)
                    if previous_response_id:
                        response["previous_response_id"] = previous_response_id
                    event["response"] = response
                    if not final_response_stored:
                        store.remember(
                            str(response["id"]),
                            effective_input=prepared["input"],
                            response=response,
                        )
                        final_response_stored = True

            yield _sse_data(event)
            sequence_number += 1
    except CodexBackendError as exc:
        yield _sse_data(
            {
                "type": "error",
                "sequence_number": sequence_number,
                "code": None,
                "message": str(exc),
                "param": None,
            }
        )

    yield b"data: [DONE]\n\n"


async def _chat_completion_event_stream(
    *,
    backend: CodexBackend,
    response_payload: dict[str, Any],
    chat_payload: dict[str, Any],
) -> AsyncIterator[bytes]:
    created = int(time.time())
    state = _ChatStreamState(
        chat_id=f"chatcmpl_{created}",
        created=created,
        model=str(response_payload.get("model") or DEFAULT_MODEL),
    )
    include_usage = bool(
        isinstance(chat_payload.get("stream_options"), dict)
        and chat_payload["stream_options"].get("include_usage")
    )
    emitted_tool_arguments: dict[int, str] = {}

    try:
        async for raw_event in backend.stream_response(response_payload):
            event = _normalize_stream_event(raw_event)
            event_type = event.get("type")

            if event_type == "response.created":
                response = event.get("response")
                if isinstance(response, dict):
                    _update_chat_stream_state(state, response)
                async for chunk in _ensure_chat_role_chunk(state):
                    yield chunk
                continue

            if event_type == "response.output_text.delta":
                async for chunk in _ensure_chat_role_chunk(state):
                    yield chunk
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    state.saw_text_delta = True
                    yield _sse_data(_chat_chunk(state, {"content": delta}))
                continue

            if event_type == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    async for chunk in _ensure_chat_role_chunk(state):
                        yield chunk
                    output_index = int(event.get("output_index") or 0)
                    emitted_tool_arguments.setdefault(output_index, "")
                    yield _sse_data(
                        _chat_chunk(
                            state,
                            {
                                "tool_calls": [
                                    _chat_tool_call_delta(
                                        item,
                                        index=output_index,
                                        arguments="",
                                        include_identity=True,
                                    )
                                ]
                            },
                        )
                    )
                continue

            if event_type == "response.function_call_arguments.delta":
                async for chunk in _ensure_chat_role_chunk(state):
                    yield chunk
                output_index = int(event.get("output_index") or 0)
                delta = str(event.get("delta") or "")
                emitted_tool_arguments[output_index] = (
                    emitted_tool_arguments.get(output_index, "") + delta
                )
                yield _sse_data(
                    _chat_chunk(
                        state,
                        {
                            "tool_calls": [
                                {
                                    "index": output_index,
                                    "function": {"arguments": delta},
                                }
                            ]
                        },
                    )
                )
                continue

            if event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    async for chunk in _chat_output_item_done_chunks(
                        state, item, event, emitted_tool_arguments
                    ):
                        yield chunk
                continue

            if event_type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                response = event.get("response")
                if isinstance(response, dict):
                    response = ensure_response_defaults(
                        response, request_payload=response_payload
                    )
                    _update_chat_stream_state(state, response)
                    async for chunk in _ensure_chat_role_chunk(state):
                        yield chunk
                    completion = response_to_chat_completion(
                        response, fallback_model=state.model
                    )
                    finish_reason = completion["choices"][0]["finish_reason"]
                    yield _sse_data(
                        _chat_chunk(state, {}, finish_reason=finish_reason)
                    )
                    if include_usage:
                        yield _sse_data(
                            _chat_chunk(state, {}, choices=[], usage=completion["usage"])
                        )
                continue
    except CodexBackendError as exc:
        yield _sse_data(
            {
                "error": {
                    "message": str(exc),
                    "type": "api_error",
                    "param": None,
                    "code": None,
                }
            }
        )

    yield b"data: [DONE]\n\n"


async def _ensure_chat_role_chunk(state: _ChatStreamState) -> AsyncIterator[bytes]:
    if not state.role_sent:
        state.role_sent = True
        yield _sse_data(_chat_chunk(state, {"role": "assistant"}))


async def _chat_output_item_done_chunks(
    state: _ChatStreamState,
    item: dict[str, Any],
    event: dict[str, Any],
    emitted_tool_arguments: dict[int, str],
) -> AsyncIterator[bytes]:
    if item.get("type") == "function_call":
        output_index = int(event.get("output_index") or 0)
        if output_index not in emitted_tool_arguments:
            async for chunk in _ensure_chat_role_chunk(state):
                yield chunk
            emitted_tool_arguments[output_index] = str(item.get("arguments") or "")
            yield _sse_data(
                _chat_chunk(
                    state,
                    {
                        "tool_calls": [
                            _chat_tool_call_delta(
                                item,
                                index=output_index,
                                arguments=emitted_tool_arguments[output_index],
                                include_identity=True,
                            )
                        ]
                    },
                )
            )
        return

    if item.get("type") == "message" and not state.saw_text_delta:
        text = extract_response_text({"output": [item]})
        if text:
            async for chunk in _ensure_chat_role_chunk(state):
                yield chunk
            state.saw_text_delta = True
            yield _sse_data(_chat_chunk(state, {"content": text}))


def _update_chat_stream_state(
    state: _ChatStreamState, response: dict[str, Any]
) -> None:
    response_id = str(response.get("id") or state.chat_id)
    state.chat_id = (
        response_id.replace("resp_", "chatcmpl_", 1)
        if response_id.startswith("resp_")
        else f"chatcmpl_{response_id}"
    )
    state.created = int(response.get("created_at") or state.created)
    state.model = str(response.get("model") or state.model)


def _chat_chunk(
    state: _ChatStreamState,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    choices: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": state.chat_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": choices
        if choices is not None
        else [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _chat_tool_call_delta(
    item: dict[str, Any],
    *,
    index: int,
    arguments: str,
    include_identity: bool,
) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if include_identity:
        function["name"] = item.get("name") or ""

    tool_call: dict[str, Any] = {
        "index": index,
        "function": function,
    }
    if include_identity:
        tool_call["id"] = item.get("call_id") or item.get("id") or f"call_{index}"
        tool_call["type"] = "function"
    return tool_call

def _normalize_stream_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    if hasattr(event, "model_dump"):
        dumped = event.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {"type": "error", "sequence_number": 0, "message": repr(event)}


def _sse_data(data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()
