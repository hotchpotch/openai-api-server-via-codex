from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .backend import CodexBackend, CodexBackendError, OpenAICodexBackend
from .compat import (
    DEFAULT_MODEL,
    ResponseStore,
    chat_request_to_response_payload,
    ensure_response_defaults,
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

    @app.post("/v1/responses")
    async def create_response(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        compat_request = OpenAICompatRequest.model_validate(body)
        payload = compat_request.model_dump(exclude_none=True)
        if payload.get("stream"):
            return _openai_error(
                400,
                "Streaming responses are not implemented by this compatibility server yet.",
                param="stream",
            )

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

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        compat_request = OpenAICompatRequest.model_validate(body)
        payload = compat_request.model_dump(exclude_none=True)
        if payload.get("stream"):
            return _openai_error(
                400,
                "Streaming chat completions are not implemented by this compatibility server yet.",
                param="stream",
            )

        response_payload = chat_request_to_response_payload(
            payload, default_model=request.app.state.default_model
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
