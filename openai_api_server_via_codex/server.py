from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import hmac
import inspect
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from . import __version__
from .auth import AUTH_JSON_ENV, BorrowKeyError, CodexAuthConfig, borrow_codex_key
from .backend import (
    CODEX_BASE_URL,
    CodexBackend,
    CodexBackendError,
    CodexHttpBackend,
    _forward_proxy_request_headers,
    _forward_proxy_response_headers,
    _validate_proxy_path,
)
from . import config as config_module
from .compat import (
    ChatCompletionStore,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MAX_STORED_ITEMS,
    DEFAULT_MODEL,
    ResponseStore,
    chat_request_to_response_payload,
    ensure_response_defaults,
    extract_response_text,
    prepare_response_payload,
    response_to_chat_completion,
    uses_legacy_chat_functions,
)
from .daemon import (
    DaemonError,
    DaemonPaths,
    LOG_FILE_ENV,
    PID_FILE_ENV,
    STATE_DIR_ENV,
    daemon_status,
    find_daemon_pid_files,
    resolve_daemon_paths,
    run_supervised,
    start_background,
    stop_background,
)
from .redaction import (
    install_redacting_filter,
    redact_sensitive_data,
    redact_sensitive_text,
)


LOGGER = logging.getLogger("openai_api_server_via_codex")
install_redacting_filter(LOGGER)
MAX_STORED_ITEMS_ENV = "OPENAI_VIA_CODEX_MAX_STORED_ITEMS"
MAX_CONCURRENT_REQUESTS_ENV = "OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS"
API_KEY_ENV = "OPENAI_VIA_CODEX_API_KEY"


class OpenAICompatRequest(BaseModel):
    model: str | None = None
    stream: bool | None = False

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    backend_base_url: str
    client_version: str
    timeout: float
    default_model: str
    max_stored_items: int
    max_concurrent_requests: int
    api_key: str | None = None
    verbose: bool = False
    auth_json: Path | None = None


ConfigData = dict[str, Any]


@dataclass(frozen=True)
class DaemonPathSelection:
    paths: DaemonPaths
    ambiguous_pid_files: tuple[Path, ...] = ()


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def create_app(
    *,
    backend: CodexBackend | None = None,
    default_model: str | None = None,
    auth_json: str | Path | None = None,
    backend_base_url: str | None = None,
    client_version: str | None = None,
    timeout: float | None = None,
    verbose: bool = False,
    max_stored_items: int | None = None,
    max_concurrent_requests: int | None = None,
    api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _close_backend(getattr(app.state, "backend", None))

    app = FastAPI(title="OpenAI API Server via Codex", lifespan=lifespan)
    app.state.verbose = verbose
    selected_timeout = (
        timeout
        if timeout is not None
        else float(
            os.environ.get("OPENAI_VIA_CODEX_TIMEOUT", str(config_module.DEFAULT_TIMEOUT))
        )
    )
    selected_max_stored_items = _non_negative_int_value(
        max_stored_items
        if max_stored_items is not None
        else os.environ.get(MAX_STORED_ITEMS_ENV, DEFAULT_MAX_STORED_ITEMS)
    )
    selected_max_concurrent_requests = _non_negative_int_value(
        max_concurrent_requests
        if max_concurrent_requests is not None
        else os.environ.get(
            MAX_CONCURRENT_REQUESTS_ENV,
            config_module.DEFAULT_MAX_CONCURRENT_REQUESTS,
        )
    )
    selected_auth_config = CodexAuthConfig(
        auth_json=_resolve_optional_path(auth_json or os.environ.get(AUTH_JSON_ENV))
    )
    selected_api_key = _optional_secret_value(
        api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    )
    app.state.backend = backend or _build_backend(
        backend_base_url=backend_base_url
        or os.environ.get("OPENAI_VIA_CODEX_BACKEND_BASE_URL", CODEX_BASE_URL),
        client_version=client_version
        or os.environ.get("OPENAI_VIA_CODEX_CLIENT_VERSION", "1.0.0"),
        timeout=selected_timeout,
        auth_config=selected_auth_config,
    )
    app.state.max_stored_items = selected_max_stored_items
    app.state.max_concurrent_requests = selected_max_concurrent_requests
    app.state.backend_semaphore = (
        asyncio.Semaphore(selected_max_concurrent_requests)
        if selected_max_concurrent_requests > 0
        else None
    )
    app.state.response_store = ResponseStore(max_entries=selected_max_stored_items)
    app.state.chat_completion_store = ChatCompletionStore(
        max_entries=selected_max_stored_items
    )
    app.state.default_model = default_model or os.environ.get(
        "OPENAI_VIA_CODEX_DEFAULT_MODEL", DEFAULT_MODEL
    )
    app.state.api_key = selected_api_key
    _install_api_key_auth(app)
    _install_verbose_request_logging(app)
    _install_unhandled_exception_middleware(app)
    if verbose:
        LOGGER.info(
            "server.create_app backend=codex-http default_model=%s timeout=%s max_stored_items=%s auth_json=%s "
            "backend_base_url=%s max_concurrent_requests=%s api_key_configured=%s",
            app.state.default_model,
            selected_timeout,
            selected_max_stored_items,
            selected_auth_config.auth_json,
            backend_base_url
            or os.environ.get("OPENAI_VIA_CODEX_BACKEND_BASE_URL", CODEX_BASE_URL),
            selected_max_concurrent_requests,
            bool(selected_api_key),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        backend = _get_backend(request)
        _log_verbose(request, "models.list backend=%s", _backend_name(backend))
        async with _backend_slot(request):
            model_ids = await backend.list_models()
        _log_verbose(request, "models.list.done count=%d", len(model_ids))
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
        backend = _get_backend(request)
        previous_response_id = prepared.get("previous_response_id")
        _log_verbose(
            request,
            "responses.create model=%s stream=%s input_items=%d "
            "previous_response_id=%s tools=%d backend=%s",
            prepared.get("model"),
            bool(payload.get("stream")),
            len(prepared.get("input") or []),
            bool(previous_response_id),
            len(prepared.get("tools") or []),
            _backend_name(backend),
        )
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
                key: value for key, value in prepared.items() if key != "previous_response_id"
            }
            return _sse_response(
                _responses_event_stream(
                    backend=backend,
                    store=store,
                    prepared=prepared,
                    backend_payload=backend_payload,
                    previous_response_id=previous_response_id,
                    backend_semaphore=_get_backend_semaphore(request),
                )
            )

        backend_payload = {
            key: value for key, value in prepared.items() if key != "previous_response_id"
        }
        try:
            async with _backend_slot(request):
                response = await backend.create_response(backend_payload)
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "responses.create.error status=%s message=%s",
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        except Exception as exc:
            LOGGER.error(
                "responses.create.unhandled_error type=%s message=%s",
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )
        response = ensure_response_defaults(response, request_payload=prepared)
        if previous_response_id:
            response["previous_response_id"] = previous_response_id

        store.remember(
            str(response["id"]),
            effective_input=prepared["input"],
            response=response,
        )
        _log_verbose(
            request,
            "responses.create.done response_id=%s status=%s output_items=%d",
            response.get("id"),
            response.get("status"),
            len(response.get("output") or []),
        )
        return JSONResponse(response)

    @app.post("/v1/responses/input_tokens", response_model=None)
    async def count_response_input_tokens(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        prepared = prepare_response_payload(
            body, default_model=request.app.state.default_model
        )
        _log_verbose(
            request,
            "responses.input_tokens model=%s input_items=%d tools=%d",
            prepared.get("model"),
            len(prepared.get("input") or []),
            len(prepared.get("tools") or []),
        )
        previous_response_id = prepared.get("previous_response_id")
        if previous_response_id:
            stored = _get_response_store(request).get(str(previous_response_id))
            if stored is None:
                return _openai_error(
                    404,
                    f"Unknown previous_response_id: {previous_response_id}",
                    param="previous_response_id",
                )
            prepared["input"] = stored.context_items + prepared["input"]

        return JSONResponse(
            {
                "object": "response.input_tokens",
                "input_tokens": _estimate_input_tokens(prepared),
            }
        )

    @app.get("/v1/responses/{response_id}", response_model=None)
    async def retrieve_response(
        request: Request, response_id: str
    ) -> JSONResponse | StreamingResponse:
        stored = _get_response_store(request).get(response_id)
        _log_verbose(
            request,
            "responses.retrieve response_id=%s stream=%s found=%s",
            response_id,
            request.query_params.get("stream") == "true",
            stored is not None,
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown response_id: {response_id}",
                param="response_id",
            )
        if request.query_params.get("stream") == "true":
            return _sse_response(_stored_response_event_stream(stored.response))
        return JSONResponse(copy.deepcopy(stored.response))

    @app.delete("/v1/responses/{response_id}", response_model=None)
    async def delete_response(request: Request, response_id: str) -> Response | JSONResponse:
        deleted = _get_response_store(request).delete(response_id)
        _log_verbose(
            request,
            "responses.delete response_id=%s deleted=%s",
            response_id,
            deleted,
        )
        if not deleted:
            return _openai_error(
                404,
                f"Unknown response_id: {response_id}",
                param="response_id",
            )
        return Response(status_code=204)

    @app.post("/v1/responses/{response_id}/cancel", response_model=None)
    async def cancel_response(request: Request, response_id: str) -> JSONResponse:
        stored = _get_response_store(request).get(response_id)
        _log_verbose(
            request,
            "responses.cancel response_id=%s found=%s",
            response_id,
            stored is not None,
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown response_id: {response_id}",
                param="response_id",
            )
        if stored.response.get("status") not in {"queued", "in_progress"}:
            return _openai_error(
                409,
                "Only queued or in-progress background responses can be cancelled.",
                param="response_id",
            )
        cancelled = _get_response_store(request).cancel(response_id)
        assert cancelled is not None
        return JSONResponse(copy.deepcopy(cancelled.response))

    @app.get("/v1/responses/{response_id}/input_items", response_model=None)
    async def list_response_input_items(
        request: Request, response_id: str
    ) -> JSONResponse:
        stored = _get_response_store(request).get(response_id)
        _log_verbose(
            request,
            "responses.input_items response_id=%s found=%s limit=%s order=%s after=%s",
            response_id,
            stored is not None,
            request.query_params.get("limit"),
            request.query_params.get("order"),
            request.query_params.get("after"),
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown response_id: {response_id}",
                param="response_id",
            )

        items = _response_input_page_items(stored.effective_input)
        order = request.query_params.get("order")
        if order == "desc":
            items.reverse()
        after = request.query_params.get("after")
        if after:
            items = _items_after_cursor(items, after)
        limit = _positive_int(request.query_params.get("limit"))
        has_more = False
        if limit is not None:
            has_more = len(items) > limit
            items = items[:limit]

        return JSONResponse(
            {
                "object": "list",
                "data": items,
                "first_id": _input_item_id(items[0], 0) if items else None,
                "last_id": _input_item_id(items[-1], len(items) - 1) if items else None,
                "has_more": has_more,
            }
        )

    @app.get("/v1/chat/completions", response_model=None)
    async def list_chat_completions(request: Request) -> JSONResponse:
        metadata = _metadata_query_params(request)
        _log_verbose(
            request,
            "chat.completions.list model=%s metadata_keys=%s limit=%s order=%s after=%s",
            request.query_params.get("model"),
            sorted(metadata),
            request.query_params.get("limit"),
            request.query_params.get("order"),
            request.query_params.get("after"),
        )
        items, has_more = _get_chat_completion_store(request).list(
            model=request.query_params.get("model"),
            metadata=metadata,
            order=request.query_params.get("order"),
            after=request.query_params.get("after"),
            limit=_positive_int(request.query_params.get("limit")),
        )
        return JSONResponse(_cursor_page(items, has_more=has_more))

    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse | StreamingResponse:
        compat_request = OpenAICompatRequest.model_validate(body)
        payload = compat_request.model_dump(exclude_none=True)
        response_payload = chat_request_to_response_payload(
            payload, default_model=request.app.state.default_model
        )
        legacy_functions = uses_legacy_chat_functions(payload)
        _log_verbose(
            request,
            "chat.completions.create model=%s stream=%s messages=%d store=%s "
            "tools=%d legacy_functions=%s n=%s backend=%s",
            response_payload.get("model"),
            bool(payload.get("stream")),
            len(payload.get("messages") or []),
            payload.get("store") is True,
            len(payload.get("tools") or []) + len(payload.get("functions") or []),
            legacy_functions,
            payload.get("n") or 1,
            _backend_name(_get_backend(request)),
        )
        if payload.get("stream"):
            return _sse_response(
                _chat_completion_event_stream(
                    backend=_get_backend(request),
                    response_payload=response_payload,
                    chat_payload=payload,
                    legacy_functions=legacy_functions,
                    chat_store=_get_chat_completion_store(request),
                    backend_semaphore=_get_backend_semaphore(request),
                )
            )

        try:
            async with _backend_slot(request):
                response = await _get_backend(request).create_response(response_payload)
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "chat.completions.create.error status=%s message=%s",
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        except Exception as exc:
            LOGGER.error(
                "chat.completions.create.unhandled_error type=%s message=%s",
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )
        response = ensure_response_defaults(response, request_payload=response_payload)
        chat_completion = response_to_chat_completion(
            response,
            fallback_model=response_payload["model"],
            legacy_functions=legacy_functions,
            n=_chat_choice_count(payload.get("n")),
        )
        if payload.get("metadata") is not None:
            chat_completion["metadata"] = copy.deepcopy(payload["metadata"])
        if payload.get("store") is True:
            _get_chat_completion_store(request).remember(
                str(chat_completion["id"]),
                completion=chat_completion,
                metadata=payload.get("metadata") or {},
            )
        _log_verbose(
            request,
            "chat.completions.create.done completion_id=%s choices=%d finish_reason=%s",
            chat_completion.get("id"),
            len(chat_completion.get("choices") or []),
            (chat_completion.get("choices") or [{}])[0].get("finish_reason"),
        )
        return JSONResponse(chat_completion)

    @app.get("/v1/chat/completions/{completion_id}", response_model=None)
    async def retrieve_chat_completion(
        request: Request, completion_id: str
    ) -> JSONResponse:
        stored = _get_chat_completion_store(request).get(completion_id)
        _log_verbose(
            request,
            "chat.completions.retrieve completion_id=%s found=%s",
            completion_id,
            stored is not None,
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown completion_id: {completion_id}",
                param="completion_id",
            )
        return JSONResponse(copy.deepcopy(stored.completion))

    @app.post("/v1/chat/completions/{completion_id}", response_model=None)
    async def update_chat_completion(
        request: Request,
        completion_id: str,
        body: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        metadata = body.get("metadata")
        _log_verbose(
            request,
            "chat.completions.update completion_id=%s metadata_keys=%s",
            completion_id,
            sorted(metadata) if isinstance(metadata, dict) else None,
        )
        if metadata is not None and not isinstance(metadata, dict):
            return _openai_error(
                400,
                "metadata must be an object or null.",
                param="metadata",
            )
        stored = _get_chat_completion_store(request).update_metadata(
            completion_id, metadata
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown completion_id: {completion_id}",
                param="completion_id",
            )
        return JSONResponse(copy.deepcopy(stored.completion))

    @app.delete("/v1/chat/completions/{completion_id}", response_model=None)
    async def delete_chat_completion(
        request: Request, completion_id: str
    ) -> JSONResponse:
        deleted = _get_chat_completion_store(request).delete(completion_id)
        _log_verbose(
            request,
            "chat.completions.delete completion_id=%s deleted=%s",
            completion_id,
            deleted,
        )
        if not deleted:
            return _openai_error(
                404,
                f"Unknown completion_id: {completion_id}",
                param="completion_id",
            )
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion.deleted",
                "deleted": True,
            }
        )

    @app.get("/v1/chat/completions/{completion_id}/messages", response_model=None)
    async def list_chat_completion_messages(
        request: Request, completion_id: str
    ) -> JSONResponse:
        stored = _get_chat_completion_store(request).get(completion_id)
        _log_verbose(
            request,
            "chat.completions.messages completion_id=%s found=%s limit=%s order=%s after=%s",
            completion_id,
            stored is not None,
            request.query_params.get("limit"),
            request.query_params.get("order"),
            request.query_params.get("after"),
        )
        if stored is None:
            return _openai_error(
                404,
                f"Unknown completion_id: {completion_id}",
                param="completion_id",
            )

        messages = [copy.deepcopy(message) for message in stored.messages]
        order = request.query_params.get("order")
        if order == "desc":
            messages.reverse()
        after = request.query_params.get("after")
        if after:
            messages = _messages_after_cursor(messages, after)
        limit = _positive_int(request.query_params.get("limit"))
        has_more = False
        if limit is not None:
            has_more = len(messages) > limit
            messages = messages[:limit]

        return JSONResponse(_cursor_page(messages, has_more=has_more))

    @app.post("/v1/audio/transcriptions", response_model=None)
    async def create_audio_transcription(request: Request) -> Response | JSONResponse:
        backend = _get_backend(request)
        body = await request.body()
        headers = _forward_proxy_request_headers(request.headers)
        _log_verbose(
            request,
            "audio.transcriptions.create bytes=%d backend=%s",
            len(body),
            _backend_name(backend),
        )
        try:
            async with _backend_slot(request):
                upstream = await backend.transcribe_audio(
                    headers=headers,
                    body=body,
                )
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "audio.transcriptions.create.error status=%s message=%s",
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        except Exception as exc:
            LOGGER.error(
                "audio.transcriptions.create.unhandled_error type=%s message=%s",
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )

        response_headers = _forward_proxy_response_headers(upstream.headers)
        response_headers["x-openai-via-codex-proxy"] = "codex-http"
        _log_verbose(
            request,
            "audio.transcriptions.create.done status=%s bytes=%d",
            upstream.status_code,
            len(upstream.body),
        )
        return Response(
            content=upstream.body,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    @app.post("/v1/images/generations", response_model=None)
    async def create_image_generation(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        backend = _get_backend(request)
        _log_verbose(
            request,
            "images.generations.create model=%s n=%s size=%s quality=%s output_format=%s backend=%s",
            body.get("model"),
            body.get("n") or 1,
            body.get("size"),
            body.get("quality"),
            body.get("output_format"),
            _backend_name(backend),
        )
        validation_error = _validate_image_generation_request(body)
        if validation_error is not None:
            return validation_error

        count = int(body.get("n") or 1)
        try:
            async with _backend_slot(request):
                response_items = [
                    await backend.create_response(
                        _image_generation_response_payload(
                            body,
                            default_model=request.app.state.default_model,
                        )
                    )
                    for _ in range(count)
                ]
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "images.generations.create.error status=%s message=%s",
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        except Exception as exc:
            LOGGER.error(
                "images.generations.create.unhandled_error type=%s message=%s",
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )

        try:
            images_response = _responses_to_images_response(body, response_items)
        except CodexBackendError as exc:
            return _openai_error(exc.status_code, str(exc), error_type="api_error")

        _log_verbose(
            request,
            "images.generations.create.done count=%d output_format=%s",
            len(images_response.get("data") or []),
            images_response.get("output_format"),
        )
        return JSONResponse(images_response)

    @app.api_route(
        "/v1/{proxy_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        response_model=None,
    )
    async def proxy_unknown_v1_endpoint(
        request: Request, proxy_path: str
    ) -> Response | JSONResponse:
        backend = _get_backend(request)
        method = request.method.upper()
        try:
            _validate_proxy_path(proxy_path)
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "proxy.request.rejected method=%s path=/v1/%s status=%s message=%s",
                method,
                proxy_path,
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        body = await request.body()
        headers = _forward_proxy_request_headers(request.headers)
        query = bytes(request.scope.get("query_string") or b"")
        _log_verbose(
            request,
            "proxy.request method=%s path=/v1/%s query=%s bytes=%d backend=%s",
            method,
            proxy_path,
            redact_sensitive_text(request.url.query),
            len(body),
            _backend_name(backend),
        )
        try:
            async with _backend_slot(request):
                upstream = await backend.proxy_request(
                    method,
                    proxy_path,
                    query=query,
                    headers=headers,
                    body=body,
                )
        except CodexBackendError as exc:
            message = redact_sensitive_text(str(exc))
            _log_verbose(
                request,
                "proxy.request.error status=%s message=%s",
                exc.status_code,
                message,
            )
            return _openai_error(exc.status_code, message, error_type="api_error")
        except Exception as exc:
            LOGGER.error(
                "proxy.request.unhandled_error method=%s path=/v1/%s type=%s message=%s",
                method,
                proxy_path,
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )

        response_headers = _forward_proxy_response_headers(upstream.headers)
        response_headers["x-openai-via-codex-proxy"] = "codex-http"
        _log_verbose(
            request,
            "proxy.request.done method=%s path=/v1/%s status=%s bytes=%d",
            method,
            proxy_path,
            upstream.status_code,
            len(upstream.body),
        )
        return Response(
            content=upstream.body,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    return app


def _build_backend(
    *,
    backend_base_url: str,
    client_version: str,
    timeout: float,
    auth_config: CodexAuthConfig,
) -> CodexBackend:
    return CodexHttpBackend(
        base_url=backend_base_url,
        client_version=client_version,
        timeout=timeout,
        auth_config=auth_config,
    )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (
        argv[0].startswith("-") and argv[0] not in {"-h", "--help", "--version"}
    ):
        argv = ["serve", *argv]

    parser = argparse.ArgumentParser(
        prog="openai-api-server-via-codex",
        description="OpenAI-compatible API server backed by Codex credentials.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve", help="run the server in the foreground", allow_abbrev=False
    )
    _add_config_option(serve)
    _add_server_options(serve)

    daemon_run = subparsers.add_parser(
        "daemon-run", help=argparse.SUPPRESS, allow_abbrev=False
    )
    _add_config_option(daemon_run)
    _add_server_options(daemon_run)

    start = subparsers.add_parser(
        "start", help="run the server in the background", allow_abbrev=False
    )
    _add_config_option(start)
    _add_server_options(start)
    _add_daemon_options(start)

    stop = subparsers.add_parser(
        "stop", help="stop the background server", allow_abbrev=False
    )
    _add_config_option(stop)
    _add_daemon_selector_options(stop)
    _add_verbose_option(stop)
    stop.add_argument(
        "--stop-timeout",
        type=float,
        default=None,
        help="seconds to wait after SIGTERM before SIGKILL",
    )

    status = subparsers.add_parser(
        "status", help="show background server status", allow_abbrev=False
    )
    _add_config_option(status)
    _add_daemon_selector_options(status)
    _add_verbose_option(status)

    config_generate = subparsers.add_parser(
        "config-generate",
        help="write a default TOML configuration",
        allow_abbrev=False,
    )
    config_generate.add_argument("--config", help="config file path")
    config_generate.add_argument(
        "--force", action="store_true", help="overwrite an existing config file"
    )
    config_generate.add_argument(
        "--stdout", action="store_true", help="print the template instead of writing"
    )

    return parser.parse_args(argv)


def load_config_for_args(args: argparse.Namespace) -> ConfigData:
    return config_module.load_config(getattr(args, "config", None))


def server_settings_from_args(
    args: argparse.Namespace, loaded_config: ConfigData | None = None
) -> ServerSettings:
    config_data = loaded_config or {}
    return ServerSettings(
        host=_arg_env_config_str(
            args,
            "host",
            "OPENAI_VIA_CODEX_HOST",
            config_data,
            "server",
            "host",
            config_module.DEFAULT_HOST,
        ),
        port=_arg_env_config_int(
            args,
            "port",
            "OPENAI_VIA_CODEX_PORT",
            config_data,
            "server",
            "port",
            config_module.DEFAULT_PORT,
        ),
        backend_base_url=_arg_env_config_str(
            args,
            "backend_base_url",
            "OPENAI_VIA_CODEX_BACKEND_BASE_URL",
            config_data,
            "codex",
            "backend_base_url",
            CODEX_BASE_URL,
        ),
        client_version=_arg_env_config_str(
            args,
            "client_version",
            "OPENAI_VIA_CODEX_CLIENT_VERSION",
            config_data,
            "codex",
            "client_version",
            config_module.DEFAULT_CLIENT_VERSION,
        ),
        timeout=_arg_env_config_float(
            args,
            "timeout",
            "OPENAI_VIA_CODEX_TIMEOUT",
            config_data,
            "server",
            "timeout",
            config_module.DEFAULT_TIMEOUT,
        ),
        default_model=_arg_env_config_str(
            args,
            "default_model",
            "OPENAI_VIA_CODEX_DEFAULT_MODEL",
            config_data,
            "server",
            "default_model",
            DEFAULT_MODEL,
        ),
        verbose=_arg_env_config_bool(
            args,
            "verbose",
            "OPENAI_VIA_CODEX_VERBOSE",
            config_data,
            "server",
            "verbose",
            False,
        ),
        max_stored_items=_arg_env_config_non_negative_int(
            args,
            "max_stored_items",
            MAX_STORED_ITEMS_ENV,
            config_data,
            "server",
            "max_stored_items",
            config_module.DEFAULT_MAX_STORED_ITEMS,
        ),
        max_concurrent_requests=_arg_env_config_non_negative_int(
            args,
            "max_concurrent_requests",
            MAX_CONCURRENT_REQUESTS_ENV,
            config_data,
            "server",
            "max_concurrent_requests",
            config_module.DEFAULT_MAX_CONCURRENT_REQUESTS,
        ),
        api_key=_optional_secret_value(
            _arg_env_config_optional_str(
                args, "api_key", API_KEY_ENV, config_data, "server", "api_key"
            )
        ),
        auth_json=_resolve_optional_path(
            _arg_env_config_optional_str(
                args, "auth_json", AUTH_JSON_ENV, config_data, "codex", "auth_json"
            )
        ),
    )


def daemon_paths_from_args(
    args: argparse.Namespace,
    settings: ServerSettings,
    loaded_config: ConfigData | None = None,
) -> DaemonPaths:
    config_data = loaded_config or {}
    return resolve_daemon_paths(
        host=settings.host,
        port=settings.port,
        state_dir=_arg_env_config_optional_str(
            args, "state_dir", STATE_DIR_ENV, config_data, "daemon", "state_dir"
        ),
        pid_file=_arg_env_config_optional_str(
            args, "pid_file", PID_FILE_ENV, config_data, "daemon", "pid_file"
        ),
        log_file=_arg_env_config_optional_str(
            args, "log_file", LOG_FILE_ENV, config_data, "daemon", "log_file"
        ),
    )


def select_daemon_paths_from_args(
    args: argparse.Namespace,
    settings: ServerSettings,
    loaded_config: ConfigData | None = None,
) -> DaemonPathSelection:
    config_data = loaded_config or {}
    paths = daemon_paths_from_args(args, settings, config_data)
    if not _should_discover_daemon_pid_file(args, config_data):
        return DaemonPathSelection(paths=paths)
    if paths.pid_file.exists():
        return DaemonPathSelection(paths=paths)

    candidates = find_daemon_pid_files(
        state_dir=paths.state_dir,
        port=settings.port,
    )
    if len(candidates) == 1:
        return DaemonPathSelection(
            paths=_paths_for_discovered_pid_file(args, config_data, paths, candidates[0])
        )
    if len(candidates) > 1:
        return DaemonPathSelection(paths=paths, ambiguous_pid_files=tuple(candidates))
    return DaemonPathSelection(paths=paths)


def stop_timeout_from_args(
    args: argparse.Namespace, loaded_config: ConfigData | None = None
) -> float:
    return _arg_env_config_float(
        args,
        "stop_timeout",
        "OPENAI_VIA_CODEX_STOP_TIMEOUT",
        loaded_config or {},
        "daemon",
        "stop_timeout",
        config_module.DEFAULT_STOP_TIMEOUT,
    )


def serve_command(settings: ServerSettings) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "openai_api_server_via_codex.server",
        "serve",
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--backend-base-url",
        settings.backend_base_url,
        "--client-version",
        settings.client_version,
        "--timeout",
        str(settings.timeout),
        "--max-stored-items",
        str(settings.max_stored_items),
        "--max-concurrent-requests",
        str(settings.max_concurrent_requests),
        "--default-model",
        settings.default_model,
    ]
    if settings.auth_json is not None:
        command.extend(["--auth-json", str(settings.auth_json)])
    if settings.verbose:
        command.append("--verbose")
    return command


def daemon_run_command(settings: ServerSettings) -> list[str]:
    command = serve_command(settings)
    command[3] = "daemon-run"
    return command


def serve_env(settings: ServerSettings) -> dict[str, str] | None:
    if settings.api_key is None:
        return None
    env = os.environ.copy()
    env[API_KEY_ENV] = settings.api_key
    return env


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(_main(argv))


def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "config-generate":
        if args.stdout:
            print(config_module.default_config_toml(), end="")
            return 0
        try:
            path = config_module.write_default_config(
                getattr(args, "config", None), force=bool(args.force)
            )
        except FileExistsError as exc:
            existing_path = Path(exc.args[0])
            print(
                f"Config already exists: {existing_path}. Use --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        print(f"Wrote config: {path}")
        return 0

    loaded_config = load_config_for_args(args)

    if args.command == "serve":
        settings = server_settings_from_args(args, loaded_config)
        _configure_logging(settings.verbose)
        _log_settings(settings, config_path=config_module.resolve_config_path(getattr(args, "config", None)))
        if not _preflight_codex_auth_or_print(settings):
            return 1
        uvicorn.run(
            app=create_app(
                default_model=settings.default_model,
                auth_json=settings.auth_json,
                backend_base_url=settings.backend_base_url,
                client_version=settings.client_version,
                timeout=settings.timeout,
                verbose=settings.verbose,
                max_stored_items=settings.max_stored_items,
                max_concurrent_requests=settings.max_concurrent_requests,
                api_key=settings.api_key,
            ),
            host=settings.host,
            port=settings.port,
            log_level="debug" if settings.verbose else "info",
            reload=False,
        )
        return 0

    if args.command == "daemon-run":
        settings = server_settings_from_args(args, loaded_config)
        _configure_logging(settings.verbose)
        _log_settings(settings, config_path=config_module.resolve_config_path(getattr(args, "config", None)))
        return run_supervised(serve_command(settings), env=serve_env(settings))

    settings = server_settings_from_args(args, loaded_config)
    _configure_logging(settings.verbose)
    _log_settings(settings, config_path=config_module.resolve_config_path(getattr(args, "config", None)))
    selection = select_daemon_paths_from_args(args, settings, loaded_config)
    if selection.ambiguous_pid_files:
        _print_ambiguous_daemon_pid_files(settings, selection.ambiguous_pid_files)
        return 1
    paths = selection.paths

    if args.command == "start":
        if not _preflight_codex_auth_or_print(settings):
            return 1
        LOGGER.info(
            "daemon.start host=%s port=%s pid_file=%s log_file=%s command=%s",
            settings.host,
            settings.port,
            paths.pid_file,
            paths.log_file,
            daemon_run_command(settings),
        )
        try:
            pid = start_background(
                daemon_run_command(settings), paths, env=serve_env(settings)
            )
        except DaemonError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Started openai-api-server-via-codex on {settings.host}:{settings.port}")
        print(f"PID: {pid}")
        print(f"PID file: {paths.pid_file}")
        print(f"Log file: {paths.log_file}")
        return 0

    if args.command == "stop":
        LOGGER.info(
            "daemon.stop host=%s port=%s pid_file=%s timeout=%s",
            settings.host,
            settings.port,
            paths.pid_file,
            stop_timeout_from_args(args, loaded_config),
        )
        result = stop_background(paths, timeout=stop_timeout_from_args(args, loaded_config))
        if result.state == "not_running":
            print(f"Not running. PID file: {paths.pid_file}")
        elif result.state == "stale":
            print(f"Removed stale PID {result.pid}. PID file: {paths.pid_file}")
        else:
            print(f"Stopped PID {result.pid} ({result.state}).")
        return 0

    if args.command == "status":
        LOGGER.info(
            "daemon.status host=%s port=%s pid_file=%s log_file=%s",
            settings.host,
            settings.port,
            paths.pid_file,
            paths.log_file,
        )
        status = daemon_status(paths)
        if status.pid is None:
            print(f"stopped. PID file: {status.pid_file}")
        else:
            print(f"{status.state}: PID {status.pid}")
            print(f"PID file: {status.pid_file}")
            print(f"Log file: {status.log_file}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def _preflight_codex_auth_or_print(settings: ServerSettings) -> bool:
    try:
        auth_path, account_id_present = _preflight_codex_auth(settings)
    except BorrowKeyError as exc:
        message = redact_sensitive_text(str(exc))
        LOGGER.error("codex.auth.preflight.failed message=%s", message)
        print(f"Codex auth preflight failed: {message}", file=sys.stderr)
        return False
    print(
        "Codex auth preflight OK: "
        f"{auth_path} (account_id_present={account_id_present})"
    )
    return True


def _preflight_codex_auth(settings: ServerSettings) -> tuple[Path, bool]:
    _, account_id = borrow_codex_key(settings.auth_json)
    auth_path = settings.auth_json or Path(os.environ.get("CODEX_HOME", "~/.codex")) / "auth.json"
    auth_path = auth_path.expanduser().resolve()
    LOGGER.info(
        "codex.auth.preflight.ok auth_json=%s account_id_present=%s",
        auth_path,
        bool(account_id),
    )
    return auth_path, bool(account_id)


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help="server host")
    parser.add_argument("--port", type=int, help="server port")
    parser.add_argument("--auth-json", help="path to Codex auth.json")
    parser.add_argument("--api-key", help="optional API key required by this server")
    parser.add_argument("--backend-base-url", help="Codex backend base URL")
    parser.add_argument("--client-version", help="Codex client_version parameter")
    parser.add_argument("--timeout", type=float, help="backend timeout in seconds")
    parser.add_argument(
        "--max-stored-items",
        type=_non_negative_int_arg,
        help="maximum in-memory responses and chat completions",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=_non_negative_int_arg,
        help="maximum concurrent Codex backend requests; 0 means unlimited",
    )
    parser.add_argument("--default-model", help="default model when request omits model")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="enable verbose server logs",
    )


def _add_verbose_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="enable verbose command logs",
    )


def _add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to config.toml")


def _add_daemon_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", help="directory for default PID and log files")
    parser.add_argument("--pid-file", help="explicit PID file path")
    parser.add_argument("--log-file", help="explicit log file path")


def _add_daemon_selector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help="server host used to derive default PID file")
    parser.add_argument("--port", type=int, help="server port used to derive default PID file")
    _add_daemon_options(parser)


def _should_discover_daemon_pid_file(
    args: argparse.Namespace, config_data: ConfigData
) -> bool:
    return not (
        _selector_is_explicit(args, "host", "OPENAI_VIA_CODEX_HOST", config_data, "server", "host")
        or _selector_is_explicit(args, "pid_file", PID_FILE_ENV, config_data, "daemon", "pid_file")
    )


def _selector_is_explicit(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
) -> bool:
    return (
        getattr(args, arg_name, None) is not None
        or os.environ.get(env_name) is not None
        or _config_value(config_data, section, key) is not None
    )


def _paths_for_discovered_pid_file(
    args: argparse.Namespace,
    config_data: ConfigData,
    paths: DaemonPaths,
    pid_file: Path,
) -> DaemonPaths:
    if _selector_is_explicit(
        args,
        "log_file",
        LOG_FILE_ENV,
        config_data,
        "daemon",
        "log_file",
    ):
        log_file = paths.log_file
    else:
        log_file = pid_file.with_suffix(".log")
    return DaemonPaths(
        state_dir=paths.state_dir,
        pid_file=pid_file.expanduser().resolve(),
        log_file=log_file.expanduser().resolve(),
    )


def _print_ambiguous_daemon_pid_files(
    settings: ServerSettings,
    pid_files: tuple[Path, ...],
) -> None:
    print(
        f"Multiple PID files match port {settings.port}. "
        "Use --host or --pid-file to choose one:",
        file=sys.stderr,
    )
    for pid_file in pid_files:
        print(f"  {pid_file}", file=sys.stderr)


def _arg_env_str(
    args: argparse.Namespace, arg_name: str, env_name: str, default: str
) -> str:
    value = getattr(args, arg_name, None)
    if value is not None:
        return str(value)
    return os.environ.get(env_name, default)


def _arg_env_config_str(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
    default: str,
) -> str:
    value = getattr(args, arg_name, None)
    if value is not None:
        return str(value)
    if env_value := os.environ.get(env_name):
        return env_value
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return str(config_value)
    return default


def _arg_env_int(
    args: argparse.Namespace, arg_name: str, env_name: str, default: int
) -> int:
    value = getattr(args, arg_name, None)
    if value is not None:
        return int(value)
    return int(os.environ.get(env_name, str(default)))


def _arg_env_config_int(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
    default: int,
) -> int:
    value = getattr(args, arg_name, None)
    if value is not None:
        return int(value)
    if env_value := os.environ.get(env_name):
        return int(env_value)
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return int(config_value)
    return default


def _arg_env_config_non_negative_int(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
    default: int,
) -> int:
    value = getattr(args, arg_name, None)
    if value is not None:
        return _non_negative_int_value(value)
    if env_value := os.environ.get(env_name):
        return _non_negative_int_value(env_value)
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return _non_negative_int_value(config_value)
    return default


def _arg_env_float(
    args: argparse.Namespace, arg_name: str, env_name: str, default: float
) -> float:
    value = getattr(args, arg_name, None)
    if value is not None:
        return float(value)
    return float(os.environ.get(env_name, str(default)))


def _arg_env_config_float(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
    default: float,
) -> float:
    value = getattr(args, arg_name, None)
    if value is not None:
        return float(value)
    if env_value := os.environ.get(env_name):
        return float(env_value)
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return float(config_value)
    return default


def _arg_env_config_bool(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
    default: bool,
) -> bool:
    value = getattr(args, arg_name, None)
    if value is not None:
        return bool(value)
    if env_value := os.environ.get(env_name):
        return _parse_bool(env_value)
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return _parse_bool(config_value)
    return default


def _arg_env_optional_str(
    args: argparse.Namespace, arg_name: str, env_name: str
) -> str | None:
    value = getattr(args, arg_name, None)
    if value is not None:
        return str(value)
    return os.environ.get(env_name)


def _arg_env_config_optional_str(
    args: argparse.Namespace,
    arg_name: str,
    env_name: str,
    config_data: ConfigData,
    section: str,
    key: str,
) -> str | None:
    value = getattr(args, arg_name, None)
    if value is not None:
        return str(value)
    if env_value := os.environ.get(env_name):
        return env_value
    config_value = _config_value(config_data, section, key)
    if config_value is not None:
        return str(config_value)
    return None


def _config_value(config_data: ConfigData, section: str, key: str) -> Any:
    section_data = config_data.get(section)
    if not isinstance(section_data, dict):
        return None
    return section_data.get(key)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _non_negative_int_arg(value: str) -> int:
    try:
        return _non_negative_int_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _non_negative_int_value(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _optional_secret_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    LOGGER.setLevel(level)
    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        install_redacting_filter(handler)


def _log_settings(settings: ServerSettings, *, config_path: Path) -> None:
    if not settings.verbose:
        return
    LOGGER.info(
        "config.loaded path=%s exists=%s",
        config_path,
        config_path.exists(),
    )
    LOGGER.info(
        "settings.resolved backend=codex-http host=%s port=%s default_model=%s "
        "timeout=%s max_stored_items=%s max_concurrent_requests=%s auth_json=%s "
        "backend_base_url=%s client_version=%s api_key_configured=%s verbose=%s",
        settings.host,
        settings.port,
        settings.default_model,
        settings.timeout,
        settings.max_stored_items,
        settings.max_concurrent_requests,
        settings.auth_json,
        settings.backend_base_url,
        settings.client_version,
        bool(settings.api_key),
        settings.verbose,
    )


def _install_api_key_auth(app: FastAPI) -> None:
    @app.middleware("http")
    async def _api_key_auth(request: Request, call_next: Any) -> Response:
        api_key = getattr(request.app.state, "api_key", None)
        if not api_key or not request.url.path.startswith("/v1/"):
            return await call_next(request)
        if _authorization_matches_api_key(
            request.headers.get("authorization"), str(api_key)
        ):
            return await call_next(request)
        return _openai_error(
            401,
            "Incorrect API key provided.",
            error_type="invalid_api_key",
            code="invalid_api_key",
        )


def _authorization_matches_api_key(authorization: str | None, api_key: str) -> bool:
    if not authorization:
        return False
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    return hmac.compare_digest(token.strip(), api_key)


def _install_verbose_request_logging(app: FastAPI) -> None:
    if not bool(getattr(app.state, "verbose", False)):
        return

    @app.middleware("http")
    async def _verbose_request_logger(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        LOGGER.info(
            "request.start method=%s path=%s query=%s client=%s",
            request.method,
            request.url.path,
            redact_sensitive_text(request.url.query),
            request.client.host if request.client else None,
        )
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "request.error method=%s path=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                (time.perf_counter() - started) * 1000,
            )
            raise
        LOGGER.info(
            "request.end method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response


def _install_unhandled_exception_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _openai_unhandled_exception_boundary(
        request: Request, call_next: Any
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            if not request.url.path.startswith("/v1/"):
                raise
            LOGGER.error(
                "request.unhandled_error method=%s path=%s type=%s message=%s",
                request.method,
                request.url.path,
                exc.__class__.__name__,
                redact_sensitive_text(str(exc)),
            )
            return _openai_error(
                500,
                _unexpected_error_message(),
                error_type="api_error",
            )


def _log_verbose(request: Request, message: str, *args: Any) -> None:
    if bool(getattr(request.app.state, "verbose", False)):
        LOGGER.info(message, *redact_sensitive_data(args))


def _backend_name(backend: Any) -> str:
    return backend.__class__.__name__


def _get_backend(request: Request) -> CodexBackend:
    return request.app.state.backend


def _get_backend_semaphore(request: Request) -> asyncio.Semaphore | None:
    semaphore = getattr(request.app.state, "backend_semaphore", None)
    return semaphore if isinstance(semaphore, asyncio.Semaphore) else None


def _backend_slot(request: Request) -> Any:
    return _backend_semaphore_slot(_get_backend_semaphore(request))


@asynccontextmanager
async def _backend_semaphore_slot(
    semaphore: asyncio.Semaphore | None,
) -> AsyncIterator[None]:
    if semaphore is None:
        yield
        return
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _get_response_store(request: Request) -> ResponseStore:
    return request.app.state.response_store


def _get_chat_completion_store(request: Request) -> ChatCompletionStore:
    return request.app.state.chat_completion_store


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 1:
        return None
    return parsed


def _chat_choice_count(value: Any) -> int:
    return _positive_int(value) or 1


def _validate_image_generation_request(body: dict[str, Any]) -> JSONResponse | None:
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _openai_error(400, "prompt is required.", param="prompt")

    if body.get("stream") is True:
        return _openai_error(
            400,
            "stream=true is not supported for image generations.",
            param="stream",
        )

    response_format = body.get("response_format")
    if response_format not in {None, "b64_json"}:
        return _openai_error(
            400,
            "Only response_format='b64_json' is supported for image generations.",
            param="response_format",
        )

    try:
        count = int(body.get("n") or 1)
    except (TypeError, ValueError):
        return _openai_error(400, "n must be an integer between 1 and 10.", param="n")
    if count < 1 or count > 10:
        return _openai_error(400, "n must be between 1 and 10.", param="n")

    output_format = _image_output_format(body)
    if output_format not in {"png", "jpeg", "webp"}:
        return _openai_error(
            400,
            "output_format must be one of png, jpeg, or webp.",
            param="output_format",
        )

    return None


def _image_generation_response_payload(
    body: dict[str, Any], *, default_model: str
) -> dict[str, Any]:
    output_format = _image_output_format(body)
    return {
        "model": default_model,
        "instructions": (
            f"{DEFAULT_INSTRUCTIONS} Use the image_generation tool when the user "
            "asks for an image. Return the generated image without adding extra "
            "requirements."
        ),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _image_generation_prompt(body),
                    }
                ],
            }
        ],
        "tools": [{"type": "image_generation", "output_format": output_format}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
    }


def _image_generation_prompt(body: dict[str, Any]) -> str:
    lines = [
        "Generate an image using the image_generation tool.",
        f"Primary request: {str(body.get('prompt') or '').strip()}",
    ]
    for key, label in (
        ("size", "Requested size"),
        ("quality", "Requested quality"),
        ("background", "Requested background"),
        ("style", "Requested style"),
    ):
        value = body.get(key)
        if isinstance(value, str) and value:
            lines.append(f"{label}: {value}")
    lines.append("Do not include text, logos, or watermarks unless explicitly requested.")
    return "\n".join(lines)


def _responses_to_images_response(
    body: dict[str, Any], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    data: list[dict[str, str]] = []
    created_values: list[int] = []
    for response in responses:
        created_at = response.get("created_at")
        if isinstance(created_at, (int, float)):
            created_values.append(int(created_at))
        image = _image_from_response(response)
        if image is None:
            raise CodexBackendError(
                "Codex backend did not return an image_generation_call.",
                status_code=502,
            )
        data.append(image)

    result: dict[str, Any] = {
        "created": created_values[0] if created_values else int(time.time()),
        "data": data,
        "output_format": _image_output_format(body),
    }
    background = body.get("background")
    if background in {"transparent", "opaque"}:
        result["background"] = background
    quality = body.get("quality")
    if quality in {"low", "medium", "high"}:
        result["quality"] = quality
    size = body.get("size")
    if size in {"1024x1024", "1024x1536", "1536x1024"}:
        result["size"] = size
    return result


def _image_from_response(response: dict[str, Any]) -> dict[str, str] | None:
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "image_generation_call":
            continue
        result = item.get("result")
        if not isinstance(result, str) or not result:
            continue
        image: dict[str, str] = {"b64_json": result}
        revised_prompt = item.get("revised_prompt")
        if isinstance(revised_prompt, str) and revised_prompt:
            image["revised_prompt"] = revised_prompt
        return image
    return None


def _image_output_format(body: dict[str, Any]) -> str:
    output_format = body.get("output_format")
    if isinstance(output_format, str) and output_format:
        return "jpeg" if output_format == "jpg" else output_format
    return "png"


def _input_item_id(item: Any, index: int) -> str:
    if isinstance(item, dict):
        item_id = item.get("id") or item.get("call_id")
        if item_id:
            return str(item_id)
    return f"input_{index}"


def _response_input_page_items(items: list[Any]) -> list[Any]:
    return [
        _response_input_page_item(item, index)
        for index, item in enumerate(copy.deepcopy(items))
    ]


def _items_after_cursor(items: list[Any], after: str) -> list[Any]:
    for index, item in enumerate(items):
        if _input_item_id(item, index) == after:
            return items[index + 1 :]
    return items


def _messages_after_cursor(items: list[dict[str, Any]], after: str) -> list[dict[str, Any]]:
    for index, item in enumerate(items):
        if str(item.get("id") or f"message_{index}") == after:
            return items[index + 1 :]
    return items


def _cursor_page(items: list[dict[str, Any]], *, has_more: bool) -> dict[str, Any]:
    return {
        "object": "list",
        "data": items,
        "first_id": str(items[0].get("id")) if items else None,
        "last_id": str(items[-1].get("id")) if items else None,
        "has_more": has_more,
    }


def _metadata_query_params(request: Request) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key.startswith("metadata[") and key.endswith("]"):
            metadata[key[len("metadata[") : -1]] = value
        elif key == "metadata":
            parsed = _parse_metadata_query_value(value)
            if parsed:
                metadata.update(parsed)
    return metadata


def _parse_metadata_query_value(value: str) -> dict[str, Any]:
    if not value:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return {str(key): parsed[key] for key in parsed}
    return {}


def _estimate_input_tokens(prepared: dict[str, Any]) -> int:
    text_parts = [str(prepared.get("instructions") or "")]
    text_parts.extend(_text_values(prepared.get("input")))
    text_parts.extend(_text_values(prepared.get("tools")))
    text_parts.extend(_text_values(prepared.get("text")))
    text = " ".join(part for part in text_parts if part)
    image_count = _typed_item_count(prepared.get("input"), "input_image")
    file_count = _typed_item_count(prepared.get("input"), "input_file")
    char_tokens = max(1, (len(text) + 3) // 4)
    word_tokens = len([part for part in text.replace("\n", " ").split(" ") if part])
    return max(char_tokens, word_tokens) + image_count * 85 + file_count * 85


def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"text", "content", "instructions", "name", "description"}:
                values.extend(_text_values(item))
            elif isinstance(item, dict | list):
                values.extend(_text_values(item))
        return values
    if isinstance(value, list):
        for item in value:
            values.extend(_text_values(item))
        return values
    return []


def _typed_item_count(value: Any, item_type: str) -> int:
    if isinstance(value, dict):
        return int(value.get("type") == item_type) + sum(
            _typed_item_count(item, item_type) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_typed_item_count(item, item_type) for item in value)
    return 0


def _response_input_page_item(item: Any, index: int) -> Any:
    if not isinstance(item, dict):
        return {
            "id": f"input_{index}",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": str(item)}],
        }

    item_type = item.get("type")
    role = item.get("role")
    if role in {"user", "system", "developer"}:
        return {
            "id": _input_item_id(item, index),
            "type": "message",
            "role": role,
            "status": item.get("status") or "completed",
            "content": _response_input_content_list(item.get("content")),
        }
    if role == "assistant":
        page_item = {
            "id": _input_item_id(item, index),
            "type": "message",
            "role": "assistant",
            "status": item.get("status") or "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": _message_content_text(item.get("content")),
                    "annotations": [],
                }
            ],
        }
        if item.get("phase"):
            page_item["phase"] = item["phase"]
        return page_item
    if item_type == "function_call":
        page_item = copy.deepcopy(item)
        page_item.setdefault("id", _input_item_id(page_item, index))
        page_item.setdefault("status", "completed")
        return page_item
    if item_type == "function_call_output":
        page_item = copy.deepcopy(item)
        page_item.setdefault("id", _input_item_id(page_item, index))
        page_item.setdefault("status", "completed")
        return page_item
    return copy.deepcopy(item)


def _response_input_content_list(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return [{"type": "input_text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]
    return [
        part
        for part in (copy.deepcopy(part) for part in content)
        if isinstance(part, dict)
    ]


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"text", "input_text", "output_text"}:
            texts.append(str(part.get("text") or ""))
    return "".join(texts)


async def _close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _openai_error(
    status_code: int,
    message: str,
    *,
    param: str | None = None,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _unexpected_error_message() -> str:
    return "Internal server error."


@dataclass
class _ChatStreamState:
    chat_id: str
    created: int
    model: str
    choice_count: int = 1
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
    backend_semaphore: asyncio.Semaphore | None,
) -> AsyncIterator[bytes]:
    output_items: list[dict[str, Any]] = []
    final_response_stored = False
    sequence_number = 0

    try:
        async with _backend_semaphore_slot(backend_semaphore):
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
    except Exception as exc:
        LOGGER.error(
            "responses.stream.unhandled_error type=%s message=%s",
            exc.__class__.__name__,
            redact_sensitive_text(str(exc)),
        )
        yield _sse_data(
            {
                "type": "error",
                "sequence_number": sequence_number,
                "code": None,
                "message": _unexpected_error_message(),
                "param": None,
            }
        )

    yield b"data: [DONE]\n\n"


async def _stored_response_event_stream(
    response: dict[str, Any],
) -> AsyncIterator[bytes]:
    stored_response = copy.deepcopy(response)
    created_response = copy.deepcopy(stored_response)
    created_response["status"] = "in_progress"
    created_response["output"] = []

    yield _sse_data(
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": created_response,
        }
    )
    sequence_number = 1
    for output_index, item in enumerate(stored_response.get("output") or []):
        if not isinstance(item, dict):
            continue
        yield _sse_data(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence_number,
                "output_index": output_index,
                "item": item,
            }
        )
        sequence_number += 1
    yield _sse_data(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": stored_response,
        }
    )
    yield b"data: [DONE]\n\n"


async def _chat_completion_event_stream(
    *,
    backend: CodexBackend,
    response_payload: dict[str, Any],
    chat_payload: dict[str, Any],
    legacy_functions: bool = False,
    chat_store: ChatCompletionStore | None = None,
    backend_semaphore: asyncio.Semaphore | None = None,
) -> AsyncIterator[bytes]:
    created = int(time.time())
    state = _ChatStreamState(
        chat_id=f"chatcmpl_{created}",
        created=created,
        model=str(response_payload.get("model") or DEFAULT_MODEL),
        choice_count=_chat_choice_count(chat_payload.get("n")),
    )
    include_usage = bool(
        isinstance(chat_payload.get("stream_options"), dict)
        and chat_payload["stream_options"].get("include_usage")
    )
    emitted_tool_arguments: dict[int, str] = {}
    output_items: list[dict[str, Any]] = []

    try:
        async with _backend_semaphore_slot(backend_semaphore):
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
                        if legacy_functions:
                            delta = {
                                "function_call": {
                                    "name": item.get("name") or "",
                                    "arguments": "",
                                }
                            }
                        else:
                            delta = {
                                "tool_calls": [
                                    _chat_tool_call_delta(
                                        item,
                                        index=output_index,
                                        arguments="",
                                        include_identity=True,
                                    )
                                ]
                            }
                        yield _sse_data(
                            _chat_chunk(state, delta)
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
                    if legacy_functions:
                        chunk_delta = {"function_call": {"arguments": delta}}
                    else:
                        chunk_delta = {
                            "tool_calls": [
                                {
                                    "index": output_index,
                                    "function": {"arguments": delta},
                                }
                            ]
                        }
                    yield _sse_data(
                        _chat_chunk(state, chunk_delta)
                    )
                    continue

                if event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict):
                        output_items.append(item)
                        async for chunk in _chat_output_item_done_chunks(
                            state,
                            item,
                            event,
                            emitted_tool_arguments,
                            legacy_functions=legacy_functions,
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
                        if output_items and not response.get("output"):
                            response["output"] = output_items
                        response = ensure_response_defaults(
                            response, request_payload=response_payload
                        )
                        _update_chat_stream_state(state, response)
                        async for chunk in _ensure_chat_role_chunk(state):
                            yield chunk
                        completion = response_to_chat_completion(
                            response,
                            fallback_model=state.model,
                            legacy_functions=legacy_functions,
                            n=state.choice_count,
                        )
                        if chat_payload.get("metadata") is not None:
                            completion["metadata"] = copy.deepcopy(chat_payload["metadata"])
                        if chat_payload.get("store") is True and chat_store is not None:
                            chat_store.remember(
                                str(completion["id"]),
                                completion=completion,
                                metadata=chat_payload.get("metadata") or {},
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
    except Exception as exc:
        LOGGER.error(
            "chat.completions.stream.unhandled_error type=%s message=%s",
            exc.__class__.__name__,
            redact_sensitive_text(str(exc)),
        )
        yield _sse_data(
            {
                "error": {
                    "message": _unexpected_error_message(),
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
    *,
    legacy_functions: bool = False,
) -> AsyncIterator[bytes]:
    if item.get("type") == "function_call":
        output_index = int(event.get("output_index") or 0)
        if output_index not in emitted_tool_arguments:
            async for chunk in _ensure_chat_role_chunk(state):
                yield chunk
            emitted_tool_arguments[output_index] = str(item.get("arguments") or "")
            if legacy_functions:
                delta = {
                    "function_call": {
                        "name": item.get("name") or "",
                        "arguments": emitted_tool_arguments[output_index],
                    }
                }
            else:
                delta = {
                    "tool_calls": [
                        _chat_tool_call_delta(
                            item,
                            index=output_index,
                            arguments=emitted_tool_arguments[output_index],
                            include_identity=True,
                        )
                    ]
                }
            yield _sse_data(
                _chat_chunk(state, delta)
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
                "index": index,
                "delta": copy.deepcopy(delta),
                "finish_reason": finish_reason,
                "logprobs": None,
            }
            for index in range(state.choice_count)
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


_app_instance: FastAPI | None = None


def __getattr__(name: str) -> Any:
    if name == "app":
        global _app_instance
        if _app_instance is None:
            _app_instance = create_app()
        return _app_instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
