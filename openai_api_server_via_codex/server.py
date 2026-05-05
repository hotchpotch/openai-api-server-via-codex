from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
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

from .app_server import (
    CODEX_APP_SERVER_CWD_ENV,
    CODEX_BACKEND_APP_SERVER,
    CODEX_BIN_ENV,
    CodexAppServerBackend,
    CodexAppServerConfig,
)
from .auth import AUTH_JSON_ENV, CodexAuthConfig
from .backend import (
    CODEX_BACKEND_HTTP,
    CODEX_BASE_URL,
    CodexBackend,
    CodexBackendError,
    CodexHttpBackend,
)
from . import config as config_module
from .compat import (
    ChatCompletionStore,
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
    resolve_daemon_paths,
    start_background,
    stop_background,
)


class OpenAICompatRequest(BaseModel):
    model: str | None = None
    stream: bool | None = False

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class ServerSettings:
    backend: str
    host: str
    port: int
    backend_base_url: str
    client_version: str
    timeout: float
    default_model: str
    codex_bin: str | None = None
    app_server_cwd: Path | None = None
    auth_json: Path | None = None


ConfigData = dict[str, Any]


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def create_app(
    *,
    backend: CodexBackend | None = None,
    default_model: str | None = None,
    backend_name: str | None = None,
    auth_json: str | Path | None = None,
    backend_base_url: str | None = None,
    client_version: str | None = None,
    timeout: float | None = None,
    codex_bin: str | None = None,
    app_server_cwd: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _close_backend(getattr(app.state, "backend", None))

    app = FastAPI(title="OpenAI API Server via Codex", lifespan=lifespan)
    selected_backend = backend_name or os.environ.get(
        "OPENAI_VIA_CODEX_BACKEND", CODEX_BACKEND_HTTP
    )
    selected_timeout = (
        timeout
        if timeout is not None
        else float(os.environ.get("OPENAI_VIA_CODEX_TIMEOUT", "180"))
    )
    selected_auth_config = CodexAuthConfig(
        auth_json=_resolve_optional_path(auth_json or os.environ.get(AUTH_JSON_ENV))
    )
    app.state.backend = backend or _build_backend(
        backend=selected_backend,
        backend_base_url=backend_base_url
        or os.environ.get("OPENAI_VIA_CODEX_BACKEND_BASE_URL", CODEX_BASE_URL),
        client_version=client_version
        or os.environ.get("OPENAI_VIA_CODEX_CLIENT_VERSION", "1.0.0"),
        timeout=selected_timeout,
        auth_config=selected_auth_config,
        codex_bin=codex_bin or os.environ.get(CODEX_BIN_ENV),
        app_server_cwd=app_server_cwd or os.environ.get(CODEX_APP_SERVER_CWD_ENV),
    )
    app.state.response_store = ResponseStore()
    app.state.chat_completion_store = ChatCompletionStore()
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
        backend = _get_backend(request)
        native_sessions = _backend_supports_native_sessions(backend)
        previous_response_id = prepared.get("previous_response_id")
        if previous_response_id and not native_sessions:
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
                if native_sessions or key != "previous_response_id"
            }
            return _sse_response(
                _responses_event_stream(
                    backend=backend,
                    store=store,
                    prepared=prepared,
                    backend_payload=backend_payload,
                    previous_response_id=previous_response_id,
                )
            )

        backend_payload = {
            key: value
            for key, value in prepared.items()
            if native_sessions or key != "previous_response_id"
        }
        try:
            response = await backend.create_response(backend_payload)
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

    @app.post("/v1/responses/input_tokens", response_model=None)
    async def count_response_input_tokens(
        request: Request, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        prepared = prepare_response_payload(
            body, default_model=request.app.state.default_model
        )
        previous_response_id = prepared.get("previous_response_id")
        if previous_response_id and not _backend_supports_native_sessions(
            _get_backend(request)
        ):
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
        if payload.get("stream"):
            return _sse_response(
                _chat_completion_event_stream(
                    backend=_get_backend(request),
                    response_payload=response_payload,
                    chat_payload=payload,
                    legacy_functions=legacy_functions,
                    chat_store=_get_chat_completion_store(request),
                )
            )

        try:
            response = await _get_backend(request).create_response(response_payload)
        except CodexBackendError as exc:
            return _openai_error(exc.status_code, str(exc), error_type="api_error")
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
        return JSONResponse(chat_completion)

    @app.get("/v1/chat/completions/{completion_id}", response_model=None)
    async def retrieve_chat_completion(
        request: Request, completion_id: str
    ) -> JSONResponse:
        stored = _get_chat_completion_store(request).get(completion_id)
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

    return app


def _build_backend(
    *,
    backend: str,
    backend_base_url: str,
    client_version: str,
    timeout: float,
    auth_config: CodexAuthConfig,
    codex_bin: str | None,
    app_server_cwd: str | Path | None,
) -> CodexBackend:
    if backend == CODEX_BACKEND_APP_SERVER:
        return CodexAppServerBackend(
            config=CodexAppServerConfig(
                codex_bin=codex_bin or "codex",
                cwd=_resolve_optional_path(app_server_cwd),
                timeout=timeout,
                auth_config=auth_config,
            )
        )
    if backend != CODEX_BACKEND_HTTP:
        raise ValueError(
            "OPENAI_VIA_CODEX_BACKEND must be 'codex-http' or 'codex-app-server'."
        )
    return CodexHttpBackend(
        base_url=backend_base_url,
        client_version=client_version,
        timeout=timeout,
        auth_config=auth_config,
    )


app = create_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in {"-h", "--help"}):
        argv = ["serve", *argv]

    parser = argparse.ArgumentParser(
        prog="openai-api-server-via-codex",
        description="OpenAI-compatible API server backed by Codex credentials.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the server in the foreground")
    _add_config_option(serve)
    _add_server_options(serve)

    start = subparsers.add_parser("start", help="run the server in the background")
    _add_config_option(start)
    _add_server_options(start)
    _add_daemon_options(start)

    stop = subparsers.add_parser("stop", help="stop the background server")
    _add_config_option(stop)
    _add_daemon_selector_options(stop)
    stop.add_argument(
        "--stop-timeout",
        type=float,
        default=None,
        help="seconds to wait after SIGTERM before SIGKILL",
    )

    status = subparsers.add_parser("status", help="show background server status")
    _add_config_option(status)
    _add_daemon_selector_options(status)

    config_generate = subparsers.add_parser(
        "config-generate", help="write a default TOML configuration"
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
        backend=_arg_env_config_str(
            args,
            "backend",
            "OPENAI_VIA_CODEX_BACKEND",
            config_data,
            "server",
            "backend",
            CODEX_BACKEND_HTTP,
        ),
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
        codex_bin=_arg_env_config_optional_str(
            args, "codex_bin", CODEX_BIN_ENV, config_data, "codex", "codex_bin"
        ),
        app_server_cwd=_resolve_optional_path(
            _arg_env_config_optional_str(
                args,
                "app_server_cwd",
                CODEX_APP_SERVER_CWD_ENV,
                config_data,
                "codex",
                "app_server_cwd",
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
        "--backend",
        settings.backend,
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
        "--default-model",
        settings.default_model,
    ]
    if settings.codex_bin is not None:
        command.extend(["--codex-bin", settings.codex_bin])
    if settings.app_server_cwd is not None:
        command.extend(["--app-server-cwd", str(settings.app_server_cwd)])
    if settings.auth_json is not None:
        command.extend(["--auth-json", str(settings.auth_json)])
    return command


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
        uvicorn.run(
            create_app(
                default_model=settings.default_model,
                backend_name=settings.backend,
                auth_json=settings.auth_json,
                backend_base_url=settings.backend_base_url,
                client_version=settings.client_version,
                timeout=settings.timeout,
                codex_bin=settings.codex_bin,
                app_server_cwd=settings.app_server_cwd,
            ),
            host=settings.host,
            port=settings.port,
            reload=False,
        )
        return 0

    settings = server_settings_from_args(args, loaded_config)
    paths = daemon_paths_from_args(args, settings, loaded_config)

    if args.command == "start":
        try:
            pid = start_background(serve_command(settings), paths)
        except DaemonError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Started openai-api-server-via-codex on {settings.host}:{settings.port}")
        print(f"PID: {pid}")
        print(f"PID file: {paths.pid_file}")
        print(f"Log file: {paths.log_file}")
        return 0

    if args.command == "stop":
        result = stop_background(paths, timeout=stop_timeout_from_args(args, loaded_config))
        if result.state == "not_running":
            print(f"Not running. PID file: {paths.pid_file}")
        elif result.state == "stale":
            print(f"Removed stale PID {result.pid}. PID file: {paths.pid_file}")
        else:
            print(f"Stopped PID {result.pid} ({result.state}).")
        return 0

    if args.command == "status":
        status = daemon_status(paths)
        if status.pid is None:
            print(f"stopped. PID file: {status.pid_file}")
        else:
            print(f"{status.state}: PID {status.pid}")
            print(f"PID file: {status.pid_file}")
            print(f"Log file: {status.log_file}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=(CODEX_BACKEND_HTTP, CODEX_BACKEND_APP_SERVER),
        help="backend implementation",
    )
    parser.add_argument("--host", help="server host")
    parser.add_argument("--port", type=int, help="server port")
    parser.add_argument("--auth-json", help="path to Codex auth.json")
    parser.add_argument("--backend-base-url", help="Codex backend base URL")
    parser.add_argument("--client-version", help="Codex client_version parameter")
    parser.add_argument("--timeout", type=float, help="backend timeout in seconds")
    parser.add_argument("--default-model", help="default model when request omits model")
    parser.add_argument("--codex-bin", help="codex binary used by app-server backend")
    parser.add_argument(
        "--app-server-cwd",
        help="working directory used by the Codex app-server backend",
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


def _get_backend(request: Request) -> CodexBackend:
    return request.app.state.backend


def _get_response_store(request: Request) -> ResponseStore:
    return request.app.state.response_store


def _get_chat_completion_store(request: Request) -> ChatCompletionStore:
    return request.app.state.chat_completion_store


def _backend_supports_native_sessions(backend: CodexBackend) -> bool:
    return bool(getattr(backend, "supports_native_sessions", False))


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
    return max(1, char_tokens, word_tokens) + image_count * 85 + file_count * 85


def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key, item in value.items():
            if key in {"text", "content", "instructions", "name", "description"}:
                values.extend(_text_values(item))
            elif isinstance(item, (dict, list)):
                values.extend(_text_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
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


if __name__ == "__main__":
    main()
