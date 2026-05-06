from __future__ import annotations

import asyncio
import copy
import logging
import platform
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI

from .auth import BorrowKeyError, CodexAuthConfig, borrow_codex_key
from .redaction import install_redacting_filter, redact_sensitive_text


LOGGER = logging.getLogger("openai_api_server_via_codex.backend")
install_redacting_filter(LOGGER)
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_BACKEND_HTTP = "codex-http"
DEFAULT_MODELS = [
    "gpt-5.1",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
]
CODEX_REASONING_INCLUDE = "reasoning.encrypted_content"
CODEX_RESPONSE_STATUSES = {
    "completed",
    "incomplete",
    "failed",
    "cancelled",
    "queued",
    "in_progress",
}


class CodexBackend(Protocol):
    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a Responses API response through Codex."""

    def stream_response(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Stream Responses API events through Codex."""

    async def list_models(self) -> list[str]:
        """Return model ids exposed by Codex."""


class CodexBackendError(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class CodexHttpBackend:
    def __init__(
        self,
        *,
        base_url: str = CODEX_BASE_URL,
        client_version: str = "1.0.0",
        timeout: float = 300.0,
        auth_config: CodexAuthConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_version = client_version
        self.timeout = timeout
        self.auth_config = auth_config or CodexAuthConfig()

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        stream = self.stream_response(payload)
        return await _collect_streamed_response(stream, payload)

    async def stream_response(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        LOGGER.info(
            "codex-http.stream.start model=%s input_items=%d tools=%d base_url=%s timeout=%s",
            payload.get("model"),
            _list_len(payload.get("input")),
            _list_len(payload.get("tools")),
            self.base_url,
            self.timeout,
        )
        token, account_id = await self._borrow_key()
        codex_payload = _prepare_codex_payload(payload)
        request_id = codex_payload.get("prompt_cache_key")
        headers = self._headers(
            account_id,
            client_version=self.client_version,
            request_id=request_id if isinstance(request_id, str) else None,
            event_stream=True,
        )
        client = AsyncOpenAI(
            api_key=token,
            base_url=self.base_url,
            default_headers=headers,
            timeout=self.timeout,
        )
        event_count = 0
        try:
            stream = await client.responses.create(**codex_payload)
            async for event in stream:
                dumped = _normalize_codex_stream_event(_dump_openai_model(event))
                event_count += 1
                LOGGER.debug(
                    "codex-http.stream.event type=%s sequence_number=%s",
                    dumped.get("type"),
                    dumped.get("sequence_number"),
                )
                yield dumped
        except APIStatusError as exc:
            message = _status_error_message(exc)
            LOGGER.warning(
                "codex-http.stream.status_error status=%s message=%s",
                exc.status_code,
                message,
            )
            raise CodexBackendError(
                message, status_code=exc.status_code
            ) from exc
        except APIError as exc:
            message = redact_sensitive_text(str(exc))
            LOGGER.warning("codex-http.stream.api_error message=%s", message)
            raise CodexBackendError(message) from exc
        finally:
            LOGGER.info(
                "codex-http.stream.end model=%s events=%d",
                payload.get("model"),
                event_count,
            )
            await client.close()

    async def list_models(self) -> list[str]:
        try:
            token, account_id = await self._borrow_key()
        except CodexBackendError:
            LOGGER.info("codex-http.models.fallback reason=auth_unavailable")
            return DEFAULT_MODELS

        headers = self._headers(account_id, client_version=self.client_version)
        headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    params={"client_version": self.client_version},
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            LOGGER.info("codex-http.models.fallback reason=%s", exc)
            return DEFAULT_MODELS

        models = [
            str(model["slug"])
            for model in data.get("models", [])
            if isinstance(model, dict)
            and model.get("slug")
            and model.get("supported_in_api")
            and model.get("visibility") == "list"
        ]
        if not models:
            LOGGER.info("codex-http.models.fallback reason=empty_model_list")
            return DEFAULT_MODELS
        LOGGER.info("codex-http.models.loaded count=%d", len(models))
        return models

    async def _borrow_key(self) -> tuple[str, str | None]:
        try:
            token, account_id = await asyncio.to_thread(
                borrow_codex_key, self.auth_config.auth_json
            )
            LOGGER.debug(
                "codex-http.auth.borrowed auth_json=%s account_id_present=%s",
                self.auth_config.auth_json,
                bool(account_id),
            )
            return token, account_id
        except BorrowKeyError as exc:
            message = redact_sensitive_text(str(exc))
            LOGGER.warning("codex-http.auth.error message=%s", message)
            raise CodexBackendError(message, status_code=401) from exc

    @staticmethod
    def _headers(
        account_id: str | None,
        *,
        client_version: str,
        request_id: str | None = None,
        event_stream: bool = False,
    ) -> dict[str, str]:
        headers = {
            "originator": "openai-api-server-via-codex",
            "User-Agent": _user_agent(client_version),
        }
        if event_stream:
            headers["OpenAI-Beta"] = "responses=experimental"
            headers["Accept"] = "text/event-stream"
            headers["Content-Type"] = "application/json"
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        if request_id:
            headers["session_id"] = request_id
            headers["x-client-request-id"] = request_id
        return headers


def _dump_openai_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return value
    raise CodexBackendError(f"Unsupported Codex response type: {type(value)!r}")


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _prepare_codex_payload(payload: dict[str, Any]) -> dict[str, Any]:
    codex_payload = copy.deepcopy(payload)
    codex_payload.pop("max_output_tokens", None)
    codex_payload["stream"] = True
    codex_payload["store"] = False
    codex_payload.setdefault("tool_choice", "auto")
    codex_payload.setdefault("parallel_tool_calls", True)

    text = codex_payload.get("text")
    text_config = dict(text) if isinstance(text, dict) else {}
    text_config.setdefault("verbosity", "low")
    codex_payload["text"] = text_config

    include = codex_payload.get("include")
    include_values = list(include) if isinstance(include, list) else []
    if CODEX_REASONING_INCLUDE not in include_values:
        include_values.append(CODEX_REASONING_INCLUDE)
    codex_payload["include"] = include_values
    return codex_payload


def _normalize_codex_stream_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(event)
    if normalized.get("type") == "response.done":
        normalized["type"] = "response.completed"

    response = normalized.get("response")
    if isinstance(response, dict):
        status = response.get("status")
        if isinstance(status, str) and status not in CODEX_RESPONSE_STATUSES:
            response.pop("status", None)
    return normalized


async def _collect_streamed_response(
    stream: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    text_parts: list[str] = []
    output_items: list[dict[str, Any]] = []
    completed_response: dict[str, Any] | None = None
    last_response_id: str | None = None

    async for event in stream:
        normalized = (
            _normalize_codex_stream_event(event)
            if isinstance(event, dict)
            else _normalize_codex_stream_event(_dump_openai_model(event))
        )
        event_type = _event_value(normalized, "type")
        if event_type == "response.created":
            response = _event_value(normalized, "response")
            if response is not None:
                dumped = _dump_openai_model(response)
                last_response_id = str(dumped.get("id") or last_response_id or "")
        elif event_type == "response.output_text.delta":
            delta = _event_value(normalized, "delta")
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type == "response.output_item.done":
            item = _event_value(normalized, "item")
            if item is not None:
                output_items.append(_dump_openai_model(item))
        elif event_type in {"response.completed", "response.incomplete"}:
            response = _event_value(normalized, "response")
            if response is not None:
                completed_response = _dump_openai_model(response)

    if completed_response is not None:
        if output_items and not completed_response.get("output"):
            completed_response["output"] = output_items
        return completed_response

    created_at = time.time()
    text = "".join(text_parts)
    return {
        "id": last_response_id or f"resp_{int(created_at * 1000)}",
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": payload.get("model"),
        "output": [
            {
                "id": f"msg_{int(created_at * 1000)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": payload.get("tool_choice") or "auto",
        "tools": payload.get("tools") or [],
    }


def _status_error_message(exc: APIStatusError) -> str:
    try:
        body = exc.response.json()
    except Exception:
        return redact_sensitive_text(exc.message)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("type") or "")
            if exc.status_code == 429 or (
                "usage_limit" in code or "rate_limit" in code
            ):
                plan = str(error.get("plan_type") or "").lower()
                plan_text = f" ({plan} plan)" if plan else ""
                reset_text = _reset_time_text(error.get("resets_at"))
                return f"You have hit your ChatGPT usage limit{plan_text}.{reset_text}"
            if error.get("message"):
                return redact_sensitive_text(str(error["message"]))
    return redact_sensitive_text(exc.message)


def _reset_time_text(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    minutes = max(0, round((float(value) - time.time()) / 60))
    return f" Try again in ~{minutes} min."


def _user_agent(client_version: str) -> str:
    return (
        f"openai-api-server-via-codex/{client_version} "
        f"({platform.system().lower()} {platform.release()}; {platform.machine()})"
    )
