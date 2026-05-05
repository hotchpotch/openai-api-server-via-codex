from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI

from .auth import BorrowKeyError, borrow_codex_key


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


class CodexBackend(Protocol):
    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a Responses API response through Codex."""

    async def list_models(self) -> list[str]:
        """Return model ids exposed by Codex."""


class CodexBackendError(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICodexBackend:
    def __init__(
        self,
        *,
        base_url: str = CODEX_BASE_URL,
        client_version: str = "1.0.0",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_version = client_version
        self.timeout = timeout

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        token, account_id = await self._borrow_key()
        headers = self._headers(account_id)
        client = AsyncOpenAI(
            api_key=token,
            base_url=self.base_url,
            default_headers=headers,
            timeout=self.timeout,
        )
        codex_payload = dict(payload)
        codex_payload["stream"] = True
        try:
            stream = await client.responses.create(**codex_payload)
            response = await _collect_streamed_response(stream, payload)
        except APIStatusError as exc:
            raise CodexBackendError(
                _status_error_message(exc), status_code=exc.status_code
            ) from exc
        except APIError as exc:
            raise CodexBackendError(str(exc)) from exc
        finally:
            await client.close()
        return response

    async def list_models(self) -> list[str]:
        try:
            token, account_id = await self._borrow_key()
        except CodexBackendError:
            return DEFAULT_MODELS

        headers = self._headers(account_id)
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
        except Exception:
            return DEFAULT_MODELS

        models = [
            str(model["slug"])
            for model in data.get("models", [])
            if isinstance(model, dict)
            and model.get("slug")
            and model.get("supported_in_api")
            and model.get("visibility") == "list"
        ]
        return models or DEFAULT_MODELS

    async def _borrow_key(self) -> tuple[str, str | None]:
        try:
            return await asyncio.to_thread(borrow_codex_key)
        except BorrowKeyError as exc:
            raise CodexBackendError(str(exc), status_code=401) from exc

    @staticmethod
    def _headers(account_id: str | None) -> dict[str, str]:
        if not account_id:
            return {}
        return {"ChatGPT-Account-ID": account_id}


def _dump_openai_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return value
    raise CodexBackendError(f"Unsupported Codex response type: {type(value)!r}")


async def _collect_streamed_response(
    stream: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    text_parts: list[str] = []
    output_items: list[dict[str, Any]] = []
    completed_response: dict[str, Any] | None = None
    last_response_id: str | None = None

    async for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "response.created":
            response = getattr(event, "response", None)
            if response is not None:
                dumped = _dump_openai_model(response)
                last_response_id = str(dumped.get("id") or last_response_id or "")
        elif event_type == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is not None:
                output_items.append(_dump_openai_model(item))
        elif event_type == "response.completed":
            response = getattr(event, "response", None)
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
        return exc.message
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return exc.message
