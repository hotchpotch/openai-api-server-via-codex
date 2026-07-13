from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIStatusError

import openai_api_server_via_codex.backend as backend_module
from openai_api_server_via_codex.backend import (
    CodexBackendError,
    CodexHttpBackend,
    DEFAULT_MODELS,
    RESPONSES_LITE_MODELS,
    _collect_streamed_response,
    _forward_proxy_request_headers,
    _forward_proxy_response_headers,
    _normalize_codex_stream_event,
    _prepare_codex_payload,
    _resolve_proxy_url,
    _resolve_transcribe_url,
    _status_error_message,
)


def test_prepare_codex_payload_adds_codex_http_defaults_without_overwriting() -> None:
    payload = {
        "model": "gpt-5.4-mini",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "store": True,
        "max_output_tokens": 20,
        "text": {"format": {"type": "json_object"}},
        "include": ["file_search_call.results"],
        "tool_choice": "none",
        "parallel_tool_calls": False,
    }

    prepared = _prepare_codex_payload(payload)

    assert prepared["stream"] is True
    assert prepared["store"] is False
    assert "max_output_tokens" not in prepared
    assert prepared["tool_choice"] == "none"
    assert prepared["parallel_tool_calls"] is False
    assert prepared["text"] == {
        "format": {"type": "json_object"},
        "verbosity": "low",
    }
    assert prepared["include"] == [
        "file_search_call.results",
        "reasoning.encrypted_content",
    ]
    assert payload["stream"] is False
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert payload["max_output_tokens"] == 20


def test_prepare_codex_payload_defaults_parallel_tool_calls_off() -> None:
    prepared = _prepare_codex_payload(
        {
            "model": "gpt-5.4-mini",
            "input": [{"role": "user", "content": "hello"}],
        }
    )

    assert prepared["stream"] is True
    assert prepared["store"] is False
    assert prepared["tool_choice"] == "auto"
    assert prepared["parallel_tool_calls"] is False
    assert prepared["text"] == {"verbosity": "low"}
    assert prepared["include"] == ["reasoning.encrypted_content"]


def test_prepare_codex_payload_preserves_explicit_parallel_tool_calls_for_regular_models() -> None:
    prepared = _prepare_codex_payload(
        {
            "model": "gpt-5.4-mini",
            "input": [{"role": "user", "content": "hello"}],
            "parallel_tool_calls": True,
        }
    )

    assert prepared["parallel_tool_calls"] is True


def test_prepare_codex_payload_uses_responses_lite_defaults_for_gpt_5_6_models() -> None:
    for model in RESPONSES_LITE_MODELS:
        prepared = _prepare_codex_payload(
            {
                "model": model,
                "input": [{"role": "user", "content": "hello"}],
            }
        )

        assert prepared["model"] == model
        assert prepared["stream"] is True
        assert prepared["store"] is False
        assert prepared["tool_choice"] == "auto"
        assert prepared["parallel_tool_calls"] is False
        assert prepared["reasoning"] == {"context": "all_turns"}
        assert prepared["include"] == ["reasoning.encrypted_content"]


def test_prepare_codex_payload_preserves_responses_lite_reasoning_options() -> None:
    prepared = _prepare_codex_payload(
        {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "hello"}],
            "reasoning": {"effort": "medium"},
        }
    )

    assert prepared["reasoning"] == {"effort": "medium", "context": "all_turns"}


def test_prepare_codex_payload_forces_responses_lite_reasoning_context() -> None:
    prepared = _prepare_codex_payload(
        {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "hello"}],
            "reasoning": {"effort": "medium", "context": "last_turn"},
        }
    )

    assert prepared["reasoning"] == {"effort": "medium", "context": "all_turns"}


def test_responses_lite_models_are_the_exact_gpt_5_6_lite_set() -> None:
    assert RESPONSES_LITE_MODELS == {
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }


def test_headers_add_responses_lite_flag_only_when_requested() -> None:
    regular_headers = CodexHttpBackend._headers(
        "account-1",
        client_version="1.2.3",
        event_stream=True,
    )
    lite_headers = CodexHttpBackend._headers(
        "account-1",
        client_version="1.2.3",
        event_stream=True,
        responses_lite=True,
    )

    assert "x-openai-internal-codex-responses-lite" not in regular_headers
    assert lite_headers["x-openai-internal-codex-responses-lite"] == "true"


async def test_collect_streamed_response_uses_request_parallel_tool_calls_default() -> None:
    async def stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}

    response = await _collect_streamed_response(
        stream(), {"model": "gpt-5.4-mini", "input": "hello"}
    )

    assert response["parallel_tool_calls"] is False


async def test_collect_streamed_response_preserves_request_parallel_tool_calls() -> None:
    async def stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}

    response = await _collect_streamed_response(
        stream(),
        {"model": "gpt-5.4-mini", "input": "hello", "parallel_tool_calls": True},
    )

    assert response["parallel_tool_calls"] is True


async def test_stream_response_sends_responses_lite_header_and_payload(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponses:
        async def create(self, **payload):
            captured["payload"] = payload

            async def stream():
                yield {"type": "response.completed", "response": {"id": "resp_1"}}

            return stream()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            captured["closed"] = True

    backend = CodexHttpBackend()

    async def fake_borrow_key() -> tuple[str, str | None]:
        return "token-1", "account-1"

    monkeypatch.setattr(backend, "_borrow_key", fake_borrow_key)
    monkeypatch.setattr(backend_module, "AsyncOpenAI", FakeAsyncOpenAI)

    events = [
        event
        async for event in backend.stream_response(
            {
                "model": "gpt-5.6-luna",
                "input": "hello",
                "parallel_tool_calls": True,
                "reasoning": {"effort": "medium", "context": "last_turn"},
            }
        )
    ]

    assert events == [{"type": "response.completed", "response": {"id": "resp_1"}}]
    assert captured["payload"]["model"] == "gpt-5.6-luna"
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["payload"]["reasoning"] == {
        "effort": "medium",
        "context": "all_turns",
    }
    assert (
        captured["client_kwargs"]["default_headers"][
            "x-openai-internal-codex-responses-lite"
        ]
        == "true"
    )
    assert captured["closed"] is True


def test_default_models_match_codex_http_fallback_catalog() -> None:
    assert DEFAULT_MODELS == [
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
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ]


def test_codex_http_backend_default_timeout_is_300_seconds() -> None:
    assert CodexHttpBackend().timeout == 300.0


def test_forward_proxy_request_headers_keeps_only_safe_openai_headers() -> None:
    headers = {
        "Authorization": "Bearer local-secret",
        "Cookie": "session=local",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "OpenAI-Beta": "responses=experimental",
        "Idempotency-Key": "idem-1",
        "X-Should-Not-Forward": "nope",
        "Host": "127.0.0.1:18080",
        "Content-Length": "17",
    }

    forwarded = _forward_proxy_request_headers(headers)

    assert forwarded == {
        "accept": "application/json",
        "content-type": "application/json",
        "openai-beta": "responses=experimental",
        "idempotency-key": "idem-1",
    }


def test_resolve_proxy_url_builds_url_under_base() -> None:
    url = _resolve_proxy_url(
        "https://chatgpt.com/backend-api/codex",
        "tokenizer",
        b"model=gpt-5.4&input=one",
    )

    assert str(url) == (
        "https://chatgpt.com/backend-api/codex/tokenizer?model=gpt-5.4&input=one"
    )


def test_resolve_transcribe_url_uses_backend_api_sibling() -> None:
    url = _resolve_transcribe_url("https://chatgpt.com/backend-api/codex")

    assert str(url) == "https://chatgpt.com/backend-api/transcribe"


def test_resolve_transcribe_url_allows_non_codex_base() -> None:
    url = _resolve_transcribe_url("https://example.test/backend-api")

    assert str(url) == "https://example.test/backend-api/transcribe"


def test_resolve_proxy_url_strips_redundant_slashes_and_dot_segments() -> None:
    url = _resolve_proxy_url(
        "https://chatgpt.com/backend-api/codex",
        "//responses/./compact",
        b"",
    )

    assert str(url) == "https://chatgpt.com/backend-api/codex/responses/compact"


def test_resolve_proxy_url_rejects_dotdot_segment() -> None:
    with pytest.raises(CodexBackendError) as excinfo:
        _resolve_proxy_url(
            "https://chatgpt.com/backend-api/codex",
            "../auth/me",
            b"",
        )

    assert excinfo.value.status_code == 400


def test_resolve_proxy_url_rejects_dotdot_after_normal_segment() -> None:
    with pytest.raises(CodexBackendError) as excinfo:
        _resolve_proxy_url(
            "https://chatgpt.com/backend-api/codex",
            "x/../auth/me",
            b"",
        )

    assert excinfo.value.status_code == 400


def test_resolve_proxy_url_rejects_dotdot_terminal_segment() -> None:
    with pytest.raises(CodexBackendError) as excinfo:
        _resolve_proxy_url(
            "https://chatgpt.com/backend-api/codex",
            "deep/nest/..",
            b"",
        )

    assert excinfo.value.status_code == 400


@pytest.mark.parametrize(
    "encoded",
    [
        "%2e%2e/auth/me",
        "x/%2e%2e/auth",
        "%2E%2E/auth",
        "x%2f..%2fy",
        "x%2f%2e%2e%2fy",
        "%2e./auth",
        ".%2e/auth",
    ],
)
def test_resolve_proxy_url_rejects_percent_encoded_traversal(encoded: str) -> None:
    with pytest.raises(CodexBackendError) as excinfo:
        _resolve_proxy_url(
            "https://chatgpt.com/backend-api/codex",
            encoded,
            b"",
        )

    assert excinfo.value.status_code == 400


def test_proxy_request_rejects_invalid_path_before_borrowing_codex_key(
    monkeypatch,
) -> None:
    import asyncio

    backend = CodexHttpBackend()

    async def _should_not_borrow() -> tuple[str, str | None]:
        raise AssertionError("_borrow_key must not run for invalid proxy paths")

    monkeypatch.setattr(backend, "_borrow_key", _should_not_borrow)

    with pytest.raises(CodexBackendError) as excinfo:
        asyncio.run(
            backend.proxy_request(
                "GET",
                "%2e%2e/auth/me",
                query=b"",
                headers={},
                body=b"",
            )
        )

    assert excinfo.value.status_code == 400


def test_forward_proxy_response_headers_drops_hop_by_hop_and_cookie_headers() -> None:
    headers = {
        "content-type": "application/json",
        "x-request-id": "upstream-1",
        "set-cookie": "session=secret",
        "content-length": "999",
        "content-encoding": "gzip",
        "transfer-encoding": "chunked",
        "connection": "close",
    }

    forwarded = _forward_proxy_response_headers(headers)

    assert forwarded == {
        "content-type": "application/json",
        "x-request-id": "upstream-1",
    }


def test_normalize_codex_stream_event_maps_response_done_to_completed() -> None:
    event = {
        "type": "response.done",
        "sequence_number": 7,
        "response": {"id": "resp_1", "status": "completed"},
    }

    normalized = _normalize_codex_stream_event(event)

    assert normalized == {
        "type": "response.completed",
        "sequence_number": 7,
        "response": {"id": "resp_1", "status": "completed"},
    }


def test_normalize_codex_stream_event_drops_unknown_status() -> None:
    event = {
        "type": "response.done",
        "response": {"id": "resp_1", "status": "mystery"},
    }

    normalized = _normalize_codex_stream_event(event)

    assert normalized["response"] == {"id": "resp_1"}


def test_status_error_message_formats_chatgpt_usage_limit() -> None:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        429,
        request=request,
        json={
            "error": {
                "code": "usage_limit_reached",
                "message": "backend raw message",
                "plan_type": "PLUS",
                "resets_at": 1_800_000_000,
            }
        },
    )
    exc = APIStatusError("rate limited", response=response, body=response.json())

    message = _status_error_message(exc)

    assert "ChatGPT usage limit" in message
    assert "plus plan" in message


def test_status_error_message_redacts_auth_values() -> None:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(
        500,
        request=request,
        json={
            "error": {
                "message": (
                    "upstream failed with Authorization: Bearer "
                    "abcdefghijklmnopqrstuvwxyz and access_token="
                    "tok_abcdefghijklmnopqrstuvwxyz"
                )
            }
        },
    )
    exc = APIStatusError("failed", response=response, body=response.json())

    message = _status_error_message(exc)

    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "Bearer abcdef******" in message
    assert "access_token=tok_ab******" in message
