from __future__ import annotations

import httpx
from openai import APIStatusError

from openai_api_server_via_codex.backend import (
    DEFAULT_MODELS,
    _prepare_codex_payload,
    _status_error_message,
    _normalize_codex_stream_event,
)


def test_prepare_codex_payload_adds_codex_http_defaults_without_overwriting() -> None:
    payload = {
        "model": "gpt-5.4-mini",
        "input": [{"role": "user", "content": "hello"}],
        "stream": False,
        "store": True,
        "text": {"format": {"type": "json_object"}},
        "include": ["file_search_call.results"],
        "tool_choice": "none",
        "parallel_tool_calls": False,
    }

    prepared = _prepare_codex_payload(payload)

    assert prepared["stream"] is True
    assert prepared["store"] is False
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


def test_prepare_codex_payload_adds_missing_tool_and_reasoning_defaults() -> None:
    prepared = _prepare_codex_payload(
        {
            "model": "gpt-5.4-mini",
            "input": [{"role": "user", "content": "hello"}],
        }
    )

    assert prepared["stream"] is True
    assert prepared["store"] is False
    assert prepared["tool_choice"] == "auto"
    assert prepared["parallel_tool_calls"] is True
    assert prepared["text"] == {"verbosity": "low"}
    assert prepared["include"] == ["reasoning.encrypted_content"]


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
    ]


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
