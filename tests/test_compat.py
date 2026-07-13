from __future__ import annotations

from openai_api_server_via_codex.compat import ensure_response_defaults


def test_ensure_response_defaults_uses_request_parallel_tool_calls_default() -> None:
    response = ensure_response_defaults(
        {"id": "resp_1", "output": []},
        request_payload={"model": "gpt-5.4-mini", "input": "hello"},
    )

    assert response["parallel_tool_calls"] is False


def test_ensure_response_defaults_preserves_request_parallel_tool_calls() -> None:
    response = ensure_response_defaults(
        {"id": "resp_1", "output": []},
        request_payload={
            "model": "gpt-5.4-mini",
            "input": "hello",
            "parallel_tool_calls": True,
        },
    )

    assert response["parallel_tool_calls"] is True


def test_ensure_response_defaults_preserves_backend_parallel_tool_calls() -> None:
    response = ensure_response_defaults(
        {"id": "resp_1", "output": [], "parallel_tool_calls": True},
        request_payload={"model": "gpt-5.4-mini", "input": "hello"},
    )

    assert response["parallel_tool_calls"] is True
