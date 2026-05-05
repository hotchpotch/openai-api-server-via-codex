from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from openai_api_server_via_codex.server import create_app


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        text = "fake: " + _flatten_input_text(payload.get("input"))
        response_number = len(self.requests)
        return {
            "id": f"resp_fake_{response_number}",
            "object": "response",
            "created_at": time.time(),
            "status": "completed",
            "model": payload["model"],
            "output": [
                {
                    "id": f"msg_fake_{response_number}",
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
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        }

    async def list_models(self) -> list[str]:
        return ["gpt-5.4", "gpt-5.4-mini"]


def _flatten_input_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if not isinstance(input_value, list):
        return ""

    parts: list[str] = []
    for item in input_value:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for content_part in content:
                if isinstance(content_part, dict):
                    parts.append(content_part.get("text") or "")
    return " ".join(part for part in parts if part)


@pytest.fixture
async def openai_client_with_backend():
    backend = RecordingBackend()
    app = create_app(backend=backend)
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://testserver/v1",
        http_client=http_client,
    )
    try:
        yield client, backend
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_responses_create_round_trips_with_openai_client(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    response = await client.responses.create(
        model="gpt-5.4",
        input="Reply with PONG.",
        reasoning={"effort": "low"},
    )

    assert response.id == "resp_fake_1"
    assert response.output_text == "fake: Reply with PONG."
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert backend.requests[0]["reasoning"] == {"effort": "low"}
    assert backend.requests[0]["store"] is False


@pytest.mark.asyncio
async def test_responses_previous_response_id_expands_local_context(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    first = await client.responses.create(model="gpt-5.4", input="My name is Ada.")
    second = await client.responses.create(
        model="gpt-5.4",
        input="What name did I give?",
        previous_response_id=first.id,
    )

    assert second.output_text == (
        "fake: My name is Ada. fake: My name is Ada. What name did I give?"
    )
    assert "previous_response_id" not in backend.requests[1]
    assert backend.requests[1]["input"] == [
        {"role": "user", "content": "My name is Ada."},
        {
            "role": "assistant",
            "content": "fake: My name is Ada.",
            "phase": "final_answer",
        },
        {"role": "user", "content": "What name did I give?"},
    ]


@pytest.mark.asyncio
async def test_chat_completions_translate_messages_images_and_reasoning(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    completion = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": "You are terse."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one line."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo=",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        reasoning_effort="high",
        max_completion_tokens=20,
    )

    assert completion.object == "chat.completion"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == (
        "fake: Describe this image in one line."
    )
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 3
    assert completion.usage.completion_tokens == 5

    request = backend.requests[0]
    assert request["instructions"] == "You are terse."
    assert request["reasoning"] == {"effort": "high"}
    assert request["max_output_tokens"] == 20
    assert request["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image in one line."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                    "detail": "low",
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_models_list_uses_codex_models(openai_client_with_backend):
    client, _backend = openai_client_with_backend

    models = await client.models.list()

    assert [model.id for model in models.data] == ["gpt-5.4", "gpt-5.4-mini"]
