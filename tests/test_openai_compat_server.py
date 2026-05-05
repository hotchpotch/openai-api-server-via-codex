from __future__ import annotations

import logging
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
        return self._response_for_payload(payload)

    async def stream_response(self, payload: dict[str, Any]):
        self.requests.append(payload)
        response = self._response_for_payload(payload)
        text = _response_output_text(response)
        midpoint = max(1, len(text) // 2)
        first_delta = text[:midpoint]
        second_delta = text[midpoint:]
        message = response["output"][0]

        yield {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": response["id"],
                "object": "response",
                "created_at": response["created_at"],
                "status": "in_progress",
                "model": response["model"],
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        yield {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "output_index": 0,
            "content_index": 0,
            "item_id": message["id"],
            "delta": first_delta,
            "logprobs": [],
        }
        yield {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "output_index": 0,
            "content_index": 0,
            "item_id": message["id"],
            "delta": second_delta,
            "logprobs": [],
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 0,
            "item": message,
        }
        yield {
            "type": "response.completed",
            "sequence_number": 4,
            "response": response,
        }

    def _response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
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


class ToolCallStreamingBackend(RecordingBackend):
    async def stream_response(self, payload: dict[str, Any]):
        self.requests.append(payload)
        created_at = time.time()
        item = {
            "id": "fc_fake_1",
            "type": "function_call",
            "call_id": "call_fake_1",
            "name": "lookup_weather",
            "arguments": '{"city":"Tokyo"}',
            "status": "completed",
        }
        response = {
            "id": "resp_tool_1",
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": payload["model"],
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": payload.get("tools") or [],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
        }

        yield {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": response["id"],
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": payload["model"],
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": payload.get("tools") or [],
            },
        }
        yield {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"},
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": item["id"],
            "delta": '{"city"',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": item["id"],
            "delta": ':"Tokyo"}',
        }
        yield {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": item,
        }
        yield {
            "type": "response.completed",
            "sequence_number": 5,
            "response": response,
        }


class ToolCallResponseBackend(RecordingBackend):
    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        created_at = time.time()
        return {
            "id": "resp_tool_1",
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": payload["model"],
            "output": [
                {
                    "id": "fc_fake_1",
                    "type": "function_call",
                    "call_id": "call_fake_1",
                    "name": "lookup_weather",
                    "arguments": '{"city":"Tokyo"}',
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": payload.get("tool_choice") or "auto",
            "tools": payload.get("tools") or [],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
        }


class MixedTextToolCallResponseBackend(RecordingBackend):
    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        created_at = time.time()
        return {
            "id": "resp_mixed_1",
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": payload["model"],
            "output": [
                {
                    "id": "msg_preamble_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "phase": "preamble",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I will check the weather.",
                            "annotations": [],
                        }
                    ],
                },
                {
                    "id": "fc_fake_1",
                    "type": "function_call",
                    "call_id": "call_fake_1",
                    "name": "lookup_weather",
                    "arguments": '{"city":"Tokyo"}',
                    "status": "completed",
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": payload.get("tool_choice") or "auto",
            "tools": payload.get("tools") or [],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 6,
                "total_tokens": 13,
            },
        }


class NativeSessionRecordingBackend(RecordingBackend):
    supports_native_sessions = True


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


def _response_output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for output in response.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


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
async def test_verbose_app_logs_request_and_response_summary(caplog):
    backend = RecordingBackend()
    app = create_app(backend=backend, verbose=True)
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
        with caplog.at_level(logging.DEBUG, logger="openai_api_server_via_codex"):
            await client.responses.create(
                model="gpt-5.4",
                input="Log this request.",
                reasoning={"effort": "low"},
            )
    finally:
        await http_client.aclose()

    log_text = caplog.text
    assert "request.start method=POST path=/v1/responses" in log_text
    assert "responses.create model=gpt-5.4 stream=False input_items=1" in log_text
    assert "backend=RecordingBackend" in log_text
    assert "request.end method=POST path=/v1/responses status=200" in log_text


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
async def test_responses_previous_response_id_preserves_function_calls_for_tool_outputs():
    backend = ToolCallResponseBackend()
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
        first = await client.responses.create(
            model="gpt-5.4",
            input="Call the weather tool for Tokyo.",
        )
        await client.responses.create(
            model="gpt-5.4",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_fake_1",
                    "output": "Tokyo is sunny.",
                }
            ],
            previous_response_id=first.id,
        )

        assert "previous_response_id" not in backend.requests[1]
        assert backend.requests[1]["input"] == [
            {"role": "user", "content": "Call the weather tool for Tokyo."},
            {
                "type": "function_call",
                "call_id": "call_fake_1",
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_fake_1",
                "output": "Tokyo is sunny.",
            },
        ]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_responses_create_sanitizes_output_items_used_as_manual_context(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    await client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "user", "content": "Remember Ada."},
            {
                "id": "rs_previous",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
            },
            {
                "id": "msg_previous",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will remember Ada.",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_previous",
                "type": "function_call",
                "call_id": "call_previous",
                "name": "lookup_name",
                "arguments": '{"name":"Ada"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_previous",
                "output": "Ada is stored.",
                "status": "completed",
            },
            {"role": "user", "content": "What did you remember?"},
        ],
    )

    assert backend.requests[0]["input"] == [
        {"role": "user", "content": "Remember Ada."},
        {
            "role": "assistant",
            "content": "I will remember Ada.",
            "phase": "final_answer",
        },
        {
            "type": "function_call",
            "call_id": "call_previous",
            "name": "lookup_name",
            "arguments": '{"name":"Ada"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_previous",
            "output": "Ada is stored.",
        },
        {"role": "user", "content": "What did you remember?"},
    ]


@pytest.mark.asyncio
async def test_responses_retrieve_and_input_items_list_use_local_store(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    created = await client.responses.create(
        model="gpt-5.4",
        input="Remember retrieve-marker.",
    )
    retrieved = await client.responses.retrieve(created.id)
    input_items_page = await client.responses.input_items.list(created.id)

    assert retrieved.id == created.id
    assert retrieved.output_text == created.output_text
    assert input_items_page.object == "list"
    assert input_items_page.data[0].model_dump(mode="json", exclude_none=True) == {
        "id": "input_0",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_text", "text": "Remember retrieve-marker."}],
    }


@pytest.mark.asyncio
async def test_responses_retrieve_can_stream_stored_response_events(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    created = await client.responses.create(
        model="gpt-5.4",
        input="Stream retrieve-marker from the local store.",
    )
    stream = await client.responses.retrieve(created.id, stream=True)

    event_types: list[str] = []
    completed_response_id: str | None = None
    output_item_text: str | None = None
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_item.done":
            item = getattr(event, "item")
            output_item_text = _response_output_text({"output": [item.model_dump()]})
        elif event.type == "response.completed":
            completed_response_id = getattr(event, "response").id

    assert event_types == [
        "response.created",
        "response.output_item.done",
        "response.completed",
    ]
    assert completed_response_id == created.id
    assert output_item_text == created.output_text


@pytest.mark.asyncio
async def test_responses_delete_removes_stored_response(openai_client_with_backend):
    client, _backend = openai_client_with_backend

    created = await client.responses.create(
        model="gpt-5.4",
        input="Delete this response from the local store.",
    )

    assert await client.responses.delete(created.id) is None
    with pytest.raises(Exception) as exc_info:
        await client.responses.retrieve(created.id)

    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_responses_cancel_completed_response_returns_openai_error(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    created = await client.responses.create(
        model="gpt-5.4",
        input="Completed responses cannot be cancelled.",
    )

    with pytest.raises(Exception) as exc_info:
        await client.responses.cancel(created.id)

    assert getattr(exc_info.value, "status_code", None) == 409


@pytest.mark.asyncio
async def test_responses_input_tokens_count_uses_openai_client_shape(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    count = await client.responses.input_tokens.count(
        model="gpt-5.4",
        instructions="Count this test input.",
        input=[
            {"role": "user", "content": "alpha beta gamma"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "delta"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                ],
            },
        ],
        reasoning={"effort": "low"},
    )

    assert count.object == "response.input_tokens"
    assert count.input_tokens > 0


@pytest.mark.asyncio
async def test_responses_input_items_list_supports_basic_cursor_params(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    created = await client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "user", "content": "first input item"},
            {"role": "user", "content": "second input item"},
        ],
    )
    first_page = await client.responses.input_items.list(created.id, limit=1)
    next_page = await client.responses.input_items.list(created.id, after="input_0")
    desc_page = await client.responses.input_items.list(created.id, limit=1, order="desc")

    assert [item.id for item in first_page.data] == ["input_0"]
    assert first_page.has_more is True
    assert [item.id for item in next_page.data] == ["input_1"]
    assert next_page.has_more is False
    assert [item.id for item in desc_page.data] == ["input_1"]
    assert desc_page.has_more is True


@pytest.mark.asyncio
async def test_responses_retrieve_unknown_id_returns_openai_error(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    with pytest.raises(Exception) as exc_info:
        await client.responses.retrieve("resp_missing")

    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_responses_previous_response_id_is_forwarded_to_native_session_backend():
    backend = NativeSessionRecordingBackend()
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
        first = await client.responses.create(model="gpt-5.4", input="My name is Ada.")
        second = await client.responses.create(
            model="gpt-5.4",
            input="What name did I give?",
            previous_response_id=first.id,
        )
    finally:
        await http_client.aclose()

    assert second.previous_response_id == first.id
    assert backend.requests[1]["previous_response_id"] == first.id
    assert backend.requests[1]["input"] == [
        {"role": "user", "content": "What name did I give?"}
    ]


@pytest.mark.asyncio
async def test_responses_create_streams_openai_events_and_stores_context(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    stream = await client.responses.create(
        model="gpt-5.4",
        input="Stream PONG.",
        stream=True,
        reasoning={"effort": "low"},
    )

    event_types: list[str] = []
    text_parts: list[str] = []
    completed_response_id: str | None = None
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_text.delta":
            text_parts.append(str(getattr(event, "delta")))
        elif event.type == "response.completed":
            response = getattr(event, "response")
            completed_response_id = response.id

    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert "".join(text_parts) == "fake: Stream PONG."
    assert completed_response_id == "resp_fake_1"
    assert backend.requests[0]["reasoning"] == {"effort": "low"}

    followup = await client.responses.create(
        model="gpt-5.4",
        input="What did I ask you to stream?",
        previous_response_id=completed_response_id,
    )
    assert followup.output_text == (
        "fake: Stream PONG. fake: Stream PONG. What did I ask you to stream?"
    )


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
async def test_chat_completions_create_streams_openai_chunks(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    stream = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Stream chat PONG."},
        ],
        stream=True,
        stream_options={"include_usage": True},
        reasoning_effort="low",
    )

    roles: list[str] = []
    content_parts: list[str] = []
    finish_reasons: list[str] = []
    usage_total_tokens: int | None = None
    async for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.role:
                roles.append(choice.delta.role)
            if choice.delta.content:
                content_parts.append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)
        if chunk.usage:
            usage_total_tokens = chunk.usage.total_tokens

    assert roles == ["assistant"]
    assert "".join(content_parts) == "fake: Stream chat PONG."
    assert finish_reasons == ["stop"]
    assert usage_total_tokens == 8
    assert backend.requests[0]["instructions"] == "You are terse."
    assert backend.requests[0]["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_chat_completions_streaming_n_duplicates_choice_chunks(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    stream = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Stream two choices."}],
        stream=True,
        n=2,
        reasoning_effort="low",
    )

    roles: dict[int, list[str]] = {0: [], 1: []}
    content_parts: dict[int, list[str]] = {0: [], 1: []}
    finish_reasons: dict[int, list[str]] = {0: [], 1: []}
    async for chunk in stream:
        for choice in chunk.choices:
            if choice.delta.role:
                roles[choice.index].append(choice.delta.role)
            if choice.delta.content:
                content_parts[choice.index].append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons[choice.index].append(choice.finish_reason)

    assert roles == {0: ["assistant"], 1: ["assistant"]}
    assert {
        index: "".join(parts) for index, parts in content_parts.items()
    } == {
        0: "fake: Stream two choices.",
        1: "fake: Stream two choices.",
    }
    assert finish_reasons == {0: ["stop"], 1: ["stop"]}


@pytest.mark.asyncio
async def test_chat_completions_create_returns_tool_calls_without_streaming():
    backend = ToolCallResponseBackend()
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
        completion = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather tool for Tokyo."}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        message = completion.choices[0].message
        assert message.content is None
        assert message.tool_calls is not None
        tool_call = message.tool_calls[0]
        function = getattr(tool_call, "function")
        assert tool_call.id == "call_fake_1"
        assert tool_call.type == "function"
        assert function.name == "lookup_weather"
        assert function.arguments == '{"city":"Tokyo"}'
        assert completion.choices[0].finish_reason == "tool_calls"
        assert backend.requests[0]["tools"][0]["name"] == "lookup_weather"
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_preserves_tool_preamble_text_without_streaming():
    backend = MixedTextToolCallResponseBackend()
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
        completion = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather tool for Tokyo."}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        message = completion.choices[0].message
        assert message.content == "I will check the weather."
        assert message.tool_calls is not None
        tool_call = message.tool_calls[0].model_dump(mode="json")
        assert tool_call["function"]["name"] == "lookup_weather"
        assert completion.choices[0].finish_reason == "tool_calls"
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_n_duplicates_choices_for_compatibility(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    completion = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Reply once."}],
        n=3,
    )

    assert [choice.index for choice in completion.choices] == [0, 1, 2]
    assert [
        choice.message.content for choice in completion.choices
    ] == ["fake: Reply once."] * 3


@pytest.mark.asyncio
async def test_chat_completions_legacy_functions_translate_to_response_tools():
    backend = ToolCallResponseBackend()
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
        completion = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather function for Tokyo."}
            ],
            functions=[
                {
                    "name": "lookup_weather",
                    "description": "Look up weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            function_call={"name": "lookup_weather"},
        )

        message = completion.choices[0].message
        assert message.content is None
        assert message.function_call is not None
        assert message.function_call.name == "lookup_weather"
        assert message.function_call.arguments == '{"city":"Tokyo"}'
        assert message.tool_calls is None
        assert completion.choices[0].finish_reason == "function_call"
        assert backend.requests[0]["tools"] == [
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Look up weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": False,
            }
        ]
        assert backend.requests[0]["tool_choice"] == {
            "type": "function",
            "name": "lookup_weather",
        }
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_with_tool_result_preserves_assistant_tool_call():
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
        await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather tool for Tokyo."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fake_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": '{"city":"Tokyo"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_fake_1",
                    "content": "Tokyo is sunny.",
                },
                {"role": "user", "content": "Summarize the tool result."},
            ],
        )

        assert backend.requests[0]["input"] == [
            {"role": "user", "content": "Call the weather tool for Tokyo."},
            {
                "type": "function_call",
                "call_id": "call_fake_1",
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_fake_1",
                "output": "Tokyo is sunny.",
            },
            {"role": "user", "content": "Summarize the tool result."},
        ]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_with_legacy_function_result_preserves_function_call():
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
        await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather function for Tokyo."},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Tokyo"}',
                    },
                },
                {
                    "role": "function",
                    "name": "lookup_weather",
                    "content": "Tokyo is sunny.",
                },
                {"role": "user", "content": "Summarize the function result."},
            ],
        )

        assert backend.requests[0]["input"] == [
            {"role": "user", "content": "Call the weather function for Tokyo."},
            {
                "type": "function_call",
                "call_id": "lookup_weather",
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
            {
                "type": "function_call_output",
                "call_id": "lookup_weather",
                "output": "Tokyo is sunny.",
            },
            {"role": "user", "content": "Summarize the function result."},
        ]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_response_format_maps_to_responses_text_config(
    openai_client_with_backend,
):
    client, backend = openai_client_with_backend

    completion = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "user", "content": "Return a JSON object with answer=yes."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "compat_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0.2,
        top_p=0.9,
        metadata={"case": "response-format"},
        user="compat-user",
    )

    assert backend.requests[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "compat_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    }
    assert backend.requests[0]["temperature"] == 0.2
    assert backend.requests[0]["top_p"] == 0.9
    assert "metadata" not in backend.requests[0]
    assert backend.requests[0]["user"] == "compat-user"
    assert completion.model_dump(mode="json")["metadata"] == {"case": "response-format"}


@pytest.mark.asyncio
async def test_chat_completions_store_true_supports_sdk_lifecycle(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    completion = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Remember stored-chat-marker."}],
        metadata={"suite": "stored-chat"},
        store=True,
    )
    retrieved = await client.chat.completions.retrieve(completion.id)
    listed_page = await client.chat.completions.list(
        limit=10,
        metadata={"suite": "stored-chat"},
        model="gpt-5.4",
    )
    messages_page = await client.chat.completions.messages.list(completion.id)
    updated = await client.chat.completions.update(
        completion.id,
        metadata={"suite": "stored-chat-updated"},
    )
    deleted = await client.chat.completions.delete(completion.id)

    assert retrieved.id == completion.id
    assert retrieved.choices[0].message.content == "fake: Remember stored-chat-marker."
    assert [item.id for item in listed_page.data] == [completion.id]
    assert listed_page.has_more is False
    assert messages_page.object == "list"
    assert [message.role for message in messages_page.data] == ["assistant"]
    assert messages_page.data[0].content == "fake: Remember stored-chat-marker."
    assert updated.model_dump(mode="json")["metadata"] == {
        "suite": "stored-chat-updated"
    }
    assert deleted.id == completion.id
    assert deleted.deleted is True
    assert deleted.object == "chat.completion.deleted"

    with pytest.raises(Exception) as exc_info:
        await client.chat.completions.retrieve(completion.id)
    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_chat_completions_store_false_is_not_retrievable_or_listed(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    completion = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Do not store this completion."}],
        metadata={"suite": "not-stored-chat"},
    )
    listed_page = await client.chat.completions.list(
        limit=10,
        metadata={"suite": "not-stored-chat"},
    )

    assert listed_page.data == []
    with pytest.raises(Exception) as exc_info:
        await client.chat.completions.retrieve(completion.id)

    assert getattr(exc_info.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_chat_completions_store_true_stream_is_retrievable(
    openai_client_with_backend,
):
    client, _backend = openai_client_with_backend

    stream = await client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Stream stored chat marker."}],
        stream=True,
        store=True,
    )

    chunk_id: str | None = None
    text_parts: list[str] = []
    async for chunk in stream:
        chunk_id = chunk.id
        for choice in chunk.choices:
            if choice.delta.content:
                text_parts.append(choice.delta.content)

    assert chunk_id is not None
    retrieved = await client.chat.completions.retrieve(chunk_id)
    assert retrieved.choices[0].message.content == "".join(text_parts)


@pytest.mark.asyncio
async def test_chat_completions_streams_tool_call_chunks():
    backend = ToolCallStreamingBackend()
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
        stream = await client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "user", "content": "Call the weather tool for Tokyo."}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Look up weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            stream=True,
        )

        tool_call_id: str | None = None
        tool_name: str | None = None
        argument_parts: list[str] = []
        finish_reasons: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                tool_call_id = tool_call.id or tool_call_id
                if tool_call.function:
                    tool_name = tool_call.function.name or tool_name
                    if tool_call.function.arguments:
                        argument_parts.append(tool_call.function.arguments)
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        assert tool_call_id == "call_fake_1"
        assert tool_name == "lookup_weather"
        assert "".join(argument_parts) == '{"city":"Tokyo"}'
        assert finish_reasons == ["tool_calls"]
        assert backend.requests[0]["tools"][0]["name"] == "lookup_weather"
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_models_list_uses_codex_models(openai_client_with_backend):
    client, _backend = openai_client_with_backend

    models = await client.models.list()

    assert [model.id for model in models.data] == ["gpt-5.4", "gpt-5.4-mini"]
