from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI


ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
RED_SQUARE_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJ0lEQVR42u3NsQkAAAjAsP7/tF7hIASyp6lTCQQCgUAgEAgEgi/BAjLD/C5w/SM9AAAAAElFTkSuQmCC"
)
LIVE_LONG_TEST_MODEL = "gpt-5.4-mini"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to call real Codex backends.",
)
async def test_live_http_and_app_server_backends_support_same_openai_calls() -> None:
    http_server = await _start_server("chatgpt-http")
    app_server = await _start_server("codex-app-server")
    try:
        http_client = AsyncOpenAI(api_key="test", base_url=f"{http_server.base_url}/v1")
        app_client = AsyncOpenAI(api_key="test", base_url=f"{app_server.base_url}/v1")
        try:
            http_models = await http_client.models.list()
            app_models = await app_client.models.list()
            model = _select_model(
                [model.id for model in http_models.data],
                [model.id for model in app_models.data],
            )

            async for client in _clients(http_client, app_client):
                first = await client.responses.create(
                    model=model,
                    input="Reply with exactly: dual-backend-ok",
                    reasoning={"effort": "low"},
                )
                assert first.output_text.strip()

                followup = await client.responses.create(
                    model=model,
                    input="What exact token did you just output?",
                    previous_response_id=first.id,
                    reasoning={"effort": "low"},
                )
                assert followup.output_text.strip()

                response_stream = await client.responses.create(
                    model=model,
                    input="Stream a short reply with the token: dual-stream-ok",
                    stream=True,
                    reasoning={"effort": "low"},
                )
                response_stream_text: list[str] = []
                async for event in response_stream:
                    if event.type == "response.output_text.delta":
                        response_stream_text.append(str(getattr(event, "delta")))
                assert "".join(response_stream_text).strip()

                chat = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Answer with concise text only."},
                        {"role": "user", "content": "Reply with exactly: dual-chat-ok"},
                    ],
                    reasoning_effort="low",
                )
                assert chat.choices[0].message.content

                image_chat = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Look at this image and answer with one short sentence.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": ONE_PIXEL_PNG_DATA_URL,
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                    reasoning_effort="low",
                )
                assert image_chat.choices[0].message.content
        finally:
            await http_client.close()
            await app_client.close()
    finally:
        await http_server.stop()
        await app_server.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to call real Codex backends.",
)
async def test_live_dual_backends_handle_images_multi_turn_and_reasoning() -> None:
    http_server = await _start_server("chatgpt-http")
    app_server = await _start_server("codex-app-server")
    try:
        http_client = AsyncOpenAI(api_key="test", base_url=f"{http_server.base_url}/v1")
        app_client = AsyncOpenAI(api_key="test", base_url=f"{app_server.base_url}/v1")
        try:
            http_models = await http_client.models.list()
            app_models = await app_client.models.list()
            model = _select_model(
                [model.id for model in http_models.data],
                [model.id for model in app_models.data],
            )

            async for backend_name, client in _named_clients(http_client, app_client):
                await _assert_responses_multi_turn_with_reasoning(
                    client, model, backend_name
                )
                await _assert_chat_multi_turn_with_reasoning(
                    client, model, backend_name
                )
                await _assert_image_inputs(client, model, backend_name)
        finally:
            await http_client.close()
            await app_client.close()
    finally:
        await http_server.stop()
        await app_server.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to call real Codex backends.",
)
async def test_live_dual_backends_handle_long_multi_turn_tool_and_text_flows() -> None:
    http_server = await _start_server("chatgpt-http")
    app_server = await _start_server("codex-app-server")
    try:
        http_client = AsyncOpenAI(api_key="test", base_url=f"{http_server.base_url}/v1")
        app_client = AsyncOpenAI(api_key="test", base_url=f"{app_server.base_url}/v1")
        try:
            await _assert_model_list_contains(
                http_client,
                app_client,
                LIVE_LONG_TEST_MODEL,
            )

            async for backend_name, client in _named_clients(http_client, app_client):
                await _assert_responses_multi_turn_tool_calling(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_multi_turn_tool_calling(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_long_plain_text_responses_conversation(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
        finally:
            await http_client.close()
            await app_client.close()
    finally:
        await http_server.stop()
        await app_server.stop()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to call real Codex backends.",
)
async def test_live_dual_backends_handle_openai_client_compatibility_matrix() -> None:
    http_server = await _start_server("chatgpt-http")
    app_server = await _start_server("codex-app-server")
    try:
        http_client = AsyncOpenAI(api_key="test", base_url=f"{http_server.base_url}/v1")
        app_client = AsyncOpenAI(api_key="test", base_url=f"{app_server.base_url}/v1")
        try:
            await _assert_model_list_contains(
                http_client,
                app_client,
                LIVE_LONG_TEST_MODEL,
            )

            async for backend_name, client in _named_clients(http_client, app_client):
                await _assert_responses_retrieve_and_input_items(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_responses_auxiliary_sdk_methods(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_responses_manual_context_without_previous_response_id(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_responses_json_object_format(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_responses_structured_outputs(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_multiple_choices(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_stored_completion_lifecycle(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_json_object_format(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_structured_outputs(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_tool_choice_none_text_only(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_legacy_functions(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_stream_multiple_choices(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_stream_include_usage(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_responses_streaming_tool_call(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )
                await _assert_chat_streaming_tool_call(
                    client,
                    LIVE_LONG_TEST_MODEL,
                    backend_name,
                )

            await asyncio.to_thread(
                _assert_sync_client_smoke,
                http_server.base_url,
                LIVE_LONG_TEST_MODEL,
                "chatgpt-http",
            )
            await asyncio.to_thread(
                _assert_sync_client_smoke,
                app_server.base_url,
                LIVE_LONG_TEST_MODEL,
                "codex-app-server",
            )
        finally:
            await http_client.close()
            await app_client.close()
    finally:
        await http_server.stop()
        await app_server.stop()


class _RunningServer:
    def __init__(self, process: subprocess.Popen[str], base_url: str) -> None:
        self.process = process
        self.base_url = base_url

    async def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


async def _start_server(backend: str) -> _RunningServer:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openai_api_server_via_codex.server",
            "serve",
            "--backend",
            backend,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    server = _RunningServer(process, base_url)
    try:
        await _wait_for_server(base_url)
    except Exception:
        await server.stop()
        raise
    return server


async def _clients(
    http_client: AsyncOpenAI, app_client: AsyncOpenAI
) -> AsyncIterator[AsyncOpenAI]:
    yield http_client
    yield app_client


async def _named_clients(
    http_client: AsyncOpenAI, app_client: AsyncOpenAI
) -> AsyncIterator[tuple[str, AsyncOpenAI]]:
    yield "chatgpt-http", http_client
    yield "codex-app-server", app_client


async def _assert_responses_multi_turn_with_reasoning(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-RESP-MARKER-731"
    first = await client.responses.create(
        model=model,
        instructions=(
            "Preserve exact marker strings. When asked later, return the marker "
            "verbatim."
        ),
        input=f"Remember this exact marker: {marker}. Reply with exactly: ready",
        reasoning={"effort": "low"},
    )
    assert first.output_text.strip(), backend_name

    followup = await client.responses.create(
        model=model,
        input="Return the exact marker I asked you to remember.",
        previous_response_id=first.id,
        reasoning={"effort": "high"},
    )
    assert _contains_marker(followup.output_text, marker), (
        backend_name,
        followup.output_text,
    )


async def _assert_chat_multi_turn_with_reasoning(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-CHAT-MARKER-582"
    first = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Preserve exact marker strings. When asked later, return the "
                    "marker verbatim."
                ),
            },
            {
                "role": "user",
                "content": f"Remember this exact marker: {marker}. Reply with ready.",
            },
        ],
        reasoning_effort="low",
    )
    first_message = first.choices[0].message.content
    assert first_message, backend_name

    followup = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Preserve exact marker strings. When asked later, return the "
                    "marker verbatim."
                ),
            },
            {
                "role": "user",
                "content": f"Remember this exact marker: {marker}. Reply with ready.",
            },
            {"role": "assistant", "content": first_message},
            {"role": "user", "content": "Return the exact marker I asked you to remember."},
        ],
        reasoning_effort="high",
    )
    followup_message = followup.choices[0].message.content or ""
    assert _contains_marker(followup_message, marker), (
        backend_name,
        followup_message,
    )


async def _assert_image_inputs(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    response_image = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "The image is a simple solid-color square. What is "
                            "the dominant color? Reply with one word."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": RED_SQUARE_PNG_DATA_URL,
                        "detail": "low",
                    },
                ],
            }
        ],
        reasoning={"effort": "low"},
    )
    assert _mentions_red(response_image.output_text), (
        backend_name,
        response_image.output_text,
    )

    chat_image = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The image is a simple solid-color square. What is "
                            "the dominant color? Reply with one word."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": RED_SQUARE_PNG_DATA_URL,
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        reasoning_effort="low",
    )
    chat_image_text = chat_image.choices[0].message.content or ""
    assert _mentions_red(chat_image_text), (backend_name, chat_image_text)


async def _assert_responses_multi_turn_tool_calling(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-RESP-TOOL-4242"
    tool: dict[str, Any] = {
        "type": "function",
        "name": "lookup_order_status",
        "description": "Look up an order status by order id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id to look up.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    first = await client.responses.create(
        model=model,
        instructions=(
            "Use the provided function tool when the user asks for order status. "
            "Do not invent order status data."
        ),
        input=(
            "Call lookup_order_status for order_id ORDER-4242. "
            "Do not answer from memory."
        ),
        tools=cast(Any, [tool]),
        tool_choice={"type": "function", "name": "lookup_order_status"},
        reasoning={"effort": "low"},
    )
    function_call = _first_function_call(first)
    assert function_call is not None, (backend_name, _dump_model(first))
    assert function_call["name"] == "lookup_order_status", (
        backend_name,
        function_call,
    )
    call_arguments = _json_object(function_call.get("arguments"))
    assert call_arguments.get("order_id") == "ORDER-4242", (
        backend_name,
        function_call,
    )
    print(
        f"{backend_name} responses tool call: "
        f"name={function_call['name']} args={function_call.get('arguments')}"
    )

    second = await client.responses.create(
        model=model,
        input=[
            {
                "type": "function_call_output",
                "call_id": function_call["call_id"],
                "output": (
                    f"ORDER-4242 status: shipped. Carrier: Yamato. "
                    f"Tracking marker: {marker}. Delivery window: Friday morning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using only the tool output just provided, state the order "
                    "status, carrier, and tracking marker in one sentence."
                ),
            },
        ],
        previous_response_id=first.id,
        reasoning={"effort": "low"},
    )
    second_text = second.output_text
    print(f"{backend_name} responses tool result: {_short_text(second_text)}")
    assert _contains_marker(second_text, marker), (backend_name, second_text)
    assert "shipped" in second_text.lower(), (backend_name, second_text)

    third = await client.responses.create(
        model=model,
        input="Return only the tracking marker from the tool result.",
        previous_response_id=second.id,
        reasoning={"effort": "low"},
    )
    third_text = third.output_text
    print(f"{backend_name} responses tool follow-up: {_short_text(third_text)}")
    assert _contains_marker(third_text, marker), (backend_name, third_text)


async def _assert_chat_multi_turn_tool_calling(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-CHAT-TOOL-7788"
    first = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the provided function tool when the user asks for customer "
                    "tier data. Do not invent customer tier data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Call lookup_customer_tier for customer_id CUST-7788. "
                    "Do not answer from memory."
                ),
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer_tier",
                    "description": "Look up customer tier by customer id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {
                                "type": "string",
                                "description": "The customer id to look up.",
                            }
                        },
                        "required": ["customer_id"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_customer_tier"}},
        reasoning_effort="low",
    )
    message = first.choices[0].message
    tool_calls = message.tool_calls or []
    assert tool_calls, (backend_name, first.model_dump(mode="json"))
    tool_call = tool_calls[0]
    tool_call_data = _dump_model(tool_call)
    tool_function = tool_call_data.get("function") or {}
    assert isinstance(tool_function, dict), (backend_name, tool_call_data)
    tool_call_id = str(tool_call_data.get("id") or "")
    tool_name = str(tool_function.get("name") or "")
    tool_arguments_text = str(tool_function.get("arguments") or "")
    tool_arguments = _json_object(tool_arguments_text)
    assert tool_name == "lookup_customer_tier", (
        backend_name,
        tool_call_data,
    )
    assert tool_arguments.get("customer_id") == "CUST-7788", (
        backend_name,
        tool_call_data,
    )
    print(
        f"{backend_name} chat tool call: "
        f"name={tool_name} args={tool_arguments_text}"
    )

    second = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer from supplied tool results only.",
            },
            {
                "role": "user",
                "content": (
                    "Call lookup_customer_tier for customer_id CUST-7788. "
                    "Do not answer from memory."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_arguments_text,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": (
                    f"CUST-7788 tier: platinum. Renewal marker: {marker}. "
                    "Account owner: Sato."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using only the tool result, state the customer tier and "
                    "renewal marker."
                ),
            },
        ],
        reasoning_effort="low",
    )
    second_text = second.choices[0].message.content or ""
    print(f"{backend_name} chat tool result: {_short_text(second_text)}")
    assert _contains_marker(second_text, marker), (backend_name, second_text)
    assert "platinum" in second_text.lower(), (backend_name, second_text)

    third = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Preserve marker strings exactly."},
            {
                "role": "user",
                "content": (
                    "Call lookup_customer_tier for customer_id CUST-7788. "
                    "Do not answer from memory."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_arguments_text,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": (
                    f"CUST-7788 tier: platinum. Renewal marker: {marker}. "
                    "Account owner: Sato."
                ),
            },
            {"role": "assistant", "content": second_text},
            {
                "role": "user",
                "content": "Return only the renewal marker from the tool result.",
            },
        ],
        reasoning_effort="low",
    )
    third_text = third.choices[0].message.content or ""
    print(f"{backend_name} chat tool follow-up: {_short_text(third_text)}")
    assert _contains_marker(third_text, marker), (backend_name, third_text)


async def _assert_long_plain_text_responses_conversation(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    markers = [
        f"{backend_name}-PLAIN-ALPHA-101",
        f"{backend_name}-PLAIN-BRAVO-202",
        f"{backend_name}-PLAIN-CHARLIE-303",
        f"{backend_name}-PLAIN-DELTA-404",
    ]
    first = await client.responses.create(
        model=model,
        instructions=(
            "This is a plain text memory check. Do not use tools. Preserve all "
            "marker strings exactly."
        ),
        input=(
            f"Remember marker one: {markers[0]}. "
            "Reply with a short acknowledgement."
        ),
        reasoning={"effort": "low"},
    )
    print(f"{backend_name} plain turn 1: {_short_text(first.output_text)}")

    second = await client.responses.create(
        model=model,
        input=(
            f"Add marker two: {markers[1]}. "
            "List the markers you know so far in order."
        ),
        previous_response_id=first.id,
        reasoning={"effort": "low"},
    )
    print(f"{backend_name} plain turn 2: {_short_text(second.output_text)}")
    _assert_contains_markers(second.output_text, markers[:2], backend_name)

    third = await client.responses.create(
        model=model,
        input=(
            f"Add marker three: {markers[2]}. "
            "Write one Japanese sentence that includes all known markers."
        ),
        previous_response_id=second.id,
        reasoning={"effort": "low"},
    )
    print(f"{backend_name} plain turn 3: {_short_text(third.output_text)}")
    _assert_contains_markers(third.output_text, markers[:3], backend_name)

    fourth = await client.responses.create(
        model=model,
        input=(
            f"Add marker four: {markers[3]}. "
            "Return the four markers in chronological order."
        ),
        previous_response_id=third.id,
        reasoning={"effort": "low"},
    )
    print(f"{backend_name} plain turn 4: {_short_text(fourth.output_text)}")
    _assert_contains_markers(fourth.output_text, markers, backend_name)

    final = await client.responses.create(
        model=model,
        input="Return only the four remembered markers separated by pipes.",
        previous_response_id=fourth.id,
        reasoning={"effort": "low"},
    )
    final_text = final.output_text
    print(f"{backend_name} plain final: {_short_text(final_text)}")
    _assert_contains_markers(final_text, markers, backend_name)


async def _assert_responses_retrieve_and_input_items(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-RETRIEVE-818"
    response = await client.responses.create(
        model=model,
        instructions="Return exact marker strings without modification.",
        input=[
            {"role": "user", "content": f"Remember marker {marker}."},
            {
                "role": "user",
                "content": f"Reply with one sentence containing {marker}.",
            },
        ],
        reasoning={"effort": "low"},
    )
    retrieved = await client.responses.retrieve(response.id)
    input_items = await client.responses.input_items.list(response.id, limit=1)
    next_items = await client.responses.input_items.list(response.id, after="input_0")
    desc_items = await client.responses.input_items.list(
        response.id, limit=1, order="desc"
    )

    _assert_response_api_shape(response, backend_name)
    _assert_response_api_shape(retrieved, backend_name)
    assert retrieved.id == response.id, (backend_name, retrieved.id, response.id)
    assert retrieved.output_text == response.output_text, (
        backend_name,
        retrieved.output_text,
        response.output_text,
    )
    assert _contains_marker(retrieved.output_text, marker), (
        backend_name,
        retrieved.output_text,
    )
    input_items_dump = input_items.model_dump(mode="json")
    assert input_items_dump.get("object") == "list", (backend_name, input_items_dump)
    assert [item.id for item in input_items.data] == ["input_0"], (
        backend_name,
        input_items.model_dump(mode="json"),
    )
    assert input_items.has_more is True, (backend_name, input_items)
    assert [item.id for item in next_items.data] == ["input_1"], (
        backend_name,
        next_items.model_dump(mode="json"),
    )
    assert [item.id for item in desc_items.data] == ["input_1"], (
        backend_name,
        desc_items.model_dump(mode="json"),
    )
    print(
        f"{backend_name} responses retrieve/input_items: "
        f"{_short_text(retrieved.output_text)}"
    )


async def _assert_responses_auxiliary_sdk_methods(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-RESP-AUX-626"
    response = await client.responses.create(
        model=model,
        instructions="Return exact marker strings without modification.",
        input=f"Reply with one sentence containing {marker}.",
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(response, backend_name)

    stream = await client.responses.retrieve(response.id, stream=True)
    event_types: list[str] = []
    replayed_output_text_parts: list[str] = []
    completed_response_id: str | None = None
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_item.done":
            item = getattr(event, "item", None)
            item_text = _response_item_text(_dump_model(item))
            if item_text:
                replayed_output_text_parts.append(item_text)
        elif event.type == "response.completed":
            completed_response_id = getattr(event, "response").id

    assert event_types[0] == "response.created", (backend_name, event_types)
    assert event_types[-1] == "response.completed", (backend_name, event_types)
    assert "response.output_item.done" in event_types, (backend_name, event_types)
    assert completed_response_id == response.id, (
        backend_name,
        completed_response_id,
        response.id,
    )
    replayed_output_text = "".join(replayed_output_text_parts)
    assert replayed_output_text == response.output_text, (
        backend_name,
        replayed_output_text,
        response.output_text,
    )

    token_count = await client.responses.input_tokens.count(
        model=model,
        instructions="Count marker prompts for compatibility testing.",
        input=cast(
            Any,
            [
                {"role": "user", "content": f"Count this marker: {marker}."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "plus one tiny image"},
                        {"type": "input_image", "image_url": ONE_PIXEL_PNG_DATA_URL},
                    ],
                },
            ],
        ),
        reasoning={"effort": "low"},
    )
    assert token_count.object == "response.input_tokens", (
        backend_name,
        token_count,
    )
    assert token_count.input_tokens > 0, (backend_name, token_count)

    delete_target = await client.responses.create(
        model=model,
        input=f"Create a temporary response for deletion: {marker}.",
        reasoning={"effort": "low"},
    )
    assert await client.responses.delete(delete_target.id) is None
    with pytest.raises(Exception) as exc_info:
        await client.responses.retrieve(delete_target.id)
    assert getattr(exc_info.value, "status_code", None) == 404, backend_name

    with pytest.raises(Exception) as cancel_exc_info:
        await client.responses.cancel(response.id)
    assert getattr(cancel_exc_info.value, "status_code", None) == 409, backend_name

    print(
        f"{backend_name} responses auxiliary SDK methods: "
        f"events={event_types} tokens={token_count.input_tokens} "
        f"text={_short_text(response.output_text)}"
    )


async def _assert_responses_manual_context_without_previous_response_id(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-MANUAL-CONTEXT-909"
    first_input = f"Remember exact marker {marker}. Reply with exactly: ready"
    first = await client.responses.create(
        model=model,
        instructions="Preserve exact marker strings.",
        input=first_input,
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(first, backend_name)

    manual_context = [
        {"role": "user", "content": first_input},
        *_dump_model(first).get("output", []),
        {"role": "user", "content": "Return only the exact marker."},
    ]
    second = await client.responses.create(
        model=model,
        instructions="Return marker strings exactly.",
        input=cast(Any, manual_context),
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(second, backend_name)
    print(
        f"{backend_name} responses manual context: "
        f"{_short_text(second.output_text)}"
    )
    assert _contains_marker(second.output_text, marker), (
        backend_name,
        second.output_text,
    )


async def _assert_responses_json_object_format(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    response = await client.responses.create(
        model=model,
        instructions=(
            "Return only valid JSON. Do not include markdown fences or commentary."
        ),
        input=(
            'Return exactly this JSON object: {"kind":"responses_json_object",'
            '"ok":true,"count":3}'
        ),
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(response, backend_name)
    parsed = _parse_json_text(response.output_text)
    print(f"{backend_name} responses json_object: {parsed}")
    assert parsed == {
        "kind": "responses_json_object",
        "ok": True,
        "count": 3,
    }, (backend_name, response.output_text)


async def _assert_responses_structured_outputs(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    schema = _compat_structured_schema()
    response = await client.responses.create(
        model=model,
        instructions="Return only JSON that satisfies the supplied schema.",
        input="Set code to RESP-STRUCT-204 and count to 7.",
        text={
            "format": {
                "type": "json_schema",
                "name": "compat_structured_response",
                "strict": True,
                "schema": schema,
            }
        },
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(response, backend_name)
    parsed = _parse_json_text(response.output_text)
    print(f"{backend_name} responses structured output: {parsed}")
    assert parsed == {"code": "RESP-STRUCT-204", "count": 7}, (
        backend_name,
        response.output_text,
    )


async def _assert_chat_multiple_choices(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-CHAT-N-202"
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Preserve exact marker strings.",
            },
            {
                "role": "user",
                "content": f"Reply with one sentence containing {marker}.",
            },
        ],
        n=2,
        reasoning_effort="low",
    )
    _assert_chat_api_shape(completion, backend_name)
    assert [choice.index for choice in completion.choices] == [0, 1], (
        backend_name,
        completion.model_dump(mode="json"),
    )
    texts = [choice.message.content or "" for choice in completion.choices]
    assert all(_contains_marker(text, marker) for text in texts), (
        backend_name,
        texts,
    )
    print(f"{backend_name} chat n=2 choices: {[_short_text(text) for text in texts]}")


async def _assert_chat_stored_completion_lifecycle(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-CHAT-STORE-414"
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Preserve exact marker strings.",
            },
            {"role": "user", "content": f"Reply with exactly: {marker}"},
        ],
        metadata={"backend": backend_name, "case": "live-stored-chat"},
        store=True,
        reasoning_effort="low",
    )
    _assert_chat_api_shape(completion, backend_name)
    retrieved = await client.chat.completions.retrieve(completion.id)
    listed = await client.chat.completions.list(
        limit=5,
        metadata={"backend": backend_name, "case": "live-stored-chat"},
        model=completion.model,
    )
    messages = await client.chat.completions.messages.list(completion.id)
    updated = await client.chat.completions.update(
        completion.id,
        metadata={"backend": backend_name, "case": "live-stored-chat-updated"},
    )

    retrieved_text = retrieved.choices[0].message.content or ""
    assert retrieved.id == completion.id, (backend_name, retrieved.id, completion.id)
    assert retrieved_text == (completion.choices[0].message.content or ""), (
        backend_name,
        retrieved_text,
        completion.choices[0].message.content,
    )
    assert _contains_marker(retrieved_text, marker), (backend_name, retrieved_text)
    assert [item.id for item in listed.data] == [completion.id], (
        backend_name,
        listed.model_dump(mode="json"),
    )
    assert [message.role for message in messages.data] == ["assistant"], (
        backend_name,
        messages.model_dump(mode="json"),
    )
    assert messages.data[0].content == retrieved_text, (
        backend_name,
        messages.model_dump(mode="json"),
        retrieved_text,
    )
    assert updated.model_dump(mode="json").get("metadata") == {
        "backend": backend_name,
        "case": "live-stored-chat-updated",
    }, (backend_name, updated.model_dump(mode="json"))

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Reply with one sentence containing {marker}."}
        ],
        stream=True,
        store=True,
        reasoning_effort="low",
    )
    stream_completion_id: str | None = None
    stream_text_parts: list[str] = []
    async for chunk in stream:
        stream_completion_id = chunk.id
        for choice in chunk.choices:
            if choice.delta.content:
                stream_text_parts.append(choice.delta.content)
    assert stream_completion_id is not None, backend_name
    stream_text = "".join(stream_text_parts)
    stream_retrieved = await client.chat.completions.retrieve(stream_completion_id)
    assert stream_retrieved.choices[0].message.content == stream_text, (
        backend_name,
        stream_retrieved.model_dump(mode="json"),
        stream_text,
    )

    deleted = await client.chat.completions.delete(completion.id)
    stream_deleted = await client.chat.completions.delete(stream_completion_id)
    assert deleted.deleted is True, (backend_name, deleted)
    assert stream_deleted.deleted is True, (backend_name, stream_deleted)
    with pytest.raises(Exception) as exc_info:
        await client.chat.completions.retrieve(completion.id)
    assert getattr(exc_info.value, "status_code", None) == 404, backend_name

    print(
        f"{backend_name} chat stored lifecycle: "
        f"retrieved={_short_text(retrieved_text)} "
        f"stream={_short_text(stream_text)}"
    )


async def _assert_chat_json_object_format(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only valid JSON. Do not include markdown fences or "
                    "commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Return exactly this JSON object: {"kind":"chat_json_object",'
                    '"ok":true,"count":4}'
                ),
            },
        ],
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    _assert_chat_api_shape(completion, backend_name)
    text = completion.choices[0].message.content or ""
    parsed = _parse_json_text(text)
    print(f"{backend_name} chat json_object: {parsed}")
    assert parsed == {"kind": "chat_json_object", "ok": True, "count": 4}, (
        backend_name,
        text,
    )


async def _assert_chat_structured_outputs(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    schema = _compat_structured_schema()
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return only JSON that satisfies the supplied schema.",
            },
            {
                "role": "user",
                "content": "Set code to CHAT-STRUCT-305 and count to 11.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "compat_structured_chat",
                "strict": True,
                "schema": schema,
            },
        },
        reasoning_effort="low",
    )
    _assert_chat_api_shape(completion, backend_name)
    text = completion.choices[0].message.content or ""
    parsed = _parse_json_text(text)
    print(f"{backend_name} chat structured output: {parsed}")
    assert parsed == {"code": "CHAT-STRUCT-305", "count": 11}, (
        backend_name,
        text,
    )


async def _assert_tool_choice_none_text_only(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    response_marker = f"{backend_name}-RESP-NO-TOOL-909"
    response = await client.responses.create(
        model=model,
        instructions=(
            "Do not call tools for this request. Return the marker as plain text."
        ),
        input=f"Reply with exactly: {response_marker}",
        tools=cast(Any, [_probe_tool("never_call_response_probe")]),
        tool_choice="none",
        reasoning={"effort": "low"},
    )
    _assert_response_api_shape(response, backend_name)
    response_output = _dump_model(response).get("output") or []
    assert not any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in response_output
    ), (backend_name, _dump_model(response))
    assert _contains_marker(response.output_text, response_marker), (
        backend_name,
        response.output_text,
    )

    chat_marker = f"{backend_name}-CHAT-NO-TOOL-910"
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Do not call tools for this request. Return the marker as "
                    "plain text."
                ),
            },
            {"role": "user", "content": f"Reply with exactly: {chat_marker}"},
        ],
        tools=cast(
            Any,
            [
                {
                    "type": "function",
                    "function": _probe_function("never_call_chat_probe"),
                }
            ],
        ),
        tool_choice="none",
        reasoning_effort="low",
    )
    _assert_chat_api_shape(completion, backend_name)
    message = completion.choices[0].message
    assert not message.tool_calls, (backend_name, completion.model_dump(mode="json"))
    chat_text = message.content or ""
    assert _contains_marker(chat_text, chat_marker), (backend_name, chat_text)
    print(
        f"{backend_name} tool_choice none: "
        f"responses={_short_text(response.output_text)} chat={_short_text(chat_text)}"
    )


async def _assert_chat_legacy_functions(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-LEGACY-FUNC-5150"
    first = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the provided legacy function when invoice status is "
                    "requested. Do not invent invoice data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Call lookup_invoice_status for invoice_id INV-5150. "
                    "Do not answer from memory."
                ),
            },
        ],
        functions=[
            {
                "name": "lookup_invoice_status",
                "description": "Look up invoice status by invoice id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "invoice_id": {
                            "type": "string",
                            "description": "The invoice id to look up.",
                        }
                    },
                    "required": ["invoice_id"],
                    "additionalProperties": False,
                },
            }
        ],
        function_call={"name": "lookup_invoice_status"},
        reasoning_effort="low",
    )
    _assert_chat_api_shape(first, backend_name)
    function_call = first.choices[0].message.function_call
    assert function_call is not None, (backend_name, first.model_dump(mode="json"))
    arguments = _json_object(function_call.arguments)
    assert arguments.get("invoice_id") == "INV-5150", (
        backend_name,
        function_call,
    )
    print(
        f"{backend_name} chat legacy function call: "
        f"name={function_call.name} args={function_call.arguments}"
    )

    second = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer from function results only."},
            {
                "role": "user",
                "content": (
                    "Call lookup_invoice_status for invoice_id INV-5150. "
                    "Do not answer from memory."
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_call.name,
                    "arguments": function_call.arguments,
                },
            },
            {
                "role": "function",
                "name": function_call.name,
                "content": (
                    f"INV-5150 status: paid. Ledger marker: {marker}. "
                    "Payment rail: bank transfer."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Using only the function result, state the invoice status "
                    "and ledger marker."
                ),
            },
        ],
        reasoning_effort="low",
    )
    _assert_chat_api_shape(second, backend_name)
    second_text = second.choices[0].message.content or ""
    print(f"{backend_name} chat legacy function result: {_short_text(second_text)}")
    assert _contains_marker(second_text, marker), (backend_name, second_text)
    assert "paid" in second_text.lower(), (backend_name, second_text)


async def _assert_chat_stream_include_usage(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-STREAM-USAGE-616"
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Reply with one short sentence containing {marker}.",
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
        reasoning_effort="low",
    )

    text_parts: list[str] = []
    usage_seen = False
    async for chunk in stream:
        if chunk.usage is not None:
            usage_seen = True
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            text_parts.append(delta.content)

    text = "".join(text_parts)
    assert usage_seen, backend_name
    assert _contains_marker(text, marker), (backend_name, text)
    print(f"{backend_name} chat stream include_usage: {_short_text(text)}")


async def _assert_chat_stream_multiple_choices(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    marker = f"{backend_name}-STREAM-N-313"
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Preserve exact marker strings.",
            },
            {
                "role": "user",
                "content": f"Reply with one short sentence containing {marker}.",
            },
        ],
        stream=True,
        n=2,
        reasoning_effort="low",
    )

    content_parts: dict[int, list[str]] = {0: [], 1: []}
    finish_reasons: dict[int, list[str]] = {0: [], 1: []}
    async for chunk in stream:
        for choice in chunk.choices:
            if choice.delta.content:
                content_parts.setdefault(choice.index, []).append(choice.delta.content)
            if choice.finish_reason:
                finish_reasons.setdefault(choice.index, []).append(choice.finish_reason)

    texts = {index: "".join(parts) for index, parts in content_parts.items()}
    assert sorted(texts) == [0, 1], (backend_name, texts)
    assert all(_contains_marker(text, marker) for text in texts.values()), (
        backend_name,
        texts,
    )
    assert finish_reasons == {0: ["stop"], 1: ["stop"]}, (
        backend_name,
        finish_reasons,
    )
    print(
        f"{backend_name} chat stream n=2 choices: "
        f"{[_short_text(texts[index]) for index in sorted(texts)]}"
    )


async def _assert_responses_streaming_tool_call(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    tool = _probe_tool("emit_response_probe")
    stream = await client.responses.create(
        model=model,
        instructions="Use the supplied function tool when asked for a probe.",
        input=(
            "Call emit_response_probe with probe_id RESP-STREAM-606 and "
            "do not answer directly."
        ),
        tools=cast(Any, [tool]),
        tool_choice={"type": "function", "name": "emit_response_probe"},
        stream=True,
        reasoning={"effort": "low"},
    )

    event_types: list[str] = []
    function_call: dict[str, Any] | None = None
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_item.done":
            item = getattr(event, "item", None)
            dumped = _dump_model(item)
            if dumped.get("type") == "function_call":
                function_call = dumped
    assert "response.completed" in event_types, (backend_name, event_types)
    assert function_call is not None, (backend_name, event_types)
    assert function_call["name"] == "emit_response_probe", (
        backend_name,
        function_call,
    )
    assert _json_object(function_call.get("arguments")).get("probe_id") == (
        "RESP-STREAM-606"
    ), (backend_name, function_call)
    print(
        f"{backend_name} responses streaming tool call: "
        f"events={event_types} call={function_call}"
    )


async def _assert_chat_streaming_tool_call(
    client: AsyncOpenAI, model: str, backend_name: str
) -> None:
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Use the supplied function tool when asked for a probe.",
            },
            {
                "role": "user",
                "content": (
                    "Call emit_chat_probe with probe_id CHAT-STREAM-707 and "
                    "do not answer directly."
                ),
            },
        ],
        tools=cast(
            Any,
            [
                {
                    "type": "function",
                    "function": _probe_function("emit_chat_probe"),
                }
            ],
        ),
        tool_choice=cast(
            Any, {"type": "function", "function": {"name": "emit_chat_probe"}}
        ),
        stream=True,
        reasoning_effort="low",
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

    arguments_text = "".join(argument_parts)
    arguments = _json_object(arguments_text)
    assert tool_call_id, backend_name
    assert tool_name == "emit_chat_probe", (backend_name, tool_name)
    assert arguments.get("probe_id") == "CHAT-STREAM-707", (
        backend_name,
        arguments_text,
    )
    assert finish_reasons == ["tool_calls"], (backend_name, finish_reasons)
    print(
        f"{backend_name} chat streaming tool call: "
        f"id={tool_call_id} name={tool_name} args={arguments_text}"
    )


def _assert_sync_client_smoke(
    base_url: str, model: str, backend_name: str
) -> None:
    client = OpenAI(api_key="test", base_url=f"{base_url}/v1")
    try:
        response = client.responses.create(
            model=model,
            input=f"Reply with a short sentence containing {backend_name}-SYNC-OK.",
            reasoning={"effort": "low"},
        )
        _assert_response_api_shape(response, backend_name)
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Reply with a short sentence containing "
                        f"{backend_name}-SYNC-CHAT-OK."
                    ),
                }
            ],
            reasoning_effort="low",
        )
        _assert_chat_api_shape(chat, backend_name)
        print(
            f"{backend_name} sync responses: {_short_text(response.output_text)}"
        )
        print(
            f"{backend_name} sync chat: "
            f"{_short_text(chat.choices[0].message.content or '')}"
        )
    finally:
        client.close()


async def _assert_model_list_contains(
    http_client: AsyncOpenAI, app_client: AsyncOpenAI, model: str
) -> None:
    http_models = await http_client.models.list()
    app_models = await app_client.models.list()
    assert model in [item.id for item in http_models.data], (
        "chatgpt-http",
        model,
        [item.id for item in http_models.data],
    )
    assert model in [item.id for item in app_models.data], (
        "codex-app-server",
        model,
        [item.id for item in app_models.data],
    )


def _first_function_call(response: Any) -> dict[str, Any] | None:
    for item in _dump_model(response).get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if call_id:
                item["call_id"] = str(call_id)
            return item
    return None


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, dict):
        return value
    return {}


def _response_item_text(item: dict[str, Any]) -> str:
    texts: list[str] = []
    for content in item.get("content") or []:
        if isinstance(content, dict) and content.get("type") == "output_text":
            texts.append(str(content.get("text") or ""))
    return "".join(texts)


def _assert_response_api_shape(response: Any, backend_name: str) -> None:
    dumped = _dump_model(response)
    assert dumped.get("object") == "response", (backend_name, dumped)
    assert dumped.get("id"), (backend_name, dumped)
    assert dumped.get("model"), (backend_name, dumped)
    assert isinstance(dumped.get("output"), list), (backend_name, dumped)
    assert getattr(response, "output_text", "") is not None, (backend_name, dumped)


def _assert_chat_api_shape(completion: Any, backend_name: str) -> None:
    dumped = _dump_model(completion)
    assert dumped.get("object") == "chat.completion", (backend_name, dumped)
    assert dumped.get("id"), (backend_name, dumped)
    assert dumped.get("model"), (backend_name, dumped)
    assert dumped.get("choices"), (backend_name, dumped)
    first_choice = dumped["choices"][0]
    assert first_choice.get("index") == 0, (backend_name, dumped)
    assert isinstance(first_choice.get("message"), dict), (backend_name, dumped)


def _compat_structured_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["code", "count"],
        "additionalProperties": False,
    }


def _probe_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        **_probe_function(name),
    }


def _probe_function(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Emit a probe function call for compatibility testing.",
        "parameters": {
            "type": "object",
            "properties": {
                "probe_id": {
                    "type": "string",
                    "description": "The probe id to echo.",
                }
            },
            "required": ["probe_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _parse_json_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Expected JSON text, got: {text}") from exc
    assert isinstance(parsed, dict), parsed
    return parsed


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assert_contains_markers(text: str, markers: list[str], backend_name: str) -> None:
    missing = [marker for marker in markers if not _contains_marker(text, marker)]
    assert not missing, (backend_name, missing, text)


def _short_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _contains_marker(text: str, marker: str) -> bool:
    normalized_text = _normalize_marker_text(text)
    normalized_marker = _normalize_marker_text(marker)
    return normalized_marker in normalized_text


def _normalize_marker_text(text: str) -> str:
    return "".join(char.upper() for char in text if char.isalnum())


def _mentions_red(text: str) -> bool:
    normalized = _normalize_marker_text(text)
    return "RED" in normalized or "赤" in text


def _select_model(http_models: list[str], app_models: list[str]) -> str:
    requested = os.environ.get("OPENAI_VIA_CODEX_TEST_MODEL")
    if requested:
        return requested
    for preferred in ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5"):
        if preferred in http_models and preferred in app_models:
            return preferred
    common = [model for model in http_models if model in app_models]
    return common[0] if common else (http_models[0] if http_models else "gpt-5.4")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_server(base_url: str) -> None:
    async with httpx.AsyncClient() as client:
        for _ in range(100):
            try:
                response = await client.get(f"{base_url}/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise AssertionError(f"server did not become ready: {base_url}")
