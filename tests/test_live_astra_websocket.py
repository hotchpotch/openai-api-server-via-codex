from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from openai import AsyncOpenAI, BadRequestError, NotFoundError, OpenAI
from openai.types.responses import ResponseInputParam, ToolParam

from tests.websocket_helpers import (
    API_KEY,
    ASYNC_TOOL,
    MODEL,
    output_text,
    print_observation,
    receive_terminal,
    spawned_server,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to use existing Codex authentication and gpt-6-astra.",
)


@pytest.fixture(scope="module")
def astra_server(
    go_binary: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    work = tmp_path_factory.mktemp("astra-live")
    host = os.environ.get("OPENAI_VIA_CODEX_TEST_HOST", "127.0.0.1")
    with spawned_server(go_binary, work, host=host) as base:
        yield base


async def test_live_astra_websocket_async_tools(astra_server: str) -> None:
    async with (
        AsyncOpenAI(
            api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        tools = [cast(ToolParam, ASYNC_TOOL)]
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=tools,
            input='Call lookup once with key="demo". Then say exactly INDEPENDENT_OK without waiting for its result.',
        )
        first, items, types = await receive_terminal(connection)
        print_observation("ws_async_initial", first, items, types)
        assert first["status"] == "completed"
        call = next(item for item in items if item["type"] == "function_call")
        assert call["name"] == "lookup" and call["async"] is True
        assert json.loads(call["arguments"]) == {"key": "demo"}
        assert "INDEPENDENT_OK" in output_text(items)

        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=tools,
            previous_response_id=first["id"],
            input="lookup is still pending. Do not call it again. Reply exactly PENDING_OK now.",
        )
        pending, items, types = await receive_terminal(connection)
        print_observation("ws_async_pending", pending, items, types)
        assert (
            pending["status"] == "completed"
            and output_text(items).strip() == "PENDING_OK"
        )
        # Cross the relay keepalive interval while a tool result is pending.
        idle_seconds = int(os.environ.get("CODEX_WS_IDLE_SECONDS", "25"))
        assert idle_seconds >= 25
        for elapsed in range(0, idle_seconds, 30):
            await asyncio.sleep(min(30, idle_seconds - elapsed))
            print(
                f"ws pending idle: {min(elapsed + 30, idle_seconds)}/{idle_seconds}s",
                flush=True,
            )
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=tools,
            previous_response_id=pending["id"],
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "VALUE_OK",
                },
                {
                    "role": "user",
                    "content": "Repeat exactly the lookup result that just arrived.",
                },
            ],
        )
        final, items, types = await receive_terminal(connection)
        print_observation("ws_async_result", final, items, types)
        assert (
            final["status"] == "completed" and output_text(items).strip() == "VALUE_OK"
        )
        with pytest.raises(NotFoundError):
            await client.responses.retrieve(first["id"])


async def test_live_astra_websocket_steering(astra_server: str) -> None:
    async with (
        AsyncOpenAI(
            api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "medium"},
            input="Draft a 12-step implementation plan for a task-tracking app, including tests and deployment. Keep the entire answer under 200 words.",
        )
        while True:
            created = await asyncio.wait_for(connection.recv(), 120)
            assert created.type not in {"error", "response.failed"}
            if created.type == "response.created":
                break
        original = created.model_dump()["response"]["id"]
        await connection.response.steer(
            previous_response_id=original,
            input="Change the task: stop the plan and reply exactly STEER_OK.",
        )
        first, items, types = await receive_terminal(connection)
        print_observation("ws_steer_original", first, items, types)
        assert first["id"] == original
        accepted = "response.steer.accepted" in types
        if first["status"] == "incomplete":
            assert first["incomplete_details"]["reason"] == "steered"
        else:
            assert first["status"] == "completed"
        successor, items, types = await receive_terminal(connection)
        print_observation("ws_steer_successor", successor, items, types)
        assert accepted or "response.steer.accepted" in types
        assert successor["id"] != original and successor["status"] == "completed"
        assert output_text(items).strip() == "STEER_OK"


def test_live_astra_websocket_sync_context(astra_server: str) -> None:
    with (
        OpenAI(api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        previous = None
        for prompt in [
            "Reply exactly WS_CONTEXT_OK",
            "What exact token did you just say? Reply only that token.",
        ]:
            args: dict[str, Any] = {
                "model": MODEL,
                "input": prompt,
                "reasoning": {"effort": "low"},
            }
            if previous:
                args["previous_response_id"] = previous
            connection.response.create(**args)
            items: list[dict[str, Any]] = []
            types: list[str] = []
            completed = False
            for event in connection:
                data = event.model_dump(exclude_none=True, by_alias=True)
                types.append(data["type"])
                assert data["type"] not in {
                    "error",
                    "response.failed",
                    "response.incomplete",
                }
                if data["type"] == "response.output_item.done":
                    items.append(data["item"])
                if data["type"] == "response.completed":
                    print_observation("ws_sync_context", data["response"], items, types)
                    previous = data["response"]["id"]
                    assert output_text(items).strip() == "WS_CONTEXT_OK"
                    completed = True
                    break
            assert completed


async def test_live_astra_http_async_history(astra_server: str) -> None:
    async with AsyncOpenAI(
        api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
    ) as client:
        tools = [cast(ToolParam, ASYNC_TOOL)]
        initial = 'Call lookup once with key="demo". Then say HTTP_INDEPENDENT_OK without waiting for its result.'
        first = await client.responses.create(
            model=MODEL, reasoning={"effort": "low"}, tools=tools, input=initial
        )
        first_items = [
            item.model_dump(exclude_none=True, by_alias=True) for item in first.output
        ]
        print_observation(
            "http_async_initial", first.model_dump(by_alias=True), first_items, []
        )
        call = next(item for item in first_items if item["type"] == "function_call")
        assert call["async"] is True and call["name"] == "lookup"
        assert json.loads(call["arguments"]) == {"key": "demo"}
        assert "HTTP_INDEPENDENT_OK" in first.output_text
        # Exercise both the compatibility ID store and caller-managed history.
        for mode in ["previous_response_id", "explicit_history"]:
            prompt = "lookup is still pending. Do not call it again. Reply exactly HTTP_PENDING_OK now."
            kwargs: dict[str, Any]
            if mode == "previous_response_id":
                kwargs = {"previous_response_id": first.id, "input": prompt}
            else:
                kwargs = {
                    "input": [
                        {"role": "user", "content": initial},
                        *first_items,
                        {"role": "user", "content": prompt},
                    ]
                }
            pending = await client.responses.create(
                model=MODEL, reasoning={"effort": "low"}, tools=tools, **kwargs
            )
            pending_items = [
                item.model_dump(exclude_none=True, by_alias=True)
                for item in pending.output
            ]
            print_observation(
                "http_pending_" + mode,
                pending.model_dump(by_alias=True),
                pending_items,
                [],
            )
            assert (
                pending.status == "completed"
                and pending.output_text.strip() == "HTTP_PENDING_OK"
            )
            final = await client.responses.create(
                model=MODEL,
                reasoning={"effort": "low"},
                tools=tools,
                previous_response_id=pending.id,
                input=cast(
                    ResponseInputParam,
                    [
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": "HTTP_VALUE_OK",
                        },
                        {
                            "role": "user",
                            "content": "Repeat exactly the lookup result that just arrived.",
                        },
                    ],
                ),
            )
            print_observation(
                "http_result_" + mode,
                final.model_dump(by_alias=True),
                [item.model_dump(by_alias=True) for item in final.output],
                [],
            )
            assert (
                final.status == "completed"
                and final.output_text.strip() == "HTTP_VALUE_OK"
            )


async def test_live_astra_background_rejected(astra_server: str) -> None:
    async with AsyncOpenAI(
        api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
    ) as client:
        with pytest.raises(BadRequestError) as caught:
            await client.responses.create(model=MODEL, input="hello", background=True)
        assert (
            caught.value.code == "unsupported_parameter"
            and caught.value.param == "background"
        )
        print(
            "background=true rejected: HTTP 400 unsupported_parameter / background",
            flush=True,
        )


async def test_live_astra_websocket_custom_async_tool(astra_server: str) -> None:
    custom: ToolParam = cast(
        ToolParam,
        {
            "type": "custom",
            "name": "custom_lookup",
            "async": True,
            "description": "Look up a demo value. Pass exactly demo as the free-form input. Result arrives later.",
            "format": {"type": "text"},
        },
    )
    async with (
        AsyncOpenAI(
            api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=[custom],
            input="Call custom_lookup exactly once with input demo. Then say CUSTOM_INDEPENDENT_OK without waiting for its result.",
        )
        first, items, types = await receive_terminal(connection)
        print_observation("ws_custom_initial", first, items, types)
        assert first["status"] == "completed"
        call = next(item for item in items if item["type"] == "custom_tool_call")
        assert call["async"] is True and call["name"] == "custom_lookup"
        assert call["input"].strip() == "demo"
        assert "CUSTOM_INDEPENDENT_OK" in output_text(items)
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=[custom],
            previous_response_id=first["id"],
            input="The custom_lookup result is still pending. Do not call it again. Reply exactly CUSTOM_PENDING_OK.",
        )
        pending, items, types = await receive_terminal(connection)
        print_observation("ws_custom_pending", pending, items, types)
        assert (
            pending["status"] == "completed"
            and output_text(items).strip() == "CUSTOM_PENDING_OK"
        )
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            tools=[custom],
            previous_response_id=pending["id"],
            input=[
                {
                    "type": "custom_tool_call_output",
                    "call_id": call["call_id"],
                    "output": "CUSTOM_VALUE_OK",
                },
                {
                    "role": "user",
                    "content": "Repeat exactly the custom_lookup result that just arrived.",
                },
            ],
        )
        final, items, types = await receive_terminal(connection)
        print_observation("ws_custom_result", final, items, types)
        assert (
            final["status"] == "completed"
            and output_text(items).strip() == "CUSTOM_VALUE_OK"
        )


async def test_live_astra_websocket_upstream_stream_id_rejection(
    astra_server: str,
) -> None:
    # Codex currently lacks named lanes even though the public API has them.
    # Keep this a real rejection contract, not a skipped multiplexing test.
    async with (
        AsyncOpenAI(
            api_key=API_KEY, base_url=astra_server + "/v1", max_retries=0
        ) as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(
            model=MODEL,
            reasoning={"effort": "low"},
            stream_id="alpha",
            input="Reply exactly ALPHA_OK",
        )
        while True:
            event = await asyncio.wait_for(connection.recv(), 120)
            data = event.model_dump(by_alias=True, exclude_none=True)
            if data["type"] == "error":
                error = data["error"]
                print("stream_id upstream rejection:", error, flush=True)
                assert error["type"] == "invalid_request_error"
                assert "Unsupported parameter: stream_id" in error["message"]
                break
            assert data["type"] not in {"response.created", "response.completed"}
