from __future__ import annotations

from typing import Any

import pytest

from openai_api_server_via_codex.app_server import (
    CodexAppServerBackend,
    response_input_to_app_server_input,
)
from openai_api_server_via_codex.backend import CodexBackendError


class FakeAppServerClient:
    def __init__(self) -> None:
        self.started = False
        self.login_params: dict[str, Any] | None = None
        self.thread_start_calls: list[dict[str, Any]] = []
        self.turn_start_calls: list[dict[str, Any]] = []
        self.model_list_calls = 0
        self._thread_counter = 0
        self._turn_counter = 0
        self._notifications: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        self.started = True

    async def login_with_chatgpt_tokens(
        self, *, access_token: str, account_id: str
    ) -> None:
        self.login_params = {
            "accessToken": access_token,
            "chatgptAccountId": account_id,
        }

    async def thread_start(self, params: dict[str, Any]) -> dict[str, Any]:
        self.thread_start_calls.append(params)
        self._thread_counter += 1
        return {"thread": {"id": f"thread_{self._thread_counter}"}}

    async def turn_start(
        self, *, thread_id: str, input_items: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        self.turn_start_calls.append(
            {"thread_id": thread_id, "input": input_items, "params": params}
        )
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter}"
        text = f"reply {self._turn_counter}"
        self._notifications.extend(
            [
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id, "items": [], "status": "inProgress"},
                    },
                },
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "itemId": f"item_{self._turn_counter}",
                        "delta": text[:6],
                    },
                },
                {
                    "method": "item/agentMessage/delta",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "itemId": f"item_{self._turn_counter}",
                        "delta": text[6:],
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {
                            "id": turn_id,
                            "items": [],
                            "status": "completed",
                            "completedAt": 1_700_000_000,
                        },
                    },
                },
            ]
        )
        return {"turn": {"id": turn_id, "items": [], "status": "inProgress"}}

    async def next_notification(self) -> dict[str, Any]:
        return self._notifications.pop(0)

    async def model_list(self) -> dict[str, Any]:
        self.model_list_calls += 1
        return {
            "data": [
                {"id": "gpt-5.4", "hidden": False},
                {"id": "gpt-5.4-mini", "hidden": False},
                {"id": "hidden-model", "hidden": True},
            ]
        }

    async def close(self) -> None:
        self.started = False


class FakeToolCallAppServerClient(FakeAppServerClient):
    async def turn_start(
        self, *, thread_id: str, input_items: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        self.turn_start_calls.append(
            {"thread_id": thread_id, "input": input_items, "params": params}
        )
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter}"
        item = {
            "type": "dynamicToolCall",
            "id": "call_weather_1",
            "namespace": None,
            "tool": "lookup_weather",
            "arguments": {"city": "Tokyo"},
            "status": "inProgress",
            "contentItems": None,
            "success": None,
            "durationMs": None,
        }
        self._notifications.extend(
            [
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id, "items": [], "status": "inProgress"},
                    },
                },
                {
                    "method": "item/started",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "item": item,
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "item": {**item, "status": "completed"},
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {
                            "id": turn_id,
                            "items": [],
                            "status": "completed",
                            "completedAt": 1_700_000_000,
                        },
                    },
                },
            ]
        )
        return {"turn": {"id": turn_id, "items": [], "status": "inProgress"}}


@pytest.mark.asyncio
async def test_app_server_backend_creates_thread_and_binds_response_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAppServerClient()
    monkeypatch.setattr(
        "openai_api_server_via_codex.app_server.borrow_codex_key",
        lambda auth_json=None: ("access-token", "account-id"),
    )
    backend = CodexAppServerBackend(client_factory=lambda: fake_client)

    first = await backend.create_response(
        {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "Remember blue."}],
            "instructions": "Be terse.",
            "reasoning": {"effort": "low"},
        }
    )
    second = await backend.create_response(
        {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": "What color?"}],
            "previous_response_id": first["id"],
            "reasoning": {"effort": "high"},
        }
    )

    assert first["id"] == "resp_turn_1"
    assert first["output"][0]["content"][0]["text"] == "reply 1"
    assert second["id"] == "resp_turn_2"
    assert fake_client.login_params == {
        "accessToken": "access-token",
        "chatgptAccountId": "account-id",
    }
    assert len(fake_client.thread_start_calls) == 1
    assert fake_client.thread_start_calls[0]["model"] == "gpt-5.4"
    assert fake_client.thread_start_calls[0]["baseInstructions"] == "Be terse."
    assert fake_client.turn_start_calls[0]["thread_id"] == "thread_1"
    assert fake_client.turn_start_calls[0]["params"]["effort"] == "low"
    assert fake_client.turn_start_calls[1]["thread_id"] == "thread_1"
    assert fake_client.turn_start_calls[1]["params"]["effort"] == "high"
    assert fake_client.turn_start_calls[1]["input"] == [
        {"type": "text", "text": "user: What color?"}
    ]


@pytest.mark.asyncio
async def test_app_server_backend_streams_responses_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAppServerClient()
    monkeypatch.setattr(
        "openai_api_server_via_codex.app_server.borrow_codex_key",
        lambda auth_json=None: ("access-token", "account-id"),
    )
    backend = CodexAppServerBackend(client_factory=lambda: fake_client)

    events = [
        event
        async for event in backend.stream_response(
            {
                "model": "gpt-5.4",
                "input": [{"role": "user", "content": "Stream."}],
            }
        )
    ]

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[0]["response"]["id"] == "resp_turn_1"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "reply 1"


@pytest.mark.asyncio
async def test_app_server_backend_projects_dynamic_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeToolCallAppServerClient()
    monkeypatch.setattr(
        "openai_api_server_via_codex.app_server.borrow_codex_key",
        lambda auth_json=None: ("access-token", "account-id"),
    )
    backend = CodexAppServerBackend(client_factory=lambda: fake_client)

    events = [
        event
        async for event in backend.stream_response(
            {
                "model": "gpt-5.4",
                "input": [{"role": "user", "content": "Call weather for Tokyo."}],
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_weather",
                        "description": "Look up weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            }
        )
    ]

    assert fake_client.thread_start_calls[0]["dynamicTools"] == [
        {
            "name": "lookup_weather",
            "description": "Look up weather.",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "deferLoading": False,
        }
    ]
    assert fake_client.thread_start_calls[0]["experimentalRawEvents"] is True
    assert fake_client.thread_start_calls[0]["persistExtendedHistory"] is True

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[1]["item"]["type"] == "function_call"
    assert events[1]["item"]["name"] == "lookup_weather"
    assert events[2]["delta"] == '{"city":"Tokyo"}'
    assert events[3]["item"] == {
        "id": "call_weather_1",
        "type": "function_call",
        "call_id": "call_weather_1",
        "name": "lookup_weather",
        "arguments": '{"city":"Tokyo"}',
        "status": "completed",
    }
    assert events[-1]["response"]["output"] == [events[3]["item"]]
    assert events[-1]["response"]["tools"][0]["name"] == "lookup_weather"


@pytest.mark.asyncio
async def test_app_server_backend_unknown_previous_response_id_raises_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAppServerClient()
    monkeypatch.setattr(
        "openai_api_server_via_codex.app_server.borrow_codex_key",
        lambda auth_json=None: ("access-token", "account-id"),
    )
    backend = CodexAppServerBackend(client_factory=lambda: fake_client)

    with pytest.raises(CodexBackendError) as exc_info:
        await backend.create_response(
            {
                "model": "gpt-5.4",
                "input": [{"role": "user", "content": "Hello."}],
                "previous_response_id": "resp_missing",
            }
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_app_server_backend_lists_visible_models() -> None:
    fake_client = FakeAppServerClient()
    backend = CodexAppServerBackend(client_factory=lambda: fake_client)

    models = await backend.list_models()

    assert models == ["gpt-5.4", "gpt-5.4-mini"]


def test_response_input_to_app_server_input_maps_text_roles_and_images() -> None:
    assert response_input_to_app_server_input(
        [
            {"role": "user", "content": "Describe."},
            {"role": "assistant", "content": "Previous."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look."},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                        "detail": "low",
                    },
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ]
    ) == [
        {"type": "text", "text": "user: Describe."},
        {"type": "text", "text": "assistant: Previous."},
        {"type": "text", "text": "user: Look."},
        {"type": "image", "url": "data:image/png;base64,AAAA"},
        {
            "type": "text",
            "text": 'assistant tool call call_2 lookup_weather: {"city":"Tokyo"}',
        },
        {"type": "text", "text": "tool call_1: 42"},
    ]
