from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI, ConflictError, NotFoundError


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(filter(None, (_flatten_text(item) for item in value)))
    if isinstance(value, dict):
        if value.get("type") in {"input_text", "output_text", "text"}:
            return str(value.get("text") or "")
        return " ".join(
            filter(
                None, (_flatten_text(value.get(key)) for key in ("content", "input"))
            )
        )
    return ""


class _FakeCodexHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counter = 0

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_GET(self) -> None:
        if self.path.startswith("/backend-api/codex/models"):
            self._json(
                {
                    "models": [
                        {
                            "slug": "gpt-5.6-luna",
                            "supported_in_api": True,
                            "visibility": "list",
                        },
                        {
                            "slug": "hidden-model",
                            "supported_in_api": True,
                            "visibility": "hidden",
                        },
                    ]
                }
            )
            return
        self._proxy_response()

    def do_POST(self) -> None:
        if self.path == "/backend-api/codex/responses":
            size = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(size))
            self._responses_stream(payload)
            return
        if self.path == "/backend-api/transcribe":
            size = int(self.headers.get("content-length", "0"))
            self.rfile.read(size)
            self._json({"text": "transcribed by fake Codex"})
            return
        self._proxy_response()

    def do_DELETE(self) -> None:
        self._proxy_response()

    def _responses_stream(self, payload: dict[str, Any]) -> None:
        if payload.get("metadata", {}).get("contract") == "async_history":
            self._async_history_stream(payload)
            return
        type(self).counter += 1
        number = type(self).counter
        response_id = f"resp_contract_{number}"
        created = time.time()
        tools = payload.get("tools") or []
        is_image = any(tool.get("type") == "image_generation" for tool in tools)
        is_tool = any(tool.get("type") == "function" for tool in tools)

        if is_image:
            item = {
                "id": f"ig_contract_{number}",
                "type": "image_generation_call",
                "status": "completed",
                "result": base64.b64encode(b"\x89PNG\r\n\x1a\ncontract").decode(),
                "revised_prompt": "revised contract image",
            }
        elif is_tool:
            item = {
                "id": f"fc_contract_{number}",
                "type": "function_call",
                "call_id": f"call_contract_{number}",
                "name": "lookup_weather",
                "arguments": '{"city":"Tokyo"}',
                "status": "completed",
            }
        else:
            text = "fake contract: " + _flatten_text(payload.get("input"))
            item = {
                "id": f"msg_contract_{number}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        response = {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": payload.get("model"),
            "output": [item],
            "parallel_tool_calls": True,
            "tool_choice": payload.get("tool_choice", "auto"),
            "tools": tools,
            "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        }
        created_response = {**response, "status": "in_progress", "output": []}
        events: list[dict[str, Any]] = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": created_response,
            }
        ]
        if is_tool:
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {**item, "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": 2,
                        "output_index": 0,
                        "item_id": item["id"],
                        "delta": '{"city":"Tokyo"}',
                    },
                ]
            )
        elif not is_image:
            message_text = cast(dict[str, Any], cast(list[Any], item["content"])[0])[
                "text"
            ]
            events.append(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": item["id"],
                    "delta": message_text,
                    "logprobs": [],
                }
            )
        events.extend(
            [
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": response,
                },
            ]
        )
        chunks = (
            "".join(
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                for event in events
            )
            + "data: [DONE]\n\n"
        )
        encoded = chunks.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _proxy_response(self) -> None:
        self._json(
            {
                "object": "list",
                "data": [],
                "has_more": False,
                "request_method": self.command,
                "request_path": self.path,
            },
            headers={"x-upstream-request-id": "contract-upstream"},
        )

    def _async_history_stream(self, payload: dict[str, Any]) -> None:
        type(self).counter += 1
        rid = f"resp_history_{type(self).counter}"
        history = payload["input"]
        calls = [item for item in history if item.get("type") == "function_call"]
        if calls:
            assert calls[0]["async"] is True
            assert calls[0]["future_field"] == "preserved"
            assert any(
                item.get("type") == "custom_tool_call" and item.get("async")
                for item in history
            )
            assert any(item.get("encrypted_content") == "opaque" for item in history)
            assert any(item.get("phase") == "commentary" for item in history)
            final = any(item.get("type") == "function_call_output" for item in history)
            if final:
                result = next(
                    item
                    for item in history
                    if item.get("type") == "function_call_output"
                )
                assert result["output"] == [{"type": "input_text", "text": "value"}]
                assert any(
                    item.get("type") == "custom_tool_call_output"
                    and item["output"] == "custom result"
                    for item in history
                )
            items = [
                {
                    "id": "msg_history",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "FINAL_OK" if final else "PENDING_OK",
                            "annotations": [],
                        }
                    ],
                }
            ]
        else:
            items = [
                {
                    "id": "fc_history",
                    "type": "function_call",
                    "call_id": "call_history",
                    "name": "lookup",
                    "arguments": "{}",
                    "async": True,
                    "future_field": "preserved",
                    "status": "completed",
                },
                {
                    "id": "custom_history",
                    "type": "custom_tool_call",
                    "call_id": "call_custom",
                    "name": "custom_lookup",
                    "input": "demo",
                    "async": True,
                },
                {
                    "id": "rs_history",
                    "type": "reasoning",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
                {
                    "id": "msg_history",
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "INDEPENDENT_OK",
                            "annotations": [],
                        }
                    ],
                },
            ]
        response = {
            "id": rid,
            "object": "response",
            "created_at": 0,
            "model": payload["model"],
            "status": "completed",
            "output": items,
        }
        events: list[dict[str, Any]] = [
            {
                "type": "response.created",
                "response": {**response, "status": "in_progress", "output": []},
            }
        ]
        events.extend(
            {"type": "response.output_item.done", "output_index": index, "item": item}
            for index, item in enumerate(items)
        )
        events.append({"type": "response.completed", "response": response})
        encoded = "".join(
            "data: " + json.dumps({**event, "sequence_number": index}) + "\n\n"
            for index, event in enumerate(events)
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json(
        self,
        value: Any,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        for name, header_value in (headers or {}).items():
            self.send_header(name, header_value)
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture(scope="module")
def fake_codex_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCodexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = cast(tuple[str, int], server.server_address)
    host, port = address
    try:
        yield f"http://{host}:{port}/backend-api/codex"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def runtime_server(
    fake_codex_url: str,
    go_binary: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, str]]:
    runtime = "go"
    work = tmp_path_factory.mktemp("contract-go")
    auth_json = work / "auth.json"
    auth_json.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.payload.signature",
                    "account_id": "contract-account",
                },
            }
        )
    )
    port = _free_port()
    command = [
        str(go_binary),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--auth-json",
        str(auth_json),
        "--backend-base-url",
        fake_codex_url,
        "--max-stored-items",
        "100",
        "--api-key",
        "contract-server-key",
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"{runtime} server exited early:\n{output}")
            try:
                if httpx.get(f"{base_url}/healthz", timeout=0.2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise AssertionError(f"{runtime} server did not become healthy")
        yield runtime, base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
async def contract_client(
    runtime_server: tuple[str, str],
) -> AsyncIterator[AsyncOpenAI]:
    _, base_url = runtime_server
    async with AsyncOpenAI(
        api_key="contract-server-key", base_url=f"{base_url}/v1"
    ) as client:
        yield client


async def test_contract_health_and_models(runtime_server: tuple[str, str]) -> None:
    runtime, base_url = runtime_server
    async with httpx.AsyncClient(base_url=base_url) as direct:
        health = await direct.get("/healthz")
        invalid_health_method = await direct.post("/healthz")
        outside_v1 = await direct.get("/outside-v1")
        unauthorized = await direct.get("/v1/models")
        unauthorized_v1_root = await direct.get("/v1")
    assert health.json() == {"status": "ok"}, runtime
    assert invalid_health_method.status_code == 405, runtime
    assert outside_v1.status_code == 404, runtime
    assert unauthorized.status_code == 401, runtime
    assert unauthorized.json()["error"]["code"] == "invalid_api_key", runtime
    assert unauthorized_v1_root.status_code == 401, runtime
    async with AsyncOpenAI(
        api_key="contract-server-key", base_url=f"{base_url}/v1"
    ) as client:
        models = await client.models.list()
    assert [model.id for model in models.data] == ["gpt-5.6-luna"], runtime

    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": "bearer contract-server-key"},
    ) as direct:
        root = await direct.get("/v1")
        root_with_slash = await direct.get("/v1/")
    assert root.status_code == 307, runtime
    assert root.headers["location"].endswith("/v1/"), runtime
    assert root_with_slash.status_code == 200, runtime
    assert root_with_slash.json()["request_path"].endswith("/backend-api/codex/"), (
        runtime
    )


async def test_contract_responses_lifecycle_and_streaming(
    runtime_server: tuple[str, str], contract_client: AsyncOpenAI
) -> None:
    runtime, _ = runtime_server
    created = await contract_client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {"role": "user", "content": "response lifecycle marker"},
            {"role": "user", "content": "second input marker"},
        ],
    )
    assert "response lifecycle marker" in created.output_text, runtime
    assert "second input marker" in created.output_text, runtime

    retrieved = await contract_client.responses.retrieve(created.id)
    assert retrieved.output_text == created.output_text, runtime

    with pytest.raises(ConflictError):
        await contract_client.responses.cancel(created.id)

    input_items = await contract_client.responses.input_items.list(created.id, limit=1)
    assert input_items.has_more is True, runtime
    assert input_items.data[0].model_dump(mode="json", exclude_none=True) == {
        "id": "input_0",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_text", "text": "response lifecycle marker"}],
    }, runtime
    next_input_items = await contract_client.responses.input_items.list(
        created.id, after="input_0", limit=2
    )
    assert next_input_items.has_more is False, runtime
    assert [item.id for item in next_input_items.data] == ["input_1"], runtime

    counted = await contract_client.responses.input_tokens.count(
        model="gpt-5.6-luna", input="count this input"
    )
    assert counted.input_tokens > 0, runtime

    continued = await contract_client.responses.create(
        model="gpt-5.6-luna",
        previous_response_id=created.id,
        input="continued marker",
    )
    assert continued.previous_response_id == created.id, runtime
    assert "response lifecycle marker" in continued.output_text, runtime
    assert "continued marker" in continued.output_text, runtime

    stream = await contract_client.responses.create(
        model="gpt-5.6-luna", input="response stream marker", stream=True
    )
    event_types: list[str] = []
    deltas: list[str] = []
    streamed_id = ""
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_text.delta":
            deltas.append(cast(Any, event).delta)
        if event.type == "response.completed":
            streamed_id = cast(Any, event).response.id
    assert "response.completed" in event_types, runtime
    assert "response stream marker" in "".join(deltas), runtime

    replay = await contract_client.responses.retrieve(streamed_id, stream=True)
    replay_types = [event.type async for event in replay]
    assert replay_types[-1] == "response.completed", runtime

    await contract_client.responses.delete(created.id)
    with pytest.raises(NotFoundError):
        await contract_client.responses.retrieve(created.id)


async def test_contract_chat_lifecycle_streaming_and_tools(
    runtime_server: tuple[str, str], contract_client: AsyncOpenAI
) -> None:
    runtime, _ = runtime_server
    completion = await contract_client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "chat lifecycle marker"}],
        n=2,
        store=True,
        metadata={"suite": "contract"},
    )
    assert "chat lifecycle marker" in (completion.choices[0].message.content or ""), (
        runtime
    )

    retrieved = await contract_client.chat.completions.retrieve(completion.id)
    assert retrieved.id == completion.id, runtime
    listed = await contract_client.chat.completions.list(metadata={"suite": "contract"})
    assert completion.id in [item.id for item in listed.data], runtime
    updated = await contract_client.chat.completions.update(
        completion.id, metadata={"suite": "updated"}
    )
    assert updated.model_dump(by_alias=True).get("metadata") == {"suite": "updated"}, (
        runtime
    )
    messages = await contract_client.chat.completions.messages.list(completion.id)
    assert messages.data[0].role == "assistant", runtime
    descending_messages = await contract_client.chat.completions.messages.list(
        completion.id, order="desc"
    )
    assert [message.id for message in descending_messages.data] == [
        f"{completion.id}_msg_1",
        f"{completion.id}_msg_0",
    ], runtime
    remaining_messages = await contract_client.chat.completions.messages.list(
        completion.id, after=f"{completion.id}_msg_0", limit=2
    )
    assert remaining_messages.has_more is False, runtime
    assert [message.id for message in remaining_messages.data] == [
        f"{completion.id}_msg_1"
    ], runtime

    stream = await contract_client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "chat stream marker"}],
        n=2,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]
    streamed_by_choice = {
        index: "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in chunk.choices
            if choice.index == index
        )
        for index in (0, 1)
    }
    assert all("chat stream marker" in text for text in streamed_by_choice.values()), (
        runtime
    )
    assert any(chunk.usage is not None for chunk in chunks), runtime

    tool = await contract_client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "use the weather tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
    )
    tool_calls = tool.choices[0].message.tool_calls
    assert tool_calls is not None, runtime
    assert (
        tool_calls[0].model_dump(by_alias=True)["function"]["name"] == "lookup_weather"
    ), runtime

    deleted = await contract_client.chat.completions.delete(completion.id)
    assert deleted.deleted is True, runtime


async def test_contract_images_audio_and_unknown_proxy(
    runtime_server: tuple[str, str], contract_client: AsyncOpenAI
) -> None:
    runtime, base_url = runtime_server
    image = await contract_client.images.generate(
        model="gpt-5.6-luna", prompt="draw a contract", size="256x256"
    )
    assert image.data is not None and image.data[0].b64_json is not None, runtime
    assert base64.b64decode(image.data[0].b64_json).startswith(b"\x89PNG"), runtime

    transcription = await contract_client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("sample.wav", b"RIFF-contract", "audio/wav"),
    )
    assert transcription.text == "transcribed by fake Codex", runtime

    async with httpx.AsyncClient(
        base_url=base_url, headers={"Authorization": "Bearer contract-server-key"}
    ) as direct:
        proxied = await direct.get("/v1/batches?limit=3")
        encoded_delimiters = await direct.get(
            "/v1/files/report%3Fformat%23section?limit=1"
        )
        literal_percent = await direct.get("/v1/files/100%25done")
        encoded_traversal = await direct.get("/v1/files/%2e%2e/auth")
    assert proxied.json()["object"] == "list", runtime
    assert proxied.headers["x-openai-via-codex-proxy"] == "codex-http", runtime
    assert encoded_delimiters.json()["request_path"].endswith(
        "/backend-api/codex/files/report%3Fformat%23section?limit=1"
    ), runtime
    assert literal_percent.json()["request_path"].endswith(
        "/backend-api/codex/files/100%25done"
    ), runtime
    assert encoded_traversal.status_code == 400, runtime


@pytest.fixture
def websocket_runtime(
    go_binary: Path,
    tmp_path: Path,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import ServerConnection, serve

    from tests.websocket_helpers import spawned_server

    received: list[dict[str, Any]] = []
    failures: list[str] = []

    def handler(connection: ServerConnection) -> None:
        assert connection.request is not None
        assert connection.request.headers["Authorization"] == "Bearer fake-ws-token"
        assert connection.request.headers["ChatGPT-Account-ID"] == "fake-ws-account"
        sequence = 0

        def emit(event: dict[str, Any]) -> None:
            nonlocal sequence
            connection.send(json.dumps({"sequence_number": sequence, **event}))
            sequence += 1

        def response(rid: str, status: str = "completed") -> dict[str, Any]:
            return {
                "id": rid,
                "object": "response",
                "created_at": 0,
                "model": "gpt-6-astra",
                "status": status,
                "output": [],
            }

        def finish(rid: str, text: str, call: bool = False) -> None:
            emit({"type": "response.created", "response": response(rid, "in_progress")})
            if call:
                emit(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": "fc_async",
                            "type": "function_call",
                            "call_id": "call_async",
                            "name": "lookup",
                            "arguments": '{"key":"demo"}',
                            "async": True,
                            "status": "completed",
                            "future_field": "preserved",
                        },
                    }
                )
            emit(
                {
                    "type": "response.output_item.done",
                    "output_index": int(call),
                    "item": {
                        "id": "msg_" + rid,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "phase": "final_answer",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    },
                }
            )
            # Codex may leave final response.output empty; consumers must collect
            # output_item.done. The proxy must not synthesize response contents.
            emit({"type": "response.completed", "response": response(rid)})

        try:
            for raw in connection:
                event = json.loads(raw)
                received.append(event)
                if "stream_id" in event:
                    emit(
                        {
                            "type": "error",
                            "status": 400,
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Unsupported parameter: stream_id",
                            },
                        }
                    )
                    continue
                if event["type"] == "response.steer":
                    assert event["previous_response_id"] == "resp_steering"
                    assert event["input"] == "Reply exactly STEER_OK"
                    emit(
                        {
                            "type": "response.steer.accepted",
                            "steer": {
                                "id": "steer_1",
                                "previous_response_id": "resp_steering",
                            },
                        }
                    )
                    emit(
                        {
                            "type": "response.incomplete",
                            "response": {
                                **response("resp_steering", "incomplete"),
                                "incomplete_details": {"reason": "steered"},
                            },
                        }
                    )
                    finish("resp_successor", "STEER_OK")
                    continue
                assert event["type"] == "response.create"
                assert event["store"] is False and "stream" not in event
                if _flatten_text(event["input"]) == "steer-start":
                    emit(
                        {
                            "type": "response.created",
                            "response": response("resp_steering", "in_progress"),
                        }
                    )
                elif event.get("previous_response_id") == "resp_async":
                    assert event["input"] == [{"role": "user", "content": "pending"}]
                    finish("resp_pending", "PENDING_OK")
                elif event.get("previous_response_id") == "resp_pending":
                    assert event["input"][0]["call_id"] == "call_async"
                    assert event["input"][0]["output"] == "VALUE_OK"
                    finish("resp_result", "VALUE_OK")
                elif event.get("tools"):
                    assert event["tools"][0]["async"] is True
                    finish("resp_async", "INDEPENDENT_OK", call=True)
                else:
                    finish("resp_basic", "WS_OK")
        except ConnectionClosed:
            pass
        except Exception as error:
            failures.append(str(error))
            raise

    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "fake-ws-token",
                    "account_id": "fake-ws-account",
                },
            }
        )
    )
    with serve(handler, "127.0.0.1", 0) as backend:
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        port = backend.socket.getsockname()[1]
        try:
            with spawned_server(
                go_binary,
                tmp_path,
                backend_url=f"http://127.0.0.1:{port}",
                auth_json=auth,
            ) as base:
                yield base, received
        finally:
            backend.shutdown()
            thread.join(timeout=5)
    assert not failures, failures


async def test_contract_websocket_async_tools_and_upstream_context(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    from openai.types.responses import ToolParam

    from tests.websocket_helpers import (
        API_KEY,
        ASYNC_TOOL,
        MODEL,
        output_text,
        receive_terminal,
    )

    base, received = websocket_runtime
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=base + "/v1") as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(
            model=MODEL, input="initial", tools=[cast(ToolParam, ASYNC_TOOL)]
        )
        first, items, _ = await receive_terminal(connection)
        call = next(item for item in items if item["type"] == "function_call")
        assert call["async"] is True and call["future_field"] == "preserved"
        assert output_text(items) == "INDEPENDENT_OK"
        await connection.response.create(
            model=MODEL, input="pending", previous_response_id=first["id"]
        )
        pending, items, _ = await receive_terminal(connection)
        assert output_text(items) == "PENDING_OK"
        await connection.response.create(
            model=MODEL,
            previous_response_id=pending["id"],
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "VALUE_OK",
                }
            ],
        )
        result, items, _ = await receive_terminal(connection)
        assert result["status"] == "completed" and output_text(items) == "VALUE_OK"
        with pytest.raises(NotFoundError):
            await client.responses.retrieve(first["id"])
    assert len(received) == 3


async def test_contract_websocket_steering(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    from tests.websocket_helpers import API_KEY, MODEL, output_text, receive_terminal

    base, _ = websocket_runtime
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=base + "/v1") as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(model=MODEL, input="steer-start")
        created = await connection.recv()
        assert created.type == "response.created"
        await connection.response.steer(
            previous_response_id=created.model_dump()["response"]["id"],
            input="Reply exactly STEER_OK",
        )
        interrupted, _, types = await receive_terminal(connection)
        assert "response.steer.accepted" in types
        assert interrupted["incomplete_details"]["reason"] == "steered"
        successor, items, _ = await receive_terminal(connection)
        assert successor["id"] != interrupted["id"]
        assert successor["status"] == "completed" and output_text(items) == "STEER_OK"


def test_contract_websocket_sync_and_background_rejection(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    from openai import BadRequestError, OpenAI

    from tests.websocket_helpers import API_KEY, MODEL

    base, received = websocket_runtime
    with OpenAI(api_key=API_KEY, base_url=base + "/v1") as client:
        with pytest.raises(BadRequestError) as error:
            client.responses.create(model=MODEL, input="hello", background=True)
        assert error.value.code == "unsupported_parameter"
        assert error.value.param == "background"
        with client.responses.connect(max_retries=0) as connection:
            connection.response.create(model=MODEL, input="hello", background=True)
            rejected = connection.recv()
            assert (
                rejected.type == "error"
                and rejected.model_dump()["error"]["code"] == "unsupported_parameter"
            )
            connection.response.create(model=MODEL, input="hello", background=False)
            text = ""
            completed = False
            for event in connection:
                data = event.model_dump(by_alias=True)
                if (
                    data["type"] == "response.output_item.done"
                    and data["item"]["type"] == "message"
                ):
                    text += data["item"]["content"][0]["text"]
                if event.type == "response.completed":
                    assert text == "WS_OK"
                    completed = True
                    break
            assert completed
    assert len(received) == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_contract_http_async_history(
    contract_client: AsyncOpenAI, stream: bool
) -> None:
    from openai.types.responses import ResponseInputParam

    first_items: list[dict[str, Any]] = []
    metadata = {"contract": "async_history"}
    if stream:
        events = await contract_client.responses.create(
            model="gpt-6-astra", input="initial", metadata=metadata, stream=True
        )
        first_id = ""
        async for event in events:
            data = event.model_dump(by_alias=True, exclude_none=True)
            if data["type"] == "response.output_item.done":
                first_items.append(data["item"])
            if data["type"] == "response.completed":
                first_id = data["response"]["id"]
        assert first_id
    else:
        response = await contract_client.responses.create(
            model="gpt-6-astra", input="initial", metadata=metadata
        )
        first_id = response.id
        first_items = [
            item.model_dump(by_alias=True, exclude_none=True)
            for item in response.output
        ]
    assert first_items[0]["async"] is True
    for mode in ["stored", "explicit"]:
        kwargs: dict[str, Any] = (
            {"previous_response_id": first_id, "input": "pending"}
            if mode == "stored"
            else {
                "input": [
                    {"role": "user", "content": "initial"},
                    *first_items,
                    {"role": "user", "content": "pending"},
                ],
            }
        )
        pending = await contract_client.responses.create(
            model="gpt-6-astra", metadata=metadata, **kwargs
        )
        assert pending.output_text == "PENDING_OK"
        final = await contract_client.responses.create(
            model="gpt-6-astra",
            metadata=metadata,
            previous_response_id=pending.id,
            input=cast(
                ResponseInputParam,
                [
                    {
                        "type": "function_call_output",
                        "call_id": "call_history",
                        "output": [{"type": "input_text", "text": "value"}],
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_custom",
                        "output": "custom result",
                    },
                ],
            ),
        )
        assert final.output_text == "FINAL_OK"


async def test_contract_websocket_upstream_named_stream_error(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    from tests.websocket_helpers import API_KEY, MODEL

    base, received = websocket_runtime
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=base + "/v1") as client,
        client.responses.connect(max_retries=0) as connection,
    ):
        await connection.response.create(model=MODEL, stream_id="alpha", input="hello")
        event = (await connection.recv()).model_dump(by_alias=True, exclude_none=True)
        assert event["type"] == "error" and event["status"] == 400
        assert event["error"] == {
            "type": "invalid_request_error",
            "message": "Unsupported parameter: stream_id",
        }
    assert received[0]["stream_id"] == "alpha"


def test_contract_saturated_connections_return_sdk_error(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    from contextlib import ExitStack

    from openai import InternalServerError, OpenAI

    from tests.websocket_helpers import API_KEY

    base, _ = websocket_runtime
    with (
        OpenAI(
            api_key=API_KEY, base_url=base + "/v1", max_retries=0, timeout=3
        ) as client,
        ExitStack() as connections,
    ):
        for _ in range(10):
            connections.enter_context(client.responses.connect(max_retries=0))
        with pytest.raises(InternalServerError) as error:
            client.models.list()
        assert error.value.status_code == 503
        assert error.value.response.headers["retry-after"] == "1"
        assert "concurrency limit" in error.value.message


async def test_contract_websocket_saturation_recovers_under_load(
    websocket_runtime: tuple[str, list[dict[str, Any]]],
) -> None:
    import asyncio
    from contextlib import AsyncExitStack

    from openai import InternalServerError

    from tests.websocket_helpers import API_KEY, MODEL, receive_terminal

    base, _ = websocket_runtime
    async with AsyncOpenAI(
        api_key=API_KEY, base_url=base + "/v1", max_retries=0, timeout=5
    ) as client:

        async def overloaded_request() -> None:
            with pytest.raises(InternalServerError) as error:
                await client.responses.create(model=MODEL, input="overloaded")
            assert error.value.status_code == 503
            assert error.value.response.headers["retry-after"] == "1"

        for cycle in range(10):
            async with AsyncExitStack() as connections:
                for _ in range(10):
                    await connections.enter_async_context(
                        client.responses.connect(max_retries=0)
                    )
                await asyncio.wait_for(
                    asyncio.gather(*(overloaded_request() for _ in range(50))), 10
                )
                async with httpx.AsyncClient() as health:
                    assert (await health.get(base + "/healthz")).status_code == 200
            # Exercise model traffic again after all capacity is released.
            async with client.responses.connect(max_retries=0) as connection:
                await connection.response.create(model=MODEL, input="recovery")
                response, _, _ = await receive_terminal(connection)
                assert response["status"] == "completed"
            print(
                f"saturation cycle {cycle + 1}/10: 50 rejected, recovery completed",
                flush=True,
            )
