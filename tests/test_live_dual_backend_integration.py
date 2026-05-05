from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator

import httpx
import pytest
from openai import AsyncOpenAI


ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
RED_SQUARE_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJ0lEQVR42u3NsQkAAAjAsP7/tF7hIASyp6lTCQQCgUAgEAgEgi/BAjLD/C5w/SM9AAAAAElFTkSuQmCC"
)


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
