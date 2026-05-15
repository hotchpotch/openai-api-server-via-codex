from __future__ import annotations

import asyncio
import base64
import os
import socket
import subprocess
import sys

import httpx
import pytest
from openai import AsyncOpenAI


ONE_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_CODEX_LIVE_TESTS") != "1",
    reason="Set RUN_CODEX_LIVE_TESTS=1 to call the real Codex backend.",
)
async def test_live_openai_client_requests_through_uvicorn_server() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "openai_api_server_via_codex.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        await _wait_for_server(base_url)
        client = AsyncOpenAI(api_key="test", base_url=f"{base_url}/v1")
        try:
            models = await client.models.list()
            model = os.environ.get("OPENAI_VIA_CODEX_TEST_MODEL") or models.data[0].id

            first = await client.responses.create(
                model=model,
                input="Reply with exactly: codex-ok",
                reasoning={"effort": "low"},
            )
            assert first.output_text.strip()

            second = await client.responses.create(
                model=model,
                input="What exact token did you just output?",
                previous_response_id=first.id,
                reasoning={"effort": "low"},
            )
            assert second.output_text.strip()

            response_stream = await client.responses.create(
                model=model,
                input="Stream a short reply with the token: response-stream-ok",
                stream=True,
                reasoning={"effort": "low"},
            )
            response_stream_text: list[str] = []
            response_stream_completed = False
            async for event in response_stream:
                if event.type == "response.output_text.delta":
                    response_stream_text.append(str(getattr(event, "delta")))
                elif event.type == "response.completed":
                    response_stream_completed = True
            assert "".join(response_stream_text).strip()
            assert response_stream_completed

            chat = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer with concise text only."},
                    {"role": "user", "content": "Reply with exactly: chat-ok"},
                ],
                reasoning_effort="low",
            )
            assert chat.choices[0].message.content

            chat_stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer with concise text only."},
                    {
                        "role": "user",
                        "content": "Stream a short reply with the token: chat-stream-ok",
                    },
                ],
                stream=True,
                reasoning_effort="low",
            )
            chat_stream_text: list[str] = []
            chat_stream_finished = False
            async for chunk in chat_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chat_stream_text.append(chunk.choices[0].delta.content)
                if chunk.choices and chunk.choices[0].finish_reason:
                    chat_stream_finished = True
            assert "".join(chat_stream_text).strip()
            assert chat_stream_finished

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

            generated = await client.images.generate(
                model="gpt-image-2",
                prompt=(
                    "A clean pixel art image of a ramen bowl in a red bowl, "
                    "with noodles, egg, nori, and green scallions. No text."
                ),
                size="1024x1024",
                quality="medium",
                output_format="png",
            )
            assert generated.data
            image_b64 = generated.data[0].b64_json
            assert image_b64
            image_bytes = base64.b64decode(image_b64)
            width, height = _png_dimensions(image_bytes)
            assert width > 0
            assert height > 0

            image_description = await client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Describe the main subject in this generated "
                                    "image in one short sentence."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{image_b64}",
                                "detail": "low",
                            },
                        ],
                    }
                ],
                reasoning={"effort": "low"},
            )
            description = image_description.output_text.lower()
            print(
                "live image generation "
                f"dimensions={width}x{height} description={description!r}"
            )
            assert any(word in description for word in ("ramen", "noodle", "bowl")), (
                description,
                width,
                height,
            )

            async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as direct:
                proxy_attempts = [
                    (
                        "POST",
                        "/v1/tokenizer",
                        {"model": model, "input": "live proxy tokenizer marker"},
                    ),
                    (
                        "POST",
                        "/v1/responses/compact",
                        {
                            "model": model,
                            "input": [
                                {
                                    "role": "user",
                                    "content": "live proxy compact marker",
                                }
                            ],
                        },
                    ),
                    (
                        "POST",
                        "/v1/embeddings",
                        {"model": model, "input": "live proxy embedding marker"},
                    ),
                ]
                for method, path, payload in proxy_attempts:
                    proxied = await direct.request(
                        method,
                        path,
                        headers={
                            "Authorization": "Bearer test",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    print(
                        "live proxy "
                        f"{method} {path} status={proxied.status_code} "
                        f"content_type={proxied.headers.get('content-type')} "
                        f"body={proxied.text[:240]!r}"
                    )
                    assert proxied.headers.get("x-openai-via-codex-proxy") == "codex-http"

                traversal_probes = [
                    "/v1/%2e%2e/auth/me",
                    "/v1/foo/%2e%2e/%2e%2e/auth/me",
                ]
                for probe in traversal_probes:
                    rejected = await direct.get(probe)
                    print(
                        "live proxy traversal "
                        f"GET {probe} status={rejected.status_code} "
                        f"body={rejected.text[:200]!r}"
                    )
                    assert rejected.status_code == 400
                    assert (
                        rejected.headers.get("x-openai-via-codex-proxy")
                        is None
                    )

                healthz = await direct.get("/healthz")
                assert healthz.status_code == 200
        finally:
            await client.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _png_dimensions(image_bytes: bytes) -> tuple[int, int]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not image_bytes.startswith(signature) or image_bytes[12:16] != b"IHDR":
        raise AssertionError("generated image is not a PNG")
    width = int.from_bytes(image_bytes[16:20], "big")
    height = int.from_bytes(image_bytes[20:24], "big")
    return width, height


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
    raise AssertionError("uvicorn server did not become ready")
