from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from openai.resources.responses.responses import AsyncResponsesConnection

API_KEY = "websocket-e2e-client-key"
MODEL = "gpt-6-astra"
ASYNC_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "lookup",
    "async": True,
    "description": "Look up a demo value. The application returns the result later.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
}


@contextmanager
def spawned_server(
    binary: Path,
    work: Path,
    *,
    backend_url: str | None = None,
    auth_json: Path | None = None,
    host: str = "127.0.0.1",
) -> Iterator[str]:
    config = work / "config.toml"
    config.write_text("")
    args = [
        str(binary),
        "serve",
        "--config",
        str(config),
        "--host",
        host,
        "--port",
        "0",
        "--default-model",
        MODEL,
        "--timeout",
        "120",
        "--max-stored-items",
        "100",
    ]
    if backend_url is not None:
        args.extend(["--backend-base-url", backend_url])
    if auth_json is not None:
        args.extend(["--auth-json", str(auth_json)])
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_VIA_CODEX_")}
    env["OPENAI_VIA_CODEX_API_KEY"] = API_KEY
    log_path = work / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(
                        "Go server exited during auth preflight/startup"
                    )
                match = re.search(r"listening on (http://\S+)", log_path.read_text())
                if match:
                    yield match[1]
                    return
                time.sleep(0.05)
            raise AssertionError("Go server startup timed out")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("Go server did not shut down within 10 seconds")


async def receive_terminal(
    connection: AsyncResponsesConnection,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    types: list[str] = []
    while True:
        event = await asyncio.wait_for(connection.recv(), timeout=120)
        data = event.model_dump(exclude_none=True, by_alias=True)
        kind = data["type"]
        types.append(kind)
        assert kind not in {"error", "response.failed", "response.steer.failed"}, (
            kind,
            data.get("code"),
        )
        if kind == "response.output_item.done":
            items.append(data["item"])
        if kind in {"response.completed", "response.incomplete"}:
            return data["response"], items, types


def output_text(items: list[dict[str, Any]]) -> str:
    return "".join(
        part.get("text", "")
        for item in items
        if item["type"] == "message"
        for part in item.get("content", [])
        if part["type"] == "output_text"
    )


def print_observation(
    label: str, response: dict[str, Any], items: list[dict[str, Any]], types: list[str]
) -> None:
    # Never print full response objects: they include encrypted reasoning.
    print(
        json.dumps(
            {
                "case": label,
                "status": response["status"],
                "events": types,
                "text": output_text(items),
                "tools": [
                    {
                        k: item[k]
                        for k in ("type", "name", "arguments", "input", "async")
                        if k in item
                    }
                    for item in items
                    if item["type"] in {"function_call", "custom_tool_call"}
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
