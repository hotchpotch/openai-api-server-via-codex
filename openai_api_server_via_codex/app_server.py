from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth import BorrowKeyError, CodexAuthConfig, borrow_codex_key
from .backend import DEFAULT_MODELS, CodexBackendError, _collect_streamed_response


CODEX_BACKEND_APP_SERVER = "codex-app-server"
CODEX_BIN_ENV = "OPENAI_VIA_CODEX_CODEX_BIN"
CODEX_APP_SERVER_CWD_ENV = "OPENAI_VIA_CODEX_APP_SERVER_CWD"


@dataclass(frozen=True)
class CodexAppServerConfig:
    codex_bin: str = "codex"
    cwd: Path | None = None
    timeout: float = 180.0
    auth_config: CodexAuthConfig | None = None


@dataclass(frozen=True)
class _ThreadBinding:
    thread_id: str
    model: str | None
    dynamic_tools_fingerprint: str


class JsonRpcCodexAppServerClient:
    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        cwd: Path | None = None,
        auth_config: CodexAuthConfig | None = None,
    ) -> None:
        self.codex_bin = codex_bin
        self.cwd = cwd
        self.auth_config = auth_config or CodexAuthConfig()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_task: asyncio.Task[None] | None = None
        self._transport_lock = asyncio.Lock()
        self._pending_notifications: deque[dict[str, Any]] = deque()

    async def initialize(self) -> None:
        await self._start()
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "openai_api_server_via_codex",
                    "title": "OpenAI API Server via Codex",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.notify("initialized", None)

    async def login_with_chatgpt_tokens(
        self, *, access_token: str, account_id: str
    ) -> None:
        await self.request(
            "account/login/start",
            {
                "type": "chatgptAuthTokens",
                "accessToken": access_token,
                "chatgptAccountId": account_id,
                "chatgptPlanType": None,
            },
        )

    async def thread_start(self, params: dict[str, Any]) -> dict[str, Any]:
        result = await self.request("thread/start", params)
        return _expect_dict(result, "thread/start")

    async def turn_start(
        self, *, thread_id: str, input_items: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {**params, "threadId": thread_id, "input": input_items}
        result = await self.request("turn/start", payload)
        return _expect_dict(result, "turn/start")

    async def model_list(self) -> dict[str, Any]:
        result = await self.request("model/list", {"includeHidden": False})
        return _expect_dict(result, "model/list")

    async def request(self, method: str, params: dict[str, Any] | None) -> Any:
        async with self._transport_lock:
            request_id = str(uuid.uuid4())
            await self._write_message(
                {"id": request_id, "method": method, "params": params or {}}
            )
            while True:
                message = await self._read_message()
                if "method" in message and "id" in message:
                    response = await self._handle_server_request(message)
                    await self._write_message({"id": message["id"], "result": response})
                    continue
                if "method" in message and "id" not in message:
                    self._pending_notifications.append(message)
                    continue
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise CodexBackendError(
                        _jsonrpc_error_message(method, message["error"])
                    )
                return message.get("result")

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:
        async with self._transport_lock:
            await self._write_message({"method": method, "params": params or {}})

    async def next_notification(self) -> dict[str, Any]:
        async with self._transport_lock:
            if self._pending_notifications:
                return self._pending_notifications.popleft()
            while True:
                message = await self._read_message()
                if "method" in message and "id" in message:
                    response = await self._handle_server_request(message)
                    await self._write_message({"id": message["id"], "result": response})
                    continue
                if "method" in message and "id" not in message:
                    return message

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
            await process.stdin.wait_closed()
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2)
        except Exception:
            process.kill()
            await process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()

    async def _start(self) -> None:
        if self._process is not None:
            return
        env = os.environ.copy()
        self._process = await asyncio.create_subprocess_exec(
            self.codex_bin,
            "app-server",
            "--listen",
            "stdio://",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while line := await process.stderr.readline():
            self._stderr_lines.append(line.decode(errors="replace").rstrip("\n"))

    async def _write_message(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexBackendError("Codex app-server is not running")
        process.stdin.write((json.dumps(payload) + "\n").encode())
        await process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise CodexBackendError("Codex app-server is not running")
        line = await process.stdout.readline()
        if not line:
            stderr = "\n".join(self._stderr_lines)
            raise CodexBackendError(
                f"Codex app-server closed stdout. stderr_tail={stderr[:2000]}"
            )
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexBackendError(
                f"Invalid Codex app-server JSON-RPC line: {line!r}"
            ) from exc
        if not isinstance(message, dict):
            raise CodexBackendError(f"Invalid Codex app-server payload: {message!r}")
        return message

    async def _handle_server_request(self, message: dict[str, Any]) -> dict[str, Any]:
        method = message.get("method")
        if method == "account/chatgptAuthTokens/refresh":
            try:
                access_token, account_id = await asyncio.to_thread(
                    borrow_codex_key, self.auth_config.auth_json
                )
            except BorrowKeyError as exc:
                raise CodexBackendError(str(exc), status_code=401) from exc
            if not account_id:
                raise CodexBackendError(
                    "Codex app-server requested token refresh but account id is missing.",
                    status_code=401,
                )
            return {
                "accessToken": access_token,
                "chatgptAccountId": account_id,
                "chatgptPlanType": None,
            }
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            return {
                "decision": "decline",
                "reason": "OpenAI-compatible app-server adapter does not grant native approvals.",
            }
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        if method == "item/tool/call":
            params = _dict_value(message.get("params"))
            tool_name = str(params.get("tool") or "tool")
            call_id = str(params.get("callId") or params.get("call_id") or "unknown")
            return {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": (
                            f"Tool call {call_id} for {tool_name} was surfaced to "
                            "the OpenAI-compatible client. Provide the result as "
                            "function_call_output in a later request."
                        ),
                    }
                ],
                "success": False,
            }
        return {}


class CodexAppServerBackend:
    supports_native_sessions = True

    def __init__(
        self,
        *,
        config: CodexAppServerConfig | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or CodexAppServerConfig()
        self._client_factory = client_factory or (
            lambda: JsonRpcCodexAppServerClient(
                codex_bin=self.config.codex_bin,
                cwd=self.config.cwd,
                auth_config=self.config.auth_config,
            )
        )
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._bindings: dict[str, _ThreadBinding] = {}

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await _collect_streamed_response(self.stream_response(payload), payload)

    async def stream_response(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async with self._turn_lock:
            client = await self._ensure_client()
            thread_id, new_thread = await self._resolve_thread(client, payload)
            input_items = response_input_to_app_server_input(payload.get("input") or [])
            turn_response = await client.turn_start(
                thread_id=thread_id,
                input_items=input_items,
                params=_turn_start_params(payload),
            )
            turn = _expect_dict(turn_response.get("turn"), "turn/start turn")
            turn_id = str(turn.get("id") or f"turn_{int(time.time() * 1000)}")
            response_id = f"resp_{_safe_id(turn_id)}"
            self._bindings[response_id] = _ThreadBinding(
                thread_id=thread_id,
                model=str(payload.get("model") or "") or None,
                dynamic_tools_fingerprint=_dynamic_tools_fingerprint(
                    response_tools_to_app_server_dynamic_tools(payload.get("tools"))
                ),
            )
            created_at = time.time()
            output_item_id = f"msg_{_safe_id(turn_id)}"
            text_parts: list[str] = []
            completed_text_item: dict[str, Any] | None = None
            output_items: list[dict[str, Any] | None] = []
            function_call_indexes: dict[str, int] = {}
            emitted_function_call_done: set[str] = set()

            yield {
                "type": "response.created",
                "sequence_number": 0,
                "response": _response_payload(
                    response_id=response_id,
                    created_at=created_at,
                    model=payload.get("model"),
                    output=[],
                    status="in_progress",
                    request_payload=payload,
                ),
            }

            sequence_number = 1
            async for event in _turn_notifications(client, turn_id):
                method = event.get("method")
                params = _event_params(event)
                if method == "item/agentMessage/delta":
                    delta = str(params.get("delta") or "")
                    if not delta:
                        continue
                    text_parts.append(delta)
                    output_item_id = str(params.get("itemId") or output_item_id)
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": output_item_id,
                        "delta": delta,
                        "logprobs": [],
                    }
                    sequence_number += 1
                elif method == "item/started":
                    item = params.get("item")
                    if isinstance(item, dict) and item.get("type") == "dynamicToolCall":
                        response_item = _dynamic_tool_call_item_to_response_item(
                            item, status="in_progress"
                        )
                        call_id = response_item["call_id"]
                        if call_id not in function_call_indexes:
                            output_index = len(output_items)
                            function_call_indexes[call_id] = output_index
                            output_items.append(response_item)
                            yield {
                                "type": "response.output_item.added",
                                "sequence_number": sequence_number,
                                "output_index": output_index,
                                "item": {**response_item, "arguments": ""},
                            }
                            sequence_number += 1
                            arguments = str(response_item.get("arguments") or "")
                            if arguments:
                                yield {
                                    "type": "response.function_call_arguments.delta",
                                    "sequence_number": sequence_number,
                                    "output_index": output_index,
                                    "item_id": response_item["id"],
                                    "delta": arguments,
                                }
                                sequence_number += 1
                elif method == "item/completed":
                    item = params.get("item")
                    if isinstance(item, dict) and item.get("type") == "agentMessage":
                        completed_text_item = _agent_message_item_to_response_item(item)
                        output_item_id = str(completed_text_item["id"])
                    elif isinstance(item, dict) and item.get("type") == "dynamicToolCall":
                        response_item = _dynamic_tool_call_item_to_response_item(
                            item, status="completed"
                        )
                        call_id = response_item["call_id"]
                        output_index = function_call_indexes.setdefault(
                            call_id, len(output_items)
                        )
                        if output_index == len(output_items):
                            output_items.append(response_item)
                        else:
                            output_items[output_index] = response_item
                        if call_id not in emitted_function_call_done:
                            yield {
                                "type": "response.output_item.done",
                                "sequence_number": sequence_number,
                                "output_index": output_index,
                                "item": response_item,
                            }
                            sequence_number += 1
                            emitted_function_call_done.add(call_id)
                    elif isinstance(item, dict):
                        response_item = _raw_response_item_to_response_item(item)
                        if response_item and response_item.get("type") == "function_call":
                            call_id = response_item["call_id"]
                            output_index = function_call_indexes.setdefault(
                                call_id, len(output_items)
                            )
                            if output_index == len(output_items):
                                output_items.append(response_item)
                            else:
                                output_items[output_index] = response_item
                            if call_id not in emitted_function_call_done:
                                yield {
                                    "type": "response.output_item.done",
                                    "sequence_number": sequence_number,
                                    "output_index": output_index,
                                    "item": response_item,
                                }
                                sequence_number += 1
                                emitted_function_call_done.add(call_id)
                elif method == "rawResponseItem/completed":
                    response_item = _raw_response_item_to_response_item(params.get("item"))
                    if response_item and response_item.get("type") == "function_call":
                        call_id = response_item["call_id"]
                        output_index = function_call_indexes.setdefault(
                            call_id, len(output_items)
                        )
                        if output_index == len(output_items):
                            output_items.append(response_item)
                        else:
                            output_items[output_index] = response_item
                        if call_id not in emitted_function_call_done:
                            yield {
                                "type": "response.output_item.done",
                                "sequence_number": sequence_number,
                                "output_index": output_index,
                                "item": response_item,
                            }
                            sequence_number += 1
                            emitted_function_call_done.add(call_id)
                elif method == "turn/completed":
                    turn_payload = params.get("turn")
                    if (
                        isinstance(turn_payload, dict)
                        and turn_payload.get("status") == "failed"
                    ):
                        error = turn_payload.get("error")
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else "Codex app-server turn failed."
                        )
                        raise CodexBackendError(str(message))
                    break

            final_output = [item for item in output_items if item is not None]
            if completed_text_item is None:
                if not final_output:
                    completed_text_item = _text_response_item(
                        item_id=output_item_id,
                        text="".join(text_parts),
                    )
            elif not text_parts:
                final_text = _response_item_text(completed_text_item)
                if final_text:
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "output_index": 0,
                        "content_index": 0,
                        "item_id": str(completed_text_item["id"]),
                        "delta": final_text,
                        "logprobs": [],
                    }
                    sequence_number += 1

            if completed_text_item is not None:
                final_output.append(completed_text_item)
                yield {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": len(final_output) - 1,
                    "item": completed_text_item,
                }
                sequence_number += 1
            yield {
                "type": "response.completed",
                "sequence_number": sequence_number,
                "response": _response_payload(
                    response_id=response_id,
                    created_at=created_at,
                    model=payload.get("model"),
                    output=final_output,
                    previous_response_id=payload.get("previous_response_id"),
                    request_payload=payload,
                ),
            }

    async def list_models(self) -> list[str]:
        try:
            client = await self._ensure_client(login=False)
            response = await client.model_list()
        except Exception:
            return DEFAULT_MODELS
        models = [
            str(model.get("id") or model.get("model"))
            for model in response.get("data", [])
            if isinstance(model, dict) and not model.get("hidden")
        ]
        return [model for model in models if model] or DEFAULT_MODELS

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def _ensure_client(self, *, login: bool = True) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            client = self._client_factory()
            await asyncio.wait_for(client.initialize(), timeout=self.config.timeout)
            if login:
                await self._login(client)
            self._client = client
            return client

    async def _login(self, client: Any) -> None:
        auth_config = self.config.auth_config or CodexAuthConfig()
        try:
            access_token, account_id = await asyncio.to_thread(
                borrow_codex_key, auth_config.auth_json
            )
        except BorrowKeyError as exc:
            raise CodexBackendError(str(exc), status_code=401) from exc
        if not account_id:
            raise CodexBackendError(
                "Codex auth did not include a ChatGPT account id.", status_code=401
            )
        await client.login_with_chatgpt_tokens(
            access_token=access_token,
            account_id=account_id,
        )

    async def _resolve_thread(
        self, client: Any, payload: dict[str, Any]
    ) -> tuple[str, bool]:
        previous_response_id = payload.get("previous_response_id")
        if previous_response_id:
            binding = self._bindings.get(str(previous_response_id))
            if binding is None:
                raise CodexBackendError(
                    f"Unknown previous_response_id for Codex app-server session: {previous_response_id}",
                    status_code=404,
                )
            return binding.thread_id, False

        thread_response = await client.thread_start(_thread_start_params(payload, self.config))
        thread = _expect_dict(thread_response.get("thread"), "thread/start thread")
        thread_id = thread.get("id")
        if not thread_id:
            raise CodexBackendError("Codex app-server thread/start did not return a thread id.")
        return str(thread_id), True


def response_input_to_app_server_input(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"type": "text", "text": input_value}]
    if not isinstance(input_value, list):
        input_value = [input_value]

    items: list[dict[str, Any]] = []
    for value in input_value:
        if isinstance(value, str):
            items.append({"type": "text", "text": value})
        elif isinstance(value, dict):
            items.extend(_response_item_to_app_server_items(value))
    return items


def _response_item_to_app_server_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("type") == "function_call":
        call_id = value.get("call_id") or value.get("id") or "unknown"
        name = value.get("name") or "function"
        arguments = value.get("arguments")
        if not isinstance(arguments, str):
            arguments = _json_arguments(arguments)
        return [
            {
                "type": "text",
                "text": f"assistant tool call {call_id} {name}: {arguments}",
            }
        ]

    if value.get("type") == "function_call_output":
        return [
            {
                "type": "text",
                "text": f"tool {value.get('call_id') or 'unknown'}: {value.get('output') or ''}",
            }
        ]

    role = str(value.get("role") or "user")
    content = value.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": f"{role}: {content}"}]
    if not isinstance(content, list):
        return [{"type": "text", "text": f"{role}: {content or ''}"}]

    items: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "text"}:
            items.append({"type": "text", "text": f"{role}: {part.get('text') or ''}"})
        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url:
                items.append({"type": "image", "url": str(image_url)})
    return items


def _thread_start_params(
    payload: dict[str, Any], config: CodexAppServerConfig
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": payload.get("model"),
        "baseInstructions": payload.get("instructions"),
        "serviceName": "OpenAI API Server via Codex",
        "experimentalRawEvents": True,
        "persistExtendedHistory": True,
    }
    dynamic_tools = response_tools_to_app_server_dynamic_tools(payload.get("tools"))
    if dynamic_tools:
        params["dynamicTools"] = dynamic_tools
    if config.cwd is not None:
        params["cwd"] = str(config.cwd)
    return {key: value for key, value in params.items() if value is not None}


def _turn_start_params(payload: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        params["effort"] = str(reasoning["effort"])
    if payload.get("model"):
        params["model"] = payload["model"]
    if payload.get("service_tier"):
        params["serviceTier"] = payload["service_tier"]
    text_config = payload.get("text")
    if isinstance(text_config, dict) and isinstance(text_config.get("format"), dict):
        params["outputSchema"] = text_config["format"].get("schema")
    return params


async def _turn_notifications(
    client: Any, turn_id: str
) -> AsyncIterator[dict[str, Any]]:
    while True:
        event = await client.next_notification()
        params = _event_params(event)
        event_turn_id = params.get("turnId")
        if event.get("method") == "turn/completed":
            turn = _dict_value(params.get("turn"))
            event_turn_id = turn.get("id")
        if event_turn_id is not None and str(event_turn_id) != turn_id:
            continue
        yield event
        if event.get("method") == "turn/completed":
            return


def _response_payload(
    *,
    response_id: str,
    created_at: float,
    model: Any,
    output: list[dict[str, Any]],
    status: str = "completed",
    previous_response_id: Any = None,
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_payload = request_payload or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": request_payload.get("tool_choice") or "auto",
        "tools": request_payload.get("tools") or [],
        "previous_response_id": previous_response_id,
        "usage": None,
    }


def _event_params(event: dict[str, Any]) -> dict[str, Any]:
    return _dict_value(event.get("params"))


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _agent_message_item_to_response_item(item: dict[str, Any]) -> dict[str, Any]:
    return _text_response_item(
        item_id=str(item.get("id") or f"msg_{int(time.time() * 1000)}"),
        text=str(item.get("text") or ""),
        phase=str(item["phase"]) if item.get("phase") else "final_answer",
    )


def response_tools_to_app_server_dynamic_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    dynamic_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if tool.get("type") == "function" else None
        if isinstance(function, dict):
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters")
        else:
            if tool.get("type") != "function":
                continue
            name = tool.get("name")
            description = tool.get("description")
            parameters = tool.get("parameters")
        if not name:
            continue
        dynamic_tools.append(
            {
                "name": str(name),
                "description": str(description or ""),
                "inputSchema": parameters or {"type": "object", "properties": {}},
                "deferLoading": False,
            }
        )
    return dynamic_tools


def _dynamic_tools_fingerprint(dynamic_tools: list[dict[str, Any]]) -> str:
    return json.dumps(dynamic_tools, sort_keys=True, separators=(",", ":"))


def _dynamic_tool_call_item_to_response_item(
    item: dict[str, Any], *, status: str
) -> dict[str, Any]:
    call_id = str(item.get("id") or f"call_{int(time.time() * 1000)}")
    return {
        "id": call_id,
        "type": "function_call",
        "call_id": call_id,
        "name": str(item.get("tool") or ""),
        "arguments": _json_arguments(item.get("arguments")),
        "status": status,
    }


def _raw_response_item_to_response_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return None
    call_id = str(item.get("call_id") or item.get("id") or f"call_{int(time.time() * 1000)}")
    return {
        "id": str(item.get("id") or call_id),
        "type": "function_call",
        "call_id": call_id,
        "name": str(item.get("name") or ""),
        "arguments": _json_arguments(item.get("arguments")),
        "status": str(item.get("status") or "completed"),
    }


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _text_response_item(
    *, item_id: str, text: str, phase: str = "final_answer"
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _response_item_text(item: dict[str, Any]) -> str:
    texts: list[str] = []
    for content in item.get("content") or []:
        if isinstance(content, dict) and content.get("type") == "output_text":
            texts.append(str(content.get("text") or ""))
    return "".join(texts)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in value)


def _expect_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CodexBackendError(f"Codex app-server {label} response must be an object.")
    return value


def _jsonrpc_error_message(method: str, error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return f"Codex app-server {method} failed: {message}"
    return f"Codex app-server {method} failed: {error!r}"
