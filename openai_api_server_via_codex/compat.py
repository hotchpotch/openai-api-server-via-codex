from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, TypeAlias


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
DEFAULT_MAX_STORED_ITEMS = 1000

_ChatCompletionPage: TypeAlias = tuple[list[dict[str, Any]], bool]


@dataclass(slots=True)
class StoredResponse:
    effective_input: list[Any]
    context_items: list[Any]
    response: dict[str, Any]


@dataclass(slots=True)
class StoredChatCompletion:
    completion: dict[str, Any]
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


class ResponseStore:
    def __init__(self, *, max_entries: int = DEFAULT_MAX_STORED_ITEMS) -> None:
        self.max_entries = max(0, int(max_entries))
        self._responses: OrderedDict[str, StoredResponse] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self._responses)

    def get(self, response_id: str) -> StoredResponse | None:
        return self._responses.get(response_id)

    def delete(self, response_id: str) -> bool:
        return self._responses.pop(response_id, None) is not None

    def cancel(self, response_id: str) -> StoredResponse | None:
        stored = self._responses.get(response_id)
        if stored is None:
            return None
        stored.response["status"] = "cancelled"
        return stored

    def remember(
        self,
        response_id: str,
        *,
        effective_input: list[Any],
        response: dict[str, Any],
    ) -> None:
        if self.max_entries <= 0:
            return
        self._responses.pop(response_id, None)
        self._responses[response_id] = StoredResponse(
            effective_input=copy.deepcopy(effective_input),
            context_items=copy.deepcopy(effective_input)
            + response_output_as_input_messages(response),
            response=copy.deepcopy(response),
        )
        self._evict_oldest()

    def _evict_oldest(self) -> None:
        while len(self._responses) > self.max_entries:
            self._responses.popitem(last=False)


class ChatCompletionStore:
    def __init__(self, *, max_entries: int = DEFAULT_MAX_STORED_ITEMS) -> None:
        self.max_entries = max(0, int(max_entries))
        self._completions: dict[str, StoredChatCompletion] = {}
        self._order: list[str] = []

    @property
    def size(self) -> int:
        return len(self._completions)

    def get(self, completion_id: str) -> StoredChatCompletion | None:
        return self._completions.get(completion_id)

    def remember(
        self,
        completion_id: str,
        *,
        completion: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.max_entries <= 0:
            return
        metadata = copy.deepcopy(metadata or {})
        stored_completion = copy.deepcopy(completion)
        stored_completion["metadata"] = metadata
        self._completions[completion_id] = StoredChatCompletion(
            completion=stored_completion,
            messages=_chat_completion_store_messages(stored_completion),
            metadata=metadata,
        )
        if completion_id not in self._order:
            self._order.append(completion_id)
        self._evict_oldest()

    def update_metadata(
        self, completion_id: str, metadata: dict[str, Any] | None
    ) -> StoredChatCompletion | None:
        stored = self._completions.get(completion_id)
        if stored is None:
            return None
        stored.metadata = copy.deepcopy(metadata or {})
        stored.completion["metadata"] = copy.deepcopy(stored.metadata)
        return stored

    def delete(self, completion_id: str) -> bool:
        if self._completions.pop(completion_id, None) is None:
            return False
        self._order = [item_id for item_id in self._order if item_id != completion_id]
        return True

    def list(
        self,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        order: str | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> _ChatCompletionPage:
        completion_ids = list(self._order)
        if order == "desc":
            completion_ids.reverse()
        if after in completion_ids:
            completion_ids = completion_ids[completion_ids.index(after) + 1 :]

        items: list[dict[str, Any]] = []
        for completion_id in completion_ids:
            stored = self._completions[completion_id]
            completion = stored.completion
            if model and completion.get("model") != model:
                continue
            if metadata and not _metadata_matches(stored.metadata, metadata):
                continue
            items.append(copy.deepcopy(completion))

        has_more = False
        if limit is not None:
            has_more = len(items) > limit
            items = items[:limit]
        return items, has_more

    def _evict_oldest(self) -> None:
        while len(self._order) > self.max_entries:
            completion_id = self._order.pop(0)
            self._completions.pop(completion_id, None)


def prepare_response_payload(
    payload: dict[str, Any], *, default_model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    prepared = copy.deepcopy(payload)
    prepared["model"] = prepared.get("model") or default_model
    prepared["input"] = normalize_response_input(prepared.get("input", ""))
    prepared.setdefault("instructions", DEFAULT_INSTRUCTIONS)
    prepared.setdefault("store", False)
    return prepared


def normalize_response_input(input_value: Any) -> list[Any]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, list):
        return _normalize_response_input_list(input_value)
    if isinstance(input_value, dict):
        normalized = _normalize_response_input_item(input_value)
        return [] if normalized is None else [normalized]
    return [{"role": "user", "content": str(input_value)}]


def _normalize_response_input_list(input_value: list[Any]) -> list[Any]:
    normalized_items: list[Any] = []
    for item in input_value:
        normalized = _normalize_response_input_item(item)
        if normalized is not None:
            normalized_items.append(normalized)
    return normalized_items


def _normalize_response_input_item(item: Any) -> Any | None:
    if not isinstance(item, dict):
        return copy.deepcopy(item)

    item_type = item.get("type")
    if item_type == "reasoning":
        encrypted_content = item.get("encrypted_content")
        if isinstance(encrypted_content, str) and encrypted_content:
            summary = item.get("summary")
            return {
                "type": "reasoning",
                "encrypted_content": encrypted_content,
                "summary": summary if isinstance(summary, list) else [],
            }
        return None
    if item_type == "message" and item.get("role") == "assistant":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": _output_message_text(item),
        }
        if item.get("phase"):
            message["phase"] = item["phase"]
        return message
    if item_type == "function_call":
        function_call = _function_call_as_input_item(item)
        if function_call:
            return function_call
    if item_type == "function_call_output":
        return {
            "type": "function_call_output",
            "call_id": item.get("call_id") or "unknown",
            "output": _string_output(item.get("output")),
        }
    return copy.deepcopy(item)


def response_output_as_input_messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            function_call = _function_call_as_input_item(item)
            if function_call:
                messages.append(function_call)
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            text = _output_message_text(item)
            if not text:
                continue
            message: dict[str, Any] = {"role": "assistant", "content": text}
            if item.get("phase"):
                message["phase"] = item["phase"]
            messages.append(message)

    if not messages:
        text = extract_response_text(response)
        if text:
            messages.append({"role": "assistant", "content": text})
    return messages


def _chat_completion_store_messages(completion: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    completion_id = str(completion.get("id") or "chatcmpl")
    choices = completion.get("choices") or []
    if not isinstance(choices, list):
        return messages
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        store_message = copy.deepcopy(message)
        store_message["id"] = f"{completion_id}_msg_{choice.get('index', index)}"
        messages.append(store_message)
    return messages


def _metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(str(metadata.get(key)) == str(value) for key, value in expected.items())


def _function_call_as_input_item(item: dict[str, Any]) -> dict[str, Any] | None:
    call_id = item.get("call_id") or item.get("id")
    name = item.get("name")
    if not call_id or not name:
        return None
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        arguments = "{}"
    return {
        "type": "function_call",
        "call_id": str(call_id),
        "name": str(name),
        "arguments": arguments,
    }


def _string_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def chat_request_to_response_payload(
    payload: dict[str, Any], *, default_model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    response_payload: dict[str, Any] = {
        "model": payload.get("model") or default_model,
        "input": [],
        "store": False,
    }

    instructions = _collect_instructions(payload.get("messages") or [])
    response_payload["instructions"] = instructions or DEFAULT_INSTRUCTIONS

    response_payload["input"] = _chat_messages_to_response_input(
        payload.get("messages") or []
    )

    _copy_if_present(
        payload,
        response_payload,
        (
            "parallel_tool_calls",
            "service_tier",
            "temperature",
            "top_p",
            "user",
        ),
    )
    _copy_first_present(
        payload,
        response_payload,
        target="max_output_tokens",
        sources=("max_completion_tokens", "max_tokens"),
    )

    if payload.get("reasoning") is not None:
        response_payload["reasoning"] = copy.deepcopy(payload["reasoning"])
    elif payload.get("reasoning_effort") is not None:
        response_payload["reasoning"] = {"effort": payload["reasoning_effort"]}

    if payload.get("verbosity") is not None:
        response_payload.setdefault("text", {})["verbosity"] = payload["verbosity"]

    tools = _chat_tools_to_response_tools(payload.get("tools"))
    tools.extend(_chat_functions_to_response_tools(payload.get("functions")))
    if tools:
        response_payload["tools"] = tools

    tool_choice = _chat_tool_choice_to_response_tool_choice(payload.get("tool_choice"))
    if tool_choice is None:
        tool_choice = _chat_function_call_to_response_tool_choice(
            payload.get("function_call")
        )
    if tool_choice is not None:
        response_payload["tool_choice"] = tool_choice

    text_config = _chat_response_format_to_text_config(payload.get("response_format"))
    if text_config:
        existing_text = response_payload.setdefault("text", {})
        existing_text.update(text_config)

    return response_payload


def response_to_chat_completion(
    response: dict[str, Any],
    *,
    fallback_model: str = DEFAULT_MODEL,
    legacy_functions: bool = False,
    n: int = 1,
) -> dict[str, Any]:
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    created = int(response.get("created_at") or time.time())
    response_id = str(response.get("id") or f"resp_{created}")
    tool_calls = _response_function_calls_to_chat_tool_calls(response)
    legacy_function_call = (
        _response_function_calls_to_chat_function_call(response)
        if legacy_functions
        else None
    )
    text = extract_response_text(response)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text or (None if tool_calls else ""),
    }
    if legacy_function_call is not None:
        message["function_call"] = legacy_function_call
    elif tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = _chat_finish_reason(response, legacy_functions=legacy_functions)
    choices = [
        {
            "index": index,
            "message": copy.deepcopy(message),
            "finish_reason": finish_reason,
            "logprobs": None,
        }
        for index in range(max(1, n))
    ]

    return {
        "id": response_id.replace("resp_", "chatcmpl_", 1)
        if response_id.startswith("resp_")
        else f"chatcmpl_{response_id}",
        "object": "chat.completion",
        "created": created,
        "model": str(response.get("model") or fallback_model),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def ensure_response_defaults(
    response: dict[str, Any], *, request_payload: dict[str, Any]
) -> dict[str, Any]:
    prepared = copy.deepcopy(response)
    created_at = float(prepared.get("created_at") or time.time())
    response_id = str(prepared.get("id") or f"resp_{int(created_at * 1000)}")
    prepared["id"] = response_id
    prepared.setdefault("object", "response")
    prepared.setdefault("created_at", created_at)
    prepared.setdefault("status", "completed")
    prepared.setdefault("model", request_payload.get("model") or DEFAULT_MODEL)
    prepared.setdefault("output", [])
    prepared.setdefault("parallel_tool_calls", True)
    prepared.setdefault("tool_choice", request_payload.get("tool_choice") or "auto")
    prepared.setdefault("tools", request_payload.get("tools") or [])
    prepared.setdefault("previous_response_id", request_payload.get("previous_response_id"))
    prepared.setdefault("usage", None)
    return prepared


def extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    texts: list[str] = []
    for item in response.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            text = _output_message_text(item)
            if text:
                texts.append(text)
    return "".join(texts)


def _output_message_text(item: dict[str, Any]) -> str:
    texts: list[str] = []
    for part in item.get("content") or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def _response_function_calls_to_chat_tool_calls(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or ""),
                },
            }
        )
    return tool_calls


def _response_function_calls_to_chat_function_call(
    response: dict[str, Any],
) -> dict[str, str] | None:
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        return {
            "name": str(item.get("name") or ""),
            "arguments": str(item.get("arguments") or ""),
        }
    return None


def _collect_instructions(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    instructions = [
        _chat_content_to_text(message.get("content"))
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "developer"}
        and _chat_content_to_text(message.get("content"))
    ]
    return "\n\n".join(instructions) or None


def _chat_messages_to_response_input(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []

    response_input: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("name") or "unknown"
            response_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _chat_content_to_text(message.get("content")),
                }
            )
            continue
        if role == "function":
            response_input.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("name") or "function",
                    "output": _chat_content_to_text(message.get("content")),
                }
            )
            continue

        if role == "assistant":
            legacy_function_call = _chat_function_call_to_response_function_call(
                message.get("function_call")
            )
            if legacy_function_call:
                response_input.append(legacy_function_call)
            for function_call in _chat_tool_calls_to_response_function_calls(
                message.get("tool_calls")
            ):
                response_input.append(function_call)
            content = _chat_content_to_text(message.get("content"))
            if content or not (message.get("tool_calls") or legacy_function_call):
                response_input.append({"role": "assistant", "content": content})
            continue

        if role == "user":
            response_input.append(
                {
                    "role": "user",
                    "content": _chat_content_to_response_content(message.get("content")),
                }
            )
    return response_input


def _chat_tool_calls_to_response_function_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []

    function_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        call_id = tool_call.get("id")
        name = function.get("name")
        if not call_id or not name:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = "{}"
        function_calls.append(
            {
                "type": "function_call",
                "call_id": str(call_id),
                "name": str(name),
                "arguments": arguments,
            }
        )
    return function_calls


def _chat_function_call_to_response_function_call(
    function_call: Any,
) -> dict[str, Any] | None:
    if not isinstance(function_call, dict):
        return None
    name = function_call.get("name")
    if not name:
        return None
    arguments = function_call.get("arguments")
    if not isinstance(arguments, str):
        arguments = "{}"
    return {
        "type": "function_call",
        "call_id": str(name),
        "name": str(name),
        "arguments": arguments,
    }


def _chat_content_to_response_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            parts.append({"type": "input_text", "text": str(part.get("text") or "")})
        elif part_type in {"image_url", "input_image"}:
            parts.append(_chat_image_part_to_response_image(part))
    return parts


def _chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in {"text", "input_text"}:
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "output_text":
                texts.append(str(part.get("text") or ""))
    return "".join(texts)


def _chat_image_part_to_response_image(part: dict[str, Any]) -> dict[str, Any]:
    image_url = part.get("image_url")
    detail = part.get("detail")
    if isinstance(image_url, dict):
        detail = image_url.get("detail") or detail
        image_url = image_url.get("url")
    return {
        "type": "input_image",
        "image_url": str(image_url or ""),
        "detail": str(detail or "auto"),
    }


def _chat_tools_to_response_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    response_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            response_tools.append(copy.deepcopy(tool))
            continue
        function = tool.get("function") or {}
        response_tool = _chat_function_to_response_tool(function)
        if response_tool:
            response_tools.append(response_tool)
    return response_tools


def _chat_functions_to_response_tools(functions: Any) -> list[dict[str, Any]]:
    if not isinstance(functions, list):
        return []
    response_tools: list[dict[str, Any]] = []
    for function in functions:
        response_tool = _chat_function_to_response_tool(function)
        if response_tool:
            response_tools.append(response_tool)
    return response_tools


def _chat_function_to_response_tool(function: Any) -> dict[str, Any] | None:
    if not isinstance(function, dict) or not function.get("name"):
        return None
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description"),
        "parameters": function.get("parameters")
        or {"type": "object", "properties": {}},
        "strict": bool(function.get("strict", False)),
    }


def _chat_tool_choice_to_response_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") == "function":
        function = tool_choice.get("function") or {}
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
    return copy.deepcopy(tool_choice)


def _chat_function_call_to_response_tool_choice(function_call: Any) -> Any:
    if function_call is None:
        return None
    if isinstance(function_call, str):
        return function_call
    if not isinstance(function_call, dict):
        return None
    name = function_call.get("name")
    if name:
        return {"type": "function", "name": name}
    return None


def uses_legacy_chat_functions(payload: dict[str, Any]) -> bool:
    return bool(payload.get("functions")) and not bool(payload.get("tools"))


def _chat_response_format_to_text_config(response_format: Any) -> dict[str, Any]:
    if not isinstance(response_format, dict):
        return {}
    format_type = response_format.get("type")
    if format_type == "json_object":
        return {"format": {"type": "json_object"}}
    if format_type != "json_schema":
        return {}

    json_schema = response_format.get("json_schema") or {}
    if not isinstance(json_schema, dict):
        json_schema = {}
    format_payload: dict[str, Any] = {
        "type": "json_schema",
        "name": json_schema.get("name") or "response",
        "schema": json_schema.get("schema") or {},
    }
    if json_schema.get("strict") is not None:
        format_payload["strict"] = json_schema["strict"]
    return {"format": format_payload}


def _copy_if_present(
    source: dict[str, Any], target: dict[str, Any], fields: tuple[str, ...]
) -> None:
    for field in fields:
        if source.get(field) is not None:
            target[field] = copy.deepcopy(source[field])


def _copy_first_present(
    source: dict[str, Any],
    destination: dict[str, Any],
    *,
    target: str,
    sources: tuple[str, ...],
) -> None:
    for source_field in sources:
        if source.get(source_field) is not None:
            destination[target] = copy.deepcopy(source[source_field])
            return


def _chat_finish_reason(
    response: dict[str, Any], *, legacy_functions: bool = False
) -> str:
    if response.get("status") == "incomplete":
        return "length"
    for item in response.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            return "function_call" if legacy_functions else "tool_calls"
    return "stop"
