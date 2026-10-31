package app

import (
	"encoding/json"
	"testing"
)

func TestChatToResponsePairsRepeatedLegacyFunctionCalls(t *testing.T) {
	call := func(arguments string) map[string]any {
		return map[string]any{"role": "assistant", "content": nil, "function_call": map[string]any{"name": "lookup", "arguments": arguments}}
	}
	result := func(content string) map[string]any {
		return map[string]any{"role": "function", "name": "lookup", "content": content}
	}
	got := sliceAny(chatToResponse(map[string]any{"messages": []any{
		call(`{"city":"Paris"}`), result("sunny"),
		call(`{"city":"Tokyo"}`), result("rainy"),
		map[string]any{"role": "function", "name": "orphan", "content": "kept"},
	}}, defaultModel)["input"])
	if len(got) != 5 {
		t.Fatalf("input = %#v", got)
	}
	for index, want := range []struct{ kind, callID string }{
		{"function_call", "call_legacy_1"},
		{"function_call_output", "call_legacy_1"},
		{"function_call", "call_legacy_2"},
		{"function_call_output", "call_legacy_2"},
		{"function_call_output", "orphan"},
	} {
		item := mapAny(got[index])
		if item["type"] != want.kind || item["call_id"] != want.callID {
			t.Fatalf("input[%d] = %#v, want %+v", index, item, want)
		}
	}
}

func TestChatToResponseKeepsAssistantTextBeforeCalls(t *testing.T) {
	for name, message := range map[string]map[string]any{
		"tool_calls": {"role": "assistant", "content": "Checking", "tool_calls": []any{map[string]any{
			"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": "{}"},
		}}},
		"function_call": {"role": "assistant", "content": "Checking", "function_call": map[string]any{"name": "lookup", "arguments": "{}"}},
	} {
		t.Run(name, func(t *testing.T) {
			got := sliceAny(chatToResponse(map[string]any{"messages": []any{message}}, defaultModel)["input"])
			if len(got) != 2 || mapAny(got[0])["content"] != "Checking" || mapAny(got[1])["type"] != "function_call" {
				t.Fatalf("input = %#v", got)
			}
		})
	}
}

func TestResponseToChatKeepsTextAndLegacyFinishReason(t *testing.T) {
	call := map[string]any{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
	text := map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "Checking"}}}
	for name, test := range map[string]struct {
		output  []any
		legacy  bool
		content any
		finish  string
	}{
		"tool call with text":    {[]any{text, call}, false, "Checking", "tool_calls"},
		"tool call without text": {[]any{call}, false, nil, "tool_calls"},
		"legacy with text":       {[]any{text, call}, true, "Checking", "function_call"},
		"legacy without text":    {[]any{call}, true, nil, "function_call"},
		"text only":              {[]any{text}, false, "Checking", "stop"},
	} {
		t.Run(name, func(t *testing.T) {
			completion := responseToChat(map[string]any{"id": "resp_1", "output": test.output}, defaultModel, test.legacy, 1)
			choice := mapAny(sliceAny(completion["choices"])[0])
			if content := mapAny(choice["message"])["content"]; content != test.content {
				t.Fatalf("content = %#v, want %#v", content, test.content)
			}
			if choice["finish_reason"] != test.finish {
				t.Fatalf("finish_reason = %#v, want %q", choice["finish_reason"], test.finish)
			}
		})
	}
}

func TestChatToResponseTranslatesMessagesToolsAndStructuredOutput(t *testing.T) {
	body := map[string]any{
		"model": "gpt-test",
		"messages": []any{
			map[string]any{"role": "system", "content": "Be terse."},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "weather"}}},
		},
		"tools":           []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "parameters": map[string]any{"type": "object"}}}},
		"tool_choice":     map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}},
		"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "answer", "strict": true, "schema": map[string]any{"type": "object"}}},
	}
	got := chatToResponse(body, defaultModel)
	if got["instructions"] != "Be terse." {
		t.Fatalf("instructions = %#v", got["instructions"])
	}
	input := sliceAny(got["input"])
	if len(input) != 1 || mapAny(input[0])["role"] != "user" {
		t.Fatalf("input = %#v", input)
	}
	tools := sliceAny(got["tools"])
	if len(tools) != 1 || mapAny(tools[0])["name"] != "lookup" {
		t.Fatalf("tools = %#v", tools)
	}
	choice := mapAny(got["tool_choice"])
	if choice["name"] != "lookup" {
		t.Fatalf("tool choice = %#v", choice)
	}
	format := mapAny(mapAny(got["text"])["format"])
	if format["type"] != "json_schema" || format["name"] != "answer" {
		t.Fatalf("format = %#v", format)
	}
}

func TestResponseAndChatStoresAreBoundedAndIsolated(t *testing.T) {
	responses := newResponseStore(1)
	responses.remember("one", []any{map[string]any{"role": "user", "content": "one"}}, map[string]any{"id": "one", "output": []any{outputMessage("answer")}})
	responses.remember("two", []any{}, map[string]any{"id": "two", "output": []any{}})
	if responses.get("one") != nil {
		t.Fatal("oldest response was not evicted")
	}
	got := responses.get("two")
	got.Response["status"] = "mutated"
	if responses.get("two").Response["status"] == "mutated" {
		t.Fatal("store leaked mutable state")
	}

	chats := newChatStore(1)
	completion := map[string]any{"id": "chatcmpl_one", "model": "gpt-test", "choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "hi"}}}}
	chats.remember("chatcmpl_one", completion, map[string]any{"key": "value"})
	stored := chats.get("chatcmpl_one")
	if len(stored.Messages) != 1 || stored.Messages[0]["id"] != "chatcmpl_one_msg_0" {
		t.Fatalf("messages = %#v", stored.Messages)
	}
	if _, err := json.Marshal(stored.Completion); err != nil {
		t.Fatal(err)
	}
}

func TestResponseStoreRefreshesExistingEntryEvictionOrder(t *testing.T) {
	responses := newResponseStore(2)
	responses.remember("one", []any{}, map[string]any{"id": "one", "output": []any{}})
	responses.remember("two", []any{}, map[string]any{"id": "two", "output": []any{}})
	responses.remember("one", []any{}, map[string]any{"id": "one", "output": []any{}})
	responses.remember("three", []any{}, map[string]any{"id": "three", "output": []any{}})
	if responses.get("one") == nil || responses.get("two") != nil || responses.get("three") == nil {
		t.Fatalf("unexpected eviction order: %#v", responses.order)
	}
}

func TestPrepareResponseNormalizesReasoningAndDefaults(t *testing.T) {
	got := prepareResponse(map[string]any{"input": []any{
		map[string]any{"type": "reasoning", "summary": []any{}},
		map[string]any{"type": "reasoning", "encrypted_content": "cipher", "summary": []any{}},
	}}, "gpt-default")
	if got["model"] != "gpt-default" || got["store"] != false {
		t.Fatalf("defaults = %#v", got)
	}
	input := sliceAny(got["input"])
	if len(input) != 1 || mapAny(input[0])["encrypted_content"] != "cipher" {
		t.Fatalf("input = %#v", input)
	}
}

func TestAsyncResponseHistoryPreservesPendingCalls(t *testing.T) {
	items := []any{
		map[string]any{"type": "function_call", "call_id": "call_async", "name": "lookup", "arguments": "{}", "async": true, "future_field": "kept"},
		map[string]any{"type": "custom_tool_call", "call_id": "custom_async", "name": "custom", "input": "query", "async": true},
		map[string]any{"type": "reasoning", "encrypted_content": "opaque", "summary": []any{}},
		map[string]any{"type": "message", "role": "assistant", "phase": "commentary", "content": []any{map[string]any{"type": "output_text", "text": "independent"}}},
	}
	for _, history := range [][]any{normalizeResponseInput(items), responseContext(map[string]any{"output": items})} {
		if len(history) != 4 {
			t.Fatalf("lost history items: %#v", history)
		}
		if mapAny(history[0])["async"] != true || mapAny(history[0])["future_field"] != "kept" || mapAny(history[1])["async"] != true || mapAny(history[2])["encrypted_content"] != "opaque" || mapAny(history[3])["phase"] != "commentary" {
			t.Fatalf("lost history fields: %#v", history)
		}
	}
}
