package app

import "testing"

func TestNormalizeCodexPayloadStripsPromptCacheBreakpoint(t *testing.T) {
	got := normalizeCodexPayload(map[string]any{
		"prompt_cache_key": "cache-key",
		"input": []any{
			map[string]any{"role": "user", "content": "plain"},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "static", "prompt_cache_breakpoint": map[string]any{"mode": "explicit"}},
				map[string]any{"type": "input_text", "text": "dynamic"},
			}},
		},
	}, nil)
	if got["prompt_cache_key"] != "cache-key" {
		t.Fatalf("prompt_cache_key = %#v", got["prompt_cache_key"])
	}
	input := sliceAny(got["input"])
	if mapAny(input[0])["content"] != "plain" {
		t.Fatalf("string content = %#v", input[0])
	}
	parts := sliceAny(mapAny(input[1])["content"])
	if _, ok := mapAny(parts[0])["prompt_cache_breakpoint"]; ok {
		t.Fatalf("content part = %#v", parts[0])
	}
	if mapAny(parts[0])["text"] != "static" || mapAny(parts[1])["text"] != "dynamic" {
		t.Fatalf("content parts = %#v", parts)
	}
}
