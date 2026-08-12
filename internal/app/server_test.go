package app

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAPIKeyProtectsV1ButNotHealth(t *testing.T) {
	s := &server{cfg: config{APIKey: "expected-secret"}}

	health := httptest.NewRecorder()
	s.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d", health.Code)
	}

	unauthorized := httptest.NewRecorder()
	s.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d", unauthorized.Code)
	}
}

func TestInvalidProxyTraversalIsRejectedBeforeBackend(t *testing.T) {
	s := &server{cfg: config{}}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/files/%2e%2e/secrets", nil)
	s.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestImageGenerationUsesCompatibilityModelAndCodexPayload(t *testing.T) {
	body := map[string]any{
		"model": "gpt-image-2", "prompt": "  draw ramen  ", "size": "1024x1024",
		"quality": "medium", "output_format": "png", "output_compression": 80,
	}
	if validation := validateImage(body); validation != nil {
		t.Fatalf("validation = %#v", validation)
	}
	payload := imageResponsePayload(body, "gpt-5.6-luna")
	if payload["model"] != "gpt-5.6-luna" || payload["tool_choice"] != "auto" {
		t.Fatalf("payload = %#v", payload)
	}
	input := mapAny(sliceAny(payload["input"])[0])
	content := mapAny(sliceAny(input["content"])[0])
	if content["text"] != "draw ramen" {
		t.Fatalf("content = %#v", content)
	}
	tool := mapAny(sliceAny(payload["tools"])[0])
	if tool["type"] != "image_generation" || intValue(tool["output_compression"]) != 80 {
		t.Fatalf("tool = %#v", tool)
	}
}

func TestImageGenerationValidationMatchesPublicContract(t *testing.T) {
	tests := []struct {
		name, param string
		body        map[string]any
	}{
		{"prompt", "prompt", map[string]any{}},
		{"stream", "stream", map[string]any{"prompt": "x", "stream": true}},
		{"response format", "response_format", map[string]any{"prompt": "x", "response_format": "b64_json"}},
		{"fractional n", "n", map[string]any{"prompt": "x", "n": 1.5}},
		{"quality", "quality", map[string]any{"prompt": "x", "quality": "ultra"}},
		{"compression", "output_compression", map[string]any{"prompt": "x", "output_compression": 101}},
		{"style", "style", map[string]any{"prompt": "x", "style": "vivid"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateImage(test.body)
			if err == nil || err.Param != test.param {
				t.Fatalf("validation = %#v", err)
			}
		})
	}
}

func TestCompatibilityRoutesDoNotCaptureUnknownProxyEndpoints(t *testing.T) {
	if isResponseResourceRoute(http.MethodPost, "compact") {
		t.Fatal("responses/compact must use fallback proxy")
	}
	if isResponseResourceRoute(http.MethodPost, "resp_1") {
		t.Fatal("POST response resource must use fallback proxy")
	}
	if !isResponseResourceRoute(http.MethodPost, "resp_1/cancel") {
		t.Fatal("response cancel route not recognized")
	}
	if !isResponseResourceRoute(http.MethodGet, "resp_1/input_items") {
		t.Fatal("response input_items route not recognized")
	}
	if isChatResourceRoute(http.MethodPatch, "chatcmpl_1") {
		t.Fatal("PATCH chat resource must use fallback proxy")
	}
	if !isChatResourceRoute(http.MethodGet, "chatcmpl_1/messages") {
		t.Fatal("chat messages route not recognized")
	}
}
