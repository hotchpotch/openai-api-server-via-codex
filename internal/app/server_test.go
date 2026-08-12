package app

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
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

	exactV1 := httptest.NewRecorder()
	s.ServeHTTP(exactV1, httptest.NewRequest(http.MethodGet, "/v1", nil))
	if exactV1.Code != http.StatusUnauthorized {
		t.Fatalf("exact /v1 status = %d", exactV1.Code)
	}

	if !validBearer("bearer expected-secret", "expected-secret") {
		t.Fatal("bearer scheme should be case-insensitive")
	}
}

func TestStartupLogMessageIsStableForDynamicPorts(t *testing.T) {
	got := startupLogMessage("test-version", "127.0.0.1:43210")
	want := "openai-api-server-via-codex test-version (Go) listening on http://127.0.0.1:43210"
	if got != want {
		t.Fatalf("startup message = %q, want %q", got, want)
	}
}

func TestRouterRejectsNonV1AndNonGETHealthRequests(t *testing.T) {
	s := &server{cfg: config{}}
	for _, test := range []struct {
		method string
		path   string
		status int
	}{
		{http.MethodPost, "/healthz", http.StatusMethodNotAllowed},
		{http.MethodGet, "/outside", http.StatusNotFound},
	} {
		response := httptest.NewRecorder()
		s.ServeHTTP(response, httptest.NewRequest(test.method, test.path, nil))
		if response.Code != test.status {
			t.Errorf("%s %s status = %d, want %d", test.method, test.path, response.Code, test.status)
		}
	}
}

func TestUnhandledPanicReturnsRedactedOpenAIError(t *testing.T) {
	s := &server{cfg: config{}}
	response := httptest.NewRecorder()
	s.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if response.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "Internal server error.") {
		t.Fatalf("body = %s", response.Body.String())
	}
}

func TestVerboseRequestLogRedactsQuerySecrets(t *testing.T) {
	var output bytes.Buffer
	previousOutput := log.Writer()
	previousFlags := log.Flags()
	log.SetOutput(&output)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(previousOutput)
		log.SetFlags(previousFlags)
	})

	s := &server{cfg: config{Verbose: true}}
	response := httptest.NewRecorder()
	s.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/outside?api_key=do-not-log", nil))
	if strings.Contains(output.String(), "do-not-log") {
		t.Fatalf("secret leaked in log: %s", output.String())
	}
	if !strings.Contains(output.String(), "[REDACTED]") {
		t.Fatalf("redaction marker missing: %s", output.String())
	}
}

func TestDecodeObjectRejectsNullAndTrailingData(t *testing.T) {
	for _, body := range []string{"null", `{"ok":true} {"extra":true}`} {
		request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
		if _, err := decodeObject(request); err == nil {
			t.Fatalf("decodeObject(%q) succeeded", body)
		}
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

func TestResolveProxyURLPreservesEncodedPathDelimiters(t *testing.T) {
	target, err := resolveProxyURL(
		"https://example.test/backend-api/codex",
		"files/report?format#section",
		"limit=1",
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := target.String(); got != "https://example.test/backend-api/codex/files/report%3Fformat%23section?limit=1" {
		t.Fatalf("target = %q", got)
	}
}

func TestResolveProxyURLPreservesLiteralPercentInDecodedPath(t *testing.T) {
	target, err := resolveProxyURL("https://example.test/backend-api/codex", "files/100%done", "")
	if err != nil {
		t.Fatal(err)
	}
	if got := target.String(); got != "https://example.test/backend-api/codex/files/100%25done" {
		t.Fatalf("target = %q", got)
	}
}

func TestResolveProxyURLRejectsAmbiguousOrEscapingPaths(t *testing.T) {
	for _, path := range []string{
		"../auth",
		"files/../auth",
		`files\..\auth`,
		"files/\x00auth",
	} {
		if _, err := resolveProxyURL("https://example.test/backend-api/codex", path, ""); err == nil {
			t.Errorf("resolveProxyURL accepted %q", path)
		}
	}
}

func TestResponseInputPageItemsMatchCompatibilityShape(t *testing.T) {
	tests := []struct {
		name string
		raw  any
		want map[string]any
	}{
		{
			name: "user message",
			raw:  map[string]any{"role": "user", "content": "hello"},
			want: map[string]any{"id": "input_0", "type": "message", "role": "user", "status": "completed", "content": []any{map[string]any{"type": "input_text", "text": "hello"}}},
		},
		{
			name: "assistant message",
			raw:  map[string]any{"role": "assistant", "content": "answer", "phase": "final_answer"},
			want: map[string]any{"id": "input_0", "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer", "content": []any{map[string]any{"type": "output_text", "text": "answer", "annotations": []any{}}}},
		},
		{
			name: "function output",
			raw:  map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
			want: map[string]any{"id": "call_1", "type": "function_call_output", "call_id": "call_1", "output": "sunny", "status": "completed"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := responseInputPageItem(test.raw, 0); !mapsEqual(got, test.want) {
				t.Fatalf("item = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestPaginateMapsComputesHasMoreAfterCursor(t *testing.T) {
	items := []map[string]any{{"id": "one"}, {"id": "two"}, {"id": "three"}}
	page, more := paginateMaps(items, url.Values{"after": {"two"}, "limit": {"2"}})
	if more || len(page) != 1 || page[0]["id"] != "three" {
		t.Fatalf("page = %#v, has_more = %t", page, more)
	}
}

func TestReadSSELineSupportsLinesLargerThanReaderBuffer(t *testing.T) {
	line := strings.Repeat("x", 1024)
	reader := bufio.NewReaderSize(strings.NewReader(line+"\n"), 16)
	got, err := readSSELine(reader, 2048)
	if err != nil || got != line {
		t.Fatalf("line length = %d, err = %v", len(got), err)
	}
	reader = bufio.NewReaderSize(strings.NewReader(line+"\n"), 16)
	if _, err := readSSELine(reader, 100); err == nil {
		t.Fatal("oversized line was accepted")
	}
}

func TestBackendHeadersForwardPromptCacheKey(t *testing.T) {
	b := newBackend(defaultConfig())
	if b.auth.refreshClient.Timeout != authRefreshTimeout {
		t.Fatalf("refresh timeout = %s", b.auth.refreshClient.Timeout)
	}
	headers := b.headers(credentials{AccessToken: "token"}, true, "request-123")
	if headers.Get("session_id") != "request-123" || headers.Get("x-client-request-id") != "request-123" {
		t.Fatalf("headers = %#v", headers)
	}
}

func TestNormalizeBackendEventDropsUnknownStatus(t *testing.T) {
	event := map[string]any{"type": "response.done", "response": map[string]any{"id": "resp_1", "status": "mystery"}}
	normalizeBackendEvent(event)
	if event["type"] != "response.completed" || mapAny(event["response"])["status"] != nil {
		t.Fatalf("event = %#v", event)
	}
}

func TestPublicStreamErrorMasksInternalDetails(t *testing.T) {
	if got := publicStreamError(errors.New("secret internal detail")); got != "Internal server error." {
		t.Fatalf("message = %q", got)
	}
	if got := publicStreamError(&backendError{Status: 502, Message: "redacted upstream"}); got != "redacted upstream" {
		t.Fatalf("backend message = %q", got)
	}
}

func mapsEqual(left, right map[string]any) bool {
	leftJSON, _ := json.Marshal(left)
	rightJSON, _ := json.Marshal(right)
	return bytes.Equal(leftJSON, rightJSON)
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
