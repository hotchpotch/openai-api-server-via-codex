package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/hotchpotch/openai-api-server-via-codex/internal/testutil/codexfake"
	"github.com/hotchpotch/openai-api-server-via-codex/internal/testutil/testauth"
)

const contractAPIKey = "go-contract-api-key"

type contractEnvironment struct {
	server   *httptest.Server
	upstream *codexfake.Server
	client   *http.Client
}

func newContractEnvironment(t *testing.T, mutate func(*config)) *contractEnvironment {
	t.Helper()
	upstream := codexfake.New(t)
	cfg := defaultConfig()
	cfg.BackendURL = upstream.BackendURL()
	cfg.AuthJSON = testauth.Write(t)
	cfg.APIKey = contractAPIKey
	cfg.Timeout = 5 * time.Second
	if mutate != nil {
		mutate(&cfg)
	}
	b := newBackend(cfg)
	s := &server{
		cfg: cfg, backend: b,
		responses: newResponseStore(cfg.MaxStored),
		chats:     newChatStore(cfg.MaxStored),
	}
	if cfg.Concurrency > 0 {
		s.slots = make(chan struct{}, cfg.Concurrency)
	}
	httpServer := httptest.NewServer(s)
	t.Cleanup(httpServer.Close)
	return &contractEnvironment{server: httpServer, upstream: upstream, client: httpServer.Client()}
}

func (environment *contractEnvironment) request(
	t *testing.T,
	method, path string,
	payload any,
) (*http.Response, []byte) {
	t.Helper()
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		body = bytes.NewReader(encoded)
	}
	request, err := http.NewRequest(method, environment.server.URL+path, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+contractAPIKey)
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := environment.client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	return response, data
}

func (environment *contractEnvironment) json(
	t *testing.T,
	method, path string,
	payload any,
	wantStatus int,
) map[string]any {
	t.Helper()
	response, data := environment.request(t, method, path, payload)
	if response.StatusCode != wantStatus {
		t.Fatalf("%s %s status = %d, want %d; body=%s", method, path, response.StatusCode, wantStatus, data)
	}
	if len(data) == 0 {
		return nil
	}
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatalf("%s %s returned invalid JSON: %v; body=%s", method, path, err, data)
	}
	return document
}

func parseSSE(t *testing.T, data []byte) []map[string]any {
	t.Helper()
	var events []map[string]any
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if raw == "" || raw == "[DONE]" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			t.Fatalf("invalid SSE event: %v: %s", err, raw)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return events
}

func TestGoHTTPContractResponsesLifecycleAndStreaming(t *testing.T) {
	environment := newContractEnvironment(t, nil)

	healthRequest, _ := http.NewRequest(http.MethodGet, environment.server.URL+"/healthz", nil)
	health, err := environment.client.Do(healthRequest)
	if err != nil {
		t.Fatal(err)
	}
	health.Body.Close()
	if health.StatusCode != http.StatusOK {
		t.Fatalf("health status = %d", health.StatusCode)
	}
	unauthorizedRequest, _ := http.NewRequest(http.MethodGet, environment.server.URL+"/v1/models", nil)
	unauthorized, err := environment.client.Do(unauthorizedRequest)
	if err != nil {
		t.Fatal(err)
	}
	unauthorized.Body.Close()
	if unauthorized.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d", unauthorized.StatusCode)
	}

	models := environment.json(t, http.MethodGet, "/v1/models", nil, http.StatusOK)
	modelData := sliceAny(models["data"])
	if len(modelData) != 1 || mapAny(modelData[0])["id"] != "gpt-5.6-luna" {
		t.Fatalf("models = %#v", models)
	}

	created := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna",
		"input": []any{
			map[string]any{"role": "user", "content": "GO-RESP-FIRST"},
			map[string]any{"role": "user", "content": "GO-RESP-SECOND"},
		},
		"max_output_tokens": 32,
		"prompt_cache_key":  "go-contract-request",
		"store":             true,
	}, http.StatusOK)
	createdID := stringValue(created["id"])
	if createdID == "" || created["status"] != "completed" {
		t.Fatalf("created response = %#v", created)
	}
	if text := responseText(created); !strings.Contains(text, "GO-RESP-FIRST") || !strings.Contains(text, "GO-RESP-SECOND") {
		t.Fatalf("created output = %q", text)
	}

	upstream := environment.upstream.LastResponseRequest(t)
	if upstream.JSON["stream"] != true || upstream.JSON["store"] != false || upstream.JSON["max_output_tokens"] != nil {
		t.Fatalf("normalized upstream payload = %#v", upstream.JSON)
	}
	textConfig := mapAny(upstream.JSON["text"])
	if textConfig["verbosity"] != "low" {
		t.Fatalf("upstream text config = %#v", textConfig)
	}
	if !containsString(sliceAny(upstream.JSON["include"]), "reasoning.encrypted_content") {
		t.Fatalf("upstream include = %#v", upstream.JSON["include"])
	}
	if upstream.Headers.Get("session_id") != "go-contract-request" || upstream.Headers.Get("x-client-request-id") != "go-contract-request" {
		t.Fatalf("upstream headers = %#v", upstream.Headers)
	}
	if upstream.Headers.Get("ChatGPT-Account-ID") != "acct_go_test" || upstream.Headers.Get("Authorization") == "Bearer "+contractAPIKey {
		t.Fatalf("upstream auth headers = %#v", upstream.Headers)
	}

	retrieved := environment.json(t, http.MethodGet, "/v1/responses/"+createdID, nil, http.StatusOK)
	if responseText(retrieved) != responseText(created) {
		t.Fatalf("retrieved response = %#v", retrieved)
	}
	firstPage := environment.json(t, http.MethodGet, "/v1/responses/"+createdID+"/input_items?limit=1", nil, http.StatusOK)
	if firstPage["has_more"] != true || len(sliceAny(firstPage["data"])) != 1 {
		t.Fatalf("first input page = %#v", firstPage)
	}
	nextPage := environment.json(t, http.MethodGet, "/v1/responses/"+createdID+"/input_items?after=input_0&limit=2", nil, http.StatusOK)
	if nextPage["has_more"] != false || len(sliceAny(nextPage["data"])) != 1 {
		t.Fatalf("next input page = %#v", nextPage)
	}
	tokens := environment.json(t, http.MethodPost, "/v1/responses/input_tokens", map[string]any{
		"model": "gpt-5.6-luna", "input": "count Go contract tokens",
	}, http.StatusOK)
	if intValue(tokens["input_tokens"]) < 1 {
		t.Fatalf("token count = %#v", tokens)
	}

	continued := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "previous_response_id": createdID,
		"input": "GO-RESP-CONTINUED",
	}, http.StatusOK)
	continuedText := responseText(continued)
	for _, marker := range []string{"GO-RESP-FIRST", "GO-RESP-SECOND", "GO-RESP-CONTINUED"} {
		if !strings.Contains(continuedText, marker) {
			t.Fatalf("continued output %q misses %q", continuedText, marker)
		}
	}
	if continued["previous_response_id"] != createdID {
		t.Fatalf("continued response = %#v", continued)
	}

	streamResponse, streamData := environment.request(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "GO-RESP-STREAM", "stream": true,
	})
	if streamResponse.StatusCode != http.StatusOK || !strings.HasPrefix(streamResponse.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream response = %d %#v", streamResponse.StatusCode, streamResponse.Header)
	}
	events := parseSSE(t, streamData)
	if len(events) == 0 || events[len(events)-1]["type"] != "response.completed" {
		t.Fatalf("stream events = %#v", events)
	}
	var streamedText string
	var streamedID string
	for _, event := range events {
		if event["type"] == "response.output_text.delta" {
			streamedText += stringValue(event["delta"])
		}
		if event["type"] == "response.completed" {
			streamedID = stringValue(mapAny(event["response"])["id"])
		}
	}
	if !strings.Contains(streamedText, "GO-RESP-STREAM") || streamedID == "" {
		t.Fatalf("streamed text=%q id=%q", streamedText, streamedID)
	}
	replayResponse, replayData := environment.request(t, http.MethodGet, "/v1/responses/"+streamedID+"?stream=true", nil)
	if replayResponse.StatusCode != http.StatusOK {
		t.Fatalf("replay status = %d", replayResponse.StatusCode)
	}
	replay := parseSSE(t, replayData)
	if replay[len(replay)-1]["type"] != "response.completed" {
		t.Fatalf("replay = %#v", replay)
	}

	environment.json(t, http.MethodPost, "/v1/responses/"+createdID+"/cancel", map[string]any{}, http.StatusConflict)
	response, _ := environment.request(t, http.MethodDelete, "/v1/responses/"+createdID, nil)
	if response.StatusCode != http.StatusNoContent {
		t.Fatalf("delete status = %d", response.StatusCode)
	}
	environment.json(t, http.MethodGet, "/v1/responses/"+createdID, nil, http.StatusNotFound)
}

func TestGoHTTPContractChatLifecycleToolsAndStreaming(t *testing.T) {
	environment := newContractEnvironment(t, nil)
	created := environment.json(t, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "gpt-5.6-luna", "n": 2, "store": true,
		"metadata": map[string]any{"suite": "go-contract"},
		"messages": []any{map[string]any{"role": "user", "content": "GO-CHAT-STORED"}},
	}, http.StatusOK)
	chatID := stringValue(created["id"])
	choices := sliceAny(created["choices"])
	if chatID == "" || len(choices) != 2 {
		t.Fatalf("chat = %#v", created)
	}
	for _, raw := range choices {
		content := stringValue(mapAny(mapAny(raw)["message"])["content"])
		if !strings.Contains(content, "GO-CHAT-STORED") {
			t.Fatalf("choice = %#v", raw)
		}
	}
	retrieved := environment.json(t, http.MethodGet, "/v1/chat/completions/"+chatID, nil, http.StatusOK)
	if retrieved["id"] != chatID {
		t.Fatalf("retrieved chat = %#v", retrieved)
	}
	listed := environment.json(t, http.MethodGet, "/v1/chat/completions?metadata%5Bsuite%5D=go-contract", nil, http.StatusOK)
	if len(sliceAny(listed["data"])) != 1 {
		t.Fatalf("listed chats = %#v", listed)
	}
	updated := environment.json(t, http.MethodPost, "/v1/chat/completions/"+chatID, map[string]any{
		"metadata": map[string]any{"suite": "updated"},
	}, http.StatusOK)
	if mapAny(updated["metadata"])["suite"] != "updated" {
		t.Fatalf("updated chat = %#v", updated)
	}
	messages := environment.json(t, http.MethodGet, "/v1/chat/completions/"+chatID+"/messages?order=desc", nil, http.StatusOK)
	messageData := sliceAny(messages["data"])
	if len(messageData) != 2 || !strings.HasSuffix(stringValue(mapAny(messageData[0])["id"]), "_msg_1") {
		t.Fatalf("chat messages = %#v", messages)
	}
	remaining := environment.json(t, http.MethodGet, "/v1/chat/completions/"+chatID+"/messages?after="+url.QueryEscape(chatID+"_msg_0")+"&limit=2", nil, http.StatusOK)
	if remaining["has_more"] != false || len(sliceAny(remaining["data"])) != 1 {
		t.Fatalf("remaining messages = %#v", remaining)
	}

	tool := environment.json(t, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "gpt-5.6-luna",
		"messages": []any{map[string]any{"role": "user", "content": "Use the weather tool"}},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "lookup_weather", "description": "Lookup weather",
			"parameters": map[string]any{"type": "object"},
		}}},
	}, http.StatusOK)
	toolMessage := mapAny(mapAny(sliceAny(tool["choices"])[0])["message"])
	toolCalls := sliceAny(toolMessage["tool_calls"])
	if len(toolCalls) != 1 || mapAny(mapAny(toolCalls[0])["function"])["name"] != "lookup_weather" {
		t.Fatalf("tool chat = %#v", tool)
	}

	legacy := environment.json(t, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":     "gpt-5.6-luna",
		"messages":  []any{map[string]any{"role": "user", "content": "Use legacy function"}},
		"functions": []any{map[string]any{"name": "lookup_weather", "parameters": map[string]any{"type": "object"}}},
	}, http.StatusOK)
	legacyMessage := mapAny(mapAny(sliceAny(legacy["choices"])[0])["message"])
	if mapAny(legacyMessage["function_call"])["name"] != "lookup_weather" {
		t.Fatalf("legacy chat = %#v", legacy)
	}

	environment.json(t, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "gpt-5.6-luna",
		"messages": []any{
			map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": "call_previous", "type": "function", "function": map[string]any{"name": "lookup_weather", "arguments": `{"city":"Tokyo"}`},
			}}},
			map[string]any{"role": "tool", "tool_call_id": "call_previous", "content": "sunny"},
		},
		"response_format": map[string]any{"type": "json_schema", "json_schema": map[string]any{
			"name": "weather", "strict": true, "schema": map[string]any{"type": "object"},
		}},
	}, http.StatusOK)
	translated := environment.upstream.LastResponseRequest(t).JSON
	translatedInput := sliceAny(translated["input"])
	if len(translatedInput) != 2 || mapAny(translatedInput[0])["type"] != "function_call" || mapAny(translatedInput[1])["type"] != "function_call_output" {
		t.Fatalf("translated tool context = %#v", translatedInput)
	}
	format := mapAny(mapAny(translated["text"])["format"])
	if format["type"] != "json_schema" || format["name"] != "weather" {
		t.Fatalf("translated response format = %#v", format)
	}

	streamResponse, streamData := environment.request(t, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "gpt-5.6-luna", "n": 2, "stream": true,
		"stream_options": map[string]any{"include_usage": true},
		"messages":       []any{map[string]any{"role": "user", "content": "GO-CHAT-STREAM"}},
	})
	if streamResponse.StatusCode != http.StatusOK {
		t.Fatalf("chat stream status = %d", streamResponse.StatusCode)
	}
	chunks := parseSSE(t, streamData)
	contentByChoice := map[int]string{}
	usageChunk := false
	for _, chunk := range chunks {
		if chunk["usage"] != nil {
			usageChunk = true
		}
		for _, rawChoice := range sliceAny(chunk["choices"]) {
			choice := mapAny(rawChoice)
			contentByChoice[intValue(choice["index"])] += stringValue(mapAny(choice["delta"])["content"])
		}
	}
	if !usageChunk || !strings.Contains(contentByChoice[0], "GO-CHAT-STREAM") || !strings.Contains(contentByChoice[1], "GO-CHAT-STREAM") {
		t.Fatalf("chat stream chunks = %#v", chunks)
	}

	deleted := environment.json(t, http.MethodDelete, "/v1/chat/completions/"+chatID, nil, http.StatusOK)
	if deleted["deleted"] != true {
		t.Fatalf("deleted chat = %#v", deleted)
	}
	environment.json(t, http.MethodGet, "/v1/chat/completions/"+chatID, nil, http.StatusNotFound)
}

func TestGoHTTPContractImagesAudioProxyAndErrors(t *testing.T) {
	environment := newContractEnvironment(t, nil)
	images := environment.json(t, http.MethodPost, "/v1/images/generations", map[string]any{
		"model": "gpt-image-2", "prompt": "GO-IMAGE", "n": 2,
		"size": "1024x1024", "quality": "medium", "output_format": "png",
	}, http.StatusOK)
	imageData := sliceAny(images["data"])
	if len(imageData) != 2 {
		t.Fatalf("images = %#v", images)
	}
	decoded, err := base64.StdEncoding.DecodeString(stringValue(mapAny(imageData[0])["b64_json"]))
	if err != nil || !bytes.HasPrefix(decoded, []byte("\x89PNG")) {
		t.Fatalf("image data = %q, error=%v", decoded, err)
	}

	audioRequest, _ := http.NewRequest(http.MethodPost, environment.server.URL+"/v1/audio/transcriptions", strings.NewReader("RIFF-go-contract"))
	audioRequest.Header.Set("Authorization", "Bearer "+contractAPIKey)
	audioRequest.Header.Set("Content-Type", "multipart/form-data; boundary=go-contract")
	audioResponse, err := environment.client.Do(audioRequest)
	if err != nil {
		t.Fatal(err)
	}
	audioBody, _ := io.ReadAll(audioResponse.Body)
	audioResponse.Body.Close()
	if audioResponse.StatusCode != http.StatusOK || !strings.Contains(string(audioBody), "transcribed by Go fake Codex") {
		t.Fatalf("audio response = %d %s", audioResponse.StatusCode, audioBody)
	}

	proxyRequest, _ := http.NewRequest(http.MethodPost, environment.server.URL+"/v1/batches?limit=3", strings.NewReader(`{"hello":"world"}`))
	proxyRequest.Header.Set("Authorization", "Bearer "+contractAPIKey)
	proxyRequest.Header.Set("Content-Type", "application/json")
	proxyRequest.Header.Set("Idempotency-Key", "go-idempotency")
	proxyRequest.Header.Set("X-Unsafe-Header", "must-not-forward")
	proxyResponse, err := environment.client.Do(proxyRequest)
	if err != nil {
		t.Fatal(err)
	}
	proxyBody, _ := io.ReadAll(proxyResponse.Body)
	proxyResponse.Body.Close()
	if proxyResponse.StatusCode != http.StatusOK || proxyResponse.Header.Get("X-OpenAI-Via-Codex-Proxy") != "codex-http" || proxyResponse.Header.Get("Set-Cookie") != "" {
		t.Fatalf("proxy response = %d %#v %s", proxyResponse.StatusCode, proxyResponse.Header, proxyBody)
	}
	var proxyDocument map[string]any
	_ = json.Unmarshal(proxyBody, &proxyDocument)
	if !strings.HasSuffix(stringValue(proxyDocument["request_path"]), "/batches?limit=3") {
		t.Fatalf("proxy body = %#v", proxyDocument)
	}
	requests := environment.upstream.Requests()
	last := requests[len(requests)-1]
	if last.Headers.Get("Idempotency-Key") != "go-idempotency" || last.Headers.Get("X-Unsafe-Header") != "" {
		t.Fatalf("proxy request headers = %#v", last.Headers)
	}

	traversal, _ := environment.request(t, http.MethodGet, "/v1/files/%2e%2e/auth", nil)
	if traversal.StatusCode != http.StatusBadRequest {
		t.Fatalf("traversal status = %d", traversal.StatusCode)
	}
	errorResponse := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "FAKE_HTTP_ERROR",
	}, http.StatusBadGateway)
	errorMessage := stringValue(mapAny(errorResponse["error"])["message"])
	if strings.Contains(errorMessage, "abcdefghijklmnopqrstuvwxyz") || !strings.Contains(errorMessage, "[REDACTED]") {
		t.Fatalf("unredacted backend error = %q", errorMessage)
	}
}

func TestGoHTTPContractCollectFallbackFailureAndNormalization(t *testing.T) {
	environment := newContractEnvironment(t, nil)
	fallback := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "FAKE_DELTA_ONLY",
	}, http.StatusOK)
	if fallback["status"] != "completed" || !strings.Contains(responseText(fallback), "FAKE_DELTA_ONLY") {
		t.Fatalf("fallback response = %#v", fallback)
	}
	normalized := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "FAKE_RAW_DONE",
	}, http.StatusOK)
	if normalized["status"] != "completed" {
		t.Fatalf("normalized response = %#v", normalized)
	}
	failed := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "FAKE_RESPONSE_FAILED",
	}, http.StatusOK)
	if failed["status"] != "failed" {
		t.Fatalf("failed response = %#v", failed)
	}
}

func TestGoHTTPContractConcurrencySlotCoversStreamingLifetime(t *testing.T) {
	environment := newContractEnvironment(t, func(cfg *config) { cfg.Concurrency = 1 })
	gate := make(chan struct{})
	started := make(chan struct{}, 2)
	environment.upstream.GateResponses(gate, started)

	type result struct {
		response *http.Response
		error    error
	}
	startStream := func(marker string) <-chan result {
		resultChannel := make(chan result, 1)
		go func() {
			payload, _ := json.Marshal(map[string]any{
				"model": "gpt-5.6-luna", "input": marker, "stream": true,
			})
			request, _ := http.NewRequest(http.MethodPost, environment.server.URL+"/v1/responses", bytes.NewReader(payload))
			request.Header.Set("Authorization", "Bearer "+contractAPIKey)
			request.Header.Set("Content-Type", "application/json")
			response, err := environment.client.Do(request)
			resultChannel <- result{response: response, error: err}
		}()
		return resultChannel
	}

	firstResult := startStream("GO-CONCURRENCY-FIRST")
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("first upstream stream did not start")
	}
	secondResult := startStream("GO-CONCURRENCY-SECOND")
	select {
	case <-started:
		t.Fatal("second upstream stream started before the first released its slot")
	case <-time.After(150 * time.Millisecond):
	}
	close(gate)
	for _, resultChannel := range []<-chan result{firstResult, secondResult} {
		result := <-resultChannel
		if result.error != nil {
			t.Fatal(result.error)
		}
		_, _ = io.Copy(io.Discard, result.response.Body)
		result.response.Body.Close()
	}
	if maximum := environment.upstream.MaximumActiveResponses(); maximum != 1 {
		t.Fatalf("maximum active upstream responses = %d", maximum)
	}
}

type faultBackend struct {
	err error
}

func (f faultBackend) stream(context.Context, map[string]any, func(map[string]any) error) error {
	return f.err
}
func (f faultBackend) collect(context.Context, map[string]any) (map[string]any, error) {
	return nil, f.err
}
func (f faultBackend) listModels(context.Context) []string { return nil }
func (f faultBackend) proxy(context.Context, string, string, string, http.Header, io.Reader) (*http.Response, error) {
	return nil, f.err
}
func (f faultBackend) transcribe(context.Context, http.Header, io.Reader) (*http.Response, error) {
	return nil, f.err
}

func TestBackendInterfaceMasksInjectedInternalFault(t *testing.T) {
	s := &server{
		cfg: config{}, backend: faultBackend{err: fmt.Errorf("internal secret detail")},
		responses: newResponseStore(1), chats: newChatStore(1),
	}
	response := httptest.NewRecorder()
	s.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hello"}`)))
	if response.Code != http.StatusInternalServerError || strings.Contains(response.Body.String(), "internal secret detail") {
		t.Fatalf("fault response = %d %s", response.Code, response.Body.String())
	}
}

func containsString(values []any, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
