package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
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
	cfg.MaxRetries = 0
	cfg.retryBaseDelay = time.Millisecond
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
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "GO-RESP-SECOND", "prompt_cache_breakpoint": map[string]any{"mode": "explicit"}},
			}},
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
	upstreamParts := sliceAny(mapAny(sliceAny(upstream.JSON["input"])[1])["content"])
	if _, ok := mapAny(upstreamParts[0])["prompt_cache_breakpoint"]; ok || mapAny(upstreamParts[0])["text"] != "GO-RESP-SECOND" {
		t.Fatalf("upstream content parts = %#v", upstreamParts)
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

// newScriptedEnvironment answers every Codex request with the given events.
func newScriptedEnvironment(t *testing.T, events []map[string]any, forwarded *map[string]any) *contractEnvironment {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(forwarded); err != nil {
			t.Errorf("decode forwarded request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			encoded, _ := json.Marshal(event)
			fmt.Fprintf(w, "data: %s\n\n", encoded)
		}
	}))
	t.Cleanup(upstream.Close)
	return newContractEnvironment(t, func(cfg *config) { cfg.BackendURL = upstream.URL })
}

func TestGoHTTPContractChatReportsFailedResponses(t *testing.T) {
	const generic = "Codex backend response failed."
	secret := "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
	created := map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_failed", "status": "in_progress"}}
	text := map[string]any{"type": "response.output_text.delta", "output_index": 0, "delta": "Partial"}
	failed := func(detail any) map[string]any {
		return map[string]any{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "failed", "output": []any{}, "error": detail}}
	}
	for name, test := range map[string]struct {
		event   map[string]any
		status  int
		code    any
		message string
	}{
		"server error":       {failed(map[string]any{"code": "server_error", "message": "upstream broke"}), 502, "server_error", "upstream broke"},
		"rate limit":         {failed(map[string]any{"code": "rate_limit_exceeded", "message": "slow down"}), 429, "rate_limit_exceeded", "slow down"},
		"context length":     {failed(map[string]any{"code": "context_length_exceeded", "message": "too long"}), 400, "context_length_exceeded", "too long"},
		"invalid prompt":     {failed(map[string]any{"code": "invalid_prompt", "message": "rejected"}), 400, "invalid_prompt", "rejected"},
		"no detail":          {failed(nil), 502, nil, generic},
		"status on done":     {map[string]any{"type": "response.done", "response": map[string]any{"id": "resp_failed", "status": "failed", "error": map[string]any{"code": "server_error", "message": "done failed"}}}, 502, "server_error", "done failed"},
		"flat error event":   {map[string]any{"type": "error", "code": "server_error", "message": "flat"}, 502, "server_error", "flat"},
		"nested error":       {map[string]any{"type": "error", "error": map[string]any{"code": "rate_limit_exceeded", "message": "nested"}}, 429, "rate_limit_exceeded", "nested"},
		"bare error event":   {map[string]any{"type": "error"}, 502, nil, generic},
		"missing status":     {map[string]any{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "error": map[string]any{"code": "server_error", "message": "no status"}}}, 502, "server_error", "no status"},
		"unknown status":     {map[string]any{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "broken", "error": map[string]any{"code": "server_error", "message": "odd status"}}}, 502, "server_error", "odd status"},
		"string error":       {failed("plain text"), 502, nil, "plain text"},
		"overloaded":         {failed(map[string]any{"code": "server_is_overloaded", "message": "busy"}), 503, "server_is_overloaded", "busy"},
		"invalid api key":    {failed(map[string]any{"code": "invalid_api_key", "message": "denied"}), 401, "invalid_api_key", "denied"},
		"event status":       {map[string]any{"type": "error", "status": 400, "error": map[string]any{"code": "unsupported_value", "message": "bad value"}}, 400, "unsupported_value", "bad value"},
		"string error event": {map[string]any{"type": "error", "error": "boom"}, 502, nil, "boom"},
		"redacted message":   {failed(map[string]any{"code": "server_error", "message": "denied " + secret}), 502, "server_error", "denied Bearer [REDACTED]"},
	} {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := log.Writer()
			log.SetOutput(&logs)
			t.Cleanup(func() { log.SetOutput(previous) })
			var forwarded map[string]any
			environment := newScriptedEnvironment(t, []map[string]any{created, text, test.event}, &forwarded)
			request := func(stream bool) (*http.Response, []byte) {
				return environment.request(t, http.MethodPost, "/v1/chat/completions", map[string]any{
					"model": "gpt-5.6-luna", "store": true, "stream": stream,
					"messages": []any{map[string]any{"role": "user", "content": "fail"}},
				})
			}
			assertError := func(document map[string]any) {
				t.Helper()
				detail := mapAny(document["error"])
				if detail["message"] != test.message || detail["code"] != test.code {
					t.Fatalf("error = %#v, want message %q code %v", detail, test.message, test.code)
				}
			}

			response, data := request(false)
			if response.StatusCode != test.status {
				t.Fatalf("status = %d, want %d; body=%s", response.StatusCode, test.status, data)
			}
			var document map[string]any
			if err := json.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			assertError(document)

			response, data = request(true)
			if response.StatusCode != http.StatusOK {
				t.Fatalf("stream status = %d; body=%s", response.StatusCode, data)
			}
			if !bytes.HasSuffix(data, []byte("data: [DONE]\n\n")) {
				t.Fatalf("stream did not end with DONE: %s", data)
			}
			chunks := parseSSE(t, data)
			assertError(chunks[len(chunks)-1])
			for _, chunk := range chunks {
				for _, rawChoice := range sliceAny(chunk["choices"]) {
					if reason := mapAny(rawChoice)["finish_reason"]; reason != nil {
						t.Fatalf("failed stream reported finish_reason %v", reason)
					}
				}
			}

			listed := environment.json(t, http.MethodGet, "/v1/chat/completions", nil, http.StatusOK)
			if stored := sliceAny(listed["data"]); len(stored) != 0 {
				t.Fatalf("failed responses were stored: %#v", stored)
			}
			if strings.Contains(string(data), secret) || strings.Contains(logs.String(), secret) {
				t.Fatalf("failure exposed a credential: %s; logs=%s", data, logs.String())
			}
			if !strings.Contains(logs.String(), "codex.response.failed") {
				t.Fatalf("missing failure log: %s", logs.String())
			}
		})
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// scriptedAttempt is one Codex response: an HTTP error, or events then EOF.
type scriptedAttempt struct {
	status  int
	headers map[string]string
	body    string
	events  []map[string]any
}

func newScriptedAttempts(t *testing.T, attempts []scriptedAttempt, mutate func(*config)) (*contractEnvironment, *atomic.Int32) {
	t.Helper()
	var count atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := attempts[min(int(count.Add(1))-1, len(attempts)-1)]
		for key, value := range attempt.headers {
			w.Header().Set(key, value)
		}
		if attempt.status != 0 {
			w.WriteHeader(attempt.status)
			io.WriteString(w, attempt.body)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range attempt.events {
			encoded, _ := json.Marshal(event)
			fmt.Fprintf(w, "data: %s\n\n", encoded)
		}
	}))
	t.Cleanup(upstream.Close)
	return newContractEnvironment(t, func(cfg *config) {
		cfg.BackendURL = upstream.URL
		if mutate != nil {
			mutate(cfg)
		}
	}), &count
}

func TestGoHTTPContractRetriesTransientFailuresBeforeOutput(t *testing.T) {
	created := map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_retry", "status": "in_progress"}}
	text := map[string]any{"type": "response.output_text.delta", "output_index": 0, "delta": "Recovered"}
	completed := map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_retry", "status": "completed", "output": []any{
		map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "Recovered"}}},
	}}}
	failed := func(detail map[string]any) map[string]any {
		return map[string]any{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "failed", "output": []any{}, "error": detail}}
	}
	success := scriptedAttempt{events: []map[string]any{created, text, completed}}
	serverError := failed(map[string]any{"code": "server_error", "message": "broke"})
	chat := map[string]any{"model": "gpt-5.6-luna", "messages": []any{map[string]any{"role": "user", "content": "retry"}}}
	withRetries := func(cfg *config) { cfg.MaxRetries = 2 }

	for name, first := range map[string]scriptedAttempt{
		"response failed": {events: []map[string]any{created, serverError}},
		"error event":     {events: []map[string]any{created, {"type": "error", "code": "server_error", "message": "broke"}}},
		"closed early":    {events: []map[string]any{created}},
		"http 500":        {status: 500, body: `{"error":{"message":"broke"}}`},
		"rate limit":      {events: []map[string]any{failed(map[string]any{"code": "rate_limit_exceeded", "message": "slow"})}},
		"overload advice": {events: []map[string]any{failed(map[string]any{"code": "server_is_overloaded", "headers": map[string]any{"retry-after": "0"}})}},
		"http 429 advice": {status: 429, headers: map[string]string{"Retry-After": "0"}, body: `{"error":{"message":"wait"}}`},
	} {
		t.Run(name, func(t *testing.T) {
			environment, count := newScriptedAttempts(t, []scriptedAttempt{first, success}, withRetries)
			document := environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusOK)
			if content := mapAny(mapAny(sliceAny(document["choices"])[0])["message"])["content"]; content != "Recovered" {
				t.Fatalf("completion = %#v", document)
			}
			if count.Load() != 2 {
				t.Fatalf("attempts = %d, want 2", count.Load())
			}

			environment, count = newScriptedAttempts(t, []scriptedAttempt{first, success}, withRetries)
			request := cloneMap(chat)
			request["stream"] = true
			_, data := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
			chunks := parseSSE(t, data)
			roles, content := 0, ""
			for _, chunk := range chunks {
				if chunk["error"] != nil {
					t.Fatalf("stream reported the retried failure: %s", data)
				}
				for _, rawChoice := range sliceAny(chunk["choices"]) {
					delta := mapAny(mapAny(rawChoice)["delta"])
					if delta["role"] != nil {
						roles++
					}
					content += stringValue(delta["content"])
				}
			}
			if roles != 1 || content != "Recovered" || count.Load() != 2 {
				t.Fatalf("roles = %d content = %q attempts = %d; %s", roles, content, count.Load(), data)
			}

			environment, count = newScriptedAttempts(t, []scriptedAttempt{first, success}, withRetries)
			_, data = environment.request(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "retry", "stream": true})
			events := parseSSE(t, data)
			createdEvents := 0
			for _, event := range events {
				switch event["type"] {
				case "response.created":
					createdEvents++
				case "error", "response.failed":
					t.Fatalf("responses stream reported the retried failure: %s", data)
				}
			}
			if createdEvents != 1 || events[len(events)-1]["type"] != "response.completed" {
				t.Fatalf("responses stream = %s", data)
			}
		})
	}

	t.Run("non-streaming requests retry after partial output", func(t *testing.T) {
		environment, count := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{created, text, serverError}}, success}, withRetries)
		document := environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusOK)
		if content := mapAny(mapAny(sliceAny(document["choices"])[0])["message"])["content"]; content != "Recovered" || count.Load() != 2 {
			t.Fatalf("attempts = %d; %#v", count.Load(), document)
		}
	})

	t.Run("a terminal code after an error event is not retried", func(t *testing.T) {
		environment, count := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{
			{"type": "error", "message": "failed"}, failed(map[string]any{"code": "context_length_exceeded", "message": "long"}),
		}}, success}, withRetries)
		document := environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusBadRequest)
		if mapAny(document["error"])["code"] != "context_length_exceeded" || count.Load() != 1 {
			t.Fatalf("attempts = %d; %#v", count.Load(), document)
		}
	})

	t.Run("no retry after output", func(t *testing.T) {
		environment, count := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{created, text, serverError}}, success}, withRetries)
		request := cloneMap(chat)
		request["stream"] = true
		_, data := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
		chunks := parseSSE(t, data)
		if count.Load() != 1 || mapAny(chunks[len(chunks)-1]["error"])["code"] != "server_error" {
			t.Fatalf("attempts = %d; %s", count.Load(), data)
		}
	})

	for name, test := range map[string]struct {
		attempt scriptedAttempt
		status  int
	}{
		"context length":        {scriptedAttempt{events: []map[string]any{failed(map[string]any{"code": "context_length_exceeded"})}}, 400},
		"quota":                 {scriptedAttempt{events: []map[string]any{failed(map[string]any{"code": "insufficient_quota"})}}, 429},
		"usage type":            {scriptedAttempt{status: 429, body: `{"error":{"type":"usage_limit_reached","message":"limit"}}`}, 429},
		"policy":                {scriptedAttempt{events: []map[string]any{failed(map[string]any{"code": "cyber_policy"})}}, 400},
		"overload without wait": {scriptedAttempt{events: []map[string]any{failed(map[string]any{"code": "server_is_overloaded"})}}, 503},
		"long retry advice":     {scriptedAttempt{status: 500, headers: map[string]string{"Retry-After": "120"}, body: `{"error":{"message":"later"}}`}, 500},
		"http 400":              {scriptedAttempt{status: 400, body: `{"error":{"message":"bad"}}`}, 400},
	} {
		t.Run("no retry for "+name, func(t *testing.T) {
			environment, count := newScriptedAttempts(t, []scriptedAttempt{test.attempt, success}, withRetries)
			response, data := environment.request(t, http.MethodPost, "/v1/chat/completions", chat)
			if response.StatusCode != test.status {
				t.Fatalf("status = %d, want %d; %s", response.StatusCode, test.status, data)
			}
			if count.Load() != 1 {
				t.Fatalf("attempts = %d, want 1", count.Load())
			}
		})
	}

	t.Run("retries stop at the limit", func(t *testing.T) {
		environment, count := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{serverError}}}, withRetries)
		document := environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusBadGateway)
		if mapAny(document["error"])["code"] != "server_error" || count.Load() != 3 {
			t.Fatalf("attempts = %d; %#v", count.Load(), document)
		}
		environment, _ = newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{created, serverError}}}, withRetries)
		response := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "fail"}, http.StatusOK)
		if response["status"] != "failed" || response["id"] != "resp_failed" {
			t.Fatalf("response = %#v", response)
		}
	})

	t.Run("zero disables retries", func(t *testing.T) {
		environment, count := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{serverError}}, success}, nil)
		environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusBadGateway)
		if count.Load() != 1 {
			t.Fatalf("attempts = %d", count.Load())
		}
	})
}

func TestGoHTTPContractStreamsReportEarlyFailuresAsHTTPErrors(t *testing.T) {
	limited := scriptedAttempt{status: 429, headers: map[string]string{"Retry-After": "120"}, body: `{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"slow","param":"input"}}`}
	for name, request := range map[string]struct {
		path string
		body map[string]any
	}{
		"chat":      {"/v1/chat/completions", map[string]any{"model": "gpt-5.6-luna", "stream": true, "messages": []any{map[string]any{"role": "user", "content": "fail"}}}},
		"responses": {"/v1/responses", map[string]any{"model": "gpt-5.6-luna", "stream": true, "input": "fail"}},
	} {
		t.Run(name, func(t *testing.T) {
			environment, _ := newScriptedAttempts(t, []scriptedAttempt{limited}, nil)
			response, data := environment.request(t, http.MethodPost, request.path, request.body)
			if response.StatusCode != http.StatusTooManyRequests || response.Header.Get("Retry-After") != "120" {
				t.Fatalf("status = %d Retry-After = %q; %s", response.StatusCode, response.Header.Get("Retry-After"), data)
			}
			if contentType := response.Header.Get("Content-Type"); !strings.HasPrefix(contentType, "application/json") {
				t.Fatalf("Content-Type = %q", contentType)
			}
			var document map[string]any
			if err := json.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			detail := mapAny(document["error"])
			if detail["code"] != "rate_limit_exceeded" || detail["type"] != "rate_limit_error" || detail["param"] != "input" {
				t.Fatalf("error = %#v", detail)
			}
		})
	}

	t.Run("chat failure event before output", func(t *testing.T) {
		environment, _ := newScriptedAttempts(t, []scriptedAttempt{{events: []map[string]any{
			{"type": "response.created", "response": map[string]any{"id": "resp_failed", "status": "in_progress"}},
			{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "failed", "error": map[string]any{"code": "context_length_exceeded", "message": "long"}}},
		}}}, nil)
		response, data := environment.request(t, http.MethodPost, "/v1/chat/completions", map[string]any{
			"model": "gpt-5.6-luna", "stream": true, "messages": []any{map[string]any{"role": "user", "content": "fail"}},
		})
		if response.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d; %s", response.StatusCode, data)
		}
	})
}

func TestGoHTTPContractIgnoresFailuresAfterCompletion(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		encoded, _ := json.Marshal(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_done", "status": "completed", "output": []any{}}})
		fmt.Fprintf(w, "data: %s\n\n", encoded)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(upstream.Close)
	environment := newContractEnvironment(t, func(cfg *config) {
		cfg.BackendURL = upstream.URL
		cfg.Timeout = 2 * time.Second
	})
	started := time.Now()
	response := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "done"}, http.StatusOK)
	if response["status"] != "completed" || time.Since(started) > time.Second {
		t.Fatalf("response = %#v after %s", response, time.Since(started))
	}
}

func TestBackendRetryDelayBounds(t *testing.T) {
	b := &backend{cfg: config{MaxRetries: 100, retryBaseDelay: retryBaseDelay}}
	transient := &backendError{http.StatusBadGateway, "broke"}
	for _, attempt := range []int{0, 5, 40, 80, 99} {
		delay, ok := b.retryDelay(transient, attempt)
		if !ok || delay <= 0 || delay > maxRetryAdvice+maxRetryAdvice/5 {
			t.Fatalf("attempt %d: delay = %s ok = %v", attempt, delay, ok)
		}
	}
	if _, ok := b.retryDelay(transient, 100); ok {
		t.Fatal("retried past the limit")
	}
	if _, ok := b.retryDelay(&backendError{http.StatusGatewayTimeout, "stalled"}, 0); ok {
		t.Fatal("retried a stalled stream")
	}
	if _, ok := b.retryDelay(errSSELimit, 0); ok {
		t.Fatal("retried an oversized event")
	}
}

func TestGoHTTPContractStreamClosedBeforeCompletion(t *testing.T) {
	text := map[string]any{"type": "response.output_text.delta", "output_index": 0, "delta": "Partial"}
	message := map[string]any{"type": "response.output_item.done", "output_index": 0, "item": map[string]any{
		"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "Partial"}},
	}}
	closed := "Codex stream closed before the response completed."
	var forwarded map[string]any
	environment := newScriptedEnvironment(t, []map[string]any{text, message, {"type": "response.completed"}}, &forwarded)
	chat := map[string]any{"model": "gpt-5.6-luna", "store": true, "messages": []any{map[string]any{"role": "user", "content": "cut"}}}

	document := environment.json(t, http.MethodPost, "/v1/chat/completions", chat, http.StatusBadGateway)
	if mapAny(document["error"])["message"] != closed {
		t.Fatalf("error = %#v", document)
	}
	if strings.Contains(string(mustJSON(t, document)), "Partial") {
		t.Fatalf("error exposed partial output: %#v", document)
	}
	if stored := sliceAny(environment.json(t, http.MethodGet, "/v1/chat/completions", nil, http.StatusOK)["data"]); len(stored) != 0 {
		t.Fatalf("truncated completion was stored: %#v", stored)
	}

	chat["stream"] = true
	_, data := environment.request(t, http.MethodPost, "/v1/chat/completions", chat)
	chunks := parseSSE(t, data)
	if mapAny(chunks[len(chunks)-1]["error"])["message"] != closed {
		t.Fatalf("chat stream = %s", data)
	}
	for _, chunk := range chunks {
		for _, rawChoice := range sliceAny(chunk["choices"]) {
			if reason := mapAny(rawChoice)["finish_reason"]; reason != nil {
				t.Fatalf("truncated stream reported finish_reason %v", reason)
			}
		}
	}

	_, data = environment.request(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "cut", "stream": true})
	events := parseSSE(t, data)
	if last := events[len(events)-1]; last["type"] != "error" || last["message"] != closed {
		t.Fatalf("responses stream = %s", data)
	}
}

func TestGoHTTPContractStreamTimeoutAppliesBetweenEvents(t *testing.T) {
	const gap = 60 * time.Millisecond
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	var stall atomic.Bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		write := func(event map[string]any) {
			encoded, _ := json.Marshal(event)
			fmt.Fprintf(w, "data: %s\n\n", encoded)
			w.(http.Flusher).Flush()
		}
		for range 5 {
			write(map[string]any{"type": "response.output_text.delta", "output_index": 0, "delta": "x"})
			time.Sleep(gap)
		}
		if stall.Load() {
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		write(map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_slow", "status": "completed", "output": []any{}}})
	}))
	t.Cleanup(upstream.Close)
	environment := newContractEnvironment(t, func(cfg *config) {
		cfg.BackendURL = upstream.URL
		cfg.Timeout = 3 * gap
	})
	request := map[string]any{"model": "gpt-5.6-luna", "stream": true, "messages": []any{map[string]any{"role": "user", "content": "slow"}}}

	_, data := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
	for _, chunk := range parseSSE(t, data) {
		if chunk["error"] != nil {
			t.Fatalf("a stream with regular events timed out: %s", data)
		}
	}
	if !strings.Contains(string(data), `"finish_reason":"stop"`) {
		t.Fatalf("slow stream did not finish: %s", data)
	}

	stall.Store(true)
	_, data = environment.request(t, http.MethodPost, "/v1/chat/completions", request)
	chunks := parseSSE(t, data)
	if mapAny(chunks[len(chunks)-1]["error"])["message"] != "Codex backend stream timed out." {
		t.Fatalf("stalled stream = %s", data)
	}
}

func TestGoHTTPContractChatFinishReasonForIncompleteResponses(t *testing.T) {
	for reason, want := range map[string]string{
		"content_filter":    "content_filter",
		"max_output_tokens": "length",
		"interrupted":       "length",
	} {
		t.Run(reason, func(t *testing.T) {
			var forwarded map[string]any
			environment := newScriptedEnvironment(t, []map[string]any{{
				"type":     "response.incomplete",
				"response": map[string]any{"id": "resp_cut", "status": "incomplete", "output": []any{}, "incomplete_details": map[string]any{"reason": reason}},
			}}, &forwarded)
			document := environment.json(t, http.MethodPost, "/v1/chat/completions", map[string]any{
				"model": "gpt-5.6-luna", "messages": []any{map[string]any{"role": "user", "content": "cut"}},
			}, http.StatusOK)
			if got := mapAny(sliceAny(document["choices"])[0])["finish_reason"]; got != want {
				t.Fatalf("finish_reason = %v, want %s", got, want)
			}
		})
	}
}

func TestGoHTTPContractForwardsUpstreamErrorDetails(t *testing.T) {
	chat := map[string]any{"model": "gpt-5.6-luna", "messages": []any{map[string]any{"role": "user", "content": "fail"}}}
	for name, test := range map[string]struct {
		attempt    scriptedAttempt
		status     int
		code       any
		errorType  string
		message    string
		param      any
		retryAfter string
	}{
		"redacted http":    {scriptedAttempt{status: 400, body: `{"error":{"code":"invalid_value","message":"denied Bearer abcdefghijklmnopqrstuvwxyz0123456789","param":"Bearer abcdefghijklmnopqrstuvwxyz0123456789"}}`}, 400, "invalid_value", "invalid_request_error", "denied Bearer [REDACTED]", "Bearer [REDACTED]", ""},
		"policy over http": {scriptedAttempt{status: 400, body: `{"error":{"code":"cyber_policy"}}`}, 400, "cyber_policy", "invalid_request_error", "This request has been flagged for possible cybersecurity risk.", nil, ""},
		"detail object":    {scriptedAttempt{status: 400, body: `{"detail":{"code":"invalid_value","message":"nested"}}`}, 400, "invalid_value", "invalid_request_error", "nested", nil, ""},
		"http details":     {scriptedAttempt{status: 400, body: `{"error":{"message":"bad","type":"invalid_request_error","code":"invalid_value","param":"input"}}`}, 400, "invalid_value", "invalid_request_error", "bad", "input", ""},
		"usage type":       {scriptedAttempt{status: 429, headers: map[string]string{"Retry-After": "120"}, body: `{"error":{"type":"usage_limit_reached","message":"limit"}}`}, 429, "usage_limit_reached", "usage_limit_reached", "limit", nil, "120"},
		"detail field":     {scriptedAttempt{status: 400, body: `{"detail":"model not supported"}`}, 400, nil, "invalid_request_error", "model not supported", nil, ""},
		"plain body":       {scriptedAttempt{status: 400, body: `nope`}, 400, nil, "invalid_request_error", "400 Bad Request", nil, ""},
		"event advice":     {scriptedAttempt{events: []map[string]any{{"type": "response.failed", "response": map[string]any{"status": "failed", "error": map[string]any{"code": "rate_limit_exceeded", "message": "slow", "headers": map[string]any{"Retry-After": "90"}}}}}}, 429, "rate_limit_exceeded", "rate_limit_error", "slow", nil, "90"},
		"bad advice":       {scriptedAttempt{status: 429, headers: map[string]string{"Retry-After": "soon"}, body: `{"error":{"message":"wait"}}`}, 429, nil, "rate_limit_error", "wait", nil, ""},
		"policy text":      {scriptedAttempt{events: []map[string]any{{"type": "response.failed", "response": map[string]any{"status": "failed", "error": map[string]any{"code": "bio_policy", "message": " "}}}}}, 400, "bio_policy", "invalid_request_error", "This content was flagged for possible biological risk.", nil, ""},
		"not found":        {scriptedAttempt{events: []map[string]any{{"type": "response.failed", "response": map[string]any{"status": "failed", "error": map[string]any{"code": "model_not_found", "message": "gone"}}}}}, 404, "model_not_found", "invalid_request_error", "gone", nil, ""},
	} {
		t.Run(name, func(t *testing.T) {
			environment, _ := newScriptedAttempts(t, []scriptedAttempt{test.attempt}, nil)
			response, data := environment.request(t, http.MethodPost, "/v1/chat/completions", chat)
			var document map[string]any
			if err := json.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			detail := mapAny(document["error"])
			if response.StatusCode != test.status || detail["code"] != test.code || detail["type"] != test.errorType || detail["message"] != test.message || detail["param"] != test.param {
				t.Fatalf("status = %d; %s", response.StatusCode, data)
			}
			if got := response.Header.Get("Retry-After"); got != test.retryAfter {
				t.Fatalf("Retry-After = %q, want %q", got, test.retryAfter)
			}
		})
	}
}

func TestGoHTTPContractChatFailureErrorShape(t *testing.T) {
	for name, test := range map[string]struct {
		detail    map[string]any
		wantType  string
		wantParam any
	}{
		"mapped type":   {map[string]any{"code": "context_length_exceeded", "message": "long"}, "invalid_request_error", nil},
		"upstream type": {map[string]any{"code": "invalid_prompt", "message": "bad", "type": "custom_error", "param": "input"}, "custom_error", "input"},
		"rate limit":    {map[string]any{"code": "rate_limit_exceeded", "message": "slow"}, "rate_limit_error", nil},
		"default":       {map[string]any{"code": "server_error", "message": "broke"}, "api_error", nil},
	} {
		t.Run(name, func(t *testing.T) {
			var forwarded map[string]any
			environment := newScriptedEnvironment(t, []map[string]any{
				{"type": "response.output_text.delta", "output_index": 0, "delta": "Partial"},
				{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "failed", "error": test.detail}},
			}, &forwarded)
			request := map[string]any{"model": "gpt-5.6-luna", "messages": []any{map[string]any{"role": "user", "content": "fail"}}}
			_, data := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
			var document map[string]any
			if err := json.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			request["stream"] = true
			_, streamed := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
			chunks := parseSSE(t, streamed)
			for _, source := range []map[string]any{document, chunks[len(chunks)-1]} {
				detail := mapAny(source["error"])
				if detail["type"] != test.wantType || detail["param"] != test.wantParam {
					t.Fatalf("error = %#v, want type %q param %v", detail, test.wantType, test.wantParam)
				}
			}
		})
	}
}

func TestGoHTTPContractResponsesKeepFailedResponseObject(t *testing.T) {
	failedEvent := map[string]any{"type": "response.failed", "response": map[string]any{"id": "resp_failed", "status": "failed", "output": []any{}, "error": map[string]any{"code": "server_error", "message": "upstream broke"}}}
	errorEvent := map[string]any{"type": "error", "code": "server_error", "message": "upstream broke"}
	for name, events := range map[string][]map[string]any{
		"failed only":     {failedEvent},
		"error first":     {errorEvent, failedEvent},
		"error last":      {failedEvent, errorEvent},
		"failed no error": {errorEvent, {"type": "response.failed", "response": map[string]any{"id": "resp_failed", "output": []any{}}}},
	} {
		t.Run(name, func(t *testing.T) {
			var forwarded map[string]any
			environment := newScriptedEnvironment(t, events, &forwarded)
			response := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "fail"}, http.StatusOK)
			if response["status"] != "failed" || mapAny(response["error"])["code"] != "server_error" {
				t.Fatalf("response = %#v", response)
			}
		})
	}

	t.Run("error without a response", func(t *testing.T) {
		var forwarded map[string]any
		environment := newScriptedEnvironment(t, []map[string]any{errorEvent}, &forwarded)
		document := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "fail"}, http.StatusBadGateway)
		if mapAny(document["error"])["code"] != "server_error" {
			t.Fatalf("error = %#v", document)
		}
	})

	t.Run("completed response survives a later error", func(t *testing.T) {
		var forwarded map[string]any
		environment := newScriptedEnvironment(t, []map[string]any{
			{"type": "response.completed", "response": map[string]any{"id": "resp_done", "status": "completed", "output": []any{}}},
			errorEvent,
		}, &forwarded)
		response := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{"model": "gpt-5.6-luna", "input": "fail"}, http.StatusOK)
		if response["status"] != "completed" {
			t.Fatalf("response = %#v", response)
		}
	})
}

func TestGoHTTPContractImagesReportFailedResponses(t *testing.T) {
	var forwarded map[string]any
	environment := newScriptedEnvironment(t, []map[string]any{{
		"type": "response.failed",
		"response": map[string]any{
			"id": "resp_failed", "status": "failed",
			"output": []any{map[string]any{"type": "image_generation_call", "result": "aW1hZ2U="}},
			"error":  map[string]any{"code": "rate_limit_exceeded", "message": "slow down"},
		},
	}}, &forwarded)
	document := environment.json(t, http.MethodPost, "/v1/images/generations", map[string]any{"model": "gpt-image-1", "prompt": "fail"}, http.StatusTooManyRequests)
	if detail := mapAny(document["error"]); detail["code"] != "rate_limit_exceeded" || detail["message"] != "slow down" {
		t.Fatalf("error = %#v", document)
	}
}

func TestGoHTTPContractChatStreamToolCallEdgeCases(t *testing.T) {
	functionCall := func(name, arguments string) map[string]any {
		return map[string]any{"type": "function_call", "id": "fc_" + name, "call_id": "call_" + name, "name": name, "arguments": arguments}
	}
	added := func(index int, name string) map[string]any {
		return map[string]any{"type": "response.output_item.added", "output_index": index, "item": functionCall(name, "")}
	}
	done := func(index int, name, arguments string) map[string]any {
		return map[string]any{"type": "response.output_item.done", "output_index": index, "item": functionCall(name, arguments)}
	}
	delta := func(index int, text string) map[string]any {
		return map[string]any{"type": "response.function_call_arguments.delta", "output_index": index, "delta": text}
	}
	created := map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_edge", "status": "in_progress"}}
	completed := func(items ...any) map[string]any {
		return map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_edge", "status": "completed", "output": items}}
	}
	first, second := functionCall("first", `{"a":1}`), functionCall("second", `{"b":2}`)
	reasoning := map[string]any{"type": "response.output_item.done", "output_index": 0, "item": map[string]any{"type": "reasoning", "id": "rs_edge"}}
	message := map[string]any{"type": "message", "id": "msg_edge", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "Checking"}}}
	tools := []any{map[string]any{"type": "function", "function": map[string]any{"name": "first", "parameters": map[string]any{"type": "object"}}}}
	functions := []any{map[string]any{"name": "first", "parameters": map[string]any{"type": "object"}}}

	stream := func(t *testing.T, events []map[string]any, request map[string]any) ([]map[string]any, map[string]any) {
		t.Helper()
		var forwarded map[string]any
		environment := newScriptedEnvironment(t, events, &forwarded)
		request["model"] = "gpt-5.6-luna"
		request["stream"] = true
		request["messages"] = []any{map[string]any{"role": "user", "content": "edge"}}
		response, data := environment.request(t, http.MethodPost, "/v1/chat/completions", request)
		if response.StatusCode != http.StatusOK {
			t.Fatalf("status = %d; body=%s", response.StatusCode, data)
		}
		return parseSSE(t, data), forwarded
	}
	assembleTools := func(chunks []map[string]any) map[int]string {
		arguments := map[int]string{}
		for _, chunk := range chunks {
			for _, rawChoice := range sliceAny(chunk["choices"]) {
				for _, rawCall := range sliceAny(mapAny(mapAny(rawChoice)["delta"])["tool_calls"]) {
					call := mapAny(rawCall)
					arguments[intValue(call["index"])] += stringValue(mapAny(call["function"])["arguments"])
				}
			}
		}
		return arguments
	}

	for name, events := range map[string][]map[string]any{
		"reasoning before calls": {created, reasoning, added(1, "first"), done(1, "first", `{"a":1}`), added(2, "second"), done(2, "second", `{"b":2}`), completed(first, second)},
		"text before calls": {
			created,
			{"type": "response.output_text.delta", "output_index": 0, "delta": "Checking"},
			{"type": "response.output_item.done", "output_index": 0, "item": message},
			added(1, "first"), delta(1, `{"a":1}`), done(1, "first", `{"a":1}`),
			added(2, "second"), done(2, "second", `{"b":2}`),
			completed(message, first, second),
		},
		"partial deltas": {created, added(0, "first"), delta(0, `{"a"`), done(0, "first", `{"a":1}`), added(1, "second"), delta(1, `{"b":2}`), done(1, "second", `{"b":2}`), completed(first, second)},
	} {
		t.Run(name, func(t *testing.T) {
			chunks, _ := stream(t, events, map[string]any{"tools": tools})
			arguments := assembleTools(chunks)
			if len(arguments) != 2 || arguments[0] != `{"a":1}` || arguments[1] != `{"b":2}` {
				t.Fatalf("streamed tool arguments = %#v", arguments)
			}
		})
	}

	t.Run("legacy functions keep only the first call", func(t *testing.T) {
		chunks, forwarded := stream(t, []map[string]any{
			created, added(0, "first"), done(0, "first", `{"a":1}`),
			added(1, "second"), delta(1, `{"b":2}`), done(1, "second", `{"b":2}`), completed(first, second),
		}, map[string]any{"functions": functions})
		if forwarded["parallel_tool_calls"] != false {
			t.Fatalf("forwarded parallel_tool_calls = %#v", forwarded["parallel_tool_calls"])
		}
		name, arguments := "", ""
		for _, chunk := range chunks {
			for _, rawChoice := range sliceAny(chunk["choices"]) {
				call := mapAny(mapAny(mapAny(rawChoice)["delta"])["function_call"])
				name += stringValue(call["name"])
				arguments += stringValue(call["arguments"])
			}
		}
		if name != "first" || arguments != `{"a":1}` {
			t.Fatalf("legacy function call name = %q arguments = %q", name, arguments)
		}
	})

	for name, sent := range map[string]string{
		"truncated": `{"secret": `,
		"different": `{"secret":"other"}`,
	} {
		t.Run(name+" arguments log sizes only", func(t *testing.T) {
			var logs bytes.Buffer
			previous := log.Writer()
			log.SetOutput(&logs)
			t.Cleanup(func() { log.SetOutput(previous) })
			chunks, _ := stream(t, []map[string]any{
				created, added(0, "first"), delta(0, sent), done(0, "first", `{"secret":"value"}`), completed(first),
			}, map[string]any{"tools": tools})
			if arguments := assembleTools(chunks); arguments[0] != sent {
				t.Fatalf("streamed tool arguments = %#v", arguments)
			}
			want := fmt.Sprintf("chat.stream.tool_arguments_mismatch output_index=0 sent_bytes=%d final_bytes=18", len(sent))
			if !strings.Contains(logs.String(), want) {
				t.Fatalf("missing mismatch log: %s", logs.String())
			}
			if strings.Contains(logs.String(), "secret") {
				t.Fatalf("log contains tool arguments: %s", logs.String())
			}
		})
	}

	t.Run("reformatted complete arguments do not log", func(t *testing.T) {
		var logs bytes.Buffer
		previous := log.Writer()
		log.SetOutput(&logs)
		t.Cleanup(func() { log.SetOutput(previous) })
		stream(t, []map[string]any{
			created, added(0, "first"), delta(0, `{"a":1}`), done(0, "first", `{"a": 1}`), completed(first),
		}, map[string]any{"tools": tools})
		if strings.Contains(logs.String(), "tool_arguments_mismatch") {
			t.Fatalf("unexpected mismatch log: %s", logs.String())
		}
	})
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

	for _, marker := range []string{"Use the weather tool", "Use the weather tool FAKE_NO_ARGUMENT_DELTA"} {
		toolStreamResponse, toolStreamData := environment.request(t, http.MethodPost, "/v1/chat/completions", map[string]any{
			"model": "gpt-5.6-luna", "stream": true,
			"messages": []any{map[string]any{"role": "user", "content": marker}},
			"tools": []any{map[string]any{"type": "function", "function": map[string]any{
				"name": "lookup_weather", "parameters": map[string]any{"type": "object"},
			}}},
		})
		if toolStreamResponse.StatusCode != http.StatusOK {
			t.Fatalf("tool stream status = %d", toolStreamResponse.StatusCode)
		}
		arguments := ""
		for _, chunk := range parseSSE(t, toolStreamData) {
			for _, rawChoice := range sliceAny(chunk["choices"]) {
				for _, rawCall := range sliceAny(mapAny(mapAny(rawChoice)["delta"])["tool_calls"]) {
					arguments += stringValue(mapAny(mapAny(rawCall)["function"])["arguments"])
				}
			}
		}
		if arguments != `{"city":"Tokyo"}` {
			t.Fatalf("%q: streamed tool arguments = %q", marker, arguments)
		}
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
	truncated := environment.json(t, http.MethodPost, "/v1/responses", map[string]any{
		"model": "gpt-5.6-luna", "input": "FAKE_DELTA_ONLY",
	}, http.StatusBadGateway)
	if !strings.Contains(stringValue(mapAny(truncated["error"])["message"]), "closed before") {
		t.Fatalf("truncated response = %#v", truncated)
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
