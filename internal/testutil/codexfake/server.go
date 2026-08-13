package codexfake

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type Request struct {
	Method  string
	Path    string
	Query   string
	Headers http.Header
	JSON    map[string]any
	Body    []byte
}

type Server struct {
	*httptest.Server

	counter atomic.Int64
	mu      sync.Mutex
	seen    []Request
	active  int
	maximum int
	gate    <-chan struct{}
	started chan<- struct{}
}

func New(t testing.TB) *Server {
	t.Helper()
	fake := &Server{}
	fake.Server = httptest.NewServer(http.HandlerFunc(fake.handle))
	t.Cleanup(fake.Close)
	return fake
}

func (s *Server) BackendURL() string {
	return s.URL + "/backend-api/codex"
}

func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	result := make([]Request, len(s.seen))
	copy(result, s.seen)
	return result
}

func (s *Server) LastResponseRequest(t testing.TB) Request {
	t.Helper()
	requests := s.Requests()
	for index := len(requests) - 1; index >= 0; index-- {
		if requests[index].Path == "/backend-api/codex/responses" {
			return requests[index]
		}
	}
	t.Fatal("fake Codex received no responses request")
	return Request{}
}

func (s *Server) GateResponses(gate <-chan struct{}, started chan<- struct{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gate = gate
	s.started = started
}

func (s *Server) MaximumActiveResponses() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maximum
}

func (s *Server) handle(w http.ResponseWriter, request *http.Request) {
	body, _ := io.ReadAll(request.Body)
	var document map[string]any
	_ = json.Unmarshal(body, &document)
	s.mu.Lock()
	s.seen = append(s.seen, Request{
		Method: request.Method, Path: request.URL.Path, Query: request.URL.RawQuery,
		Headers: request.Header.Clone(), JSON: document, Body: append([]byte(nil), body...),
	})
	s.mu.Unlock()

	switch {
	case request.Method == http.MethodGet && request.URL.Path == "/backend-api/codex/models":
		writeJSON(w, map[string]any{"models": []any{
			map[string]any{"slug": "gpt-5.6-luna", "supported_in_api": true, "visibility": "list"},
			map[string]any{"slug": "hidden-model", "supported_in_api": true, "visibility": "hidden"},
			map[string]any{"slug": "unsupported-model", "supported_in_api": false, "visibility": "list"},
		}})
	case request.Method == http.MethodPost && request.URL.Path == "/backend-api/codex/responses":
		s.responses(w, document)
	case request.Method == http.MethodPost && request.URL.Path == "/backend-api/transcribe":
		writeJSON(w, map[string]any{"text": "transcribed by Go fake Codex"})
	default:
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Request-ID", "fake-upstream-request")
		w.Header().Add("Set-Cookie", "session=must-not-forward")
		writeJSON(w, map[string]any{
			"object": "list", "data": []any{}, "has_more": false,
			"request_method": request.Method, "request_path": request.URL.RequestURI(),
		})
	}
}

func (s *Server) responses(w http.ResponseWriter, payload map[string]any) {
	s.mu.Lock()
	s.active++
	if s.active > s.maximum {
		s.maximum = s.active
	}
	gate, started := s.gate, s.started
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.active--
		s.mu.Unlock()
	}()

	number := s.counter.Add(1)
	responseID := fmt.Sprintf("resp_go_contract_%d", number)
	createdAt := float64(time.Now().Unix())
	text := "fake Go contract: " + flattenText(payload["input"])
	if strings.Contains(text, "FAKE_HTTP_ERROR") {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		writeJSON(w, map[string]any{"error": map[string]any{
			"message": "upstream access_token=tok_abcdefghijklmnopqrstuvwxyz failed",
		}})
		return
	}
	item := outputMessage(number, text)
	if hasTool(payload, "image_generation") {
		item = map[string]any{
			"id": fmt.Sprintf("ig_go_contract_%d", number), "type": "image_generation_call",
			"status":         "completed",
			"result":         base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nGo-contract")),
			"revised_prompt": "revised by Go fake Codex",
		}
	} else if tool := firstFunctionTool(payload); tool != nil {
		item = map[string]any{
			"id": fmt.Sprintf("fc_go_contract_%d", number), "type": "function_call",
			"call_id": fmt.Sprintf("call_go_contract_%d", number), "status": "completed",
			"name": tool["name"], "arguments": `{"city":"Tokyo"}`,
		}
	}
	response := map[string]any{
		"id": responseID, "object": "response", "created_at": createdAt,
		"status": "completed", "model": payload["model"], "output": []any{item},
		"parallel_tool_calls": true, "tool_choice": valueOr(payload["tool_choice"], "auto"),
		"tools": slice(payload["tools"]),
		"usage": map[string]any{"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
	}
	created := clone(response)
	created["status"] = "in_progress"
	created["output"] = []any{}

	w.Header().Set("Content-Type", "text/event-stream")
	flusher, _ := w.(http.Flusher)
	writeSSE(w, map[string]any{"type": "response.created", "sequence_number": 0, "response": created})
	if flusher != nil {
		flusher.Flush()
	}
	if started != nil {
		started <- struct{}{}
	}
	if gate != nil {
		<-gate
	}

	if strings.Contains(text, "FAKE_UPSTREAM_ERROR") {
		return
	}
	if item["type"] == "function_call" {
		added := clone(item)
		added["arguments"] = ""
		added["status"] = "in_progress"
		writeSSE(w, map[string]any{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": added})
		writeSSE(w, map[string]any{"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0, "delta": `{"city":"Tokyo"}`})
	} else if item["type"] == "message" {
		writeSSE(w, map[string]any{
			"type": "response.output_text.delta", "sequence_number": 1,
			"output_index": 0, "content_index": 0, "item_id": item["id"],
			"delta": text, "logprobs": []any{},
		})
	}
	writeSSE(w, map[string]any{"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": item})
	if strings.Contains(text, "FAKE_DELTA_ONLY") {
		return
	}
	if strings.Contains(text, "FAKE_RESPONSE_FAILED") {
		response["status"] = "failed"
		writeSSE(w, map[string]any{"type": "response.failed", "sequence_number": 4, "response": response})
		return
	}
	terminalType := "response.completed"
	if strings.Contains(text, "FAKE_RAW_DONE") {
		terminalType = "response.done"
		response["status"] = "backend-specific-status"
	}
	writeSSE(w, map[string]any{"type": terminalType, "sequence_number": 4, "response": response})
	w.Write([]byte("data: [DONE]\n\n"))
}

func outputMessage(number int64, text string) map[string]any {
	return map[string]any{
		"id": fmt.Sprintf("msg_go_contract_%d", number), "type": "message",
		"role": "assistant", "status": "completed", "phase": "final_answer",
		"content": []any{map[string]any{
			"type": "output_text", "text": text, "annotations": []any{},
		}},
	}
}

func firstFunctionTool(payload map[string]any) map[string]any {
	for _, raw := range slice(payload["tools"]) {
		if tool, ok := raw.(map[string]any); ok && tool["type"] == "function" {
			return tool
		}
	}
	return nil
}

func hasTool(payload map[string]any, toolType string) bool {
	for _, raw := range slice(payload["tools"]) {
		if tool, ok := raw.(map[string]any); ok && tool["type"] == toolType {
			return true
		}
	}
	return false
}

func flattenText(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case []any:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			if text := flattenText(item); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, " ")
	case map[string]any:
		if typed["type"] == "input_text" || typed["type"] == "output_text" || typed["type"] == "text" {
			return fmt.Sprint(typed["text"])
		}
		parts := []string{}
		for _, key := range []string{"content", "input", "output"} {
			if text := flattenText(typed[key]); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, " ")
	default:
		return ""
	}
}

func writeSSE(w io.Writer, event map[string]any) {
	data, _ := json.Marshal(event)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeJSON(w http.ResponseWriter, value any) {
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	_ = json.NewEncoder(w).Encode(value)
}

func slice(value any) []any {
	result, _ := value.([]any)
	return result
}

func valueOr(value, fallback any) any {
	if value == nil || value == "" {
		return fallback
	}
	return value
}

func clone(value map[string]any) map[string]any {
	data, _ := json.Marshal(value)
	var result map[string]any
	_ = json.Unmarshal(data, &result)
	return result
}
