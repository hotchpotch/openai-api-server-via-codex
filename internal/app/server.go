package app

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type server struct {
	cfg       config
	backend   codexBackend
	responses *responseStore
	chats     *chatStore
	slots     chan struct{}
}

type codexBackend interface {
	stream(context.Context, map[string]any, func(map[string]any) error) error
	collect(context.Context, map[string]any) (map[string]any, error)
	listModels(context.Context) []string
	proxy(context.Context, string, string, string, http.Header, io.Reader) (*http.Response, error)
	transcribe(context.Context, http.Header, io.Reader) (*http.Response, error)
}

const startupLogMarker = "listening on http://"

func startupLogMessage(version, address string) string {
	return fmt.Sprintf("openai-api-server-via-codex %s (Go) %s%s", version, startupLogMarker, address)
}

func serve(cfg config, version string) error {
	b := newBackend(cfg)
	if _, err := b.auth.borrow(); err != nil {
		return fmt.Errorf("Codex authentication preflight failed: %w", err)
	}
	s := &server{cfg: cfg, backend: b, responses: newResponseStore(cfg.MaxStored), chats: newChatStore(cfg.MaxStored)}
	if cfg.Concurrency > 0 {
		s.slots = make(chan struct{}, cfg.Concurrency)
	}
	address := net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port))
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	boundAddress := listener.Addr().String()
	httpServer := &http.Server{Addr: boundAddress, Handler: s, ReadHeaderTimeout: 10 * time.Second}
	log.Print(startupLogMessage(version, boundAddress))
	if cfg.Verbose {
		log.Printf(
			"settings.resolved host=%s port=%d model=%s timeout=%s max_stored_items=%d max_concurrent_requests=%d auth_json=%s backend_base_url=%s api_key_configured=%t",
			cfg.Host, cfg.Port, cfg.Model, cfg.Timeout, cfg.MaxStored, cfg.Concurrency,
			cfg.AuthJSON, cfg.BackendURL, cfg.APIKey != "",
		)
	}
	serveResult := make(chan error, 1)
	go func() { serveResult <- httpServer.Serve(listener) }()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)
	select {
	case err := <-serveResult:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-signals:
		shutdownTimeout := cfg.StopTimeout
		if shutdownTimeout <= 0 {
			shutdownTimeout = 10 * time.Second
		}
		ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			_ = httpServer.Close()
			return err
		}
		err := <-serveResult
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	capture := &responseCapture{ResponseWriter: w}
	started := time.Now()
	if s.cfg.Verbose {
		log.Printf("request.start method=%s path=%s query=%s", r.Method, redactSensitive(r.URL.Path), redactSensitive(r.URL.RawQuery))
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			log.Printf("request.unhandled_error method=%s path=%s panic_type=%T", r.Method, redactSensitive(r.URL.Path), recovered)
			if !capture.wroteHeader {
				writeError(capture, 500, "Internal server error.", "api_error", nil, nil)
			}
		}
		if s.cfg.Verbose {
			log.Printf("request.end method=%s path=%s status=%d bytes=%d duration_ms=%.1f", r.Method, redactSensitive(r.URL.Path), capture.statusCode(), capture.bytes, float64(time.Since(started).Microseconds())/1000)
		}
	}()
	s.serveHTTP(capture, r)
}

type responseCapture struct {
	http.ResponseWriter
	status      int
	bytes       int64
	wroteHeader bool
}

func (w *responseCapture) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.status = status
	w.wroteHeader = true
	w.ResponseWriter.WriteHeader(status)
}

func (w *responseCapture) Write(data []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	written, err := w.ResponseWriter.Write(data)
	w.bytes += int64(written)
	return written, err
}

func (w *responseCapture) Flush() {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *responseCapture) statusCode() int {
	if w.status == 0 {
		return http.StatusOK
	}
	return w.status
}

func (s *server) serveHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		if r.Method == http.MethodGet {
			writeJSON(w, 200, map[string]any{"status": "ok"})
		} else {
			methodNotAllowed(w)
		}
		return
	}
	if r.URL.Path != "/v1" && !strings.HasPrefix(r.URL.Path, "/v1/") {
		http.NotFound(w, r)
		return
	}
	if s.cfg.APIKey != "" && !validBearer(r.Header.Get("Authorization"), s.cfg.APIKey) {
		writeError(w, 401, "Incorrect API key provided.", "invalid_request_error", nil, "invalid_api_key")
		return
	}
	if r.URL.Path == "/v1" {
		http.Redirect(w, r, "/v1/", http.StatusTemporaryRedirect)
		return
	}
	w.Header().Set("X-Content-Type-Options", "nosniff")
	path := strings.TrimPrefix(r.URL.Path, "/v1/")
	switch {
	case r.Method == http.MethodGet && path == "models":
		s.models(w, r)
	case r.Method == http.MethodPost && path == "responses":
		s.createResponse(w, r)
	case r.Method == http.MethodPost && path == "responses/input_tokens":
		s.inputTokens(w, r)
	case strings.HasPrefix(path, "responses/") && isResponseResourceRoute(r.Method, strings.TrimPrefix(path, "responses/")):
		s.responseResource(w, r, strings.TrimPrefix(path, "responses/"))
	case (r.Method == http.MethodGet || r.Method == http.MethodPost) && path == "chat/completions":
		s.chatCollection(w, r)
	case strings.HasPrefix(path, "chat/completions/") && isChatResourceRoute(r.Method, strings.TrimPrefix(path, "chat/completions/")):
		s.chatResource(w, r, strings.TrimPrefix(path, "chat/completions/"))
	case r.Method == http.MethodPost && path == "audio/transcriptions":
		s.audio(w, r)
	case r.Method == http.MethodPost && path == "images/generations":
		s.images(w, r)
	default:
		s.proxy(w, r, path)
	}
}

func isResponseResourceRoute(method, resource string) bool {
	parts := strings.Split(resource, "/")
	if len(parts) == 1 && parts[0] != "" {
		return method == http.MethodGet || method == http.MethodDelete
	}
	if len(parts) != 2 || parts[0] == "" {
		return false
	}
	return (parts[1] == "cancel" && method == http.MethodPost) ||
		(parts[1] == "input_items" && method == http.MethodGet)
}

func isChatResourceRoute(method, resource string) bool {
	parts := strings.Split(resource, "/")
	if len(parts) == 1 && parts[0] != "" {
		return method == http.MethodGet || method == http.MethodPost || method == http.MethodDelete
	}
	return len(parts) == 2 && parts[0] != "" && parts[1] == "messages" && method == http.MethodGet
}

func validBearer(header, key string) bool {
	scheme, token, ok := strings.Cut(strings.TrimSpace(header), " ")
	if !ok || !strings.EqualFold(scheme, "bearer") {
		return false
	}
	got := []byte(strings.TrimSpace(token))
	want := []byte(key)
	return len(got) == len(want) && subtle.ConstantTimeCompare(got, want) == 1
}
func (s *server) acquire(ctx context.Context) error {
	if s.slots == nil {
		return nil
	}
	select {
	case s.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (s *server) release() {
	if s.slots != nil {
		<-s.slots
	}
}

func (s *server) models(w http.ResponseWriter, r *http.Request) {
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	ids := s.backend.listModels(r.Context())
	data := make([]any, len(ids))
	for i, id := range ids {
		data[i] = map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "codex"}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func (s *server) createResponse(w http.ResponseWriter, r *http.Request) {
	body, err := decodeObject(r)
	if err != nil {
		writeError(w, 400, "Invalid JSON body.", "invalid_request_error", nil, nil)
		return
	}
	prepared := prepareResponse(body, s.cfg.Model)
	previous := stringValue(prepared["previous_response_id"])
	if previous != "" {
		stored := s.responses.get(previous)
		if stored == nil {
			writeError(w, 404, "Unknown previous_response_id: "+previous, "invalid_request_error", strPtr("previous_response_id"), nil)
			return
		}
		prepared["input"] = append(stored.Context, sliceAny(prepared["input"])...)
	}
	downstream := cloneMap(prepared)
	delete(downstream, "previous_response_id")
	if boolValue(body["stream"]) {
		s.streamResponse(w, r, prepared, downstream, previous)
		return
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	response, err := s.backend.collect(r.Context(), downstream)
	if err != nil {
		writeBackendError(w, err)
		return
	}
	response = ensureResponse(response, prepared)
	if previous != "" {
		response["previous_response_id"] = previous
	}
	s.responses.remember(stringValue(response["id"]), sliceAny(prepared["input"]), response)
	writeJSON(w, 200, response)
}

func (s *server) streamResponse(w http.ResponseWriter, r *http.Request, prepared, downstream map[string]any, previous string) {
	setSSE(w)
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, 500, "Streaming unsupported.", "api_error", nil, nil)
		return
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	var outputs []any
	stored := false
	seq := 0
	err := s.backend.stream(r.Context(), downstream, func(event map[string]any) error {
		eventType := stringValue(event["type"])
		if item := mapAny(event["item"]); eventType == "response.output_item.done" && item != nil {
			outputs = append(outputs, item)
		}
		if eventType == "response.completed" || eventType == "response.incomplete" || eventType == "response.failed" {
			if response := mapAny(event["response"]); response != nil {
				if len(sliceAny(response["output"])) == 0 && len(outputs) > 0 {
					response["output"] = outputs
				}
				response = ensureResponse(response, prepared)
				if previous != "" {
					response["previous_response_id"] = previous
				}
				event["response"] = response
				if !stored {
					s.responses.remember(stringValue(response["id"]), sliceAny(prepared["input"]), response)
					stored = true
				}
			}
		}
		if event["sequence_number"] == nil {
			event["sequence_number"] = seq
		}
		seq++
		writeSSE(w, event)
		flusher.Flush()
		return nil
	})
	if err != nil {
		writeSSE(w, map[string]any{"type": "error", "sequence_number": seq, "code": nil, "message": publicStreamError(err), "param": nil})
	}
	io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *server) inputTokens(w http.ResponseWriter, r *http.Request) {
	body, err := decodeObject(r)
	if err != nil {
		writeError(w, 400, "Invalid JSON body.", "invalid_request_error", nil, nil)
		return
	}
	prepared := prepareResponse(body, s.cfg.Model)
	if previous := stringValue(prepared["previous_response_id"]); previous != "" {
		stored := s.responses.get(previous)
		if stored == nil {
			writeError(w, 404, "Unknown previous_response_id: "+previous, "invalid_request_error", strPtr("previous_response_id"), nil)
			return
		}
		prepared["input"] = append(stored.Context, sliceAny(prepared["input"])...)
	}
	writeJSON(w, 200, map[string]any{"object": "response.input_tokens", "input_tokens": estimateTokens(prepared)})
}

func (s *server) responseResource(w http.ResponseWriter, r *http.Request, resource string) {
	parts := strings.Split(resource, "/")
	id := parts[0]
	stored := s.responses.get(id)
	if stored == nil {
		writeError(w, 404, "Unknown response_id: "+id, "invalid_request_error", strPtr("response_id"), nil)
		return
	}
	if len(parts) == 1 {
		switch r.Method {
		case http.MethodGet:
			if r.URL.Query().Get("stream") == "true" {
				streamStoredResponse(w, stored.Response)
			} else {
				writeJSON(w, 200, stored.Response)
			}
		case http.MethodDelete:
			s.responses.delete(id)
			w.WriteHeader(204)
		default:
			methodNotAllowed(w)
		}
		return
	}
	if len(parts) == 2 && parts[1] == "cancel" && r.Method == http.MethodPost {
		status := stringValue(stored.Response["status"])
		if status != "queued" && status != "in_progress" {
			writeError(w, 409, "Only queued or in-progress background responses can be cancelled.", "invalid_request_error", strPtr("response_id"), nil)
			return
		}
		writeJSON(w, 200, s.responses.cancel(id))
		return
	}
	if len(parts) == 2 && parts[1] == "input_items" && r.Method == http.MethodGet {
		s.responseInputItems(w, r, stored)
		return
	}
	http.NotFound(w, r)
}

func streamStoredResponse(w http.ResponseWriter, response map[string]any) {
	setSSE(w)
	created := cloneMap(response)
	created["status"] = "in_progress"
	created["output"] = []any{}
	writeSSE(w, map[string]any{"type": "response.created", "sequence_number": 0, "response": created})
	seq := 1
	for i, item := range sliceAny(response["output"]) {
		writeSSE(w, map[string]any{"type": "response.output_item.done", "sequence_number": seq, "output_index": i, "item": item})
		seq++
	}
	writeSSE(w, map[string]any{"type": "response.completed", "sequence_number": seq, "response": response})
	io.WriteString(w, "data: [DONE]\n\n")
}

func (s *server) responseInputItems(w http.ResponseWriter, r *http.Request, stored *storedResponse) {
	items := make([]map[string]any, 0, len(stored.EffectiveInput))
	for i, raw := range stored.EffectiveInput {
		items = append(items, responseInputPageItem(raw, i))
	}
	items, more := paginateMaps(items, r.URL.Query())
	writeJSON(w, 200, cursorPage(items, more))
}

func responseInputPageItem(raw any, index int) map[string]any {
	item := mapAny(raw)
	if item == nil {
		return map[string]any{
			"id":      fmt.Sprintf("input_%d", index),
			"type":    "message",
			"role":    "user",
			"status":  "completed",
			"content": []any{map[string]any{"type": "input_text", "text": stringValue(raw)}},
		}
	}
	item = cloneMap(item)
	itemType := stringValue(item["type"])
	role := stringValue(item["role"])
	id := inputItemID(item, index)
	switch role {
	case "user", "system", "developer":
		return map[string]any{
			"id":      id,
			"type":    "message",
			"role":    role,
			"status":  valueOr(item["status"], "completed"),
			"content": responseInputContent(item["content"]),
		}
	case "assistant":
		page := map[string]any{
			"id": id, "type": "message", "role": role,
			"status": valueOr(item["status"], "completed"),
			"content": []any{map[string]any{
				"type": "output_text", "text": contentText(item["content"]), "annotations": []any{},
			}},
		}
		if item["phase"] != nil && item["phase"] != "" {
			page["phase"] = item["phase"]
		}
		return page
	}
	if itemType == "function_call" || itemType == "function_call_output" {
		setDefault(item, "id", id)
		setDefault(item, "status", "completed")
	}
	return item
}

func inputItemID(item map[string]any, index int) string {
	if id := stringValue(valueOr(item["id"], item["call_id"])); id != "" {
		return id
	}
	return fmt.Sprintf("input_%d", index)
}

func responseInputContent(value any) []any {
	if value == nil {
		return []any{map[string]any{"type": "input_text", "text": ""}}
	}
	if text, ok := value.(string); ok {
		return []any{map[string]any{"type": "input_text", "text": text}}
	}
	parts, ok := value.([]any)
	if !ok {
		return []any{map[string]any{"type": "input_text", "text": stringValue(value)}}
	}
	result := make([]any, 0, len(parts))
	for _, raw := range parts {
		if part := mapAny(raw); part != nil {
			result = append(result, cloneMap(part))
		}
	}
	return result
}

func (s *server) chatCollection(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		s.listChats(w, r)
		return
	}
	body, err := decodeObject(r)
	if err != nil {
		writeError(w, 400, "Invalid JSON body.", "invalid_request_error", nil, nil)
		return
	}
	responsePayload := chatToResponse(body, s.cfg.Model)
	legacy := body["functions"] != nil || body["function_call"] != nil
	if boolValue(body["stream"]) {
		s.streamChat(w, r, responsePayload, body, legacy)
		return
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	response, err := s.backend.collect(r.Context(), responsePayload)
	if err != nil {
		writeBackendError(w, err)
		return
	}
	response = ensureResponse(response, responsePayload)
	completion := responseToChat(response, stringValue(responsePayload["model"]), legacy, chatChoiceCount(body["n"]))
	if metadata := mapAny(body["metadata"]); metadata != nil {
		completion["metadata"] = cloneMap(metadata)
	}
	if boolValue(body["store"]) {
		s.chats.remember(stringValue(completion["id"]), completion, mapOrEmpty(body["metadata"]))
	}
	writeJSON(w, 200, completion)
}

func (s *server) streamChat(w http.ResponseWriter, r *http.Request, responsePayload, body map[string]any, legacy bool) {
	setSSE(w)
	flusher, ok := w.(http.Flusher)
	if !ok {
		return
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	state := chatStreamState{ID: newID("chatcmpl"), Created: int(time.Now().Unix()), Model: stringValue(responsePayload["model"]), ChoiceCount: chatChoiceCount(body["n"])}
	includeUsage := boolValue(mapAny(body["stream_options"])["include_usage"])
	var outputs []any
	emitted := map[int]string{}
	emit := func(v map[string]any) { writeSSE(w, v); flusher.Flush() }
	role := func() {
		if !state.RoleSent {
			state.RoleSent = true
			emit(chatChunk(state, map[string]any{"role": "assistant"}, nil, nil, nil))
		}
	}
	err := s.backend.stream(r.Context(), responsePayload, func(event map[string]any) error {
		typ := stringValue(event["type"])
		switch typ {
		case "response.created":
			if resp := mapAny(event["response"]); resp != nil {
				state.update(resp)
			}
			role()
		case "response.output_text.delta":
			role()
			if delta := stringValue(event["delta"]); delta != "" {
				state.SawText = true
				emit(chatChunk(state, map[string]any{"content": delta}, nil, nil, nil))
			}
		case "response.output_item.added":
			item := mapAny(event["item"])
			if item != nil && item["type"] == "function_call" {
				role()
				idx := intValue(event["output_index"])
				emitted[idx] = ""
				emit(chatChunk(state, toolDelta(item, idx, "", true, legacy), nil, nil, nil))
			}
		case "response.function_call_arguments.delta":
			role()
			idx := intValue(event["output_index"])
			delta := stringValue(event["delta"])
			emitted[idx] += delta
			var d map[string]any
			if legacy {
				d = map[string]any{"function_call": map[string]any{"arguments": delta}}
			} else {
				d = map[string]any{"tool_calls": []any{map[string]any{"index": idx, "function": map[string]any{"arguments": delta}}}}
			}
			emit(chatChunk(state, d, nil, nil, nil))
		case "response.output_item.done":
			item := mapAny(event["item"])
			if item != nil {
				outputs = append(outputs, item)
				if item["type"] == "function_call" {
					idx := intValue(event["output_index"])
					if _, ok := emitted[idx]; !ok {
						role()
						emitted[idx] = stringValue(item["arguments"])
						emit(chatChunk(state, toolDelta(item, idx, emitted[idx], true, legacy), nil, nil, nil))
					}
				} else if item["type"] == "message" && !state.SawText {
					if text := messageText(item); text != "" {
						role()
						state.SawText = true
						emit(chatChunk(state, map[string]any{"content": text}, nil, nil, nil))
					}
				}
			}
		case "response.completed", "response.incomplete", "response.failed":
			resp := mapAny(event["response"])
			if resp != nil {
				if len(sliceAny(resp["output"])) == 0 && len(outputs) > 0 {
					resp["output"] = outputs
				}
				resp = ensureResponse(resp, responsePayload)
				state.update(resp)
				role()
				completion := responseToChat(resp, state.Model, legacy, state.ChoiceCount)
				if metadata := mapAny(body["metadata"]); metadata != nil {
					completion["metadata"] = cloneMap(metadata)
				}
				if boolValue(body["store"]) {
					s.chats.remember(stringValue(completion["id"]), completion, mapOrEmpty(body["metadata"]))
				}
				finish := stringValue(mapAny(sliceAny(completion["choices"])[0])["finish_reason"])
				emit(chatChunk(state, map[string]any{}, &finish, nil, nil))
				if includeUsage {
					empty := []any{}
					emit(chatChunk(state, map[string]any{}, nil, &empty, mapAny(completion["usage"])))
				}
			}
		}
		return nil
	})
	if err != nil {
		emit(map[string]any{"error": map[string]any{"message": publicStreamError(err), "type": "api_error", "param": nil, "code": nil}})
	}
	io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

type chatStreamState struct {
	ID                string
	Created           int
	Model             string
	ChoiceCount       int
	RoleSent, SawText bool
}

func (s *chatStreamState) update(resp map[string]any) {
	if id := stringValue(resp["id"]); id != "" {
		if strings.HasPrefix(id, "resp_") {
			s.ID = strings.Replace(id, "resp_", "chatcmpl_", 1)
		} else {
			s.ID = "chatcmpl_" + id
		}
	}
	if created := intValue(resp["created_at"]); created != 0 {
		s.Created = created
	}
	if model := stringValue(resp["model"]); model != "" {
		s.Model = model
	}
}
func chatChunk(s chatStreamState, delta map[string]any, finish *string, choices *[]any, usage map[string]any) map[string]any {
	var c []any
	if choices != nil {
		c = *choices
	} else {
		for i := 0; i < s.ChoiceCount; i++ {
			var reason any
			if finish != nil {
				reason = *finish
			}
			c = append(c, map[string]any{"index": i, "delta": cloneMap(delta), "finish_reason": reason, "logprobs": nil})
		}
	}
	result := map[string]any{"id": s.ID, "object": "chat.completion.chunk", "created": s.Created, "model": s.Model, "choices": c}
	if usage != nil {
		result["usage"] = usage
	}
	return result
}
func toolDelta(item map[string]any, index int, args string, identity, legacy bool) map[string]any {
	if legacy {
		fn := map[string]any{"arguments": args}
		if identity {
			fn["name"] = stringValue(item["name"])
		}
		return map[string]any{"function_call": fn}
	}
	fn := map[string]any{"arguments": args}
	call := map[string]any{"index": index, "function": fn}
	if identity {
		fn["name"] = stringValue(item["name"])
		call["id"] = stringValue(valueOr(item["call_id"], valueOr(item["id"], fmt.Sprintf("call_%d", index))))
		call["type"] = "function"
	}
	return map[string]any{"tool_calls": []any{call}}
}

func (s *server) listChats(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	metadata := map[string]any{}
	for key, values := range q {
		if strings.HasPrefix(key, "metadata[") && strings.HasSuffix(key, "]") && len(values) > 0 {
			metadata[strings.TrimSuffix(strings.TrimPrefix(key, "metadata["), "]")] = values[0]
		}
	}
	items, more := s.chats.list(q.Get("model"), metadata, q.Get("order") == "desc", q.Get("after"), positiveInt(q.Get("limit")))
	writeJSON(w, 200, cursorPage(items, more))
}
func (s *server) chatResource(w http.ResponseWriter, r *http.Request, resource string) {
	parts := strings.Split(resource, "/")
	id := parts[0]
	stored := s.chats.get(id)
	if stored == nil {
		writeError(w, 404, "Unknown completion_id: "+id, "invalid_request_error", strPtr("completion_id"), nil)
		return
	}
	if len(parts) == 2 && parts[1] == "messages" && r.Method == http.MethodGet {
		items := stored.Messages
		items, more := paginateMaps(items, r.URL.Query())
		writeJSON(w, 200, cursorPage(items, more))
		return
	}
	if len(parts) > 1 {
		http.NotFound(w, r)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, 200, stored.Completion)
	case http.MethodPost:
		body, err := decodeObject(r)
		if err != nil {
			writeError(w, 400, "Invalid JSON body.", "invalid_request_error", nil, nil)
			return
		}
		metadata := mapAny(body["metadata"])
		if body["metadata"] != nil && metadata == nil {
			writeError(w, 400, "metadata must be an object or null.", "invalid_request_error", strPtr("metadata"), nil)
			return
		}
		writeJSON(w, 200, s.chats.update(id, mapOrEmpty(metadata)))
	case http.MethodDelete:
		s.chats.delete(id)
		writeJSON(w, 200, map[string]any{"id": id, "object": "chat.completion.deleted", "deleted": true})
	default:
		methodNotAllowed(w)
	}
}

func (s *server) audio(w http.ResponseWriter, r *http.Request) {
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	resp, err := s.backend.transcribe(r.Context(), r.Header, r.Body)
	if err != nil {
		writeBackendError(w, err)
		return
	}
	defer resp.Body.Close()
	copyProxyResponse(w, resp)
}

func (s *server) images(w http.ResponseWriter, r *http.Request) {
	body, err := decodeObject(r)
	if err != nil {
		writeError(w, 400, "Invalid JSON body.", "invalid_request_error", nil, nil)
		return
	}
	if err := validateImage(body); err != nil {
		writeError(w, 400, err.Message, "invalid_request_error", strPtr(err.Param), nil)
		return
	}
	count := intValue(body["n"])
	if count == 0 {
		count = 1
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	data := make([]any, 0, count)
	created := int(time.Now().Unix())
	for i := 0; i < count; i++ {
		payload := imageResponsePayload(body, s.cfg.Model)
		resp, err := s.backend.collect(r.Context(), payload)
		if err != nil {
			writeBackendError(w, err)
			return
		}
		if v := intValue(resp["created_at"]); v != 0 {
			created = v
		}
		image := imageFromResponse(resp)
		if image == nil {
			writeError(w, 502, "Codex backend did not return an image_generation_call.", "api_error", nil, nil)
			return
		}
		data = append(data, image)
	}
	result := map[string]any{"created": created, "data": data, "output_format": valueOr(body["output_format"], "png")}
	for _, allowed := range []struct {
		key    string
		values map[string]bool
	}{
		{"background", map[string]bool{"transparent": true, "opaque": true}},
		{"quality", map[string]bool{"low": true, "medium": true, "high": true}},
		{"size", map[string]bool{"1024x1024": true, "1024x1536": true, "1536x1024": true}},
	} {
		if value := stringValue(body[allowed.key]); allowed.values[value] {
			result[allowed.key] = value
		}
	}
	writeJSON(w, 200, result)
}

func (s *server) proxy(w http.ResponseWriter, r *http.Request, path string) {
	if invalidProxyPath(path) {
		writeError(w, 400, "Invalid proxy path.", "api_error", nil, nil)
		return
	}
	if s.acquire(r.Context()) != nil {
		return
	}
	defer s.release()
	resp, err := s.backend.proxy(r.Context(), r.Method, path, r.URL.RawQuery, r.Header, r.Body)
	if err != nil {
		writeBackendError(w, err)
		return
	}
	defer resp.Body.Close()
	copyProxyResponse(w, resp)
}

func decodeObject(r *http.Request) (map[string]any, error) {
	defer r.Body.Close()
	decoder := json.NewDecoder(io.LimitReader(r.Body, 32<<20))
	decoder.UseNumber()
	var body map[string]any
	if err := decoder.Decode(&body); err != nil {
		return nil, err
	}
	if body == nil {
		return nil, errors.New("expected a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("unexpected data after JSON object")
		}
		return nil, err
	}
	return body, nil
}
func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func writeError(w http.ResponseWriter, status int, message, typ string, param *string, code any) {
	var p any
	if param != nil {
		p = *param
	}
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": message, "type": typ, "param": p, "code": code}})
}
func writeBackendError(w http.ResponseWriter, err error) {
	var be *backendError
	if errors.As(err, &be) {
		writeError(w, be.Status, be.Message, "api_error", nil, nil)
	} else {
		writeError(w, 500, "Internal server error.", "api_error", nil, nil)
	}
}
func publicStreamError(err error) string {
	var be *backendError
	if errors.As(err, &be) {
		return be.Message
	}
	return "Internal server error."
}
func setSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
}
func writeSSE(w io.Writer, value any) {
	data, _ := json.Marshal(value)
	fmt.Fprintf(w, "data: %s\n\n", data)
}
func copyProxyResponse(w http.ResponseWriter, resp *http.Response) {
	for name, values := range resp.Header {
		switch strings.ToLower(name) {
		case "connection", "content-encoding", "content-length", "keep-alive", "proxy-authenticate", "proxy-authorization", "set-cookie", "set-cookie2", "te", "trailer", "transfer-encoding", "upgrade":
			continue
		}
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.Header().Set("x-openai-via-codex-proxy", "codex-http")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}
func methodNotAllowed(w http.ResponseWriter) {
	writeError(w, 405, "Method not allowed.", "invalid_request_error", nil, nil)
}
func strPtr(s string) *string { return &s }
func positiveInt(s string) int {
	v, _ := strconv.Atoi(s)
	if v > 0 {
		return v
	}
	return 0
}
func chatChoiceCount(v any) int {
	n := intValue(v)
	if n < 1 {
		return 1
	}
	return n
}
func mapOrEmpty(value any) map[string]any {
	if m := mapAny(value); m != nil {
		return m
	}
	return map[string]any{}
}
func reverseMaps[T any](v []T) {
	for i, j := 0, len(v)-1; i < j; i, j = i+1, j-1 {
		v[i], v[j] = v[j], v[i]
	}
}
func paginateMaps[T interface{ map[string]any }](items []T, q url.Values) ([]T, bool) {
	if q.Get("order") == "desc" {
		reverseMaps(items)
	}
	if after := q.Get("after"); after != "" {
		for i, item := range items {
			if stringValue(item["id"]) == after {
				items = items[i+1:]
				break
			}
		}
	}
	limit := positiveInt(q.Get("limit"))
	more := limit > 0 && len(items) > limit
	if more {
		items = items[:limit]
	}
	return items, more
}
func cursorPage[T interface{ map[string]any }](items []T, more bool) map[string]any {
	var first, last any
	if len(items) > 0 {
		first = items[0]["id"]
		last = items[len(items)-1]["id"]
	}
	return map[string]any{"object": "list", "data": items, "first_id": first, "last_id": last, "has_more": more}
}

func estimateTokens(prepared map[string]any) int {
	data, _ := json.Marshal(prepared["input"])
	words := len(strings.Fields(string(data)))
	images := bytes.Count(data, []byte("input_image"))
	tools := len(sliceAny(prepared["tools"]))
	n := (len(data) + 3) / 4
	if words > n {
		n = words
	}
	n += images*85 + tools*16
	if n < 1 {
		n = 1
	}
	return n
}

type imageValidationError struct{ Message, Param string }

func validateImage(body map[string]any) *imageValidationError {
	prompt, ok := body["prompt"].(string)
	if !ok || strings.TrimSpace(prompt) == "" {
		return &imageValidationError{"prompt is required.", "prompt"}
	}
	if boolValue(body["stream"]) {
		return &imageValidationError{"stream=true is not supported for image generations.", "stream"}
	}
	if body["response_format"] != nil {
		return &imageValidationError{"response_format is not supported for GPT image generations; images are always returned as b64_json.", "response_format"}
	}
	n := 1
	if body["n"] != nil {
		var valid bool
		n, valid = exactInt(body["n"])
		if !valid {
			return &imageValidationError{"n must be an integer between 1 and 10.", "n"}
		}
	}
	if n < 1 || n > 10 {
		return &imageValidationError{"n must be between 1 and 10.", "n"}
	}
	format := stringValue(valueOr(body["output_format"], "png"))
	if !map[string]bool{"png": true, "jpeg": true, "webp": true}[format] {
		return &imageValidationError{"output_format must be one of png, jpeg, or webp.", "output_format"}
	}
	if size := stringValue(body["size"]); size != "" && size != "auto" {
		parts := strings.Split(size, "x")
		if len(parts) != 2 {
			return &imageValidationError{"size must be 'auto' or a WIDTHxHEIGHT pixel string.", "size"}
		}
		w, e1 := strconv.Atoi(parts[0])
		h, e2 := strconv.Atoi(parts[1])
		if e1 != nil || e2 != nil || w < 1 || h < 1 {
			return &imageValidationError{"size must be 'auto' or a WIDTHxHEIGHT pixel string.", "size"}
		}
	}
	for _, allowed := range []struct {
		key, message string
		values       map[string]bool
	}{
		{"quality", "quality must be one of low, medium, high, or auto.", map[string]bool{"low": true, "medium": true, "high": true, "auto": true}},
		{"background", "background must be one of transparent, opaque, or auto.", map[string]bool{"transparent": true, "opaque": true, "auto": true}},
		{"moderation", "moderation must be one of low or auto.", map[string]bool{"low": true, "auto": true}},
	} {
		if body[allowed.key] != nil && !allowed.values[stringValue(body[allowed.key])] {
			return &imageValidationError{allowed.message, allowed.key}
		}
	}
	if body["output_compression"] != nil {
		compression, valid := exactInt(body["output_compression"])
		if !valid {
			return &imageValidationError{"output_compression must be an integer between 0 and 100.", "output_compression"}
		}
		if compression < 0 || compression > 100 {
			return &imageValidationError{"output_compression must be between 0 and 100.", "output_compression"}
		}
	}
	if body["style"] != nil {
		return &imageValidationError{"style is not supported for GPT image generations.", "style"}
	}
	if body["partial_images"] != nil {
		return &imageValidationError{"partial_images requires streamed image generations, which are not supported.", "partial_images"}
	}
	return nil
}

func exactInt(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case float64:
		converted := int(typed)
		return converted, float64(converted) == typed
	case json.Number:
		converted, err := strconv.Atoi(string(typed))
		return converted, err == nil
	case string:
		converted, err := strconv.Atoi(typed)
		return converted, err == nil
	default:
		return 0, false
	}
}
func imageResponsePayload(body map[string]any, model string) map[string]any {
	format := stringValue(body["output_format"])
	if format == "" {
		format = "png"
	}
	tool := map[string]any{"type": "image_generation", "action": "generate", "output_format": format}
	for _, k := range []string{"size", "quality", "background", "moderation"} {
		if body[k] != nil {
			tool[k] = body[k]
		}
	}
	if body["output_compression"] != nil {
		compression, _ := exactInt(body["output_compression"])
		tool["output_compression"] = compression
	}
	return map[string]any{
		"model":        model,
		"instructions": "Use the image_generation tool to generate the requested image.",
		"input": []any{map[string]any{
			"role": "user",
			"content": []any{map[string]any{
				"type": "input_text", "text": strings.TrimSpace(stringValue(body["prompt"])),
			}},
		}},
		"store": false, "tools": []any{tool}, "tool_choice": "auto", "parallel_tool_calls": true,
	}
}
func imageFromResponse(response map[string]any) map[string]any {
	for _, raw := range sliceAny(response["output"]) {
		item := mapAny(raw)
		if item["type"] == "image_generation_call" && stringValue(item["result"]) != "" {
			result := map[string]any{"b64_json": item["result"]}
			if item["revised_prompt"] != nil {
				result["revised_prompt"] = item["revised_prompt"]
			}
			return result
		}
	}
	return nil
}
