package app

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
)

var defaultModels = []string{"gpt-5.1", "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.2", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

const maxSSEEventBytes = 128 << 20
const replayBodyMemoryLimit = 1 << 20

type backend struct {
	cfg    config
	client *http.Client
	auth   *authProvider
}
type backendError struct {
	Status  int
	Message string
}

func (e *backendError) Error() string { return e.Message }

func newBackend(cfg config) *backend {
	client := &http.Client{Timeout: cfg.Timeout}
	return &backend{
		cfg:    cfg,
		client: client,
		auth: &authProvider{
			path:          cfg.AuthJSON,
			refreshClient: &http.Client{Timeout: authRefreshTimeout},
		},
	}
}

func (b *backend) headers(cred credentials, stream bool, requestID string) http.Header {
	return b.headersWithTier(cred, stream, requestID, "")
}

// headersWithTier builds the upstream request headers. serviceTier travels as
// X-Service-Tier rather than only in the body: the payload field alone did not
// take effect against the Codex backend, so the header carries it as well.
func (b *backend) headersWithTier(cred credentials, stream bool, requestID, serviceTier string) http.Header {
	h := make(http.Header)
	h.Set("Authorization", "Bearer "+cred.AccessToken)
	originator := b.cfg.Originator
	if originator == "" {
		originator = defaultOriginator
	}
	h.Set("originator", originator)
	userAgent := b.cfg.UserAgent
	if userAgent == "" {
		userAgent = fmt.Sprintf("openai-api-server-via-codex/%s (%s; %s)", b.cfg.ClientVersion, runtime.GOOS, runtime.GOARCH)
	}
	h.Set("User-Agent", userAgent)
	if serviceTier != "" {
		h.Set("X-Service-Tier", serviceTier)
	}
	if cred.AccountID != "" {
		h.Set("ChatGPT-Account-ID", cred.AccountID)
	}
	if stream {
		h.Set("Accept", "text/event-stream")
		h.Set("Content-Type", "application/json")
		h.Set("OpenAI-Beta", "responses=experimental")
	}
	if requestID != "" {
		h.Set("session_id", requestID)
		h.Set("x-client-request-id", requestID)
	}
	return h
}

func (b *backend) doAuthenticated(makeRequest func(credentials) (*http.Request, error)) (*http.Response, error) {
	cred, err := b.auth.borrow()
	if err != nil {
		logAuthFailure("request", err)
		return nil, &backendError{401, err.Error()}
	}
	for attempt := 0; attempt < 2; attempt++ {
		req, err := makeRequest(cred)
		if err != nil {
			return nil, err
		}
		resp, err := b.client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusUnauthorized {
			return resp, nil
		}
		if attempt == 1 {
			log.Printf("codex.auth.unauthorized code=upstream_unauthorized stage=upstream attempt=2 action=return_401")
			return resp, nil
		}
		log.Printf("codex.auth.unauthorized code=upstream_unauthorized stage=upstream attempt=1 action=reload_and_retry")
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		_ = resp.Body.Close()
		previousAccessToken := cred.AccessToken
		cred, err = b.auth.reload()
		if err != nil {
			logAuthFailure("reload_after_401", err)
			return nil, &backendError{401, err.Error()}
		}
		b.debugf("codex.auth.reloaded_after_unauthorized credentials_changed=%t", previousAccessToken != cred.AccessToken)
	}
	panic("unreachable")
}

func logAuthFailure(stage string, err error) {
	log.Printf(
		"codex.auth.error stage=%s code=%s message=%q",
		stage,
		authFailureCode(err),
		redactSensitive(err.Error()),
	)
}

func (b *backend) stream(ctx context.Context, payload map[string]any, fn func(map[string]any) error) error {
	b.debugf("codex.stream.start model=%s endpoint=%s/responses", stringValue(payload["model"]), b.cfg.BackendURL)
	prepared := cloneMap(payload)
	for _, name := range b.cfg.DropParams {
		delete(prepared, name)
	}
	delete(prepared, "max_output_tokens")
	prepared["stream"], prepared["store"] = true, false
	setDefault(prepared, "tool_choice", "auto")
	setDefault(prepared, "parallel_tool_calls", true)
	text, _ := prepared["text"].(map[string]any)
	if text == nil {
		text = map[string]any{}
	}
	setDefault(text, "verbosity", "low")
	prepared["text"] = text
	include := sliceAny(prepared["include"])
	found := false
	for _, v := range include {
		if v == "reasoning.encrypted_content" {
			found = true
		}
	}
	if !found {
		include = append(include, "reasoning.encrypted_content")
	}
	prepared["include"] = include
	body, _ := json.Marshal(prepared)
	resp, err := b.doAuthenticated(func(cred credentials) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.BackendURL+"/responses", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		// Read the tier from prepared, not payload, so --drop-params service_tier
		// removes it from the header too rather than only from the body.
		req.Header = b.headersWithTier(cred, true, stringValue(prepared["prompt_cache_key"]), stringValue(prepared["service_tier"]))
		return req, nil
	})
	if err != nil {
		var backendErr *backendError
		if errors.As(err, &backendErr) {
			b.debugf("codex.stream.auth_error message=%s", redactSensitive(err.Error()))
			return err
		}
		return &backendError{502, "Codex backend request failed."}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return decodeBackendError(resp)
	}
	reader := bufio.NewReaderSize(resp.Body, 64*1024)
	var data strings.Builder
	events := 0
	flush := func() error {
		if data.Len() == 0 {
			return nil
		}
		raw := data.String()
		data.Reset()
		if raw == "[DONE]" {
			return nil
		}
		var event map[string]any
		if json.Unmarshal([]byte(raw), &event) != nil {
			return nil
		}
		normalizeBackendEvent(event)
		events++
		return fn(event)
	}
	for {
		line, readErr := readSSELine(reader, maxSSEEventBytes)
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			b.debugf("codex.stream.error message=%s", redactSensitive(readErr.Error()))
			return readErr
		}
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			part := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data.Len()+len(part)+1 > maxSSEEventBytes {
				return fmt.Errorf("Codex SSE event exceeds %d bytes", maxSSEEventBytes)
			}
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(part)
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
	}
	if err := flush(); err != nil {
		return err
	}
	b.debugf("codex.stream.end events=%d", events)
	return nil
}

func readSSELine(reader *bufio.Reader, limit int) (string, error) {
	var line []byte
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(line)+len(fragment) > limit {
			return "", fmt.Errorf("Codex SSE line exceeds %d bytes", limit)
		}
		line = append(line, fragment...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}
		line = bytes.TrimSuffix(line, []byte{'\n'})
		line = bytes.TrimSuffix(line, []byte{'\r'})
		return string(line), err
	}
}

func normalizeBackendEvent(event map[string]any) {
	if event["type"] == "response.done" {
		event["type"] = "response.completed"
	}
	response := mapAny(event["response"])
	if response == nil {
		return
	}
	status := stringValue(response["status"])
	if status != "" && !validResponseStatus(status) {
		delete(response, "status")
	}
}

func validResponseStatus(status string) bool {
	switch status {
	case "completed", "failed", "in_progress", "cancelled", "queued", "incomplete":
		return true
	default:
		return false
	}
}

func (b *backend) collect(ctx context.Context, payload map[string]any) (map[string]any, error) {
	var completed map[string]any
	var output []any
	var text strings.Builder
	var responseID string
	err := b.stream(ctx, payload, func(event map[string]any) error {
		switch event["type"] {
		case "response.created":
			if r := mapAny(event["response"]); r != nil {
				responseID = stringValue(r["id"])
			}
		case "response.output_text.delta":
			text.WriteString(stringValue(event["delta"]))
		case "response.output_item.done":
			if item := mapAny(event["item"]); item != nil {
				output = append(output, item)
			}
		case "response.completed", "response.incomplete":
			completed = cloneMap(mapAny(event["response"]))
		case "response.failed":
			completed = cloneMap(mapAny(event["response"]))
			if completed == nil {
				return &backendError{502, "Codex backend response failed."}
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if completed != nil {
		if len(sliceAny(completed["output"])) == 0 && len(output) > 0 {
			completed["output"] = output
		}
		return completed, nil
	}
	if responseID == "" {
		responseID = newID("resp")
	}
	return map[string]any{"id": responseID, "object": "response", "created_at": nowFloat(), "status": "completed", "model": payload["model"], "output": []any{outputMessage(text.String())}, "parallel_tool_calls": true, "tool_choice": valueOr(payload["tool_choice"], "auto"), "tools": sliceAny(payload["tools"])}, nil
}

func (b *backend) listModels(ctx context.Context) []string {
	resp, err := b.doAuthenticated(func(cred credentials) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.cfg.BackendURL+"/models?client_version="+url.QueryEscape(b.cfg.ClientVersion), nil)
		if err != nil {
			return nil, err
		}
		req.Header = b.headers(cred, false, "")
		return req, nil
	})
	if err != nil {
		b.debugf("codex.models.fallback reason=request_or_auth_error")
		return append([]string(nil), defaultModels...)
	}
	defer resp.Body.Close()
	var doc map[string]any
	if resp.StatusCode/100 != 2 || json.NewDecoder(resp.Body).Decode(&doc) != nil {
		b.debugf("codex.models.fallback reason=invalid_response status=%d", resp.StatusCode)
		return append([]string(nil), defaultModels...)
	}
	var result []string
	for _, raw := range sliceAny(doc["models"]) {
		model := mapAny(raw)
		if boolValue(model["supported_in_api"]) && model["visibility"] == "list" && stringValue(model["slug"]) != "" {
			result = append(result, stringValue(model["slug"]))
		}
	}
	if len(result) == 0 {
		b.debugf("codex.models.fallback reason=empty_model_list")
		return append([]string(nil), defaultModels...)
	}
	b.debugf("codex.models.loaded count=%d", len(result))
	return result
}

func (b *backend) proxy(ctx context.Context, method, path, query string, headers http.Header, body io.Reader) (*http.Response, error) {
	return b.proxyTo(ctx, method, b.cfg.BackendURL, path, query, headers, body)
}

func (b *backend) proxyTo(ctx context.Context, method, baseURL, path, query string, headers http.Header, body io.Reader) (*http.Response, error) {
	target, err := resolveProxyURL(baseURL, path, query)
	if err != nil {
		return nil, &backendError{400, "Invalid proxy path."}
	}
	b.debugf("codex.proxy.start method=%s path=%s", method, redactSensitive(target.EscapedPath()))
	replay, err := newReplayBody(body)
	if err != nil {
		return nil, &backendError{400, "Read proxy request body failed."}
	}
	defer replay.Close()
	resp, err := b.doAuthenticated(func(cred credentials) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, method, target.String(), replay.Reader())
		if err != nil {
			return nil, err
		}
		req.Header = b.headers(cred, false, "")
		for _, name := range []string{"Accept", "Content-Type", "Idempotency-Key", "OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project", "OpenAI-Version"} {
			if v := headers.Get(name); v != "" {
				req.Header.Set(name, v)
			}
		}
		return req, nil
	})
	if err != nil {
		var backendErr *backendError
		if errors.As(err, &backendErr) {
			return nil, err
		}
		return nil, &backendError{502, "Codex backend proxy request failed."}
	}
	b.debugf("codex.proxy.headers status=%d", resp.StatusCode)
	return resp, nil
}

type replayBody struct {
	data []byte
	file *os.File
	size int64
}

func newReplayBody(body io.Reader) (*replayBody, error) {
	if body == nil {
		return &replayBody{}, nil
	}
	var memory bytes.Buffer
	if _, err := io.Copy(&memory, io.LimitReader(body, replayBodyMemoryLimit+1)); err != nil {
		return nil, err
	}
	if memory.Len() <= replayBodyMemoryLimit {
		return &replayBody{data: memory.Bytes()}, nil
	}
	file, err := os.CreateTemp("", "openai-via-codex-request-*")
	if err != nil {
		return nil, err
	}
	cleanup := func() {
		_ = file.Close()
		_ = os.Remove(file.Name())
	}
	if _, err := file.Write(memory.Bytes()); err != nil {
		cleanup()
		return nil, err
	}
	written, err := io.Copy(file, body)
	if err != nil {
		cleanup()
		return nil, err
	}
	return &replayBody{file: file, size: int64(memory.Len()) + written}, nil
}

func (b *replayBody) Reader() io.Reader {
	if b.file == nil {
		return bytes.NewReader(b.data)
	}
	return io.NewSectionReader(b.file, 0, b.size)
}

func (b *replayBody) Close() {
	if b.file != nil {
		name := b.file.Name()
		_ = b.file.Close()
		_ = os.Remove(name)
	}
}

func (b *backend) debugf(format string, args ...any) {
	if b.cfg.Verbose {
		log.Printf(format, args...)
	}
}

func (b *backend) transcribe(ctx context.Context, headers http.Header, body io.Reader) (*http.Response, error) {
	base := strings.TrimSuffix(b.cfg.BackendURL, "/codex")
	return b.proxyTo(ctx, http.MethodPost, base, "transcribe", "", headers, body)
}

func decodeBackendError(resp *http.Response) error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var doc map[string]any
	message := resp.Status
	if json.Unmarshal(data, &doc) == nil {
		if e := mapAny(doc["error"]); e != nil && stringValue(e["message"]) != "" {
			message = stringValue(e["message"])
		}
	}
	return &backendError{resp.StatusCode, redactSensitive(message)}
}
func invalidProxyPath(path string) bool {
	_, err := cleanProxyPath(path)
	return err != nil
}

func cleanProxyPath(path string) (string, error) {
	if strings.Contains(path, "\\") {
		return "", errors.New("invalid proxy path")
	}
	for _, r := range path {
		if r == 0 || r < 0x20 || r == 0x7f {
			return "", errors.New("invalid proxy path")
		}
	}
	var segments []string
	for _, p := range strings.Split(path, "/") {
		if p == ".." {
			return "", errors.New("invalid proxy path")
		}
		if p != "" && p != "." {
			segments = append(segments, p)
		}
	}
	return strings.Join(segments, "/"), nil
}

func resolveProxyURL(baseURL, path, query string) (*url.URL, error) {
	cleaned, err := cleanProxyPath(path)
	if err != nil {
		return nil, err
	}
	target, err := url.Parse(baseURL)
	if err != nil || (target.Scheme != "http" && target.Scheme != "https") || target.Host == "" || target.RawQuery != "" || target.Fragment != "" {
		return nil, errors.New("invalid backend base URL")
	}
	target.Path = strings.TrimRight(target.Path, "/") + "/" + cleaned
	target.RawPath = ""
	target.RawQuery = query
	target.Fragment = ""
	return target, nil
}
