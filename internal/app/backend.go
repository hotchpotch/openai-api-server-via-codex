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
	"runtime"
	"strings"
)

var defaultModels = []string{"gpt-5.1", "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.2", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

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
	return &backend{cfg: cfg, client: client, auth: &authProvider{path: cfg.AuthJSON, client: client}}
}

func (b *backend) headers(cred credentials, stream bool) http.Header {
	h := make(http.Header)
	h.Set("Authorization", "Bearer "+cred.AccessToken)
	h.Set("originator", "openai-api-server-via-codex")
	h.Set("User-Agent", fmt.Sprintf("openai-api-server-via-codex/%s (%s; %s)", b.cfg.ClientVersion, runtime.GOOS, runtime.GOARCH))
	if cred.AccountID != "" {
		h.Set("ChatGPT-Account-ID", cred.AccountID)
	}
	if stream {
		h.Set("Accept", "text/event-stream")
		h.Set("Content-Type", "application/json")
		h.Set("OpenAI-Beta", "responses=experimental")
	}
	return h
}

func (b *backend) stream(ctx context.Context, payload map[string]any, fn func(map[string]any) error) error {
	b.debugf("codex.stream.start model=%s endpoint=%s/responses", stringValue(payload["model"]), b.cfg.BackendURL)
	cred, err := b.auth.borrow()
	if err != nil {
		b.debugf("codex.stream.auth_error message=%s", redactSensitive(err.Error()))
		return &backendError{401, err.Error()}
	}
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.BackendURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header = b.headers(cred, true)
	resp, err := b.client.Do(req)
	if err != nil {
		return &backendError{502, "Codex backend request failed."}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return decodeBackendError(resp)
	}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
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
		if event["type"] == "response.done" {
			event["type"] = "response.completed"
		}
		events++
		return fn(event)
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := flush(); err != nil {
		return err
	}
	if err := scanner.Err(); err != nil {
		b.debugf("codex.stream.error message=%s", redactSensitive(err.Error()))
		return err
	}
	b.debugf("codex.stream.end events=%d", events)
	return nil
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
	cred, err := b.auth.borrow()
	if err != nil {
		b.debugf("codex.models.fallback reason=auth_error")
		return append([]string(nil), defaultModels...)
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, b.cfg.BackendURL+"/models?client_version="+url.QueryEscape(b.cfg.ClientVersion), nil)
	req.Header = b.headers(cred, false)
	resp, err := b.client.Do(req)
	if err != nil {
		b.debugf("codex.models.fallback reason=request_error")
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
	cred, err := b.auth.borrow()
	if err != nil {
		return nil, &backendError{401, err.Error()}
	}
	req, err := http.NewRequestWithContext(ctx, method, target.String(), body)
	if err != nil {
		return nil, err
	}
	req.Header = b.headers(cred, false)
	for _, name := range []string{"Accept", "Content-Type", "Idempotency-Key", "OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project", "OpenAI-Version"} {
		if v := headers.Get(name); v != "" {
			req.Header.Set(name, v)
		}
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return nil, &backendError{502, "Codex backend proxy request failed."}
	}
	b.debugf("codex.proxy.headers status=%d", resp.StatusCode)
	return resp, nil
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
	decoded, err := url.PathUnescape(path)
	if err != nil || strings.Contains(decoded, "\\") {
		return "", errors.New("invalid proxy path")
	}
	for _, r := range decoded {
		if r == 0 || r < 0x20 || r == 0x7f {
			return "", errors.New("invalid proxy path")
		}
	}
	var segments []string
	for _, p := range strings.Split(decoded, "/") {
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
