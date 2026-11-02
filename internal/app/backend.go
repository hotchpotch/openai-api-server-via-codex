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
	"math"
	"math/rand/v2"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

var defaultModels = []string{"gpt-5.1", "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.2", "gpt-5.2-codex", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

const maxSSEEventBytes = 128 << 20
const replayBodyMemoryLimit = 1 << 20

type backend struct {
	cfg          config
	streamClient *http.Client
	client       *http.Client
	auth         *authProvider
}
type backendError struct {
	Status  int
	Message string
}

func (e *backendError) Error() string { return e.Message }

const (
	retryBaseDelay = 200 * time.Millisecond
	// A longer server-advised wait is left to the client through Retry-After.
	maxRetryAdvice = 30 * time.Second
)

var (
	errSSELimit    = errors.New("Codex SSE size limit exceeded")
	errStreamIdle  = errors.New("Codex stream idle timeout")
	errRetryStream = errors.New("retry Codex stream")
)

// responseFailureError is a failure Codex reported through a response status or
// an event in a 2xx stream.
type responseFailureError struct {
	backendError
	Type, Code, RetryAfter string
	Param                  any
}

func (e *responseFailureError) Unwrap() error { return &e.backendError }

// responseFailure builds the client error for a failure detail from Codex. Text
// fields are redacted. The code is also logged because a stream has already
// reported HTTP 200.
func responseFailure(detail map[string]any) error {
	code := redactSensitive(stringValue(detail["code"]))
	upstreamType := redactSensitive(stringValue(detail["type"]))
	if code == "" && responseFailureStatus(upstreamType) == http.StatusTooManyRequests {
		// Usage errors name themselves in type rather than code.
		code = upstreamType
	}
	message := failureFallbackMessages[code]
	if message == "" {
		message = "Codex backend response failed."
	}
	if text := strings.TrimSpace(stringValue(detail["message"])); text != "" {
		message = redactSensitive(text)
	}
	status := responseFailureStatus(code)
	if explicit := intValue(detail["status"]); explicit >= 400 && explicit < 600 {
		status = explicit
	}
	errorType := upstreamType
	if errorType == "" {
		switch status {
		case http.StatusBadRequest, http.StatusNotFound:
			errorType = "invalid_request_error"
		case http.StatusUnauthorized:
			errorType = "authentication_error"
		case http.StatusTooManyRequests:
			errorType = "rate_limit_error"
		default:
			errorType = "api_error"
		}
	}
	var param any
	if text := stringValue(detail["param"]); text != "" {
		param = redactSensitive(text)
	}
	log.Printf("codex.response.failed code=%s status=%d", code, status)
	return &responseFailureError{backendError{status, message}, errorType, code, retryAfterAdvice(detail), param}
}

var failureFallbackMessages = map[string]string{
	"invalid_prompt":                "Invalid request.",
	"cyber_policy":                  "This request has been flagged for possible cybersecurity risk.",
	"bio_policy":                    "This content was flagged for possible biological risk.",
	"misalignment_policy_violation": "This request was blocked due to a misalignment policy violation.",
}

func responseFailureStatus(code string) int {
	switch code {
	case "invalid_api_key":
		return http.StatusUnauthorized
	case "model_not_found":
		return http.StatusNotFound
	case "rate_limit_exceeded", "insufficient_quota", "usage_limit_reached", "usage_not_included",
		"credit_balance_exhausted", "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
		"organization_usage_limit_exceeded", "flex_unavailable":
		return http.StatusTooManyRequests
	case "server_is_overloaded", "slow_down", "policy_capacity_unavailable":
		return http.StatusServiceUnavailable
	case "context_length_exceeded", "unsupported_parameter", "content_filter",
		"cyber_policy", "bio_policy", "misalignment_policy_violation",
		"audio_too_large", "audio_too_short", "missing_audio_sample_rate":
		return http.StatusBadRequest
	}
	if strings.HasPrefix(code, "invalid_") {
		return http.StatusBadRequest
	}
	return http.StatusBadGateway
}

// retryAfterAdvice returns a valid Retry-After value from a failure's headers.
func retryAfterAdvice(detail map[string]any) string {
	for key, value := range mapAny(detail["headers"]) {
		if strings.EqualFold(key, "retry-after") {
			advice := strings.TrimSpace(stringValue(value))
			if _, ok := parseRetryAfter(advice); ok {
				return advice
			}
		}
	}
	return ""
}

func parseRetryAfter(value string) (time.Duration, bool) {
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), seconds >= 0 && seconds <= math.MaxInt32
	}
	if at, err := http.ParseTime(value); err == nil {
		return max(time.Until(at), 0), true
	}
	return 0, false
}

// failureDetail reads an upstream error value, which is an object or a message.
func failureDetail(value any) map[string]any {
	if detail := mapAny(value); detail != nil {
		return detail
	}
	if text := stringValue(value); text != "" {
		return map[string]any{"message": text}
	}
	return nil
}

// errorEventDetail reads an error event, which holds its fields either directly
// or under "error", and may give an HTTP status beside them.
func errorEventDetail(event map[string]any) map[string]any {
	detail := cloneMap(failureDetail(event["error"]))
	if detail == nil {
		detail = cloneMap(event)
		delete(detail, "type")
	}
	if detail["status"] == nil {
		detail["status"] = event["status"]
	}
	return detail
}

// failedResponseError returns the client error for a response that failed.
func failedResponseError(response map[string]any) error {
	if stringValue(response["status"]) != "failed" {
		return nil
	}
	return responseFailure(failureDetail(response["error"]))
}

func newBackend(cfg config) *backend {
	client := &http.Client{Timeout: cfg.Timeout}
	return &backend{
		cfg:    cfg,
		client: client,
		// stream applies cfg.Timeout between events, so a long response can finish.
		streamClient: &http.Client{},
		auth: &authProvider{
			path:          cfg.AuthJSON,
			refreshClient: &http.Client{Timeout: authRefreshTimeout},
		},
	}
}

func (b *backend) headers(cred credentials, stream bool, requestID string) http.Header {
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
	if requestID != "" {
		h.Set("session_id", requestID)
		h.Set("x-client-request-id", requestID)
	}
	return h
}

func (b *backend) doAuthenticated(makeRequest func(credentials) (*http.Request, error)) (*http.Response, error) {
	return b.doAuthenticatedWith(b.client, makeRequest)
}

func (b *backend) doAuthenticatedWith(client *http.Client, makeRequest func(credentials) (*http.Request, error)) (*http.Response, error) {
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
		resp, err := client.Do(req)
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

func prepareCodexPayload(payload map[string]any, dropParams []string) map[string]any {
	return normalizeCodexPayload(cloneMap(payload), dropParams)
}

func normalizeCodexPayload(prepared map[string]any, dropParams []string) map[string]any {
	for _, name := range dropParams {
		delete(prepared, name)
	}
	delete(prepared, "max_output_tokens")
	for _, item := range sliceAny(prepared["input"]) {
		for _, part := range sliceAny(mapAny(item)["content"]) {
			delete(mapAny(part), "prompt_cache_breakpoint")
		}
	}
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
	return prepared
}

// stream delivers Codex events to fn, retrying transient failures while no output
// has reached fn. Lifecycle events wait until the attempt produces output, so a
// client never sees events from an abandoned attempt.
func (b *backend) stream(ctx context.Context, payload map[string]any, fn func(map[string]any) error) error {
	for attempt := 0; ; attempt++ {
		var held []map[string]any
		var retryable error
		committed := false
		commit := func() error {
			committed = true
			for _, event := range held {
				if err := fn(event); err != nil {
					return err
				}
			}
			return nil
		}
		err := b.streamOnce(ctx, payload, func(event map[string]any) error {
			if committed {
				return fn(event)
			}
			switch stringValue(event["type"]) {
			case "response.created", "response.in_progress", "response.metadata", "codex.response.metadata":
				held = append(held, event)
				return nil
			case "error", "response.failed":
				failure := eventFailure(event)
				if _, ok := b.retryDelay(failure, attempt); ok {
					retryable = failure
					return errRetryStream
				}
			}
			if err := commit(); err != nil {
				return err
			}
			return fn(event)
		})
		if committed {
			return err
		}
		failure := err
		if errors.Is(err, errRetryStream) {
			failure = retryable
		}
		delay, ok := b.retryDelay(failure, attempt)
		if !ok || ctx.Err() != nil {
			if commitErr := commit(); commitErr != nil {
				return commitErr
			}
			return err
		}
		log.Printf("codex.stream.retry attempt=%d max_retries=%d delay_ms=%d", attempt+1, b.cfg.MaxRetries, delay.Milliseconds())
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}
}

func eventFailure(event map[string]any) error {
	if stringValue(event["type"]) == "error" {
		return responseFailure(errorEventDetail(event))
	}
	return responseFailure(failureDetail(mapAny(event["response"])["error"]))
}

// retryDelay returns the wait before retrying after err, if err is transient and
// retries remain. Only a backend-reported failure can be transient.
func (b *backend) retryDelay(err error, attempt int) (time.Duration, bool) {
	var failure *backendError
	if attempt >= b.cfg.MaxRetries || !errors.As(err, &failure) {
		return 0, false
	}
	var code, advice string
	var reported *responseFailureError
	if errors.As(err, &reported) {
		code, advice = reported.Code, reported.RetryAfter
	}
	var retry bool
	switch {
	case failure.Status == http.StatusTooManyRequests:
		retry = code == "rate_limit_exceeded" || (code == "" && advice != "")
	case code == "server_is_overloaded" || code == "policy_capacity_unavailable":
		retry = advice != ""
	default:
		retry = failure.Status >= 500
	}
	if !retry {
		return 0, false
	}
	if advice != "" {
		delay, _ := parseRetryAfter(advice)
		return delay, delay <= maxRetryAdvice
	}
	jitter := 0.9 + rand.Float64()*0.2
	return time.Duration(float64(b.cfg.retryBaseDelay<<attempt) * jitter), true
}

func (b *backend) streamOnce(ctx context.Context, payload map[string]any, fn func(map[string]any) error) error {
	b.debugf("codex.stream.start model=%s endpoint=%s/responses", stringValue(payload["model"]), b.cfg.BackendURL)
	prepared := prepareCodexPayload(payload, b.cfg.DropParams)
	body, _ := json.Marshal(prepared)
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	idle := time.AfterFunc(b.cfg.Timeout, func() { cancel(errStreamIdle) })
	defer idle.Stop()
	idleError := func() error {
		if errors.Is(context.Cause(ctx), errStreamIdle) {
			return &backendError{http.StatusGatewayTimeout, "Codex backend stream timed out."}
		}
		return nil
	}
	resp, err := b.doAuthenticatedWith(b.streamClient, func(cred credentials) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.cfg.BackendURL+"/responses", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header = b.headers(cred, true, stringValue(prepared["prompt_cache_key"]))
		return req, nil
	})
	if err != nil {
		var backendErr *backendError
		if errors.As(err, &backendErr) {
			b.debugf("codex.stream.auth_error message=%s", redactSensitive(err.Error()))
			return err
		}
		if err := idleError(); err != nil {
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
	terminal := false
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
		switch event["type"] {
		case "error", "response.failed":
			terminal = true
		case "response.completed", "response.incomplete":
			terminal = terminal || mapAny(event["response"]) != nil
		}
		// A slow client must not count as an idle backend.
		idle.Stop()
		defer idle.Reset(b.cfg.Timeout)
		return fn(event)
	}
	for {
		line, readErr := readSSELine(reader, maxSSEEventBytes)
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			b.debugf("codex.stream.error message=%s", redactSensitive(readErr.Error()))
			if err := idleError(); err != nil {
				return err
			}
			if errors.Is(readErr, errSSELimit) {
				return readErr
			}
			return &backendError{502, "Codex backend stream failed."}
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
				return fmt.Errorf("%w: event exceeds %d bytes", errSSELimit, maxSSEEventBytes)
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
	if !terminal {
		return &backendError{502, "Codex stream closed before the response completed."}
	}
	return nil
}

func readSSELine(reader *bufio.Reader, limit int) (string, error) {
	var line []byte
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(line)+len(fragment) > limit {
			return "", fmt.Errorf("%w: line exceeds %d bytes", errSSELimit, limit)
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
	var failure map[string]any
	err := b.stream(ctx, payload, func(event map[string]any) error {
		switch event["type"] {
		case "response.output_item.done":
			if item := mapAny(event["item"]); item != nil {
				output = append(output, item)
			}
		case "error":
			failure = errorEventDetail(event)
		case "response.completed", "response.incomplete":
			completed = cloneMap(mapAny(event["response"]))
		case "response.failed":
			completed = cloneMap(mapAny(event["response"]))
			if completed == nil {
				return responseFailure(failure)
			}
			completed["status"] = "failed"
			if completed["error"] == nil && failure != nil {
				completed["error"] = failure
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
	return nil, responseFailure(failure)
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
	detail := map[string]any{}
	if json.Unmarshal(data, &doc) == nil {
		if upstream := failureDetail(doc["error"]); upstream != nil {
			detail = cloneMap(upstream)
		} else if text := stringValue(doc["detail"]); text != "" {
			detail["message"] = text
		}
	}
	if stringValue(detail["message"]) == "" {
		detail["message"] = resp.Status
	}
	detail["status"] = resp.StatusCode
	detail["headers"] = map[string]any{"retry-after": resp.Header.Get("Retry-After")}
	return responseFailure(detail)
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
