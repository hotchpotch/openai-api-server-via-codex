package app

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
)

// WebSocket conversations belong to the upstream connection, not the HTTP
// compatibility stores. Keep events (including unknown fields) intact.
type websocketBackend interface {
	dialWebSocket(context.Context, http.Header, string) (*websocket.Conn, string, error)
}

func (b *backend) dialWebSocket(ctx context.Context, incoming http.Header, query string) (*websocket.Conn, string, error) {
	target, err := resolveProxyURL(b.cfg.BackendURL, "responses", query)
	if err != nil {
		return nil, "", &backendError{400, "Invalid backend URL."}
	}
	cred, err := b.auth.borrow()
	if err != nil {
		logAuthFailure("websocket", err)
		return nil, "", &backendError{401, err.Error()}
	}
	// A client-wide timeout would also terminate the upgraded connection. Only
	// bound the handshake; established connections use bounded writes below.
	client := *b.client
	client.Timeout = 0
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	for attempt := 0; attempt < 2; attempt++ {
		headers := b.headers(cred, true, incoming.Get("x-client-request-id"))
		headers.Del("Accept")
		headers.Del("Content-Type")
		for _, key := range []string{"OpenAI-Beta", "session_id", "x-client-request-id", "x-codex-turn-state", "x-codex-turn-metadata"} {
			if value := incoming.Get(key); value != "" {
				headers.Set(key, value)
			}
		}
		dialCtx, cancel := context.WithTimeout(ctx, b.cfg.Timeout)
		conn, resp, dialErr := websocket.Dial(dialCtx, target.String(), &websocket.DialOptions{HTTPClient: &client, HTTPHeader: headers})
		cancel()
		if dialErr == nil {
			return conn, cred.AccessToken, nil
		}
		if resp == nil {
			return nil, "", &backendError{502, "Codex WebSocket connection failed."}
		}
		if resp.StatusCode == http.StatusUnauthorized {
			action := "return_401"
			if attempt == 0 {
				action = "reload_and_retry"
			}
			log.Printf("codex.auth.unauthorized code=upstream_unauthorized stage=websocket attempt=%d action=%s", attempt+1, action)
			if attempt == 0 {
				cred, err = b.auth.reload()
				if err != nil {
					logAuthFailure("websocket_reload_after_401", err)
					return nil, "", &backendError{401, err.Error()}
				}
				continue
			}
		}
		status := resp.StatusCode
		if status < 400 || status > 599 {
			status = 502
		}
		// Never expose a handshake response body or a websocket library error: both
		// can contain credentials, response headers, or an arbitrary upstream body.
		return nil, "", &backendError{status, "Codex WebSocket handshake failed."}
	}
	return nil, "", &backendError{502, "Codex WebSocket connection failed."}
}

func (s *server) websocketResponse(w http.ResponseWriter, r *http.Request) {
	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		writeError(w, 426, "Use a WebSocket upgrade for GET /v1/responses.", "invalid_request_error", nil, nil)
		return
	}
	// Reject cross-origin browser handshakes before opening an upstream socket.
	if origin := r.Header.Get("Origin"); origin != "" {
		u, err := url.Parse(origin)
		if err != nil || !strings.EqualFold(u.Host, r.Host) {
			writeError(w, 403, "Cross-origin WebSocket connections are not allowed.", "invalid_request_error", nil, nil)
			return
		}
	}
	b, ok := s.backend.(websocketBackend)
	if !ok {
		writeError(w, 501, "WebSocket backend unavailable.", "api_error", nil, nil)
		return
	}
	s.wsMu.Lock()
	closing := s.wsClosing
	s.wsMu.Unlock()
	if closing {
		writeError(w, 503, "Server is shutting down.", "api_error", nil, nil)
		return
	}
	if !s.acquireRequest(w, r) {
		return
	}
	defer s.release()
	upstream, token, err := b.dialWebSocket(r.Context(), r.Header, r.URL.RawQuery)
	if err != nil {
		writeBackendError(w, err)
		return
	}
	defer upstream.CloseNow()
	downstream, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer downstream.CloseNow()
	upstream.SetReadLimit(maxSSEEventBytes)
	downstream.SetReadLimit(32 << 20)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stopCtx, stop := context.WithCancel(context.Background())
	defer stop()
	if !s.trackWebSocket(downstream, stop, cancel) {
		return
	}
	defer s.untrackWebSocket(downstream)
	s.debugWebSocket("connected")
	defer s.debugWebSocket("disconnected")

	// One backend HTTP request / slot remains occupied for the full connection,
	// including idle periods and all turns on that connection.
	var wg sync.WaitGroup
	wg.Add(3)
	results := make(chan error, 3)
	go func() {
		defer wg.Done()
		results <- s.keepWebSocketAlive(ctx, upstream, 20*time.Second)
	}()
	go func() {
		defer wg.Done()
		results <- s.relayWebSocket(ctx, upstream, downstream, false, token)
	}()
	go func() {
		defer wg.Done()
		results <- s.relayWebSocket(ctx, downstream, upstream, true, token)
	}()
	code := websocket.StatusGoingAway
	select {
	case err = <-results:
		code = relayCloseStatus(err)
	case <-stopCtx.Done():
	}
	// Close before canceling reads: cancellation immediately tears down the socket.
	closeWebSocketPair(downstream, upstream, code, "")
	cancel()
	_ = downstream.CloseNow()
	_ = upstream.CloseNow()
	wg.Wait()
}

func relayCloseStatus(err error) websocket.StatusCode {
	code := websocket.CloseStatus(err)
	switch code {
	case 1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014:
		return code
	}
	if code >= 3000 && code <= 4999 {
		return code
	}
	return websocket.StatusInternalError
}

func (s *server) keepWebSocketAlive(ctx context.Context, conn *websocket.Conn, interval time.Duration) error {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			pingCtx, cancel := context.WithTimeout(ctx, s.cfg.Timeout)
			err := conn.Ping(pingCtx)
			cancel()
			if err != nil {
				return err
			}
		}
	}
}

func (s *server) relayWebSocket(ctx context.Context, from, to *websocket.Conn, client bool, token string) error {
	for {
		kind, data, err := from.Read(ctx)
		if err != nil {
			return err
		}
		if client && kind == websocket.MessageText {
			event := decodeWebSocketEvent(data)
			if event != nil && event["type"] == "response.create" {
				if boolValue(event["background"]) {
					rejected := map[string]any{"type": "error", "status": 400, "error": map[string]any{"type": "invalid_request_error", "code": "unsupported_parameter", "param": "background", "message": backgroundUnsupportedMessage}}
					for _, key := range []string{"event_id", "stream_id"} {
						if event[key] != nil {
							rejected[key] = event[key]
						}
					}
					encoded, _ := json.Marshal(rejected)
					encoded = redactWebSocketError(encoded, s.cfg.APIKey, token)
					if err := s.writeWebSocket(ctx, from, websocket.MessageText, encoded); err != nil {
						return err
					}
					continue
				}
				// Defaults and Codex transport requirements only. Do not rebuild input
				// items or expand previous_response_id; upstream owns that conversation.
				if stringValue(event["model"]) == "" {
					event["model"] = s.cfg.Model
				}
				setDefault(event, "instructions", "You are a helpful assistant.")
				if input, ok := event["input"].(string); ok {
					event["input"] = []any{map[string]any{"role": "user", "content": input}}
				}
				event = normalizeCodexPayload(event, s.cfg.DropParams)
				delete(event, "stream")
				delete(event, "background")
				data, _ = json.Marshal(event)
			}
		} else if !client {
			data = redactWebSocketError(data, s.cfg.APIKey, token)
		}
		if err := s.writeWebSocket(ctx, to, kind, data); err != nil {
			return err
		}
	}
}

func (s *server) writeWebSocket(ctx context.Context, conn *websocket.Conn, kind websocket.MessageType, data []byte) error {
	writeCtx, cancel := context.WithTimeout(ctx, s.cfg.Timeout)
	defer cancel()
	return conn.Write(writeCtx, kind, data)
}

// Only error events/subobjects are scrubbed. Normal generated text and opaque
// encrypted reasoning must not be rewritten by a transport intermediary.
func redactWebSocketError(data []byte, secrets ...string) []byte {
	event := decodeWebSocketEvent(data)
	if event == nil {
		return data
	}
	var scrub func(any) any
	scrub = func(value any) any {
		switch v := value.(type) {
		case string:
			for _, secret := range secrets {
				if secret != "" {
					v = strings.ReplaceAll(v, secret, "[REDACTED]")
				}
			}
			return redactSensitive(v)
		case map[string]any:
			for key, child := range v {
				switch strings.ToLower(key) {
				case "access_token", "refresh_token", "id_token", "api_key", "authorization":
					v[key] = "[REDACTED]"
				default:
					v[key] = scrub(child)
				}
			}
		case []any:
			for i := range v {
				v[i] = scrub(v[i])
			}
		}
		return value
	}
	t := stringValue(event["type"])
	changed := false
	if t == "error" || strings.HasSuffix(t, ".failed") {
		scrub(event)
		changed = true
	} else if response := mapAny(event["response"]); response != nil && response["error"] != nil {
		response["error"] = scrub(response["error"])
		changed = true
	}
	if event["error"] != nil {
		event["error"] = scrub(event["error"])
		changed = true
	}
	if !changed {
		return data
	}
	encoded, err := json.Marshal(event)
	if err != nil {
		return data
	}
	return encoded
}

func (s *server) debugWebSocket(action string) {
	if s.cfg.Verbose {
		log.Printf("codex.websocket.%s", action)
	}
}

type websocketLifecycle struct {
	stop  context.CancelFunc
	abort context.CancelFunc
}

func (s *server) trackWebSocket(conn *websocket.Conn, stop, abort context.CancelFunc) bool {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	if s.wsClosing {
		abort()
		return false
	}
	if s.wsCancels == nil {
		s.wsCancels = make(map[*websocket.Conn]websocketLifecycle)
	}
	s.wsWG.Add(1)
	s.wsCancels[conn] = websocketLifecycle{stop: stop, abort: abort}
	return true
}
func (s *server) untrackWebSocket(conn *websocket.Conn) {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	delete(s.wsCancels, conn)
	s.wsWG.Done()
}
func (s *server) closeWebSockets() {
	timeout := s.cfg.StopTimeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	_ = s.closeWebSocketsContext(ctx)
}

func (s *server) closeWebSocketsContext(ctx context.Context) error {
	s.wsMu.Lock()
	s.wsClosing = true
	for _, lifecycle := range s.wsCancels {
		lifecycle.stop()
	}
	s.wsMu.Unlock()
	done := make(chan struct{})
	go func() { s.wsWG.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		s.abortWebSockets()
		return ctx.Err()
	}
}

// Cancel relay reads to tear down both sockets without waiting for close replies.
func (s *server) abortWebSockets() {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	s.wsClosing = true
	for _, lifecycle := range s.wsCancels {
		lifecycle.abort()
	}
}

const backgroundUnsupportedMessage = "Codex does not support background responses. Use foreground HTTP streaming or WebSocket mode."

func closeWebSocketPair(a, b *websocket.Conn, code websocket.StatusCode, reason string) {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); _ = a.Close(code, reason) }()
	_ = b.Close(code, reason)
	wg.Wait()
}

func decodeWebSocketEvent(data []byte) map[string]any {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var event map[string]any
	if decoder.Decode(&event) != nil {
		return nil
	}
	var extra any
	if decoder.Decode(&extra) != io.EOF {
		return nil
	}
	return event
}
