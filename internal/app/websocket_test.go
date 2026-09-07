package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/hotchpotch/openai-api-server-via-codex/internal/testutil/testauth"
)

func websocketTestServer(t *testing.T, handler http.HandlerFunc, mutate func(*config)) (*server, string) {
	t.Helper()
	upstream := httptest.NewServer(handler)
	t.Cleanup(upstream.Close)
	cfg := defaultConfig()
	cfg.BackendURL = upstream.URL
	cfg.AuthJSON = testauth.Write(t)
	cfg.APIKey = contractAPIKey
	cfg.Timeout = time.Second * 3
	if mutate != nil {
		mutate(&cfg)
	}
	s := &server{cfg: cfg, backend: newBackend(cfg), responses: newResponseStore(cfg.MaxStored), chats: newChatStore(cfg.MaxStored)}
	if cfg.Concurrency > 0 {
		s.slots = make(chan struct{}, cfg.Concurrency)
	}
	downstream := httptest.NewServer(s)
	t.Cleanup(func() { s.closeWebSockets(); downstream.Close() })
	return s, downstream.URL + "/v1/responses"
}

func dialTestWebSocket(t *testing.T, target string) *websocket.Conn {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, target, &websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + contractAPIKey}}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.CloseNow() })
	return conn
}

func TestWebSocketTransparentEventsAndLifecycle(t *testing.T) {
	captured := make(chan map[string]any, 4)
	s, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer "+contractAPIKey || r.Header.Get("ChatGPT-Account-ID") != "acct_go_test" {
			t.Error("upstream credentials not substituted")
		}
		if r.URL.RawQuery != "probe=1" {
			t.Error("query not forwarded")
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		ctx := context.Background()
		for {
			var event map[string]any
			if wsjson.Read(ctx, conn, &event) != nil {
				return
			}
			captured <- event
			if wsjson.Write(ctx, conn, event) != nil {
				return
			}
		}
	}, func(c *config) { c.Concurrency = 1 })
	conn := dialTestWebSocket(t, target+"?probe=1")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	input := []any{map[string]any{"type": "function_call", "call_id": "call_async", "name": "lookup", "arguments": "{}", "async": true, "future_field": "intact"}, map[string]any{"type": "reasoning", "encrypted_content": "opaque", "summary": []any{}}}
	create := map[string]any{"type": "response.create", "model": "gpt-6-astra", "previous_response_id": "upstream_only", "stream_id": "lane-2", "input": input, "tools": []any{map[string]any{"type": "custom", "name": "lookup", "async": true}}}
	if err := wsjson.Write(ctx, conn, create); err != nil {
		t.Fatal(err)
	}
	var echo map[string]any
	echo = nil
	if err := wsjson.Read(ctx, conn, &echo); err != nil {
		t.Fatal(err)
	}
	received := <-captured
	if !reflect.DeepEqual(received["input"], input) || received["previous_response_id"] != "upstream_only" || received["stream_id"] != "lane-2" {
		t.Fatalf("conversation was modified: %#v", received)
	}
	if received["store"] != false || received["stream"] != nil || received["parallel_tool_calls"] != true {
		t.Fatalf("invalid transport settings: %#v", received)
	}
	for _, event := range []map[string]any{
		{"type": "response.steer", "previous_response_id": "upstream_only", "input": "correction"},
		{"type": "future.event", "opaque": map[string]any{"async": true, "value": "encrypted_content_is_opaque"}},
	} {
		if err := wsjson.Write(ctx, conn, event); err != nil {
			t.Fatal(err)
		}
		echo = nil
		if err := wsjson.Read(ctx, conn, &echo); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(<-captured, event) || !reflect.DeepEqual(echo, event) {
			t.Fatal("event not forwarded intact")
		}
	}
	if len(s.slots) != 1 {
		t.Fatal("connection must hold concurrency slot")
	}
	if len(s.responses.values) != 0 {
		t.Fatal("websocket wrote local conversation store")
	}
	s.closeWebSockets()
	if _, _, err := conn.Read(ctx); err == nil {
		t.Fatal("shutdown left websocket open")
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(s.slots) != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(s.slots) != 0 {
		t.Fatal("connection leaked concurrency slot")
	}
}

func TestWebSocketHandshakeAuthRetryAndRedaction(t *testing.T) {
	authPath := filepath.Join(t.TempDir(), "auth.json")
	writeAuth := func(token string) {
		data, _ := json.Marshal(map[string]any{"auth_mode": "chatgpt", "tokens": map[string]any{"access_token": token}})
		if err := os.WriteFile(authPath, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeAuth("old-upstream-secret")
	var calls atomic.Int32
	var logs synchronizedLogBuffer
	oldLog := log.Writer()
	log.SetOutput(&logs)
	defer log.SetOutput(oldLog)
	_, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n == 1 {
			if r.Header.Get("Authorization") != "Bearer old-upstream-secret" {
				t.Error("wrong first credential")
			}
			writeAuth("new-upstream-secret")
			http.Error(w, "old-upstream-secret", 401)
			return
		}
		if r.Header.Get("Authorization") != "Bearer new-upstream-secret" {
			t.Error("credential not reloaded")
		}
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		_ = wsjson.Write(context.Background(), conn, map[string]any{"type": "error", "message": "new-upstream-secret " + contractAPIKey, "error": map[string]any{"refresh_token": "refresh-secret", "id_token": "id-secret"}, "code": "invalid_request"})
		_, _, _ = conn.Read(context.Background())
	}, func(c *config) { c.AuthJSON = authPath })
	conn := dialTestWebSocket(t, target)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("handshakes=%d", calls.Load())
	}
	for _, secret := range []string{"new-upstream-secret", contractAPIKey, "refresh-secret", "id-secret"} {
		if strings.Contains(string(data), secret) {
			t.Fatalf("credential leaked: %s", data)
		}
	}
	_ = conn.CloseNow()
	if !strings.Contains(logs.String(), "code=upstream_unauthorized") {
		t.Fatal("missing auth reason log")
	}
	if strings.Contains(logs.String(), "old-upstream-secret") || strings.Contains(logs.String(), "new-upstream-secret") {
		t.Fatal("credential leaked to logs")
	}
}

func TestWebSocketRejectsUnauthorizedAndUpstreamFailures(t *testing.T) {
	for _, status := range []int{401, 403, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			_, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				http.Error(w, "api_key=secret access_token=upstream-secret", status)
			}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			_, resp, err := websocket.Dial(ctx, target, nil)
			if err == nil || resp.StatusCode != 401 || calls.Load() != 0 {
				t.Fatal("incoming auth not enforced before upstream")
			}
			_, resp, err = websocket.Dial(ctx, target, &websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + contractAPIKey}}})
			if err == nil || resp.StatusCode != status {
				t.Fatalf("status=%v error=%v", resp, err)
			}
			want := int32(1)
			if status == 401 {
				want = 2
			}
			if calls.Load() != want {
				t.Fatalf("attempts=%d want=%d", calls.Load(), want)
			}
			if strings.Contains(err.Error(), "upstream-secret") {
				t.Fatal("handshake leaked upstream body")
			}
		})
	}
}

func TestWebSocketRejectsBackgroundWithoutClosingConnection(t *testing.T) {
	var calls atomic.Int32
	_, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		for {
			var v map[string]any
			if wsjson.Read(context.Background(), conn, &v) != nil {
				return
			}
			calls.Add(1)
			if wsjson.Write(context.Background(), conn, v) != nil {
				return
			}
		}
	}, nil)
	conn := dialTestWebSocket(t, target)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for _, background := range []bool{true, false} {
		if err := wsjson.Write(ctx, conn, map[string]any{"type": "response.create", "background": background, "input": "hello", "stream_id": "lane"}); err != nil {
			t.Fatal(err)
		}
		var event map[string]any
		if err := wsjson.Read(ctx, conn, &event); err != nil {
			t.Fatal(err)
		}
		if background && (mapAny(event["error"])["code"] != "unsupported_parameter" || mapAny(event["error"])["param"] != "background" || event["stream_id"] != "lane") {
			t.Fatalf("wrong error: %#v", event)
		}
		if !background && (event["type"] != "response.create" || event["background"] != nil) {
			t.Fatalf("connection did not recover: %#v", event)
		}
	}
	if calls.Load() != 1 {
		t.Fatal("background forwarded upstream")
	}
}

func TestWebSocketErrorScrubbingPreservesNormalOutput(t *testing.T) {
	for _, typ := range []string{"error", "response.steer.failed", "response.failed", "response.completed"} {
		raw, _ := json.Marshal(map[string]any{"type": typ, "response": map[string]any{"error": map[string]any{"message": "api_key=secret-key Authorization: Bearer token-value eyJabc.eyJdef.signature", "refresh_token": "refresh-value"}}})
		result := string(redactWebSocketError(raw))
		for _, secret := range []string{"secret-key", "token-value", "eyJabc.eyJdef.signature", "refresh-value"} {
			if strings.Contains(result, secret) {
				t.Fatalf("%s leaked %s", typ, secret)
			}
		}
	}
	raw := []byte(`{"type":"response.output_text.delta","delta":"literal api_key=example","unknown":true}`)
	if !bytes.Equal(redactWebSocketError(raw), raw) {
		t.Fatal("normal text modified")
	}
}

type synchronizedLogBuffer struct {
	mu   sync.Mutex
	data bytes.Buffer
}

func (b *synchronizedLogBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data.Write(p)
}
func (b *synchronizedLogBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data.String()
}

func TestWebSocketPreservesLargeIntegersAndDoesNotReplayErrors(t *testing.T) {
	var handshakes atomic.Int32
	_, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		handshakes.Add(1)
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		_, raw, err := conn.Read(context.Background())
		if err != nil {
			return
		}
		if !bytes.Contains(raw, []byte(`9007199254740993`)) {
			t.Errorf("large integer rounded: %s", raw)
		}
		_ = conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"error","status":401,"error":{"message":"Bearer arbitrary-secret","type":"authentication_error"}}`))
		_, _, _ = conn.Read(context.Background())
	}, nil)
	conn := dialTestWebSocket(t, target)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := conn.Write(ctx, websocket.MessageText, []byte(`{"type":"response.create","model":"gpt-6-astra","input":"hello","future_integer":9007199254740993}`)); err != nil {
		t.Fatal(err)
	}
	_, raw, err := conn.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "arbitrary-secret") || !strings.Contains(string(raw), `"status":401`) {
		t.Fatalf("bad error relay: %s", raw)
	}
	if handshakes.Load() != 1 {
		t.Fatal("retried after upgrade")
	}
}

func TestWebSocketHandshakeValidationAndTimeout(t *testing.T) {
	var calls atomic.Int32
	_, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		<-r.Context().Done()
	}, func(c *config) { c.Timeout = 50 * time.Millisecond })
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, resp, err := websocket.Dial(ctx, target, &websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + contractAPIKey}, "Origin": {"https://untrusted.example"}}})
	if err == nil || resp.StatusCode != 403 || calls.Load() != 0 {
		t.Fatal("cross origin handshake opened upstream")
	}
	_, resp, err = websocket.Dial(ctx, target, &websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + contractAPIKey}}})
	if err == nil || resp.StatusCode != 502 || calls.Load() != 1 {
		t.Fatal("handshake timeout not enforced")
	}
}

func TestWebSocketSaturationCloseAndRelease(t *testing.T) {
	for _, code := range []websocket.StatusCode{websocket.StatusPolicyViolation, websocket.StatusServiceRestart} {
		t.Run(code.String(), func(t *testing.T) {
			closeUpstream := make(chan struct{})
			s, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				conn, err := websocket.Accept(w, r, nil)
				if err != nil {
					return
				}
				defer conn.CloseNow()
				<-closeUpstream
				_ = conn.Close(code, "secret close reason")
			}, func(cfg *config) { cfg.Concurrency = 1 })
			conn := dialTestWebSocket(t, target)
			req := httptest.NewRequest("GET", "/v1/models", nil)
			req.Header.Set("Authorization", "Bearer "+contractAPIKey)
			rec := httptest.NewRecorder()
			s.ServeHTTP(rec, req)
			if rec.Code != 503 || rec.Header().Get("Retry-After") != "1" {
				t.Fatalf("saturation: %d %s", rec.Code, rec.Body.String())
			}
			close(closeUpstream)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			_, _, err := conn.Read(ctx)
			if websocket.CloseStatus(err) != code || strings.Contains(err.Error(), "secret close reason") {
				t.Fatalf("close: %v", err)
			}
			deadline := time.Now().Add(3 * time.Second)
			for len(s.slots) != 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if len(s.slots) != 0 {
				t.Fatal("slot not released")
			}
		})
	}
}

func TestWebSocketKeepAliveAndGracefulShutdown(t *testing.T) {
	pings := make(chan struct{}, 8)
	s, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OnPingReceived: func(context.Context, []byte) bool { pings <- struct{}{}; return true }})
		if err != nil {
			return
		}
		defer conn.CloseNow()
		_, _, _ = conn.Read(context.Background())
	}, nil)
	conn := dialTestWebSocket(t, target)
	done := make(chan error, 1)
	upstream, _, err := s.backend.(websocketBackend).dialWebSocket(context.Background(), nil, "")
	if err != nil {
		t.Fatal(err)
	}
	defer upstream.CloseNow()
	readCtx, stopRead := context.WithCancel(context.Background())
	defer stopRead()
	go upstream.Read(readCtx)
	pingCtx, stopPing := context.WithCancel(context.Background())
	go func() { done <- s.keepWebSocketAlive(pingCtx, upstream, 10*time.Millisecond) }()
	select {
	case <-pings:
	case <-time.After(time.Second):
		t.Fatal("no upstream ping")
	}
	stopPing()
	<-done
	closed := make(chan struct{})
	go func() { s.closeWebSockets(); close(closed) }()
	_, _, err = conn.Read(context.Background())
	if websocket.CloseStatus(err) != websocket.StatusGoingAway {
		t.Fatalf("shutdown: %v", err)
	}
	<-closed
}

func TestResponseContextDropsMalformedOutput(t *testing.T) {
	history := responseContext(map[string]any{"output": []any{nil, "stray", 7, map[string]any{"type": "function_call", "name": "lookup", "call_id": "call1", "arguments": "{}"}}})
	if len(history) != 1 || mapAny(history[0])["call_id"] != "call1" {
		t.Fatalf("malformed history: %#v", history)
	}
}

func TestWebSocketAbruptDisconnectReleasesSlot(t *testing.T) {
	s, target := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()
		_, _, _ = conn.Read(context.Background())
	}, func(cfg *config) { cfg.Concurrency = 1 })
	conn := dialTestWebSocket(t, target)
	_ = conn.CloseNow()
	deadline := time.Now().Add(3 * time.Second)
	for len(s.slots) != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(s.slots) != 0 {
		t.Fatal("abrupt disconnect leaked slot")
	}
	_ = dialTestWebSocket(t, target)
}

func TestShutdownDeadlineIncludesUnresponsiveWebSocket(t *testing.T) {
	for _, stalledHTTP := range []bool{false, true} {
		t.Run(fmt.Sprint("stalled_http=", stalledHTTP), func(t *testing.T) {
			s, _ := websocketTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				conn, err := websocket.Accept(w, r, nil)
				if err != nil {
					return
				}
				defer conn.CloseNow()
				_, _, _ = conn.Read(context.Background())
			}, func(cfg *config) { cfg.StopTimeout = 100 * time.Millisecond })
			httpStarted := make(chan struct{})
			front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/slow" {
					close(httpStarted)
					<-r.Context().Done()
					return
				}
				s.ServeHTTP(w, r)
			}))
			defer front.Close()
			// Deliberately never read: this client cannot answer the close handshake.
			_ = dialTestWebSocket(t, front.URL+"/v1/responses")
			if stalledHTTP {
				go func() {
					resp, err := http.Get(front.URL + "/slow")
					if err == nil {
						resp.Body.Close()
					}
				}()
				<-httpStarted
			}
			ctx, cancel := context.WithTimeout(context.Background(), s.cfg.StopTimeout)
			defer cancel()
			started := time.Now()
			done := make(chan error, 1)
			go func() { done <- s.shutdown(ctx, front.Config) }()
			// The listener must close while the unresponsive WebSocket is still draining.
			deadline := time.Now().Add(80 * time.Millisecond)
			listenerClosed := false
			for time.Now().Before(deadline) {
				c, err := net.DialTimeout("tcp", front.Listener.Addr().String(), 10*time.Millisecond)
				if err != nil {
					listenerClosed = true
					break
				}
				c.Close()
				time.Sleep(time.Millisecond)
			}
			if !listenerClosed {
				t.Error("HTTP listener stayed open during WebSocket drain")
			}
			err := <-done
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("shutdown error: %v", err)
			}
			if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
				t.Fatalf("100ms stop deadline took %s", elapsed)
			}
			deadline = time.Now().Add(time.Second)
			for len(s.slots) != 0 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if len(s.slots) != 0 {
				t.Fatal("forced shutdown leaked relay slot")
			}
		})
	}
}
