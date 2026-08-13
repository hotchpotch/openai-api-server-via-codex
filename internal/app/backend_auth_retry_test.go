package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func replaceTestAuth(path, accessToken string) error {
	data, err := json.Marshal(authDocument(accessToken, map[string]any{"account_id": "acct_retry"}))
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func retryTestBackend(t *testing.T, upstream *httptest.Server, authPath string) *backend {
	t.Helper()
	cfg := defaultConfig()
	cfg.BackendURL = upstream.URL
	cfg.AuthJSON = authPath
	return newBackend(cfg)
}

func TestBackendReloadsExternalAuthAndRetriesStreamOnce(t *testing.T) {
	authPath := writeTestAuth(t, authDocument("access-old", map[string]any{"account_id": "acct_retry"}))
	originalStat, err := os.Stat(authPath)
	if err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	var authorizations []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorizations = append(authorizations, r.Header.Get("Authorization"))
		if calls.Add(1) == 1 {
			if err := replaceTestAuth(authPath, "access-new"); err != nil {
				t.Error(err)
				http.Error(w, "test auth replacement failed", http.StatusInternalServerError)
				return
			}
			if err := os.Chtimes(authPath, originalStat.ModTime(), originalStat.ModTime()); err != nil {
				t.Error(err)
				http.Error(w, "test auth timestamp restore failed", http.StatusInternalServerError)
				return
			}
			http.Error(w, `{"error":{"message":"expired"}}`, http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_retry\",\"status\":\"completed\",\"output\":[]}}\n\ndata: [DONE]\n\n")
	}))
	defer upstream.Close()

	var events []map[string]any
	err = retryTestBackend(t, upstream, authPath).stream(context.Background(), map[string]any{
		"model": "gpt-5.6-luna",
		"input": []any{map[string]any{"role": "user", "content": "hello"}},
	}, func(event map[string]any) error {
		events = append(events, event)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want 2", calls.Load())
	}
	wantAuth := []string{"Bearer access-old", "Bearer access-new"}
	if len(authorizations) != len(wantAuth) || authorizations[0] != wantAuth[0] || authorizations[1] != wantAuth[1] {
		t.Fatalf("authorizations = %#v, want %#v", authorizations, wantAuth)
	}
	if len(events) != 1 || events[0]["type"] != "response.completed" {
		t.Fatalf("events = %#v", events)
	}
}

func TestBackendRetriesUnauthorizedAtMostOnce(t *testing.T) {
	var logs bytes.Buffer
	previousOutput := log.Writer()
	previousFlags := log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(previousOutput)
		log.SetFlags(previousFlags)
	})

	authPath := writeTestAuth(t, authDocument("access-unchanged", nil))
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, `{"error":{"message":"still unauthorized"}}`, http.StatusUnauthorized)
	}))
	defer upstream.Close()

	err := retryTestBackend(t, upstream, authPath).stream(context.Background(), map[string]any{
		"model": "gpt-5.6-luna",
		"input": "hello",
	}, func(map[string]any) error { return nil })
	var backendErr *backendError
	if !errors.As(err, &backendErr) || backendErr.Status != http.StatusUnauthorized {
		t.Fatalf("error = %#v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want exactly 2", calls.Load())
	}
	for _, expected := range []string{
		"codex.auth.unauthorized code=upstream_unauthorized stage=upstream attempt=1 action=reload_and_retry",
		"codex.auth.unauthorized code=upstream_unauthorized stage=upstream attempt=2 action=return_401",
	} {
		if !strings.Contains(logs.String(), expected) {
			t.Fatalf("auth log missing %q: %s", expected, logs.String())
		}
	}
	if strings.Contains(logs.String(), "still unauthorized") {
		t.Fatalf("upstream response leaked into auth log: %s", logs.String())
	}
}

func TestBackendLogsLocalAuthFailureWithoutVerbose(t *testing.T) {
	var logs bytes.Buffer
	previousOutput := log.Writer()
	previousFlags := log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(previousOutput)
		log.SetFlags(previousFlags)
	})

	missing := filepath.Join(t.TempDir(), "missing-auth.json")
	cfg := defaultConfig()
	cfg.AuthJSON = missing
	err := newBackend(cfg).stream(context.Background(), map[string]any{
		"model": "gpt-5.6-luna",
		"input": "hello",
	}, func(map[string]any) error { return nil })
	var backendErr *backendError
	if !errors.As(err, &backendErr) || backendErr.Status != http.StatusUnauthorized {
		t.Fatalf("error = %#v", err)
	}
	for _, expected := range []string{
		"codex.auth.error stage=request code=auth_file_not_found",
		"run `codex login`",
	} {
		if !strings.Contains(logs.String(), expected) {
			t.Fatalf("auth log missing %q: %s", expected, logs.String())
		}
	}
}

func TestBackendReloadsExternalAuthAndReplaysProxyBody(t *testing.T) {
	authPath := writeTestAuth(t, authDocument("access-old", nil))
	payload := bytes.Repeat([]byte("replayable-body-"), replayBodyMemoryLimit/16+1)
	var calls atomic.Int32
	var bodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if calls.Add(1) == 1 {
			if err := replaceTestAuth(authPath, "access-new"); err != nil {
				t.Error(err)
				http.Error(w, "test auth replacement failed", http.StatusInternalServerError)
				return
			}
			http.Error(w, `{"error":{"message":"expired"}}`, http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Authorization") != "Bearer access-new" {
			t.Errorf("retry authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	resp, err := retryTestBackend(t, upstream, authPath).proxy(
		context.Background(), http.MethodPost, "files", "", http.Header{"Content-Type": {"application/octet-stream"}}, bytes.NewReader(payload),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || calls.Load() != 2 {
		t.Fatalf("status = %d, calls = %d", resp.StatusCode, calls.Load())
	}
	if len(bodies) != 2 || !bytes.Equal(bodies[0], payload) || !bytes.Equal(bodies[1], payload) {
		t.Fatalf("replayed body lengths = %v, want [%d %d]", bodyLengths(bodies), len(payload), len(payload))
	}
}

func TestBackendReloadsExternalAuthBeforeRetryingModels(t *testing.T) {
	authPath := writeTestAuth(t, authDocument("access-old", nil))
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			if err := replaceTestAuth(authPath, "access-new"); err != nil {
				t.Error(err)
				http.Error(w, "test auth replacement failed", http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Authorization") != "Bearer access-new" {
			t.Errorf("retry authorization = %q", r.Header.Get("Authorization"))
		}
		_, _ = io.WriteString(w, `{"models":[{"slug":"gpt-reloaded","supported_in_api":true,"visibility":"list"}]}`)
	}))
	defer upstream.Close()

	models := retryTestBackend(t, upstream, authPath).listModels(context.Background())
	if calls.Load() != 2 || len(models) != 1 || models[0] != "gpt-reloaded" {
		t.Fatalf("calls = %d, models = %#v", calls.Load(), models)
	}
}

func bodyLengths(bodies [][]byte) []int {
	lengths := make([]int, len(bodies))
	for index, body := range bodies {
		lengths[index] = len(body)
	}
	return lengths
}

func TestReplayBodySupportsRepeatedSmallReads(t *testing.T) {
	replay, err := newReplayBody(strings.NewReader("small body"))
	if err != nil {
		t.Fatal(err)
	}
	defer replay.Close()
	for range 2 {
		data, err := io.ReadAll(replay.Reader())
		if err != nil || string(data) != "small body" {
			t.Fatalf("data = %q, err = %v", data, err)
		}
	}
}

func TestReplayBodySupportsRepeatedLargeReadsAndRemovesFile(t *testing.T) {
	payload := bytes.Repeat([]byte("x"), replayBodyMemoryLimit+1)
	replay, err := newReplayBody(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	if replay.file == nil {
		t.Fatal("large body was not spooled to a file")
	}
	name := replay.file.Name()
	for range 2 {
		data, err := io.ReadAll(replay.Reader())
		if err != nil || !bytes.Equal(data, payload) {
			t.Fatalf("read length = %d, err = %v", len(data), err)
		}
	}
	replay.Close()
	if _, err := os.Stat(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("temporary request body still exists: %v", err)
	}
}
