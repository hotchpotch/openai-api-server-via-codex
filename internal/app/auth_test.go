package app

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func testJWT(t *testing.T, payload map[string]any) string {
	t.Helper()
	header, err := json.Marshal(map[string]any{"alg": "none", "typ": "JWT"})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return base64.RawURLEncoding.EncodeToString(header) + "." +
		base64.RawURLEncoding.EncodeToString(body) + ".sig"
}

func writeTestAuth(t *testing.T, value map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "auth.json")
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func authDocument(accessToken string, extra map[string]any) map[string]any {
	tokens := map[string]any{"access_token": accessToken}
	for key, value := range extra {
		tokens[key] = value
	}
	return map[string]any{"auth_mode": "chatgpt", "tokens": tokens}
}

func noRefreshClient(t *testing.T) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("unexpected token refresh")
		return nil, errors.New("unreachable")
	})}
}

func TestJWTNumberDistinguishesExpiryFromOpaqueToken(t *testing.T) {
	want := float64(time.Now().Add(time.Hour).Unix())
	got, ok := jwtNumber(testJWT(t, map[string]any{"exp": want}), "exp")
	if !ok || got != want {
		t.Fatalf("jwtNumber = (%v, %t), want (%v, true)", got, ok, want)
	}
	if _, ok := jwtNumber("opaque-access-token", "exp"); ok {
		t.Fatal("opaque token reported a parsed expiry")
	}
	if tokenFresh(0, true) {
		t.Fatal("literal exp=0 must be expired")
	}
	if !tokenFresh(0, false) {
		t.Fatal("uninspectable token must be forwarded to the upstream")
	}
}

func TestAuthProviderBorrowsFreshTokenAndCachesOpaqueToken(t *testing.T) {
	path := writeTestAuth(t, authDocument("opaque-access-token", map[string]any{
		"account_id": "acct_opaque",
	}))
	provider := &authProvider{path: path, refreshClient: noRefreshClient(t)}

	for range 2 {
		credential, err := provider.borrow()
		if err != nil {
			t.Fatal(err)
		}
		if credential.AccessToken != "opaque-access-token" || credential.AccountID != "acct_opaque" {
			t.Fatalf("credential = %#v", credential)
		}
	}
	if provider.cache == nil || provider.cache.HasExp {
		t.Fatalf("cache = %#v", provider.cache)
	}
}

func TestAuthProviderRefreshesExpiredTokenOnceConcurrently(t *testing.T) {
	path := writeTestAuth(t, authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(-time.Minute).Unix(),
	}), map[string]any{
		"refresh_token": "refresh-old",
		"account_id":    "acct_refresh",
	}))
	refreshedAccess := testJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	var calls atomic.Int32
	provider := &authProvider{
		path: path,
		refreshClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			calls.Add(1)
			if request.URL.String() != refreshURL || request.Method != http.MethodPost {
				t.Errorf("refresh request = %s %s", request.Method, request.URL)
			}
			data, _ := json.Marshal(map[string]any{
				"access_token":  refreshedAccess,
				"refresh_token": "refresh-new",
			})
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(string(data))),
				Request:    request,
			}, nil
		})},
	}

	const workers = 8
	var group sync.WaitGroup
	errorsSeen := make(chan error, workers)
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			credential, err := provider.borrow()
			if err == nil && credential.AccessToken != refreshedAccess {
				err = errors.New("borrow returned the wrong refreshed token")
			}
			errorsSeen <- err
		}()
	}
	group.Wait()
	close(errorsSeen)
	for err := range errorsSeen {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	tokens := mapAny(document["tokens"])
	if tokens["refresh_token"] != "refresh-new" || tokens["access_token"] != refreshedAccess {
		t.Fatalf("refreshed auth = %#v", document)
	}
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode := stat.Mode().Perm(); mode != 0600 {
		t.Fatalf("refreshed auth mode = %o", mode)
	}
}

func TestAuthProviderCachesOpaqueTokenReturnedByRefresh(t *testing.T) {
	path := writeTestAuth(t, authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(-time.Minute).Unix(),
	}), map[string]any{"refresh_token": "refresh"}))
	var calls atomic.Int32
	provider := &authProvider{
		path: path,
		refreshClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			calls.Add(1)
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body: io.NopCloser(strings.NewReader(
					`{"access_token":"new-opaque-token","refresh_token":"refresh"}`,
				)),
				Request: request,
			}, nil
		})},
	}

	for range 2 {
		credential, err := provider.borrow()
		if err != nil {
			t.Fatal(err)
		}
		if credential.AccessToken != "new-opaque-token" {
			t.Fatalf("credential = %#v", credential)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestAuthProviderRejectsRefreshResponseWithoutAccessToken(t *testing.T) {
	path := writeTestAuth(t, authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(-time.Minute).Unix(),
	}), map[string]any{"refresh_token": "refresh"}))
	provider := &authProvider{
		path: path,
		refreshClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"refresh_token":"replacement"}`)),
				Request:    request,
			}, nil
		})},
	}

	if _, err := provider.borrow(); err == nil || !strings.Contains(err.Error(), "missing access token") {
		t.Fatalf("error = %v", err)
	}
}

func TestAuthProviderRejectsExpiredTokenWithoutRefresh(t *testing.T) {
	path := writeTestAuth(t, authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(-time.Minute).Unix(),
	}), nil))
	provider := &authProvider{path: path, refreshClient: noRefreshClient(t)}
	if _, err := provider.borrow(); err == nil || !strings.Contains(err.Error(), "no refresh token") {
		t.Fatalf("error = %v", err)
	}
}

func TestAuthProviderReloadsChangedFileAndReadsAccountClaims(t *testing.T) {
	path := writeTestAuth(t, authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_access",
		},
	}), nil))
	provider := &authProvider{path: path, refreshClient: noRefreshClient(t)}
	first, err := provider.borrow()
	if err != nil {
		t.Fatal(err)
	}
	if first.AccountID != "acct_access" {
		t.Fatalf("first account = %q", first.AccountID)
	}

	document := authDocument(testJWT(t, map[string]any{
		"exp": time.Now().Add(time.Hour).Unix(),
	}), map[string]any{
		"padding":  "changed-size",
		"id_token": testJWT(t, map[string]any{"chatgpt_account_id": "acct_id"}),
	})
	data, _ := json.Marshal(document)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	changed := time.Now().Add(time.Second)
	if err := os.Chtimes(path, changed, changed); err != nil {
		t.Fatal(err)
	}
	second, err := provider.borrow()
	if err != nil {
		t.Fatal(err)
	}
	if second.AccountID != "acct_id" {
		t.Fatalf("second account = %q", second.AccountID)
	}
}

func TestAuthProviderRejectsInvalidDocuments(t *testing.T) {
	tests := []struct {
		name, contents, message string
	}{
		{"invalid JSON", "{not-json", "invalid Codex auth JSON"},
		{"non-object JSON", "[]", "invalid Codex auth JSON"},
		{"wrong mode", `{"auth_mode":"api_key","tokens":{"access_token":"token"}}`, "expected Codex auth_mode"},
		{"missing token", `{"auth_mode":"chatgpt","tokens":{}}`, "no ChatGPT tokens"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "auth.json")
			if err := os.WriteFile(path, []byte(test.contents), 0600); err != nil {
				t.Fatal(err)
			}
			provider := &authProvider{path: path, refreshClient: noRefreshClient(t)}
			if _, err := provider.borrow(); err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestAuthRefreshTransportErrorsAreRedacted(t *testing.T) {
	secret := "refresh_abcdefghijklmnopqrstuvwxyz"
	provider := &authProvider{refreshClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("request failed with refresh_token=" + secret)
	})}}
	_, err := provider.refresh(secret)
	if err == nil {
		t.Fatal("refresh succeeded")
	}
	if strings.Contains(err.Error(), secret) || !strings.Contains(err.Error(), "[REDACTED]") {
		t.Fatalf("unredacted error = %q", err)
	}
}
