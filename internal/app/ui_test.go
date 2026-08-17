package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newUITestServer(t *testing.T, apiKey string) (*server, string) {
	t.Helper()
	dir := t.TempDir()
	cfg := defaultConfig()
	cfg.APIKey = apiKey
	cfg.AuthJSON = filepath.Join(dir, "auth.json")
	return &server{
		cfg:      cfg,
		sessions: newUISessions(),
		device:   &deviceLogin{},
	}, cfg.AuthJSON
}

func uiRequest(s *server, method, path, body string, cookie *http.Cookie) *httptest.ResponseRecorder {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, reader)
	if cookie != nil {
		req.AddCookie(cookie)
	}
	rec := httptest.NewRecorder()
	s.serveUI(rec, req)
	return rec
}

// Without an API key there is nothing to authenticate against, so the console
// must refuse rather than default open — it can hand over the ChatGPT account.
func TestUIRefusesToServeWithoutAPIKey(t *testing.T) {
	s, _ := newUITestServer(t, "")
	for _, path := range []string{"/ui", "/ui/api/status", "/ui/api/login"} {
		rec := uiRequest(s, http.MethodGet, path, "", nil)
		if rec.Code != 503 {
			t.Fatalf("%s = %d, want 503", path, rec.Code)
		}
	}
}

func TestUIRejectsUnauthenticatedAPICalls(t *testing.T) {
	s, _ := newUITestServer(t, "secret")
	for _, path := range []string{
		"/ui/api/status", "/ui/api/signin/start", "/ui/api/auth/paste", "/ui/api/auth/delete",
	} {
		method := http.MethodPost
		if strings.HasSuffix(path, "status") {
			method = http.MethodGet
		}
		rec := uiRequest(s, method, path, "{}", nil)
		if rec.Code != 401 {
			t.Fatalf("%s = %d, want 401", path, rec.Code)
		}
	}
}

func TestUILoginRejectsWrongKeyAndIssuesSessionForRight(t *testing.T) {
	s, _ := newUITestServer(t, "secret")

	if rec := uiRequest(s, http.MethodPost, "/ui/api/login", `{"key":"wrong"}`, nil); rec.Code != 401 {
		t.Fatalf("wrong key = %d, want 401", rec.Code)
	}

	rec := uiRequest(s, http.MethodPost, "/ui/api/login", `{"key":"secret"}`, nil)
	if rec.Code != 200 {
		t.Fatalf("right key = %d, want 200", rec.Code)
	}
	cookies := rec.Result().Cookies()
	if len(cookies) == 0 {
		t.Fatal("no session cookie issued")
	}
	session := cookies[0]
	if session.Name != uiSessionCookie {
		t.Fatalf("cookie name = %q", session.Name)
	}
	if !session.HttpOnly {
		t.Fatal("session cookie must be HttpOnly so page scripts cannot read it")
	}
	if session.SameSite != http.SameSiteStrictMode {
		t.Fatalf("cookie SameSite = %v, want Strict", session.SameSite)
	}

	// A forged cookie value must not be accepted.
	forged := &http.Cookie{Name: uiSessionCookie, Value: "not-a-real-session"}
	if rec := uiRequest(s, http.MethodGet, "/ui/api/status", "", forged); rec.Code != 401 {
		t.Fatalf("forged cookie = %d, want 401", rec.Code)
	}

	if rec := uiRequest(s, http.MethodGet, "/ui/api/status", "", session); rec.Code != 200 {
		t.Fatalf("valid session status = %d, want 200", rec.Code)
	}
}

func TestUILogoutRevokesSession(t *testing.T) {
	s, _ := newUITestServer(t, "secret")
	login := uiRequest(s, http.MethodPost, "/ui/api/login", `{"key":"secret"}`, nil)
	session := login.Result().Cookies()[0]

	if rec := uiRequest(s, http.MethodPost, "/ui/api/logout", "{}", session); rec.Code != 200 {
		t.Fatalf("logout = %d", rec.Code)
	}
	if rec := uiRequest(s, http.MethodGet, "/ui/api/status", "", session); rec.Code != 401 {
		t.Fatalf("status after logout = %d, want 401", rec.Code)
	}
}

func TestValidateAuthJSONRejectsUnusableContent(t *testing.T) {
	cases := []struct {
		name, content, wantSubstring string
	}{
		{"empty", "   ", "Paste the contents"},
		{"not json", "hello", "not valid JSON"},
		{"api key mode", `{"auth_mode":"apikey","tokens":{"access_token":"a","refresh_token":"b"}}`, "auth_mode"},
		{"no tokens", `{"auth_mode":"chatgpt"}`, "tokens"},
		{"no access token", `{"auth_mode":"chatgpt","tokens":{"refresh_token":"b"}}`, "access_token"},
		{"no refresh token", `{"auth_mode":"chatgpt","tokens":{"access_token":"a"}}`, "refresh_token"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAuthJSON([]byte(tc.content))
			if err == nil {
				t.Fatal("expected rejection")
			}
			if !strings.Contains(err.Error(), tc.wantSubstring) {
				t.Fatalf("error = %q, want it to mention %q", err.Error(), tc.wantSubstring)
			}
		})
	}

	valid := `{"auth_mode":"chatgpt","tokens":{"access_token":"a","refresh_token":"b"}}`
	if err := validateAuthJSON([]byte(valid)); err != nil {
		t.Fatalf("valid content rejected: %v", err)
	}
}

func TestUIPasteWritesAuthFilePrivatelyAndDeleteRemovesIt(t *testing.T) {
	s, authPath := newUITestServer(t, "secret")
	session := uiRequest(s, http.MethodPost, "/ui/api/login", `{"key":"secret"}`, nil).Result().Cookies()[0]

	content := `{"auth_mode":"chatgpt","tokens":{"access_token":"a","refresh_token":"b"}}`
	payload, _ := json.Marshal(map[string]string{"content": content})
	if rec := uiRequest(s, http.MethodPost, "/ui/api/auth/paste", string(payload), session); rec.Code != 200 {
		t.Fatalf("paste = %d body=%s", rec.Code, rec.Body.String())
	}

	written, err := os.ReadFile(authPath)
	if err != nil {
		t.Fatalf("auth.json not written: %v", err)
	}
	if string(written) != content {
		t.Fatalf("content = %q", string(written))
	}
	info, err := os.Stat(authPath)
	if err != nil {
		t.Fatal(err)
	}
	// The file holds a live ChatGPT credential; it must not be group/world readable.
	if perm := info.Mode().Perm(); perm != 0600 {
		t.Fatalf("auth.json mode = %o, want 600", perm)
	}
	// No temp file may be left behind next to it.
	entries, _ := os.ReadDir(filepath.Dir(authPath))
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".auth-") {
			t.Fatalf("temp file left behind: %s", entry.Name())
		}
	}

	if rec := uiRequest(s, http.MethodPost, "/ui/api/auth/delete", "{}", session); rec.Code != 200 {
		t.Fatalf("delete = %d", rec.Code)
	}
	if _, err := os.Stat(authPath); !os.IsNotExist(err) {
		t.Fatal("auth.json should be gone after delete")
	}
}

// A bad paste must not destroy a working login.
func TestUIPasteRejectionLeavesExistingAuthIntact(t *testing.T) {
	s, authPath := newUITestServer(t, "secret")
	session := uiRequest(s, http.MethodPost, "/ui/api/login", `{"key":"secret"}`, nil).Result().Cookies()[0]

	original := `{"auth_mode":"chatgpt","tokens":{"access_token":"good","refresh_token":"good"}}`
	if err := os.WriteFile(authPath, []byte(original), 0600); err != nil {
		t.Fatal(err)
	}

	payload, _ := json.Marshal(map[string]string{"content": "garbage"})
	if rec := uiRequest(s, http.MethodPost, "/ui/api/auth/paste", string(payload), session); rec.Code != 400 {
		t.Fatalf("bad paste = %d, want 400", rec.Code)
	}
	after, err := os.ReadFile(authPath)
	if err != nil || string(after) != original {
		t.Fatalf("existing auth.json was disturbed: %q err=%v", string(after), err)
	}
}

func TestDropParamsReadFromEnvironment(t *testing.T) {
	t.Setenv("OPENAI_VIA_CODEX_DROP_PARAMS", " fast_mode , ,service_tier,")
	cfg := defaultConfig()
	cfg.applyEnvironment()
	want := []string{"fast_mode", "service_tier"}
	if len(cfg.DropParams) != len(want) {
		t.Fatalf("DropParams = %#v, want %#v", cfg.DropParams, want)
	}
	for i, name := range want {
		if cfg.DropParams[i] != name {
			t.Fatalf("DropParams = %#v, want %#v", cfg.DropParams, want)
		}
	}
}

// The console lives outside /v1 and must not be reachable with only a bearer.
func TestUIRoutesAreSeparateFromV1BearerAuth(t *testing.T) {
	s, _ := newUITestServer(t, "secret")
	req := httptest.NewRequest(http.MethodGet, "/ui/api/status", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	s.serveUI(rec, req)
	if rec.Code != 401 {
		t.Fatalf("bearer-only console access = %d, want 401 (session cookie required)", rec.Code)
	}
}
