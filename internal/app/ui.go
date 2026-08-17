package app

// A small operator console for signing this proxy in to a ChatGPT account.
//
// Why it exists: the server only ever *reads* auth.json. Producing that file
// normally means shell access to run `codex login`, which is exactly what an
// unattended container on a shared host does not offer. The console closes that
// gap with two paths — a device-code sign-in and a paste fallback — so the login
// can be established and re-established from a browser.
//
// The device-code path drives the official Codex CLI rather than calling the
// device-auth API directly. That API is undocumented (usercode + an unspecified
// poll step), so the CLI stays the safer dependency: it owns the poll loop, the
// token exchange, and the exact auth.json layout the reader expects.
//
// Access control: the console is gated by the same API key as /v1, exchanged for
// an HttpOnly session cookie because a browser cannot attach a bearer header to
// a navigation. Signing in here grants control of the ChatGPT account, so when
// no API key is configured the console refuses to serve rather than defaulting
// open.

import (
	"bufio"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	uiSessionCookie = "codex_ui_session"
	uiSessionTTL    = 12 * time.Hour
	// The CLI prints this as the page the operator opens to enter the code.
	uiDeviceVerificationURL = "https://auth.openai.com/codex/device"
)

// ansiPattern strips the colour codes the CLI wraps around the URL and code.
var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

// userCodePattern matches the one-time code, e.g. "WXET-EA6X0".
var userCodePattern = regexp.MustCompile(`\b([A-Z0-9]{4}-[A-Z0-9]{4,6})\b`)

type uiSessions struct {
	mu     sync.Mutex
	tokens map[string]time.Time
}

func newUISessions() *uiSessions { return &uiSessions{tokens: map[string]time.Time{}} }

func (s *uiSessions) issue() (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	token := hex.EncodeToString(raw)
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for existing, expiry := range s.tokens {
		if now.After(expiry) {
			delete(s.tokens, existing)
		}
	}
	s.tokens[token] = now.Add(uiSessionTTL)
	return token, nil
}

func (s *uiSessions) valid(token string) bool {
	if token == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	expiry, ok := s.tokens[token]
	if !ok {
		return false
	}
	if time.Now().After(expiry) {
		delete(s.tokens, token)
		return false
	}
	return true
}

func (s *uiSessions) revoke(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.tokens, token)
}

// deviceLogin tracks the single in-flight `codex login --device-auth` process.
// Only one may run at a time: concurrent logins would race to write auth.json.
type deviceLogin struct {
	mu       sync.Mutex
	cmd      *exec.Cmd
	running  bool
	UserCode string
	URL      string
	Message  string
	Failed   bool
	started  time.Time
}

func (d *deviceLogin) snapshot() map[string]any {
	d.mu.Lock()
	defer d.mu.Unlock()
	return map[string]any{
		"running":   d.running,
		"user_code": d.UserCode,
		"url":       d.URL,
		"message":   d.Message,
		"failed":    d.Failed,
	}
}

func (d *deviceLogin) cancel() {
	d.mu.Lock()
	cmd, running := d.cmd, d.running
	if running {
		d.Message, d.Failed = "Sign-in cancelled.", true
	}
	d.mu.Unlock()
	if running && cmd != nil && cmd.Process != nil {
		// The whole group, not just the direct child — see configureProcessGroup.
		_ = killProcessGroup(cmd.Process.Pid)
	}
}

// start launches the CLI and streams its output into the shared state until it
// exits. Returns an error only if the process could not be started at all.
func (d *deviceLogin) start(codexPath, codexHome string) error {
	d.mu.Lock()
	if d.running {
		d.mu.Unlock()
		return errors.New("a sign-in is already in progress")
	}
	cmd := exec.Command(codexPath, "login", "--device-auth")
	cmd.Env = append(os.Environ(), "CODEX_HOME="+codexHome)
	configureProcessGroup(cmd)
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		d.mu.Unlock()
		return err
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		d.mu.Unlock()
		return err
	}
	d.cmd, d.running, d.started = cmd, true, time.Now()
	d.UserCode, d.URL, d.Message, d.Failed = "", "", "Starting sign-in…", false
	d.mu.Unlock()

	go d.consume(cmd, pipe)
	return nil
}

func (d *deviceLogin) consume(cmd *exec.Cmd, pipe io.ReadCloser) {
	scanner := bufio.NewScanner(pipe)
	scanner.Buffer(make([]byte, 0, 8192), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(ansiPattern.ReplaceAllString(scanner.Text(), ""))
		if line == "" {
			continue
		}
		d.mu.Lock()
		if match := userCodePattern.FindStringSubmatch(line); match != nil && d.UserCode == "" {
			d.UserCode = match[1]
			if d.URL == "" {
				d.URL = uiDeviceVerificationURL
			}
			// The page renders the code and URL itself, so the CLI's numbered
			// instruction line ("2. Enter this one-time code…") would only
			// repeat what is already on screen.
			d.Message = "Waiting for you to enter the code…"
		}
		if strings.Contains(line, "http") && d.URL == "" {
			if idx := strings.Index(line, "http"); idx >= 0 {
				d.URL = strings.Fields(line[idx:])[0]
			}
		}
		if strings.HasPrefix(strings.ToLower(line), "error") {
			d.Message, d.Failed = line, true
		} else if d.UserCode == "" {
			d.Message = line
		}
		d.mu.Unlock()
	}
	err := cmd.Wait()
	d.mu.Lock()
	d.running = false
	if err != nil && !d.Failed {
		d.Failed, d.Message = true, "Sign-in did not complete: "+err.Error()
	} else if err == nil && !d.Failed {
		d.Message = "Signed in."
	}
	d.mu.Unlock()
}

// uiEnabled reports whether the console may serve. Without an API key there is
// nothing to authenticate against, so it stays off rather than open.
func (s *server) uiEnabled() bool { return s.cfg.APIKey != "" }

func (s *server) uiAuthorized(r *http.Request) bool {
	cookie, err := r.Cookie(uiSessionCookie)
	if err != nil {
		return false
	}
	return s.sessions.valid(cookie.Value)
}

func (s *server) serveUI(w http.ResponseWriter, r *http.Request) {
	if !s.uiEnabled() {
		writeJSON(w, 503, map[string]any{
			"error": "The console is disabled because no API key is configured. " +
				"Set OPENAI_VIA_CODEX_API_KEY and restart.",
		})
		return
	}
	// The console must never be cached by an intermediary: it is authenticated.
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")

	path := strings.TrimPrefix(r.URL.Path, "/ui")
	path = strings.TrimPrefix(path, "/")

	switch {
	case path == "" && r.Method == http.MethodGet:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, uiHTML)
	case path == "api/login" && r.Method == http.MethodPost:
		s.uiLogin(w, r)
	case path == "api/logout" && r.Method == http.MethodPost:
		s.uiLogout(w, r)
	case path == "api/status" && r.Method == http.MethodGet:
		s.uiGuard(w, r, s.uiStatus)
	case path == "api/signin/start" && r.Method == http.MethodPost:
		s.uiGuard(w, r, s.uiSignInStart)
	case path == "api/signin/cancel" && r.Method == http.MethodPost:
		s.uiGuard(w, r, s.uiSignInCancel)
	case path == "api/auth/paste" && r.Method == http.MethodPost:
		s.uiGuard(w, r, s.uiPasteAuth)
	case path == "api/auth/delete" && r.Method == http.MethodPost:
		s.uiGuard(w, r, s.uiDeleteAuth)
	default:
		http.NotFound(w, r)
	}
}

func (s *server) uiGuard(w http.ResponseWriter, r *http.Request, next func(http.ResponseWriter, *http.Request)) {
	if !s.uiAuthorized(r) {
		writeJSON(w, 401, map[string]any{"error": "Not signed in to the console."})
		return
	}
	next(w, r)
}

func (s *server) uiLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Key string `json:"key"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "Malformed request."})
		return
	}
	// Constant-time: the console key is the only thing protecting the account.
	if subtle.ConstantTimeCompare([]byte(strings.TrimSpace(body.Key)), []byte(s.cfg.APIKey)) != 1 {
		writeJSON(w, 401, map[string]any{"error": "Incorrect key."})
		return
	}
	token, err := s.sessions.issue()
	if err != nil {
		writeJSON(w, 500, map[string]any{"error": "Could not start a session."})
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     uiSessionCookie,
		Value:    token,
		Path:     "/ui",
		HttpOnly: true,
		SameSite: http.SameSiteStrictMode,
		Secure:   r.TLS != nil,
		MaxAge:   int(uiSessionTTL.Seconds()),
	})
	writeJSON(w, 200, map[string]any{"ok": true})
}

func (s *server) uiLogout(w http.ResponseWriter, r *http.Request) {
	if cookie, err := r.Cookie(uiSessionCookie); err == nil {
		s.sessions.revoke(cookie.Value)
	}
	http.SetCookie(w, &http.Cookie{
		Name: uiSessionCookie, Value: "", Path: "/ui", HttpOnly: true, MaxAge: -1,
	})
	writeJSON(w, 200, map[string]any{"ok": true})
}

func (s *server) uiStatus(w http.ResponseWriter, _ *http.Request) {
	status := map[string]any{
		"auth_path":     s.cfg.AuthJSON,
		"default_model": s.cfg.Model,
		"codex_cli":     s.codexCLIPath() != "",
		"signin":        s.device.snapshot(),
	}
	if info, err := os.Stat(expandHome(s.cfg.AuthJSON)); err == nil {
		status["auth_file_present"] = true
		status["auth_file_modified"] = info.ModTime().UTC().Format(time.RFC3339)
	} else {
		status["auth_file_present"] = false
	}

	// borrow() refreshes when needed, so a success here means the credentials
	// genuinely work — not merely that a file exists.
	cred, err := s.backendAuth().borrow()
	if err != nil {
		status["signed_in"] = false
		status["error"] = err.Error()
		writeJSON(w, 200, status)
		return
	}
	status["signed_in"] = true
	status["account_id"] = cred.AccountID
	if exp, ok := jwtNumber(cred.AccessToken, "exp"); ok {
		expiry := time.Unix(int64(exp), 0).UTC()
		status["expires_at"] = expiry.Format(time.RFC3339)
		status["expires_in_seconds"] = int64(time.Until(expiry).Seconds())
	}
	if payload := jwtPayload(cred.AccessToken); payload != nil {
		if email, ok := payload["email"].(string); ok {
			status["email"] = email
		}
	}
	writeJSON(w, 200, status)
}

// codexCLIPath resolves the Codex CLI, which only the device path needs. An
// empty result means the console offers the paste fallback alone.
func (s *server) codexCLIPath() string {
	if override := os.Getenv("OPENAI_VIA_CODEX_CLI_PATH"); override != "" {
		if _, err := os.Stat(override); err == nil {
			return override
		}
	}
	path, err := exec.LookPath("codex")
	if err != nil {
		return ""
	}
	return path
}

func (s *server) uiSignInStart(w http.ResponseWriter, _ *http.Request) {
	cli := s.codexCLIPath()
	if cli == "" {
		writeJSON(w, 501, map[string]any{
			"error": "The Codex CLI is not installed in this image, so device sign-in is unavailable. " +
				"Paste an auth.json instead.",
		})
		return
	}
	home := filepath.Dir(expandHome(s.cfg.AuthJSON))
	if err := os.MkdirAll(home, 0700); err != nil {
		writeJSON(w, 500, map[string]any{"error": "Could not prepare " + home + ": " + err.Error()})
		return
	}
	if err := s.device.start(cli, home); err != nil {
		writeJSON(w, 409, map[string]any{"error": err.Error()})
		return
	}
	// Give the CLI a moment to print the code so the first status carries it.
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		snap := s.device.snapshot()
		if snap["user_code"] != "" || snap["failed"] == true || snap["running"] == false {
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	writeJSON(w, 200, s.device.snapshot())
}

func (s *server) uiSignInCancel(w http.ResponseWriter, _ *http.Request) {
	s.device.cancel()
	writeJSON(w, 200, map[string]any{"ok": true})
}

func (s *server) uiPasteAuth(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": "Could not read the request."})
		return
	}
	var body struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		writeJSON(w, 400, map[string]any{"error": "Malformed request."})
		return
	}
	if err := validateAuthJSON([]byte(body.Content)); err != nil {
		writeJSON(w, 400, map[string]any{"error": err.Error()})
		return
	}
	path := expandHome(s.cfg.AuthJSON)
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	// Write via a temp file in the same directory, then rename: a partial write
	// must never replace a working login.
	tmp, err := os.CreateTemp(filepath.Dir(path), ".auth-*.json")
	if err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if err := tmp.Chmod(0600); err != nil {
		_ = tmp.Close()
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	if _, err := tmp.WriteString(body.Content); err != nil {
		_ = tmp.Close()
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	if err := tmp.Close(); err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	if err := os.Rename(tmpName, path); err != nil {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	// The reader caches on (mtime, size); drop it so the new file is picked up.
	s.backendAuth().invalidate()
	writeJSON(w, 200, map[string]any{"ok": true})
}

func (s *server) uiDeleteAuth(w http.ResponseWriter, _ *http.Request) {
	path := expandHome(s.cfg.AuthJSON)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		writeJSON(w, 500, map[string]any{"error": err.Error()})
		return
	}
	s.backendAuth().invalidate()
	writeJSON(w, 200, map[string]any{"ok": true})
}

// validateAuthJSON rejects content the reader would later choke on, so a bad
// paste fails at the point of paste rather than on the next API call.
func validateAuthJSON(content []byte) error {
	if len(strings.TrimSpace(string(content))) == 0 {
		return errors.New("Paste the contents of auth.json.")
	}
	var parsed map[string]any
	if err := json.Unmarshal(content, &parsed); err != nil {
		return fmt.Errorf("That is not valid JSON: %w", err)
	}
	if mode, _ := parsed["auth_mode"].(string); mode != "chatgpt" {
		return fmt.Errorf("Expected auth_mode \"chatgpt\", got %q. Use a ChatGPT login, not an API key.", mode)
	}
	tokens, ok := parsed["tokens"].(map[string]any)
	if !ok {
		return errors.New("Missing the \"tokens\" object.")
	}
	if token, _ := tokens["access_token"].(string); token == "" {
		return errors.New("Missing tokens.access_token.")
	}
	if token, _ := tokens["refresh_token"].(string); token == "" {
		return errors.New("Missing tokens.refresh_token — without it the login expires within the hour and cannot renew.")
	}
	return nil
}
