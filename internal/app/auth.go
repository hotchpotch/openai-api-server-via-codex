package app

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const refreshURL = "https://auth.openai.com/oauth/token"
const codexClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

type credentials struct{ AccessToken, AccountID string }
type authCacheEntry struct {
	ModTime time.Time
	Size    int64
	Cred    credentials
	Exp     float64
}
type authProvider struct {
	path   string
	client *http.Client
	mu     sync.Mutex
	cache  *authCacheEntry
}

func (a *authProvider) borrow() (credentials, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	path := expandHome(a.path)
	stat, err := os.Stat(path)
	if err != nil {
		return credentials{}, fmt.Errorf("Codex auth file not found at %s", path)
	}
	if a.cache != nil && a.cache.Size == stat.Size() && a.cache.ModTime.Equal(stat.ModTime()) && tokenFresh(a.cache.Exp) {
		return a.cache.Cred, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return credentials{}, fmt.Errorf("read Codex auth JSON: %w", err)
	}
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		return credentials{}, fmt.Errorf("invalid Codex auth JSON at %s", path)
	}
	if doc["auth_mode"] != "chatgpt" {
		return credentials{}, fmt.Errorf("expected Codex auth_mode 'chatgpt'")
	}
	tokens, ok := doc["tokens"].(map[string]any)
	if !ok || stringValue(tokens["access_token"]) == "" {
		return credentials{}, fmt.Errorf("no ChatGPT tokens found; run `codex login` first")
	}
	access := stringValue(tokens["access_token"])
	exp := jwtNumber(access, "exp")
	if !tokenFresh(exp) {
		refresh := stringValue(tokens["refresh_token"])
		if refresh == "" {
			return credentials{}, fmt.Errorf("no refresh token available; run `codex login` again")
		}
		newTokens, err := a.refresh(refresh)
		if err != nil {
			return credentials{}, err
		}
		for _, key := range []string{"access_token", "refresh_token", "id_token"} {
			if newTokens[key] != nil {
				tokens[key] = newTokens[key]
			}
		}
		doc["tokens"] = tokens
		updated, _ := json.MarshalIndent(doc, "", "  ")
		tmp := path + ".tmp"
		if err := os.WriteFile(tmp, updated, 0600); err != nil {
			return credentials{}, err
		}
		if err := os.Rename(tmp, path); err != nil {
			return credentials{}, err
		}
		stat, _ = os.Stat(path)
		access = stringValue(tokens["access_token"])
		exp = jwtNumber(access, "exp")
	}
	cred := credentials{AccessToken: access, AccountID: accountID(tokens)}
	a.cache = &authCacheEntry{ModTime: stat.ModTime(), Size: stat.Size(), Cred: cred, Exp: exp}
	return cred, nil
}

func (a *authProvider) refresh(token string) (map[string]any, error) {
	body, _ := json.Marshal(map[string]string{"client_id": codexClientID, "grant_type": "refresh_token", "refresh_token": token})
	req, _ := http.NewRequest(http.MethodPost, refreshURL, strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("token refresh failed: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("token refresh failed (HTTP %d)", resp.StatusCode)
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("invalid token refresh response")
	}
	return result, nil
}

func tokenFresh(exp float64) bool { return exp == 0 || float64(time.Now().Unix()) < exp-30 }
func jwtPayload(token string) map[string]any {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return nil
	}
	data, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil
	}
	var value map[string]any
	if json.Unmarshal(data, &value) != nil {
		return nil
	}
	return value
}
func jwtNumber(token, key string) float64 {
	if value, ok := jwtPayload(token)[key].(float64); ok {
		return value
	}
	return 0
}
func accountID(tokens map[string]any) string {
	if v := stringValue(tokens["account_id"]); v != "" {
		return v
	}
	for _, key := range []string{"id_token", "access_token"} {
		p := jwtPayload(stringValue(tokens[key]))
		if p == nil {
			continue
		}
		if v := stringValue(p["chatgpt_account_id"]); v != "" {
			return v
		}
		if nested, ok := p["https://api.openai.com/auth"].(map[string]any); ok {
			if v := stringValue(nested["chatgpt_account_id"]); v != "" {
				return v
			}
		}
	}
	return ""
}
func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(path, "~/"))
		}
	}
	return path
}
