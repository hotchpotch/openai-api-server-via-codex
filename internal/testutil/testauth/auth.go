package testauth

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func JWT(t testing.TB, payload map[string]any) string {
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

func Write(t testing.TB) string {
	t.Helper()
	return WriteAt(t, filepath.Join(t.TempDir(), "auth.json"))
}

func WriteAt(t testing.TB, path string) string {
	t.Helper()
	document := map[string]any{
		"auth_mode": "chatgpt",
		"tokens": map[string]any{
			"access_token": JWT(t, map[string]any{
				"exp": time.Now().Add(time.Hour).Unix(),
				"https://api.openai.com/auth": map[string]any{
					"chatgpt_account_id": "acct_go_test",
				},
			}),
			"account_id": "acct_go_test",
		},
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}
