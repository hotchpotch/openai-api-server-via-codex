package app

import (
	"strings"
	"testing"
)

func TestRedactSensitive(t *testing.T) {
	raw := "Authorization: Bearer secret-token access_token=another-secret eyJabc.def.ghi\nforged-log-line"
	got := redactSensitive(raw)
	for _, secret := range []string{"secret-token", "another-secret", "eyJabc.def.ghi"} {
		if strings.Contains(got, secret) {
			t.Fatalf("secret %q remains in %q", secret, got)
		}
	}
	if strings.ContainsAny(got, "\r\n") || !strings.Contains(got, "forged-log-line") {
		t.Fatalf("control characters were not flattened in %q", got)
	}
}
