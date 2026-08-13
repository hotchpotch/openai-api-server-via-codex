package app

import (
	"bufio"
	"strings"
	"testing"
)

func FuzzProxyPath(f *testing.F) {
	for _, seed := range []string{
		"models", "files/report?format#section", "files/100%done", "../auth",
		"files/../auth", `files\..\auth`, "files/\x00auth", "//responses/./compact",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, path string) {
		cleaned, err := cleanProxyPath(path)
		if err != nil {
			return
		}
		for _, segment := range strings.Split(cleaned, "/") {
			if segment == ".." {
				t.Fatalf("accepted traversal path %q as %q", path, cleaned)
			}
		}
		for _, character := range cleaned {
			if character == '\\' || character == 0 || character < 0x20 || character == 0x7f {
				t.Fatalf("accepted unsafe path %q as %q", path, cleaned)
			}
		}
		target, err := resolveProxyURL("https://example.test/backend-api/codex", path, "limit=1")
		if err != nil {
			t.Fatalf("cleanProxyPath accepted %q but resolveProxyURL rejected it: %v", path, err)
		}
		if target.Scheme != "https" || target.Host != "example.test" || target.Fragment != "" {
			t.Fatalf("path %q escaped base URL: %s", path, target)
		}
	})
}

func FuzzSSELineReader(f *testing.F) {
	for _, seed := range []string{"data: {}\n", "data: [DONE]\r\n", strings.Repeat("x", 128) + "\n", "final-line"} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		reader := bufio.NewReaderSize(strings.NewReader(input), 16)
		line, err := readSSELine(reader, 4096)
		if err == nil && len(line) > 4096 {
			t.Fatalf("returned oversized line of %d bytes", len(line))
		}
		if err == nil && strings.Contains(line, "\n") {
			t.Fatalf("returned line delimiter in %q", line)
		}
	})
}

func FuzzJWTPayload(f *testing.F) {
	for _, seed := range []string{
		"opaque", "header.payload.signature", "e30.e30.sig",
		"eyJhbGciOiJub25lIn0.eyJleHAiOjE3ODAwMDAwMDB9.sig",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, token string) {
		payload := jwtPayload(token)
		if payload == nil {
			return
		}
		if exp, ok := jwtNumber(token, "exp"); ok {
			if value, valueOK := payload["exp"].(float64); !valueOK || value != exp {
				t.Fatalf("inconsistent expiry for payload %#v", payload)
			}
		}
	})
}
