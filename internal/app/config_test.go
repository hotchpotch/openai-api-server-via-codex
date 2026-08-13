package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestApplyConfigFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	data := `[server]
host = "127.0.0.2"
port = 19090
default_model = "gpt-config"
timeout = 12.5
verbose = true
max_stored_items = 25
max_concurrent_requests = 4
api_key = "local-key"

[codex]
auth_json = "~/custom-auth.json"
backend_base_url = "https://example.test/codex/"
client_version = "9.9.9"

[compat]
drop_params = ["temperature", "top_p"]
`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultConfig()
	if err := cfg.applyConfigFile(path); err != nil {
		t.Fatal(err)
	}
	if cfg.Host != "127.0.0.2" || cfg.Port != 19090 || cfg.Model != "gpt-config" {
		t.Fatalf("server config = %#v", cfg)
	}
	if cfg.Timeout != 12500*time.Millisecond || !cfg.Verbose || cfg.MaxStored != 25 || cfg.Concurrency != 4 {
		t.Fatalf("numeric config = %#v", cfg)
	}
	if cfg.BackendURL != "https://example.test/codex" || len(cfg.DropParams) != 2 {
		t.Fatalf("Codex config = %#v", cfg)
	}
}

func TestConfigEnvironmentOverridesFileValues(t *testing.T) {
	t.Setenv("OPENAI_VIA_CODEX_PORT", "20001")
	t.Setenv("OPENAI_VIA_CODEX_DEFAULT_MODEL", "gpt-env")
	cfg := defaultConfig()
	cfg.Port = 19090
	cfg.Model = "gpt-file"
	cfg.applyEnvironment()
	if cfg.Port != 20001 || cfg.Model != "gpt-env" {
		t.Fatalf("config = %#v", cfg)
	}
}

func TestConfigUsesFullTOMLSyntax(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	data := `[server]
api_key = 'value#with-comment-character'

[compat]
drop_params = ["parameter,with,commas", "top_p"] # an actual comment
`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultConfig()
	if err := cfg.applyConfigFile(path); err != nil {
		t.Fatal(err)
	}
	if cfg.APIKey != "value#with-comment-character" {
		t.Fatalf("api key = %q", cfg.APIKey)
	}
	if len(cfg.DropParams) != 2 || cfg.DropParams[0] != "parameter,with,commas" {
		t.Fatalf("drop params = %#v", cfg.DropParams)
	}
}

func TestConfigRejectsInvalidKnownValueTypes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(path, []byte("[server]\nport = \"not-an-integer\"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := defaultConfig()
	if err := cfg.applyConfigFile(path); err == nil {
		t.Fatal("invalid port type was accepted")
	}
}

func TestHelpReturnsSuccess(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {"start", "--help"}, {"stop", "--help"}, {"config-generate", "--help"}} {
		if err := Run(args, "test-version"); err != nil {
			t.Fatalf("Run(%v) = %v", args, err)
		}
	}
}

func TestUnsupportedCommandListsAllPublicCommands(t *testing.T) {
	err := Run([]string{"unknown"}, "test-version")
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, command := range []string{"serve", "start", "stop", "status", "config-generate"} {
		if !strings.Contains(err.Error(), command) {
			t.Fatalf("error %q does not mention %q", err, command)
		}
	}
}

func TestForegroundServeAllowsPortZeroButDaemonRunRejectsIt(t *testing.T) {
	missingAuth := filepath.Join(t.TempDir(), "missing-auth.json")
	err := Run([]string{"serve", "--port", "0", "--auth-json", missingAuth}, "test")
	if err == nil || !strings.Contains(err.Error(), "authentication preflight failed code=auth_file_not_found") {
		t.Fatalf("serve --port 0 error = %v", err)
	}
	err = Run([]string{"daemon-run", "--port", "0", "--auth-json", missingAuth}, "test")
	if err == nil || !strings.Contains(err.Error(), "port") {
		t.Fatalf("daemon-run --port 0 error = %v", err)
	}
}
