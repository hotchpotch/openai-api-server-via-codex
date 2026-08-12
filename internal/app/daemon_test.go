package app

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestResolveDaemonPathsAndDiscoverUniquePID(t *testing.T) {
	state := t.TempDir()
	cfg := defaultConfig()
	cfg.Host = "127.0.0.1"
	cfg.Port = 19191
	cfg.StateDir = state
	paths, err := resolveDaemonPaths(cfg, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(paths.PIDFile) != "server-127.0.0.1-19191.pid" {
		t.Fatalf("paths = %#v", paths)
	}

	discovered := filepath.Join(state, "server-100.64.0.1-19191.pid")
	if err := os.WriteFile(discovered, []byte("123\n"), 0600); err != nil {
		t.Fatal(err)
	}
	paths, err = resolveDaemonPaths(cfg, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if paths.PIDFile != discovered {
		t.Fatalf("discovered paths = %#v", paths)
	}
	if paths.LogFile != strings.TrimSuffix(discovered, ".pid")+".log" {
		t.Fatalf("log path = %s", paths.LogFile)
	}
}

func TestResolveDaemonPathsRefusesAmbiguousPIDDiscovery(t *testing.T) {
	state := t.TempDir()
	for index, host := range []string{"0.0.0.0", "100.64.0.1"} {
		path := filepath.Join(state, "server-"+host+"-19192.pid")
		if err := os.WriteFile(path, []byte(strconv.Itoa(index+1)), 0600); err != nil {
			t.Fatal(err)
		}
	}
	cfg := defaultConfig()
	cfg.Port = 19192
	cfg.StateDir = state
	if _, err := resolveDaemonPaths(cfg, false, false); err == nil || !strings.Contains(err.Error(), "multiple PID files") {
		t.Fatalf("error = %v", err)
	}
}

func TestPIDFileRoundTripAndConditionalRemoval(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.pid")
	if err := writePID(path, 4321); err != nil {
		t.Fatal(err)
	}
	if got := readPID(path); got != 4321 {
		t.Fatalf("pid = %d", got)
	}
	if err := removeMatchingPID(path, 1234); err != nil {
		t.Fatal(err)
	}
	if readPID(path) != 4321 {
		t.Fatal("mismatched PID removed")
	}
	if err := removeMatchingPID(path, 4321); err != nil {
		t.Fatal(err)
	}
	if readPID(path) != 0 {
		t.Fatal("matching PID remains")
	}
}

func TestGeneratedConfigIncludesDaemonSettings(t *testing.T) {
	text := defaultConfigTOML()
	if !strings.Contains(text, "[daemon]") || !strings.Contains(text, "stop_timeout = 10.0") || strings.Contains(text, "%!q(MISSING)") {
		t.Fatalf("config = %s", text)
	}
}

func TestServerCommandArgsPreserveDaemonSettings(t *testing.T) {
	cfg := defaultConfig()
	cfg.Host = "127.0.0.2"
	cfg.Port = 19193
	cfg.APIKey = "must-not-be-command-line"
	cfg.DropParams = []string{"temperature", "top_p"}
	cfg.Verbose = true
	args := serverCommandArgs("daemon-run", cfg)
	joined := strings.Join(args, " ")
	for _, expected := range []string{"daemon-run", "--host 127.0.0.2", "--port 19193", "--drop-params temperature,top_p", "--verbose"} {
		if !strings.Contains(joined, expected) {
			t.Fatalf("args %q do not contain %q", joined, expected)
		}
	}
	if strings.Contains(joined, cfg.APIKey) {
		t.Fatalf("API key leaked into args: %q", joined)
	}
}
