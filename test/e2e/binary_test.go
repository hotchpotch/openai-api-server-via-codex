package e2e

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/hotchpotch/openai-api-server-via-codex/internal/testutil/codexfake"
	"github.com/hotchpotch/openai-api-server-via-codex/internal/testutil/testauth"
)

const startupMarker = "listening on http://"

func TestGoBinaryForegroundE2E(t *testing.T) {
	binary := buildBinary(t)

	t.Run("preflight failure occurs before bind", func(t *testing.T) {
		missingAuth := filepath.Join(t.TempDir(), "missing-auth.json")
		command := exec.Command(binary, "serve", "--host", "127.0.0.1", "--port", "0", "--auth-json", missingAuth)
		output, err := command.CombinedOutput()
		if err == nil {
			t.Fatal("server started with missing auth")
		}
		text := string(output)
		if !strings.Contains(text, "authentication preflight failed") || strings.Contains(text, startupMarker) {
			t.Fatalf("unexpected preflight output: %s", text)
		}
	})

	t.Run("dynamic port request and shutdown", func(t *testing.T) {
		upstream := codexfake.New(t)
		authPath := testauth.Write(t)
		apiKey := "go-binary-e2e-key"
		command := exec.Command(
			binary, "serve", "--host", "127.0.0.1", "--port", "0",
			"--auth-json", authPath, "--backend-base-url", upstream.BackendURL(),
		)
		command.Env = append(os.Environ(), "OPENAI_VIA_CODEX_API_KEY="+apiKey)
		reader, writer, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		command.Stdout = writer
		command.Stderr = writer
		if err := command.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = command.Process.Kill() })
		writer.Close()
		var logs bytes.Buffer
		var logsMu sync.Mutex
		addressChannel := make(chan string, 1)
		drainDone := make(chan struct{})
		go func() {
			defer close(drainDone)
			scanner := bufio.NewScanner(reader)
			for scanner.Scan() {
				line := scanner.Text()
				logsMu.Lock()
				logs.WriteString(line)
				logs.WriteByte('\n')
				logsMu.Unlock()
				if _, tail, ok := strings.Cut(line, startupMarker); ok {
					select {
					case addressChannel <- strings.TrimSpace(tail):
					default:
					}
				}
			}
		}()

		processDone := make(chan error, 1)
		go func() { processDone <- command.Wait() }()
		var address string
		select {
		case address = <-addressChannel:
		case err := <-processDone:
			t.Fatalf("server exited before listen: %v; logs=%s", err, readLogs(&logs, &logsMu))
		case <-time.After(15 * time.Second):
			_ = command.Process.Kill()
			t.Fatalf("timed out waiting for dynamic port; logs=%s", readLogs(&logs, &logsMu))
		}
		baseURL := "http://" + address
		client := &http.Client{Timeout: 10 * time.Second}
		waitForHealth(t, client, baseURL)

		payload, _ := json.Marshal(map[string]any{
			"model": "gpt-5.6-luna", "input": "GO-BINARY-E2E-MARKER",
		})
		request, _ := http.NewRequest(http.MethodPost, baseURL+"/v1/responses", bytes.NewReader(payload))
		request.Header.Set("Authorization", "Bearer "+apiKey)
		request.Header.Set("Content-Type", "application/json")
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		responseBody, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != http.StatusOK || !bytes.Contains(responseBody, []byte("GO-BINARY-E2E-MARKER")) {
			t.Fatalf("response = %d %s", response.StatusCode, responseBody)
		}

		if runtime.GOOS == "windows" {
			if err := command.Process.Kill(); err != nil {
				t.Fatal(err)
			}
		} else if err := command.Process.Signal(syscall.SIGTERM); err != nil {
			t.Fatal(err)
		}
		select {
		case err := <-processDone:
			if runtime.GOOS != "windows" && err != nil {
				t.Fatalf("graceful shutdown failed: %v; logs=%s", err, readLogs(&logs, &logsMu))
			}
		case <-time.After(15 * time.Second):
			_ = command.Process.Kill()
			t.Fatal("server did not stop")
		}
		reader.Close()
		<-drainDone
		serverLogs := readLogs(&logs, &logsMu)
		for _, expected := range []string{"request.end method=POST", "path=/v1/responses", "status=200"} {
			if !strings.Contains(serverLogs, expected) {
				t.Fatalf("server access log missing %q; logs=%s", expected, serverLogs)
			}
		}
	})
}

func buildBinary(t *testing.T) string {
	t.Helper()
	if configured := os.Getenv("OPENAI_VIA_CODEX_E2E_EXECUTABLE"); configured != "" {
		binary, err := filepath.Abs(configured)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(binary)
		if err != nil {
			t.Fatalf("stat configured E2E executable: %v", err)
		}
		if info.IsDir() || info.Size() == 0 {
			t.Fatalf("configured E2E executable is not a nonempty file: %s", binary)
		}
		return binary
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	name := "openai-api-server-via-codex"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(t.TempDir(), name)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	command := exec.CommandContext(ctx, "go", "build", "-trimpath", "-o", binary, "./cmd/openai-api-server-via-codex")
	command.Dir = root
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build Go binary: %v\n%s", err, output)
	}
	return binary
}

func waitForHealth(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		response, err := client.Get(baseURL + "/healthz")
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("server at %s did not become healthy", baseURL)
}

func readLogs(buffer *bytes.Buffer, mutex *sync.Mutex) string {
	mutex.Lock()
	defer mutex.Unlock()
	return buffer.String()
}
