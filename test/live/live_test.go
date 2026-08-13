package live_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	startupMarker = "listening on http://"
	liveAPIKey    = "go-live-e2e-local-key"
)

var onePixelPNGDataURL = "data:image/png;base64," +
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

func TestLiveGoServerCompatibilityMatrix(t *testing.T) {
	if os.Getenv("RUN_CODEX_LIVE_TESTS") != "1" {
		t.Skip("set RUN_CODEX_LIVE_TESTS=1 to use real Codex credentials and network requests")
	}
	server := startLiveServer(t)
	client := &liveClient{
		baseURL: server.baseURL + "/v1",
		apiKey:  liveAPIKey,
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
	model := chooseLiveModel(t, client)
	t.Logf("Go live E2E server=%s model=%s", server.baseURL, model)

	t.Run("responses lifecycle streaming and context", func(t *testing.T) {
		marker := "GO-LIVE-RESP-731"
		created := client.json(t, http.MethodPost, "/responses", map[string]any{
			"model":        model,
			"instructions": "Preserve exact marker strings and answer concisely.",
			"input":        "Reply with exactly: " + marker,
			"reasoning":    map[string]any{"effort": "low"},
		}, http.StatusOK)
		id := stringValue(created["id"])
		assertMarker(t, responseText(created), marker)

		retrieved := client.json(t, http.MethodGet, "/responses/"+url.PathEscape(id), nil, http.StatusOK)
		assertMarker(t, responseText(retrieved), marker)
		items := client.json(t, http.MethodGet, "/responses/"+url.PathEscape(id)+"/input_items?limit=10", nil, http.StatusOK)
		if len(sliceValue(items["data"])) == 0 {
			t.Fatalf("response input_items is empty: %#v", items)
		}
		tokens := client.json(t, http.MethodPost, "/responses/input_tokens", map[string]any{
			"model": model, "input": "count these live E2E tokens",
		}, http.StatusOK)
		if numberValue(tokens["input_tokens"]) <= 0 {
			t.Fatalf("invalid token count: %#v", tokens)
		}

		continuedMarker := "GO-LIVE-CONTINUED-732"
		continued := client.json(t, http.MethodPost, "/responses", map[string]any{
			"model":                model,
			"instructions":         "Return both exact marker strings, separated by a space.",
			"input":                "Also remember " + continuedMarker + " and return both markers.",
			"previous_response_id": id,
			"reasoning":            map[string]any{"effort": "low"},
		}, http.StatusOK)
		assertMarker(t, responseText(continued), marker)
		assertMarker(t, responseText(continued), continuedMarker)

		streamMarker := "GO-LIVE-RESP-STREAM-733"
		events := client.sse(t, http.MethodPost, "/responses", map[string]any{
			"model": model, "input": "Reply with exactly: " + streamMarker,
			"stream": true, "reasoning": map[string]any{"effort": "low"},
		}, http.StatusOK)
		var text strings.Builder
		terminal := false
		for _, event := range events {
			switch stringValue(event["type"]) {
			case "response.output_text.delta":
				text.WriteString(stringValue(event["delta"]))
			case "response.completed":
				terminal = true
			}
		}
		if !terminal {
			t.Fatalf("stream has no response.completed event: %v", eventTypes(events))
		}
		assertMarker(t, text.String(), streamMarker)
		t.Logf("responses: created=%s stream_events=%v output=%q", id, eventTypes(events), shortText(text.String()))
	})

	t.Run("structured outputs", func(t *testing.T) {
		response := client.json(t, http.MethodPost, "/responses", map[string]any{
			"model":        model,
			"instructions": "Return only JSON that satisfies the supplied schema.",
			"input":        "Set code to GO-RESP-STRUCT-204 and count to 7.",
			"text": map[string]any{"format": map[string]any{
				"type": "json_schema", "name": "go_live_response", "strict": true,
				"schema": structuredSchema(),
			}},
			"reasoning": map[string]any{"effort": "low"},
		}, http.StatusOK)
		assertStructured(t, responseText(response), "GO-RESP-STRUCT-204", 7)

		chat := client.json(t, http.MethodPost, "/chat/completions", map[string]any{
			"model": model,
			"messages": []any{
				map[string]any{"role": "system", "content": "Return only JSON that satisfies the supplied schema."},
				map[string]any{"role": "user", "content": "Set code to GO-CHAT-STRUCT-305 and count to 11."},
			},
			"response_format": map[string]any{
				"type": "json_schema", "json_schema": map[string]any{
					"name": "go_live_chat", "strict": true, "schema": structuredSchema(),
				},
			},
			"reasoning_effort": "low",
		}, http.StatusOK)
		assertStructured(t, chatText(chat), "GO-CHAT-STRUCT-305", 11)
	})

	t.Run("forced tools nonstreaming and streaming", func(t *testing.T) {
		tool := map[string]any{
			"type": "function", "name": "emit_go_probe",
			"description": "Emit the exact requested probe identifier.",
			"parameters": map[string]any{
				"type": "object", "additionalProperties": false,
				"properties": map[string]any{"probe_id": map[string]any{"type": "string"}},
				"required":   []any{"probe_id"},
			},
			"strict": true,
		}
		response := client.json(t, http.MethodPost, "/responses", map[string]any{
			"model":        model,
			"instructions": "Use the supplied function and do not answer directly.",
			"input":        "Call emit_go_probe with probe_id GO-TOOL-4242.",
			"tools":        []any{tool},
			"tool_choice":  map[string]any{"type": "function", "name": "emit_go_probe"},
			"reasoning":    map[string]any{"effort": "low"},
		}, http.StatusOK)
		call := findResponseToolCall(t, response)
		assertToolCall(t, call, "emit_go_probe", "GO-TOOL-4242")

		chatTool := map[string]any{"type": "function", "function": map[string]any{
			"name": tool["name"], "description": tool["description"],
			"parameters": tool["parameters"], "strict": true,
		}}
		events := client.sse(t, http.MethodPost, "/chat/completions", map[string]any{
			"model": model,
			"messages": []any{
				map[string]any{"role": "system", "content": "Use the supplied function and do not answer directly."},
				map[string]any{"role": "user", "content": "Call emit_go_probe with probe_id GO-CHAT-TOOL-4343."},
			},
			"tools":       []any{chatTool},
			"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "emit_go_probe"}},
			"stream":      true, "reasoning_effort": "low",
		}, http.StatusOK)
		var name, arguments string
		var finish string
		for _, event := range events {
			for _, rawChoice := range sliceValue(event["choices"]) {
				choice := mapValue(rawChoice)
				if value := stringValue(choice["finish_reason"]); value != "" {
					finish = value
				}
				for _, rawCall := range sliceValue(mapValue(choice["delta"])["tool_calls"]) {
					function := mapValue(mapValue(rawCall)["function"])
					if value := stringValue(function["name"]); value != "" {
						name = value
					}
					arguments += stringValue(function["arguments"])
				}
			}
		}
		assertToolCall(t, map[string]any{"name": name, "arguments": arguments}, "emit_go_probe", "GO-CHAT-TOOL-4343")
		if finish != "tool_calls" {
			t.Fatalf("chat tool stream finish_reason = %q, want tool_calls", finish)
		}
		t.Logf("tools: response_args=%s chat_args=%s", call["arguments"], arguments)
	})

	t.Run("chat stored lifecycle and multiple-choice stream", func(t *testing.T) {
		marker := "GO-LIVE-CHAT-STORE-414"
		created := client.json(t, http.MethodPost, "/chat/completions", map[string]any{
			"model": model, "n": 2, "store": true,
			"metadata": map[string]any{"suite": "go-live", "case": "stored"},
			"messages": []any{
				map[string]any{"role": "system", "content": "Preserve exact marker strings."},
				map[string]any{"role": "user", "content": "Reply with exactly: " + marker},
			},
			"reasoning_effort": "low",
		}, http.StatusOK)
		id := stringValue(created["id"])
		choices := sliceValue(created["choices"])
		if id == "" || len(choices) != 2 {
			t.Fatalf("invalid stored n=2 chat: %#v", created)
		}
		for _, choice := range choices {
			assertMarker(t, stringValue(mapValue(mapValue(choice)["message"])["content"]), marker)
		}

		retrieved := client.json(t, http.MethodGet, "/chat/completions/"+url.PathEscape(id), nil, http.StatusOK)
		assertMarker(t, chatText(retrieved), marker)
		listed := client.json(t, http.MethodGet, "/chat/completions?limit=5&metadata%5Bsuite%5D=go-live&metadata%5Bcase%5D=stored", nil, http.StatusOK)
		if len(sliceValue(listed["data"])) != 1 || stringValue(mapValue(sliceValue(listed["data"])[0])["id"]) != id {
			t.Fatalf("stored chat list mismatch: %#v", listed)
		}
		updated := client.json(t, http.MethodPost, "/chat/completions/"+url.PathEscape(id), map[string]any{
			"metadata": map[string]any{"suite": "go-live", "case": "updated"},
		}, http.StatusOK)
		if stringValue(mapValue(updated["metadata"])["case"]) != "updated" {
			t.Fatalf("chat metadata was not updated: %#v", updated)
		}
		messages := client.json(t, http.MethodGet, "/chat/completions/"+url.PathEscape(id)+"/messages", nil, http.StatusOK)
		if len(sliceValue(messages["data"])) != 2 {
			t.Fatalf("stored chat messages mismatch: %#v", messages)
		}

		streamMarker := "GO-LIVE-CHAT-STREAM-616"
		events := client.sse(t, http.MethodPost, "/chat/completions", map[string]any{
			"model": model, "n": 2, "stream": true,
			"stream_options":   map[string]any{"include_usage": true},
			"messages":         []any{map[string]any{"role": "user", "content": "Reply with exactly: " + streamMarker}},
			"reasoning_effort": "low",
		}, http.StatusOK)
		content := map[int]string{}
		finished := map[int]bool{}
		usageSeen := false
		for _, event := range events {
			if event["usage"] != nil {
				usageSeen = true
			}
			for _, rawChoice := range sliceValue(event["choices"]) {
				choice := mapValue(rawChoice)
				index := int(numberValue(choice["index"]))
				content[index] += stringValue(mapValue(choice["delta"])["content"])
				if stringValue(choice["finish_reason"]) != "" {
					finished[index] = true
				}
			}
		}
		for _, index := range []int{0, 1} {
			assertMarker(t, content[index], streamMarker)
			if !finished[index] {
				t.Fatalf("chat stream choice %d did not finish", index)
			}
		}
		if !usageSeen {
			t.Fatal("chat stream did not include a usage chunk")
		}
		deleted := client.json(t, http.MethodDelete, "/chat/completions/"+url.PathEscape(id), nil, http.StatusOK)
		if deleted["deleted"] != true {
			t.Fatalf("chat delete mismatch: %#v", deleted)
		}
		t.Logf("chat: stored=%s stream_events=%d", id, len(events))
	})

	t.Run("vision and image generation", func(t *testing.T) {
		marker := "GO-LIVE-VISION-818"
		vision := client.json(t, http.MethodPost, "/responses", map[string]any{
			"model":        model,
			"instructions": "Preserve exact marker strings.",
			"input": []any{map[string]any{
				"role": "user", "content": []any{
					map[string]any{"type": "input_text", "text": "Inspect the image, then reply with exactly: " + marker},
					map[string]any{"type": "input_image", "image_url": onePixelPNGDataURL, "detail": "low"},
				},
			}},
			"reasoning": map[string]any{"effort": "low"},
		}, http.StatusOK)
		assertMarker(t, responseText(vision), marker)

		generated := client.json(t, http.MethodPost, "/images/generations", map[string]any{
			"model":  "gpt-image-2",
			"prompt": "A simple centered red circle on a plain white background. No text.",
			"size":   "1024x1024", "quality": "low", "output_format": "png",
		}, http.StatusOK)
		images := sliceValue(generated["data"])
		if len(images) != 1 {
			t.Fatalf("image generation returned %d images: %#v", len(images), generated)
		}
		decoded, err := base64.StdEncoding.DecodeString(stringValue(mapValue(images[0])["b64_json"]))
		if err != nil || len(decoded) < 24 || !bytes.Equal(decoded[:8], []byte("\x89PNG\r\n\x1a\n")) {
			t.Fatalf("invalid generated PNG: bytes=%d err=%v", len(decoded), err)
		}
		width := binary.BigEndian.Uint32(decoded[16:20])
		height := binary.BigEndian.Uint32(decoded[20:24])
		if width == 0 || height == 0 {
			t.Fatalf("invalid generated PNG dimensions: %dx%d", width, height)
		}
		t.Logf("image generation: %dx%d bytes=%d", width, height, len(decoded))
	})

	t.Run("audio transcription and fallback proxy", func(t *testing.T) {
		var multipartBody bytes.Buffer
		writer := multipart.NewWriter(&multipartBody)
		file, err := writer.CreateFormFile("file", "go-live.wav")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write(silentWAV()); err != nil {
			t.Fatal(err)
		}
		if err := writer.WriteField("model", "whisper-1"); err != nil {
			t.Fatal(err)
		}
		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
		response, data := client.raw(t, http.MethodPost, "/audio/transcriptions", &multipartBody, map[string]string{
			"Content-Type": writer.FormDataContentType(),
		})
		var transcription map[string]any
		switch response.StatusCode {
		case http.StatusOK:
			if err := json.Unmarshal(data, &transcription); err != nil || transcription["text"] == nil {
				t.Fatalf("invalid transcription response: %s (err=%v)", data, err)
			}
		case http.StatusForbidden:
			// ChatGPT may protect this sibling backend endpoint with a browser
			// challenge even when Codex Responses is available. The deterministic
			// contract suite verifies its successful proxy path; this live probe
			// verifies that the real upstream rejection is passed through.
			if !bytes.Contains(bytes.ToLower(data), []byte("challenge")) &&
				!bytes.Contains(bytes.ToLower(data), []byte("enable javascript")) {
				t.Fatalf("unexpected audio transcription rejection: %s", data)
			}
			t.Log("audio transcription: real upstream requires a browser challenge (403); successful proxy shape is covered by the deterministic contract")
		default:
			t.Fatalf("audio transcription status=%d body=%s", response.StatusCode, data)
		}

		proxyCases := []struct {
			path    string
			payload map[string]any
		}{
			{"/tokenizer", map[string]any{"model": model, "input": "go live proxy tokenizer"}},
			{"/responses/compact", map[string]any{"model": model, "input": []any{map[string]any{"role": "user", "content": "go live proxy compact"}}}},
			{"/embeddings", map[string]any{"model": model, "input": "go live proxy embedding"}},
		}
		for _, probe := range proxyCases {
			encoded, _ := json.Marshal(probe.payload)
			response, data := client.raw(t, http.MethodPost, probe.path, bytes.NewReader(encoded), map[string]string{"Content-Type": "application/json"})
			if response.Header.Get("X-OpenAI-Via-Codex-Proxy") != "codex-http" {
				t.Fatalf("fallback proxy %s missing marker header: status=%d body=%s", probe.path, response.StatusCode, data)
			}
			t.Logf("proxy: POST %s status=%d", probe.path, response.StatusCode)
		}
		response, data = client.raw(t, http.MethodGet, "/files/%2e%2e/auth/me", nil, nil)
		if response.StatusCode != http.StatusBadRequest || response.Header.Get("X-OpenAI-Via-Codex-Proxy") != "" {
			t.Fatalf("proxy traversal was not rejected: status=%d headers=%v body=%s", response.StatusCode, response.Header, data)
		}
		if transcription != nil {
			t.Logf("audio transcription: text=%q", shortText(stringValue(transcription["text"])))
		}
	})
}

type liveServer struct {
	baseURL string
}

func startLiveServer(t *testing.T) *liveServer {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	name := "openai-api-server-via-codex"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binaryPath := filepath.Join(t.TempDir(), name)
	buildContext, cancelBuild := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelBuild()
	build := exec.CommandContext(buildContext, "go", "build", "-trimpath", "-o", binaryPath, "./cmd/openai-api-server-via-codex")
	build.Dir = root
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build live Go server: %v\n%s", err, output)
	}

	command := exec.Command(binaryPath, "serve", "--host", "127.0.0.1", "--port", "0")
	command.Env = append(os.Environ(), "OPENAI_VIA_CODEX_API_KEY="+liveAPIKey)
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	command.Stdout = writer
	command.Stderr = writer
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
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

	t.Cleanup(func() {
		_ = command.Process.Kill()
		select {
		case <-processDone:
		case <-time.After(10 * time.Second):
		}
		_ = reader.Close()
		select {
		case <-drainDone:
		case <-time.After(time.Second):
		}
		if t.Failed() {
			t.Logf("Go live server logs:\n%s", readLogs(&logs, &logsMu))
		}
	})

	var address string
	select {
	case address = <-addressChannel:
	case err := <-processDone:
		t.Fatalf("live server exited before listen: %v; logs=%s", err, readLogs(&logs, &logsMu))
	case <-time.After(30 * time.Second):
		t.Fatalf("timed out waiting for live server: logs=%s", readLogs(&logs, &logsMu))
	}
	return &liveServer{baseURL: "http://" + address}
}

type liveClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

func (client *liveClient) json(t *testing.T, method, path string, payload any, status int) map[string]any {
	t.Helper()
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		body = bytes.NewReader(encoded)
	}
	response, data := client.raw(t, method, path, body, map[string]string{"Content-Type": "application/json"})
	if response.StatusCode != status {
		t.Fatalf("%s %s status=%d, want=%d; body=%s", method, path, response.StatusCode, status, data)
	}
	if len(data) == 0 {
		return nil
	}
	var document map[string]any
	if err := json.Unmarshal(data, &document); err != nil {
		t.Fatalf("%s %s returned invalid JSON: %v; body=%s", method, path, err, data)
	}
	return document
}

func (client *liveClient) sse(t *testing.T, method, path string, payload any, status int) []map[string]any {
	t.Helper()
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	response, data := client.raw(t, method, path, bytes.NewReader(encoded), map[string]string{
		"Accept": "text/event-stream", "Content-Type": "application/json",
	})
	if response.StatusCode != status {
		t.Fatalf("%s %s status=%d, want=%d; body=%s", method, path, response.StatusCode, status, data)
	}
	var events []map[string]any
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if raw == "" || raw == "[DONE]" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(raw), &event); err != nil {
			t.Fatalf("invalid SSE event: %v; data=%s", err, raw)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return events
}

func (client *liveClient) raw(t *testing.T, method, path string, body io.Reader, headers map[string]string) (*http.Response, []byte) {
	t.Helper()
	request, err := http.NewRequest(method, client.baseURL+path, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+client.apiKey)
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response, err := client.http.Do(request)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	data, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	return response, data
}

func chooseLiveModel(t *testing.T, client *liveClient) string {
	t.Helper()
	models := client.json(t, http.MethodGet, "/models", nil, http.StatusOK)
	var available []string
	for _, raw := range sliceValue(models["data"]) {
		if id := stringValue(mapValue(raw)["id"]); id != "" {
			available = append(available, id)
		}
	}
	if len(available) == 0 {
		t.Fatal("GET /v1/models returned no models")
	}
	requested := os.Getenv("OPENAI_VIA_CODEX_TEST_MODEL")
	if requested == "" {
		requested = "gpt-5.4-mini"
	}
	for _, model := range available {
		if model == requested {
			return requested
		}
	}
	if os.Getenv("OPENAI_VIA_CODEX_TEST_MODEL") != "" {
		t.Fatalf("requested live model %q is not listed; available=%v", requested, available)
	}
	t.Logf("preferred live model %q is unavailable; using %q", requested, available[0])
	return available[0]
}

func responseText(response map[string]any) string {
	var text strings.Builder
	for _, rawOutput := range sliceValue(response["output"]) {
		output := mapValue(rawOutput)
		if output["type"] != "message" {
			continue
		}
		for _, rawContent := range sliceValue(output["content"]) {
			content := mapValue(rawContent)
			if content["type"] == "output_text" {
				text.WriteString(stringValue(content["text"]))
			}
		}
	}
	return text.String()
}

func chatText(chat map[string]any) string {
	choices := sliceValue(chat["choices"])
	if len(choices) == 0 {
		return ""
	}
	return stringValue(mapValue(mapValue(choices[0])["message"])["content"])
}

func findResponseToolCall(t *testing.T, response map[string]any) map[string]any {
	t.Helper()
	for _, raw := range sliceValue(response["output"]) {
		item := mapValue(raw)
		if item["type"] == "function_call" {
			return item
		}
	}
	t.Fatalf("response has no function_call: %#v", response)
	return nil
}

func assertToolCall(t *testing.T, call map[string]any, wantName, wantProbe string) {
	t.Helper()
	if stringValue(call["name"]) != wantName {
		t.Fatalf("tool name = %q, want %q; call=%#v", call["name"], wantName, call)
	}
	var arguments map[string]any
	if err := json.Unmarshal([]byte(stringValue(call["arguments"])), &arguments); err != nil {
		t.Fatalf("invalid tool arguments: %v; call=%#v", err, call)
	}
	if stringValue(arguments["probe_id"]) != wantProbe {
		t.Fatalf("tool probe_id = %q, want %q; call=%#v", arguments["probe_id"], wantProbe, call)
	}
}

func structuredSchema() map[string]any {
	return map[string]any{
		"type": "object", "additionalProperties": false,
		"properties": map[string]any{
			"code":  map[string]any{"type": "string"},
			"count": map[string]any{"type": "integer"},
		},
		"required": []any{"code", "count"},
	}
}

func assertStructured(t *testing.T, text, code string, count float64) {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &value); err != nil {
		t.Fatalf("structured output is not JSON: %v; text=%q", err, text)
	}
	if stringValue(value["code"]) != code || numberValue(value["count"]) != count {
		t.Fatalf("structured output = %#v, want code=%q count=%v", value, code, count)
	}
}

func assertMarker(t *testing.T, text, marker string) {
	t.Helper()
	if !strings.Contains(strings.ToUpper(text), strings.ToUpper(marker)) {
		t.Fatalf("text does not contain marker %q: %q", marker, text)
	}
}

func eventTypes(events []map[string]any) []string {
	result := make([]string, 0, len(events))
	for _, event := range events {
		result = append(result, stringValue(event["type"]))
	}
	return result
}

func silentWAV() []byte {
	const sampleRate = 16_000
	const seconds = 1
	dataSize := sampleRate * seconds * 2
	data := make([]byte, 44+dataSize)
	copy(data[0:4], "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], uint32(36+dataSize))
	copy(data[8:12], "WAVE")
	copy(data[12:16], "fmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1)
	binary.LittleEndian.PutUint16(data[22:24], 1)
	binary.LittleEndian.PutUint32(data[24:28], sampleRate)
	binary.LittleEndian.PutUint32(data[28:32], sampleRate*2)
	binary.LittleEndian.PutUint16(data[32:34], 2)
	binary.LittleEndian.PutUint16(data[34:36], 16)
	copy(data[36:40], "data")
	binary.LittleEndian.PutUint32(data[40:44], uint32(dataSize))
	return data
}

func mapValue(value any) map[string]any {
	result, _ := value.(map[string]any)
	return result
}

func sliceValue(value any) []any {
	result, _ := value.([]any)
	return result
}

func stringValue(value any) string {
	if result, ok := value.(string); ok {
		return result
	}
	if value == nil {
		return ""
	}
	return fmt.Sprint(value)
}

func numberValue(value any) float64 {
	result, _ := value.(float64)
	return result
}

func shortText(value string) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > 160 {
		return value[:160] + "..."
	}
	return value
}

func readLogs(buffer *bytes.Buffer, mutex *sync.Mutex) string {
	mutex.Lock()
	defer mutex.Unlock()
	return buffer.String()
}
