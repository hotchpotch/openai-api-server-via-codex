package app

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultHost        = "127.0.0.1"
	defaultPort        = 18080
	defaultModel       = "gpt-5.6-luna"
	defaultBackendURL  = "https://chatgpt.com/backend-api/codex"
	defaultClient      = "1.0.0"
	defaultMaxStored   = 1000
	defaultConcurrency = 10
)

type config struct {
	Host          string
	Port          int
	Model         string
	BackendURL    string
	ClientVersion string
	AuthJSON      string
	APIKey        string
	Timeout       time.Duration
	MaxStored     int
	Concurrency   int
	Verbose       bool
	DropParams    []string
	StateDir      string
	PIDFile       string
	LogFile       string
	StopTimeout   time.Duration
}

func defaultConfig() config {
	home, _ := os.UserHomeDir()
	auth := filepath.Join(home, ".codex", "auth.json")
	return config{
		Host: defaultHost, Port: defaultPort, Model: defaultModel,
		BackendURL: defaultBackendURL, ClientVersion: defaultClient, AuthJSON: auth,
		Timeout: 300 * time.Second, MaxStored: defaultMaxStored, Concurrency: defaultConcurrency,
		StateDir: defaultStateDir(), StopTimeout: 10 * time.Second,
	}
}

func (c *config) applyEnvironment() {
	c.Host = envString("OPENAI_VIA_CODEX_HOST", c.Host)
	c.Port = envInt("OPENAI_VIA_CODEX_PORT", c.Port)
	c.Model = envString("OPENAI_VIA_CODEX_DEFAULT_MODEL", c.Model)
	c.BackendURL = strings.TrimRight(envString("OPENAI_VIA_CODEX_BACKEND_BASE_URL", c.BackendURL), "/")
	c.ClientVersion = envString("OPENAI_VIA_CODEX_CLIENT_VERSION", c.ClientVersion)
	c.AuthJSON = envString("OPENAI_VIA_CODEX_AUTH_JSON", c.AuthJSON)
	c.APIKey = strings.TrimSpace(envString("OPENAI_VIA_CODEX_API_KEY", c.APIKey))
	c.Timeout = time.Duration(envFloat("OPENAI_VIA_CODEX_TIMEOUT", c.Timeout.Seconds()) * float64(time.Second))
	c.MaxStored = envInt("OPENAI_VIA_CODEX_MAX_STORED_ITEMS", c.MaxStored)
	c.Concurrency = envInt("OPENAI_VIA_CODEX_MAX_CONCURRENT_REQUESTS", c.Concurrency)
	c.Verbose = envBool("OPENAI_VIA_CODEX_VERBOSE", c.Verbose)
	c.StateDir = envString("OPENAI_VIA_CODEX_STATE_DIR", c.StateDir)
	c.PIDFile = envString("OPENAI_VIA_CODEX_PID_FILE", c.PIDFile)
	c.LogFile = envString("OPENAI_VIA_CODEX_LOG_FILE", c.LogFile)
	c.StopTimeout = time.Duration(envFloat("OPENAI_VIA_CODEX_STOP_TIMEOUT", c.StopTimeout.Seconds()) * float64(time.Second))
}

func Run(args []string, version string) error {
	if len(args) == 1 && args[0] == "--version" {
		fmt.Println(version)
		return nil
	}
	command := "serve"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		command, args = args[0], args[1:]
	}
	if command == "config-generate" {
		return runConfigGenerate(args)
	}
	if command == "start" || command == "stop" || command == "status" {
		return runDaemonCommand(command, args)
	}
	if command != "serve" && command != "daemon-run" {
		return fmt.Errorf("unsupported command %q (supported: serve, start, stop, status, config-generate)", command)
	}
	cfg, configPath, err := loadResolvedConfig(args)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("openai-api-server-via-codex", flag.ContinueOnError)
	fs.StringVar(&configPath, "config", configPath, "configuration file path")
	fs.StringVar(&cfg.Host, "host", cfg.Host, "server bind host")
	fs.IntVar(&cfg.Port, "port", cfg.Port, "server bind port")
	fs.StringVar(&cfg.Model, "default-model", cfg.Model, "default model")
	fs.StringVar(&cfg.BackendURL, "backend-base-url", cfg.BackendURL, "Codex backend base URL")
	fs.StringVar(&cfg.ClientVersion, "client-version", cfg.ClientVersion, "client version header")
	fs.StringVar(&cfg.AuthJSON, "auth-json", cfg.AuthJSON, "Codex auth.json path")
	fs.StringVar(&cfg.APIKey, "api-key", cfg.APIKey, "incoming API key")
	timeout := cfg.Timeout.Seconds()
	fs.Float64Var(&timeout, "timeout", timeout, "backend timeout seconds")
	fs.IntVar(&cfg.MaxStored, "max-stored-items", cfg.MaxStored, "maximum in-memory stored items")
	fs.IntVar(&cfg.Concurrency, "max-concurrent-requests", cfg.Concurrency, "maximum Codex requests")
	fs.BoolVar(&cfg.Verbose, "verbose", cfg.Verbose, "verbose logging")
	var dropParams string
	fs.StringVar(&dropParams, "drop-params", "", "comma-separated downstream parameters to drop")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if cfg.MaxStored < 0 || cfg.Concurrency < 0 || cfg.Port < 1 || cfg.Port > 65535 || timeout <= 0 {
		return errors.New("port, timeout, max-stored-items, or max-concurrent-requests is invalid")
	}
	cfg.Timeout = time.Duration(timeout * float64(time.Second))
	if dropParams != "" {
		for _, value := range strings.Split(dropParams, ",") {
			if value = strings.TrimSpace(value); value != "" {
				cfg.DropParams = append(cfg.DropParams, value)
			}
		}
	}
	if command == "daemon-run" {
		return runSupervised(cfg, version)
	}
	return serve(cfg, version)
}

func loadResolvedConfig(args []string) (config, string, error) {
	cfg := defaultConfig()
	configPath := configPathFromArgs(args)
	if configPath == "" {
		configPath = os.Getenv("OPENAI_VIA_CODEX_CONFIG")
	}
	if configPath == "" {
		home, _ := os.UserHomeDir()
		configPath = filepath.Join(home, ".config", "openai-api-server-via-codex", "config.toml")
	}
	if err := cfg.applyConfigFile(configPath); err != nil {
		return config{}, "", err
	}
	cfg.applyEnvironment()
	return cfg, configPath, nil
}

func runConfigGenerate(args []string) error {
	fs := flag.NewFlagSet("config-generate", flag.ContinueOnError)
	stdout := fs.Bool("stdout", false, "print configuration")
	path := fs.String("config", "", "configuration path")
	force := fs.Bool("force", false, "overwrite")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	text := defaultConfigTOML()
	if *stdout {
		fmt.Print(text)
		return nil
	}
	if *path == "" {
		home, _ := os.UserHomeDir()
		*path = filepath.Join(home, ".config", "openai-api-server-via-codex", "config.toml")
	}
	if !*force {
		if _, err := os.Stat(*path); err == nil {
			return fmt.Errorf("configuration already exists at %s", *path)
		}
	}
	if err := os.MkdirAll(filepath.Dir(*path), 0700); err != nil {
		return err
	}
	return os.WriteFile(*path, []byte(text), 0600)
}

func defaultConfigTOML() string {
	return fmt.Sprintf(`[server]
host = %q
port = %d
default_model = %q
timeout = 300.0
verbose = false
max_stored_items = %d
max_concurrent_requests = %d

[codex]
auth_json = "~/.codex/auth.json"
backend_base_url = %q
client_version = %q

[compat]
drop_params = []

[daemon]
state_dir = %q
stop_timeout = 10.0
`, defaultHost, defaultPort, defaultModel, defaultMaxStored, defaultConcurrency, defaultBackendURL, defaultClient, defaultStateDir())
}

func defaultStateDir() string {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "openai-api-server-via-codex", "run")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "openai-api-server-via-codex", "run")
}

func configPathFromArgs(args []string) string {
	for i, arg := range args {
		if arg == "--config" && i+1 < len(args) {
			return args[i+1]
		}
		if strings.HasPrefix(arg, "--config=") {
			return strings.TrimPrefix(arg, "--config=")
		}
	}
	return ""
}

// applyConfigFile intentionally parses only the documented scalar and string-array
// settings. Keeping the parser narrow avoids bringing a TOML library into the final
// static binary while accepting the configuration generated by both runtimes.
func (c *config) applyConfigFile(path string) error {
	data, err := os.ReadFile(expandHome(path))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read config %s: %w", path, err)
	}
	section := ""
	for number, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(strings.SplitN(raw, "#", 2)[0])
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid config line %d", number+1)
		}
		key, value := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		unquoted := func() string {
			if v, err := strconv.Unquote(value); err == nil {
				return v
			}
			return value
		}
		parseInt := func() (int, error) { return strconv.Atoi(value) }
		parseFloat := func() (float64, error) { return strconv.ParseFloat(value, 64) }
		parseBool := func() (bool, error) { return strconv.ParseBool(value) }
		var parseErr error
		switch section + "." + key {
		case "server.host":
			c.Host = unquoted()
		case "server.port":
			c.Port, parseErr = parseInt()
		case "server.default_model":
			c.Model = unquoted()
		case "server.timeout":
			var seconds float64
			seconds, parseErr = parseFloat()
			c.Timeout = time.Duration(seconds * float64(time.Second))
		case "server.verbose":
			c.Verbose, parseErr = parseBool()
		case "server.max_stored_items":
			c.MaxStored, parseErr = parseInt()
		case "server.max_concurrent_requests":
			c.Concurrency, parseErr = parseInt()
		case "server.api_key":
			c.APIKey = unquoted()
		case "codex.auth_json":
			c.AuthJSON = unquoted()
		case "codex.backend_base_url":
			c.BackendURL = strings.TrimRight(unquoted(), "/")
		case "codex.client_version":
			c.ClientVersion = unquoted()
		case "compat.drop_params":
			c.DropParams, parseErr = parseStringArray(value)
		case "daemon.state_dir":
			c.StateDir = unquoted()
		case "daemon.pid_file":
			c.PIDFile = unquoted()
		case "daemon.log_file":
			c.LogFile = unquoted()
		case "daemon.stop_timeout":
			var seconds float64
			seconds, parseErr = parseFloat()
			c.StopTimeout = time.Duration(seconds * float64(time.Second))
		}
		if parseErr != nil {
			return fmt.Errorf("invalid config %s.%s on line %d: %w", section, key, number+1, parseErr)
		}
	}
	return nil
}

func parseStringArray(value string) ([]string, error) {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "[") || !strings.HasSuffix(value, "]") {
		return nil, fmt.Errorf("expected string array")
	}
	inside := strings.TrimSpace(value[1 : len(value)-1])
	if inside == "" {
		return []string{}, nil
	}
	var result []string
	for _, raw := range strings.Split(inside, ",") {
		item, err := strconv.Unquote(strings.TrimSpace(raw))
		if err != nil || strings.TrimSpace(item) == "" {
			return nil, fmt.Errorf("expected non-empty quoted string")
		}
		result = append(result, item)
	}
	return result, nil
}

func envString(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
func envInt(name string, fallback int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil {
		return v
	}
	return fallback
}
func envFloat(name string, fallback float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(name), 64); err == nil {
		return v
	}
	return fallback
}
func envBool(name string, fallback bool) bool {
	if v, err := strconv.ParseBool(os.Getenv(name)); err == nil {
		return v
	}
	return fallback
}
