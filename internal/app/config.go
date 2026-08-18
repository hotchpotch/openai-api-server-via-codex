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

	"github.com/pelletier/go-toml/v2"
)

const (
	defaultHost        = "127.0.0.1"
	defaultPort        = 18080
	defaultModel       = "gpt-5.6-luna"
	defaultBackendURL  = "https://chatgpt.com/backend-api/codex"
	defaultClient      = "1.0.0"
	defaultMaxStored   = 1000
	defaultConcurrency = 10

	// The Codex backend does not treat every caller alike: requests identifying
	// themselves as the official `codex_exec` CLI are accepted where a
	// third-party originator is throttled or refused. Upstream ships its own
	// name; this deployment deliberately keeps the CLI's identity instead.
	// Both are overridable so the choice stays visible rather than baked in.
	defaultOriginator = "codex_exec"
	defaultUserAgent  = "codex_exec/0.146.0 (Mac OS 15.6.1; arm64) iTerm.app (codex_exec; 0.146.0)"
)

type config struct {
	Host          string
	Port          int
	Model         string
	BackendURL    string
	ClientVersion string
	Originator    string
	UserAgent     string
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
		Originator: defaultOriginator, UserAgent: defaultUserAgent,
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
	c.Originator = envString("OPENAI_VIA_CODEX_ORIGINATOR", c.Originator)
	c.UserAgent = envString("OPENAI_VIA_CODEX_USER_AGENT", c.UserAgent)
	// Every other setting is reachable from the environment; without this one a
	// container could only strip parameters by mounting a config file.
	if raw := strings.TrimSpace(os.Getenv("OPENAI_VIA_CODEX_DROP_PARAMS")); raw != "" {
		c.DropParams = splitParamList(raw)
	}
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

// splitParamList parses a comma-separated parameter list, ignoring blanks so a
// trailing comma or stray spacing does not become an empty parameter name.
func splitParamList(raw string) []string {
	var names []string
	for _, value := range strings.Split(raw, ",") {
		if value = strings.TrimSpace(value); value != "" {
			names = append(names, value)
		}
	}
	return names
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
	stopTimeout := cfg.StopTimeout.Seconds()
	if command == "daemon-run" {
		fs.Float64Var(&stopTimeout, "stop-timeout", stopTimeout, "seconds to wait before force kill")
	}
	var dropParams string
	fs.StringVar(&dropParams, "drop-params", "", "comma-separated downstream parameters to drop")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	minimumPort := 0
	if command == "daemon-run" {
		minimumPort = 1
	}
	if cfg.MaxStored < 0 || cfg.Concurrency < 0 || cfg.Port < minimumPort || cfg.Port > 65535 || timeout <= 0 || stopTimeout <= 0 {
		return errors.New("port, timeout, max-stored-items, max-concurrent-requests, or stop-timeout is invalid")
	}
	cfg.Timeout = time.Duration(timeout * float64(time.Second))
	cfg.StopTimeout = time.Duration(stopTimeout * float64(time.Second))
	if dropParams != "" {
		cfg.DropParams = splitParamList(dropParams)
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
# Identity sent upstream. Defaults impersonate the official Codex CLI, which the
# backend treats differently from a third-party originator. Clear user_agent to
# fall back to the "openai-api-server-via-codex/<version>" form.
originator = %q
user_agent = %q

[compat]
# Parameters stripped before the request reaches Codex. Empty by default: fast_mode used to be
# stripped here, but the caller now decides per request, so removing it centrally would make the
# toggle silently inert. Add names back to strip a parameter Codex rejects.
drop_params = []

[daemon]
state_dir = %q
stop_timeout = 10.0
`, defaultHost, defaultPort, defaultModel, defaultMaxStored, defaultConcurrency, defaultBackendURL, defaultClient, defaultOriginator, defaultUserAgent, defaultStateDir())
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

func (c *config) applyConfigFile(path string) error {
	data, err := os.ReadFile(expandHome(path))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read config %s: %w", path, err)
	}
	var file configFile
	if err := toml.Unmarshal(data, &file); err != nil {
		return fmt.Errorf("parse config %s: %w", path, err)
	}
	for _, name := range file.Compat.DropParams {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("parse config %s: compat.drop_params must contain non-empty strings", path)
		}
	}
	file.apply(c)
	return nil
}

type configFile struct {
	Server struct {
		Host                  *string  `toml:"host"`
		Port                  *int     `toml:"port"`
		DefaultModel          *string  `toml:"default_model"`
		Timeout               *float64 `toml:"timeout"`
		Verbose               *bool    `toml:"verbose"`
		MaxStoredItems        *int     `toml:"max_stored_items"`
		MaxConcurrentRequests *int     `toml:"max_concurrent_requests"`
		APIKey                *string  `toml:"api_key"`
	} `toml:"server"`
	Codex struct {
		AuthJSON       *string `toml:"auth_json"`
		BackendBaseURL *string `toml:"backend_base_url"`
		ClientVersion  *string `toml:"client_version"`
		Originator     *string `toml:"originator"`
		UserAgent      *string `toml:"user_agent"`
	} `toml:"codex"`
	Compat struct {
		DropParams []string `toml:"drop_params"`
	} `toml:"compat"`
	Daemon struct {
		StateDir    *string  `toml:"state_dir"`
		PIDFile     *string  `toml:"pid_file"`
		LogFile     *string  `toml:"log_file"`
		StopTimeout *float64 `toml:"stop_timeout"`
	} `toml:"daemon"`
}

func (file *configFile) apply(c *config) {
	if file.Server.Host != nil {
		c.Host = *file.Server.Host
	}
	if file.Server.Port != nil {
		c.Port = *file.Server.Port
	}
	if file.Server.DefaultModel != nil {
		c.Model = *file.Server.DefaultModel
	}
	if file.Server.Timeout != nil {
		c.Timeout = time.Duration(*file.Server.Timeout * float64(time.Second))
	}
	if file.Server.Verbose != nil {
		c.Verbose = *file.Server.Verbose
	}
	if file.Server.MaxStoredItems != nil {
		c.MaxStored = *file.Server.MaxStoredItems
	}
	if file.Server.MaxConcurrentRequests != nil {
		c.Concurrency = *file.Server.MaxConcurrentRequests
	}
	if file.Server.APIKey != nil {
		c.APIKey = *file.Server.APIKey
	}
	if file.Codex.AuthJSON != nil {
		c.AuthJSON = *file.Codex.AuthJSON
	}
	if file.Codex.BackendBaseURL != nil {
		c.BackendURL = strings.TrimRight(*file.Codex.BackendBaseURL, "/")
	}
	if file.Codex.ClientVersion != nil {
		c.ClientVersion = *file.Codex.ClientVersion
	}
	if file.Codex.Originator != nil {
		c.Originator = *file.Codex.Originator
	}
	if file.Codex.UserAgent != nil {
		c.UserAgent = *file.Codex.UserAgent
	}
	if file.Compat.DropParams != nil {
		c.DropParams = append([]string(nil), file.Compat.DropParams...)
	}
	if file.Daemon.StateDir != nil {
		c.StateDir = *file.Daemon.StateDir
	}
	if file.Daemon.PIDFile != nil {
		c.PIDFile = *file.Daemon.PIDFile
	}
	if file.Daemon.LogFile != nil {
		c.LogFile = *file.Daemon.LogFile
	}
	if file.Daemon.StopTimeout != nil {
		c.StopTimeout = time.Duration(*file.Daemon.StopTimeout * float64(time.Second))
	}
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
