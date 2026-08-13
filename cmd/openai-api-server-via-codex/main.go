package main

import (
	"fmt"
	"os"
	"runtime/debug"
	"strings"

	"github.com/hotchpotch/openai-api-server-via-codex/internal/app"
)

var version = "dev"

func main() {
	if err := app.Run(os.Args[1:], resolvedVersion()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func resolvedVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return normalizeVersion(version, "")
	}
	return normalizeVersion(version, info.Main.Version)
}

func normalizeVersion(configured, moduleVersion string) string {
	if configured != "dev" {
		return configured
	}
	if moduleVersion == "" || moduleVersion == "(devel)" {
		return configured
	}
	return strings.TrimPrefix(moduleVersion, "v")
}
