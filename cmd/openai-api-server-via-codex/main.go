package main

import (
	"fmt"
	"os"

	"github.com/hotchpotch/openai-api-server-via-codex/internal/app"
)

var version = "0.1.5-go"

func main() {
	if err := app.Run(os.Args[1:], version); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
