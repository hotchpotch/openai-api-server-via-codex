package main

import "testing"

func TestNormalizeVersion(t *testing.T) {
	tests := []struct {
		name          string
		configured    string
		moduleVersion string
		want          string
	}{
		{name: "checkout", configured: "dev", moduleVersion: "(devel)", want: "dev"},
		{name: "go install", configured: "dev", moduleVersion: "v0.2.0", want: "0.2.0"},
		{name: "release ldflags", configured: "0.2.0", moduleVersion: "(devel)", want: "0.2.0"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := normalizeVersion(test.configured, test.moduleVersion); got != test.want {
				t.Fatalf("normalizeVersion(%q, %q) = %q, want %q", test.configured, test.moduleVersion, got, test.want)
			}
		})
	}
}
